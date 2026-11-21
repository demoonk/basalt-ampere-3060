#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

constexpr int H = 2560, FF = 640;
constexpr int ROW_GU = H / 4;
constexpr int NC = H / 32;
constexpr size_t O_CODES = (size_t)2 * FF * ROW_GU;
constexpr size_t O_D_CODES = O_CODES + (size_t)H * (FF / 4);
constexpr size_t O_GU_SCALES = O_CODES + (size_t)H * (FF / 4);              // 1,228,800

static __device__ __forceinline__ uint32_t unpack4(uint32_t b) {
    return (b & 3u) | ((b >> 2 & 3u) << 8) | ((b >> 4 & 3u) << 16) | ((b >> 6 & 3u) << 24);
}
static __device__ __forceinline__ uint32_t unpack4b(uint32_t b) {   // (code - 1) per 2-bit field, signed
    return (((b & 3u) - 1u) & 0xFFu) | ((((b >> 2 & 3u) - 1u) & 0xFFu) << 8) |
           ((((b >> 4 & 3u) - 1u) & 0xFFu) << 16) | ((((b >> 6 & 3u) - 1u) & 0xFFu) << 24);
}

// VAR 0: unpack in loop + separate hx dp4a; VAR 1: shared LUT + hx; VAR 2: shared LUT with bias folded
template<int VAR>
__global__ void gu(const uint8_t* __restrict__ blob, const int8_t* __restrict__ xs,
                   const float* __restrict__ dx, const int* __restrict__ hx, float* __restrict__ out, int ne) {
    __shared__ uint32_t lut[256];
    if (VAR != 0) { lut[threadIdx.x] = (VAR == 1) ? unpack4(threadIdx.x) : unpack4b(threadIdx.x); __syncthreads(); }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int i0 = blockIdx.x * 16 + 2 * w;
    const uint8_t* c0 = blob + (size_t)i0 * ROW_GU;
    const uint8_t* c1 = c0 + ROW_GU;
    const uint8_t* sc = blob + O_GU_SCALES + (size_t)i0 * 80;
    float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
    for (int c = lane; c < NC; c += 32) {
        const uint8_t* cb0 = c0 + (size_t)c * 8;
        const uint8_t* cb1 = c1 + (size_t)c * 8;
        uint32_t m0[8], m1[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (VAR == 0) { m0[j] = unpack4(cb0[j]); m1[j] = unpack4(cb1[j]); }
            else { m0[j] = lut[cb0[j]]; m1[j] = lut[cb1[j]]; }
        }
        const float dw = __half2float(*(const __half*)(sc + (size_t)(c >> 1) * 2));
        for (int e = 0; e < ne; ++e) {
            const int8_t* x = xs + (size_t)e * NC * 32 + (size_t)c * 32;
            int s0 = 0, s1 = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int xv; memcpy(&xv, x + 4 * j, 4);
                s0 = __dp4a((int)m0[j], xv, s0);
                s1 = __dp4a((int)m1[j], xv, s1);
            }
            if (VAR == 2) { a0[e] += dw * dx[e * NC + c] * (float)s0; a1[e] += dw * dx[e * NC + c] * (float)s1; }
            else {
                int h0 = 0, h1 = 0; const int ones = 0x01010101;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int xv; memcpy(&xv, x + 4 * j, 4);
                    h0 = __dp4a(ones, xv, h0); h1 = __dp4a(ones, xv, h1);
                }
                a0[e] += dw * dx[e * NC + c] * (float)(s0 - h0);
                a1[e] += dw * dx[e * NC + c] * (float)(s1 - h1);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            a0[e] += __shfl_down_sync(0xffffffffu, a0[e], off);
            a1[e] += __shfl_down_sync(0xffffffffu, a1[e], off);
        }
    if (lane == 0)
        for (int e = 0; e < ne; ++e) { out[(size_t)i0 * 8 + e] = a0[e]; out[(size_t)(i0 + 1) * 8 + e] = a1[e]; }
}

__global__ void gu_mma(const uint8_t* __restrict__ blob, const int8_t* __restrict__ xs,
                       const float* __restrict__ dx, const int* __restrict__ hx, float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r0 = blockIdx.x * 128 + w * 16;
    const int g = lane >> 2, t = lane & 3;
    float acc[4] = {0,0,0,0};
    for (int c = 0; c < NC; ++c) {
        const uint8_t* cr0 = blob + (size_t)(r0 + g) * ROW_GU + (size_t)c * 8;
        const uint8_t* cr1 = blob + (size_t)(r0 + g + 8) * ROW_GU + (size_t)c * 8;
        uint32_t a0 = unpack4(cr0[t]), a1 = unpack4(cr1[t]), a2 = unpack4(cr0[t + 4]), a3 = unpack4(cr1[t + 4]);
        const int8_t* xb = xs + (size_t)g * NC * 32 + (size_t)c * 32;
        uint32_t b0, b1;
        memcpy(&b0, xb + 4 * t, 4);
        memcpy(&b1, xb + 16 + 4 * t, 4);
        int d[4];
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
        const float w0 = __half2float(*(const __half*)(blob + O_GU_SCALES + (size_t)(r0 + g) * 80 + (size_t)(c >> 1) * 2));
        const float w1 = __half2float(*(const __half*)(blob + O_GU_SCALES + (size_t)(r0 + g + 8) * 80 + (size_t)(c >> 1) * 2));
        const float e0 = dx[2 * t * NC + c], e1 = dx[(2 * t + 1) * NC + c];
        const float h0 = (float)hx[2 * t * NC + c], h1 = (float)hx[(2 * t + 1) * NC + c];
        acc[0] += w0 * e0 * (float)(d[0] - h0);
        acc[1] += w0 * e1 * (float)(d[1] - h1);
        acc[2] += w1 * e0 * (float)(d[2] - h0);
        acc[3] += w1 * e1 * (float)(d[3] - h1);
    }
    out[(size_t)(r0 + g) * 8 + 2 * t] = acc[0];
    out[(size_t)(r0 + g) * 8 + 2 * t + 1] = acc[1];
    out[(size_t)(r0 + g + 8) * 8 + 2 * t] = acc[2];
    out[(size_t)(r0 + g + 8) * 8 + 2 * t + 1] = acc[3];
#endif
}

// VAR 4: same as VAR 0 but the codes are fetched with streaming loads (__ldcs), the L2 hint E9.1 tests.
__global__ void __launch_bounds__(256) gu_ldcs(const uint8_t* __restrict__ blob, const int8_t* __restrict__ xs,
                                              const float* __restrict__ dx, const int* __restrict__ hx,
                                              float* __restrict__ out, int ne) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int i0 = blockIdx.x * 16 + 2 * w;
    const uint8_t* c0 = blob + (size_t)i0 * ROW_GU;
    const uint8_t* c1 = c0 + ROW_GU;
    const uint8_t* sc = blob + O_GU_SCALES + (size_t)i0 * 80;
    float a0[8] = {0,0,0,0,0,0,0,0}, a1[8] = {0,0,0,0,0,0,0,0};
    for (int c = lane; c < NC; c += 32) {
        uint2 cb0 = __ldcs((const uint2*)(c0 + (size_t)c * 8));
        uint2 cb1 = __ldcs((const uint2*)(c1 + (size_t)c * 8));
        uint32_t m0[8], m1[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) { m0[j] = unpack4(cb0.x >> (8 * j)); m1[j] = unpack4(cb1.x >> (8 * j)); }
        #pragma unroll
        for (int j = 0; j < 4; ++j) { m0[4 + j] = unpack4(cb0.y >> (8 * j)); m1[4 + j] = unpack4(cb1.y >> (8 * j)); }
        const float dw = __half2float(*(const __half*)(sc + (size_t)(c >> 1) * 2));
        for (int e = 0; e < ne; ++e) {
            const int8_t* x = xs + (size_t)e * NC * 32 + (size_t)c * 32;
            int s0 = 0, s1 = 0, h0 = 0, h1 = 0; const int ones = 0x01010101;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int xv; memcpy(&xv, x + 4 * j, 4);
                s0 = __dp4a((int)m0[j], xv, s0); h0 = __dp4a(ones, xv, h0);
                s1 = __dp4a((int)m1[j], xv, s1); h1 = __dp4a(ones, xv, h1);
            }
            a0[e] += dw * dx[e * NC + c] * (float)(s0 - h0);
            a1[e] += dw * dx[e * NC + c] * (float)(s1 - h1);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            a0[e] += __shfl_down_sync(0xffffffffu, a0[e], off);
            a1[e] += __shfl_down_sync(0xffffffffu, a1[e], off);
        }
    if (lane == 0)
        for (int e = 0; e < ne; ++e) { out[(size_t)i0 * 8 + e] = a0[e]; out[(size_t)(i0 + 1) * 8 + e] = a1[e]; }
}

int main() {
    // tiny deterministic check of the bias-folded LUT: codes all 3, x all 1, scale 1.0 -> 2560 * 2
    {
        const size_t tsz = O_GU_SCALES + (size_t)2 * FF * 80;
        std::vector<uint8_t> tiny(tsz, 0xff);
        for (size_t i = O_GU_SCALES; i < tsz; i += 2) { tiny[i] = 0x00; tiny[i + 1] = 0x3c; }
        uint8_t* dt; cudaMalloc(&dt, tsz);
        cudaMemcpy(dt, tiny.data(), tiny.size(), cudaMemcpyHostToDevice);
        int8_t* dx1; cudaMalloc(&dx1, 8 * NC * 32);
        std::vector<int8_t> x1(8 * NC * 32, 1);
        cudaMemcpy(dx1, x1.data(), x1.size(), cudaMemcpyHostToDevice);
        float* ddx1; cudaMalloc(&ddx1, 8 * NC * 4);
        std::vector<float> s1(8 * NC, 1.0f);
        cudaMemcpy(ddx1, s1.data(), s1.size() * 4, cudaMemcpyHostToDevice);
        int* dhx1; cudaMalloc(&dhx1, 8 * NC * 4);
        std::vector<int> hx1(8 * NC, 32);
        cudaMemcpy(dhx1, hx1.data(), hx1.size() * 4, cudaMemcpyHostToDevice);
        float* o; cudaMalloc(&o, 40960);
        cudaDeviceSynchronize();
        gu<0><<<80, 256>>>(dt, dx1, ddx1, dhx1, o, 1);
        cudaDeviceSynchronize();
        float v0; cudaMemcpy(&v0, o, 4, cudaMemcpyDeviceToHost);
        gu<1><<<80, 256>>>(dt, dx1, ddx1, dhx1, o, 1);
        cudaDeviceSynchronize();
        float v1; cudaMemcpy(&v1, o, 4, cudaMemcpyDeviceToHost);
        gu<2><<<80, 256>>>(dt, dx1, ddx1, dhx1, o, 1);
        cudaDeviceSynchronize();
        float v2; cudaMemcpy(&v2, o, 4, cudaMemcpyDeviceToHost);
        gu_mma<<<10, 256>>>(dt, dx1, ddx1, dhx1, o);
        cudaDeviceSynchronize();
        float v3; cudaMemcpy(&v3, o, 4, cudaMemcpyDeviceToHost);
        fprintf(stderr, "bias check (expect 5120.00): dp4a=%.2f lut=%.2f lut+bias=%.2f mma=%.2f\n", v0, v1, v2, v3);
        cudaFree(dt); cudaFree(dx1); cudaFree(ddx1); cudaFree(dhx1); cudaFree(o);
    }
    fprintf(stderr, "p2 start\n");
    const int NE = 48, MAXE = 8;
    const size_t blob = 1382400;
    fprintf(stderr, "host blobs\n");
    std::vector<uint8_t> hb(blob);
    fprintf(stderr, "device blob\n");
    uint8_t* db; cudaMalloc(&db, blob * (size_t)NE);
    fprintf(stderr, "malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
    for (size_t i = 0; i < blob; ++i) hb[i] = (uint8_t)(rand() & 0xff);
    fprintf(stderr, "fill1\n");
    for (size_t i = O_GU_SCALES; i < O_GU_SCALES + (size_t)2 * FF * 80; i += 2) { hb[i] = 0x3c; hb[i + 1] = (uint8_t)(0x30 + (rand() & 15)); }
    fprintf(stderr, "fill2\n");
    cudaMemcpy(db, hb.data(), blob, cudaMemcpyHostToDevice);
    fprintf(stderr, "copy blob\n");
    for (int x = 0; x < NE; ++x) cudaMemcpy(db + (size_t)x * blob, hb.data(), blob, cudaMemcpyHostToDevice);
    fprintf(stderr, "copied %d blobs %s\n", NE, cudaGetErrorString(cudaGetLastError()));
    std::vector<int8_t> hb8((size_t)MAXE * NC * 32);
    for (size_t i = 0; i < hb8.size(); ++i) hb8[i] = (int8_t)((rand() % 255) - 128);
    fprintf(stderr, "fill3\n");
    std::vector<float> hd((size_t)MAXE * NC, 1.0f);
    std::vector<int> hh((size_t)MAXE * NC, 0);
    for (int e = 0; e < MAXE; ++e) for (int c = 0; c < NC; ++c) { int s = 0; for (int k = 0; k < 32; ++k) s += hb8[(size_t)e * NC * 32 + (size_t)c * 32 + k]; hh[(size_t)e * NC + c] = s; }

    fprintf(stderr, "filled\n");
    cudaMemcpy(db, hb.data(), blob, cudaMemcpyHostToDevice);
    fprintf(stderr, "copied blob\n");
    int8_t* dxs; cudaMalloc(&dxs, hb8.size()); cudaMemcpy(dxs, hb8.data(), hb8.size(), cudaMemcpyHostToDevice);
    float* ddx; cudaMalloc(&ddx, hd.size() * 4); cudaMemcpy(ddx, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice);
    int* dhx; cudaMalloc(&dhx, hh.size() * 4); cudaMemcpy(dhx, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice);
    float* got; cudaMalloc(&got, (size_t)2 * FF * 8 * 4);
    fprintf(stderr, "copied all\n");
    cudaDeviceSynchronize();
    fprintf(stderr, "setup: %s\n", cudaGetErrorString(cudaGetLastError()));
    printf("setup ok\n"); fflush(stdout);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const char* nm[5] = {"dp4a unpack", "dp4a lut", "dp4a lut+bias", "mma s8", "dp4a ldcs"};
    for (int ne = 1; ne <= MAXE; ++ne) {
        for (int v = 0; v < 5; ++v) {
            fprintf(stderr, "variant %d ne %d\n", v, ne);
            auto run = [&]() {
                const uint8_t* b = db;
                switch (v) {
                    case 0: gu<0><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 1: gu<1><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 2: gu<2><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 3: gu_mma<<<10, 256>>>(b, dxs, ddx, dhx, got); break;
                    default: gu_ldcs<<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                }
            };
            for (int i = 0; i < 3; ++i) run();
            cudaDeviceSynchronize();
            cudaError_t er = cudaGetLastError();
            if (er != cudaSuccess) { printf("%s: %s\n", nm[v], cudaGetErrorString(er)); fflush(stdout); continue; }
            cudaEventRecord(e0);
            for (int r = 0; r < 20; ++r) for (int x = 0; x < NE; ++x) {
                const uint8_t* b = db + (size_t)x * blob;
                switch (v) {
                    case 0: gu<0><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 1: gu<1><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 2: gu<2><<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                    case 3: gu_mma<<<10, 256>>>(b, dxs, ddx, dhx, got); break;
                    default: gu_ldcs<<<80, 256>>>(b, dxs, ddx, dhx, got, ne); break;
                }
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            std::vector<float> h((size_t)2 * FF * 8);
            cudaMemcpy(h.data(), got, h.size() * 4, cudaMemcpyDeviceToHost);
            fprintf(stderr, "  readback err=%s h[0]=%g\n", cudaGetErrorString(cudaGetLastError()), h[0]);
            double mx = 0, sum = 0;
            for (size_t i = 0; i < h.size(); ++i) { if (fabs(h[i]) > mx) mx = fabs(h[i]); sum += h[i]; }
            // host reference for row 0, entry 0
            double refv = 0;
            for (int c = 0; c < NC; ++c) {
                int s = 0, hh2 = 0;
                for (int k = 0; k < 32; ++k) {
                    int code = (hb[(size_t)c * 8 + k / 4] >> (2 * (k % 4))) & 3;
                    int x = hb8[(size_t)c * 32 + k];
                    s += code * x; hh2 += x;
                }
                uint16_t sh = (uint16_t)(hb[O_GU_SCALES + (size_t)(c >> 1) * 2] | (hb[O_GU_SCALES + (size_t)(c >> 1) * 2 + 1] << 8));
                unsigned bits = (unsigned)sh;
                int exp = (int)((bits >> 10) & 0x1F) - 15;
                double mant = 1.0 + (double)(bits & 0x3FF) / 1024.0;
                double dw = (exp >= 0 ? mant * (1 << exp) : mant / (1 << (-exp)));
                refv += dw * hd[c] * (double)(s - hh2);
            }
            printf("ne=%d  %-14s %7.1f us/expert   out[0]=%.2f ref=%.2f  max=%.1f\n",
                   ne, nm[v], ms * 1000.0 / (20 * NE), h[0], refv, mx);
            fflush(stdout);
        }
    }
    return 0;
}
