// src/models/qwen4exp/kernels/hc8_copy.cu - the 8-bit copies of the hyper-connection matrices, split out of
// hc_fused.cu so Ampere builds can use them: the fused read in hc_fused.cu needs TMA and sm_120a instructions,
// while these quantize/dequantize kernels are plain enough for sm_86.
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace basalt::kernels {
namespace {

constexpr int G8 = 32;

// 8-bit copies: q = round(w / s) in [-127, 127], s = f16(absmax / 127) per G8 consecutive values of a row; a row's
// `gpr` scales start every `srow` f16 (a multiple of 8: 16-byte aligned runs).  `err` (optional): the sums of w^2
// and of the error^2.
__global__ void hc8_quant_kernel(const uint16_t* w, int64_t groups, int gpr, int srow, int8_t* q, uint16_t* sc,
                                 double* err) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    double e[2] = {0.0, 0.0};
    if (gi < groups) {
        float x[G8], amax = 0.0f;
        for (int i = 0; i < G8; ++i) {
            x[i] = __uint_as_float((uint32_t) w[gi * G8 + i] << 16);
            amax = fmaxf(amax, fabsf(x[i]));
        }
        const __half h = __float2half_rn(amax / 127.0f);
        const float s = __half2float(h);
        sc[gi / gpr * srow + gi % gpr] = __half_as_ushort(h);
        for (int i = 0; i < G8; ++i) {
            const float v = s > 0.0f ? fminf(fmaxf(rintf(x[i] / s), -127.0f), 127.0f) : 0.0f;
            q[gi * G8 + i] = (int8_t) v;
            e[0] += (double) x[i] * x[i];
            e[1] += (double) (v * s - x[i]) * (v * s - x[i]);
        }
    }
    if (err != nullptr)
        for (int k = 0; k < 2; ++k) {
            double v = e[k];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if ((threadIdx.x & 31) == 0) atomicAdd(&err[k], v);
        }
}
struct Hc8Copy {
    int8_t* q = nullptr;   // then the scales, in the same allocation
    uint16_t* s = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0, srow = 0;
    bool owned = false;    // keyed by q itself: no bf16 matrix behind it
};
std::mutex g_hc8_mu;
std::unordered_map<const void*, Hc8Copy> g_hc8;
size_t g_hc8_bytes = 0;

// The copies are carved out of large allocations: one cudaMalloc per matrix rounds each up to the allocator's
// granularity, which costs a fraction of a MiB per matrix.  Offsets stay 256-byte aligned, as separate allocations are.
// Copies are never returned, so a chunk is never freed.
cudaError_t hc8_alloc(void** out, size_t bytes) {
    constexpr size_t kChunk = (size_t) 32 << 20, kAlign = 256;
    static char* bases[16] = {};
    static size_t lefts[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return cudaMalloc(out, bytes);
    char*& base = bases[dev];
    size_t& left = lefts[dev];
    bytes = (bytes + kAlign - 1) / kAlign * kAlign;
    if (bytes > left) {
        if (bytes > kChunk / 2) return cudaMalloc(out, bytes);
        void* p = nullptr;
        const cudaError_t e = cudaMalloc(&p, kChunk);
        if (e != cudaSuccess) return e;
        base = (char*) p;
        left = kChunk;
    }
    *out = base;
    base += bytes;
    left -= bytes;
    return cudaSuccess;
}

// the copy of the bf16 matrix w (device memory of the current card); false when there is no VRAM for it
bool hc8_make(const uint16_t* w, int rows, int cols, Hc8Copy& c) {
    const int64_t n = (int64_t) rows * cols, groups = n / G8;
    const int gpr = cols / G8, srow = (gpr + 7) & ~7;
    c.bytes = (size_t) n + (size_t) rows * srow * 2;
    c.rows = rows;
    c.cols = cols;
    c.srow = srow;
    if (hc8_alloc((void**) &c.q, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        basalt::log::error("basalt hc8", "no VRAM for an 8-bit copy (%zu bytes)", c.bytes);
        return false;
    }
    c.s = reinterpret_cast<uint16_t*>(c.q + n);
    cudaMemset(c.s, 0, (size_t) rows * srow * 2);
    hc8_quant_kernel<<<(unsigned) ((groups + 255) / 256), 256>>>(w, groups, gpr, srow, c.q, c.s, nullptr);
    cudaDeviceSynchronize();
    return true;
}

__global__ void hc8_dequant_bf16_kernel(const int8_t* q, const uint16_t* s, int cols, int srow, int64_t row0, int64_t n,
                                        uint16_t* out) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;   // 8 values, one scale group
    if (i >= n * cols) return;
    const int64_t r = row0 + i / cols;
    const int k = (int) (i % cols);
    const float sc = __half2float(__ushort_as_half(s[r * srow + k / G8]));
    const uint2 w = *reinterpret_cast<const uint2*>(q + r * cols + k);
    uint16_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = (int) (int8_t) ((j < 4 ? w.x : w.y) >> (8 * (j & 3)));
        o[j] = __bfloat16_as_ushort(__float2bfloat16_rn((float) v * sc));
    }
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
}

}  // namespace

bool hc8_enabled() { return true; }

bool hc8_only() { return true; }

void hc8_prepare(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    if (g_hc8.count(w)) return;
    int dev = 0;
    cudaPointerAttributes pa{};
    cudaGetDevice(&dev);
    if (cudaPointerGetAttributes(&pa, w) != cudaSuccess || pa.type != cudaMemoryTypeDevice || pa.device != dev) {
        cudaGetLastError();
        return;   // another card's matrix: its reads stay bf16
    }
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return;   // that read stays bf16
    g_hc8[w] = c;
    g_hc8_bytes += c.bytes;
}

const void* hc8_adopt(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return nullptr;
    c.owned = true;
    g_hc8[c.q] = c;
    g_hc8_bytes += c.bytes;
    return c.q;
}

const void* hc8_register(const uint16_t* w, int rows, int cols, int8_t** q, uint16_t** s) {
    *q = nullptr;
    *s = nullptr;
    if (rows <= 0 || cols <= 0 || cols % G8 != 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    if (w != nullptr && g_hc8.count(w)) return w;
    const int64_t n = (int64_t) rows * cols;
    Hc8Copy c;
    c.rows = rows;
    c.cols = cols;
    c.srow = ((cols / G8) + 7) & ~7;
    c.bytes = (size_t) n + (size_t) rows * c.srow * 2;
    if (hc8_alloc((void**) &c.q, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        basalt::log::error("basalt hc8", "no VRAM for an 8-bit copy (%zu bytes)", c.bytes);
        return nullptr;
    }
    c.s = reinterpret_cast<uint16_t*>(c.q + n);
    c.owned = w == nullptr;
    const void* key = w != nullptr ? (const void*) w : (const void*) c.q;
    g_hc8[key] = c;
    g_hc8_bytes += c.bytes;
    *q = c.q;
    *s = c.s;
    return key;
}

bool hc8_lookup(const void* w, const int8_t** q, const uint16_t** s, size_t* bytes) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    *q = it->second.q;
    *s = it->second.s;
    if (bytes != nullptr) *bytes = it->second.bytes;
    return true;
}

bool hc8_view(const void* w, Hc8View& v) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    const Hc8Copy& c = it->second;
    v.q = c.q; v.s = c.s; v.rows = c.rows; v.cols = c.cols; v.srow = c.srow; v.bytes = c.bytes; v.owned = c.owned;
    return true;
}

void hc8_dequant_bf16(const Hc8View& v, int64_t row0, int64_t n, uint16_t* out, void* stream) {
    if (n <= 0) return;
    const int64_t units = n * v.cols / 8;
    hc8_dequant_bf16_kernel<<<(unsigned) ((units + 255) / 256), 256, 0, (cudaStream_t) stream>>>(v.q, v.s, v.cols, v.srow,
                                                                                              row0, n, out);
}

size_t hc8_bytes() {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    return g_hc8_bytes;
}

}  // namespace basalt::kernels
