// C:\Neurolib\Tools\tmp\bw.cpp - DRAM ceiling probe for HomePC (Ryzen 7 1700, DDR4 dual-channel).
// Two patterns: (1) sequential streaming, (2) blob-like (1.382400 MB blocks, shuffled block order),
// which is what the CPU expert pool actually does.  Run outside the repo, single process.
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>
#include <random>

static const size_t BLOB = 1382400;          // one Q2_0 expert blob
static const size_t TOTAL = 32ull * BLOB;    // ~43 MiB working set per thread group

static double run_seq(int threads, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    for (size_t i = 0; i < bytes; ++i) buf[i] = (uint8_t)(i & 0xff);
    std::vector<uint64_t> acc(threads, 0);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t]() {
            size_t lo = bytes / threads * (size_t)t, hi = (t + 1 == threads) ? bytes : bytes / threads * (size_t)(t + 1);
            uint64_t s = 0;
            for (size_t i = lo; i < hi; ++i) s += buf[i];
            acc[t] = s;
        });
    for (auto& th : ts) th.join();
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)bytes / dt / 1e9;
}

static double run_blobs(int threads, int reps) {
    // a ~25 GiB pool of blobs, read in shuffled order: the real pool's access pattern
    std::vector<uint8_t> pool(8ull * 1024 * 1024 * 1024);   // 8 GiB: far beyond the 16 MB L3
    for (size_t i = 0; i < pool.size(); ++i) pool[i] = (uint8_t)(i & 0xff);
    size_t nblobs = pool.size() / BLOB;
    std::vector<size_t> order(nblobs);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(12345);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<uint64_t> acc(threads, 0);
    size_t total = nblobs * BLOB * (size_t)reps;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t]() {
            uint64_t s = 0;
            for (int r = 0; r < reps; ++r)
                for (size_t j = (size_t)t; j < nblobs; j += (size_t)threads) {
                    const uint8_t* p = pool.data() + order[j] * BLOB;
                    for (size_t i = 0; i < BLOB; ++i) s += p[i];
                }
            acc[t] = s;
        });
    for (auto& th : ts) th.join();
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)total / dt / 1e9;
}

int main() {
    for (int t : {1, 2, 4, 7, 8, 16})
        std::printf("sequential  %2d threads: %.2f GB/s\n", t, run_seq(t, 2ull * 1024 * 1024 * 1024 / 4));
    for (int t : {7, 8, 16})
        std::printf("blob-shuffle %2d threads: %.2f GB/s\n", t, run_blobs(t, 1));
    return 0;
}
