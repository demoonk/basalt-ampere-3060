// include/basalt/kernels/device_caps.hpp - what the card in front of a launch can actually execute.
//
// Basalt's kernels are written for Blackwell (sm_120): tensor-map loads, cluster barriers, the FP4 and
// block-scaled FP8 matrix products.  An Ampere build (sm_86) compiles those paths out of the device code,
// so every launch that reaches one has to ask the device first.  These checks are per device and cached.
#pragma once
#include <cuda_runtime.h>
#include <mutex>

namespace basalt::kernels {

/// The compute capability of `dev` (-1: the current device); -1 when the query failed.
inline int device_cc(int dev = -1) {
    static int cache[64] = {};
    static std::mutex mu;
    if (dev < 0) {
        int d = 0;
        if (cudaGetDevice(&d) != cudaSuccess) { (void) cudaGetLastError(); return -1; }
        dev = d;
    }
    if (dev < 0 || dev >= 64) return -1;
    std::lock_guard<std::mutex> lock(mu);
    if (cache[dev] != 0) return cache[dev];
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    cache[dev] = major * 10 + minor;
    return cache[dev];
}

/// Thread-block clusters (compute capability 9.0 and up): the cluster barriers and distributed shared memory.
inline bool device_clusters(int dev = -1) { return device_cc(dev) >= 90; }

/// The sm_120a instructions: NVFP4 and block-scaled FP8 matrix products.
inline bool device_fp4(int dev = -1) { return device_cc(dev) >= 120; }

/// An Ampere card: the tier this fork's kernels fall back to.
inline bool ampere_device(int dev = -1) { const int cc = device_cc(dev); return cc >= 80 && cc < 90; }

}  // namespace basalt::kernels
