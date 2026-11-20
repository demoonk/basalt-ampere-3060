// include/basalt/core/gpu_arch.hpp - the compute capabilities this fork builds for.
//
// Upstream basalt is Blackwell-only (sm_120).  This fork adds the Ampere tier (sm_86), where the cluster
// barriers, tensor-map loads and the FP4/FP8 block-scaled matrix products are compiled out and the kernels
// that still run take the plain Ampere paths.  A card outside the two tiers is refused before its first
// kernel launch rather than failing inside one.
#pragma once

namespace basalt::core {

inline bool supported_compute_capability(int major, int minor) {
    if (major == 12) return true;
    if (major == 8) return minor == 6 || minor == 9;
    return false;
}

inline bool blackwell_compute_capability(int major, int minor) { return major >= 12; }

}  // namespace basalt::core
