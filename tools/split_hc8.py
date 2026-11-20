import io

src = r"C:\Neurolib\Tools\basalt-ampere-3060\src\models\qwen4exp\kernels\hc_fused.cu"
dst = r"C:\Neurolib\Tools\basalt-ampere-3060\src\models\qwen4exp\kernels\hc8_copy.cu"

with io.open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()

start = 1212 - 1
end = 1407  # exclusive slice end -> lines[1211:1407] are 1212..1407
block = lines[start:end]

head = """// src/models/qwen4exp/kernels/hc8_copy.cu - the 8-bit copies of the hyper-connection matrices, split out of
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

"""

with io.open(dst, "w", encoding="utf-8") as f:
    f.write(head)
    f.writelines(block)
    f.write("\n}  // namespace basalt::kernels\n")

rest = lines[:start] + lines[end:]
with io.open(src, "w", encoding="utf-8") as f:
    f.writelines(rest)

print("moved", len(block), "lines; hc_fused.cu now", len(rest), "lines")
print("first moved:", block[0].strip()[:60])
print("last moved:", block[-1].strip()[:60])
