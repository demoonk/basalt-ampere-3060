<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

Basalt is a local inference engine for large language models: **Qwen3.8-Flash-Next**, **DeepSeek-V4.1-Flash** and **Qwen3.8-27B**. It runs
on an NVIDIA Blackwell GPU (sm_120) on Linux and Windows x64, with what does not fit in VRAM (the experts of a mixture-of-experts model)
in system RAM, and optionally on more GPUs. The server speaks the OpenAI and Anthropic APIs. Website: [basalt.jgulfo.com](https://basalt.jgulfo.com).

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

**Help wanted:** spare Blackwell compute for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

## This fork: the RTX 3060 (sm_86, 12 GB) branch

Everything measured on an Ampere 3060 lives here, in plain markdown:

| file | what it holds |
|---|---|
| [FORK-3060-v0.5.0.md](FORK-3060-v0.5.0.md) | what the fork changes for sm_86 and how to build it |
| [SPEEDUP-3060.md](SPEEDUP-3060.md) | round 1: the fork, the first measured speed-ups, the `--mmap-experts` finding |
| [SPEEDUP2-3060.md](SPEEDUP2-3060.md) | round 2: S1–S8, the 128K context, `--route-prior`, the crash analysis, the ready-to-run shortcuts |
| [basalt-3060-edits.md](basalt-3060-edits.md) | the code edits, one by one |
| [basalt-3060-speedup-task.md](basalt-3060-speedup-task.md), [task 2](basalt-3060-speedup-task-2.md) | the task statements the two rounds were written against |

Run it with `serve-3060.cmd` / `serve-3060.json` (and the `safe`, `16k`, `65k`, `spec8`,
`stable` and `nommap` variants); the raw run logs of both rounds are in `logs-speedup/` and
`logs-speedup2/`.

## Requirements

- Linux x86-64 with AVX2, or Windows 10 1903+ x64 ([platforms](docs/platforms.md)).
- An NVIDIA Blackwell GPU (compute capability 12.0) with CUDA 13.
- CMake 3.24+, GCC 13+ (Linux), Python 3.10+, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

Prebuilt Linux and Windows archives are attached to each [release](https://github.com/jesdga95/basalt/releases) ([what they need](docs/platforms.md#release-archives)); otherwise build from source:

```sh
git clone https://github.com/jesdga95/basalt && cd basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
                                        # Windows: add -G Ninja to the first line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
                                        # one model only: -DBASALT_MODELS=qwen4exp, deepseek41 or qwen35 ([building](docs/building.md#building-only-some-models))
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4-DQ6_K.basalt --local-dir models
```

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-NVFP4-DQ6_K.basalt", "gpu": "0,1", "kv": "int8" }],
  "engine": { "resident-experts": true },
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Models

| | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash | Qwen3.8-27B | Hy4-flash |
|---|---|---|---|---|
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) | default `DeepSeek-V4.1-Flash-1017-UD-IQ1_M`, [Hugging Face](https://huggingface.co/jesdga/DeepSeek-V4.1-Flash-1017-Basalt) | `Qwen3.8-27B-QUASAR-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-27B-Basalt) | `Hy4-flash-NVFP4`, [Hugging Face](https://huggingface.co/jesdga/Hy4-flash-Basalt) |
| Context | up to 262,144 (longer with YaRN) | up to 1,048,576 | up to 262,144 | up to 1,048,576 |
| Drafting | MTP, on by default | DSpark, on by default | DFlash 2, on by default | DFly2, on by default |
| Images | yes | yes | yes | yes |
| Documents (PDF, text) | yes | yes | yes | yes |
| Guide | [guide](docs/models/qwen38-flash-next.md) | [guide](docs/models/deepseek41.md) | [guide](docs/models/qwen38-27b.md) | [guide](docs/models/hy4-flash.md) |
| Precision | [precision](docs/precision/qwen38-flash-next.md) | [precision](docs/precision/deepseek41.md) | [precision](docs/precision/qwen38-27b.md) | [quality](docs/precision/quality.md) |

All models compared, with each one's context limit, flags and cards: [models](docs/models/index.md).

## Video

Qwen3.8-Flash-Next and Qwen3.8-27B read video; DeepSeek and Hy4-flash do not. A video is named like an image in a chat message, and needs
`--vision` and FFmpeg on the machine. Flash-Next decodes frames on the GPU with NVDEC; the 27B decodes them on the CPU.
How a request names a video, and the limits: [video](docs/running.md#video).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Flash-Next](docs/models/qwen38-flash-next.md), [DeepSeek](docs/models/deepseek41.md),
  [Qwen3.8-27B](docs/models/qwen38-27b.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- **Code:** [Strata](https://github.com/Niko1221/Strata), the project Basalt forks (MIT); [Infernix](https://github.com/Wallawalla47/Infernix)
  (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT); and ideas from [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Qwen](https://huggingface.co/Qwen) and [DeepSeek](https://huggingface.co/deepseek-ai), under the licences on their cards;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints;
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 27B (Apache-2.0); [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter.

Thank you to everyone who has [contributed](https://github.com/jesdga95/basalt/graphs/contributors) or
[sponsored](https://github.com/sponsors/jesdga95) the project.

## License

MIT, see [LICENSE](LICENSE).
