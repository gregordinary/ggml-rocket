# ggml-rocket

## AI disclosure

Except for the prior work it builds on, ggml-rocket was developed by AI, primarily Claude. Human
involvement was mostly limited to setting project goals and providing hardware access. This is a
side project for curiosity's sake, and it comes with no guarantee of quality, accuracy, or update
frequency.

## About ggml-rocket

ggml-rocket is a drop-in [ggml](https://github.com/ggml-org/ggml) backend for Rockchip NPUs
(validated on the RK3588). It offloads LLM and Whisper prefill to the NPU through the mainline
`rocket` DRM-accel driver.

It builds as a runtime-loadable `libggml-rocket.so` for stock llama.cpp, whisper.cpp and other
ggml hosts, such as transcribe.cpp, a multi-model speech-to-text library. Point
`GGML_BACKEND_PATH` at it, and the NPU appears as a ggml device, as ggml's own BLAS backend does.
It runs the Whisper encoder end to end on the NPU, and the prefill of Gemma-4, Qwen3.5, Qwen3.6,
Llama-3.2, Phi-4, Ministral and more.

Decode stays on the CPU. A decode step is single-row work, bound by the DDR bandwidth the NPU
shares with the CPU. Below `ROCKET_MIN_M` (default 32 rows), the dispatch and the weight pack
cost more than the NPU saves.

The backend links `librocketnpu`, which the `rocket-userspace` project builds and which also runs
on its own. The build expects the checkouts side by side:

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  ggml/               # fresh upstream clone: git clone https://github.com/ggml-org/ggml
  ggml-rocket/        # this project
```

The other documents in this repository are:

- [TUNING.md](TUNING.md): the configuration for each workload, and what each setting is worth
- [MODELS.md](MODELS.md): the results and accuracy for each model and precision
- [MOE.md](MOE.md): mixture-of-experts models, their placement and their memory budget
- [RK3576.md](RK3576.md): the backend on the RK3576
- [API.md](API.md): every `ROCKET_*` environment variable, the diagnostic logging, the precision
  declarations and the implementation notes
- [MEASUREMENTS.md](MEASUREMENTS.md): how each figure in these documents was measured

## Quick start

These commands build llama.cpp and the backend side by side, then run a prompt with the NPU
prefilling it. They need an RK3588 board with the `rocket` driver loaded, as
[Requirements](#requirements) describes.

```sh
# 1. latest llama.cpp, shared + DL-capable ggml
cd <workspace>     # the dir holding ggml-rocket/
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
cmake --build llama.cpp/build -j

# 2. libggml-rocket.so against llama.cpp's BUNDLED ggml
cd ggml-rocket
cmake -S . -B build-dl -DGGML_ROCKET_DL=ON -DHOST_DIR=$PWD/../llama.cpp
cmake --build build-dl -j      # if needed: -DGGML_LIB_DIR=/abs/path

# 3. run with the NPU backend loaded (-ngl 0; ACCEL offload is BLAS-style, scheduler-driven)
GGML_BACKEND_PATH=$PWD/build-dl/libggml-rocket.so \
  sudo -E ../llama.cpp/build/bin/llama-cli \
    -m /path/to/gemma4/gemma-4-12b-it-F16.gguf \
    -p "Explain the RK3588 NPU in two sentences." \
    -n 64 -c 4096 -ngl 0 -b 512
```

Run it under `sudo -E`: `/dev/accel/accel0` needs privilege, and `-E` keeps `GGML_BACKEND_PATH`
in the environment. The startup log lists a `ROCKET` device whose description names the detected
part and driver, such as `RK3588 NPU (mainline rocket driver)`. Cap `-c` so that the weights and
the KV cache fit in RAM.

An F16 GGUF takes the fp16 [route](#terms) with no conversion. A quantized GGUF (`Q8_0`, `Q4_K`,
`Q6_K` and the rest) also prefills on the NPU, dequantized to fp16 on the fly, with no whole-model
F16 copy. A BF16 GGUF converts to fp16 on every micro-batch, at ~0.55-0.6x the fp16 prefill rate,
so convert it once with `llama-quantize in.gguf out.gguf f16`.

> [!CAUTION]
> For a quantized GGUF, turn the host's weight repack off: pass `-nr` (`--no-repack`), or
> `--repack 0` to `llama-bench`. With the repack on, the quantized weights stay on the CPU, and
> nothing reports it.
>
> The CPU backend repacks quantized weights into a non-host `CPU_REPACK` buffer for its own
> SIMD kernels. ggml's scheduler offloads only an op whose weight is in a host buffer. A host
> built with `-DGGML_CPU_REPACK=OFF` also works, but it removes the repack from CPU-only runs
> too, where it is worth 1.3-1.7x on quantized prefill. Confirm the offload with
> `ROCKET_MM_PROFILE=1`.
>
> A host that registers each repacked weight's original bytes with the backend can leave the
> repack on, but llama.cpp and whisper.cpp do not register them.
> [API.md](API.md#repacked-weights) describes the call.

Gemma-4's chat template needs `--jinja`, or use `llama-completion -no-cnv` for raw prompt
completion. [TUNING.md](TUNING.md) gives the settings for each workload.

## What runs on the NPU

The backend offloads the ops that dominate prefill, and the CPU backend computes every other op:

- **`MUL_MAT`**: the prefill GEMMs. The backend [claims](#terms) a matmul with an F16, F32 or
  quantized weight if `M >= ROCKET_MIN_M` and the shape aligns (`K%32`, `N%16`, `K>=64`,
  `N>=64`).
- **`FLASH_ATTN_EXT`**: prefill attention, on the NPU by default at `n_kv >= 1024`.
- **`MUL_MAT_ID`**: MoE routed experts, on the NPU by default as [native-quant experts](#terms).
  The backend claims a quantized [expert stack](#terms) that fits the memory budget and whose
  per-expert GEMM repays its dispatch. Any other stack runs whole on the CPU. The first
  prefill of each `llama_context` converts the experts once, which [MOE.md](MOE.md) times.
- **Opt-in routes**: native int8 (`ROCKET_INT8=1`), int4 (`ROCKET_INT4=1`) and bf16
  (`ROCKET_BF16=1`). They fit a model in less RAM or compute an exact fp32 result, but none
  prefills faster than fp16.
- **Precision declarations**: the backend declines a matmul whose model declares a precision the
  backend cannot meet, such as Mistral-4's F32 routed down-projection activation.
  [API.md](API.md#precision-declarations) has the rules.
- **Everything else** stays on the CPU: norms, rope, the conv front-end and decode.

## Performance

The backend speeds up prefill, a batched GEMM. At `pp2048` it prefills a dense F16 model at
4.2-5.8x the 8-thread CPU backend. A dense `Q4_K_M` model gains 3.8-4.0x, and a MoE model
1.8-2.3x.
Decode runs on the CPU, where it is bound by memory bandwidth and its rate follows the
quantization, not the backend.

| Model | Params | GGUF | NPU prefill, `pp2048`, t/s | Speedup over the CPU, `pp2048` | Q4 decode, t/s | Q4 GGUF size, GiB |
|---|---:|---|---:|---:|---:|---:|
| SmolVLM2-2.2B (multimodal) | 1.8B | F16 | 117.8 | 4.2x | 20.7 | 1.0 |
| Llama-3.2-3B | 3.2B | F16 | 85.1 | 5.0x | 11.1 | 1.9 |
| Ministral-3-3B | 3.4B | F16 | 75.0 | 4.7x | 10.5 | 2.0 |
| Phi-4-mini | 3.8B | F16 | 76.1 | 5.0x | 8.8 | 2.3 |
| Ministral-3-8B | 8.5B | F16 | 39.6 | 5.8x | 4.7 | 4.8 |
| Qwen3.5-9B | 9.0B | F16 | 36.1 | 4.9x | 4.0 | 5.3 |
| Gemma-4-12B | 11.9B | F16 | 24.6 | 5.2x | 3.1 | 6.9 |
| Phi-4 (14B) | 14.7B | Q4_K_M | 21.9 | 4.0x | 2.7 | 8.3 |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | Q4_K_M | 43.4 | 1.83x | 13.5 | 9.7 |
| gpt-oss-20b (MoE) | 20.9B | MXFP4 | 37.8 | 2.29x | 9.1 | 11.3 |
| Qwen3.6-27B (hybrid) | 27.3B | Q4_K_M | 11.5 | 3.8x | 1.3 | 15.9 |

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), CPU governor `performance`, warm runs,
llama.cpp b11539 [HW sweep 2026-11-01]. Each speedup pairs the NPU and the 8-thread CPU backend on
the same GGUF. The F16 rows run at llama.cpp's default `-b` and `-ub`. The four quantized rows run
at `-b 2048 -ub 2048`, with llama.cpp's weight repack on in the CPU backend. Decode is the CPU
backend at `-t 4` on the four A76 cores (`taskset 0xf0`), `tg64`, which decodes faster than 8
threads, as [TUNING.md](TUNING.md#cpu-threads) gives.

NPU prefill's perplexity agrees with the CPU backend's on every model in the table but
gpt-oss-20b. Wikitext perplexity is not a valid measure for that reasoning model. Its offloaded
expert GEMMs measure a mean cosine of 0.9998 against an fp64 CPU reference instead
[HW sweep 2026-08-27]. [MODELS.md](MODELS.md) gives each model's measured difference.

The MoE rows gain less because more of a MoE model's prefill stays on the CPU: the router,
attention and the expert gathers.

### Speech models

For a speech model the backend offloads the encoder. The Whisper encoder runs 1.18x-2.14x as fast
as on the CPU backend, from `tiny.en` to `large-v3`. Through transcribe.cpp, a whole long-audio
pipeline gains 1.0x-1.9x. It gains most where the encoder is large or the decoder cross-attends
the audio, and least on a small decoder-only model.

**Conditions:** RK3588, warm runs, against the CPU backend on the same model. Whisper at NPU
~816 MHz (the 600 MHz request), transcribe.cpp at `Q8_0`.

### An interactive turn

On Ministral-3-8B-Instruct, the NPU shortens the wait for the first token, and the quantization
sets the decode rate. A 2048-token prompt with a 200-token answer, as in RAG, takes:

| Configuration | Time to first token | Decode, t/s | Total |
|---|---:|---:|---:|
| F16, CPU | 301 s | 1.6 | 426 s |
| F16, NPU | 52 s | 1.6 | 177 s |
| `Q4_K_M`, NPU | 45 s | 4.7 | 88 s |

**Conditions:** the Performance table's rates [host-computed]. The time to first
token is the prompt over the `pp2048` rate, and the total adds 200 tokens at the decode rate.
`Q4_K_M` runs under `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`.

On this prompt the NPU cuts the time to first token 5.8x and the turn 2.4x. A short chat turn is
mostly decode, so the quantization sets its length: 400 tokens take 250 s at F16 and 85 s at
`Q4_K_M`.

Run this model as `Q4_K_M` with the NPU. Under `ROCKET_QUANT_RESIDENT=auto` it holds an fp16 copy
of its weights, 13.8 GB, which needs a board with that RAM free. Without it, the 4.83 GB GGUF fits
an 8 GB board and prefills more slowly, as [TUNING.md](TUNING.md#quantized-ggufs) describes.

## Hardware support

| Part | Status | Scope |
|---|---|---|
| RK3588 | Supported | Every route. The fully validated target. |
| RK3576 | Supported | [The RK3576 route](#terms) only, with `ROCKET_INT8=1`. Attention runs on the CPU. |
| Other Rockchip NPUs | Not characterized | |

On the RK3576 the backend runs an int8 W8A8 matmul with a per-output-column requant, the only
matmul route on that part. It prefills Qwen2.5-1.5B F16 at 1.31x ggml's `Q8_0` CPU kernel, and at
2.37x the same F16 GGUF on the CPU [HW sweep 2026-08-23, `pp2048`]. [RK3576.md](RK3576.md)
documents the route.

> [!CAUTION]
> On the RK3576, set `ROCKET_INT8=1`. Without it the backend declines every `MUL_MAT`, and
> nothing offloads.

## Requirements

- An RK3588 or RK3576 board on a mainline kernel with the `rocket` DRM-accel driver loaded, so
  that `/dev/accel/accel0` exists (`lsmod | grep rocket`).
- `librocketnpu`, from the sibling `rocket-userspace` checkout or installed as a `rocketnpu`
  package.
- A C++20 compiler with `_Float16` in C++, which on aarch64 is GCC 13 or later. The build is
  tested with GCC 13 and GCC 16.
- Privilege to open the accel node. Run with `sudo -E`, because plain `sudo` drops the
  `ROCKET_*` environment variables.

### Clock

The NPU boots at 200 MHz, and the backend computes correctly there. Above 200 MHz the RK3588's
NPU clock follows the rail voltage, so a requested rate is not the rate the NPU runs at.

Apply the `patches/rocket` clock and voltage patches, and load the module with
`rocket_npu_clk_hz=900000000`. On the RK1's 0.80 V rail that request runs the NPU at 928 MHz.

The language-model figures in this README were measured at the 900 MHz request. The speech
figures were measured at the 600 MHz request, which runs at ~816 MHz on the same rail. The 900 MHz
request is 1.044x its speed on matmul device time.

> [!CAUTION]
> With the NPU cores pinned on (`power/control=on`), an `rmmod` at the 900 MHz request
> hard-locks the board, and only a power cycle recovers it. Unload the module only while all
> three cores are runtime-suspended.

## Build and run

To run a model, build the `.so` against the host's ggml, so that it links the same
`libggml-base.so`, and point `GGML_BACKEND_PATH` at it. [Quick start](#quick-start) does both for
llama.cpp, and the recipes below do them for whisper.cpp and transcribe.cpp.

### Drop into whisper.cpp

Upstream whisper.cpp loads `libggml-rocket.so` through `GGML_BACKEND_PATH`, with no source
changes.

```sh
# 1. whisper.cpp with a SHARED, DL-capable ggml (single libggml-base.so the host loads)
cd <workspace>     # the dir holding ggml-rocket/
git clone https://github.com/ggml-org/whisper.cpp
cmake -S whisper.cpp -B whisper.cpp/build -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
cmake --build whisper.cpp/build -j

# 2. libggml-rocket.so against whisper's BUNDLED ggml (so it links the same libggml-base.so)
cd ggml-rocket
cmake -S . -B build-dl -DGGML_ROCKET_DL=ON -DWHISPER_DIR=$PWD/../whisper.cpp
cmake --build build-dl -j      # if ggml shared libs aren't found, add -DGGML_LIB_DIR=/abs/path

# 3. run stock whisper-cli, pointing GGML_BACKEND_PATH at the .so
GGML_BACKEND_PATH=$PWD/build-dl/libggml-rocket.so \
  sudo -E ../whisper.cpp/build/bin/whisper-cli -m /path/ggml-base.en.bin -f /path/audio.wav
```

A `drm_mm "Memory manager not clean"` warning at exit comes from a known teardown race and is
harmless. `whisper-server` loads the `.so` the same way. For a server fed fixed-length chunks,
[TUNING.md](TUNING.md#transcription-service) gives the three server settings that set its CPU
cost.

### Drop into transcribe.cpp

transcribe.cpp, a ggml-based host for several speech-to-text model families, loads
`libggml-rocket.so` the same way. Through it the backend reaches Granite-Speech, Voxtral, MOSS
diarize, SenseVoice, FunASR and Parakeet. Its runner registers the NPU as an ACCEL device and
logs `using accel backend: ROCKET`.

```sh
# transcribe.cpp shared + DL-capable, CPU backend tuned for the A76 (else the CPU baseline is slow)
cmake -B build -DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_GGML_BACKEND_DL=ON \
  -DTRANSCRIBE_VULKAN=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build build -j
cp build/bin/libggml-cpu.so build/src/   # the ARM cpu module lands in build/bin; the backend scan globs build/src

GGML_BACKEND_PATH=$PWD/../ggml-rocket/build-dl/libggml-rocket.so \
  LD_LIBRARY_PATH=build/ggml/src:build/bin \
  sudo -n -E ./build/bin/transcribe-cli -m /path/model.gguf -f audio.wav
```

Build the `.so` against transcribe.cpp's ggml
(`-DGGML_ROCKET_DL=ON -DHOST_DIR=<...>/transcribe.cpp`), as with any host. A
`TRANSCRIBE_GGML_BACKEND_DL` build forces `GGML_NATIVE=OFF`, so the `-DGGML_CPU_ARM_ARCH=…` above
is required for a CPU baseline at full speed.

### Build options

The CMake options are:

- `-DGGML_ROCKET_NATIVE_FP16=OFF`. By default the convert kernels are compiled for the RK3588
  `armv8.2-a+fp16` baseline. The `.so` then stops with `SIGILL` on a core without FP16, such as
  the RK3576's. Turn this OFF for a portable `.so` with scalar convert kernels.
- `-DGGML_ROCKET_DIAGNOSTICS=ON`. Compiles in the per-op recompute and trace diagnostics that
  [API.md](API.md#diagnostics) lists, which the production `.so` leaves out.
- `-DGGML_ROCKET_PORTABLE_GLIBC=ON`. Lets the `.so` load on an older glibc and libstdc++, as
  [Troubleshooting](#version-glibc_238-not-found-at-load) describes.
- `-DGGML_ROCKET_STRICT_ABI=ON`. Fails the build on a host ggml whose
  `GGML_BACKEND_API_VERSION` this backend has not been checked against. By default that build
  continues with a warning.
- `-DGGML_ROCKET_RK3576_BENCH=ON`. Builds the RK3576 GEMM benchmark that
  [Development](#development) describes, which needs OpenBLAS.

## Development

The standalone build links the source tree's ggml and builds the tests:

```sh
# from the workspace root (the dir holding rocket-userspace/, ggml/, ggml-rocket/)
git clone https://github.com/ggml-org/ggml          # if not already present
cd ggml-rocket
cmake -S . -B build              # or: -DGGML_DIR=/path/to/ggml
cmake --build build -j
sudo ./build/test-rocket-matmul  # needs /dev/accel/accel0; each shape prints PASS
```

The tests below are registered with CTest. Run `ctest` from the build directory. The NPU tests
report `Skipped` off the device, and the host-only tests run anywhere. Each NPU test also reads
the route counters `ggml-rocket.h` declares, so it asserts which route computed its ops instead
of inferring it from the output:

| Test | What it covers |
|---|---|
| `test-rocket-matmul` | The backend against the CPU backend on real ggml graphs |
| `test-rocket-moe` | The MoE handler, both the fp16 and the native-quant expert route, against the CPU backend under outlier-channel activations |
| `test-rocket-int4` | The int4 route |
| `test-rocket-bf16` | The bf16 route |
| `test-rk3576-w8a8` | The RK3576 route against the CPU backend. `Skipped` on any other part |
| `test-rocket-placement` | The `supports_op` and `offload_op` contract |
| `test-rocket-fa` | `FLASH_ATTN_EXT` against a double-precision reference, including the logits fp16 cannot hold |
| `test-rocket-internals` | Host-only: the weight cache's identity, the MoE residency ledger across two live backends, the budget parser, the FA dump's file handling, the device-route circuit breaker, the registered sources of repacked weights |
| `test-hadamard-neon`, `test-hadamard-quant` | The Hadamard rotation construction |

`test-common.h` holds the graph and comparison code the tests share.

`test-rk3576-sgemm-ab` is a benchmark rather than a test. It times the RK3576's NPU GEMM against
ggml F32, ggml `Q8_0` and OpenBLAS on the CPU. It builds only with
`-DGGML_ROCKET_RK3576_BENCH=ON`.

> [!CAUTION]
> Running `test-rk3576-sgemm-ab` can take an RK3576 board down hard enough that only a physical
> power cycle recovers it.

> [!NOTE]
> If `find_package(rocketnpu)` resolves to an installed `rocket-userspace` package (from
> `cmake --install`, for example under `/usr/local`), the backend links that library, not the
> sibling source tree. After a change in `rocket-userspace/`, reinstall it
> (`cd ../rocket-userspace && cmake --build build && sudo cmake --install build`) before
> rebuilding here. Or build against the source tree, whatever is installed, with
> `-DROCKETNPU_DIR=$PWD/../rocket-userspace` from this directory.

## Troubleshooting

### ROCKET device not listed at startup

Most often the backend was built against a different ggml from the host's (an ABI mismatch), so
it loads but no NPU device appears. Rebuild the `.so` against that host checkout's ggml, and
rebuild again after every ggml update in the host:

```sh
# HOST_DIR is the host's own checkout: llama.cpp, whisper.cpp or transcribe.cpp
cd ggml-rocket
cmake -S . -B build-dl -DGGML_ROCKET_DL=ON -DHOST_DIR=/abs/path/to/llama.cpp
cmake --build build-dl -j
```

llama.cpp and whisper.cpp each clone their own ggml, and the host's log names the mismatch:

- `incompatible API version (backend: N, current: M)`: the host's `GGML_BACKEND_API_VERSION`
  differs from the one the backend was built against, and ggml's loader refuses the `.so`.
- `ggml-rocket: op ordinal N is X in this .so and Y in the host's ggml`, followed by
  `ggml_backend_init returned NULL`: ggml's op numbering moved, which the API version does not
  track. The backend checks the numbering at load and refuses to register.

This backend is checked against `GGML_BACKEND_API_VERSION` 2 and 3, and another version builds
with a warning. It builds against the ggml of whisper.cpp 1.8.6 through 1.9.5 and of llama.cpp
b10558 through b11539. llama.cpp moved to version 3 at b11351, and b11351 through b11401 label
that ggml 0.25.3. Read a host's API version from its `ggml/src/ggml-backend-impl.h`, not from
the ggml version.

ggml 1.0.0, the ggml of llama.cpp v1.0.0, keeps API version 3 but removes three ops. That
renumbers the ops after them, `MUL_MAT` from 29 to 28. A 1.0.0 host refuses a `.so` built
against an earlier ggml, and an earlier host refuses a 1.0.0 one. Both log the op-ordinal line
above, and the cure is to rebuild against the host. The source needs no change: it compiles
against ggml 1.0.0 with `-DGGML_ROCKET_STRICT_ABI=ON` [host-computed 2026-11-14]. It has not
been run in a llama.cpp v1.0.0 host.

If the device appears but runs zero matmuls, the backend was built without a working
`offload_op`. See [Implementation notes](API.md#implementation-notes).

### `version 'GLIBC_2.38' not found` at load

`libggml-rocket.so` was built on a host with a newer glibc (2.38 or later, such as Debian trixie
or forky) than the runtime it is deployed to (such as a Debian 12 bookworm container, glibc
2.36). The message names `GLIBC_2.43` instead when the build host has glibc 2.43, and
`GLIBCXX_3.4.31` when the target's libstdc++ is older than the build host's. Rebuild it with
`-DGGML_ROCKET_PORTABLE_GLIBC=ON`:

```sh
cd ggml-rocket
cmake -S . -B build-dl -DGGML_ROCKET_DL=ON -DGGML_ROCKET_PORTABLE_GLIBC=ON \
  -DHOST_DIR=/abs/path/to/llama.cpp
cmake --build build-dl -j
```

The option links forwarders for glibc 2.38's `__isoc23_*` symbols and glibc 2.43's `sqrtf` into
the `.so`, so it needs only glibc 2.34. It also holds the `.so` to the libstdc++ of GCC 12
(`GLIBCXX_3.4.30`), which Debian 12 provides. If the linked `.so` still needs a newer glibc or
libstdc++, the build fails and names the symbol. The option takes effect in DL mode only, and a
same-host build is byte-unchanged.

A Debian 12 host cannot build the backend for the board itself, because GCC 12 accepts `_Float16`
in C++ only on x86 (the GCC 12 manual, "Floating Types"). Build on a newer host with the option.

## Terms

The terms these documents use that ggml's and llama.cpp's documentation does not define:

| Term | Definition | Source | Nearest standard term |
|---|---|---|---|
| CBUF | Convolution buffer. The on-chip buffer of an NPU core that holds the feature data and weights the core computes on. Matmul tiles are sized to fit it. | NVDLA | |
| Claim | The backend's acceptance of a ggml op, made by `supports_op` returning true. ggml's scheduler can then run a claimed op on the NPU. Its opposite is decline. | Coined | `supports_op` returning true |
| Decline | The backend's refusal of a ggml op, made by `supports_op` returning false. The CPU backend then computes the op. Its opposite is claim. | Coined | `supports_op` returning false |
| Expert stack | One layer's `[K, N, n_expert]` weight tensor of routed experts: its `ffn_gate_exps`, `ffn_up_exps` or `ffn_down_exps`. The backend places a stack whole, on the NPU or on the CPU. | Coined | |
| Ingest | The one-time conversion of a quantized expert to int8 codes in NPU buffers, done inside the first prefill of each `llama_context`. | Coined | Weight preparation |
| IOVA window | The range of I/O virtual addresses (IOVA) through which the NPU reaches the buffers of one open device file. Each NPU worker of the backend opens its own device file. | Coined | IOMMU address space |
| K-accumulation | The summing of a matmul's partial results over successive tiles of its inner dimension, `K`. `ROCKET_KACC` selects the form that sums the partials in fp16 on the NPU. | Coined | Split-K reduction |
| Native-quant experts | The route for MoE experts that converts each quantized GGUF expert once into int8 codes, held as resident weights. Each call then quantizes only the activation (`ROCKET_MOE_NATIVE`). | Coined | |
| Pre-flight | The check in `supports_op` that reserves the RAM and IOVA window an expert stack needs, before the backend claims its `MUL_MAT_ID` op. | Coined | Admission control |
| Resident weight | A weight converted to the NPU's tile layout and written to NPU buffers once, then reused by every later call. Its opposite is a streamed weight. | Coined | Prepacked weight |
| Route | The way the backend computes one op, named by its precision and by whether its weights are resident: for example the fp16 route, the native-quant route or the RK3576 route. `ggml_backend_rocket_route_ops` counts ops per route. | Coined | Code path |
| Streamed weight | A weight converted and written to NPU buffers on every call that uses it. Its opposite is a resident weight. | Coined | |
| The RK3576 route | The int8 route for `MUL_MAT` on an RK3576, with int8 weights and activations (W8A8) and the result requantized per output column on the NPU. | Coined | W8A8 quantized matmul |
| Work per dispatch | The multiply-accumulates one expert GEMM does, `M_e · K · N`, where `M_e` is the rows routed to the expert. The backend offloads an expert stack only above a minimum. | Coined | |

## The rocket NPU stack

This backend is one frontend of an open source stack for Rockchip NPUs. The stack is a driver
library, three frontends, a submit provider for the vendor kernel, and a set of optional kernel
patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`): the
  userspace driver, matmul, and on-NPU op library. It is the dependency, and usable on its own.
- **`ggml-rocket`**: this backend, a ggml `.so` for `llama.cpp` and `whisper.cpp`.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)**: a TFLite external delegate
  for detection models, linking the same library.
- **[`ort-rocket`](https://github.com/gregordinary/ort-rocket)**: an ONNX Runtime execution
  provider for transformer encoders and per-layer convolutions, linking the same library.
- **[`rknpu-submit`](https://github.com/gregordinary/rknpu-submit)**: a submit provider that runs
  `librocketnpu` on the vendor `rknpu` BSP kernel driver instead of mainline `rocket`.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope): optional out-of-tree
  kernel-module patches for clock, voltage and IOMMU. They raise the NPU clock above its 200 MHz
  boot default.

## License and credits

`ggml-rocket` is MIT-licensed. See [LICENSE](LICENSE). It builds on
[ggml](https://github.com/ggml-org/ggml) (MIT) as an out-of-tree, runtime-loadable backend, modeled
on ggml's own BLAS backend, and on the `rocket-userspace` driver for NPU access.
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) carries ggml's license notice.
