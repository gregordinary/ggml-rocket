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

Decode stays on the CPU. It is small-`M` work, and below `ROCKET_MIN_M` (default 32 rows) the
per-call dispatch and weight packing outweigh the NPU's per-row advantage. The threshold keeps
llama.cpp's single-row GEMV decode on the CPU, which is bound by the DDR bandwidth the NPU shares
with the CPU. It also keeps batched decode there: whisper.cpp's default beam search presents 5
rows per step, far below the 32-row threshold.

The backend links `librocketnpu`, which the `rocket-userspace` project builds and which also runs
on its own. The build expects the checkouts side by side:

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  ggml/               # fresh upstream clone: git clone https://github.com/ggml-org/ggml
  ggml-rocket/        # this project
```

[API.md](API.md) is the tuning reference. It holds:

- The complete per-model benchmarks
- The [recommended configuration](API.md#recommended-configurations) for each workload (chat,
  agentic, RAG, MoE), with the RAM each needs
- Every `ROCKET_*` [environment variable](API.md#environment-variables)
- The diagnostic logging channel and the implementation notes

## What runs on the NPU

The backend offloads the ops that dominate prefill, and the CPU backend computes every other op:

- **`MUL_MAT`**: the prefill GEMMs. If `M >= ROCKET_MIN_M` and the shape aligns (`K%32`,
  `N%16`, `K>=64`, `N>=64`), the backend [claims](#terms) an F16 or F32 weight's matmul. Quantized
  weights (`Q8_0`, `Q4_K`, `Q6_K` and the rest) also offload, dequantized to fp16 on the fly.
  Decode (`M=1` GEMV) runs on the CPU.
- **`FLASH_ATTN_EXT`**: prefill attention, on the NPU by default at `n_kv >= 1024`. On
  Ministral-3-3B F16 its perplexity is within 0.03% of the CPU backend's attention. On
  Gemma-4-12B F16 it measured parity with the CPU backend's flash attention at 1K context. It
  measured 1.07x at 4K, 1.50x at 8K and 1.25x at 16K [HW sweep 2026-06].

  **Not characterized:** these ratios on the current build of the attention
  [route](#terms).
- **`MUL_MAT_ID`**: MoE routed experts, on the NPU by default as
  [native-quant experts](#terms), for a quantized expert stack that meets two conditions. The
  [pre-flight](#terms) must fit an estimate of the whole `[K, N, n_expert]` stack's footprint
  in the budget before any conversion. The per-expert GEMM must also do enough work to repay its
  own dispatch.

  On gpt-oss-20b the offload gives 1.9x the CPU backend's prefill at `pp2048`, against a CPU
  backend with its weight repack on [HW sweep 2026-09-29]. A stack that fails either condition
  runs whole on the CPU, so the default never falls below the experts-on-CPU baseline. The
  first prefill of each `llama_context` converts the experts once: ~21 s on gpt-oss-20b and
  ~18 s on DeepSeek-V2-Lite. See [MoE and MLA models](API.md#moe-and-mla-models).
- **Opt-in routes**: native int8 (`ROCKET_INT8=1`), int4 (`ROCKET_INT4=1`) and bf16
  (`ROCKET_BF16=1`). The int8 and int4 routes generate text character-identical to fp16 on the
  prompts measured, and the bf16 route computes an exact fp32 result.
  [Environment variables](API.md#environment-variables) has the detail.
- **Precision declarations**: the backend declines a matmul whose model declares a precision the
  backend cannot meet, and the CPU backend computes it. llama.cpp declares an F32 activation on
  Mistral-4's routed down-projection and an F32 result on some GLM-4 projections. A matmul
  tagged as a Hadamard rotation, which llama.cpp builds around a quantized KV cache, also stays
  on the CPU. [Precision declarations](API.md#precision-declarations) has the rules.
- **Everything else** stays on the CPU: norms, rope, the conv front-end and decode.
  `librocketnpu` can compose the offloaded matmuls into a full Whisper or transformer encoder
  block on the NPU, at cosine 1.000000 against an fp64 reference. The backend uses that block's
  attention sublayer.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), for every figure in this list.

Quantization reduces RAM and lets a larger model fit, and it does not speed prefill at this
operating point.
[Quantization and prefill throughput](API.md#quantization-and-prefill-throughput) explains why,
and API.md gives each route's contract and each datatype's detail.

## Terms

The terms this README and [API.md](API.md) use that ggml's and llama.cpp's documentation does
not define:

| Term | Definition | Source | Nearest standard term |
|---|---|---|---|
| CBUF | Convolution buffer. The on-chip buffer of an NPU core that holds the feature data and weights the core computes on. Matmul tiles are sized to fit it. | NVDLA | |
| Claim | The backend's acceptance of a ggml op, made by `supports_op` returning true. ggml's scheduler can then run a claimed op on the NPU. Its opposite is decline. | Coined | `supports_op` returning true |
| Decline | The backend's refusal of a ggml op, made by `supports_op` returning false. The CPU backend then computes the op. Its opposite is claim. | Coined | `supports_op` returning false |
| IOVA window | The range of I/O virtual addresses (IOVA) through which the NPU reaches the buffers of one open device file. Each NPU worker of the backend opens its own device file. | Coined | IOMMU address space |
| K-accumulation | The summing of a matmul's partial results over successive tiles of its inner dimension, `K`. `ROCKET_KACC` selects the form that sums the partials in fp16 on the NPU. | Coined | Split-K reduction |
| Native-quant experts | The route for MoE experts that converts each quantized GGUF expert once into int8 codes, held as resident weights. Each call then quantizes only the activation (`ROCKET_MOE_NATIVE`). | Coined | |
| Pre-flight | The check in `supports_op` that reserves the RAM and IOVA window an expert stack needs, before the backend claims its `MUL_MAT_ID` op. An expert stack is one `[K, N, n_expert]` weight tensor. | Coined | Admission control |
| Resident weight | A weight converted to the NPU's tile layout and written to NPU buffers once, then reused by every later call. Its opposite is a streamed weight. | Coined | Prepacked weight |
| Route | The way the backend computes one op, named by its precision and by whether its weights are resident: for example the fp16 route, the native-quant route or the RK3576 route. `ggml_backend_rocket_route_ops` counts ops per route. | Coined | Code path |
| Streamed weight | A weight converted and written to NPU buffers on every call that uses it. Its opposite is a resident weight. | Coined | |
| The RK3576 route | The int8 route for `MUL_MAT` on an RK3576, with int8 weights and activations (W8A8) and the result requantized per output column on the NPU. | Coined | W8A8 quantized matmul |

## Performance

The backend speeds up prefill, a batched GEMM, and its prefill speedup over the 8-thread CPU
backend grows with model size. Decode runs on the CPU, where it is bound by memory bandwidth and
its rate follows the quantization, not the backend.

| Model | Params | NPU prefill t/s, `pp2048` | Speedup over the CPU, `pp2048` | Q4 decode t/s | Q4 GGUF size |
|---|---:|---:|---:|---:|---:|
| SmolVLM2-2.2B (multimodal) | 1.8B | 52.6 F16 | 1.9x | 14.5 | 1.0 GB |
| Llama-3.2-3B | 3.2B | 56.0 F16 † | 2.6x † | 7.9 | 1.9 GB |
| Ministral-3-3B | 3.4B | 48.3 F16 † | 2.4x † | 7.7 | 2.0 GB |
| Phi-4-mini | 3.8B | 50.2 F16 † | 2.6x † | 7.0 | 2.3 GB |
| Ministral-3-8B | 8.5B | 21.4 F16 | 3.1x | 3.8 | 4.8 GB |
| Qwen3.5-9B | 9.0B | 24.9 F16 | 3.5x | 3.6 | 5.3 GB |
| Gemma-4-12B | 11.9B | 20.0 F16 † | 3.2x † | 2.5 | 6.9 GB |
| Phi-4 (14B) | 14.7B | 16.7 Q4 | 3.1x | 2.2 | 8.3 GB |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | 34.1 Q4 ‡ | 1.44x ‡ | 7.8 | 9.7 GB |
| gpt-oss-20b (MoE) | 20.9B | 30.2 MXFP4 ‡ | 1.90x ‡ | 7.2 | 11.3 GB |
| Qwen3.6-27B (hybrid) | 27.3B | 9.6 Q4 | 3.2x | 1.1 | 15.9 GB |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), warm runs, the same GGUF on the NPU
and on the 8-thread CPU backend [HW sweep]. The F16 rows run at llama.cpp's default `-b` and
`-ub`, and the unmarked F16 rows are 2026-07 sessions that measured both backends. The four
quantized rows run at `-b 2048 -ub 2048` on llama.cpp b11242, governor pinned, against a CPU
backend with llama.cpp's weight repack on (2026-09-29). The repack does not apply to F16
weights.

† The t/s is from a later build, llama.cpp b10558: 2026-08-29 for the 3B rows and 2026-09-02 for
Gemma-4-12B. The speedup is from the 2026-07 sessions, which measured both backends on an earlier
build.

‡ Measured on the 2026-09-29 build. Against the build before it, the current expert route
measures 1.09-1.10x the NPU prefill on gpt-oss-20b at `pp512`. On DeepSeek-V2-Lite at `pp2048`
it measures 1.09-1.14x [HW sweep 2026-10-09, NPU only, governor `ondemand`]. It also halves
DeepSeek-V2-Lite's one-time expert conversion, which `llama-bench`'s timed repetitions partly
include.

**Not characterized:** the speedup over the CPU on llama.cpp b10558 for the † rows, and on the
current expert route for the ‡ rows.

NPU prefill's perplexity agrees with the CPU backend's on every model in the table but
gpt-oss-20b. Wikitext perplexity is not a valid measure for that reasoning model. Its offloaded
expert GEMMs measure a mean cosine of 0.9998 against an fp64 CPU reference instead
[HW sweep 2026-08-27].
[Models and precisions](API.md#models-and-precisions) gives the measured difference for each
model where one was recorded, with its prefill, decode and interactive figures.

The MoE rows show smaller speedups because more of a MoE model's prefill stays on the CPU: the
router, attention and the expert gathers. The experts themselves run on the NPU by default.

### Speech models

For a speech model the backend offloads the encoder. The Whisper encoder runs 1.18x-2.14x
faster than on the CPU backend, and the speedup grows with model size. Through transcribe.cpp,
Granite-Speech, Voxtral, MOSS and other models gain 1.1x-1.7x on long audio. The gain is largest
where the encoder is large or the decoder cross-attends the audio.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), warm runs, against the CPU backend
on the same model.

### Ministral-3-8B-Instruct

Ministral-3-8B-Instruct shows the three figures a user sees: NPU prefill, CPU decode, and the
time an interactive turn takes.

#### Prefill

Prefill rate in t/s, CPU backend -> NPU, with the speedup:

| Precision | `pp512` | `pp1024` | `pp2048` |
|---|---|---|---|
| F16    | 7.2 -> 27.4 (3.8x) | 7.0 -> 24.4 (3.5x) | 6.8 -> 21.4 (3.1x) |
| Q8_0   | 6.9 -> 17.4 (2.5x) | 6.7 -> 19.3 (2.9x) | 6.4 -> 17.7 (2.8x) |
| Q4_K_M | 6.5 -> 17.0 (2.6x) | 6.4 -> 19.1 (3.0x) | 6.0 -> 17.4 (2.9x) |

**Conditions:** RK3588, 8-core CPU, NPU ~816 MHz (the 600 MHz request), warm runs, a 2026-07-01
session. F16 runs at the default `-b` and `-ub`, and the quantized rows at `-b 2048 -ub 2048`,
against a CPU backend without llama.cpp's weight repack.

Against a CPU backend with the repack on, Q4_K_M at `pp2048` measures 9.81 -> 19.33 t/s, 2.0x
[HW sweep 2026-07-17, 8 threads].

#### Decode

Decode runs on the CPU with either backend and is bound by memory bandwidth. Its rate therefore
follows the quantization and not the backend: F16 1.4 t/s, Q8_0 2.5 t/s and Q4_K_M 3.8 t/s.

#### Interactive turns

The NPU shortens the wait for the first token, and the quantization sets the decode rate:

| Turn | Configuration | Time to first token | Decode t/s | Total |
|---|---|---|---|---|
| 2048-tok prompt -> 200 out (RAG) | F16 CPU | 299 s | 1.4 t/s | 445 s |
| | F16 + NPU | 96 s | 1.4 t/s | 236 s |
| | Q4_K_M + NPU | 118 s | 3.8 t/s | 170 s |
| 128-tok prompt -> 400 out (chat) | F16 + NPU | 5 s | 1.4 t/s | 285 s |
| | Q4_K_M + NPU | 8 s | 3.8 t/s | 112 s |

**Conditions:** the board, clock and GGUFs of the prefill table above.

On a long prompt the NPU cuts the time to first token ~3x and nearly halves the turn. Decode sets
the length of a short chat turn, so the NPU barely shortens it, and quantization does.

On most boards, run Ministral-3-8B as Q4_K_M with the NPU. Its 4.83 GB GGUF fits an 8 GB board,
prefills ~2.0x faster than a repacked CPU backend at `pp2048`, and decodes 2.8x faster than F16.
If RAM is ample, F16 gives the fastest prefill.

## Hardware support

| Part | Status | Scope |
|---|---|---|
| RK3588 | Supported | Every route. The fully validated target. |
| RK3576 | Supported | The RK3576 route only, with `ROCKET_INT8=1`. Attention runs on the CPU. |
| Other Rockchip NPUs | Not characterized | |

### The RK3576

On the RK3576 the backend uses the RK3576 route, an int8 W8A8 matmul with a per-output-column
requant. The backend selects it from the detected part, and no environment variable selects it.
It is the only matmul route on that part, because the RK3588 routes do not run there. The
backend claims a `MUL_MAT` on the RK3576 route or declines it.

> [!CAUTION]
> On the RK3576, set `ROCKET_INT8=1`. Without it the backend declines every `MUL_MAT`, and
> nothing offloads.

Attention runs on the CPU there, with nothing to set. The backend has no RK3576 attention
route, so it declines `FLASH_ATTN_EXT`, and the CPU backend computes it with its own kernel.

Build the backend for the RK3576 with `-DGGML_ROCKET_NATIVE_FP16=OFF`, because the default
convert kernels target the RK3588's `armv8.2-a+fp16` baseline.

The backend prefills Qwen2.5-1.5B F16 at 13.36 t/s on the RK3576 [HW sweep]. That rate is 1.31x
ggml's Q8_0 kernel on the CPU, the comparison for a user who already quantizes. It is 2.37x the
same F16 GGUF on the CPU, the comparison for a drop-in user, because the backend reads an F16
GGUF. Its wikitext-2 perplexity is 1.011x the F16 CPU backend's, over eight chunks whose own
error bar is +-0.59 [HW sweep].

**Conditions:** Qwen2.5-1.5B, `pp2048`, `-ub` 512, four CPU threads on the A72 cores, governor
`performance`.

> [!NOTE]
> A `pp512` figure on the RK3576 measures calibration, not the route's speed. At that length
> every call is still a calibration pass, which sets the route's per-column output scale.

[The RK3576](API.md#the-rk3576) in API.md documents the route, its thirteen `ROCKET_RK3576_*`
environment variables, and what each is measured against.

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

The RK3588 figures in this README were measured with the 600 MHz request, which runs at ~816 MHz
on the same rail. The 900 MHz request is 1.044x faster on matmul device time.

> [!CAUTION]
> With the NPU cores pinned on (`power/control=on`), an `rmmod` at the 900 MHz request
> hard-locks the board, and only a power cycle recovers it. Unload the module only while all
> three cores are runtime-suspended.

## Build and run

### Build and test the backend

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
the backend's route counters, so it asserts which route computed its ops instead of inferring it
from the output:

| Test | What it covers |
|---|---|
| `test-rocket-matmul` | The backend against the CPU backend on real ggml graphs |
| `test-rocket-moe` | The MoE handler, both the fp16 and the native-quant expert route, against the CPU backend under outlier-channel activations |
| `test-rocket-int4` | The int4 route |
| `test-rocket-bf16` | The bf16 route |
| `test-rk3576-w8a8` | The RK3576 route against the CPU backend. `Skipped` on any other part |
| `test-rocket-placement` | The `supports_op` and `offload_op` contract |
| `test-rocket-fa` | `FLASH_ATTN_EXT` against a double-precision reference, including the logits fp16 cannot hold |
| `test-rocket-internals` | Host-only: the weight cache's identity, the MoE residency ledger across two live backends, the budget parser, the FA dump's file handling, the device-route circuit breaker |
| `test-hadamard-neon`, `test-hadamard-quant` | The Hadamard rotation construction |

`test-common.h` holds the graph and comparison code the tests share.

`test-rk3576-sgemm-ab` is a benchmark rather than a test. It times the RK3576's NPU GEMM against
ggml F32, ggml Q8_0 and OpenBLAS on the CPU. It builds only with
`-DGGML_ROCKET_RK3576_BENCH=ON`, which needs OpenBLAS.

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

The build options are:

- `-DGGML_ROCKET_NATIVE_FP16=OFF`. By default the convert kernels are compiled for the RK3588
  `armv8.2-a+fp16` baseline. The resulting `.so` stops with `SIGILL` on an older aarch64 core
  that lacks FP16. Turn this OFF for a portable `.so` with scalar convert kernels.
- `-DGGML_ROCKET_DIAGNOSTICS=ON`. Compiles in the heavy per-op recompute and trace diagnostics
  (`ROCKET_VERIFY`, `ROCKET_CPU_FORWARD`, `ROCKET_TRACE`, `ROCKET_AB`, `ROCKET_DUMP`), which the
  production `.so` leaves out by default. The lightweight `ROCKET_MM_PROFILE` and `ROCKET_DEBUG`
  lines are available in either build.
- `-DGGML_ROCKET_PORTABLE_GLIBC=ON`. Lets the `.so` load on an older glibc, down to 2.34. It
  links in forwarders for the symbols a newer build host versions above 2.34: glibc 2.38's
  `__isoc23_*` and glibc 2.43's `sqrtf`. It is needed only for a build host whose glibc is newer
  than the deployment target's, and a same-host build is byte-unchanged. It takes effect in DL
  mode only, since a static standalone build has no loader to satisfy. The `.so` also stays
  within the libstdc++ of GCC 12, which Debian 12 provides. If it still needs a newer glibc or
  libstdc++, the build fails and names the symbol (see
  [Troubleshooting](#version-glibc_238-not-found-at-load)).
- `-DGGML_ROCKET_RK3576_BENCH=ON`. Builds the RK3576 NPU-vs-CPU GEMM benchmark, which needs
  OpenBLAS. It is off by default, and the Caution above applies.
- `-DGGML_ROCKET_STRICT_ABI=ON`. Fails the build on a host ggml whose
  `GGML_BACKEND_API_VERSION` this backend has not been checked against. By default that build
  continues with a warning.

To run a model, rebuild the `.so` against the host's ggml, so that it links the same
`libggml-base.so`, and point `GGML_BACKEND_PATH` at it. Each recipe below does both.

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

Run it under `sudo -E`: `/dev/accel/accel0` needs privilege, and `-E` keeps `GGML_BACKEND_PATH`
in the environment. The startup log lists a `ROCKET` device whose description names the detected
part and driver, such as `RK3588 NPU (mainline rocket driver)` or
`RK3576 NPU (mainline rocket driver)`. A `drm_mm "Memory manager not clean"` warning at exit
comes from a known teardown race and is harmless.

`whisper-server` loads the `.so` the same way. When a client posts fixed-length chunks with no
request fields, as OpenWebRX+ does, the server's own settings decide the CPU cost. Three
settings matter:

- `-nt -sns`
- An `-ac` of 50 times the chunk length plus 4 s, capped at 1500
- The temperature fallback off, through a `temperature_inc=0` request field, because the
  server's `-nf` flag is parsed and never applied

Together they cut the CPU cost to 0.70x the defaults' on 20 s chunks, at better WER.
[Recommended configurations](API.md#recommended-configurations) gives the settings.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), F16 `ggml-small` [HW sweep].

> [!CAUTION]
> An `-ac` equal to the chunk length makes the decoder loop, and a 30 s chunk without `-nt` is
> encoded twice.

### Drop into transcribe.cpp

transcribe.cpp, a ggml-based host for several speech-to-text model families, loads
`libggml-rocket.so` the same way. Through it the backend reaches Granite-Speech, Voxtral, MOSS
diarize, SenseVoice, FunASR and Parakeet. Its runner registers the NPU as an ACCEL device and
logs `using accel backend: ROCKET`.

The encoder runs on the NPU, and on long audio so does the decoder's batched prefill.
Autoregressive decode stays on the CPU. The speedup therefore grows with encoder size and audio
length. Encoder-heavy models gain most, such as Voxtral with its Whisper-large-v3 encoder, or
Granite with its cross-attention decoder. Small decoder-only models are bound by decode and
barely change.

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

### Drop into llama.cpp

llama.cpp loads the `.so` the same way. An F16 GGUF takes the fp16 route with no conversion.

A quantized GGUF (`Q8_0`, `Q4_K`, `Q6_K` and the rest) also prefills on the NPU. Its weights are
dequantized to fp16 on the fly, with no whole-model F16 copy. The model therefore keeps its
compact footprint.

A BF16 GGUF also prefills on the NPU. Its weights are converted to fp16 on every micro-batch, at
~0.55-0.6x the prefill rate of fp16. Convert it with `llama-quantize in.gguf out.gguf f16` for
full speed, or set `ROCKET_BF16=1` for the exact fp32-output bf16 route.

> [!CAUTION]
> For a quantized GGUF, turn the host's weight repack off: pass `-nr` (`--no-repack`), or
> `--repack 0` to `llama-bench`. With the repack on, the quantized weights stay on the CPU, and
> nothing reports it.
>
> The CPU backend repacks quantized weights into a non-host `CPU_REPACK` buffer for its own
> SIMD kernels. ggml's scheduler offloads only an op whose weight is in a host buffer. A
> host built with `-DGGML_CPU_REPACK=OFF` also works, but it removes the repack from CPU-only
> runs too, where it is worth 1.3-1.7x on quantized prefill. F16 weights are never repacked.
> Confirm the offload with `ROCKET_MM_PROFILE=1` or `ROCKET_DEBUG_GRAPH=1`.

```sh
# 1. latest llama.cpp, shared + DL-capable ggml. The weight repack stays on; pass -nr to a
#    run of a QUANTIZED GGUF on the NPU (see the note above).
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

K-accumulation (`ROCKET_KACC`) gives the fastest prefill and is on by default, and with it CBUF
operand reuse (`ROCKET_REUSE`) chooses data or weight reuse per shape. Cap `-c` so that the weights and the KV cache fit in RAM.

A quantized GGUF is dequantized to fp16 on every micro-batch, and two settings reduce that cost.
If its fp16 image fits in RAM, set `ROCKET_QUANT_RESIDENT=auto`, which makes each weight a
resident weight, and keep the default `-ub`. On seven models of 1.8-11.9 B that measures
1.35-1.75x the default configuration's prefill. Combined with `-b 2048 -ub 2048` it measures
1.00-1.66x [HW sweep 2026-08].

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request).

If the fp16 image does not fit, run `-b 2048 -ub 2048`. The setting gives 0.91-1.53x the
`-ub 512` default's prefill, depending on the model, and is slower on two models of eleven, so
measure it.

Gemma-4's chat template needs `--jinja`, or use `llama-completion -no-cnv` for raw prompt
completion.

#### CPU thread placement

Pin a prefill-heavy run to the A76 big cores, and do not pin a decode-heavy one. `taskset 0xf0`
gives 1.05-1.13x on NPU prefill, largest on the smallest model [HW sweep, three models]. It
applies with streamed weights too: the 12B model streams all of its weights and still gains
1.05x. Pinning slows decode by 34% on Gemma-4-12B F16, whose decode is bound by LPDDR bandwidth
and uses all 8 cores. Measure a mixed run.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), governor pinned.

#### Warm-up runs

The NPU clock drops at idle, so a cold run measures ~15% low. Discard the first `-r 1` run, run
at least three times, and compare warm runs.

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

The option links forwarders for those glibc symbols into the `.so`, so it needs only glibc 2.34.
It also holds the `.so` to the libstdc++ of GCC 12 (`GLIBCXX_3.4.30`), which Debian 12 provides.
If the linked `.so` still needs a newer glibc or libstdc++, the build fails and names the symbol.

A Debian 12 host cannot build the backend for the board itself, because GCC 12 accepts `_Float16`
in C++ only on x86 (the GCC 12 manual, "Floating Types"). Build on a newer host with the option.

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
  boot default. The prefill figures here were measured with them, at ~816 MHz (the 600 MHz
  request).

## License and credits

`ggml-rocket` is GPL-3.0-or-later (it links the GPL-3 `rocket-userspace` driver library, whose NPU
register headers are the GPL-3 reverse-engineering by Jasbir Matharu). It builds on
[ggml](https://github.com/ggml-org/ggml) (MIT) as an out-of-tree, runtime-loadable backend, modeled
on ggml's own BLAS backend, and on the `rocket-userspace` driver for NPU access.
