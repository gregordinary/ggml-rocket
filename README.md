# ggml-rocket

## AI Disclosure

With the exception of prior work this may build on, ggml-rocket was developed by AI, primarily Claude Code (Opus 4.8). Human involvement was mostly limited to setting project goals and providing hardware access. This is a side project for curiosity's sake and comes with no guarantee of quality, accuracy, or update frequency.

## About ggml-rocket

A drop-in [ggml](https://github.com/ggml-org/ggml) backend for Rockchip NPUs (validated on
the RK3588) that offloads LLM and Whisper prefill to the NPU through the mainline `rocket`
DRM-accel driver.

It builds as a runtime-loadable `libggml-rocket.so` that drops into stock llama.cpp and
whisper.cpp: point `GGML_BACKEND_PATH` at it and the NPU appears as a ggml device, exactly like
ggml's own BLAS backend. It runs the Whisper encoder end to end and the prefill of Gemma-4,
Qwen3.5 / 3.6, Llama-3.2, Phi-4, Ministral and more on the NPU. Decode stays on the CPU — it is a
single-row GEMV, ~82× slower on the NPU.

This is a separate project from the driver: `rocket-userspace` (the `librocketnpu` dependency)
builds and runs on its own; this backend links it.

```
.
  rocket-userspace/   # the driver library (dependency): git clone https://github.com/gregordinary/rocket-userspace
  ggml/               # fresh upstream clone: git clone https://github.com/ggml-org/ggml
  ggml-rocket/        # this project
```

The complete per-model benchmarks, the full `ROCKET_*` knob table, the diagnostic-logging channel,
and the implementation notes are in [API.md](API.md).

## Performance

The NPU is a prefill / batched-GEMM engine. The prefill speedup over the 8-thread CPU grows with
model size; decode is forced to the CPU, is bandwidth-bound, and scales with the quant, not the
backend. All figures warm, RK3588 @ 600 MHz, same GGUF on NPU and CPU; NPU prefill is
perplexity-faithful to the CPU on every model. [HW sweep]

| Model | Params | Prefill NPU @pp2048 (×CPU) | Decode Q4 t/s | Fits Q4 |
|---|---:|---:|---:|---:|
| SmolVLM2-2.2B (multimodal) | 1.8B | 52.6 F16 (1.9×) | 14.5 | 1.0 GB |
| Llama-3.2-3B | 3.2B | 45.5 F16 (2.6×) | 7.9 | 1.9 GB |
| Ministral-3-3B | 3.4B | 39.8 F16 (2.4×) | 7.7 | 2.0 GB |
| Phi-4-mini | 3.8B | 39.8 F16 (2.6×) | 7.0 | 2.3 GB |
| Ministral-3-8B | 8.5B | 21.4 F16 (3.1×) | 3.8 | 4.8 GB |
| Qwen3.5-9B | 9.0B | 24.9 F16 (3.5×) | 3.6 | 5.3 GB |
| Gemma-4-12B | 11.9B | 15.0 F16 (3.2×) | 2.5 | 6.9 GB |
| Phi-4 (14B) | 14.7B | 11.8 Q4 (3.5×) | 2.2 | 8.3 GB |
| DeepSeek-V2-Lite (MoE+MLA) | 15.7B | 23.9 Q4 (1.26×) | 7.8 | 9.7 GB |
| gpt-oss-20b (MoE) | 20.9B | 13.1 MXFP4 (1.04×) | 7.2 | 11.3 GB |
| Qwen3.6-27B (hybrid) | 27.3B | 7.8 Q4 (4.4×) | 1.1 | 15.9 GB |

Where the ratio is small the architecture keeps most prefill FLOPs off the NPU: the MoE expert FFNs
of gpt-oss (1.04×) and DeepSeek (1.26×) stay on the CPU by default. The Whisper encoder is a
separate case (1.18× → 2.14× by model size). Per-model prefill / decode / interactive detail, the
Gemma-4-12B walkthrough, and why quantization does not speed prefill at this operating point are in
[API.md](API.md#models-and-precisions).

### A representative model — Ministral-3-8B-Instruct

The numbers split three ways: prefill (the NPU's job), decode (CPU-bound on both backends), and the
interactive turn you feel. RK3588, 8-core, NPU at 600 MHz, warm.

**Prefill (prompt processing), t/s — CPU → NPU:**

| | pp512 | pp1024 | pp2048 |
|---|---|---|---|
| F16    | 7.2 → 27.4 (3.8×) | 7.0 → 24.4 (3.5×) | 6.8 → 21.4 (3.1×) |
| Q8_0   | 6.9 → 17.4 (2.5×) | 6.7 → 19.3 (2.9×) | 6.4 → 17.7 (2.8×) |
| Q4_K_M | 6.5 → 17.0 (2.6×) | 6.4 → 19.1 (3.0×) | 6.0 → 17.4 (2.9×) |

**Decode (generation), t/s — CPU ≈ NPU** (off the NPU; bandwidth-bound, so it scales with the quant,
not the backend): F16 1.4 → Q8_0 2.5 → Q4_K_M 3.8 t/s.

**Interactive — the NPU shortens the first-token wait; the quant speeds the stream:**

| turn | config | time-to-first-token | stream | total |
|---|---|---|---|---|
| 2048-tok prompt → 200 out (RAG) | F16 CPU | 299 s | 1.4 t/s | 445 s |
| | F16 + NPU | 96 s | 1.4 t/s | 236 s |
| | Q4_K_M + NPU | 118 s | 3.8 t/s | 170 s |
| 128-tok prompt → 400 out (chat) | F16 + NPU | 5 s | 1.4 t/s | 285 s |
| | Q4_K_M + NPU | 8 s | 3.8 t/s | 112 s |

On a long prompt the NPU cuts time-to-first-token ~3× and nearly halves the turn; on a short chatty
turn the wall is decode-bound, so the NPU barely moves it and quantization is the lever. For most
boards Q4_K_M + NPU is the sweet spot — ~2.9× prefill and a 2.8×-faster stream than F16, in 4.83 GB
(fits an 8 GB board); F16 is the max-prefill option when RAM is ample.

## What runs on the NPU

The backend offloads the ops that dominate prefill and leaves the rest on the CPU:

- **`MUL_MAT`** — the prefill GEMMs. F16/F32 weights offload when `M ≥ ROCKET_MIN_M` and the shape
  aligns (`K%32`, `N%16`, `K≥64`, `N≥64`). Quantized weights (`Q8_0` / `Q4_K` / `Q6_K` / …) also
  offload, dequantized to fp16 on the fly. Decode (`M=1` GEMV) is forced to the CPU.
- **`FLASH_ATTN_EXT`** — prefill attention, on the NPU by default when `n_kv ≥ 1024`. Bit-faithful
  and submit-chained: 1.07× at 4K, 1.50× at 8K, 1.25× at 16K, parity below. [HW sweep, F16, 600 MHz]
- **Opt-in datapaths** — native int8 (`ROCKET_INT8=1`), int4 (`ROCKET_INT4=1`), bf16
  (`ROCKET_BF16=1`), and MoE routed experts (`ROCKET_MOE=1`). Each is numerically faithful; see the
  [knob table](API.md#runtime-knobs).
- **Everything else** — norms, rope, the conv front-end, decode — stays on the CPU. The
  `rocket-userspace` driver composes the offloaded matmuls into a full Whisper/transformer encoder
  block on the NPU (cos = 1.000000 vs an fp64 oracle); this backend wires the attention sublayer of
  that for LLM prefill.

The envelope, all HW-validated on the RK3588 and PPL-faithful to the CPU backend:

- **Prefill engine.** The NPU accelerates prefill (batched GEMM) and the Whisper encoder; decode
  (M=1 GEMV) stays on the CPU (~82× slower on the NPU).
- **Quantization buys RAM / model-fit, not prefill speed** at this operating point — resident matmul
  is ~460 GOP/s across precisions (DMA/dispatch-bound, not MAC-bound). A quantized GGUF wants
  `-b 2048 -ub 2048`. Bottleneck-conditional, not a permanent property — see
  [API.md](API.md#why-quantization-does-not-speed-prefill).
- **MoE routed experts run on the CPU by default** — offloading the quantized experts is a net loss
  (`ROCKET_MOE=1` opts in, bit-faithful but slower). The dense projections and `lm_head` still reach
  the NPU.
- **bf16 weights prefill ~0.55–0.6× native fp16** (re-decoded per micro-batch); convert to fp16 for
  full speed, or set `ROCKET_BF16=1` for the exact fp32-output bf16 datapath.

The full per-op routing contract and per-dtype detail are in [API.md](API.md).

## Requirements

- An RK3588 board on a mainline kernel carrying the `rocket` DRM-accel driver, with
  `/dev/accel/accel0` present (`lsmod | grep rocket`).
- The sibling `rocket-userspace` driver library, cloned next to this repo or installed as a
  `rocketnpu` package.
- Privilege to open the accel node — run with `sudo -E` (the `-E` preserves the `ROCKET_*` env knobs
  plain `sudo` strips).

**Clock.** The NPU boots at 200 MHz and the backend is correct there, but every prefill figure above
is at 600 MHz: apply the `patches/rocket` clock patch and load the module with
`rocket_npu_clk_hz=600000000`.

## Build and run

### Build and test the backend

```sh
# from the workspace root (the dir holding rocket-userspace/, ggml/, ggml-rocket/)
git clone https://github.com/ggml-org/ggml          # if not already present
cd ggml-rocket
cmake -S . -B build              # or: -DGGML_DIR=/path/to/ggml
cmake --build build -j
sudo ./build/test-rocket-matmul  # needs /dev/accel/accel0 — each shape prints PASS
```

The gates are all CTest-registered — run `ctest` from the build dir; the NPU gates report `Skipped`
off-device, the pure-CPU ones run anywhere. They include `test-rocket-matmul` (rocket vs CPU backend
on real ggml graphs), `test-rocket-moe` (the MoE handler, cos = 1.000000), `test-rocket-int4`,
`test-rocket-bf16`, `test-rocket-placement` (the `supports_op` / `offload_op` contract), and the
Hadamard construction tests.

> **If the driver is installed** (`find_package(rocketnpu)` resolves to a `cmake --install`d
> package, e.g. `/usr/local`), ggml-rocket links that installed lib, not a sibling
> `rocket-userspace/` source tree. So after a change in `rocket-userspace/`, reinstall it
> (`cd ../rocket-userspace && cmake --build build && sudo cmake --install build`) before rebuilding
> here, or force the source tree with
> `-DCMAKE_DISABLE_FIND_PACKAGE_rocketnpu=ON -DROCKETNPU_DIR=../rocket-userspace`.

**Build options.**

- `-DGGML_ROCKET_NATIVE_FP16=OFF` — by default the convert kernels are compiled for the RK3588
  `armv8.2-a+fp16` baseline; the resulting `.so` `SIGILL`s on an older aarch64 core lacking FP16.
  Turn this OFF for a portable (scalar-convert) `.so`.
- `-DGGML_ROCKET_DIAGNOSTICS=ON` — compile in the heavy per-op recompute/trace diagnostics
  (`ROCKET_VERIFY` / `ROCKET_CPU_FORWARD` / `ROCKET_TRACE` / `ROCKET_AB` / `ROCKET_DUMP`), out of the
  production `.so` by default. The lightweight `ROCKET_MM_PROFILE` / `ROCKET_DEBUG` lines stay
  available regardless.
- `-DGGML_ROCKET_PORTABLE_GLIBC=ON` — internalize the five glibc 2.38 `__isoc23_*` symbols so the
  `.so` loads on an older-glibc runtime (floors it at glibc 2.34). Needed only when the build host's
  glibc is newer than the deployment target's; a same-host build is byte-unchanged.

To run a real model you rebuild the `.so` against the host app's bundled ggml (so it links the same
`libggml-base.so`) and point `GGML_BACKEND_PATH` at it — the two drop-in recipes below.

### Drop into whisper.cpp

Mainline whisper.cpp loads `libggml-rocket.so` with no source changes: `whisper-cli`'s `main()`
calls `ggml_backend_load_all()` (which honors `GGML_BACKEND_PATH`), the `.so` exports
`ggml_backend_init` via `GGML_BACKEND_DL_IMPL`, and its device is type ACCEL (whisper adds every
ACCEL device, BLAS-style; the scheduler then offloads the `MUL_MAT`s).

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

`sudo -E` because `/dev/accel/accel0` needs privilege and `-E` preserves the env var. The startup log
lists a `ROCKET` / "RK3588 NPU" device; the `drm_mm "Memory manager not clean"` WARN at exit is a
known-benign teardown race.

### Drop into llama.cpp

Same mechanism. An F16 GGUF takes the fp16 NPU path with zero conversion; a quantized
(`Q8_0` / `Q4_K` / `Q6_K` / …) GGUF also prefills on the NPU (weights dequantized to fp16 on the fly,
no whole-model F16 copy — NPU prefill and the compact footprint); a BF16 GGUF also prefills on the
NPU (weights decoded to fp16 on the fly, ~0.55–0.6× fp16 — convert with
`llama-quantize in.gguf out.gguf f16` for full speed, or set `ROCKET_BF16=1` for the fp32-output
datapath).

> **Quantized GGUFs need `-DGGML_CPU_REPACK=OFF` on the host build.** ggml-cpu repacks quantized
> weights into a non-host `CPU_REPACK` buffer for its own SIMD kernels (on by default), and the
> scheduler only hands an op to this backend when the weight sits in a host buffer. Building with
> `-DGGML_CPU_REPACK=OFF` keeps quantized weights in a host (`CPU_Mapped`) buffer so they reach the
> NPU. F16 weights are never repacked. Confirm offload with `ROCKET_MM_PROFILE=1` or
> `ROCKET_DEBUG_GRAPH=1`.

```sh
# 1. latest llama.cpp, shared + DL-capable ggml. Add -DGGML_CPU_REPACK=OFF if you will
#    run a QUANTIZED GGUF on the NPU (keeps quant weights in a host buffer; see note above).
cd <workspace>     # the dir holding ggml-rocket/
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_CPU_REPACK=OFF
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

K-accum (the best-prefill operating mode; DATA_REUSE follows) is on by default. Cap `-c` so the
weights + KV fit RAM. Bigger `-b` / `-ub` (= M) means better NPU utilization; for a quantized GGUF
run `-b 2048 -ub 2048` (its per-micro-batch dequant ~halves prefill at the llama.cpp `-ub 512`
default). Gemma-4's chat template needs `--jinja` (or use `llama-completion -no-cnv` for raw prompt
completion).

- **CPU thread placement — leave it unpinned for F16.** On Gemma-4-12B F16, pinning the process to
  the A76 big cores is not a win: prefill is flat (the matmul workers are already A76-pinned) and
  decode is −34% (F16 decode is LPDDR-bandwidth-bound, so confining it to 4 A76 cores starves it).
  Let decode use all 8 cores.
- **Measurement discipline.** The clock parks at idle, so discard the first `-r 1` run (cold, ~15%
  low); run ≥3× and compare warm runs.

## Troubleshooting

**"ROCKET device not listed at startup"** (the backend loads but no NPU device appears). Almost
always an ABI mismatch. The ggml backend vtables are initialized positionally, so the backend must be
built against the same ggml the host app uses — `llama.cpp` and `whisper.cpp` each clone their own
ggml. If the host's `ggml-backend-impl.h` has a different `GGML_BACKEND_API_VERSION` or 2d-tensor
field layout than the ggml this backend was compiled against, the positional vtable drifts and the
device silently fails to register. Fix: rebuild against that host checkout's ggml headers, and
re-check on every ggml bump. This backend targets `GGML_BACKEND_API_VERSION 2` with the device vtable
that includes the `set_tensor_2d_async` / `get_tensor_2d_async` slots (verified against the in-repo
`ggml/`, tagged v0.14.0). If the device appears but runs zero matmuls, the backend was built without
a working `offload_op` — see [implementation notes](API.md#implementation-notes).

**`version 'GLIBC_2.38' not found` when the host app loads the `.so`.** `libggml-rocket.so` was built
on a host with a newer glibc (≥ 2.38, e.g. Debian trixie/forky) than the runtime it is deployed to
(e.g. a Debian 12 bookworm container, glibc 2.36). Rebuild with `-DGGML_ROCKET_PORTABLE_GLIBC=ON`
(internalizes the five `__isoc23_*` symbols so the `.so` floors at glibc 2.34), or build on a host
matching the target's glibc.

## The rocket NPU stack

This backend is one frontend of an open source stack for Rockchip NPUs — three userspace projects
plus a set of optional kernel patches:

- **[`rocket-userspace`](https://github.com/gregordinary/rocket-userspace)** (`librocketnpu`) — the userspace driver, matmul, and on-NPU op library.
  The dependency; usable on its own.
- **`ggml-rocket`** (this project) — a ggml backend `.so` for `llama.cpp` / `whisper.cpp`.
- **[`tflite-rocket`](https://github.com/gregordinary/tflite-rocket)** — a TFLite external delegate for detection models, linking the same driver.
- **[`patches`](https://github.com/gregordinary/patches)** (`rocket/` scope) — optional out-of-tree kernel-module patches
  (clock / voltage / IOMMU). They raise the NPU clock from its 200 MHz boot default to 600 MHz; the
  prefill numbers here assume them.

## License & credits

`ggml-rocket` is GPL-3.0-or-later (it links the GPL-3 `rocket-userspace` driver library, whose NPU
register headers are the GPL-3 reverse-engineering by Jasbir Matharu). It builds on
[ggml](https://github.com/ggml-org/ggml) (MIT) as an out-of-tree, runtime-loadable backend, modeled
on ggml's own BLAS backend, and on the `rocket-userspace` driver for the NPU path.
