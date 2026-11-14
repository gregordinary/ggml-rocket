# ggml-rocket: models and precisions

This document gives the measured prefill, decode and accuracy of each model the backend has run,
and of each precision route. Each entry is validated on the RK3588.

A figure dated 2026-11-01 was measured with the NPU at 928 MHz (the 900 MHz request), CPU governor
`performance` and llama.cpp b11539. Its decode rates run at `-t 4` on the four A76 cores. Unless a
Conditions line says otherwise, any other figure was measured with the NPU at ~816 MHz (the
600 MHz request), warm. A speedup's baseline is the 8-thread CPU backend on the same GGUF.

## Speech models

### Whisper

Stock `whisper-cli` and `whisper-bench` load the backend's `.so` through `GGML_BACKEND_PATH` and
run the encoder's `MUL_MAT` ops on the NPU. Transcripts are byte-identical to the CPU backend's,
at WER 0 and encoder-output cosine 0.9998. The encoder is faster than on the CPU backend at every
model size, and the speedup grows with the model:

| Model | Encoder speedup over the CPU backend |
|---|---|
| `tiny.en` | 1.18x |
| `base.en` | 1.29x |
| `small.en` | 1.54x |
| `medium.en` | 1.84x |
| `large-v3` | 2.14x |
| `large-v3-turbo` | 2.12x |

**Conditions:** `whisper-bench` encode.

The decoder runs at `M`=1 and stays on the CPU with either backend. `large-v3-turbo` keeps the
full 32-layer encoder with a 4-layer decoder, ~6x cheaper per step, so the offloaded encoder is
the larger share of its pipeline.

### transcribe.cpp models

transcribe.cpp hosts Granite-Speech, Voxtral, MOSS diarize, SenseVoice, FunASR and Parakeet, and
the backend offloads each one's encoder, at 1.0x-1.8x the CPU backend's encode. The speedup scales
with encoder size times audio length. Autoregressive decode stays on the CPU, except the batched
decode prefill over the injected audio context, which offloads on long audio.

The model's architecture sets the whole pipeline's speedup:

| Architecture | Model | Result |
|---|---|---|
| Cross-attention decoder | Granite-Speech | ~1.9x on 120 s, its decode offloading best |
| Large decoder-only over long audio | Voxtral-mini 3B | Its ~1500-token audio prefill offloads: decode 1.46x, whole pipeline 1.57x |
| Small decoder-only | MOSS, FunASR 0.6B | ~1.0-1.15x, bound by decode despite a ~1.6x encode |
| Single-pass CTC | SenseVoice | No decode: 6.2x realtime, 1.26x |

**Conditions:** `Q8_0`, the process on the A76 cores, one fresh process per run, against
transcribe.cpp's CPU backend [HW sweep 2026-07-21].

**Not characterized:** the NPU clock of these runs.

With only the encoder offloaded, the transcript is identical to the CPU backend's. A model whose
decoder cross-attention offloads, such as Granite, can diverge late, as fp16 accumulation perturbs
greedy decoding. `Q8_0` matches or beats F16 on both backends at half the RAM [HW sweep
2026-07-21], so it is the quantization to use for these models.

## Language models

### Ministral-3-8B

Prefill rate in t/s, CPU backend -> NPU, with the speedup:

| Precision | `pp512` | `pp1024` | `pp2048` |
|---|---|---|---|
| F16 | 7.2 -> 44.0 (6.1x) | 7.1 -> 41.5 (5.8x) | 6.8 -> 39.6 (5.8x) |
| `Q8_0` | 12.8 -> 50.2 (3.9x) | 12.4 -> 48.4 (3.9x) | 11.7 -> 45.0 (3.8x) |
| `Q4_K_M` | 10.3 -> 50.5 (4.9x) | 10.1 -> 47.9 (4.7x) | 9.6 -> 45.1 (4.7x) |

**Conditions:** the default `-b` and `-ub` [HW sweep 2026-11-01]. The quantized GGUFs run under
`ROCKET_QUANT_RESIDENT=auto`, against a CPU backend with llama.cpp's weight repack on.

The quantized GGUFs prefill faster on the NPU than F16 does. Quantized residency holds all 235 of
their weights resident as fp16, while the F16 GGUF streams its weights at the default. Their
speedups are lower than F16's because the CPU backend runs them faster.

Decode rates follow the quantization: F16 1.6 t/s, `Q8_0` 3.0 t/s and `Q4_K_M` 4.7 t/s. The
README's [interactive turn](README.md#an-interactive-turn) combines the two.

### Gemma-4-12B

Gemma-4-12B generates coherently through the same drop-in on llama.cpp. NPU prefill keeps
wikitext perplexity within ±2% of the CPU backend's across F16, `Q8_0` and `Q4_K_M`, inside the
±10% per-run noise. The absolute perplexity of this instruct model on raw text is not meaningful.

Prefill rate in t/s, CPU backend -> NPU, with the speedup:

| Precision | `pp512` | `pp1024` | `pp2048` |
|---|---|---|---|
| F16 | 4.98 -> 25.62 (5.1x) | 4.87 -> 25.31 (5.2x) | 4.72 -> 24.64 (5.2x) |
| `Q8_0` | 8.44 -> 26.45 (3.1x) | 8.14 -> 26.58 (3.3x) | 7.78 -> 24.75 (3.2x) |
| `Q4_K_M` | 6.84 -> 26.25 (3.8x) | 6.67 -> 25.38 (3.8x) | 6.42 -> 24.51 (3.8x) |

**Conditions:** the default `-b` and `-ub` [HW sweep 2026-11-01]. The quantized GGUFs run under
`ROCKET_QUANT_RESIDENT=auto`, against a CPU backend with llama.cpp's weight repack on.

The F16 speedup holds at 5.1-5.2x from `pp512` to `pp2048`. Under quantized residency the `Q8_0`
and `Q4_K_M` GGUFs prefill at about the F16 rate at every length, with 92-95% of their weights
resident. The rest stream, because the RAM reserve or the NPU's IOVA window stops the admission.

Decode follows the quantization: 1.07 t/s at F16, 1.97 t/s at `Q8_0` and 3.05 t/s at `Q4_K_M`.
The F16 GGUF is 22 GiB, so this model needs a quantized GGUF for interactive use. A 2048-token
prompt with a 200-token answer takes:

| Configuration | Time to first token | Decode, t/s | Total |
|---|---:|---:|---:|
| F16, CPU | 434 s | 1.07 | 621 s |
| F16, NPU | 83 s | 1.07 | 270 s |
| `Q4_K_M`, NPU | 84 s | 3.05 | 149 s |

**Conditions:** the rates above [host-computed]. The time to first token is the prompt over the
`pp2048` rate, and the total adds 200 tokens at the decode rate.

A 400-token chat answer takes 374 s at F16 and 131 s at `Q4_K_M`. `Q4_K_M` with the NPU is the
practical configuration: it decodes at 2.9x the F16 rate. Its 6.86 GB GGUF fits an 8 GB board,
where it streams its weights. Its prefill there is lower than the table's resident rate
[expected]: quantized residency measured 1.35-1.75x the streamed default on seven dense models,
as [TUNING.md](TUNING.md#quantized-weights) gives.

### Phi-4 (14B)

Phi-4 has 14.66 B parameters with dense GQA, under llama.cpp's `llama` architecture rather than
Phi-4-mini's `phi3`. Its F16 GGUF is 29.3 GB and does not fit a 31 GiB board, so it runs
quantized. At `pp2048`, `Q4_K_M` prefills at 21.9 t/s, 4.0x a CPU backend with llama.cpp's weight
repack on, which runs at 5.47 t/s [HW sweep 2026-11-01]. Its wikitext perplexity differs from the
CPU backend's by −0.09% at `Q8_0` and −0.31% at `Q4_K_M`.

**Conditions:** `-b 2048 -ub 2048`.

`Q4_K_M`, at 8.28 GB, fits a ~9 GB budget and decodes at 2.7 t/s.

### Qwen3.5 and Qwen3.6

Qwen3.5 (0.8B and 9B, dense GQA) and Qwen3.6-27B, a hybrid with Gated-DeltaNet linear attention,
prefill on the NPU through the same drop-in. The DeltaNet and SSM-scan layers are CPU-only ops,
and the FFN and the projections offload, so the hybrid keeps its speedup.

At `pp2048`, Qwen3.6-27B at `Q4_K_M` prefills at 11.5 t/s, 3.8x a CPU backend with its weight
repack on, which runs at 3.0 t/s. Qwen3.5-9B at F16 prefills at 36.1 t/s, 4.9x the CPU backend
[HW sweep 2026-11-01].

**Conditions:** the 27B at `-b 2048 -ub 2048`, the 9B at the default `-b` and `-ub`.

Qwen3.5-9B's wikitext perplexity differs from the CPU backend's by +0.05% at F16, −0.07% at
`Q4_K_M` and −0.10% at `IQ4_XS`. Qwen3.6-27B's differs by −0.26% at `Q4_K_M` [HW sweep
2026-07-02/03].
Each is well inside its run's own error estimate of ±4.5-4.7%.

**Not characterized:** Qwen3.5-0.8B's perplexity.

Importance-matrix quantizations such as `IQ4_XS` take the same route as any quantized type,
because the importance matrix is used only at quantization time. `IQ4_XS` has 256-element
super-blocks, so `K%32` holds. Qwen3.5-9B at `IQ4_XS` prefills at 3.1x a CPU backend with the
repack off at `pp2048` [HW sweep 2026-07-02]. At 4.80 GB, it is that model's fastest-decoding
quantization.

### Llama-3.2-3B, Ministral-3-3B and Phi-4-mini

These 3-4 B dense models use GQA, under llama.cpp's `llama`, `mistral3` and `phi3` architectures.
At F16 and `pp2048`, Llama-3.2-3B prefills at 85.1 t/s, 5.0x the CPU backend. Ministral-3-3B
prefills at 75.0 t/s, 4.7x, and Phi-4-mini at 76.1 t/s, 5.0x.

**Conditions:** llama-bench's default `-b` and `-ub` [HW sweep 2026-11-01].

At `Q4_K_M` they decode at 11.1, 10.5 and 8.8 t/s. At F16, their wikitext perplexity differs from
the CPU backend's by +0.02% for Llama-3.2-3B and −0.04% for Ministral-3-3B. At `Q4_K_M`,
Llama-3.2-3B is 1.87 GB and Ministral-3-3B is 1.99 GB, so each fits a 4 GB board.

### SmolVLM2-2.2B

SmolVLM2-2.2B runs its full pipeline on the NPU through llama.cpp's `mtmd` library. It pairs a
SigLIP-SO400M vision encoder with a 1.81 B SmolLM2-class language model under llama.cpp's `llama`
architecture.

The language model prefills at 117.8 t/s at F16 and `pp2048`, 4.2x the CPU backend
[HW sweep 2026-11-01]. It decodes at 20.7 t/s at `Q4_K_M` from a 1.03 GB file. Its wikitext
perplexity differs from the CPU backend's by −0.07% at F16 and −0.71% at `Q4_K_M` [HW sweep
2026-07-04].

> [!CAUTION]
> The vision encoder runs through `clip.cpp`, whose automatic device selection probes only the
> GPU and IGPU device types. Name the NPU's ACCEL device in `MTMD_BACKEND_DEVICE`. Without it, the
> encoder stays on the CPU.

With the NPU attached, the SigLIP encode is ~1.19x as fast as on the CPU backend. Its projected
image embeddings match the CPU backend's at cosine 0.99998. The gain is modest because only the
aligned static-weight GEMMs offload: the q, k, v and o projections and fc1. The second FFN
projection, fc2 (`K`=4304, not a multiple of 32), stays on the CPU with the attention core and
every norm, softmax and GELU.

## Precisions

### int8 W8A8

`ROCKET_INT8=1` runs the RK3588's offloaded matmuls as W8A8, with the weights and the activations
quantized to int8. With the Hadamard rotation on (`ROCKET_INT8_HADAMARD=1`), the output is
coherent and character-identical to fp16. `ROCKET_INT8_RESIDENT=1` holds the int8 weights
[resident](README.md#terms). Resident int8 prefill measures 0.95-0.99x resident fp16 on two 3 B
models and about resident fp16's rate on an 8 B model ([TUNING.md](TUNING.md#quantization-and-prefill-speed)).

Wikitext perplexity over 8 chunks of 512 rises with the int8 route as follows:

| Model, F16 GGUF | int8, no rotation | int8, `ROCKET_INT8_HADAMARD=1` |
|---|---:|---:|
| Llama-3.2-3B-Instruct | +1.5% | not measured |
| Ministral-3-8B-Instruct | +0.73% | +0.93% |
| Qwen3.5-9B | +2.83% | +0.97% |

**Conditions:** the resident int8 route against fp16 on the same GGUF, RK1, NPU at 928 MHz,
llama.cpp b11539. The 3 B row was measured 2026-11-06 against resident fp16 (11.0729 against
10.9114), and the 8-9 B rows 2026-11-07 against the default fp16 route.

Perplexity under-reads the route's distance from fp16. The rotation cuts the KL divergence from
the fp16 route's token distribution on all three models measured, including Ministral-3-8B, whose
perplexity it does not lower. Gemma-4-12B-it needs the rotation:

| Model, F16 GGUF | Mean KL divergence, no rotation | Mean KL divergence, `ROCKET_INT8_HADAMARD=1` | Same top token, no rotation | Same top token, rotated |
|---|---:|---:|---:|---:|
| Ministral-3-8B-Instruct | 0.0166 | 0.0063 | 93.7% | 96.7% |
| Qwen3.5-9B | 0.0330 | 0.0084 | 92.0% | 95.8% |
| Gemma-4-12B-it | 1.37 | 0.123 | 58.7% | 89.5% |

**Conditions:** `llama-perplexity --kl-divergence` on wikitext-2, 8 chunks of 512, against the
default fp16 route's logits, every int8 weight resident, RK1, NPU at 928 MHz, llama.cpp b11539.
Gemma was measured 2026-11-10, the 8-9 B models 2026-11-11.

The rotation applies only to a projection whose `K` is a power of two or 60 times one. On these
8-9 B models it therefore skips the FFN down projection. It lowers Qwen3.5-9B's perplexity at every
chunk and does not lower Ministral-3-8B-Instruct's. On both it lowers the KL divergence above.

On the RK3576, `ROCKET_INT8=1` selects [the RK3576 route](README.md#terms) instead, which
[RK3576.md](RK3576.md) describes.

### int4 W4A4

`ROCKET_INT4=1` runs prefill as native int4 x int4 matmuls from an F16 GGUF, validated on
Gemma-4-12B. Weights and activations are quantized to 4-bit, symmetric `[-7,7]`, with group-wise
scales and a Hadamard rotation, both on by default. Each scale covers a K-group of 128 elements
by default (`ROCKET_INT4_GROUP`), and the group size sets the accuracy.

Its generated text is character-identical to fp16's on the prompts tested: "Paris / Eiffel Tower /
Louvre Museum", primes and Jupiter. Per-matmul cosine against fp16 is ~0.978 per-channel with the
rotation, ~0.987 at group 128 and ~0.989 at group 64. Without the rotation, W4A4 falls to ~0.79.
The rotation needs full-precision weights, so the route quantizes from an F16 GGUF and not from a
pre-quantized Q4 one.

`ROCKET_INT4_RESIDENT=1` holds the group-wise int4 weights resident, packed as 4-bit nibbles. The
rotation is applied to the resident weight once, and each call rotates and quantizes only the
activation, at the same accuracy:

| Route | Prefill, t/s | Weights' NPU footprint |
|---|---:|---:|
| fp16 | 13.18 | ~10.5 GB |
| int4, resident | 6.94 | 2634 MB |
| int4, per call | 3.56 | |

**Conditions:** Gemma-4-12B F16 GGUF, `pp512`.

Resident int4 runs at ~2x the per-call int4 route and 0.53x fp16, with the weights' NPU footprint
at a quarter of fp16's. A deep-`K` matmul in group-wise int4 reads back ~120 K-group partials,
where fp16 with K-accumulation reads back once, and that keeps int4 slower.

`ROCKET_INT4_CACHE_MB` caps the resident total at 4096 MB by default. A full 12B model is ~6 GB of
int4 weights, so raise the cap for full coverage. The one-time rotation of each weight runs on
every CPU core and takes about 3 min for a 12B model.

### bf16

`ROCKET_BF16=1` selects the exact bf16 route, which writes an fp32 result straight to the output.
bf16 has fp32's exponent range, so the activations need no per-row scale. Without the variable, a
bf16 weight converts to fp16 on every micro-batch and takes the fp16 route at fp16 accuracy. Its
prefill then runs at ~0.55-0.6x an F16 GGUF's rate.
