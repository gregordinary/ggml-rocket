# ggml-rocket: API and tuning reference

This reference covers the settings for each workload, the results for each model and
[route](README.md#terms), and every `ROCKET_*` environment variable and diagnostic. Start with
the [README](README.md), which covers the build and a first run.

## Recommended configurations

The backend's defaults enable these optimizations, and none needs an environment variable:

- fp16 [K-accumulation](README.md#terms) on the NPU (`ROCKET_KACC`)
- [CBUF](README.md#terms) operand reuse, chosen per shape (`ROCKET_REUSE`)
- CBUF weight streaming (`ROCKET_MM_STREAM`)
- Asymmetric tiling (`ROCKET_MM_ASYM`)
- Multi-threaded dequantization (`ROCKET_DEQUANT_THREADS`)
- Attention offload (`ROCKET_FLASH_ATTN`)

Set the backend up as in [Build and run](README.md#build-and-run): the 900 MHz NPU clock request,
`sudo -E`, and an absolute `GGML_BACKEND_PATH`. The 900 MHz request runs at 928 MHz on the RK1's
0.80 V rail. Then add the settings the table gives for the workload.

Each residency setting trades RAM for prefill speed. A weight that does not fit the residency
budget is [streamed](README.md#terms) instead, so requesting residency does not fail a run.

| Workload | Precision | Add | Effect |
|---|---|---|---|
| Interactive chat, short prompts | Any | Nothing. Choose a `Q4_K_M` GGUF for decode t/s | The turn is decode-bound, so the quantization type sets decode t/s. A quantized prompt under 512 rows stays on the CPU (`ROCKET_MIN_M_QUANT`), and an F16 one under 32 (`ROCKET_MIN_M`) |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form fits RAM, wholly or partly | `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, without `-b 2048 -ub 2048` | Dequantizes and packs each weight once: 1.35-1.75x the streamed default on seven dense models of 1.8-11.9 B [HW sweep 2026-08-29/31]. Needs free RAM of about the fp16 model size. See [Quantization and prefill throughput](#quantization-and-prefill-throughput) |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form does not fit | `-b 2048 -ub 2048` | `-ub 2048` spreads the per-micro-batch dequantization over more rows. 0.91-1.53x the `-ub 512` default, and slower on two models of eleven, so measure it per model |
| Agentic or RAG, repeated prefill | F16, with RAM for ~2x the model | `ROCKET_F16_RESIDENT=auto` | Packs each weight once across turns, for a single-digit-percent gain |
| Medium or long prompts | MoE (gpt-oss, DeepSeek-V2-Lite) | `-b 2048 -ub 2048`, measured per model | Routed experts are resident as int8 by default, at 1.9x a repacked CPU backend on gpt-oss (2026-09-29 build). See [MoE models](#moe-models) |
| Model too large at F16 | n/a | A `Q4_K_M` GGUF, or `ROCKET_INT4=1` with an F16 GGUF | Reduces the footprint. Quantization does not speed prefill at this operating point |
| Transcription service (`whisper-server` fed fixed-length chunks) | F16 Whisper | `-nt -sns`, `-ac 50*(chunk_s+4)` capped at 1500, and `temperature_inc=0` as a request field | 0.70x the server defaults' CPU core-seconds per second of audio on 20 s chunks, at better WER. See [Transcription service](#transcription-service) |
| Voice commands, short utterances (Parakeet through whisper.cpp's `parakeet-cli`) | F16 | `ROCKET_MIN_M=16` in the STT process, and `OMP_WAIT_POLICY=PASSIVE` where another process shares its cores | Offloads the commands shorter than 2.6 s, which the default threshold keeps on the CPU. See [Voice commands](#voice-commands) |

**Conditions:** RK3588. The figures in the LLM rows are at `pp2048`, with the NPU at ~816 MHz
(the 600 MHz request). The speech subsections give their own conditions.

`ROCKET_INT8` and `ROCKET_INT4` generate text character-identical to fp16's, and `ROCKET_BF16`
computes an exact fp32 output. None of the three prefills faster than fp16: resident int8
measures 0.60x fp16 and resident int4 0.53x. Use them to fit a model in less RAM, never to speed
prefill.

Run once with `ROCKET_LOG_STDERR=1` to confirm that the intended route is in use. Without it,
`llama-bench` prints no backend lines.

### MoE models

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request).

The routed experts of a quantized MoE model are resident as int8 by default, and no environment
variable is needed. The placement leaves on the CPU any stack that does not fit the resident
budget, or whose per-expert GEMM is under either size threshold. The table's 1.9x is against a
CPU backend with llama.cpp's weight repack on.

At `pp2048`, `-ub 2048` measures 1.254x the default `-ub` on gpt-oss-20b and 1.323x on
DeepSeek-V2-Lite. On Qwen3-30B-A3B it measures 0.908x, and the backend declines that model's
experts [HW sweep 2026-08-29]. Measure it per model.

Where the stack nearly fits, `ROCKET_MOE=1` is 6-13% faster than the default at `pp512` and
18-21% at `pp2048`. Where the stack does not fit, it is slower than the experts-on-CPU baseline.

### Transcription service

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), `ggml-small` F16, `whisper-server`
fed fixed-length chunks.

The table's 0.70x is on 20 s chunks, against the server's defaults, at better WER. On 30 s
chunks with `-nt`, the server takes 0.65x the defaults' CPU core-seconds per second of audio
[HW sweep].

The server's `-nf` flag has no effect, so the `temperature_inc=0` request field turns the
temperature fallback off. Without `-nt`, a 30 s chunk is encoded twice.

> [!CAUTION]
> An `-ac` set to the chunk length itself makes the decoder loop at the end of the audio. Size
> `-ac` as the table gives it.

> [!NOTE]
> A quantized Whisper model costs more CPU time than the F16 one. Its unnamed tensors keep the
> encoder off the resident route, so each request dequantizes the encoder.

### Voice commands

**Conditions:** RK3588, Parakeet F16 through whisper.cpp's `parakeet-cli`.

A 1-5 s command is 16-75 encoder rows. The default `ROCKET_MIN_M` of 32 offloads a command from
2.6 s up, and a shorter one runs exactly as it does on the CPU backend. At 16, the encoder ties
the CPU at 1.3-1.8 s and runs 1.18-1.73x faster at 2-6 s. It frees 34-61% of the CPU
core-seconds, with the transcript unchanged [HW sweep].

Beside a CPU-bound process, the offloaded STT process slows more than a CPU-only one does.
`OMP_WAIT_POLICY=PASSIVE` recovers most of that slowdown.

Leave an LLM process at the default `ROCKET_MIN_M`.

### CPU thread placement

Pin a prefill-heavy run to the A76 big cores with `taskset 0xf0`, and do not pin a decode-heavy
one. Pinning gives 1.05-1.13x on NPU prefill on three models, largest on the smallest. The gain
is on the host: the NPU's share of the work is the same either way. It also holds with streamed
weights, because Gemma-4-12B F16 streams all of its weights and still gains 1.05x. Pinning slows
Gemma-4-12B F16 decode by 34%, because that decode is bound by LPDDR bandwidth and uses all eight
cores.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), CPU governor pinned [HW sweep].

### Evidence

The `taskset 0xf0` figures are rotated interleaved passes on three models, with the ratios paired
within a pass.

## Models and precisions

Each entry below is validated on RK3588 hardware unless its own text names another part.

### Whisper encoder

Stock `whisper-cli` and `whisper-bench` load the backend's `.so` through `GGML_BACKEND_PATH` and
run the encoder's `MUL_MAT` ops on the NPU. Transcripts are byte-identical to the CPU backend's,
at WER 0 and encoder-output cosine 0.9998.

The encoder is faster than on the CPU backend at every model size, and the speedup grows with the
model:

| Model | Encoder speedup over the CPU backend |
|---|---|
| tiny.en | 1.18x |
| base.en | 1.29x |
| small.en | 1.54x |
| medium.en | 1.84x |
| large-v3 | 2.14x |
| large-v3-turbo | 2.12x |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), `whisper-bench` encode, warm.

The decoder is autoregressive at `M`=1 and stays on the CPU with either backend. `large-v3-turbo`
keeps the full 32-layer encoder with a 4-layer decoder, ~6x cheaper per step. The offloaded encoder
is therefore the larger share of its pipeline.

### Multi-model STT through transcribe.cpp

The same `.so` drops into transcribe.cpp, a ggml-based host for Granite-Speech, Voxtral, MOSS
diarize, SenseVoice, FunASR and Parakeet. It offloads the encoder, as with Whisper.

The encoder always offloads, at 1.2x-2.7x, and the speedup scales with encoder size times audio
length. Autoregressive decode runs at `M`=1 and stays on the CPU. The exception is the batched
decode prefill over the injected audio context on long audio, which offloads once it is large
enough.

The model's architecture sets the overall speedup:

| Shape | Model | Result |
|---|---|---|
| Cross-attention decoder | Granite-Speech | Offloads decode best, ~1.9x on 120 s |
| Big decoder-only over long audio | Voxtral-mini 3B | Offloads its ~1500-token audio prefill: decode 1.46x, whole pipeline 1.57x |
| Small decoder-only | MOSS, FunASR 0.6B | Decode-bound, ~1.0-1.15x despite a ~1.6x encode |
| Single-pass CTC | SenseVoice | No decode: 6.2x realtime, 1.26x |

**Conditions:** RK3588, Q8_0, the process on the A76 cores, warm, one fresh process per run,
against transcribe.cpp's CPU backend [HW sweep 2026-07-21]. The record does not state the NPU
clock.

With only the encoder offloaded, the transcript is identical to the CPU backend's. A model whose
decoder cross-attention offloads, such as Granite, can diverge late, as fp16 accumulation perturbs
greedy decoding.

Q8_0 is the recommended quantization for these models. It matches or beats F16 on both backends at
half the RAM [HW sweep 2026-07-21]. These models are less matmul-dominated than LLM prefill, so
the per-micro-batch dequantize costs little.

### Gemma-4-12B

Gemma-4-12B generates coherently through the same drop-in on llama.cpp. Its prefill GEMMs
offload, and decode stays on the CPU.

NPU prefill keeps wikitext perplexity within ±2% of the CPU backend's across F16, Q8_0 and
Q4_K_M, inside the ±10% per-run noise. The absolute perplexity of this instruct model on raw text
is not meaningful.

**Conditions:** RK3588 with an 8-core CPU, NPU ~816 MHz (the 600 MHz request), warm. The CPU
backend runs the quantized GGUFs with llama.cpp's weight repack off.

#### Prefill

Prefill throughput in t/s, CPU backend -> NPU, with the speedup:

| Precision | `pp512` | `pp1024` | `pp2048` |
|---|---|---|---|
| F16    | 4.82 -> 17.42 (3.6x) | 4.78 -> 16.09 (3.4x) | 4.63 -> 14.98 (3.2x) |
| Q8_0   | 4.52 -> 11.41 (2.5x) | 4.35 -> 13.53 (3.1x) | 4.16 -> 13.43 (3.2x) |
| Q4_K_M | 4.23 -> 11.20 (2.6x) | 4.15 -> 13.19 (3.2x) | 4.02 -> 13.28 (3.3x) |

F16 runs at the default `-b` and `-ub`, and the quantized GGUFs at `-b 2048 -ub 2048`
[HW sweep 2026-07-02].

On that build, F16 prefill is fastest at pp512, 17.4 t/s, and its speedup narrows with `M` from 3.6x to
3.2x as the output readback grows. The quantized GGUFs speed up with `M` as their per-micro-batch
dequantize amortizes, and they converge on F16 by pp2048 at ~13-15 t/s. A quantized GGUF is
smaller, and at this operating point it does not prefill faster.

A later build measures F16 at 19.96 t/s at pp2048, at the default `-b` and `-ub` and the CPU
governor `performance` [HW sweep 2026-09-02, llama.cpp b10558].

**Not characterized:** the CPU backend on the 2026-09-02 build.

#### Decode

Decode runs on the CPU with or without the NPU, at about the same rate. It is bandwidth-bound, so
it follows the quantization: 0.9 t/s at F16, 1.7 t/s at Q8_0 and 2.5 t/s at Q4_K_M. At F16 the
GGUF is 22 GiB and decode is slow at 0.9 t/s, so a quantized GGUF is effectively mandatory for
interactive use of this model.

#### Interactive use

The NPU shortens the time to first token, and a quantized GGUF raises the decode rate:

| Turn | Configuration | Time to first token | Decode | Total |
|---|---|---|---|---|
| 2048-token prompt -> 200 out (RAG) | F16, CPU | 442 s | 0.9 t/s | 655 s |
| | F16, NPU | 137 s | 0.9 t/s | 350 s |
| | Q4_K_M, NPU | 154 s | 2.5 t/s | 234 s |
| 128-token prompt -> 400 out (chat) | F16, NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M, NPU | 11 s | 2.5 t/s | 171 s |

The NPU cuts the time to first token ~3.2x on a long prompt. This model's decode is slow, so
decode dominates the turn, and Q4_K_M with the NPU is the practical choice. It gives ~3.3x the
prefill of a CPU backend with the repack off at long prompts, and 2.7x the F16 decode rate. Its
GGUF is 6.86 GB, which fits an 8 GB board. Where RAM is ample, F16 gives the highest prefill,
from a 22 GiB GGUF.

#### Evidence

The 19.96 t/s figure is six rotated passes.

### Phi-4 (14B) prefill

Phi-4 has 14.66 B parameters with dense GQA, and its llama.cpp architecture is `llama`, which
differs from Phi-4-mini's `phi3`. It prefills on the NPU through the same drop-in. Its wikitext
perplexity differs from the CPU backend's by −0.09% at Q8_0 and −0.31% at Q4_K_M.

Its F16 GGUF is 29.3 GB and does not fit a 31 GB board, so it runs quantized. At pp2048, Q4_K_M
prefills at 16.7 t/s, 3.1x the 5.45 t/s of a CPU backend with llama.cpp's weight repack on
[HW sweep 2026-09-29]. Against a CPU backend with the repack off, at ~3.4 t/s, Q8_0 and Q4_K_M
measure ~3.4-3.5x. The speedup rises with prompt length as the per-micro-batch dequantize
amortizes.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), `-b 2048 -ub 2048`, CPU governor
pinned.

Q4_K_M, at 8.28 GB, is the practical choice. It fits a ~9 GB budget and decodes at ~2.2 t/s. By
size it sits between Gemma-4-12B and Qwen3.6-27B.

### Qwen3.5 and Qwen3.6 prefill

Qwen3.5 (0.8B and 9B, dense GQA) and Qwen3.6-27B, a hybrid with Gated-DeltaNet linear attention,
prefill on the NPU through the same drop-in, with perplexity agreeing with the CPU backend's.
Qwen3.5-9B's wikitext perplexity differs from the CPU backend's by +0.05% at F16, −0.07% at
Q4_K_M and −0.10% at IQ4_XS. Qwen3.6-27B's differs by −0.26% at Q4_K_M [HW sweep 2026-07-02/03,
NPU ~816 MHz (the 600 MHz request)]. Each is well inside its run's own error estimate of
±4.5-4.7%.

**Not characterized:** Qwen3.5-0.8B's perplexity.

The DeltaNet and SSM-scan layers are CPU-only ops and stay on the CPU. The FFN and the projections
offload, and the prefill speedup holds. The hybrid's linear-attention layers do not prevent the
offload.

At pp2048, Qwen3.6-27B at Q4_K_M prefills at 9.6 t/s, 3.2x the 3.0 t/s of a CPU backend with its
weight repack on [HW sweep 2026-09-29]. It is the largest speedup among the quantized models,
with the NPU at ~816 MHz (the 600 MHz request).
Against a CPU backend with the repack off, an especially slow 1.8 t/s at 27 B, it measured 4.4x.

The NPU's prefill speedup over the CPU backend grows with model size. At F16 it is 1.44x on the
0.8B at pp512 and 3.65x on the 9B at pp2048 and `-ub 2048` [HW sweep 2026-06-27/28].

**Conditions:** NPU ~816 MHz (the 600 MHz request), the best runs of a scaling sweep.

**Not characterized:** the size scaling on a later build.

For a quantized GGUF whose fp16 image fits RAM, set `ROCKET_QUANT_RESIDENT=auto` at the default
`-ub`, without `-b 2048 -ub 2048`. Use `-b 2048 -ub 2048` only where the fp16 image does not fit.
[Quantization and prefill throughput](#quantization-and-prefill-throughput) covers both, and
[Matmul offload](#matmul-offload) has `ROCKET_MIN_M_QUANT`.

### Small dense models: Llama-3.2-3B, Ministral-3-3B, Phi-4-mini

The 3-4 B dense models use GQA, with llama.cpp architecture `llama`, `mistral3` or `phi3`. Each
matmul has fewer FLOPs than a larger model's, so they prefill fast. At F16 and `pp512`,
Llama-3.2-3B prefills at 61.8 t/s, 3.4x the CPU backend. Ministral-3-3B prefills at 57.6 t/s and
Phi-4-mini at 55.4 t/s.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), llama-bench's default `-b` and `-ub`,
against the 8-thread CPU backend [HW sweep 2026-07-02/03, llama.cpp b9568].

The F16 speedup narrows with prompt length, from 3.4x at pp512 to 2.4-2.6x by pp2048. A small
model's per-op readback grows with `M`, so the speedup is largest on short-to-medium prompts.

Their pp2048 rates on llama.cpp b10558 are in the [Resident weights](#resident-weights) table.

**Not characterized:** the CPU backend and pp512 on llama.cpp b10558.

At Q4_K_M, Llama-3.2-3B is 1.87 GB and Ministral-3-3B is 1.99 GB, so each fits a 4 GB board, and
they decode fastest at ~7.7-7.9 t/s. At F16, their wikitext perplexity differs from the CPU
backend's by +0.02% for Llama-3.2-3B and −0.04% for Ministral-3-3B.

Their speedup over the CPU is modest, because their CPU baseline is comparatively fast, while
their absolute prefill rate on the NPU is high.

### Vision-language: SmolVLM2-2.2B

SmolVLM2-2.2B runs its full pipeline on the NPU through llama.cpp's `mtmd` library. It pairs a
SigLIP-SO400M vision encoder with a SmolLM2-class language model of 1.81 B, llama.cpp architecture
`llama`. The 2.2B in the name is the combined count.

The language model is the fastest in this backend. At F16 it prefills at 99 t/s at pp512, 3.5x the
CPU backend. At Q4_K_M it has the fastest decode here, 14.5 t/s from a 1.03 GB file. Its wikitext
perplexity differs from the CPU backend's by −0.07% at F16 and −0.71% at Q4_K_M.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request) [HW sweep 2026-07-04].

> [!CAUTION]
> The vision encoder runs through `clip.cpp`, whose automatic device selection probes only the
> GPU and IGPU device types. Name the NPU's ACCEL device in `MTMD_BACKEND_DEVICE`. Without it, the
> encoder stays on the CPU.

With the NPU attached, the SigLIP encode is ~1.19x faster than on the CPU backend, warm. Its
projected image embeddings match the CPU backend's at cosine 0.99998. The speedup is modest,
because the backend offloads only the aligned static-weight GEMMs: the q, k, v and o projections
and fc1.

Three parts stay on the CPU: fc2 (`K`=4304, not a multiple of 32), the attention core, and every
norm, softmax and GELU op. The graph therefore moves between the CPU and the NPU many times. The
drop-in `clip.cpp` integration is correct and gives a modest speedup.

### Quantized GGUF prefill

A `Q8_0`, `Q4_K`, `Q6_K` or similar GGUF runs its prefill GEMMs on the NPU directly. The backend
dequantizes each weight to fp16 as it is used, one `[K]` row at a time, to the exact values ggml's
CPU backend computes. It holds no whole-model F16 copy in RAM. The host's weight repack must be
off, with `-nr` or a `-DGGML_CPU_REPACK=OFF` build, as the README's
[Drop into llama.cpp](README.md#build-and-run) recipe describes.

The model therefore stays compact on disk and in memory: Gemma-4-12B at `Q8_0` is ~12 GB against
the F16 GGUF's ~24 GB. Prefill still takes the fp16 [route](README.md#terms) on the NPU, and
decode runs on the CPU from the quantized weights. Quantization reduces RAM and lets a larger model fit, and it does not
raise prefill speed. The matmul's throughput is independent of the datatype, as
[Quantization and prefill throughput](#quantization-and-prefill-throughput) describes.

The dequantize runs on every micro-batch, so quantized prefill depends on the micro-batch size. If
the fp16 image fits RAM, set `ROCKET_QUANT_RESIDENT=auto` at the default `-ub`, which dequantizes
each weight once. Otherwise run `-b 2048 -ub 2048`, for a model-dependent 0.91-1.53x over
llama.cpp's default `-ub 512`, and slower on two models of eleven. Both are under
[Quantization and prefill throughput](#quantization-and-prefill-throughput).

A prefill shorter than `ROCKET_MIN_M_QUANT`, 512 rows by default, stays on the CPU, where it is
faster than a dequantize-bound NPU offload. The default sits just above the measured crossover of
~360 rows.

Importance-matrix quantizations such as `IQ4_XS` take the same route, because the importance
matrix is used only at quantization time. At inference they are ordinary types that dequantize to
fp16. `IQ4_XS` is 4.25 bits per weight with 256-element super-blocks, so `K%32` holds. Its prefill
GEMMs offload like any quantized type instead of falling back to the CPU. Qwen3.5-9B at
`IQ4_XS` prefills at 3.1x a CPU backend with the repack off at pp2048 [HW sweep 2026-07-02]. It
is that model's fastest-decoding quantization, at 4.80 GB.

### Resident weights

A resident weight is packed into NPU buffers once and reused by every later micro-batch and
prefill. Its pack (`packB`) runs once instead of on every call. The backend makes a static F16 weight
resident by default when its `K` is at most 2048, which covers Whisper's encoder.
`ROCKET_F16_RESIDENT` extends residency to every `K`, and `ROCKET_QUANT_RESIDENT` does the same
for a quantized GGUF (see [Quantization and prefill throughput](#quantization-and-prefill-throughput)).

A weight becomes resident the first time the backend sees it at `M` >= 256, the RK3588's tile
cap. Below that, a call streams the weight through the per-call `mt` route and leaves any
resident copy in place. A resident weight serves every `M` at or above the cap without being
packed again. A weight packed during warm-up therefore serves a later prefill of any length.

The cache is keyed on the weight's name and a fingerprint of its contents (sixteen 64-byte
windows), not on its address or on `M`. A weight that is reloaded or replaced under the same
name is packed again.

The environment variables that force, disable and size residency are in
[Weight residency](#weight-residency).

#### `ROCKET_F16_RESIDENT`

`ROCKET_F16_RESIDENT=auto` makes every F16 weight resident within a budget sized from free RAM.
That includes the LLM attention and FFN weights, at `K` of 3072, 8192 and above. It takes the same
`auto`, `<N>` (MB) and `1` values and the same reserve as `ROCKET_QUANT_RESIDENT`. It does not
`madvise` the GGUF, so the file stays mapped for CPU decode.

It keeps QKV and gate-up fusion. Each fusable group becomes one resident weight, its members
concatenated along `N`. The group is packed once, and still shares one activation pack and one
submit.

| Model, F16 | Streamed weights, t/s | `ROCKET_F16_RESIDENT=auto`, t/s | Speedup |
|---|---:|---:|---:|
| Llama-3.2-3B | 56.01 | 60.54 | 1.078-1.084x |
| Ministral-3-3B | 48.28 | 52.04 | 1.078-1.084x |
| Phi-4-mini | 50.19 | 54.40 | 1.078-1.084x |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), llama.cpp b10558, `pp2048` at the
default `-b` and `-ub`, every weight resident and none streamed [HW sweep 2026-08-29].

**Not characterized:** `pp512` and `pp1024` on the current build.

The resident tiles duplicate the still-mapped GGUF, so the full benefit needs RAM for about twice
the F16 model. A model too large for that is made partly resident. Admission stops at a
`MemAvailable` minimum and the rest is streamed, so a board without swap does not run out of
memory.

#### Teardown report

Both `ROCKET_F16_RESIDENT` and `ROCKET_QUANT_RESIDENT` report their outcome when the backend is
freed:

    [f16-resident] weights offered to the resident route: N resident on the NPU (X MB),
    M streamed via the per-call pack -- P% resident

The next line names the limit that first declined a weight, with its numbers: the byte budget,
the `MemAvailable` minimum, or a full IOVA window. Below 80% resident, a warning follows.

The count is of weights offered to the resident route, and each member of a fused group counts
once. The percentage therefore means the same with fusion on and off. The report goes through the
`librocketnpu` log channel, so `ROCKET_LOG_STDERR=1` shows it under a host that silences ggml.

A matmul that was never offered to the route is in neither count, and its weight is still
packed on every call. That covers a call below `ROCKET_MIN_M` and a weight with no stable name, which the report
counts on a separate line. On a 12 B F16 model with all 328 weights resident and none
streamed, `ROCKET_MM_PROFILE` still attributes 12 s to `packB`.

> [!NOTE]
> A benchmark row does not show residency. A run with a quarter of its weights resident prints
> the same rows as a run with all of them resident that gained nothing. The two can
> also measure nearly the same t/s. Read the teardown report before concluding that residency
> had no effect.

#### Evidence

The `ROCKET_F16_RESIDENT` figures are three rotated passes per configuration. The speedup range is
the spread of the per-pass ratios, paired within a pass.

An earlier build measured a larger gain at shorter prefills: Llama-3.2-3B +9.5% at `pp512`
against +5.9% at `pp2048`, with output token-identical to the streamed configuration. On that
build Phi-4-mini, whose single `qkv` and `gate_up` weights leave nothing to fuse, measured +21%
at `pp512` and +16% at `pp1024`. The likely cause of the smaller long-prefill gain is that
attention stays on the CPU at long prefill and dilutes the offloaded matmuls' share
[hypothesis].

The key includes the content fingerprint because the scheduler reuses pooled addresses across
weights. A cache keyed on the address served one weight's packed tiles for another, and the model
produced gibberish.

### int8 W8A8 with a Hadamard rotation

`ROCKET_INT8=1` runs the RK3588's offloaded matmuls as W8A8, with the weights and the activations
quantized to int8. With the Hadamard rotation on (`ROCKET_INT8_HADAMARD=1`), the output is
coherent and char-identical to fp16. The [route](README.md#terms) is slower than the default fp16
route. Its int32 readback is not [K-accumulated](README.md#terms) on the NPU, and resident int8
prefill measures 0.60x fp16 (see
[Quantization and prefill throughput](#quantization-and-prefill-throughput)). Use it to fit a
model in less RAM, not to speed prefill.

`ROCKET_INT8_RESIDENT=1` holds the int8 weights [resident](README.md#terms), and
[Precision](#precision) lists the route's environment variables. On the RK3576, `ROCKET_INT8=1`
selects [the RK3576 route](README.md#terms) instead, which [The RK3576](#the-rk3576) describes.

### int4 W4A4 in-model

`ROCKET_INT4=1` runs prefill as native int4xint4 matmuls from an F16 GGUF, validated on
Gemma-4-12B. Weights and activations are quantized to 4-bit, symmetric `[-7,7]`, with group-wise
scales and a Hadamard rotation, both on by default. Each scale covers a K-group of 128 elements by
default, and the group size sets the int4 accuracy.

Its generated text is char-identical to fp16's on the prompts tested: "Paris / Eiffel Tower /
Louvre Museum", primes and Jupiter. Per-matmul cosine against fp16 is ~0.98-0.99: ~0.978
per-channel with the Hadamard rotation, ~0.987 at group 128 and ~0.989 at group 64. Without the
Hadamard rotation, W4A4 falls to ~0.79.

The Hadamard rotation needs full-precision weights, so the int4 route quantizes from an F16 GGUF
and not from a pre-quantized Q4 one. `ROCKET_INT4_GROUP` sets the group size (default 128, where 0
is per-channel) and `ROCKET_INT4_HADAMARD` the rotation (default 1). [Precision](#precision) lists
them with `ROCKET_INT4_CACHE_MB`.

### int4 W4A4 resident

`ROCKET_INT4_RESIDENT=1` holds the group-wise int4 weights resident in NPU buffers, packed as
4-bit nibbles. The Hadamard rotation is applied to the resident weight once, and each call
rotates and quantizes only the activation (`A`). Prefill therefore has no per-call weight pack.
Accuracy is the same as the per-call int4 route's, at cosine 0.987, and the text is
char-identical to fp16's.

| Route | Prefill, t/s | Weights' NPU footprint |
|---|---:|---:|
| fp16 | 13.18 | ~10.5 GB |
| int4, resident | 6.94 | 2634 MB |
| int4, per call | 3.56 | |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), Gemma-4-12B F16 GGUF, pp512.

Resident int4 is ~2x the per-call int4 route, because it removes the weight pack (`packB`), and
0.53x fp16. It halves int4's prefill cost and cuts the weights' NPU footprint to a quarter of
fp16's.

Against fp16, resident int4 saves memory and does not raise speed. A deep-`K` matmul in
group-wise int4 reads back ~120 K-group partials, where fp16 with K-accumulation reads back
once, and that keeps int4 slower.

`ROCKET_INT4_CACHE_MB` caps the resident total. A full 12B model is ~6 GB of int4 weights, so
raise the cap for full coverage. The offloaded subset in the table is ~2.6 GB. The one-time
Hadamard build of each weight is parallelized across the CPU cores, ~14x, and takes about 3 min
for a 12B model.

The resident int4 weight is independent of `M`, because `librocketnpu` plans its tile layout at a
fixed `M`. The weight packed during warm-up serves any later prefill without a re-pack, a short
prompt included. The cache is keyed on the weight's name and a fingerprint of its contents, not
on the padded `M`.

A resident int8 weight is reused across `M` in the same way. A resident fp16 weight serves every
`M` at or above the tile cap, as [Resident weights](#resident-weights) describes.

### Evidence

The README's performance table pairs the `pp2048` rate of Llama-3.2-3B, Ministral-3-3B,
Phi-4-mini and Gemma-4-12B on llama.cpp b10558 with a speedup from the 2026-07 sessions. In those
sessions the NPU measured 45.5, 39.8, 39.8 and 15.0 t/s on the four models.

## MoE and MLA models

Mixture-of-experts models compute their expert FFNs with `MUL_MAT_ID`. The backend's `MUL_MAT_ID`
handler groups the routed `(slot,token)` rows by expert and runs each active expert's
`[M_e,K]x[N,K]ᵀ` GEMM on the NPU, spread across the workers. `M_e` is the number of rows routed to
that expert. The handler then scatters the rows back into the output. The models measured are
gpt-oss-20b and DeepSeek-V2-Lite.

With the default placement, gpt-oss-20b prefills 1.9x faster than on the CPU backend with
llama.cpp's weight repack on, at `pp2048` [HW sweep 2026-09-29]. Against the CPU backend with the
repack off it measured ~2.4x, and was faster at every prefill length [HW sweep 2026-08-27].
The gpt-oss-20b table under Native-quant experts, below, gives those measurements.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), gpt-oss-20b MXFP4, the backend builds
of 2026-09-29 (1.9x) and 2026-08-27 (~2.4x).

**Not characterized:** either CPU ratio on the current build.

Both ratios predate the current expert route, which is faster than the build before it. The current
build measures 24.90-25.23 t/s on gpt-oss-20b at `pp512`, and 32.69-32.83 t/s on DeepSeek-V2-Lite
at `pp2048` with a one-time expert ingest of 18.2-18.4 s. Those figures are the NPU configuration
alone, at `-b 2048 -ub 2048` and governor `ondemand` [HW sweep 2026-10-09].

Whether a MoE model's experts offload depends on its architecture, and one model's ratio does not
apply to another. DeepSeek-V2-Lite, below, is a MoE whose routing puts it under the
work-per-dispatch minimum at short prefill.

### Expert routes

An expert reaches the NPU by one of two [routes](README.md#terms), and only the
[native-quant](README.md#terms) route is faster than leaving the experts on the CPU:

- The fp16 dequant route (`ROCKET_MOE_NATIVE=0`) dequantizes each expert weight to fp16 on the
  host every micro-batch. A MoE layer has `n_expert` times more distinct weights than a dense
  layer, each amortized over only the rows routed to it. The dequantization does not shrink with
  the row count, so it costs more than the GEMM saves, while the CPU's fused quantized kernel
  dequantizes nothing. On gpt-oss-20b it measures 4.59 t/s at `pp512` and 10.18 at `pp2048`,
  slower than leaving the experts on the CPU. It is the comparison baseline, and the default
  placement never claims it.
- The native-quant route (`ROCKET_MOE_NATIVE`, on by default) converts a GGUF-quantized expert
  once into int8 codes held [resident](README.md#terms) in NPU buffers. This one-time conversion
  is the ingest. Each call then quantizes only the activation, per `(row, K-group)`. Removing the
  per-micro-batch dequantization is what makes this route faster than the CPU. An F16 expert
  always takes the fp16 route, because it has nothing to dequantize.

Native-quant experts, below, describes the native-quant route in detail.

### Placement

`ROCKET_MOE` selects which `MUL_MAT_ID` ops the backend [claims](README.md#terms). The
experts-on-CPU baseline in this section is the backend with `ROCKET_MOE=0`, which offloads the
dense matmuls and leaves the experts on the CPU.

| `ROCKET_MOE` | Placement |
|---|---|
| unset (default) | On the RK3588, claims an expert op only where the offload is measured to be faster. The stack must be GGUF-quantized, and the [pre-flight](README.md#terms) must reserve its whole `[K, N, n_expert]` stack before the first ingest. Each expert's GEMM must also meet both size minimums, `ROCKET_MOE_M_BUCKET` and `ROCKET_MOE_MIN_WORK`. Every other expert op stays on the CPU. |
| `1` | Claims every `MUL_MAT_ID` the handler can compute, reserving nothing and applying neither size minimum. That includes the fp16 route, for an F16 stack or one too large to hold resident, which measured 0.25-0.82x the CPU backend's prefill on DeepSeek-V2-Lite and gpt-oss-20b at `-ub 2048`. It is the comparison configuration, not a recommended setting. |
| `0` | Experts stay on the CPU, whatever the shape. |

Under the default, a stack the budget cannot hold is never claimed, so none of its experts is
[streamed](README.md#terms). Its layers run entirely on the CPU, which is the experts-on-CPU
baseline rather than a loss. The teardown split under the default is therefore expected to show
100% resident [expected]. Partial residency is reachable only with `ROCKET_MOE=1`. At an induced 12
GB budget, gpt-oss-20b falls to 52% resident and measures 0.97x the experts-on-CPU baseline at
`pp512`.

The unit of placement is one weight stack: the `[K, N, n_expert]` tensor of one layer's
`ffn_gate_exps`, `ffn_up_exps` or `ffn_down_exps`. A model whose whole expert set does not fit
still offloads the stacks that do. gpt-oss-20b's 19.1 GiB of int8 codes and its 11.3 GiB GGUF do
not both fit a 32 GB board, and it runs 99% resident there.

A declined stack runs on the CPU, so its weights are never dequantized per call. The result is a
blend of the offloaded route and the experts-on-CPU baseline, and never falls below the baseline
at any RAM size. A declined stack also costs no copy. The backend's buffer type is the CPU buffer
type, so a CPU-placed op beside an offloaded one is a scheduler entry, not a copy.

### Pre-flight

The pre-flight reserves a whole expert stack before the backend claims its op. It runs in
`supports_op`, before the first ingest, because a stack's resident cost follows from its tensor
alone: `[K, N, n_expert]` and the GGUF source stride. It charges the stack against two budgets,
RAM and the [IOVA window](README.md#terms). Each worker has a 4 GB IOVA window, of which the
pre-flight budgets 3840 MB, so the IOVA budget is the worker count times 3840 MB. A stack that does
not fit either budget stays on the CPU whole, rather than partly ingested. Native-quant experts,
below, describes the partial-residency loss that the pre-flight avoids.

Under the default, the charge for each placed stack is its int8 codes, its scales and its GGUF
source. A declined stack is charged nothing, although the CPU reads its experts from the same
mapping every micro-batch. The budget can therefore report room after the expert weights in use
have outgrown RAM. `ROCKET_MOE_CHARGE_ALL_SOURCE=1` charges the whole mapping instead. The startup
line states which charge is in force.

#### Several live backends

Each live backend holds its own resident copy of a placed stack, and the pre-flight charges each
copy. Placement is decided once per process, and the int8 codes live in the backend that converts
them. A stack is identified by a sampled fingerprint of its bytes, so two models with the same
tensor names are two stacks.

A backend claims a copy of a placed stack before it converts the stack's first expert. It takes
the pre-flight's copy where no live backend holds it. Otherwise it charges a new copy's codes,
scales and IOVA to the same budget. The GGUF source is charged once per stack, or once per buffer
with `ROCKET_MOE_CHARGE_ALL_SOURCE=1`, because both backends read one mapping. A copy that does not
fit is refused, and that backend dequantizes the stack to fp16 on every call, correctly and more
slowly.

A freed backend releases its copies after its resident buffers, so a backend that outlives it is
not refused for memory already freed. Under `ROCKET_MOE=1` the pre-flight reserves nothing, and
each backend's own budget decides at the ingest.

### Expert budget

`ROCKET_MOE_CACHE_MB` sets the budget in MB that the pre-flight and the ingest both charge, so the
claim and the ingest are judged against one number. `0` is unlimited. The default, `auto`, is
`MemAvailable` minus a flat 6 GiB reserve. Unlike the dense residency routes, whose reserve is
`max(6 GiB, 30% of RAM)`, it has no proportional term.

A flat reserve withholds a larger fraction of a smaller board: 19% of 32 GB, 37.5% of 16 GB and
75% of 8 GB. At 8 GB, `auto` effectively disables the route [expected].

> [!NOTE]
> `ROCKET_MOE_CACHE_MB` bounds the admission charge, not the RAM the experts occupy. A value that
> suits one quantization or one board does not transfer to another by RAM arithmetic alone.

#### Charge factor

Per expert, the admission charge is `N·K` bytes of int8 codes, `N·(K/group)·4` bytes of scales,
and the expert's GGUF source stride. A budget therefore holds less resident code than it names, by
the charge factor `1 + 4/group + source_bits_per_weight/8`. The source bytes are in the charge
because the GGUF stays mapped. MoE decode reads the active experts from it on the CPU every token,
so the codes and the source must coexist.

For Qwen3-30B-A3B Q4_K_M (4.87 bits per weight, `K`=2048, group 512) the factor is 1.617. For the
same placement it is about 2.07 for a Q8_0 MoE and 1.54 for IQ4_XS, derived from bits per weight
[expected].

The pre-flight's log gives the factor directly. When the budget stops admission, it logs
`resident budget reached after N expert stacks (XMB RAM of which SMB is GGUF source, YMB IOVA)`.
Under the default charge, `X/Y` is the charge factor as the run applied it. It is 1.610 on
Qwen3-30B-A3B and 1.521 on gpt-oss-20b MXFP4, against 1.617 and 1.538 derived. The formula sizes a
budget to within ~1% before a run, and the line confirms it after one.

#### Raising the budget

Raising the budget gives back stacks that the pre-flight's whole-stack reservation declines.
Unlike `ROCKET_MOE=1`, it keeps the default's guarantees: no streamed experts, and never below the
experts-on-CPU baseline. On gpt-oss-20b, more budget places more stacks and makes prefill faster:

| `ROCKET_MOE_CACHE_MB` | Stacks placed, of 72 | `pp512` | `pp2048` |
|---|---:|---:|---:|
| `auto` (24672) | 63 | baseline | baseline |
| 26000 | 66 | +4.3% | +5.1% |
| 28000 | 71 | +8.3% | +14.1% |

**Conditions:** RK3588, 31 GiB board, gpt-oss-20b, no streamed experts in any configuration,
gains against `auto` [HW sweep 2026-08-27].

The stack counts are exact. Each gain is a single-process figure, and only the direction is
resolved: on this model, more stacks are faster. At 28000 MB the route reaches 79% of
`ROCKET_MOE=1`'s gain at `pp2048`, and exceeds it at `pp512`.

28000 MB leaves only ~2.8 GB of the headroom the 6 GiB reserve keeps, so it suits a known working
set, not a default. Set the budget explicitly also to make placement the same in every process.

#### Expert GGUF large against RAM

On a model whose expert GGUF is large against RAM, the direction inverts, and raising the budget
can cost the whole gain. On Qwen3-30B-A3B, whose experts hold 29 of its 30.5 B parameters in a
17.28 GiB GGUF, the gain holds on a plateau and then drops. `auto` is already past the plateau:

| `ROCKET_MOE_CACHE_MB` | Stacks placed | Prefill against the experts-on-CPU baseline |
|---|---:|---:|
| 12000 | 38 | 1.032x |
| 18000-21000 | 58-67 | 1.046x |
| `auto` (24425) | 79 | 1.014x |
| 28000 | 90 | 0.999x |

**Conditions:** RK3588 (RK1, 31 GiB), NPU ~816 MHz (the 600 MHz request), Qwen3-30B-A3B,
`-ub 4096` [HW sweep 2026-08-28].

The 3.2% step from the plateau to `auto` is the only difference deep enough to quote as a size.
The 12000 MB and 28000 MB rows are single measurements. At 28000 MB, the 0.999x shows one process
at parity, not that the offload gains nothing at 90 stacks.

`auto` overshoots because the default charge omits the GGUF source of the stacks left on the CPU
(see [Pre-flight](#pre-flight)). Those experts are just as unreclaimable, so the working set
outgrows RAM before the budget shows it.

Size the budget for such a model through the charge factor, not by RAM:
`budget_MB ~ (1 + source_bits_per_weight/8) * (MemTotal − the whole expert GGUF − runtime headroom)`.
On this board that is `1.610 × (31.7 − 17.3 − 1.2 GB)` = 21.3 GB, on the measured plateau. The
formula is consistent with the plateau, not an independent prediction of it: its headroom term is
its one free parameter, chosen to match it.

Which way to move `ROCKET_MOE_CACHE_MB` depends on the expert GGUF's size against RAM. A value
that suits one model or board therefore need not suit another. Where the GGUF is small against RAM,
as gpt-oss-20b's 11.3 GiB is on a 31 GiB board, raise it. Where it is large, as Qwen3-30B-A3B's
17.28 GiB is on the same board, lower it. The target is about `MemTotal` minus the whole expert
GGUF minus the reserve, converted through the charge factor. Both models' experts are most of their weights, so
that share does not decide the direction.

**Not characterized:** the GGUF size between those two at which the direction turns.

#### `ROCKET_MOE_CHARGE_ALL_SOURCE`

`ROCKET_MOE_CHARGE_ALL_SOURCE=1` charges the whole mapped weight buffer once, up front, and leaves
only the codes and scales in each stack's charge. The pre-flight then counts the source of the
stacks it declines. On the 31 GiB board above, it places about 46 Qwen3-30B-A3B stacks against the
default's 79, below the 58-67 plateau and on its conservative side. On gpt-oss-20b it places about
58 of the 63 that `auto` places, derived from the 1.521 charge factor [expected]. That model is
faster with every added stack (see [Raising the budget](#raising-the-budget)).

`test-rocket-placement` checks the contrast at one budget over four stacks in one buffer: the
default admits 3 and the variable 2.

#### Budget across processes

`auto` is resolved once per process, at the first `supports_op` or backend initialization, and
then frozen. It still varies between processes. On one board in one session, the same
DeepSeek-V2-Lite run saw 24.6 GB early and 20.7 GB after several hours of multi-GB allocation
churn. Those budgets admit all 78 of its expert stacks and 70 of them.

A long-lived server that builds a fresh `llama_context` late in its life can therefore place fewer
stacks than at startup, with no warning. The teardown line from `ggml_backend_rocket_moe_stats` is
the only place the difference shows.

To make placement reproducible, pin `ROCKET_MOE_CACHE_MB` to a value the board can hold, or drop
the page cache, as root, before the run:

```sh
echo 3 > /proc/sys/vm/drop_caches
```

### Size minimums

Residency alone does not make the offload faster. The rows an expert receives are
`M_e = n_tokens · n_used / n_expert`, which the architecture's routing sets for a given
micro-batch. gpt-oss routes 4 of 32 experts, so 512 tokens give each expert ~64 rows.
DeepSeek-V2-Lite routes 6 of 64, so the same 512 tokens give ~48. At 100% residency,
DeepSeek-V2-Lite measures 19.09 t/s at `pp512` against 26.04 with the experts on the CPU. That 27%
regression is invisible to a residency check.

The default placement therefore applies two minimums to each expert GEMM:

- The tile granule, `ROCKET_MOE_M_BUCKET` (64 rows). The backend rounds `M_e` up onto a fixed
  ladder of buckets, and computes and reads back the pad rows in full. An expert GEMM smaller than
  the smallest tile is mostly padding.
- The work per dispatch, `ROCKET_MOE_MIN_WORK` (340 million MACs of `M_e · K · N`). Each dispatch
  has a fixed cost in gather, activation quantize, submit, fence and scatter. The work it does is
  `M_e · K · N`, so the point where the offload gains is a threshold on that product.

The bucket ladder steps twice per octave from the granule (64, 96, 128, 192, 256, ...), so padding
is at most ~33% of `M_e` and ~15% on average. Every bucket is a multiple of 4, as
`librocketnpu`'s resident int8 planner requires. The ladder also holds a model to ~15 distinct row
counts. `librocketnpu` keeps scratch for 32 `(M,K,N,group)` shapes per context, and past that it
reallocates the least recently used slot.

`M_e` stands in for the work only while `K · N` is fixed. gpt-oss's expert GEMM is 2880×2880 and
DeepSeek-V2-Lite's is 2048×1408, 2.88x smaller at the same row count.

Default placement against the experts-on-CPU baseline, by `M_e`:

| `M_e` | 64 | 72 | 96 | 128 | 144 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss (2880×2880) | 1.64x | n/a | 1.77x | 1.91x | n/a | 2.05x | 2.08x |
| DeepSeek (2048×1408) | n/a | 0.948x | 1.049x | n/a | 1.225x | 1.313x | n/a |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request) pinned, governor `performance`,
`-p 512` to `-p 2048` on both models, every expert resident and none streamed [HW sweep
2026-08-27].

gpt-oss-20b is 1.64x faster at `M_e` = 64, while DeepSeek-V2-Lite is 5-6% slower at `M_e` = 72. A
row minimum low enough to admit the first admits the second, so no row minimum separates them.
Sorted by work per dispatch, the configurations of both architectures separate with no overlap.

Which minimum decides is per architecture. On gpt-oss-20b it is always the granule, because the
work minimum's threshold there is `M_e` = 41. Adding the work minimum therefore left gpt-oss-20b's
placement unchanged by construction. On DeepSeek-V2-Lite the work minimum decides. It admits from
`M_e` >= 118, past ~1250 tokens in a micro-batch.

The form of the work minimum is derived, and its value is fitted on two architectures. The default
of 340 is the geometric midpoint of the gap between two DeepSeek-V2-Lite shapes: 1.049x at 2.77e8
MACs and 1.22x at 4.15e8. It is a measured boundary between two points, not a bound.

Qwen3-30B-A3B, a third architecture, confirms the work minimum's direction but not its position.
It routes 8 of 128 experts of 2048×768 (`K·N` = 1.573e6), so at `-ub 2048` its `M_e` is 128 and
its work 201 MMAC. The row minimum alone accepts it, and the work minimum declines it. The default
measures 1.000x its own `ROCKET_MOE=0` configuration and ingests nothing. `ROCKET_MOE=1` measures
0.834x, and halving the work per dispatch at matched residency lowers that to 0.661x.

**Not characterized:** an architecture whose work per dispatch falls between 277 and 415 MMAC, the
gap the value 340 is fitted in.

The work minimum sits above 1.00x because of the ingest. The ingest takes ~18 s per
`llama_context` on DeepSeek-V2-Lite and ~21 s on gpt-oss-20b, and the backend runs it only for a
claimed op. An offload at ratio `r` saves `1 − 1/r` of prefill time. The ingest is therefore
repaid after ~7 700 tokens of prefill at 1.05x, ~2 300 at 1.22x, and ~1 100 at gpt-oss-20b's 1.64x
[host-computed]. A shape only a few percent faster than the CPU costs a short session more than it
saves.

#### Micro-batch size

Run a MoE model at `-b 2048 -ub 2048`. The native-quant route dequantizes no expert per
micro-batch, but the dense quantized weights are still dequantized per micro-batch. A smaller
micro-batch also gives each expert proportionally fewer rows, 64 at `-ub 512` against 256 at
`-ub 2048` on gpt-oss-20b. The fixed cost per dispatch, and the padding, stay the same.

### Native-quant experts

The [native-quant experts route](README.md#terms) converts each GGUF-quantized expert to int8 codes
once, and holds the codes in NPU buffers as [resident weights](README.md#terms). Its speed comes
from removing the per-micro-batch dequantization and weight pack, not from int8 arithmetic. The int8
GEMM's int32 output reads back at 8 bytes per element, so it moves more bytes than the equivalent
fp16 GEMM. Quantization here gives residency, and residency gives the speed. The route's speed is
therefore at most fp16's.

The route applies a scale per K-group. The NPU cannot apply a scale per K-block, because `K` is
fully contracted at its output stage. The integer partials already leave the chip at every K-tile
boundary, since no on-device integer [K-accumulation](README.md#terms) is implemented. The backend
keeps each K-tile inside one quant group and applies the group's scale in the readback loop, at no
added cost.

MXFP4 converts to int8 exactly. Its codebook is declared `int8_t` and reaches only ±12, and its E8M0
block scale is an exact power of two. Merging a K-group's native 32-element blocks onto one exponent
is therefore an integer shift, not a requantization. Every other quantized type is dequantized once
and requantized, which works equally well. The cost the route removes is the dequantization per
micro-batch, not the one at load.

Residency is admission-only: a converted expert stays resident, and the backend evicts nothing.
Prefill touches every expert every micro-batch, so the resident total alone decides how much of the
dequantization cost the route removes.

An expert that does not fit the RAM budget or the per-fd [IOVA window](README.md#terms) is
[streamed](README.md#terms). A streamed expert is dequantized to fp16 on every call, which is
correct and slower. `ggml_backend_rocket_moe_stats` reports the resident and streamed counts, and
the backend logs them at teardown.

On a 31 GiB board gpt-oss reaches 99% resident: ~14 GB of int8 experts beside the 11.3 GiB GGUF,
which must stay mapped for CPU decode. A board with less RAM falls short of 99%. A partial result is
correct and slower, as [Partial residency](#partial-residency) describes.

`test-rocket-moe` checks the route against the CPU backend at the primitive level, with
outlier-channel activations. It measures cosine 0.9999 for MXFP4, Q4_K and Q8_0.

### Expert ingest

The backend converts and packs the experts once per `llama_context`, lazily, inside the first
prefill. The expert ingest takes ~21 s on gpt-oss-20b and ~18 s on DeepSeek-V2-Lite:

| Model | Experts | Conversion to int8 | NPU-BO pack | Total |
|---|---:|---:|---:|---:|
| gpt-oss-20b | ~1575 | 11.2-11.8 s | 9.4-9.9 s | ~21 s, 13 ms each |
| DeepSeek-V2-Lite | 4746 | 6.4-6.6 s | 11.7-11.8 s | ~18 s |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), governor `ondemand`, `-b 2048 -ub 2048`
[HW sweep 2026-10-09, RK1].

The pack runs at ~1.3 GB/s, so its time grows with the bytes held resident, not with the expert
count. With `ROCKET_LOG_STDERR=1`, the backend prints an `[moe-int8] ingesting experts` line every
256 experts, so the ingest does not look like a hang. The teardown log reports the total.

`llama-bench`'s prefill figure includes part of the ingest. Its warm-up ingests every expert it
routes to, and an expert first routed in a timed repetition ingests inside that repetition. On
gpt-oss, three timed `pp512` repetitions measured 22.60, 22.51 and 21.27 s [HW sweep 2026-10-09].
The run exercised 1629 experts, where a one-repetition run exercises 1575. Compare MoE figures only
between runs whose ingest costs the same, or read the third and later repetitions.

`llama-bench` builds a fresh `llama_context` for each test row, and the expert cache lives on the
backend instance. It therefore runs the ingest once per row, which adds to its wall-clock time. A
long-running host such as `llama-cli` or `llama-server` runs it once.

### Performance on gpt-oss-20b

On gpt-oss-20b (MXFP4), the default configuration is 1.81x the unrepacked CPU backend at `pp512` and
2.38x at `pp2048`. The default places 63 of 72 expert stacks, 1656 experts, 100% resident and 0
streamed.

| Test | CPU, unrepacked, t/s | NPU, `ROCKET_MOE=0`, t/s | NPU, default, t/s | NPU, `ROCKET_MOE=1`, t/s |
|---|---:|---:|---:|---:|
| `pp512` | 12.17 | 14.08 | 22.0 (1.81x) | 24.81 (2.04x) |
| `pp2048` | 11.80 | 13.62 | 28.1 (2.38x) | 34.01 (2.88x) |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), clock pinned, governor `performance`,
`-b 2048 -ub 2048`, one board and one backend `.so` throughout [HW sweep 2026-08-27]. Ratios are
against the unrepacked CPU backend.

At llama.cpp's default `-ub 512`, the default configuration measures 22.2 t/s at `pp512` (range
21.7-22.7) and 22.5 at `pp2048` (range 22.4-22.8). With the experts on the CPU, the backend measures
14.13 and 13.53, so the route is about 1.6x and 1.7x faster. The route still gains at a small
micro-batch, because it has no per-micro-batch dequantization to repeat. The fp16 expert route
repeats its dequantization every micro-batch. The route gives up ~20% at `pp2048` against
`-ub 2048`, so run it at `-b 2048 -ub 2048`.

The default is not the route's peak. The [pre-flight](README.md#terms) reserves all `n_expert`
experts of a stack, while the lazy admission at ingest charges only the ~82% that the router
exercises. The default therefore places 63 of 72 stacks, and `ROCKET_MOE=1`, which reserves nothing,
places all 72. `ROCKET_MOE=1` is 6-13% faster at `pp512` and 18-21% faster at `pp2048`.

The default gives up that peak for a guarantee: no expert is streamed, and prefill stays at or above
the experts-on-CPU baseline at every budget. On a board with RAM to spare, raising
`ROCKET_MOE_CACHE_MB` recovers the peak and keeps the zero-streamed property. The gain holds on a
model whose expert GGUF is small against RAM, as gpt-oss-20b's is. Where the GGUF is large against
RAM, raising it is a loss, as its row in [MoE experts](#moe-experts) shows. `ROCKET_MOE=1` does not
keep the zero-streamed property.

### Partial residency

The route's gain requires nearly the whole expert stack resident. At 99% resident the figures above
hold. Below that, the loss is a cliff rather than a gradual decline. A streamed expert costs a
weight dequantization that does not depend on its row count. A resident expert costs a GEMM that
shrinks with `M`. The streamed remainder's share of prefill time therefore grows as the prefill
shortens: 12% of `pp2048` and 43% of `pp512`.

> [!NOTE]
> A residency percentage is not a cost percentage. On the build that first held experts resident
> (2026-07-14), 82% resident measured 12.19 t/s at `pp512`, below the 14.11 t/s with the experts
> on the CPU.

At high residency, streaming a few experts costs less than leaving whole stacks of experts on the
CPU. On the 31 GiB board, `ROCKET_MOE=1` at 91% resident was the fastest configuration measured. At
an induced 12 GB budget, `ROCKET_MOE=1` falls to 52% resident and measures 13.63 t/s at `pp512`,
below the 14.08 with the experts on the CPU. The default places 30 fully resident stacks at that
budget and measures 17.02 t/s (1.21x).

**Not characterized:** the residency between 52% and 91% at which `ROCKET_MOE=1` falls below the
experts-on-CPU baseline on the 2026-08-27 build.

The pre-flight makes that difference, and it keeps the default at or above the experts-on-CPU
baseline at every budget. Under the default, the pre-flight reserves a stack before the backend
[claims](README.md#terms) its op, so no streamed remainder arises. A low residency under the default
means the pre-flight declined stacks, and its log line names the budget that stopped it. On this
route that budget is RAM, for the reason
[Worker count and the IOVA window](#worker-count-and-the-iova-window) gives.

Below 80% resident, the backend warns at teardown. The pre-flight's `resident budget reached`
line names the budget that bound it. The 80% threshold is set conservatively, not at a measured
edge.

### Worker count and the IOVA window

`ROCKET_N_THREADS` sets the number of NPU workers, and each worker's file descriptor has its own
IOVA window. The effect of raising it from five depends on the route. The native-quant route gains
1.6-2.2%, and the f16 residency route (`ROCKET_F16_RESIDENT`) loses speed at eight workers. Six is
the recommended value on both routes measured. It gives the native-quant route its whole gain, and
places the most weights on the f16 residency route.

| `ROCKET_N_THREADS` | Native-quant route: gpt-oss-20b `pp2048`, against 5 workers | f16 residency route: Gemma-4-12B F16 `-p 512`, weights resident |
|---|---:|---:|
| 5 (default) | reference | 286 |
| 6 | 1.022x | 293 |
| 7 | 1.016x | 274 |
| 8 | 1.018x | 287 |

**Conditions:** RK3588. gpt-oss-20b with the same 63 stacks admitted at every count. Gemma-4-12B F16
at a near-identical budget, where the IOVA window first declines a weight at five workers
[HW sweep 2026-09-02, RK1].

#### Native-quant route

On the native-quant route, raising the worker count does not place more experts on gpt-oss-20b,
because RAM binds before the IOVA window. The pre-flight charges each stack
`codes + scales + source` against RAM, and `codes` alone against the IOVA budget,
`ROCKET_N_THREADS x 3840 MB`. The window binds first only where `RAM budget / IOVA budget` exceeds
`1 + 4/group + source_bits_per_weight/8`. On `gpt-oss-20b` MXFP4 the ratio at five workers is 1.28,
against a factor of 1.538, so RAM binds.

Raising `ROCKET_N_THREADS` enlarges only the IOVA budget, which moves the ratio further below the
factor. From 5 to 8 workers the window grows from 19200 to 30720 MB, and the admitted set is
byte-identical: 62 stacks and 15693 MB. Both runs report `Bound by RAM` [HW sweep 2026-09-02, RK1].
At five workers the window binds only with `MemAvailable` above 35.7 GB. On this route, read the
pre-flight line, and raise `ROCKET_MOE_CACHE_MB` to place more stacks.

A higher worker count still makes the native-quant route slightly faster, through a mechanism
unrelated to the window. Each count from six to eight measures above 1.00 against five, and no two
are distinguishable, so the gain is a step at six. Each further worker adds oversubscription cost
without adding speed. Set `ROCKET_N_THREADS=6` on this route, and no higher.

#### F16 residency route

On the f16 residency route the IOVA window can bind, and the worker count changes placement. On
Gemma-4-12B F16 at `-p 512`, the route first declines a weight on the window at the default five
workers. Placement is not monotone in the worker count: six places the most weights, and seven
places fewer than five. Try each count rather than the largest, because the value matters more than
the direction.

Eight workers also cost prefill speed on this route. On Gemma-4-12B F16 at `pp2048`, eight workers
measured 0.987x the prefill speed of five [HW sweep 2026-09-01, RK1]. Each worker pins to big core
`i` mod the big-core count, so eight workers put two on every A76. At `pp2048` eight workers also
placed fewer weights, 281 against 286 at five, because the `MemAvailable` minimum binds there, not
the window.

> [!NOTE]
> Which limit binds depends on the prefill length. Read the `admission first declined` line at the
> prefill length you will run.

### DeepSeek-V2-Lite

DeepSeek-V2-Lite combines Multi-head Latent Attention (MLA) with MoE. Its attention is asymmetric,
with DK 192 and DV 128, and it has routed experts. Its routed experts take the native-quant route by
default at long prefill only. Both size minimums in [MoE and MLA models](#moe-and-mla-models) decide
that placement.

The backend claims a `FLASH_ATTN_EXT` whose DK and DV differ. It therefore offloads MLA attention
at `n_kv` >= 1024 whenever the host runs flash attention (`-fa`). `librocketnpu`'s attention test
checks the DeepSeek-V2-Lite shape at cosine >= 0.9995 against an fp64 reference. At the prefill
lengths measured here the offloaded MLA attention is dispatch-bound and gives no prefill gain.

With the routed experts on the CPU, the backend offloads the large MLA projections, the two
always-on shared experts and `lm_head` as ordinary `MUL_MAT` ops. Prefill is 1.18-1.26x the
unrepacked CPU backend. The gain is larger than gpt-oss's dense-only gain, because those projections
are large.

| Configuration | `pp512` | `pp1024` | `pp2048` | `tg64` |
|---|---:|---:|---:|---:|
| CPU, unrepacked | 20.37 | 19.92 | 19.01 | 7.73 |
| NPU, experts on the CPU | 24.04 | 24.85 | 23.87 | 7.77 |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), Q4_K_M (15.71 B parameters, ~2.4 B
active), warm, `-b 2048 -ub 2048`, t/s [HW sweep 2026-07-03].

The NPU holds ~24 t/s at every length, and the gain rises with `M` as the CPU slows. Its wikitext
perplexity differs from the CPU backend's by −0.26%. The absolute perplexity, ~8.2, is meaningful
because this is a base model.

The fp16 expert route lowers prefill to 0.25x to 0.59x the CPU backend at `-ub 2048`: 5.05 t/s at
`pp512`, 7.82 at `pp1024` and 11.30 at `pp2048`. The loss is larger than gpt-oss's. DeepSeek's
faster CPU and the default's existing NPU gain leave more to give up. Its experts are also smaller
(`K` = 2048, `N` = 1408) and more numerous (64), so each expert's dequantization and dispatch are a
larger share of its cost. The default placement never claims that route.

On the native-quant route, DeepSeek-V2-Lite loses at `pp512` and gains at `pp2048`, which is why the
placement has a row minimum:

| Test | CPU, t/s | NPU, `ROCKET_MOE=0`, t/s | NPU, experts on the NPU, t/s |
|---|---:|---:|---:|
| `pp512` | 19.54 | 26.04 | 19.09 (0.73x) |
| `pp2048` | 18.50 | 21.63 | 27.9 (1.29x) |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), governor `performance`
[HW sweep 2026-08-27]. Ratios are against `ROCKET_MOE=0`.

The `pp512` loss is 27% at 100% residency and 0 streamed. It is not the partial-residency loss, and
the residency report does not show it. DeepSeek-V2-Lite routes 6 of 64 experts, so 512 tokens give
each one ~48 rows. That count is below the 64-row granule its GEMM is padded to. With the granule,
the default declines the routed experts at `pp512`, and `pp512` returns to 25.9 t/s, the
experts-on-CPU figure. `pp2048` keeps its gain, so the same model is offloaded at long prefill and
declined at short.

At `-c 2048`, where `M_e` is 192, the routed experts on the NPU change perplexity by −0.12% ± 0.10%
against `ROCKET_MOE=0` [HW sweep 2026-08-27]. The difference is indistinguishable from zero. Against the CPU
backend the default measures −0.42%, which includes the dense offload's −0.30%.

The granule alone also admits shapes where the offload loses or barely gains, as the `M_e` table in
[MoE and MLA models](#moe-and-mla-models) shows. What separates them from gpt-oss's gains is the
work per dispatch, not the row count, because DeepSeek's expert GEMM is 2.88x smaller.
`ROCKET_MOE_MIN_WORK` sets that minimum.

> [!NOTE]
> Do not apply gpt-oss's ratio to another MoE architecture. Both `n_used / n_expert` and the expert
> GEMM's `K · N` decide whether the offload is eligible.

### Evidence

The 99% residency on a 31 GiB board holds for the current resident layout. An earlier layout used
35% of the resident bytes for tile padding, and did not reach it.

The three `llama-bench` ingest repetition figures, 22.60, 22.51 and 21.27 s, are from one run
[HW sweep 2026-10-09].

The gpt-oss-20b default column is the mean of six runs, which spanned 20.5-22.8 at `pp512` and
26.9-28.7 at `pp2048`. Placement was identical in all six. One run measured at the bottom of that
spread, so the column gives the mean. The `ROCKET_MOE=0` control was stable to ±0.05 across the same
rebuilds. The stable control places the spread in the default configuration and not in the board.

The `ROCKET_MOE=1` gains of 6-13% at `pp512` and 18-21% at `pp2048` span two measurements, the
table's 13% and 21% among them. A third, over three rotated passes, measured 19.7% at `pp2048`
[HW sweep 2026-08-29].

The native-quant worker-count ratios are paired within a pass, over three rotated passes, and each
is resolved above 1.00. An A/B with the two configurations run adjacent measured the 5-to-8 step
higher, 1.027x in one two-configuration measurement series. The ladder places a pair half an hour
apart inside a pass, and that spacing is the candidate cause [hypothesis].

On the native-quant route the little-cluster instruction share steps from 0.120 to 0.115 at six
workers and does not move again. Migrations and the A76 IPC keep degrading with every worker added.
The benefit of fanning the expert GEMMs across more fds saturates one worker above the default, and
the oversubscription cost does not.

The worker-count measurements admitted 63 stacks of gpt-oss-20b and the RAM-budget runs 62, both
on 2026-09-02. The difference is the `auto` budget, which is read from `MemAvailable` once per
process. The 63rd stack brings the admitted total to 24529 MB. The 62-stack runs had budgets of
24517-24526 MB, and the 63-stack runs 24597-24622 MB. Within each set the admitted stacks were
identical at every worker count.

On the f16 residency route, each placement count was byte-identical across independent processes,
and the window decline at five workers reproduced. The 0.987x is over three rotated passes, and the
5-to-8 step raised the little-cluster instruction share from 0.141 to 0.150
[HW sweep 2026-09-01, RK1].

The current-build figures come from a 2026-10-09 comparison against the build before it, NPU
configuration only. On that build, gpt-oss-20b measured 22.78-22.99 t/s at `pp512` at an unchanged
ingest. DeepSeek-V2-Lite measured 28.82-29.99 t/s at `pp2048`, and its ingest fell from 38.3-38.8 s
to 18.2-18.4 s inside the same runs. The comparison measured no CPU ratio.

`ROCKET_MOE=1` is the configuration the earlier MoE measurements were taken under. The `pp512`
half of its 6-13% gain over the default is inside that configuration's own run-to-run spread.

With the page cache dropped first, three processes requesting 24679, 26000 and 28000 MB admitted
9539, 10032 and 10810 experts. A repeat of the `auto` configuration came within 13 experts of a
separate probe. The procedure is measured to work and is not explained. `MemAvailable` already
counts most reclaimable page cache, so why dropping it steadies the budget is not established.

The gpt-oss-20b budget figures are one `-r 3` process per budget, so each gain is n=1. A single
process on this board can sit ~10% off the level its own configuration repeats at. The `pp512`
column also sits inside a ±1.4-2.1 t/s within-process spread.

The Qwen3-30B-A3B plateau is pooled over seven processes. `auto` is three (rep sd ±0.6), and the
12000 MB and 28000 MB rows are one process each (28000 MB rep sd ±1.02). The per-rep spread rises
from ±0.00 to ±1.02 as placement grows, so the two single-process rows are where it is widest. The
3.2% gap is the plateau's seven processes against `auto`'s three. It is inside this board's 8-13%
per-process spread, and resolving it at 3 sigma needs 27 rotated passes.

DeepSeek-V2-Lite's ratios in the `M_e` table are means of adjacent pairs: four at `M_e` = 72 and
144, eight at 96, and two at 192. Its offloaded configuration varies ~15% run to run, so a single
pair cannot decide a ratio this close to 1.00. gpt-oss-20b's ratios are single pairs, whose margins
sit an order of magnitude outside that spread.

On the build that first held experts resident, the fp16 dequant route dequantized each gpt-oss-20b
expert in 75 ms per micro-batch. At ~1580 expert dequantizations per prefill, that is ~119 s before
any arithmetic, and it gives the route's 4.59 t/s at `pp512` [HW sweep 2026-07-14].

The DeepSeek-V2-Lite perplexity differences at `-c 2048` are paired per-chunk differences over
8 chunks, at ~816 MHz (the 600 MHz request). The absolute error bar, ±2.5%, would resolve none of
them. Pairing on the same chunks cancels the variance both configurations share.

Each break-even count is the 2026-08 count at that ratio, scaled by the model's ingest ratio. That
ratio compares the current build with the earlier pack in one session: 0.47x on DeepSeek-V2-Lite
and 0.51x on gpt-oss-20b.

## The RK3576

On the RK3576 the backend runs [the RK3576 route](README.md#terms): an int8 W8A8 matmul with a
per-output-column requant. The backend selects it from the detected part
(`rocket_hw_current()`), not from an environment variable. The other sections of this reference
describe the RK3588.

The RK3576 route is the only matmul route on that part. The RK3588 fp16, int8, int4 and bf16
generators refuse there by construction, and the RK3576 entries refuse on the RK3588. The backend
therefore [claims](README.md#terms) a matmul on the RK3576 for the RK3576 route or not at all.

> [!CAUTION]
> A run on the RK3576 must set `ROCKET_INT8=1`. Without it, `supports_op` declines every
> `MUL_MAT`, and a run at the defaults offloads nothing and reports no reason. Run once with
> `ROCKET_LOG_STDERR=1` to confirm the offload.

Build the backend with `-DGGML_ROCKET_NATIVE_FP16=OFF` for the RK3576. The default convert
kernels target the RK3588's `armv8.2-a+fp16` baseline.

### The route

The RK3588's int8 entry returns a raw int32 accumulator, and the host applies both scales. The
RK3576 route instead calls that part's int8-output entry, which applies a per-output-column
requant in the DPU epilogue. The route depends on the scale it supplies to that requant.

The route has four parts, and each is required:

1. **Rotation.** The route rotates `A` and `B` by an orthonormal Hadamard matrix along `K` on
   every call, and `ROCKET_INT8_HADAMARD` does not control it on this part. An unrotated run is
   chaotic, not merely less accurate. A 2% change in the per-column divisor moved one model's
   perplexity by nine orders of magnitude.
2. **Quantization.** `A` is quantized per tensor and `B` per output channel. Per-axis input
   scales cost nothing at this interface and score better on every per-GEMM norm. They are also
   38x worse end to end on one model of two, and `ROCKET_RK3576_AROW` selects them.
3. **Per-column output requant.** Its scale is frozen after a short calibration and divided by a
   safety factor (`ROCKET_RK3576_NCAL`, `ROCKET_RK3576_CALSAFE` and `ROCKET_RK3576_BOOTMARGIN`).
4. **Host dequantization.** The host dequantizes by the scale it asked the entry for, not by the
   gain the entry's integer ramp delivered. The two differ by the entry's `worst_rel_err`.
   Dequantizing by the delivered gain measured no net difference.

Together, the four parts give this perplexity against the per-tensor output scale they replace:

| Configuration | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---:|---:|
| The RK3576 route | 1.008x | 1.020x |
| Per-tensor output scale | 1.11x | 2.26x |

**Conditions:** wikitext-2 perplexity against fp32, simulated on the host with an exactly frozen
column maximum [host-computed 2026-08-10]. SmolLM2-1.7B's figure is at a `ROCKET_RK3576_CALSAFE` of
2.0, and Qwen2.5-1.5B's is the same at 2.0 and 3.0.

On the device, the route measures 1.011x wikitext-2 perplexity on Qwen2.5-1.5B F16. Its
perplexity is 9.8969, against 9.7872 for the same GGUF on the CPU backend [HW sweep 2026-08-11,
H96 MAX M9, llama.cpp b10356]. The figure is over eight chunks, which have an error bar of
+-0.59. The measurement confirms that no part of the device route costs the model an order of magnitude in perplexity.

> [!NOTE]
> The device figure and the host-computed figures are different quantities, so a difference
> between them measures nothing. The host-computed figures are against fp32 with an exactly
> frozen column maximum. The device figure is against the same GGUF on the CPU backend, with the
> default calibration (`ROCKET_RK3576_CALSAFE`=3.0).

To be claimed for the RK3576 route, a matmul must meet these conditions, which are stricter than
the RK3588's:

- `K%32` and `N%32`, where the RK3588 needs `N%16`
- `K >= 64` and `N >= 32`
- A 2D static weight of type F16 or F32, with an F32 activation and output. A quantized or BF16
  weight is declined
- `M >= ROCKET_MIN_M`
- A chunking of `K` that the entry accepts (see [`ROCKET_RK3576_KSPLIT`](#rocket_rk3576_ksplit))

`M` has no alignment constraint on this part, and `M=1` computes. The backend declines
`MUL_MAT_ID` on the RK3576 whatever `ROCKET_MOE` is set to, so a MoE model's expert ops run on the
CPU.

### Performance

On Qwen2.5-1.5B at `pp2048`, the RK3576 route measures 13.36 t/s:

| Configuration | t/s | RK3576 route speedup |
|---|---:|---:|
| The RK3576 route | 13.36 | |
| ggml's `Q8_0` CPU kernel | 10.18 | 1.31x |
| The same F16 GGUF on the CPU backend | 5.64 | 2.37x |

**Conditions:** RK3576 (H96 MAX M9), kernel 7.1.7, `rocket` 1.6.0, stock llama.cpp b10558
[HW sweep 2026-08-23]. `pp2048`, `-ub 512`, four threads pinned to the A72 cores, governor
`performance`, attention on the CPU.

> [!NOTE]
> Quote the two ratios together, because they answer different questions. The 2.37x is the
> speedup a drop-in user sees, because the route consumes an F16 GGUF. The 1.31x is against the
> fastest CPU configuration on the same part.

> [!NOTE]
> A `pp512` figure on the RK3576 measures the calibration, not the converged route. At that
> length every call is a calibration forward. Its 11.78 t/s, and the 2.01x and 1.02x ratios
> computed from it, are calibration measurements [HW sweep 2026-08-11, H96 MAX M9, llama.cpp
> b10356].

The CPU governor setting moves this result by 16%.

Run-to-run spread depends on the prompt length. At `pp2048` it is ±0.02 t/s within a process and
3.5% across processes, so one process gives a usable measurement. At `pp512` the RK3576 route
spreads ±1.8 t/s (15%), against ±0.01 t/s for both CPU configurations, so one process does not.

### Attention on the RK3576

The backend declines `FLASH_ATTN_EXT` on the RK3576, so attention runs on the CPU backend's own
kernel, and no setting is needed. `librocketnpu` has no attention encoder for this part. The
decline has the effect of `ROCKET_FLASH_ATTN=0`, and every RK3576 figure in this section was
measured with attention on the CPU.

> [!NOTE]
> Under a benchmark harness, a bare `llama_decode ... res = -3` means a message is hidden.
> `llama-bench` installs a no-op `ggml` log callback that discards it. Run with
> `ROCKET_LOG_STDERR=1` to see it.

### RK3576 environment variables

The thirteen `ROCKET_RK3576_*` variables apply only on the RK3576 and have no effect elsewhere. Their defaults give the route
this section describes. The first five set the route's accuracy, and two of them,
`ROCKET_RK3576_AROW` and `ROCKET_RK3576_KSPLIT`, change what the model computes:

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_RK3576_NCAL` | 2 | Calibration forwards before the per-column output scale is frozen, each one two device passes (see [RK3576 calibration](#rk3576-calibration)). A value below 1 is replaced by the default, 2 |
| `ROCKET_RK3576_CALSAFE` | 3.0 | Divisor applied to the frozen column maximum, and a value of 0 or below is replaced by the default. The default is 3.0 rather than 2.0 because the calibration estimates the maximum (see [RK3576 calibration](#rk3576-calibration)) |
| `ROCKET_RK3576_BOOTMARGIN` | 1.5 | Slack on the calibration passes' own output scale, and a value below 1 is replaced by the default. The output a calibration forward returns is correct at a scale this much looser, which costs resolution and nothing else |
| `ROCKET_RK3576_AROW` | 0 | `1` selects a per-row activation scale in place of the per-tensor one on every offloaded GEMM, and `2` only on the weights the K-split cut. A value outside 0-2 acts as `1`, not as the default. The dequantization applies the row scale, so it never crosses the ioctl. A per-GEMM norm does not predict its end-to-end effect, so measure a candidate model (see [The route](#the-route)) |
| `ROCKET_RK3576_KSPLIT` | off | Cuts a `K` of 6176 or more, which the library refuses, into chunks on the host, one whole W8A8 GEMM each, and sums the f32 partials. It costs Qwen2.5-1.5B 1.60x wikitext-2 perplexity. Off, the backend declines an FFN down-projection, which runs on the CPU (see [`ROCKET_RK3576_KSPLIT`](#rocket_rk3576_ksplit)) |

The other eight compare configurations or measure cost:

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_RK3576_KCHUNK` | unset | Moves a chunk size to the head of the preference list (2304, 2048, 1536, 1024, 512, 256, 128, 64, 32), so a candidate depth can be validated without a rebuild. Only 2304, 2048 and 1536 were measured clean. Time a model whose chunking uses another size before quoting its figures |
| `ROCKET_RK3576_FORCESPLIT` | off | Cuts a `K` the part would take whole, to separate a defect in the backend's split code from a property of the route. At `K=1536` the rotation is bit-identical either way, so an output change under it comes from the split code. Use it with `ROCKET_RK3576_KCHUNK` |
| `ROCKET_RK3576_WSA` | on | Passes the entry the weight's cached per-column `sum_k \|qB\|`, so the entry does not recompute the O(K·N) pass on every call. The output is bit-identical by construction, so only the cost changes. `=0` measures the cost of the pass |
| `ROCKET_RK3576_WDEV` | on | Packs each converged weight into a device buffer for the entry and frees the host `qB`, at no extra memory. It removes the per-call cube memset, copy and weight-buffer allocation, and `=0` restores them to measure their cost. One failed buffer creation turns it off for the whole context |
| `ROCKET_RK3576_CALSCAN` | 2 | Selects the form of the calibration column scan, to compare their cost: `0` column-outer, `1` row-major with an `int` accumulator, `2` row-major with a byte one. `2` is ~3.5x as fast as `0`, flat in `N`, because a byte lane processes 16 elements where an `int` lane processes 4. A value outside 0-2 is replaced by the default |
| `ROCKET_RK3576_CALROWS` | 0 | Caps the row count of calibration pass one, for measurement, and `0` uses every row. Pass one's estimate is a maximum over `M` accumulator rows, so fewer rows underestimate it. Pass two always runs at full `M`, and its output is what the call returns |
| `ROCKET_RK3576_CALMAP` | unset | Names a file, to which the backend writes a per-column CSV of what the two calibration passes learn (`key,chunk,fwd,M,Kc,N,n,abound,est1,est2,sat,rmsA,rmsB`). The CSV is enough to score a host-side predictor of the column maximum offline. It costs an O(M·Kc + N·Kc) sweep per forward, so keep it out of a timed run |
| `ROCKET_RK3576_PROF_CVT16` | off | With `ROCKET_MM_PROFILE`, also converts each converged call's activation to fp16 after its dequantization, and times that pass into the shape rows' `cvt16` column. It measures the one host pass an fp16 route would make in place of the rotate-and-quantize. It adds that pass to every call, so keep it out of a timed run |

#### RK3576 calibration

Each calibration forward runs two device passes. Pass one estimates the per-column accumulator
maxima from an analytic bound. Pass two refines them at that scale and returns the output.

`ROCKET_RK3576_CALSAFE` defaults to 3.0 because the frozen maximum comes from this calibration,
which has a 0.3% per-column error. At 2.0 that error costs SmolLM2-1.7B 0.018 of perplexity
ratio, against 0.0021 at 3.0. With an exactly frozen maximum, the ratio moves only 0.0017
between the two values.

#### `ROCKET_RK3576_KSPLIT`

The split is fast and exact. Its perplexity cost comes from what it reaches: `ffn_down`, an input
that the route's per-tensor activation scale does not fit. More chunks raise the cost. Cut ten
ways, the same projection scores a perplexity of 18.22, against 15.84 cut four ways.

#### `ROCKET_MM_PROFILE` on the RK3576

`ROCKET_MM_PROFILE` also reports this route, through a third accumulator. The report has the
library's buckets (`sumabs / packA / packB / coeff / gen / alloc / stamp / submit / wait / read`)
and the terms outside the entry (`valloc / quant+rot / calibrate / entry / dequant`).

The same line is followed by one row per GEMM shape `(M, Kc, N)`, over converged calls only. Each
row gives `quant+rot`, `entry`, `dequant` and `cvt16` per call. Each row also counts the calls
whose activation was the previous call's, which rotate and quantize the same activation again. On
a transformer those are `k` and `v` after `q`, and `up` after `gate`.

Both reports sum to the time of the offloaded matmuls, not to prefill time. Attention, the norms,
rope and the scheduler are in neither.

### Evidence

The `pp2048` performance table interleaves three processes per configuration in one boot. The
per-pair ratios behind the 1.31x are 1.30, 1.34 and 1.30.

A `pp512 -r 1` run reports 332 of 332 calls as calibration forwards.

## Quantization and prefill throughput

Quantization saves RAM and lets a larger model fit, and it does not raise prefill speed at this
operating point. Prefill is limited by DMA and dispatch rather than by the MAC array. While it is,
int8's 2x and int4's 4x MAC rates do not show in prefill.

With the whole-tile program, the resident matmul runs at ~460 GOP/s at every precision: fp16 461,
int8 386 and int4 413 GOP/s. The NPU then runs at ~15% of its MAC peak. In a model, resident int8
prefill is 0.60x fp16, because its int32 readback is not [K-accumulated](README.md#terms) on the
NPU.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), [resident weights](README.md#terms),
the whole-tile program that `ROCKET_MM_STREAM=0` selects.

CBUF weight streaming (`ROCKET_MM_STREAM`), the fp16 default, cuts the whole-tile program's task
count. It raises `pp512` 1.069-1.167x over `ROCKET_MM_STREAM=0` on four GGUFs [HW sweep
2026-10-21].

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), three cores, resident weights,
governor `performance`.

The fp16 prefill figures are the product of operating-point gains over the 200 MHz boot clock,
measured step by step on Gemma-4-12B:

| Step | Gemma-4-12B `pp2048`, t/s | Gain |
|---|---:|---:|
| 200 MHz boot clock | 7.98 | |
| NPU ~816 MHz (the 600 MHz request) | 11.40 | 1.43x |
| fp16 K-accumulation on the NPU (`ROCKET_KACC`) | 13.38 | +19% |
| [CBUF](README.md#terms) DATA_REUSE | ~14.5 | +7% |
| Resident weights | ~15.1 | +6% |

**Conditions:** RK3588, Gemma-4-12B F16, `pp2048`, each step added to the step before it, earlier
builds of the backend.

**Not characterized:** the steps on the current build.

K-accumulation and CBUF operand reuse are on by default. Residency is the default only for an F16 weight
with `K` at most 2048, and opt-in above it. None of the steps is quantization.

The default route is fp16 with K-accumulation, CBUF operand reuse and CBUF weight streaming. Use int8
and int4 when a model must be quantized to fit memory.

On the 2026-09-02 build, the default fp16 route measures 19.96 t/s on
Gemma-4-12B F16 at `pp2048` [HW sweep 2026-09-02, llama.cpp b10558].

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), governor `performance`, a build
without CBUF weight streaming.

**Not characterized:** Gemma-4-12B with CBUF weight streaming.

Across models, the F16 prefill speedup over the CPU backend grows with model size and then levels
off. It runs from 1.44x on a 0.8B model to ~3.6x on 9-12 B models. It is measured on Qwen3.5 and
Qwen3.6, including the 27B hybrid-DeltaNet, and on Gemma-4-12B, with perplexity agreeing with the
CPU backend's on each.

**Conditions:** RK3588. The Qwen3.5 figures are best-case runs [HW sweep 2026-06-27/28], and
Gemma-4-12B's 3.6x is at `pp512` [HW sweep 2026-07-02].

### Micro-batch size for quantized prefill

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), `pp2048` unless a figure names another
test.

A quantized GGUF is dequantized to fp16 on every micro-batch, so the llama.cpp default `-ub 512`
costs throughput. `-b 2048 -ub 2048` measures 0.91-1.53x the default: 1.18-1.53x on dense models
above 8 B, and 0.94-1.13x under 4 B. A MoE model has its own row in
[Recommended configurations](#recommended-configurations). Two models of eleven are slower with
the flag, so measure it per model. A 9B `Q4_K` goes from 19.1 to 27.3 t/s.

If the fp16 form of the model fits RAM, wholly or partly, set `ROCKET_QUANT_RESIDENT=auto` at the
default `-ub`, without `-b 2048 -ub 2048`. On seven models of 1.8-11.9 B it measures 1.35-1.75x
the streamed default, against 1.00-1.66x combined with `-b 2048 -ub 2048` [HW sweep
2026-08-29/31]. All seven are faster without the flag, and six of the seven were 100% resident.

With `ROCKET_QUANT_RESIDENT=auto`, the default `-ub` is the faster choice at every model size
measured, and on a partly resident model too. On the one such model, the default `-ub` makes 9-13
percentage points more of its weights resident. `-b 2048 -ub 2048` uses that RAM for compute buffers first, so the reserve stops
placement earlier.

Under 4 B, `-ub 2048` also has a 14-21% cost unrelated to dequantization, which its
dequantization gain hides. Above 4 B, the default `-ub` gains because residency also removes the
per-call pack and upload, which the default `-ub` incurs four times as often.

The per-micro-batch dequantization is multi-threaded by default, through
`ROCKET_DEQUANT_THREADS`. The default is the A76 big-core count, or the core count capped at 8
where no big cluster is found, and `1` forces serial dequantization. The weight rows are
independent, so threading gives +33% at `-ub 512` and +19% at `-ub 2048` over serial
dequantization on a 9B `Q4_K` [HW sweep]. It is bit-exact and needs no extra RAM.

With threaded dequantization, the quantization type does not affect NPU throughput, and `Q4_K`,
`IQ4_XS` and `Q8_0` measure about the same. A quantized GGUF approaches F16 as `M` grows, from
~0.64x F16 at `pp512` to ~0.85x at `pp2048`. The F16 weight stays resident and needs no
dequantization.

### Quantized weight residency

`ROCKET_QUANT_RESIDENT=1` removes most of the gap between a quantized GGUF and F16. It dequantizes and packs each weight into
resident fp16 NPU buffers once, so prefill has neither the per-micro-batch dequantization nor the
per-call `packB`. It uses the F16 resident route, with bit-identical perplexity. Quantized prefill
then measures 0.95-1.00x the same model's F16 GGUF with resident weights:

| Model | `Q4_K_M` streamed, t/s | `Q4_K_M` resident, t/s | F16 resident, t/s | `Q4_K_M` resident against F16 |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 39.20 | 57.72 | 60.54 | 0.95x |
| Ministral-3-3B | 34.60 | 50.00 | 52.04 | 0.96x |
| Phi-4-mini | 35.94 | 54.18 | 54.40 | 1.00x |

**Conditions:** RK3588, `pp2048` at the default `-ub`, llama.cpp b10558, NPU ~816 MHz (the 600 MHz
request), governor `performance`, every weight resident [HW sweep 2026-08-29/31]. Residency is
`ROCKET_QUANT_RESIDENT=auto` for `Q4_K_M` and `ROCKET_F16_RESIDENT=auto` for F16.

The cost is the full fp16 footprint in resident buffers, which cancels the RAM that quantization
saved. It is therefore opt-in, for a board with RAM to spare. A weight over the `ROCKET_CACHE_MB`
budget, or past a full NPU [IOVA window](README.md#terms), is [streamed](README.md#terms)
instead.

On a model larger than the default 2 GB budget, use `ROCKET_QUANT_RESIDENT=auto`, which sizes its
budget from free RAM and makes the whole model resident. A blanket `1` at the default budget makes
only part of the model resident, and can be slower than the streamed default.

On Qwen3.5-9B `Q4_K_M`, `auto` at the default `-ub` measures 33.45 t/s, 1.752x the streamed
default:

| Configuration | t/s | Speedup over the streamed default |
|---|---:|---:|
| Streamed, default `-ub` | 19.14 | |
| Streamed, `-b 2048 -ub 2048` | 27.26 | 1.424x |
| `auto` with `-b 2048 -ub 2048` | 31.78 | 1.661x |
| `auto`, default `-ub` | 33.45 | 1.752x |

**Conditions:** RK3588, Qwen3.5-9B `Q4_K_M`, `pp2048`, NPU ~816 MHz (the 600 MHz request),
governor `performance`, every resident configuration 100% resident [HW sweep 2026-08-29/31]. The
9B's ~18 GB fp16 form fits the board's 31 GB RAM and the IOVA window.

A quantized prefill shorter than `ROCKET_MIN_M_QUANT` stays on the CPU. The threshold sits near
the measured crossover of ~360 rows.

### int8 and prefill speed

int8 does not speed this backend's prefill. The reason is the datatype's role on this matmul, not
a hardware mode the backend is missing. int8 can help prefill in two ways only:

- It cuts operand bandwidth. A W8A8 scheme halves both operands' bytes, and this route quantizes
  the weights and keeps fp16 activations.
- It puts more work in each submit, which reduces the dispatch cost.

Neither raises the MAC limit, and the matmul is limited by DMA and dispatch rather than by the MAC
array. Throughput therefore rises with fewer, larger, batched submits, not with a narrower
datatype.

Integer K-accumulation on the NPU would let int8 keep its narrow weights and avoid the int32
readback. The DPU's elementwise stage adds int32 exactly, with its precision fields all set to
int32 [HW sweep 2026-09-27]. The backend does not implement it, and it does not raise prefill
speed here. Prefill is limited by dispatch and DMA, and with the whole-tile program fp16, int8 and
int4 tie at ~460 GOP/s.

### Evidence

The `ROCKET_QUANT_RESIDENT=auto` comparison over seven models is three rotated passes per
configuration. The Qwen3.5-9B `Q4_K_M` table is three rotated passes, with the ratios paired
within a pass. Its `auto` at the default `-ub` row is a second session [HW sweep 2026-08-31], whose
streamed default measured 19.10 t/s.

The CBUF weight streaming figures are three rotated passes per configuration.

The `ROCKET_DEQUANT_THREADS` gains compare the threaded and serial configurations in one session.

The residency table against F16 is three rotated passes per configuration, and each figure is the
mean of the three. The `Q4_K_M` columns are one series [HW sweep 2026-08-31]. The F16 column is
from a second series on the same llama.cpp build [HW sweep 2026-08-29], which also measured the
streamed `Q4_K_M` defaults. Those agree with the first series' to within 0.7%.

An earlier build measured residency in one session on Qwen3.5 [HW sweep 2026-06-28], at `pp2048`:

| Configuration | t/s |
|---|---|
| 0.8B `Q4_K` resident | 89, equal to F16's 89 |
| 0.8B `Q4_K` streamed, `-ub 2048` | 81 |
| 0.8B `Q4_K` streamed, `-ub 512` | 51 |
| 9B `Q4_K` resident | 24.6, ~0.92x F16 |
| 9B `Q4_K` streamed, `-ub 2048`, serial dequantization | 15.8 |

The 0.8B resident row's `-ub` was not recorded. On the 9B, residency was 1.56x, and the per-call
`packB` fell from 12440 to 889 ms.

On an earlier build, a blanket `ROCKET_QUANT_RESIDENT=1` at the default budget measured 19.3 t/s
on the 9B `Q4_K`, against 22.1 t/s streamed, at `pp2048` and `-ub 2048`.

## Environment variables

The backend and `librocketnpu` read the `ROCKET_*` environment variables below, and each row marks
a variable that `librocketnpu` reads. `sudo` drops the environment, so run under `sudo -E`.

A variable the backend reads that is set but empty is treated as unset, not as `0`, and takes its
default. A wrapper that forwards an unset `$VAR` produces that form, such as `ROCKET_MIN_M=`.

> [!NOTE]
> `librocketnpu` reads an empty `ROCKET_KACC`, `ROCKET_REUSE`, `ROCKET_MM_STREAM` or
> `ROCKET_MM_ASYM` as `0`, which turns that option off. An empty `ROCKET_WAIT_MS` takes its
> default. `ROCKET_NO_KACC` set to any value, empty included, turns K-accumulation off.

The presence-only diagnostics are the exception to the empty-value rule. Each is on whenever it is
set, so an empty value turns it on, and none of them changes a result:

- `ROCKET_DEBUG`
- `ROCKET_DEBUG_GRAPH`
- `ROCKET_MM_PROFILE`
- `ROCKET_AB`
- `ROCKET_DUMP`
- `ROCKET_FA_TIMING`

### Matmul offload

| Variable | Default | Effect |
|---|---|---|
| `GGML_BACKEND_PATH` | unset | The path from which the host's ggml loads `libggml-rocket.so`. Set it, as an absolute path, to load the backend |
| `ROCKET_KACC` | on | (read by `librocketnpu`) [K-accumulation](README.md#terms): the NPU sums the fp16 `K` partials, for +19% prefill over `=0`. `=0`, or `ROCKET_NO_KACC` set to any value, selects the byte-exact reference [route](README.md#terms), which sums the partials on the host in fp32. Leave it on, and set `0` where a byte-exact result is required |
| `ROCKET_REUSE` | auto with K-accumulation on, `0` with it off | (read by `librocketnpu`) Forces the [CBUF](README.md#terms) operand reuse mode: `0` off, `1` weight reuse, `2` data reuse. Auto picks, per shape, the operand with the longer reuse run, and data reuse on a tie. Data reuse gives +7% prefill over no reuse |
| `ROCKET_MM_STREAM` | on | (read by `librocketnpu`) CBUF weight streaming: the fp16 planner streams each weight slice through the CBUF, so one NPU task covers up to 4096 output columns. A layer then needs far fewer tasks, and `=0`, or any `ROCKET_MM_MT`, `ROCKET_MM_NT` or `ROCKET_MM_KT` override, selects the whole-tile plan. [`ROCKET_MM_STREAM`](#rocket_mm_stream) has the measured gain |
| `ROCKET_MM_ASYM` | on | (read by `librocketnpu`) Asymmetric tiling in the whole-tile plan, for a shape that tiles both `N` and `K`. The tile's `N` halves to 128, and its `K` grows from 384 to 512. `=0` forces the symmetric plan, whose fp16 output can differ because the `K` partials round at different points |
| `ROCKET_N_THREADS` | 5 | The number of NPU workers that an offloaded op is spread across, 1 to 8, and a value outside that range is ignored. Each worker has its own [IOVA window](README.md#terms), and six is the value to take on both routes measured, MoE and f16 residency ([MoE and MLA models](#moe-and-mla-models)). The figures in this document were measured at 5, and the `admission first declined` line names the limit that stops residency at the shape you will run |
| `ROCKET_MIN_M` | 32 | The fewest rows (`M`) an unquantized weight's `MUL_MAT` needs to offload, and a value below 4 is raised to 4. Below the crossover of 21-25 rows, dispatch and the per-call weight pack make the offload slower than the CPU. Set 16 for a CTC or transducer encoder on short utterances ([`ROCKET_MIN_M`](#rocket_min_m)) |
| `ROCKET_MIN_M_QUANT` | 512 | The fewest rows a quantized weight's `MUL_MAT` needs to offload, never below `ROCKET_MIN_M`. Its per-micro-batch dequantize needs more rows than F16 to amortize, and the measured crossover is ~360 rows on 9B and 27B `Q4_K`. A shorter quantized prefill stays on the CPU, where it is faster |
| `ROCKET_DEQUANT_THREADS` | auto | The number of CPU threads that dequantize a quantized weight on each micro-batch. Auto, which `0` also selects, is the A76 big-core count, or the core count capped at 8 where no big cluster is found. `1` forces one thread, and a larger value sets the count ([Quantization and prefill throughput](#quantization-and-prefill-throughput)) |
| `ROCKET_NO_FUSE` | off | `1` turns off QKV and gate-up fusion, so each matmul is routed on its own, for an A/B comparison. Fusion also turns off under `ROCKET_NO_STREAM` and `ROCKET_FORCE_PREPACK`, and under `ROCKET_INT8` and `ROCKET_INT4`, because the fused route is fp16 only |
| `ROCKET_NO_STREAM` | off | `1` turns off the persistent context that the per-call `mt` route uses for [streamed weights](README.md#terms), so each call opens and releases its own NPU file descriptors and buffers. It also turns fusion off, which needs that context |
| `ROCKET_WAIT_MS` | 8000 | (read by `librocketnpu`) The time in ms to wait for a job's output fence, and a value below 1 takes the default. The default covers a multi-second CPU-side stall, such as a large model faulting in under low RAM |

#### `ROCKET_MIN_M`

`ROCKET_MIN_M` sets the fewest rows an unquantized weight's matmul needs to offload. Below the
crossover, the per-call dispatch and weight pack cost more than the NPU's per-row speed saves, and
the offload is slower than the CPU. A weight becomes [resident](README.md#terms) only at `M` >= 256
(`max_tile`), so below that it is packed again on every call.

The crossover is 21-25 rows on all three models measured. The default of 32 is above every
measured crossover, so no model measured is slower than on the CPU:

| F16 model | pp16 | pp32 | pp64 | pp128 |
|---|---:|---:|---:|---:|
| 0.8B | 0.75 | 1.20 | 1.71 | 1.96 |
| 3B | 0.78 | 1.40 | 2.51 | 3.68 |
| 8B | 0.77 | 1.46 | 2.75 | 3.90 |

**Conditions:** RK3588, F16, NPU prefill t/s over the faster of 4 and 8 CPU threads, CPU governor
`performance` [HW sweep 2026-10, llama.cpp a7fb71f].

The crossover barely depends on the model, because the weight pack the offload adds and the
compute it saves both scale with `K`·`N`. The dispatch cost does not scale with `K`·`N`, and it
causes the small drift between models.

whisper.cpp runs the same at 32 as at 128. whisper-small F16 measures 0.995-1.006x its speed at
`ROCKET_MIN_M=128`, with a byte-identical transcript.

> [!NOTE]
> `ROCKET_MIN_M` also keeps batched decode on the CPU. whisper.cpp's default beam search presents
> `M`=5 per step. At `ROCKET_MIN_M=4` those steps offload and run 2.3x slower, a 1.40x net loss
> end to end.

For a CTC or transducer encoder on short utterances, set 16 in that process. Where another
process shares its cores, also set `OMP_WAIT_POLICY=PASSIVE`
([Recommended configurations](#recommended-configurations)). Do not set it to 256. Below 256 rows
a llama.cpp model streams its weights and still offloads faster than the CPU, 3.90x at pp128 on
the 8B.

#### `ROCKET_MM_STREAM`

CBUF weight streaming is bit-exact, and perplexity is unchanged. Prefill speedup over
`ROCKET_MM_STREAM=0` at pp512:

| Model | GGUF | Speedup |
|---|---|---:|
| Llama-3.2-3B | F16 | 1.134-1.143x |
| Llama-3.2-3B | `Q4_K_M` under `ROCKET_QUANT_RESIDENT=auto` | 1.163-1.167x |
| Ministral-3-3B | F16 | 1.120-1.153x |
| Qwen3.5-9B | `Q4_K_M`, quant-resident | 1.069-1.110x |

**Conditions:** RK3588, NPU 928 MHz (the 900 MHz request), three cores, resident weights, CPU
governor `performance` [HW sweep 2026-10-21].

#### Evidence

The `ROCKET_MM_STREAM` speedups are three rotated passes per configuration.

On the f16 residency route, the prefill-time difference between 5 and 6 workers
(`ROCKET_N_THREADS`) is below the measurement's resolution.

### Weight residency

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_F16_RESIDENT` | off | Takes the same values as `ROCKET_QUANT_RESIDENT`, and `auto` makes every static F16 weight resident at any `K` within a budget sized from free RAM. Off, only weights with `K` <= 2048 are resident, which covers Whisper's encoder. Set `auto` for repeated prefills of an F16 model that fits about twice in RAM ([Resident weights](#resident-weights)) |
| `ROCKET_QUANT_RESIDENT` | off | Dequantizes each quantized weight to fp16 once and holds it resident, which lifts prefill to F16 speed at the cost of the full fp16 footprint. `auto` sizes the budget from free RAM less `ROCKET_QUANT_RESIDENT_RESERVE_MB`, `<N>` sets an `N`-MB budget, and `1` uses the `ROCKET_CACHE_MB` budget. Set `auto` at the default `-ub` where the fp16 image fits RAM, wholly or partly ([Quantization and prefill throughput](#quantization-and-prefill-throughput)) |
| `ROCKET_QUANT_RESIDENT_RESERVE_MB` | `max(6 GiB, 30 % of RAM)` | The RAM, in MB, that the `auto` budgets of `ROCKET_QUANT_RESIDENT` and `ROCKET_F16_RESIDENT` hold back from `MemAvailable`. It is also the `MemAvailable` minimum at which their admission stops, and it replaces the MoE expert budget's 6 GiB reserve ([MoE and MLA models](#moe-and-mla-models)). Set it where the KV cache and activations need a different headroom |
| `ROCKET_CACHE_MB` | 2048 | The resident fp16 weight budget in MB, `0` for unlimited. `ROCKET_QUANT_RESIDENT` and `ROCKET_F16_RESIDENT` set it from their `auto` or `<N>` value, `ROCKET_F16_RESIDENT`'s where both are on. A valid `ROCKET_CACHE_MB` overrides both, and an empty or malformed one is ignored |
| `ROCKET_DEQUANT_CACHE_MB` | off | A host cache, in MB, of each quantized weight's fp16 form on the streamed route, `0` for unlimited, so a weight is dequantized once. It is a diagnostic that isolates the dequantize cost, and `ROCKET_QUANT_RESIDENT` gives more for the same fp16 footprint by removing the per-call pack and upload. A weight over the budget is dequantized per call, correctly, and the teardown `[dq-cache]` line reports the weights held, skipped and still streamed |
| `ROCKET_FORCE_PREPACK` | off | `1` makes every F16 weight resident at any `K`, each on its own, within a 2 GB budget unless `ROCKET_CACHE_MB` sets one. It is the diagnostic force and turns QKV and gate-up fusion off, and `ROCKET_F16_RESIDENT` is the setting for a production run |
| `ROCKET_NO_PREPACK` | off | `1` turns off resident F16 weights on the single-matmul route, the `K` <= 2048 default included, so those weights are streamed. Set it to compare against resident weights |
| `ROCKET_PREPACK_MADVISE` | off | `1` releases each F16 weight's GGUF pages with `madvise` once the weight is resident, which frees RAM. It breaks CPU decode, so set it only for a prefill-only run |

### Precision

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_INT8` | off | `1` selects the W8A8 int8 route, which on the RK3588 is slower than fp16 with K-accumulation ([Models and precisions](#models-and-precisions)). On the RK3576 it is required, because [the RK3576 route](README.md#terms) is the only matmul route there and `supports_op` [declines](README.md#terms) every `MUL_MAT` without it ([The RK3576](#the-rk3576)) |
| `ROCKET_INT8_HADAMARD` | off | `1` adds a Hadamard rotation to the RK3588 int8 route, and with it the route's output is coherent. The RK3576 route always rotates and ignores it |
| `ROCKET_INT8_RESIDENT` | off | With `ROCKET_INT8=1` on the RK3588, `1` holds the int8 weights resident on the NPU, within `ROCKET_INT8_CACHE_MB` |
| `ROCKET_INT8_CACHE_MB` | 4096 | The resident int8 weight budget in MB, `0` for unlimited, counted against the weights' actual NPU buffer size. A weight over the budget takes the fp16 route |
| `ROCKET_INT4` | off | `1` selects the native W4A4 int4 route from an F16 GGUF, with group-wise scales and a Hadamard rotation on by default. Its output is char-identical to fp16, and it saves memory without raising prefill speed ([Models and precisions](#models-and-precisions)) |
| `ROCKET_INT4_RESIDENT` | off | With `ROCKET_INT4=1`, `1` holds the int4 weights resident on the NPU at ~¼ of the fp16 footprint, with no per-call weight pack. It needs group-wise scales (`ROCKET_INT4_GROUP>0`, the default). A per-channel weight, or one over the budget or past a full IOVA window, takes the per-call int4 route |
| `ROCKET_INT4_GROUP` | 128 | The number of `K` elements that share one int4 scale, which must divide `K` and be a multiple of 32, and `0` selects per-channel scales. A smaller group is more accurate and reads back more partials |
| `ROCKET_INT4_HADAMARD` | 1 | The Hadamard rotation of the int4 route, which raises per-matmul cosine from ~0.79 to ~0.98. Leave it on, and `0` turns it off |
| `ROCKET_INT4_CACHE_MB` | 4096 | The int4 weight budget in MB, `0` for unlimited, which caps both the per-call route's host cache and the resident total. A weight over the budget falls back. A full 12B model's resident int4 weights are ~6 GB, so raise it for full coverage |
| `ROCKET_BF16` | off | `1` selects the exact bf16 route, which writes its fp32 result straight to `dst`. The bf16 format has fp32's exponent range, so the activations need no per-row scale and go through unscaled. Without it, a bf16 weight converts to fp16 and takes the shared fp16 route, at fp16 accuracy and a similar speed |
| `ROCKET_BF16_CACHE_MB` | off | A host cache, in MB, of the `ROCKET_BF16` route's fp32 weights, `0` for unlimited, and a weight over the budget is converted per call. It is off by default because it holds fp32, twice the bf16 weight's size, so caching a 24 GB bf16 model whole takes 48 GB. It removes the per-micro-batch bf16-to-fp32 conversion, the route's largest host cost, and the `dequant` bucket of `ROCKET_MM_PROFILE` shows whether that is worth the RAM |

### MoE experts

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_MOE` | unset | Selects which MoE expert ops (`MUL_MAT_ID`) the backend [claims](README.md#terms): unset claims a quantized stack the [pre-flight](README.md#terms) can reserve and that meets both size minimums, `1` claims every one it can compute, and `0` none. The default prefills gpt-oss-20b 1.9x faster than the CPU backend with llama.cpp's weight repack on, at `pp2048` [HW sweep 2026-09-29]. Set `1` or `0` only to compare configurations (see [MoE and MLA models](#moe-and-mla-models)). |
| `ROCKET_MOE_MIN_TOKENS` | `512` | The smallest micro-batch, in tokens, at which the backend claims a `MUL_MAT_ID` op, in any `ROCKET_MOE` state. It keeps short prefills and decode on the CPU, and is never below `ROCKET_MIN_M`. Each expert's GEMM receives about `n_tokens · n_expert_used / n_expert` rows. |
| `ROCKET_MOE_NATIVE` | on | Selects the [native-quant](README.md#terms) [route](README.md#terms): a GGUF-quantized expert is converted once into int8 codes held in NPU buffers, and each call quantizes only the activation. `0` selects the fp16 dequant route, the comparison baseline, which the default placement never claims, so `0` takes effect only with `ROCKET_MOE=1`. An F16 expert always takes the fp16 route. |
| `ROCKET_MOE_CACHE_MB` | `auto` | The resident expert budget in MB, which the pre-flight and the ingest both charge, with `0` for unlimited. `auto` is `MemAvailable` minus 6 GiB, resolved once per process. Raise it where the expert GGUF is small against RAM, and lower it where the GGUF is large (see [MoE and MLA models](#moe-and-mla-models)). |
| `ROCKET_MOE_CHARGE_ALL_SOURCE` | off | `1` charges the whole mapped GGUF weight buffer once, so the pre-flight also counts the source of the stacks it leaves on the CPU. It places fewer stacks, about 46 against the default's 79 on Qwen3-30B-A3B on a 31 GiB board. Its effect differs by model class (see [MoE and MLA models](#moe-and-mla-models)). |
| `ROCKET_MOE_GROUP` | `0` | The K-group the native-quant route quantizes on. `0` picks the largest divisor of `K` that is a multiple of 32 and fits the [CBUF](README.md#terms) as one K-tile, which minimizes readback: 576 for gpt-oss-20b's `K`=2880, in five K-tiles. Set a smaller group only to trade speed for accuracy: the route is readback-bound and readback scales as `K/group`, so a smaller group is more accurate and proportionally slower. |
| `ROCKET_MOE_M_BUCKET` | `64` | The smallest `M_e` the default placement claims, rounded up to a power of two. Each expert's row count is rounded up onto buckets that step twice per octave from it (64, 96, 128, 192, 256, ...), each a multiple of 4. Raise it only to trade more padding for fewer distinct row counts. |
| `ROCKET_MOE_MIN_WORK` | `340` | The least work one expert dispatch must do for the default placement to claim the op, in millions of MACs of `M_e · K · N`. `0` disables it, and the value is fitted on two architectures. Set `240` to also claim the shape just below it, which repays its ingest only after ~7 700 tokens of prefill (see [MoE and MLA models](#moe-and-mla-models)). |
| `ROCKET_MOE_COSINE` | off | Reports, for each offloaded `MUL_MAT_ID`, the cosine between the NPU's output and a CPU reference. A diagnostic. |

### Attention

The backend offloads `FLASH_ATTN_EXT`, prefill attention, to the NPU by default. These variables
choose which attention ops it [claims](README.md#terms) and how it computes them.

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_FLASH_ATTN` | on | Offloads `FLASH_ATTN_EXT` to the NPU. `0` disables the offload. See [`ROCKET_FLASH_ATTN`](#rocket_flash_attn) for what it computes, which ops it declines and its speed. |
| `ROCKET_FA_CHAIN` | on | `librocketnpu` batches each worker's per-head QK submits into one NPU job, and its AV submits into a second, through a resident batched-matmul context. The fewer submits move the context length at which NPU attention matches the CPU's down to ~2K. `0` forces the per-head [route](README.md#terms), and the output is bit-identical either way. |
| `ROCKET_FA_ROWMAJOR` | on | Read by `librocketnpu`: the NPU writes the attention scores and reads the softmax's output as plain rows, so the host converts neither to or from the tile layout. `0` restores the tiled handoff. See [`ROCKET_FA_ROWMAJOR`](#rocket_fa_rowmajor) for its speed and its one difference in output. |
| `ROCKET_FLASH_ATTN_MIN_KV` | 1024 | The minimum `n_kv`, the context length, at which the backend offloads an attention op. The default, 1024, is about Gemma-4's sliding-window length, so its windowed local layers offload while shorter prompts stay on the CPU. The threshold is on `n_kv` because every micro-batch has `n_tokens` ~512 at the default `-ub`, whatever the context. |
| `ROCKET_FLASH_ATTN_MIN_T` | 16 | The minimum `n_tokens` at which the backend offloads an attention op, which keeps single-token decode on the CPU. |
| `ROCKET_FLASH_ATTN_UNMASKED` | off | Claims `FLASH_ATTN_EXT` ops that have no mask: an encoder's self-attention, as in whisper.cpp's audio encoder and the mtmd vision encoders. It is off because the one encoder measured was slower with it. See [`ROCKET_FLASH_ATTN_UNMASKED`](#rocket_flash_attn_unmasked). |
| `ROCKET_ATTN_HOST_SOFTMAX` | host | Read by `librocketnpu`: a nonzero value runs the attention softmax on the host, and `0` runs it on the NPU. Unset, flash attention runs it on the host, because its scores are already there for the additive mask. |
| `ROCKET_FA_QSCALE` | 1 | Folds the op's scale into Q as the handler converts Q to fp16, so the fp16 scores hold `scale*q.k` and overflow only past `65504/scale`. `0` restores the unscaled scores. See [`ROCKET_FA_QSCALE`](#rocket_fa_qscale). |
| `ROCKET_FA_THREADS` | 1 | Sets the number of host workers for the attention handler's gather and scatter, whose time adds directly to prefill time. Set it to `4` for the measured gain, since more workers add almost nothing. See [`ROCKET_FA_THREADS`](#rocket_fa_threads). |
| `ROCKET_FA_TIMING` | off | Prints the attention handler's host split at exit: gather, NPU compute and scatter, in ms. The gather is host code that `ROCKET_MM_PROFILE` does not see. It is a diagnostic, with near-zero cost while off. |
| `ROCKET_FA_CHECKSUM` | off | `1` prints at exit an FNV-1a hash of every byte the attention handler writes, over every offloaded op. `2` adds a hash per op, and `3` adds the hashes of each op's inputs. Read [`ROCKET_FA_CHECKSUM`](#rocket_fa_checksum) before comparing two runs. |
| `ROCKET_FA_DUMP_OP` | unset | With `ROCKET_FA_CHECKSUM` on, `ROCKET_FA_DUMP_OP=N` writes op `N`'s four input tiles and its fp16 output to raw files, for a value-by-value comparison of two runs. |
| `ROCKET_FA_DUMP_DIR` | `/tmp` | The directory for `ROCKET_FA_DUMP_OP`'s files. Each file is created new with mode 0600, and a name that already exists, a symlink included, is refused and reported. A dump under `sudo -E` therefore cannot be redirected onto another file. |
| `ROCKET_FLASH_ATTN_NO_CTX` | off | Forces attention onto the per-call `_mt` route, with its own file descriptors and per-call scratch, instead of the persistent `rocket_fa_ctx`. Use it to measure what the resident score-matrix scratch is worth. |

#### `ROCKET_FLASH_ATTN`

The backend computes `softmax(scale·QKᵀ + mask)·V` on the NPU, with the heads spread across its
NPU workers and their submits chained (`ROCKET_FA_CHAIN`). On Ministral-3-3B F16 its
perplexity is within 0.03% of the CPU backend's attention, 13.8994 against 13.9024. It claims an
op only at `ROCKET_FLASH_ATTN_MIN_KV` keys and `ROCKET_FLASH_ATTN_MIN_T` query tokens or more.

| Context | Speedup over the CPU's flash attention |
|---|---:|
| 1K or less | Parity |
| 4K | 1.07x |
| 8K | 1.50x |
| 16K | 1.25x |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), Gemma-4-12B F16, governor
`performance` [HW sweep 2026-06-26/28].

**Not characterized:** these ratios on the current build.

The backend declines an op that has attention sinks (`src[4]`), a learned per-head logit in the
softmax denominator. gpt-oss, `mimo2` and `deepseek4` use them. The handler has no sink term, so
the CPU computes those ops.

If an NPU job fails, the backend computes that layer on its host reference and logs a warning.
The graph continues, as it does after a failed `MUL_MAT` or `MUL_MAT_ID`. The result is correct,
and slower.

#### `ROCKET_FA_ROWMAJOR`

The row handoff's output is bit-identical to the tiled handoff's, except for a head that runs
alone in its chained group at long context. That head differs by a few fp16 ulps, because the row
handoff accumulates it in fp32 and the tiled handoff uses fp16
[K-accumulation](README.md#terms).

| Prefill | Speedup over `ROCKET_FA_ROWMAJOR=0` |
|---|---:|
| `pp2048` | 1.058-1.068x |
| `pp4096` | 1.080-1.116x |

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), Llama-3.2-3B F16, `-fa 1`, default
`-b` and `-ub`, governor `performance` [HW sweep 2026-10-11, RK1].

#### `ROCKET_FLASH_ATTN_UNMASKED`

With this variable set, the backend claims `FLASH_ATTN_EXT` ops that have no mask, which is an
encoder's self-attention. The handler and `librocketnpu` take a missing mask as unmasked, so the
op computes correctly: the whisper transcript is md5-identical. The variable does not affect a
masked op, such as an LLM decoder's.

The offload is off by default because, at the one shape measured, it is slower than the CPU kernel
it replaces. On whisper small (12 heads, `d`=64, `T` = `n_kv` = 1500), the offload uses as many
CPU core-seconds as the CPU kernel (16.2 -> 16.1 on a 20 s window), and encoding takes 36% longer.
The `[T, n_kv]` scores make a round trip through the host softmax, and at `d`=64 that traffic
costs what the QK and AV matmuls save. With the on-NPU softmax (`ROCKET_ATTN_HOST_SOFTMAX=0`),
encoding takes 96% longer.

**Conditions:** RK3588, NPU ~816 MHz (the 600 MHz request), host pinned to the A76 cores
[HW sweep 2026-09-03].

**Not characterized:** whisper small's encoder on the current build.

#### `ROCKET_FA_QSCALE`

With this variable on, the handler folds the op's scale into Q as it converts Q to fp16, and
passes `librocketnpu` a scale of 1. The QK contraction accumulates in fp32 but writes fp16 scores,
so an unscaled `q.k` past 65504 overflows there. Folded, the scores hold `scale*q.k` and overflow
only past `65504/scale`, which is 8x further at head_dim 64 and 16x at 256. An op past that limit
runs on the full-precision host reference (see [Precision declarations](#precision-declarations)).

The score is the same function either way, `softcap*tanh(scale*q.k/softcap) + mask`, with one
fp16 rounding of Q. `0` restores the unscaled scores.

`test-rocket-fa` scores an unscaled `q.k` of 60000-135000 exactly with the fold, and wrong by up to
1.0 without it. It scores random operands within 3.2e-5 and 3.4e-5 of a double reference either
way [HW sweep, RK3588, ~816 MHz (the 600 MHz request)]. On this ARM build, the CPU backend returns
NaN on that case, because its fp16 dot product accumulates in fp16.

#### `ROCKET_FA_THREADS`

The attention handler runs five host walks: the Q conversion, the K copy, the V transpose, the
mask copy and the output scatter. They run on the dispatch thread while the scheduler waits, so
their time adds directly to prefill time. On Gemma-4-12B F16 at `pp2048` they take 5.6% of a
pinned prefill. Threading them can therefore save at most 5.6% x (1 - 1/k) of prefill time, at
`k` workers.

| `ROCKET_FA_THREADS` | Host walk time per prefill (`G_k`) |
|---:|---:|
| 1 | 5.80 s |
| 2 | 3.17 s |
| 4 | 1.87 s |

**Conditions:** RK3588, Gemma-4-12B F16, `pp2048`, host pinned to the A76 cores
[HW sweep 2026-09-07].

At 4 workers, prefill measured 1.0389x as fast as at 1. A fit of `G_k = A/k + B` to the table
leaves a fixed 0.55 s. By that fit, workers beyond the four A76 cores can save about 0.6% of
prefill time between them. `4` is therefore the value to set. The value is clamped to the shared
host pool's worker count, which `ROCKET_DEQUANT_THREADS` sets and which is the A76 count by
default.

The default is 1, which runs the walks on the dispatch thread without the host pool. The gain is
measured on one model at one shape. The workers split each walk into disjoint index ranges.

> [!NOTE]
> Two runs of the same fp16 attention can differ, because of which NPU core ran each job. The
> RK3588's three cores differ only on output lane 0 (`n % 16 == 0`), where each rarely writes a
> result 2^-16 low on its own set of inputs. The `rocket` driver places jobs by load, so a
> comparison of one run against one run compares the cores each job ran on.

**Not characterized:** whether the output is bit-identical at every `ROCKET_FA_THREADS` value.

#### `ROCKET_FA_CHECKSUM`

`ROCKET_FA_CHECKSUM` hashes the attention handler's output, so that two runs can be compared by one
number. Each level adds to the one below it:

- `1` prints, at exit, an FNV-1a hash over every byte the handler writes to its output,
  accumulated over every offloaded op.
- `2` adds one hash per op with its shape beside it, which locates a hash that moves.
- `3` adds, on each op's line, the hashes of the four dense fp16 tiles the handler passes
  `librocketnpu`: `q`, `k`, `v` and the mask. An op whose output differs between two runs then
  shows whether its inputs already differed.

`ROCKET_FA_DUMP_OP` and `ROCKET_FA_DUMP_DIR` then write that op's tiles to files, for a
value-by-value comparison.

A comparison across `ROCKET_FA_THREADS` values measures the split and also which NPU core ran
each job (see the Note under [`ROCKET_FA_THREADS`](#rocket_fa_threads)). On the op traced, the
disagreement is one fp16 ulp at one lane-0 element, and the hash cannot separate the two causes.
Repeat one configuration and bound its self-disagreement rate before reading any comparison
across thread counts. Read the op count first, since a run that offloaded no attention prints
nothing.

> [!NOTE]
> The checksum and `ROCKET_FA_TIMING`'s summary print from an `atexit` handler. A tool built on
> llama.cpp's `common_init` drops both, because `common_log` is asynchronous and discards
> messages once its worker is paused. `llama-bench -v` prints them. `llama-perplexity` prints
> neither, although it runs the attention handler.

#### Evidence

The `ROCKET_FLASH_ATTN` context ratios are from a build that predates the current softmax and the
row handoff of the scores between the matmuls.

The `ROCKET_FLASH_ATTN_UNMASKED` figures were measured on 2026-09-03. That build predates the
fused NEON softmax and the direct score de-tile, which act on the host passes those figures
measure.

The 5.6% share in `ROCKET_FA_THREADS` is from a `ROCKET_FA_TIMING=1` readout [HW sweep, RK3588,
~816 MHz (the 600 MHz request)]. The walks took 5.44 s of a 97.8 s pinned prefill: 3.46 s
gathering and 1.98 s scattering. The thread-count sweep ran three configurations in three
passes, rotated, with each ratio paired within a pass.

The `ROCKET_FA_CHECKSUM` hash agreed across `k` = 1, 2, 3, 4, 5. Two runs at one `k` on
Gemma-4-12B F16 also disagreed, twice in seven runs on one build [HW sweep 2026-09-07, RK1]. The
runs had the same graph, the same 288 ops and the same 2818572288 bytes. A replay of the dumped op
gave the odd value in 7-91% of trials at 5 workers [HW sweep 2026-09-26/27, RK1]. At 1 worker it
never did (200 of 200). An equality across `k` is therefore no evidence about the split.

`llama-perplexity` runs the attention handler without printing either summary. On Ministral-3-3B
F16, its perplexity measured 13.8994 with `ROCKET_FLASH_ATTN=1` and 13.9024 with `0`, and two `=1`
processes agreed to every digit [HW sweep 2026-09-07].

### Diagnostics

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_VERIFY` | off | Checks each offloaded op's output against a CPU recomputation. Read `max_abs`, and ignore `max_rel`. It is very slow, and needs a build with `-DGGML_ROCKET_DIAGNOSTICS=ON`. |
| `ROCKET_MM_PROFILE` | off | Prints a per-phase breakdown of host and `librocketnpu` time at exit. It includes the dequantize bucket for [streamed weights](README.md#terms) (quantized or bf16 to fp16), the largest host cost of a quantized GGUF's prefill. The profile adds noise, so leave it off for headline t/s. |
| `ROCKET_TRACE` | off | Prints one trace line per offloaded matmul, with its weight name, data pointer and amax. Needs a build with `-DGGML_ROCKET_DIAGNOSTICS=ON`. |
| `ROCKET_DEBUG` | off | Prints the fusion groups, among other debug lines. |
| `ROCKET_DEBUG_GRAPH` | off | Prints one census line per scheduler split. |
| `ROCKET_LOG_STDERR` | off | Copies the `librocketnpu` log channel, which the backend also writes to, to stderr. A host that silences its ggml logger, as `llama-bench` does without `-v`, then still shows those lines. See [Diagnostic logging](#diagnostic-logging). |

The production `.so` compiles out the heavy recompute and trace diagnostics: `ROCKET_VERIFY`,
`ROCKET_CPU_FORWARD`, `ROCKET_TRACE`, `ROCKET_AB` and `ROCKET_DUMP`. Build with
`-DGGML_ROCKET_DIAGNOSTICS=ON` to enable them. `ROCKET_MM_PROFILE`, `ROCKET_DEBUG` and
`ROCKET_DEBUG_GRAPH` are always available.

## Diagnostic logging

Every diagnostic the backend emits goes through ggml's log channel: errors, warnings, the
`ROCKET_MM_PROFILE` and `ROCKET_FA_TIMING` breakdowns, and the traces the variables in
[Diagnostics](#diagnostics) enable. The host, llama.cpp or whisper.cpp, therefore redirects or
silences them with the same `ggml_log_set` callback it uses for its own output.

Severity maps onto ggml's levels. Failures go to `ERROR`, recoverable fallbacks to `WARN`, the
profile breakdowns and one-time hints to `INFO`, and the per-op traces to `DEBUG`. An environment
variable decides whether a trace is emitted at all, and the channel decides only where it goes.

`librocketnpu` has its own log channel. The backend bridges it into ggml's at startup, so a
`librocketnpu` message reaches the same host callback instead of raw stderr. `librocketnpu`'s own
threshold applies first. A `librocketnpu` `DEBUG` trace therefore needs both `ROCKET_DEBUG` and a
host logger that prints `DEBUG`.

A host that silences its ggml logger silences the backend too. `llama-bench` installs a no-op
ggml callback unless run with `-v`. Its default run therefore hides every backend line, including
one-shot decisions such as the `ROCKET_QUANT_RESIDENT=auto` resident-weight budget. The budget
changes the benchmarked prefill number.

For that case, the backend also sends the lines a benchmark needs through the `librocketnpu`
channel, among them:

- Its configuration lines: the budget decision, and the "raise `-ub`" quant hint
- Its residency outcome lines, `[f16-resident]` and `[moe-int8]`
- Its own `ROCKET_MM_PROFILE` breakdowns

`ROCKET_LOG_STDERR=1` copies them to stderr whatever the host's logger does. Set it beside a
benchmark to confirm which budget or mode engaged, and how far it got. A benchmark row answers
neither question.

## Route counters

Two functions report which route computed the backend's ops, for a caller that
must assert it rather than infer it from the output.

`ggml_backend_rocket_route_ops(backend, route)` returns the number of ops the named route
completed. The `MUL_MAT` routes are:

| Route | What it computes |
|---|---|
| `mt` | The per-call fp16 route, through the persistent context unless `ROCKET_NO_STREAM` is set |
| `prepacked` | [Resident weights](README.md#terms) in fp16 |
| `prepacked-q` | A quantized weight held resident as fp16 |
| `bf16` | The bf16 route |
| `int8`, `int8r` | int8, per call and resident |
| `i8-76` | [The RK3576 route](README.md#terms) |
| `int4`, `int4r` | int4, per call and resident |
| `fused` | A fused QKV or gate-up group, per call |
| `fused-resident` | A fused group with resident weights |

`moe` counts a `MUL_MAT_ID` the expert handler computed, and `fa` a `FLASH_ATTN_EXT` the NPU
computed. A `NULL` route returns the total over every route.

`ggml_backend_rocket_cpu_fallbacks(backend)` returns the number of slices, MoE experts and
attention ops that the backend computed on the CPU instead. A fallback follows a failed NPU job,
or an attention op whose range fp16 cannot hold. The CPU reference of the `ROCKET_MOE_COSINE`
check is not counted. An `mt` or `moe` op counts once in its route however many of its slices or
experts fell back, so read the two counts together.

Both counts are cumulative over the backend's life. The backend's tests take a mark before each
run and assert over the difference.

> [!CAUTION]
> A matmul fallback computes in fp64, as a test's CPU reference does. A test that compares only
> the output therefore passes on a device that computed nothing.

## Precision declarations

A ggml graph can declare, per op, the lowest precision its implementation must keep
(`ggml_prec_set_acc` and `ggml_prec_set_src`, ggml 0.25). A backend that cannot meet a declaration
must leave the op to one that can, and `supports_op` declines these:

| Declaration | Declined when | Why |
|---|---|---|
| Accumulator of a `MUL_MAT` / `MUL_MAT_ID` | F32 or BF16 | No route guarantees an fp32 result. The fp16 route writes fp16, and with `ROCKET_KACC` on, its default, sums the K-partials in fp16. The RK3576 route writes int8 through its requant. The bf16, int8 and int4 routes write fp32 or int32 but fall back to the fp16 route when they decline. |
| Activation (`src1`) of a `MUL_MAT` / `MUL_MAT_ID` | Declared above the route's own rank | The fp16 route converts the activation to fp16. `ROCKET_INT8`, the RK3576 route and the [native-quant expert](README.md#terms) route quantize it to int8, and `ROCKET_INT4` to int4. `supports_op` takes the lowest rank the op could reach, because every route falls back to fp16 and the quantizing routes are tried first. |
| `GGML_HINT_SRC0_IS_HADAMARD` on a `MUL_MAT` | Always | The CPU backend runs it as a fast Walsh-Hadamard transform, O(n log n) a row and exact in f32. Here it would be a dense n × n fp16 GEMM. |

The ranks run F32, BF16, F16, Q8, Q4, highest first, and an accumulator declared F16 is met by
every route. In llama.cpp b11242 these declarations reach an offloadable op:

- An F32 activation on Mistral-4's routed down-projection
- An F32 accumulator on the `ffn_down` and attention output of GLM-4, GLM-4-MoE and JAIS-2, on
  MiniMax-M3's attention output, and on DeepSeek-V4's expert router
- A per-weight activation rank, which a GGUF can store in `general.tensor_extra`

The Hadamard tag appears around a quantized KV cache (`-ctk` / `-ctv` with a quantized type) and
in DeepSeek's lightning indexer. None of the models in this document's tables has one.

### `FLASH_ATTN_EXT` accumulator

The backend claims a `FLASH_ATTN_EXT` that declares an F32 accumulator, and guards its range.
llama.cpp declares F32 on every attention op it builds, and every attention figure in this
document was measured with it set.

The CPU backend is the declaration's reference implementation, and it runs F32 and the default
through one kernel. For F16 K and V, that kernel narrows Q to fp16 and accumulates P·V in fp16
(ggml-cpu's `ops.cpp`, the `VKQ16` path). The backend accumulates both contractions in fp32 on the
NPU, and runs the soft-cap, mask and softmax in fp32 on the host. It holds the scaled Q, the
scores, P and the output in fp16, so its rounding is the reference's.

The NPU's fp16 scores saturate at 65504 instead of becoming inf, so an overflowed logit ties with
every other saturated key. The softmax then returns a finite output of the right size with wrong
values [HW sweep 2026-10-09, RK3588]. Under `ROCKET_FA_QSCALE=0`, `test-rocket-fa` measures an
error of 1.0 and no inf, at ~816 MHz (the 600 MHz request).

The backend therefore guards the range at both ends. A scaled Q element past 65504 is caught as Q
is gathered, and `librocketnpu`'s score pass flags a scaled logit past it (`ROCKET_E_RANGE`).
Either sends the op to a full-precision host reference. The reference reads Q as F32, computes q·k
and the softmax in double, and rounds the output once to F32. The op counts as a fallback, and the
reference is slower than both the NPU and the CPU kernel. No model has been measured reaching it
[expected].

## Implementation notes

The backend uses ggml's private headers, `ggml-impl.h` and `ggml-backend-impl.h`, as is expected
of an out-of-tree backend. Its buffers and its placement work as follows:

- A host built with `GGML_BACKEND_DL` loads the backend through `ggml_backend_load_all()`, which
  honors `GGML_BACKEND_PATH`, and `whisper-cli`'s `main()` calls it. The `.so` exports
  `ggml_backend_init` through `GGML_BACKEND_DL_IMPL`, and its device type is ACCEL, which the host
  adds as it adds the BLAS device.
- The backend's device uses the host's CPU buffer type and accepts any host buffer, so weights
  stay in host buffers.
- For ggml's scheduler to move an op in a host buffer onto an accelerator device, both
  `supports_op` and `offload_op` must return true. The BLAS backend works the same way. A NULL
  `offload_op` leaves the device listed at startup, running zero matmuls.
- `ggml_fp16_t` (`uint16_t`) has the IEEE half-precision bit layout of `_Float16`, so the
  backend fills `librocketnpu`'s `_Float16*` buffers with `ggml_fp32_to_fp16_row` and a cast.

### ggml version compatibility

The backend fills ggml's vtables, `ggml_backend_i` and the rest, with C++20 designated
initializers, so each function binds to its slot by name. Only the slots the backend implements
are named, and ggml reads every other slot as an unimplemented optional one. A slot ggml inserts
therefore builds unchanged, and a change to a slot the backend fills is a compile error.

The backend is audited against `GGML_BACKEND_API_VERSION` 2 and 3. Version 3 adds two slots to the
buffer-type vtable, which this backend does not implement, because its device uses the host's CPU
buffer type. Any other version builds with a warning, because a new slot that ggml requires is the
one change naming cannot see. `-DGGML_ROCKET_STRICT_ABI=ON` turns that warning into an error.

It builds against ggml 0.14.0 through 0.25.3, including the ggml of llama.cpp b11351 through
b11401, which has version 3 under the same 0.25.3 label. It also builds with
`-DGGML_ROCKET_STRICT_ABI=ON` against llama.cpp b11539 and whisper.cpp v1.9.5, both at version 3.
On both, the NPU's greedy completion and transcripts are byte-identical to the CPU's [HW sweep
2026-10-09, RK1].

> [!CAUTION]
> llama.cpp and whisper.cpp each clone their own ggml. Build the backend against that checkout's
> headers. ggml's loader refuses a `.so` built for another `GGML_BACKEND_API_VERSION`,
> logging `incompatible API version (backend: N, current: M)`, and the device never appears.

### Op numbering check

At load, the backend checks the host's op numbering. ggml numbers its ops by position in an
enum, and inserting one shifts every op after it without moving `GGML_BACKEND_API_VERSION`. ggml
0.15 inserted `COL2IM_1D`, so `FLASH_ATTN_EXT` moved from 73 to 74 while `MUL_MAT` stayed at 29.

The load entry asks the host's `ggml_op_name` to name each op number the backend handles, and
returns `NULL` on a mismatch. The loader reports that as `ggml_backend_init returned NULL` and
unloads the `.so`. Before that line, the backend logs the op, for example `op ordinal 73 is
FLASH_ATTN_EXT in this .so and FILL in the host's ggml`. Loaded into whisper.cpp 1.9.4, a `.so`
built for whisper.cpp 1.8.6 logs that line.
