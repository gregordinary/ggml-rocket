# ggml-rocket: API and tuning reference

The exhaustive per-model benchmarks, the full `ROCKET_*` runtime-knob table, the
diagnostic-logging channel, and the backend implementation notes. The [README](README.md) is
the guide, and this is the reference.

## Models and precisions

Every entry below is HW-validated on the RK3588 unless its own text says otherwise.

### Whisper encoder

Stock `whisper-cli` and `whisper-bench` load the `.so` via `GGML_BACKEND_PATH` and run the
encoder `MUL_MAT`s on the NPU. Transcripts are byte-identical to the CPU, at WER 0 and
encoder-output cosine 0.9998.

Encoder latency beats the CPU at every size, and the win **grows with the model**.
`whisper-bench` encode, warm, RK3588 @ 600 MHz:

| Model | Encoder speedup |
|---|---|
| tiny.en | 1.18x |
| base.en | 1.29x |
| small.en | 1.54x |
| medium.en | 1.84x |
| large-v3 | 2.14x |
| large-v3-turbo | 2.12x |

The decoder is autoregressive at M=1 and stays on the CPU on both backends.

**large-v3-turbo** keeps the full 32-layer encoder but a 4-layer decoder, ~6x cheaper per step.
The NPU-accelerated encoder is therefore the larger share of its pipeline.

### Multi-model STT through transcribe.cpp

The same `.so` drops into transcribe.cpp, a ggml-based host for Granite-Speech, Voxtral, MOSS
diarize, SenseVoice, FunASR and Parakeet. It offloads the **encoder**, under the same law as
Whisper.

Encode always offloads, at 1.2x-2.7x, scaling with encoder size times audio length.
Autoregressive decode is M=1 and stays on the CPU. The exception is the batched decode-prefill
over the injected audio context on long audio, which offloads when it is large enough.

So the win is set by architecture:

| Shape | Model | Result |
|---|---|---|
| Cross-attention decoder | Granite-Speech | Offloads decode best, ~1.9x on 120 s |
| Big decoder-only over long audio | Voxtral-mini 3B | Offloads its ~1500-token audio prefill: decode 1.46x, whole-pipeline 1.57x |
| Small decoder-only | MOSS, FunASR 0.6B | Decode-bound, ~1.0-1.15x despite a ~1.6x encode |
| Single-pass CTC | SenseVoice | No decode: 6.2x realtime, 1.26x |

Encoder-only offload is bit-faithful CPU-against-NPU. A model whose decoder cross-attention
offloads, such as Granite, can diverge late as fp16 accumulation perturbs greedy decoding.

**Q8_0 is the STT default.** It ties or beats F16 on both backends at half the RAM. These models
are less matmul-dominated than LLM prefill, so the per-micro-batch dequant tax is small. [HW A/B, 2026-07-21]

### Gemma-4-12B prefill

Coherent generation via the same drop-in on llama.cpp. Prefill GEMMs offload, and decode stays
on the CPU.

### Phi-4 (14B) prefill

The 14.66 B Phi-4 is arch `llama`, dense GQA, a different architecture from Phi-4-mini's `phi3`.
It prefills on the NPU through the same drop-in, **PPL-faithful** to the CPU, at Q8_0 and Q4_K_M
wikitext Δ −0.09% and −0.31%.

Its F16 GGUF is 29.3 GB and does not fit a 31 GB board, so it runs quantized. **Q8_0 and Q4_K_M
reach ~3.4-3.5x CPU at pp2048**, about 12 t/s over a ~3.5 t/s 14 B CPU baseline. The win rises
with prompt length as the per-micro-batch dequant amortizes.

Q4_K_M at 8.28 GB is the practical pick. It fits a ~9 GB budget and streams decode ~2.2 t/s. It
is a mid-large point between Gemma-4-12B and Qwen3.6-27B.

### Qwen3.5 and Qwen3.6 prefill

Qwen3.5 (0.8B and 9B, dense GQA) and **Qwen3.6-27B, a hybrid with Gated-DeltaNet linear
attention**, prefill on the NPU through the same drop-in, **PPL-faithful** to the CPU.

The DeltaNet and SSM-scan layers are CPU-only ops and stay there. The FFN and projections offload
and the prefill win holds. The hybrid's linear-attention layers do not block it.

**27B Q4_K_M reaches 4.4x CPU at pp2048**, the largest prefill win measured. Its quant laps an
especially slow 27B CPU baseline.

The NPU prefill advantage **grows with model size**: F16 0.8B 1.44x to 9B 3.65x CPU. A quantized
GGUF wants `-b 2048 -ub 2048`, in Performance and `ROCKET_MIN_M_QUANT` below.

### Small dense models: Llama-3.2-3B, Ministral-3-3B, Phi-4-mini

The 3-4 B dense end, arch `llama`, `mistral3` or `phi3` with GQA, posts the **fastest prefill**
through this backend. Llama-3.2-3B F16 reaches **61.8 t/s at pp512, 3.4x CPU**, Ministral-3-3B
57.6 and Phi-4-mini 55.4, because there are fewer FLOPs per matmul.

The F16 win narrows with prompt length, from 3.4x to 2.4-2.6x by pp2048. A small model's per-op
readback grows with M. It therefore lands hardest on short-to-medium prompts.

A Q4_K_M fits a 4 GB board, at Llama 1.87 GB and Ministral 1.99 GB, and streams decode fastest at
~7.7-7.9 t/s. NPU prefill is PPL-faithful to the CPU, at Llama and Ministral F16 wikitext Δ
+0.02% and −0.04%.

The NPU-to-CPU prefill *ratio* is modest on these, because their CPU baseline is comparatively
fast. The NPU's absolute prefill rate is the headline.

### Vision-language: SmolVLM2-2.2B

The full pipeline runs on the NPU through llama.cpp's `mtmd` path. That is a **SigLIP-SO400M
vision encoder** plus a **SmolLM2-class language model**, arch `llama` at 1.81 B. The "2.2B" is
the combined count.

The LLM half is the **fastest in this backend**. F16 reaches **99 t/s at pp512, 3.5x CPU**, with
the briskest decode at Q4_K_M **14.5 t/s** from a 1.03 GB file. It is PPL-faithful, at F16 and
Q4_K_M wikitext Δ −0.07% and −0.71%.

The vision encoder runs through `clip.cpp`. Attach the NPU with **`MTMD_BACKEND_DEVICE`**.
clip's auto-path only probes GPU and IGPU device types. The ACCEL device must therefore be named
explicitly, or the encoder stays on the CPU.

Offloaded, the SigLIP encode is **~1.19x faster warm** and **faithful at cosine 0.99998** on
projected image embeddings against the CPU. That is a modest win. ggml-rocket offloads only the
aligned static-weight GEMMs, the q/k/v/o projections and fc1.

Three things stay on the CPU: fc2 (K=4304, off the K%32 grid), the attention core, and all norm,
softmax and GELU ops. The graph therefore pays many CPU↔NPU handoffs. A resident whole-encoder path would widen this, and the drop-in clip route is
correct and modestly positive as it stands.

### Quantized GGUF prefill

A `Q8_0`, `Q4_K`, `Q6_K` or similar GGUF runs its prefill GEMMs on the NPU directly. Each weight
is dequantized to fp16 on the fly, one `[K]` row at a time. Those are the exact values ggml's CPU
backend decodes, with **no whole-model F16 copy in RAM**.

So the model stays compact on disk and in memory, at Gemma-4-12B `Q8_0` ~12 GB against the F16
GGUF's ~24 GB. Prefill still takes the fp16 NPU path, and decode stays on the CPU on the
quantized weights. Quantization buys RAM and model-fit here rather than prefill speed, because
the matmul is dtype-independent, in Performance below.

The dequant is paid **per micro-batch**, so quantized prefill is micro-batch-sensitive. Run with
**`-b 2048 -ub 2048`** for a model-dependent 0.91-1.53x over the llama.cpp default `-ub 512`,
which re-dequantizes every 512-row chunk. Short prefills below `ROCKET_MIN_M_QUANT`, default 512
rows, just past the measured ~360-row crossover, stay on the CPU, where they beat a dequant-bound
NPU offload.

Importance-matrix quants such as `IQ4_XS` take this same path, because an imatrix is a
*quantize-time* construct. At inference they are ordinary dequant-to-fp16 types, where `IQ4_XS` is
4.25 bpw with 256-element super-blocks and `K%32` holds. Their prefill GEMMs offload like any quant rather than falling back to the CPU. Qwen3.5-9B
`IQ4_XS` prefill is 3.1x CPU at pp2048, the fastest-decoding rung at 4.80 GB.

It requires the host built `-DGGML_CPU_REPACK=OFF`, in *Drop-in use with llama.cpp* below.

### Resident weights

Static F16 weight tensors are packed into resident NPU BOs and cached by **weight name**, not by
`src0->data`. The scheduler reuses pooled copy addresses across weights, so keying on the address
aliases distinct weights and produces gibberish. `supports_buft = is_host`.

Holding a weight resident pays its scatter (`packB`) **once** and reuses it across every later
micro-batch and prefill, instead of re-scattering per call.

The cache is keyed on the name alone and **not** on the prefill M.

A weight packed at one prefill length is reused at any other where M >= the tile cap. There is no
re-pack: the resident layout is M-independent there.

A weight is made resident only when first seen at a reuse-worthy M.

A smaller one-shot M streams via the per-call mt path: a short prompt, or a long one's tail
ubatch. That leaves the resident weight intact for the big prefills, with no re-pack thrash.
Residency is default-gated to **K<=2048** weights, which is whisper's encoder, resident by
default.

#### Extending residency to all-K weights

**`ROCKET_F16_RESIDENT=auto`** extends residency to **all-K** F16 weights, the LLM attention
  and FFN at K∈{3072,8192,…}, with a free-RAM-sized budget. It is the F16 sibling of
  `ROCKET_QUANT_RESIDENT`, and **decode-safe**: no `madvise`, so the GGUF stays mapped for CPU
  decode.

  It disables QKV and gate-up fusion so every weight goes resident. Residenting the fused weights,
  about two thirds of the `packB` bytes, beats fusing them.

  Llama-3.2-3B-F16 at 600 MHz and `-ub 512` reads pp2048 39.8->42.1 (**+5.9%**) and pp512
  54.5->59.6 (**+9.5%**), token-identical to streaming [HW sweep].

  The win is **larger at shorter prefills**, because long-prefill attention stays on the CPU and
  dilutes the offloaded-matmul share. It is also **larger on combined-projection architectures**.
  Phi-4-mini has single `qkv` and `gate_up` weights, so nothing to fuse, and reads **+21%, +16%
  and +14%** at pp512, 1024 and 2048.

  The `auto`, `<N>`-MB and `1` modes and the reserve mirror `ROCKET_QUANT_RESIDENT`. For F16
  models that fit ~2x in RAM, the resident tiles duplicate the still-mapped GGUF. A too-large
  model degrades to partial residency via a runtime `MemAvailable` floor, so there is no OOM on a
  swapless board.
#### Both modes report their outcome at teardown

They report the outcome rather than only the budget:

    [f16-resident] weights offered to the resident route: N resident on the NPU (X MB),
    M streamed via the per-call pack -- P% resident

The limit that stopped admission follows, with its numbers: the byte budget, the `MemAvailable`
floor, or a full IOVA window. Under 80% resident it adds a warning.

The denominator is the weights *offered to the route*, and a fused group's members count
individually. The number therefore means the same thing with fusion on and off. It goes through
the driver log channel, so `ROCKET_LOG_STDERR=1` shows it under a host that silences ggml.

**Read it before calling a residency measurement a zero.** A run that placed a quarter of its
weights produces the same rows as one that placed all of them and gained nothing. It can produce
nearly the same t/s. The budget line alone cannot tell those apart.

One further knob sits beside them:

- `ROCKET_FORCE_PREPACK=1` is the diagnostic force (default 2 GB budget unless `ROCKET_CACHE_MB`
  set) with an optional prefill-only `ROCKET_PREPACK_MADVISE` that reclaims the F16 source (breaks
  CPU decode).

### int8 W8A8 with a Hadamard rotation

coherent and char-identical to fp16, but a **net loss** in
prefill speed (see below).

### int4 W4A4 in-model

Native int4xint4 prefill for a Gemma-4-12B F16 GGUF, via `ROCKET_INT4=1`. Weights and
activations are quantized to 4-bit, symmetric `[-7,7]`, with **group-wise scales**, the per-128
K-group W4 quality lever, and a **Hadamard rotation**. Both are on by default.

It generates **char-identically to fp16**: "Paris / Eiffel Tower / Louvre Museum", primes and
Jupiter. Per-matmul cosine against fp16 is ~**0.98-0.99**. Per-channel with Hadamard reads ~0.978,
group-128 ~0.987, and group-64 ~0.989. Without Hadamard W4A4 collapses to ~0.79.

Hadamard needs full-precision weights, so int4 quantizes from an **F16** GGUF rather than a
pre-quantized Q4 one. The knobs are `ROCKET_INT4_GROUP` (group size, default 128, where 0 is
per-channel), `ROCKET_INT4_HADAMARD` (default 1), and `ROCKET_INT4_CACHE_MB`.

### int4 W4A4 resident

`ROCKET_INT4_RESIDENT=1` holds the group-wise int4 weights in **resident NPU nibble BOs**. The
Hadamard rotation is baked into the resident weight, and only A is rotated and quantized per call.
Prefill therefore pays **no per-call weight scatter**. Quality is the same as the non-resident
path, at cosine 0.987 and char-identical to fp16.

Gemma-4-12B pp512 at 600 MHz: **resident int4 6.94 t/s against non-resident int4 3.56**, ~2x from
the packB elimination, **and fp16 13.18**, so 0.53x. It both **halves int4's prefill cost** *and*
cuts the weights' NPU footprint to **a quarter**, 2634 MB resident against ~10.5 GB at fp16.

Against fp16 it stays a RAM play. The group-wise int4's ~120 K-group readbacks per deep-K matmul
keep it behind fp16-plus-KACC's single readback.

`ROCKET_INT4_CACHE_MB` caps the resident BO total, and is worth raising: a full 12B is ~6 GB of
nibble BOs, and the offloaded subset here is ~2.6 GB. The one-time per-weight Hadamard build is
parallelized across CPU cores, ~14x, about 3 min for 12B.

The resident weight is **M-independent**. The driver plans its tile layout at a canonical `M`.
The weight packed during warmup therefore serves any later prefill `M` with **no re-pack**,
a short-prompt one included.

The cache is keyed on weight name rather than on the padded M.

The same holds for the resident int8 and fp16 weights, so warmup-M serves prefill-M across the
board.

## Gemma-4-12B

The large-end model in this stack. RK3588, 8-core, NPU at 600 MHz, warm.

**Prefill (prompt processing), t/s, CPU -> NPU:**

| | pp512 | pp1024 | pp2048 |
|---|---|---|---|
| F16    | 4.82 -> **17.42** (3.6x) | 4.78 -> 16.09 (3.4x) | 4.63 -> 14.98 (3.2x) |
| Q8_0   | 4.52 -> 11.41 (2.5x) | 4.35 -> 13.53 (3.1x) | 4.16 -> 13.43 (3.2x) |
| Q4_K_M | 4.23 -> 11.20 (2.6x) | 4.15 -> 13.19 (3.2x) | 4.02 -> 13.28 (3.3x) |

F16 prefills fastest at 17.4 t/s, and its win narrows with M, from 3.6x to 3.2x, as the output
readback grows. The quants rise with M as their per-micro-batch dequant amortizes, converging on
F16 by pp2048 at ~13-15 t/s. A quantized GGUF fits smaller, and it does not prefill faster at this
operating point.

**Decode (generation), t/s, CPU ~ NPU.** Decode is off the NPU and bandwidth-bound, so it scales
with the quant: F16 **0.9** -> Q8_0 **1.7** -> Q4_K_M **2.5** t/s. At 12 B and F16, 22 GiB, decode
is slow at 0.9 t/s, so a quant is effectively mandatory for interactive use on this model.

**Interactive.** The NPU shortens the first-token wait, and the quant speeds the stream:

| turn | config | time-to-first-token | stream | total |
|---|---|---|---|---|
| 2048-tok prompt -> 200 out (RAG) | F16 CPU | 442 s | 0.9 t/s | 655 s |
| | F16 + NPU | **137 s** | 0.9 t/s | 350 s |
| | Q4_K_M + NPU | 154 s | 2.5 t/s | **234 s** |
| 128-tok prompt -> 400 out (chat) | F16 + NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M + NPU | 11 s | 2.5 t/s | **171 s** |

The NPU cuts first-token wait ~3.2x on a long prompt. This model's slow decode means the stream
dominates the turn, so **Q4_K_M + NPU** is the practical pick. That is ~3.3x prefill at length and
a 2.7x-faster stream, in 6.86 GB, which fits an 8 GB board. F16 is the max-prefill option when RAM
is ample, and its GGUF is 22 GiB.

NPU prefill is perplexity-faithful to the CPU. Differential wikitext PPL Δ stays within ±2% across
F16, Q8_0 and Q4_K_M, inside the ±10% per-run noise. The absolute PPL of this instruct model on
raw text is not meaningful.

## MoE and MLA models

The `MUL_MAT_ID` expert path (gpt-oss-20b, DeepSeek-V2-Lite).

Mixture-of-experts models route the expert FFNs through `GGML_OP_MUL_MAT_ID`. This backend has a
`MUL_MAT_ID` handler. It buckets the routed `(slot,token)` rows by expert and runs each active
expert's GEMM on the NPU, bit-faithfully.

It is worth **~2.4x the CPU** at pp2048 on gpt-oss-20b, and it wins at every prefill length there,
in the table below. Eligibility is per architecture rather than universal. The DeepSeek-V2-Lite
section covers a MoE whose routing puts it under the per-dispatch work floor at short prefill.

**`ROCKET_MOE` has three states**, because the route's two regimes have opposite signs:

| `ROCKET_MOE` | placement |
|---|---|
| unset (**default**) | Claim an expert op only where the offload is measured to win: a **GGUF-quantized** stack, on the RK3588, whose whole `[K, N, n_expert]` stack the residency pre-flight can reserve before the first ingest, **and** whose per-expert GEMM clears both size floors: the tile granule (`ROCKET_MOE_M_BUCKET`) and the per-dispatch work floor (`ROCKET_MOE_MIN_WORK`). Anything else stays on the CPU. |
| `1` | Claim every `MUL_MAT_ID` the handler can compute, the fp16 streaming route included, with no reservation. The A/B arm; not the recommended setting. |
| `0` | Experts stay on the CPU whatever the shape. |

**The pre-flight is what makes the default safe.** An expert stack's resident cost is knowable
from its tensor alone, `[K, N, n_expert]` plus the GGUF source stride. The decision is therefore taken in `supports_op`, *before* the first ingest. It is taken against
both the RAM budget and the aggregate NPU IOVA window, `n_threads x 4 GB`, where the window is per
fd.

A stack that will not fit is left on the CPU **whole**, rather than half-ingested into the
partial-residency loss described under **Residency is the route**.

**The budget is read once per process, and it moves.** `ROCKET_MOE_CACHE_MB`'s auto value is
`MemAvailable − 6 GiB`, resolved at the **first** `supports_op` and then frozen. A re-read per
context would shrink under its own success, because the ingest it authorizes is what lowers
`MemAvailable`.

Frozen per process still leaves it varying *between* processes. Measured on one board in one
session, the same DeepSeek-V2-Lite run saw **24.6 GB** early and **20.7 GB** after several hours of
multi-GB allocation churn. That is the difference between admitting all 78 of its expert stacks
and 70.

A long-lived server that builds a fresh `llama_context` late in its life can therefore place
fewer stacks on the NPU than at startup. Nothing warns about it. The teardown line from
`ggml_backend_rocket_moe_stats` is the only place it shows.

Two things make it reproducible, and both are the caller's. **Pin `ROCKET_MOE_CACHE_MB`** to a
value the board can hold, or **drop the page cache before the run**
(`echo 3 > /proc/sys/vm/drop_caches`).

With the cache dropped first, three processes requesting 24679, 26000 and 28000 MB admitted 9539,
10032 and 10810 experts. A repeat of the auto arm landed within 13 experts of a separate probe.

That is a procedure measured to work rather than an explained one. `MemAvailable` already counts
most reclaimable page cache, so why dropping it steadies the number is not established.

**The default degrades in the safe direction as the board shrinks, and both halves of that are
design, not luck.**

#### An unreservable stack is never claimed, so it is not streamed

The pre-flight runs in `supports_op`. A stack the budget cannot hold stays with the CPU backend,
and those layers run entirely on the CPU. That is the experts-on-CPU baseline rather than a loss.

The partial-residency failure mode is a **`ROCKET_MOE=1`** property, meaning forced claims without
reserving. At an induced 12 GB budget it falls to 52% resident and reads 0.97x the experts-on-CPU
baseline at pp512. The default cannot reach that state, which is what lets it be on by default at
any RAM size.

#### The 6 GiB reserve is flat on purpose

Flatness is what makes it conservative on a small board. It is 19% of a 32 GB board, 37.5% of
16 GB and 75% of 8 GB. So `auto` withholds a larger fraction the smaller the board is, and at 8 GB
it effectively disables the route. That is the right direction.

It deliberately does **not** use the dense paths' `max(6 GiB, 30% of RAM)` rule. The 30% arm
exists to cover a resident set that duplicates a still-mapped GGUF. This route charges that same
GGUF explicitly, per expert, so applying both would count the mapping twice. Do not "fix" it into
a proportional reserve.

**Two size floors are the second half of it, and residency implies neither.** The rows an expert
actually receives are `M_e = n_tokens · n_used / n_expert`. That is a property of the
**architecture** rather than of the prompt. gpt-oss routes 4-of-32, so 512 tokens give each expert
~64 rows. DeepSeek-V2-Lite routes 6-of-64, so the same 512 tokens give ~48. At **100% residency** DeepSeek at
pp512 reads **19.09 t/s against 26.04 with the experts left on the CPU**, a 27% regression that
residency cannot see. Two floors bound it:

- **The tile granule** (`ROCKET_MOE_M_BUCKET`, 64 rows). `M_e` is rounded up onto a fixed bucket
  ladder, and the pad rows are computed and read back in full. An expert GEMM smaller than the
  smallest tile the resident path can run is therefore mostly padding. This one is tile
  geometry.
- **The work one dispatch carries**, `ROCKET_MOE_MIN_WORK`, 340 mega-MACs of `M_e · K · N`. The
  per-expert cost the granule exists to amortize is paid **per dispatch**: one gather, one
  activation quantize, one submit, one fence, one scatter. The work a dispatch carries is
  `M_e · K · N`.

  So the crossover is a threshold on that product, and `M_e` is a proxy for it only while `K · N`
  is constant. It is not. gpt-oss's expert GEMM is 2880×2880 and DeepSeek-V2-Lite's is 2048×1408,
  **2.88x smaller at the same row count**.

Mapping `M_e` -> (default ÷ experts-on-CPU) over `-p 512…2048` on both models, 100% resident and
0 streamed throughout [HW sweep 2026-08-27, 600 MHz pinned, governor `performance`]:

| `M_e` | 64 | 72 | 96 | 128 | 144 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss (2880×2880) | 1.64x | n/a | **1.77x** | 1.91x | n/a | 2.05x | 2.08x |
| DeepSeek (2048×1408) | n/a | **0.948x** | 1.049x | n/a | 1.225x | 1.313x | n/a |

**gpt-oss wins 1.64x at `M_e` = 64 while DeepSeek loses 5-6% at `M_e` = 72.** Any row floor low
enough to admit the first admits the second, so no row floor separates them. Work per dispatch
does: the cells sort by it with no overlap, across both architectures at once.

DeepSeek's cells are means of four adjacent pairs. Its offloaded arm varies ~15% run to run, so a
single pair cannot decide a cell this close to 1.00. `M_e` = 72 and 144 are four pairs, `M_e` = 96
is eight, and `M_e` = 192 is two.

gpt-oss's margins sit an order of magnitude outside that spread, so one pair each suffices.

**Which floor binds is per architecture.** On gpt-oss it is always the granule. The work floor
would first bind at `M_e` = 41, which the granule already excludes. Adding it therefore left
gpt-oss's placement unchanged by construction rather than only by measurement. On DeepSeek-V2-Lite the work
floor binds at `M_e` >= 118, past ~1250 tokens in a micro-batch.

The **form** is derived. The **number** is fitted on two architectures, and a third is where it
gets tested. It changes placement there. A 128-expert and 8-used model with a 2048×768 expert
would be declined at `M_e` = 128, 201 MMAC, where the row floor alone accepted it.

**The floor is a MATERIALITY bar rather than `> 1.00`, and the ingest is why.** The one-time
expert ingest is ~32 s per `llama_context` on DeepSeek and ~36 s on gpt-oss. It is charged **only
if the gate accepts**, so the gate is the one place it can be avoided.

An offload at ratio `r` saves `1 − 1/r` of prefill wall. The ingest therefore breaks even after
**~16 300 tokens** of prefill at a 1.05x cell, **~4 800** at 1.22x, and **~2 100** at gpt-oss's
1.64x. A cell a few percent above parity costs a short session more than it saves. That is what
makes a bar above 1.00 the correct default rather than a cautious one.

The grain is **one weight stack**, meaning one layer's `ffn_gate_exps`, `ffn_up_exps` and
`ffn_down_exps`, rather than the whole model.

"Decline unless the entire model fits" would decline gpt-oss-20b on the very 32 GB board its win
was measured on. The codes are decoded from the GGUF, and 19.1 GiB of int8 codes plus the
11.3 GiB GGUF do not both fit. The winning run was 99% resident rather than 100%.

Per-stack admission keeps that win and still has no cliff, because a **declined** stack runs on the
CPU rather than streaming. The M-independent dequant is then never paid at all. What is left is a
blend of the offloaded route and the CPU baseline, bounded below by the baseline at every RAM size.

A declined stack costs nothing at the seam either. This backend's buffer type *is* the CPU buffer
type. A CPU-placed op beside an offloaded neighbour is therefore a scheduler bookkeeping entry
rather than a copy.

**How an expert reaches the NPU decides everything.** There are two routes, and only one of them wins:

- **The fp16 dequant route** (`ROCKET_MOE_NATIVE=0`) dequantizes each expert weight to fp16 on
  the host *every micro-batch*. MoE has `n_expert` times more distinct weights per layer than a
  dense model, each amortized over only `M_e` rows. The decode does not shrink with the row count.
  So it costs more than the GEMM saves, where the CPU's fused quantized kernel pays no dequant at
  all. Measured on gpt-oss-20b: **4.59 t/s at pp512, 10.18 at pp2048**, *worse than
  leaving the experts on the CPU*. It is the A/B baseline, and the default placement does not
  claim it.
- **The native-quant route** (`ROCKET_MOE_NATIVE`, **on by default**) is the only route the
  default placement claims. It ingests a **GGUF-quantized** expert **once** into int8 codes that
  stay resident in NPU BOs, and each call quantizes only the activation, per `(row, K-group)`.
  That deletes the per-micro-batch dequant entirely, and it is what turns the loss into the win.
  An **F16** expert always takes the fp16 route, since it has no dequant to delete.

### Native-quant experts

Three facts make this work, and each is load-bearing.

#### The NPU cannot apply a K-blocked scale on chip

At the output stage `K` is fully contracted, so nothing in the DPU is indexed by a K-block.

Integer partials **already** leave the chip at every K-tile boundary, because on-device integer
K-accumulation is architecturally impossible. So a per-K-group scale rides along **free at a
boundary already being paid for**. Keep each K-tile inside one quant group, and fold that group's
scale into the readback loop that exists anyway.

#### An MXFP4 code already *is* an exact int8 value

The codebook is declared `int8_t` and reaches only ±12, and MXFP4's block scale is E8M0, an exact
power of two. So merging the native 32-element blocks of a K-group onto one exponent is an
**integer shift rather than a requantization**.

Every other quantized type takes a dequantize-once-requantize route instead, which is equally
fine. What costs is a dequant *per micro-batch*, not one at load.

#### The speed does not come from the quantization

The int8 GEMM's int32 output reads back at 8 bytes per element, so it moves *more* bytes than the
equivalent fp16 GEMM. The entire win is deleting the per-micro-batch dequant and weight scatter.

Quantization here buys residency, and residency buys the speed. That also means the ceiling is
what fp16 would do, not better.

Residency is **admission-only**. Prefill touches every expert every micro-batch, so there is no
hotness for an eviction policy to exploit. Only the resident *total* decides how much of the
dequant tax is removed.

An expert that does not fit the budget or the NPU's per-fd IOVA window streams on the fp16 route
instead, correctly. `ggml_backend_rocket_moe_stats` reports the split, and it is logged at
teardown.

On a 31 GiB board gpt-oss reaches **99%** resident, ~14 GB of int8 experts alongside the 11.3 GiB
GGUF, which must stay mapped for CPU decode. That holds only once the resident weights stopped
wasting 35% of their bytes on tile padding. A smaller board lands short. A partial blend is a
correct outcome rather than a failure, and simply a slower one, below.

**The ingest is not free, and it is paid once per context.** It happens lazily, inside the first
prefill, and it is **~70 s** on gpt-oss-20b: ~1750 experts at **42 ms each**. The NPU-BO scatter is
the larger half at 50 s, and the MXFP4-to-int8 decode the smaller at 21 s [HW sweep, 600 MHz].

`ROCKET_LOG_STDERR=1` prints an `[moe-int8] ingesting experts` tick every 256, so the stall is
visible rather than looking like a hang. The total is reported at teardown.

**It does not contaminate `llama-bench`.** That tool's warmup is a *full* prompt run, so the ingest
lands there and the reported t/s is clean.

What it does inflate is wall clock. `llama-bench` builds a fresh `llama_context` per test row,
and the expert cache lives on the backend instance. The ingest is therefore paid **once per row**
rather than once per process. A long-running host such as `llama-cli` or `llama-server` pays it
once.

The route is gated bit-faithful against the CPU backend at the primitive level (`test-rocket-moe`, with
outlier-channel activations: MXFP4 cosine 0.9999, Q4_K 0.9999, Q8_0 0.9999).

**Measured end to end on gpt-oss-20b (MXFP4), `-b 2048 -ub 2048`, 600 MHz, clock pinned, governor
`performance`, one board and one `.so` throughout:**

| test | CPU | NPU, `ROCKET_MOE=0` | **NPU, default** | NPU, `ROCKET_MOE=1` |
|---|---:|---:|---:|---:|
| pp512  | 12.17 | 14.08 | **22.0 (1.81x CPU)** | 24.81 (2.04x) |
| pp2048 | 11.80 | 13.62 | **28.1 (2.38x)** | 34.01 (2.88x) |

The default column is the **mean of six runs**, spanning 20.5-22.8 at pp512 and 26.9-28.7 at
pp2048. Placement was identical each time: 63 of 72 expert stacks, 1656 experts, **0 streamed**.

It is quoted as a mean because one run of this arm landed at the bottom of that spread, and
briefly read as a regression. The `ROCKET_MOE=0` control was stable to ±0.05 across the same
rebuilds. That is what identified the spread as this arm's own rather than the board's.

At the llama.cpp **default `-ub 512`** the default reads **22.2 (21.7-22.7) and 22.5
(22.4-22.8)**. That is against 14.13 and 13.53 with the experts on the CPU, about **1.6x and
1.7x**.

The route does *not* collapse at a small micro-batch. That tax belonged to the fp16 streaming
route's per-micro-batch dequant, and native-quant has none to multiply. It does give up ~20% at
pp2048 against `-ub 2048`, so `-b 2048 -ub 2048` is still the setting to run it at.

**The default is deliberately not the peak.** The pre-flight reserves all `n_expert` experts of a
stack, where the lazy admission it front-runs charges only the ~82% the router actually exercises.

So the default takes 63 of 72 stacks, 100% resident and **0 streamed**, and `ROCKET_MOE=1` takes
all 72. That is worth 8% at pp512 and 16% at pp2048.

The trade is **sign for peak**. On a board with RAM to spare the peak comes back through
`ROCKET_MOE_CACHE_MB`, which keeps the zero-streamed property. That holds on a model whose experts
are not most of its weight. Raising it is a loss on one that is, in the knob's own row.
`ROCKET_MOE=1` does not keep the property.

**Residency is the route.** The win is conditional on nearly the whole expert stack fitting. At
99% resident the numbers above hold. At **82%** resident the same route reads 12.19 at pp512,
*below* the 14.11 that leaving the experts on the CPU gives.

That is a cliff rather than a gradient, and the reason is the cost structure. A streamed expert
keeps paying a weight dequant that is **independent of its row count**. A resident one pays a GEMM
that shrinks with `M`. So the streamed remainder's share of the wall clock *grows as the prefill
shortens*: 12% of pp2048, and 43% of pp512. **A residency percentage is not a cost
percentage.**

**But "any streaming is a cliff" is too strong**, and it was the premise the pre-flight was
designed against. Measured on this board, `ROCKET_MOE=1` at **91% resident was the fastest arm of
all**. At high residency, streaming a few experts is *cheaper* than CPU-placing whole stacks.

The crossover sits between **52% and 82%**. At an induced 12 GB budget the same forced arm falls
to 52% resident. It reads **13.63 at pp512 against a 14.08 experts-on-CPU baseline**, a net loss.
The default's 30 fully-resident stacks read **17.02 (1.21x)** there. That is what the pre-flight buys,
and it is the whole reason the default is bounded below at every budget.

Under the default a stack is reserved before its op is claimed, so the streamed remainder does not
arise at all. Reaching a low residency under the default therefore means a limit no pre-flight can
see ahead of the ingest. The likeliest is an exhausted NPU IOVA window, whose exit is
`ROCKET_N_THREADS`, meaning more fds and more per-fd window, rather than more RAM.

The backend warns below 80% and names which budget bound it. That threshold is conservative rather
than measured at its edge.

**DeepSeek-V2-Lite** is a **Multi-head Latent Attention (MLA) plus MoE** model. It has both an
asymmetric attention, DK=192 ≠ DV=128, and routed experts.

The FA gate accepts DK≠DV as a bit-faithful primitive, so MLA is *engageable* with `-fa`. It is
dispatch-bound and pp-neutral at these lengths, so by default it stays on the CPU.

The routed experts take the native-quant route by default **at long prefill only**, below. This is
the model that showed both size floors.

Below them the NPU takes the large MLA projections, the 2 always-on shared experts, and `lm_head`
as an ordinary `MUL_MAT`. That is a **modest but real** win, *larger* than pure-MoE gpt-oss's
dense-only case, because those projections are substantial. Q4_K_M, 15.71 B with ~2.4 B active,
RK3588 @ 600 MHz, warm:

| | pp512 | pp1024 | pp2048 | tg64 |
|---|---|---|---|---|
| CPU | 20.37 | 19.92 | 19.01 | 7.73 |
| NPU (default) | 24.04 | 24.85 | 23.87 | 7.77 |

Prefill gains **1.18-1.26x**, with the NPU flat at ~24 t/s and the win rising with M as the CPU
declines. It is faithful to the CPU, at differential wikitext PPL Δ −0.26%. This is a base model,
so the absolute ~8.2 is meaningful.

On the **fp16 expert route** the handler drops prefill to **0.25x to 0.59x** the CPU, at 5.05,
7.82 and 11.30 with `-ub 2048`. That is a *bigger* loss than gpt-oss's. DeepSeek's faster CPU and
already-winning default NPU leave more to give up. Its experts are also smaller (`K=2048, N=1408`)
and more numerous (64), so the per-expert dequant and dispatch are a larger share of each one.
That route is never taken by default.

**The native-quant route on DeepSeek, measured, and the reason the row floor exists.**
[HW sweep 2026-08-27, 600 MHz, governor `performance`.]

| test | CPU | NPU, `ROCKET_MOE=0` | NPU, experts on the NPU |
|---|---:|---:|---:|
| pp512  | 19.54 | **26.04** | 19.09, **0.73x**, a regression |
| pp2048 | 18.50 | 21.63 | **27.9 (1.29x)** |

The pp512 row is a 27% regression at **100% residency, 0 streamed**. It is therefore not the
partial-residency mechanism, and a residency check alone would never have caught it.

DeepSeek routes **6 of 64** experts, so 512 tokens give each one ~48 rows. That is below the
64-row granule its GEMM is padded up to. gpt-oss routes 4 of 32, and gets 64 rows at the same
count.

With the granule in place DeepSeek's pp512 returns to **25.9**, the experts-on-CPU number, and
pp2048 keeps its win. Same model, offloaded at long prefill and declined at short.

**And the granule alone was not enough.** Mapping the whole range showed the default still
reading **0.96x at `M_e` = 72 and 0.94x at `M_e` = 96**. It accepted both cells. gpt-oss reads
1.77x at the same `M_e` = 96.

What separates them is the work per dispatch rather than the row count, because DeepSeek's expert
GEMM is 2.88x smaller. See **Two size floors** above.

**Do not read gpt-oss's ratio across to another MoE architecture.** `n_used / n_expert` *and*
`K · N` both decide whether the offload is eligible, and the second is the easy one to forget.

## The RK3576 (second target)

Everything above describes the RK3588. The **RK3576** runs a different route, selected by the
**detected part** (`rocket_hw_current()`) rather than by a knob.

It is the **only** matmul route on that SoC. The RK3588 fp16, int8, int4 and bf16 generators
refuse there by construction, and the RK3576 entries refuse on the RK3588. So an op this backend
claims on that part is claimed by the W8A8 handler or not at all.

**`ROCKET_INT8=1` is mandatory there.** `supports_op` declines every `MUL_MAT` without it, so a
run left at the defaults offloads nothing and says nothing about why. Set it, and confirm with
`ROCKET_LOG_STDERR=1`.

**Build it with `-DGGML_ROCKET_NATIVE_FP16=OFF`.** The default convert kernels target the
RK3588's `armv8.2-a+fp16` baseline.

### The route

The RK3588's int8 entry hands back a raw int32 accumulator and the host applies both scales. The
RK3576's writes **int8**.

On that part any output element wider than one byte poisons the *next* submit, across processes.
The int32 sibling is therefore not a route a frontend can take. What the part
offers instead is a per-output-**column** requant in the DPU epilogue, and the whole route hangs
off supplying its scale. Four pieces, each load-bearing:

1. **Rotate A and B by an orthonormal Hadamard along K.** Mandatory, not a knob: the unrotated
   route is not merely less accurate, it is *chaotic* (a 2% per-column divisor change moved one
   model's perplexity nine orders of magnitude). `ROCKET_INT8_HADAMARD` does **not** gate it here.
2. **Quantize A per tensor and B per output channel.** Per-axis *input* scales are free at this
   interface and better on every per-GEMM norm. They are also 38x worse end to end on one model of
   two. The per-tensor activation scale is therefore the measured choice rather than the lazy one,
   and `ROCKET_RK3576_AROW` is the A/B.
3. **Per-column output requant**, its scale frozen from a short calibration pass and divided by a
   safety factor (`ROCKET_RK3576_NCAL` / `CALSAFE` / `BOOTMARGIN`).
4. **Host dequantize by the scale the entry was *asked* for**, not the gain its integer ramp
   delivered. The two differ by the entry's `worst_rel_err`. De-quantizing by the achieved gain is
   a measured wash, and is deliberately not done.

Composed, that measures **1.008x / 1.020x wikitext-2 perplexity against fp32** on Qwen2.5-1.5B and
SmolLM2-1.7B, against 1.11x / 2.26x for the per-tensor output scale it replaces.
[host arithmetic over two models, 2026-08-10]

Taken off the device inside a model, the composed route reads **1.011x** on Qwen2.5-1.5B: 9.8969
against the F16 CPU arm's 9.7872 over eight wikitext-2 chunks. **That is not the same quantity as
the 1.008x above, and the two must not be diffed.**

The simulator figure is against fp32 with an exactly frozen colmax. This one is against the same
GGUF's F16 CPU arm, with the shipped bootstrap at `CALSAFE`=3.0. The eight chunks carry their own
+-0.59 error bar. What it settles is that nothing on the composed device route costs a model
an order of magnitude.
[HW sweep, H96 MAX M9, llama.cpp b10356, Qwen2.5-1.5B F16, 2026-08-11]

**Shape contract**, stricter than the RK3588's: `K%32`, `N%32` (not `N%16`), `K >= 64`, `N >= 32`,
2D static weight, `M >= ROCKET_MIN_M`, and a chunking of `K` the entry will take (see
`ROCKET_RK3576_KSPLIT`). **`M` carries no constraint** on this part, and `M=1` computes.

That is the opposite of the RK3588, where rows are the convolution's spatial height and a height
below 4 mis-computes. There is deliberately no `M%4` padding here.

### Performance, and the denominator that matters

Qwen2.5-1.5B at `pp2048` under stock llama.cpp, `-ub` 512, four threads pinned to the A72s,
governor `performance`. Attention is on the CPU as below, and three processes an arm were
interleaved in one boot:

| Arm | t/s | Against the port |
|---|---:|---|
| The port | 13.36 | |
| ggml's Q8_0 kernel | 10.18 | 1.31x |
| The same F16 GGUF on the CPU | 5.64 | 2.37x |

Both numbers are real and they answer different questions. The 2.37x is what a drop-in user
reads, because the port consumes an F16 GGUF. The 1.31x is against the strongest CPU arm on the
same silicon. Quote them together. Per-pair ratios are 1.30, 1.34 and 1.30.
[HW sweep, H96 MAX M9, 7.1.7 / `rocket` 1.6.0, llama.cpp b10558, 2026-08-23]

**A `pp512` figure on this part prices the calibration path, not the route.** At that length the
route has not converged: a `pp512 -r 1` run reports 332 of 332 calls as calibration forwards. The
11.78 t/s it gives, and the 2.01x / 1.02x pair that follows from it, are a calibration-regime
measurement and are not the route's speed.
[HW sweep, H96 MAX M9, llama.cpp b10356, 2026-08-11]

Two operating notes. The governor is worth **16%**.

And run-to-run spread is a function of prompt length rather than a property of the route. At
`pp2048` it is **±0.02 t/s within a process and 3.5% across processes**, so one process is a
measurement. At `pp512` the NPU arm spreads **±1.8 t/s, 15%**, against ±0.01 on both CPU arms, and
one process is not.

### Attention stays on the CPU there

`FLASH_ATTN_EXT` is **declined** on the RK3576. The handler's only compute call goes through
`rocket_matmul_fp16`, which refuses on any profile that is not `rk3588`. The attention primitive
has no encoder for this part. Claiming the op would hand the scheduler something only the
single-threaded host reference can compute, slower than the CPU backend's own kernel.

The gate declines instead. That is what `ROCKET_FLASH_ATTN=0` did by hand, and is the
configuration every number above was measured under. Nothing to set.

This is worth knowing because of how it used to fail. The gate had no RK3576 branch, and a prompt
of 512 kept `n_kv` under the offload floor, so the whole W8A8 corpus ran. Anything longer returned
`llama_decode ... res = -3`, with the explanation swallowed by `llama-bench`'s no-op `ggml` log
callback.

Read a bare `res = -3` under a bench harness as "a message you are not being shown", and reach for
`ROCKET_LOG_STDERR=1`.

### RK3576 knobs

All twelve are RK3576-only and no-ops elsewhere. The defaults are the shipping route.

The rest are A/B and instrument knobs, and the last two of the accuracy group change what the
model computes.

| var | default | meaning |
|---|---|---|
| `ROCKET_RK3576_NCAL` | 2 | calibration forwards before the per-column output scale is frozen. Each is two device passes: pass one estimates the per-column accumulator maxima from an analytic bound, pass two refines at that scale and returns the surface. Clamped to >=1, since a value below that would freeze an all-zero scale vector and divide by it |
| `ROCKET_RK3576_CALSAFE` | 3.0 | divisor applied to the frozen column maximum. 3.0 rather than 2.0 because the frozen maximum is *bootstrapped*: a bootstrap's 0.3% per-column error costs SmolLM2-1.7B 0.018 of perplexity ratio at 2.0 against 0.0021 at 3.0, while the exactly-frozen arm moves only 0.0017 between them |
| `ROCKET_RK3576_BOOTMARGIN` | 1.5 | slack on the calibration passes' own output scale. The returned surface is correct at a scale this much loose, which costs resolution and nothing else |
| `ROCKET_RK3576_AROW` | 0 | per-**row** activation scale instead of the per-tensor one. Free at this interface (the row scale never crosses the ioctl; it is applied in the dequantize) and better on every per-GEMM norm, and **38x worse end to end on one model of two**, which is why it is off. `=1` everywhere, `=2` only on the weights the K-split cut. Measure a candidate model; do not read a norm |
| `ROCKET_RK3576_KSPLIT` | **off** | cut a `K` past the library's `K >= 6176` refusal into chunks on the host, one whole W8A8 GEMM each, f32 partials summed. Fast and exact, and it costs Qwen2.5-1.5B **1.60x** wikitext-2 perplexity, because what it reaches is `ffn_down` and the route's per-tensor activation scale does not carry that input. More chunks is worse, not better (ten ways scores 18.22 against four ways' 15.84). Off, an FFN down-projection is declined and runs on the CPU |
| `ROCKET_RK3576_KCHUNK` | auto | move the head of the chunk-size preference list (2304, 2048, 1536, 1024, 512, 256, 128, 64, 32) so a candidate depth can be validated without a rebuild. Only 2304 / 2048 / 1536 were measured clean; a model whose chunking lands elsewhere should have that chunk timed before its numbers are quoted |
| `ROCKET_RK3576_FORCESPLIT` | off | cut a `K` the part would take whole. The arm that separates a defect in this code from a property of the route: at `K=1536` the rotation is bit-identical either way, so anything the output does under it is this code's doing. Use with `KCHUNK` |
| `ROCKET_RK3576_WSA` | on | hand the entry this weight's cached per-column `sum_k \|qB\|` instead of letting it recompute the O(K·N) pass every call. Bit-identical by construction, a cost knob rather than an accuracy one; `=0` prices the pass |
| `ROCKET_RK3576_WDEV` | on | once a weight's calibration converges, pack it into a device BO and hand the entry that, then drop the host `qB`. The cube is the same bytes, so this is memory-neutral; it deletes the per-call cube memset, copy and weight-BO churn. `=0` restores the per-call pack for pricing. Latches off for the whole context if one create fails |
| `ROCKET_RK3576_CALSCAN` | 2 | which form of the calibration column scan runs: `0` column-outer, `1` row-major with an `int` accumulator, `2` row-major with a byte one. Exists to price them: `2` is ~3.5x `0`, flat in `N`, because a byte lane processes 16 elements where an `int` lane processes 4 |
| `ROCKET_RK3576_CALROWS` | 0 (all) | cap the row count of calibration pass **one**. An instrument, not a policy: pass one's estimate is a maximum over `M` accumulator rows, so cutting rows undershoots it. Pass two always runs at full `M` and its surface is what the call returns |
| `ROCKET_RK3576_CALMAP` | off | write a per-column CSV readout of what the two calibration passes learn (`key,chunk,fwd,M,Kc,N,n,abound,est1,est2,sat,rmsA,rmsB`), enough to score a host-side predictor of the column maximum offline. Costs an O(M·Kc + N·Kc) sweep per forward; keep it out of a timed arm |

`ROCKET_MM_PROFILE` covers this route too, through a third accumulator. That is the library's
`sumabs / packA / packB / coeff / gen / alloc / stamp / submit / wait / read`, plus the terms
outside the entry: `valloc / quant+rot / calibrate / entry / dequant`.

Both sum to the **offloaded path** rather than to the prefill wall. Attention, the norms, rope and
the scheduler are in neither.

## Why quantization does not speed prefill

The dispatch floor is why. The fp16 prefill in the benchmarks is a stack of operating-point wins
over the 200 MHz boot clock:

| Step | Gemma-4-12B pp2048 |
|---|---|
| 200 MHz boot clock | 7.98 t/s |
| 600 MHz (×1.43) | 11.40 |
| NPU-side fp16 K-accumulation (`ROCKET_KACC`, +19%) | 13.38 |
| CBUF DATA_REUSE (+7%) | ~14.5 |
| Resident weights (+6%) | ~15.1 |

KACC and DATA_REUSE are the default operating mode, and resident weights are opt-in. What that
stack does *not* include is quantization.

**At the current operating point, quantization does not speed up prefill.** Resident matmul is
**~460 GOP/s across precisions**: fp16 461, int8 386, int4 413 GOP/s. The NPU runs at ~15% of MAC
peak, so it is **DMA/dispatch-bound rather than MAC-bound**, and int8's 2x and int4's 4x MAC do not
express.

In-model **resident int8 prefill is 0.60x fp16**. Its int32 readback cannot be K-accumulated,
because the NPU's eltwise operand DMA is <=16-bit.

So quantization's payoff *today* is **RAM, model size and decode-coexistence**, not throughput.
Treat this as **bottleneck-conditional rather than permanent**. Quant's MAC advantage is gated
behind the dtype-independent dispatch floor, which has not yet been attacked. It is a later-stage
lever rather than a dead one.

*Today's* production path is therefore **fp16 with KACC and DATA_REUSE**, ~14.5 t/s, optionally
with resident weights for prefill at ~15.1. int8 and int4 are there for when a model must be
quantized to fit memory. Whether they can also win on prefill speed is gated by the
dtype-independent dispatch floor rather than by the datatype.

### Quantized prefill is micro-batch-sensitive

A quantized GGUF dequantizes to fp16 **per micro-batch**, so the llama.cpp default `-ub 512` costs
throughput. **`-b 2048 -ub 2048` recovers 0.91-1.53x**: dense 1.18-1.53x above 8 B, and 0.94-1.13x
under 4 B, with MoE per its own row. Measure it, because two of eleven models lose. A 9B `Q4_K`
goes 19.1->27.3 t/s.

That per-micro-batch dequant is itself **multi-threaded by default**, through
`ROCKET_DEQUANT_THREADS`, auto being cores capped at 8, with 1 for the serial path. The weight rows
are independent, so fanning them across the CPU recovers **+33% at `-ub 512` and +19% at
`-ub 2048`** over the serial decode. It is bit-exact and needs no extra RAM (9B `Q4_K`, same-session
HW A/B).

Quant *type* is then irrelevant to NPU throughput, and `Q4_K`, `IQ4_XS` and `Q8_0` converge. Quant
closes toward F16 with M, from ~0.64x at pp512 to ~0.85x at pp2048, because F16 stays resident with
no dequant.

**`ROCKET_QUANT_RESIDENT=1` closes that gap.** It dequantizes and packs each weight to resident
fp16 NPU BOs **once**, so prefill then pays neither the per-micro-batch dequant nor the per-call
packB. It reuses the F16 prepacked path and is bit-identical on PPL. That lifts quant prefill to
**F16 parity** and recovers the `-ub 512` halving:

| Model and arm | pp2048 |
|---|---|
| 0.8B `Q4_K` resident | 89 t/s, equal to F16's 89 |
| 0.8B `Q4_K` streaming, `-ub 2048` | 81 |
| 0.8B `Q4_K` streaming, `-ub 512` | 51 |
| 9B `Q4_K` resident | 24.6, ~0.92x F16 |
| 9B `Q4_K` streaming | 15.8 |

So it is **1.5x** at the llama.cpp-default `-ub 512` on the 0.8B, and **1.56x** on the 9B. The
per-call `packB` collapses from 12440 to 889 ms. The 9B's ~18 GB fp16 fits the 31 GB RAM and the
IOVA window.

The cost is the full fp16 resident footprint, which negates the quant RAM saving. It is therefore
opt-in for the RAM-to-spare case. Over `ROCKET_CACHE_MB`, or once the NPU IOVA window fills,
weights fall back to streaming.

On a model larger than the default 2 GB budget, use **`ROCKET_QUANT_RESIDENT=auto`**, whose budget
is sized from free RAM, so it residents in full. A blanket `1` at the default budget residents only
part and can **net-lose to streaming**. 9B `Q4_K` pp2048 at ub2048 reads 19.3 against 22.1 t/s.
`auto` reads 25.8 there, about F16's 25.1. The win grows at the llama.cpp-default `-ub 512`, where
streaming re-dequants 4x: `auto` 26.1 against streaming 15.3, **1.7x** [HW sweep, 600 MHz].

Short quant prefills below `ROCKET_MIN_M_QUANT`, around the measured ~360-row crossover, auto-route
to the CPU.

Across models, the **F16 NPU prefill win grows with size then plateaus**, from 0.8B 1.44x to 9-12B
~3.6x CPU. That is validated on Qwen3.5 and Qwen3.6, including the 27B hybrid-DeltaNet, and on
Gemma-4-12B at 3.6x on pp512, all PPL-faithful.

### What int8 would have to change

Quantized int8 does not speed this backend's prefill. That is about the datatype's role here rather
than a hardware mode the backend is missing.

int8 helps prefill in two ways only. It cuts operand *bandwidth*, where a W8A8 scheme halves both
operands' bytes and this path quantizes weights while keeping fp16 activations. And it batches more
work per kernel submit, lowering the dispatch floor.

Neither lifts the MAC ceiling. The matmul is DMA/dispatch-bound rather than MAC-bound. Throughput
therefore rises by attacking the dispatch floor, meaning fewer, bigger, batched submits, rather
than by narrowing the datatype.

On-device integer K-accumulation would let int8 keep its narrow weights *and* avoid the int32
readback. It does not exist on this silicon.

## Recommended configurations

The performance datapath is default-on: fp16 K-accumulation (`ROCKET_KACC`), CBUF DATA_REUSE
(`ROCKET_REUSE=2`), asymmetric tiling (`ROCKET_MM_ASYM`), threaded dequant (`ROCKET_DEQUANT_THREADS`),
and attention offload (`ROCKET_FLASH_ATTN`) all engage without configuration. Set the backend up as in
[Build and run](README.md#build-and-run) (600 MHz clock, `sudo -E`, absolute `GGML_BACKEND_PATH`), then
add opt-ins by workload. Each residency lever trades RAM for speed and falls back to streaming if the
budget does not fit, so it is safe to request.

| Workload | Precision | Add | Effect |
|---|---|---|---|
| Interactive chat, short prompts | any | nothing; pick `Q4_K_M` for decode speed | Prefill is below the offload floor (`ROCKET_MIN_M`); the turn is decode-bound, and quant sets the stream rate |
| Agentic / RAG / long prompts | quantized GGUF | `-b 2048 -ub 2048` | A quant GGUF re-dequantizes per micro-batch; `-ub 2048` ~doubles prefill over the `-ub 512` default |
| Agentic / RAG, repeated prefill | quantized GGUF, fp16 fits RAM | `+ ROCKET_QUANT_RESIDENT=auto` | Dequant + pack once -> fp16 prefill parity (~1.5x). Needs ~the fp16 model size free |
| Agentic / RAG, repeated prefill | F16, fits ~2x RAM | `+ ROCKET_F16_RESIDENT=auto` | Pack weights once across turns; single-digit-percent gain |
| Any | MoE (gpt-oss, …) | `-b 2048 -ub 2048` | Routed experts resident as int8, on by default: ~2.4x at pp2048 on gpt-oss. A stack that will not fit the resident budget, or whose per-expert row count is under the tile granule, is left on the CPU by the placement gate, so there is nothing to set. `ROCKET_MOE=1` is 13-18% faster where the stack nearly fits and *below* the experts-on-CPU baseline where it does not |
| Model too big at F16 | n/a | a `Q4_K_M` GGUF, or `ROCKET_INT4=1` from an F16 GGUF | Footprint, not speed; quantization does not speed prefill here |

`ROCKET_INT8`, `ROCKET_INT4` and `ROCKET_BF16` are numerically faithful, and they tie the prefill
throughput of fp16. Use them to fit a model in less RAM, never to speed prefill.

Confirm the intended mode engaged by running once with `ROCKET_LOG_STDERR=1`. llama-bench
otherwise prints no backend lines.

## Runtime knobs

The backend reads a set of `ROCKET_*` env vars. `sudo` strips the environment, so always use
`sudo -E`.

A set-but-empty var is treated as **unset** and takes the default. `ROCKET_MIN_M=` is the shape a
wrapper produces when it forwards an unset `$VAR`. It does not parse as `0`. That would collapse
a routing threshold to its clamp floor, or move a knob whose default is non-zero to a different
arithmetic.

The one exception is the presence-only diagnostics: `ROCKET_DEBUG`, `ROCKET_MM_PROFILE`,
`ROCKET_AB`, `ROCKET_DUMP` and `ROCKET_FA_TIMING`. They are documented as "set to anything to
arm", so an empty value arms them. None of them changes a result.

| var | default | meaning |
|---|---|---|
| `GGML_BACKEND_PATH` | unset | path to `libggml-rocket.so` (how the host loads it) |
| **`ROCKET_KACC`** | **on** | **the operating mode**: fp16 NPU-side K-accumulation (+19%). Default-on; `=0` (or `ROCKET_NO_KACC`) opts out to the byte-exact host fp64-accum path. Under it, `ROCKET_REUSE` defaults to 2 |
| **`ROCKET_REUSE`** | 2 (KACC on by default) | CBUF operand reuse: 0 off / 1 WEIGHT_REUSE / 2 DATA_REUSE (+7%). Defaults to 2 whenever K-accum is on, which is the default |
| `ROCKET_N_THREADS` | 5 | worker count (knee ~5; "one above #cores") |
| `ROCKET_MIN_M` | 128 | min M to offload. Below the crossover the per-call dispatch + weight packing (a weight only goes resident at `max_tile`=256, so below that it re-packs every call) outweigh the NPU's per-row advantage and the offload **loses to the CPU**. Measured F16 ratio NPU/CPU, 0.8B: pp16 0.36 / pp64 0.83 / pp96 1.04 / pp128 1.15; 3B: pp64 1.03 / pp128 1.60; 8B: pp48 0.93 / pp64 1.24 / pp128 1.89. The crossover is nearly model-independent (the packB you pay and the compute you gain both scale with K·N, so it cancels); the residual drift (~86 rows at 0.8B down to ~55 at 8B) is the dispatch term, which does *not* scale with K·N and so weighs more when the weights are small. **128 is at or above every measured crossover, so no model regresses below CPU**; bigger models cross earlier still. Also covers **batched** decode: whisper.cpp's default beam search presents M=5 per step, which the old floor of 4 wrongly offloaded (2.3x slower; a 1.40x net loss end-to-end). Do **not** set it to 256: llama streams below 256 and still wins, so 256 costs pp128 −46% |
| `ROCKET_MIN_M_QUANT` | 512 | min M to offload **quantized** weights (the dequant->fp16 path). Its per-microbatch dequant needs more rows than F16 to amortize; measured crossover ~360 on 9B/27B `Q4_K`, so short quant prefills below this stay on the CPU (avoids a net-loss offload). Floored at `ROCKET_MIN_M` |
| **`ROCKET_QUANT_RESIDENT`** | off | dequant a quantized GGUF weight to fp16 **once** and hold it in resident NPU BOs (reuses the F16 prepacked path) instead of re-dequantizing **and** re-packing it every micro-batch, lifting quant prefill to F16 parity (closes the per-microbatch dequant tax, and the `-ub 512` / short-follow-up cases `-ub 2048`'s amortization can't) at the cost of the **full fp16 resident footprint**. Modes: `1` = blanket, bounded by `ROCKET_CACHE_MB` (default 2 GB); **`auto`** = size the budget from free RAM (MemAvailable − a swap-safe reserve `max(6 GiB, 30 % of RAM)`, override `ROCKET_QUANT_RESIDENT_RESERVE_MB`); `N` = an explicit N-MB budget. Prefer `auto` for the "quantized-for-download, RAM-to-spare" case: on a model **larger than the default budget**, blanket `1` residents only part of it and is a **net loss vs streaming**; `auto` reaches parity (Qwen3.5-9B-`Q4_K` pp2048: `auto` 25.8 ~ F16 25.1 vs streaming 22.1 t/s [HW sweep, 600 MHz]). Over-budget / IOVA-full weights fall back to streaming |
| **`ROCKET_FLASH_ATTN`** | on | offload the attention op (`FLASH_ATTN_EXT`) to the NPU: heads fanned across the worker fds + submit-chained (see `ROCKET_FA_CHAIN`), bit-faithful (PPL == CPU) **for the attention it implements**: `softmax(scale·QKᵀ + mask)·V`. An op carrying **attention sinks** (`src[4]`, a learned per-head logit in the softmax denominator: gpt-oss, `mimo2`, `deepseek4`) is **declined**: the handler has no sink term, so accepting it would compute a *different* attention, silently. **Parity at <=1K, a growing win above: 1.07x at 4K, 1.50x at 8K, 1.25x at 16K** [HW sweep, `-r3`]. Context-gated (see `MIN_KV`); `=0` disables. A driver failure degrades the layer to the host reference rather than failing the graph, as `MUL_MAT` and `MUL_MAT_ID` do: correct but slow, and it logs a warning saying so |
| `ROCKET_FA_CHAIN` | on | batch each worker's per-head QK (and AV) submits into one NPU job through a resident batched-matmul context, the dispatch-floor lever that pulls the crossover in to ~2K; `=0` forces the per-head path. Bit-identical |
| `ROCKET_FLASH_ATTN_MIN_KV` | 1024 | min `n_kv` (context length) to offload an attention op; 1024 ~ the sliding-window length, so the windowed local layers offload while shorter prompts stay on CPU. Gates on `n_kv`, not `n_tokens` (every ubatch has `n_tokens~512` regardless of total context) |
| `ROCKET_FLASH_ATTN_MIN_T` | 16 | min prefill `n_tokens` to offload attention (single-token decode stays on CPU) |
| `ROCKET_ATTN_HOST_SOFTMAX` | host (FA) | attention softmax placement; `=0` forces the on-NPU softmax (default host, since scores are already host-side for the additive mask) |
| `ROCKET_FA_TIMING` | off | print the FA handler's host split at exit: gather / on-NPU compute / scatter ms (the `FLASH_ATTN_EXT` outer gather is host glue the driver's `ROCKET_MM_PROFILE` does not see). Diagnostic only; near-zero when off |
| `ROCKET_MOE` | **auto (on)** | MoE routed-expert FFNs (`MUL_MAT_ID`): bucket the `(slot,token)` rows by expert id, run each active expert's `[M_e,K]x[N,K]ᵀ` GEMM fanned across the worker fds, scatter the rows back. **Three states.** *Unset* = auto: claim the op only for a **GGUF-quantized** stack, on the RK3588, whose whole `[K, N, n_expert]` stack the residency pre-flight can reserve before the first ingest **and** whose per-expert GEMM clears both the tile granule (`ROCKET_MOE_M_BUCKET`) and the per-dispatch work floor (`ROCKET_MOE_MIN_WORK`); that is the native-quant route (`ROCKET_MOE_NATIVE`), which holds each expert resident as int8 and is worth **~2.4x the CPU** at pp2048 on gpt-oss-20b MXFP4 (~1.8x at pp512). *`=1`* = forced: claim every `MUL_MAT_ID` the handler can compute, reserving nothing and ignoring both size floors, which includes the fp16 streaming route (an F16 stack, or a stack too large to hold resident), which measured 0.42-0.90x. Forced is **6-13% faster at pp512 and 18-21% at pp2048 where the stack nearly fits** (measured twice; the pp512 half is inside that arm's own spread) and **below the experts-on-CPU baseline where it does not** (0.97x at pp512 on an induced 12 GB budget), and `ROCKET_MOE_CACHE_MB=28000` buys back 79% of the pp2048 gap on a 31 GiB board while keeping the sign guarantee forced gives up; it is the A/B arm and the configuration the archived MoE measurements were taken under. *`=0`* = off. Run at **`-b 2048 -ub 2048`**, not because the experts re-dequantize (native-quant is what stops that) but because the **dense** quantized weights still do, and because a smaller micro-batch gives each expert proportionally fewer rows (64 at `-ub 512` vs 256 at `-ub 2048`) while the per-expert dispatch/gather/scatter/padding stays flat. Measured at `-ub 512`: ~1.6x/1.7x over experts-on-CPU, i.e. **no collapse**; that tax belonged to the fp16 route. Costs a one-time expert ingest per `llama_context` (~36 s on gpt-oss-20b, ~32 s on DeepSeek-V2-Lite; the dominant NPU-BO pack term is bytes-bound at ~500-545 MB/s, not per expert). See the MoE note above |
| `ROCKET_MOE_MIN_TOKENS` | 512 | in every `ROCKET_MOE` state, the min micro-batch `n_tokens` for a `MUL_MAT_ID` op to offload; `M_e ~ n_tokens · n_expert_used / n_expert` sets the per-expert GEMM size; short prefills and decode stay on the CPU. Floored at `ROCKET_MIN_M` |
| `ROCKET_MOE_NATIVE` | **on** | route a **GGUF-quantized** expert through the resident int8 group-wise path: ingest its quant blocks **once** into int8 codes held in NPU BOs, then quantize only the activation per call. This is what removes the per-micro-batch host dequant that makes the fp16 expert route a loss, and it is the only route the default placement claims; `=0` forces the fp16 dequant route (the A/B baseline) and leaves auto placement with nothing to claim, so it takes effect under `ROCKET_MOE=1`. An **F16** expert always takes the fp16 route, since it has no dequant to delete |
| `ROCKET_MOE_CACHE_MB` | **auto** | resident native-quant **expert** budget in MB (`0` = unlimited). Auto is `MemAvailable` minus the reserve. **This is the knob that buys back what the pre-flight's conservatism costs**, and unlike `ROCKET_MOE=1` it keeps both the zero-streamed property and the sign guarantee. Measured ladder, gpt-oss-20b on a 31 GiB board, every arm 0 streamed, **one `-r 3` process per arm, so every rung is n=1** [HW sweep 2026-08-27]: auto (24672 MB) takes 63 of 72 stacks; **26000 takes 66 (+4.3% / +5.1% at pp512 / pp2048); 28000 takes 71 (+8.3% / +14.1%)**, which is 79% of the `ROCKET_MOE=1` ceiling and above it at pp512. **The stack counts are exact; the four percentages are not.** A single process on this board can sit ~10% off the level its own configuration repeats at, so read this ladder as "more budget places more stacks, and on this model class that direction pays" rather than as those deltas; the pp512 column additionally sits inside a ±1.4-2.1 t/s within-process spread and should not be read finely at all. 28000 leaves only ~2.8 GB of the headroom the 6 GiB auto reserve exists for, so it is a setting for a **known working set**, not a new default; pin it explicitly if you also need placement to be reproducible across processes. **On an expert-dominated model the direction inverts, and raising the budget costs the whole offload.** On Qwen3-30B-A3B (29 of 30.5 B in the experts, 17.28 GiB GGUF) at `-ub` 4096 the placed-fraction curve is a plateau then a cliff, and the auto budget is over the edge: **18000-21000 MB** takes 58-67 stacks and reads **1.046x** the experts-on-CPU baseline (pooled over **seven** processes), auto (24425) takes 79 and reads **1.014x** (**n=3**, rep sd ±0.6), and 28000 takes 90 and reads **0.999x** (**n=1**, rep sd ±1.02), with per-rep spread rising from ±0.00 to ±1.02 as placement grows [HW sweep 2026-08-28, RK1, 600 MHz]. **Only the plateau-to-auto step is deep enough to quote as a size**: the plateau's n=7 against auto's n=3 is the 3.2% gap. The rungs at each end are one process each -- 12000 MB (38 stacks, 1.032x) and 28000 MB -- so 28000's 0.999x supports "one process read parity", not "at 90 stacks the offload buys nothing", and the two single-process rungs are the ones taken where the per-rep spread is widest. The cause is that the admission charge counts the GGUF source bytes of the experts it *places* but not of the ones it leaves on the CPU, which are read from the same mmap every micro-batch and are equally unreclaimable, so the true hot set crosses RAM before the budget says it has. **THE KNOB'S UNITS ARE NOT BYTES OF RAM**, and that is why a recommendation cannot be carried to another quant or another board by arithmetic on RAM alone. What it bounds is the admission charge, which per expert is `N·K` int8 code bytes + `N·(K/group)·4` scale bytes + that expert's GGUF source stride -- so a budget buys less residency than it names, by a **charge factor** `1 + 4/group + source_bits_per_weight/8`. On Qwen3-30B-A3B Q4_K_M (4.87 bits/weight, `K`=2048 -> group 512) that is **1.617**; the same physical placement would need about **2.07** typed in for a Q8_0 MoE and **1.54** for IQ4_XS **[expected -- derived from bits/weight, not measured]**. **You do not have to compute it for your own model: the route prints it.** The pre-flight line `resident budget reached after N expert stacks (X MB RAM, Y MB IOVA)` has the charge and the codes side by side, so `X/Y` is the factor with nothing inferred -- **1.610** on Qwen3-30B-A3B and **1.521** on gpt-oss-20b MXFP4, against 1.617 and 1.538 derived. So the formula is good to ~1% for sizing a budget **before** a run, and the log is what to read **after** one. So convert rather than copy: `budget_MB ~ (1 + source_bits_per_weight/8) * (MemTotal − the whole expert GGUF − runtime headroom)`, which on this board is `1.610 × (31.7 − 17.3 − 1.2 GB)` = 21.3 GB and lands on the measured plateau. That is a consistent **form**, not an independent prediction: the headroom term is its one free parameter and it was chosen to land there. **Read a `ROCKET_MOE_CACHE_MB` recommendation as scoped to the model class, not to the board size**: raise it on a model whose non-expert weights dominate, and on an expert-dominated one *lower* it, to roughly `MemTotal` minus the whole expert GGUF minus the reserve **converted through the factor**. The source bytes are in the charge because the GGUF is mmapped and cannot be reclaimed (MoE **decode** reads the active experts from it on the CPU every token), so both copies must coexist. Read by the placement pre-flight as well as by admission, so the decision to claim an op and the charge taken when its experts are ingested are against the same number. Admission-only: an expert that does not fit streams on the fp16 route, correctly, and the split is logged at teardown (`ggml_backend_rocket_moe_stats`); under the default that split should read 100% resident, because an unreservable stack was never claimed |
| `ROCKET_MOE_GROUP` | **auto** | the K-group the native-quant path quantizes on. Auto picks the largest divisor of `K` that is a multiple of 32 and that the CBUF can hold as one K-tile, the readback floor (gpt-oss `K=2880` -> **576**, `nKt=5`). Readback scales as `K/group` and this path is readback-bound, so a finer group is more faithful and proportionally slower; the knob exists for that A/B |
| `ROCKET_MOE_M_BUCKET` | 64 | the FLOOR of the ladder the ragged per-expert row count `M_e` is rounded up onto (rounded up to a power of two). Not a tuning knob so much as a requirement: `M%4` is a hardware contract, and the driver caches its resident scratch per `(M,K,N,group)` in a fixed 32-slot table that a distinct `M` per expert would exhaust mid-prefill, after which every remaining expert degrades to the CPU. The ladder is fixed at two rungs per octave from the granule (64, 96, 128, 192, 256, …), so it bounds the distinct slot count by construction (~15 values) and caps padding at ~33% of `M_e` (~15% on average). It does **not** adapt: an earlier adaptive granule that coarsened as the table filled could not un-coarsen (the driver's scratch slots are permanent, so the headroom test never became true again), ratcheted to its 4096 ceiling on the first overflow, and left a 356-row expert computing 4096 rows, 88% padding, silently. Raise the granule only to trade padding for slots |
| `ROCKET_MOE_MIN_WORK` | 340 | the least work ONE expert dispatch must carry, in **mega-MACs** of `M_e · K · N`, for the offload to pay its fixed cost (`0` disables the floor). This is the floor that separates the architectures, and the row floor above cannot do its job: the per-expert cost is paid **per dispatch** while the work a dispatch carries is `M_e · K · N`, so `M_e` is a proxy for it only while `K · N` is constant, and gpt-oss's expert GEMM is 2880×2880 against DeepSeek-V2-Lite's 2048×1408, **2.88x larger at the same row count**. Measured over `-p 512…2048` on both models at 100% residency: `M_e` = 64 reads **1.64x** on gpt-oss while `M_e` = 72 reads **0.94x** on DeepSeek, so no row floor separates them; against work per dispatch the measured cells separate with no overlap. The default is the geometric midpoint of the gap between the marginal boundary cell (2.77e8, 1.049x over eight pairs, which does not repay its own ~32 s ingest until ~16 600 tokens of prefill at that shape) and the first materially winning one (4.15e8, 1.22x). Set it to **240** to take that boundary cell on a workload that really does prefill that much. The form is derived, the number is **fitted on two architectures**, so treat it as a measured boundary, not a bound. A third, **Qwen3-30B-A3B** (`K·N` = 1.573e6, 201 MMAC at `-ub` 2048), confirms the floor's SIGN and not its position: the default reads **1.000x** its own `ROCKET_MOE=0` control and ingests nothing, while `ROCKET_MOE=1` reads **0.834x**, and halving the work per dispatch at matched residency takes that to **0.661x**. 201 is far below 340, so what would move the number is a model landing between 277 and 415 |
| `ROCKET_INT8` (+ `ROCKET_INT8_HADAMARD`) | off | W8A8 int8 path (coherent with Hadamard; net loss vs fp16+KACC on the RK3588). **On the RK3576 this knob is mandatory, not optional**: the W8A8 route is the only matmul route there and `supports_op` declines every `MUL_MAT` without it, and the rotation is unconditional there rather than gated by `ROCKET_INT8_HADAMARD`. See [The RK3576](#the-rk3576-second-target) |
| `ROCKET_INT8_RESIDENT` | off | resident int8 weights (on top of `ROCKET_INT8`) |
| `ROCKET_INT8_CACHE_MB` | 4096 | resident rotated-int8 weight cache cap, measured against the actual resident NPU-BO tile footprint (over-budget weights fall back to fp16) |
| **`ROCKET_INT4`** | off | native W4A4 int4 path (from an F16 GGUF); group-wise + Hadamard on by default. Char-identical to fp16; a RAM play, not faster |
| **`ROCKET_INT4_RESIDENT`** | off | hold the int4 weights in **resident NPU nibble BOs** (~¼ the fp16 footprint, no per-call weight scatter) instead of re-packing per call. Group-wise only (`ROCKET_INT4_GROUP>0`, the default); per-channel / over-budget / IOVA-full weights fall back to the one-shot int4 path |
| `ROCKET_INT4_GROUP` | 128 | int4 K-group scale granularity (must divide K, %32; `0` = per-channel). Smaller = higher fidelity, more readback |
| `ROCKET_INT4_HADAMARD` | 1 | int4 outlier rotation (near-mandatory: cos ~0.79 -> ~0.98 with it). `0` disables |
| `ROCKET_INT4_CACHE_MB` | 4096 | int4 weight budget in MB (caps the one-shot host cache AND the resident NPU-BO total; over-budget weights fall back). A full 12B's int4 resident BOs are ~6 GB, so raise this for full coverage |
| `ROCKET_BF16` | off | the exact fp32-output bf16 datapath. bf16 carries fp32's exponent range, so activations need no per-row scale: A goes through unscaled and the fp32 result is written straight to `dst`. Without it a bf16 weight decodes to fp16 and takes the shared fp16 streaming route (fp16-faithful, similar speed) |
| `ROCKET_BF16_CACHE_MB` | **0 (off)** | host fp32 weight cache for the `ROCKET_BF16` path, in MB (`0` here means off; pass a budget to enable, and see the note). Off by default where every other cache is on, because this one holds **fp32, twice the bf16 weight it came from**, so a blanket cache of a 24 GB bf16 model asks for 48 GB. What it buys is the per-micro-batch BF16->fp32 conversion, which is that route's dominant host term; run `ROCKET_MM_PROFILE` and read the **dequant** bucket to decide whether the RAM is worth it on a given model. Over-budget weights convert per call, correctly |
| `ROCKET_FORCE_PREPACK` / `ROCKET_NO_PREPACK` | unset | force / disable the resident-weights path |
| `ROCKET_PREPACK_MADVISE` | off | madvise the F16 source after packing (**prefill-only; breaks CPU decode**) |
| `ROCKET_CACHE_MB` | unset | resident-weight byte budget in MB (the lower-level knob `ROCKET_QUANT_RESIDENT=auto`/`N` set; an explicit value here wins over both) |
| `ROCKET_NO_FUSE` | off | disable QKV / gate-up graph fusion (per-node routing, for a clean A/B). Fusion also turns itself off under `ROCKET_NO_STREAM`, `ROCKET_FORCE_PREPACK`, `ROCKET_INT8` and `ROCKET_INT4`, because the fused path is fp16-only; leaving it on under a precision knob would run the fusable projections (QKV, gate/up) in fp16 and reach only `o_proj` / `ffn_down` / `lm_head` with the selected precision |
| `ROCKET_NO_STREAM` | off | disable the persistent streaming matmul context (per-call `mt` path instead). Also turns fusion off, which needs it |
| `ROCKET_MOE_COSINE` | off | report a per-op NPU-vs-CPU cosine for each offloaded `MUL_MAT_ID`. Diagnostic |
| `ROCKET_FLASH_ATTN_NO_CTX` | off | force attention onto the per-call `_mt` path (own fds, per-call scratch) instead of the persistent `rocket_fa_ctx`. The A/B for what the resident score-matrix scratch is worth |
| `ROCKET_VERIFY` | off | per-op NPU-vs-CPU correctness gate (`max_abs` is the signal; ignore `max_rel`; very slow); needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_MM_PROFILE` | off | per-phase host+driver bucket breakdown at exit, including the streaming **weight-dequant** (quant/bf16->fp16) bucket, the dominant host term of a quantized-GGUF prefill (adds noise -> drop for headline t/s) |
| `ROCKET_TRACE` / `ROCKET_DEBUG` / `ROCKET_DEBUG_GRAPH` | off | per-op weight/ptr/amax trace; fusion groups; split census (`ROCKET_TRACE` needs `-DGGML_ROCKET_DIAGNOSTICS=ON`) |
| `ROCKET_WAIT_MS` | 8000 | output-fence wait deadline |
| `ROCKET_LOG_STDERR` | off | tee the driver/backend log channel to stderr even when the host silences its `ggml` logger (e.g. `llama-bench` without `-v`); see Diagnostic logging |

The heavy recompute and trace diagnostics are compiled out of the production `.so`:
`ROCKET_VERIFY`, `ROCKET_CPU_FORWARD`, `ROCKET_TRACE`, `ROCKET_AB` and `ROCKET_DUMP`. Build with
`-DGGML_ROCKET_DIAGNOSTICS=ON` to enable them. `ROCKET_MM_PROFILE`, `ROCKET_DEBUG` and
`ROCKET_DEBUG_GRAPH` are always available.

K-accum is on by default, and DATA_REUSE follows automatically. `ROCKET_KACC=0` opts out.

## Diagnostic logging

Every diagnostic the backend emits flows through ggml's log channel: errors, warnings, the
`ROCKET_MM_PROFILE` and `ROCKET_FA_TIMING` breakdowns, and the env-gated traces above. The host,
llama.cpp or whisper.cpp, therefore redirects or silences it with the same `ggml_log_set` callback
it uses for its own output.

Severity maps the obvious way. Failures go to `ERROR`, recoverable fallbacks to `WARN`, the
profile breakdowns and one-time hints to `INFO`, and the per-op traces to `DEBUG`. The env knobs
above still gate whether a trace is emitted at all, and the channel only governs where it goes.

The driver it links, librocketnpu, has its own log channel. The backend bridges it into ggml's at
startup, so a driver message reaches the same host callback rather than landing on raw stderr.

The driver's own threshold applies first. A driver `DEBUG` trace therefore needs both its
driver-side gate (`ROCKET_DEBUG`) and a host logger that prints `DEBUG`.

One consequence is that a host which *silences* its `ggml` logger silences the backend too.
`llama-bench` installs a no-op `ggml` callback unless run with `-v`. Its default run therefore
hides every backend line, including one-shot decisions such as the `ROCKET_QUANT_RESIDENT=auto`
resident-weight budget. That budget changes the benchmarked prefill number, so it is exactly what
you want to see.

For that case the backend routes two groups through the driver channel. The first is its
user-facing config lines: the budget decision, and the "raise `-ub`" quant hint. The second is its
residency outcome lines, `[f16-resident]` and `[moe-int8]`. `ROCKET_LOG_STDERR=1` tees them to
stderr regardless of the host's logger.

Set it alongside a benchmark to confirm which budget or mode actually engaged **and how far it
got**. Those are two different questions, and a benchmark row answers neither on its own.

## Implementation notes

- Uses ggml's private headers, `ggml-impl.h` and `ggml-backend-impl.h`, as expected for
  an out-of-tree backend.

  The vtables, `ggml_backend_i` and the rest, are initialized positionally. The backend
  must therefore be built against the **host's bundled ggml**, matching both field order
  and `GGML_BACKEND_API_VERSION`.

  The concrete ABI this backend targets is **`GGML_BACKEND_API_VERSION 2`**, with the
  device vtable that *includes* the `set_tensor_2d_async` and `get_tensor_2d_async`
  slots. It is built and verified against the in-repo `ggml/` tagged v0.14.0.

  The host apps, llama.cpp and whisper.cpp, clone their own ggml. Build this backend
  against **that checkout's** headers and re-check on every bump. A host
  `ggml-backend-impl.h` with a different `GGML_BACKEND_API_VERSION` or 2d-field layout
  drifts the positional vtable, and you get a "ROCKET device not listed at startup"
  failure.

Two further implementation facts:

- `ggml_fp16_t` (uint16_t) shares the IEEE-half bit layout with `_Float16`, so the
  driver's `_Float16*` buffers are filled via `ggml_fp32_to_fp16_row` and cast.
- **Why offload happens:** ggml's scheduler only pulls a host-resident op onto an
  ACCEL device when **both** `supports_op` and `offload_op` return true (BLAS does the
  same). A NULL `offload_op` makes the device appear at startup but run zero matmuls.
