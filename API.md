# ggml-rocket: API and tuning reference

The exhaustive per-model benchmarks, the full `ROCKET_*` runtime-knob table, the
diagnostic-logging channel, and the backend implementation notes. The [README](README.md)
is the guide; this is the reference.

## Models and precisions

- **Whisper encoder (HW-validated):** stock `whisper-cli` / `whisper-bench` load the `.so`
  via `GGML_BACKEND_PATH` and run the encoder `MUL_MAT`s on the NPU, transcripts
  byte-identical to the CPU (WER 0; encoder-output cosine 0.9998). Encoder latency beats
  the CPU at every size and the win **grows with the model**: `whisper-bench` encode,
  warm, RK3588 @ 600 MHz: tiny.en 1.18x, base.en 1.29x, small.en 1.54x, medium.en 1.84x,
  large-v3 2.14x, large-v3-turbo 2.12x. The decoder (autoregressive, M=1) stays on the CPU
  on both backends. **large-v3-turbo** keeps the full 32-layer encoder but a 4-layer decoder
  (~6x cheaper per step), so the NPU-accelerated encoder is the larger share of its pipeline.
- **Multi-model STT via transcribe.cpp (HW-validated):** the same `.so` drops into transcribe.cpp
  (a ggml-based host for Granite-Speech, Voxtral, MOSS diarize, SenseVoice, FunASR, Parakeet, …) and
  offloads the **encoder**, with the same law as Whisper: encode always offloads (1.2x-2.7x, scaling
  with encoder size x audio length), autoregressive decode is M=1 and stays on the CPU, except the
  batched decode-prefill over the injected audio context on long audio, which offloads when large
  enough. So the win is set by architecture: a **cross-attention decoder** (Granite-Speech) offloads
  decode best (~1.9x on 120 s); a **big decoder-only over long audio** (Voxtral-mini 3B) offloads its
  ~1500-token audio prefill (decode 1.46x, whole-pipeline 1.57x); a **small decoder-only** (MOSS /
  FunASR 0.6B) is decode-bound and gains ~1.0-1.15x despite a ~1.6x encode; a **single-pass CTC**
  (SenseVoice) has no decode (6.2x realtime, 1.26x). Encoder-only offload is bit-faithful CPU-vs-NPU;
  a model whose decoder cross-attn offloads (Granite) can diverge late as fp16 accumulation perturbs
  greedy decoding. **Q8_0 is the STT default** (ties or beats F16 on both backends at half the RAM,
  because these models are less matmul-dominated than LLM prefill so the per-micro-batch dequant tax
  is small). [HW A/B, 2026-07-21]
- **Gemma-4-12B prefill (HW-validated):** coherent generation via the same drop-in on
  llama.cpp. Prefill GEMMs offload, decode stays on CPU.
- **Phi-4 (14B) prefill (HW-validated):** the 14.66 B Phi-4 (arch `llama`, dense GQA, a different
  architecture from Phi-4-mini's `phi3`) prefills on the NPU through the same drop-in, **PPL-faithful**
  to the CPU (Q8_0 / Q4_K_M wikitext Δ −0.09% / −0.31%). Its F16 GGUF (29.3 GB) does not fit a 31 GB
  board, so it runs quantized: **Q8_0 / Q4_K_M ~3.4-3.5x CPU at pp2048** (~12 t/s over a ~3.5 t/s 14 B
  CPU baseline), the win rising with prompt length as the per-micro-batch dequant amortizes. Q4_K_M
  (8.28 GB) is the practical pick: it fits a ~9 GB budget and streams decode ~2.2 t/s. A mid-large point
  between Gemma-4-12B and Qwen3.6-27B.
- **Qwen3.5 / Qwen3.6 prefill (HW-validated):** Qwen3.5 (0.8B/9B, dense GQA) and
  **Qwen3.6-27B (hybrid Gated-DeltaNet linear attention)** prefill on the NPU through the same
  drop-in, **PPL-faithful** to the CPU. The DeltaNet / SSM-scan layers are CPU-only ops and stay
  there; the FFN + projections offload and the prefill win holds; the hybrid's linear-attention
  layers do not block it (**27B Q4_K_M 4.4x CPU at pp2048**, the largest prefill win measured, its
  quant lapping an especially slow 27B CPU baseline). The NPU prefill advantage **grows with model
  size**: F16 0.8B 1.44x -> 9B 3.65x CPU. A quantized GGUF wants `-b 2048 -ub 2048` (see
  Performance / `ROCKET_MIN_M_QUANT`).
- **Small dense models, Llama-3.2-3B / Ministral-3-3B / Phi-4-mini (HW-validated):** the 3-4 B
  dense end (arch `llama` / `mistral3` / `phi3`, GQA) posts the **fastest prefill** through this
  backend: Llama-3.2-3B F16 **61.8 t/s at pp512 (3.4x CPU)**, Ministral-3-3B 57.6 and Phi-4-mini
  55.4, because fewer FLOPs per matmul. The F16 win narrows with prompt length (3.4x->2.4-2.6x by
  pp2048) as a small model's per-op readback grows with M, so it lands hardest on short-to-medium prompts; a
  Q4_K_M fits a 4 GB board (Llama 1.87 GB, Ministral 1.99 GB) and streams decode fastest (~7.7-7.9
  t/s). NPU prefill is PPL-faithful to the CPU (Llama / Ministral F16 wikitext Δ +0.02% / −0.04%).
  The NPU/CPU prefill *ratio* is modest on these because their CPU baseline is comparatively fast:
  the NPU's absolute prefill rate is the headline.
- **Vision-language / multimodal, SmolVLM2-2.2B (HW-validated):** the full pipeline runs on the
  NPU through llama.cpp's `mtmd` path: a **SigLIP-SO400M vision encoder** plus a **SmolLM2-class
  language model** (arch `llama`, 1.81 B; the "2.2B" is the combined count). The LLM half is the
  **fastest in this backend**: F16 **99 t/s at pp512 (3.5x CPU)** and the briskest decode (Q4_K_M
  **14.5 t/s** from a 1.03 GB file), PPL-faithful (F16 / Q4_K_M wikitext Δ −0.07% / −0.71%). The
  vision encoder runs through `clip.cpp`; attach the NPU with **`MTMD_BACKEND_DEVICE`** (clip's
  auto-path only probes GPU/IGPU device types, so the ACCEL device must be named explicitly, else
  the encoder stays on the CPU). Offloaded, the SigLIP encode is **~1.19x faster warm** and
  **faithful at cosine 0.99998** (projected image embeddings vs CPU), a modest win: ggml-rocket
  offloads only the aligned static-weight GEMMs (q/k/v/o projections + fc1), while fc2 (K=4304 off
  the K%32 grid), the attention core, and all norm/softmax/GELU ops stay on the CPU, so the graph
  pays many CPU↔NPU handoffs. A resident whole-encoder path would widen this; the drop-in clip
  route is correct and modestly positive as-is.
- **Quantized GGUF prefill (HW-validated):** a `Q8_0` / `Q4_K` / `Q6_K` / … GGUF runs
  its prefill GEMMs on the NPU directly: each weight is dequantized to fp16 on the fly
  (one [K] row at a time, the exact values ggml's CPU backend decodes), with **no
  whole-model F16 copy in RAM**. So the model stays compact on disk and in memory (e.g.
  Gemma-4-12B `Q8_0` ~12 GB vs the F16 GGUF's ~24 GB) and prefill still takes the fp16
  NPU path; decode stays on the CPU on the quantized weights. Quantization buys
  RAM / model-fit here, not prefill speed (the matmul is dtype-independent; see
  Performance). The dequant is paid **per micro-batch**, so quantized prefill is
  micro-batch-sensitive: run with **`-b 2048 -ub 2048`** for ~2x over the llama.cpp
  default `-ub 512` (which re-dequantizes every 512-row chunk). Short prefills below
  `ROCKET_MIN_M_QUANT` (default 512 rows ~ just past the measured ~360-row crossover)
  stay on the CPU, where they beat a dequant-bound NPU offload. Importance-matrix quants
  (`IQ4_XS`, …) take this same path, since an imatrix is a *quantize-time* construct, so at inference
  they are ordinary dequant-to-fp16 types (`IQ4_XS` is 4.25 bpw with 256-element super-blocks,
  `K%32` holds) whose prefill GEMMs offload like any quant, not a CPU fallback (Qwen3.5-9B
  `IQ4_XS` prefill 3.1x CPU at pp2048, the fastest-decoding rung at 4.80 GB). Requires the host
  built `-DGGML_CPU_REPACK=OFF` (see *Drop-in use with llama.cpp* below).
- **Resident weights:** static F16 weight tensors are packed into resident NPU BOs and
  cached by **weight name** (not `src0->data`: the scheduler reuses pooled copy addresses
  across weights, so keying on the address aliases distinct weights and produces gibberish;
  `supports_buft = is_host`). Holding a weight resident pays its scatter (`packB`) **once** and
  reuses it across every later micro-batch / prefill, instead of re-scattering per call. The cache
  is keyed on the name alone and **not** on the prefill M, so a weight packed at one prefill length is
  reused at any other (M >= the tile cap) with no re-pack (the resident layout is M-independent there).
  A weight is made resident only when first seen at a reuse-worthy M; a smaller one-shot M (a short
  prompt, or the tail ubatch of a long prompt) streams via the per-call mt path, leaving the resident
  weight intact for the big prefills (no re-pack thrash on mixed-length workloads).
  - Default-gated to **K<=2048** weights (whisper's encoder, resident by default).
  - **`ROCKET_F16_RESIDENT=auto`** extends residency to **all-K** F16 weights (the LLM attn/FFN,
    K∈{3072,8192,…}) with a free-RAM-sized budget, the F16 sibling of `ROCKET_QUANT_RESIDENT`.
    **Decode-safe:** no `madvise`, so the GGUF stays mapped for CPU decode. It disables QKV/gate-up
    fusion so every weight goes resident: residenting the fused weights (~⅔ of the `packB` bytes)
    beats fusing them. Llama-3.2-3B-F16 @600 MHz `-ub 512`: pp2048 39.8->42.1 (**+5.9%**), pp512
    54.5->59.6 (**+9.5%**), token-identical to streaming [HW sweep]. The win is **larger at shorter
    prefills** (long-prefill attention stays on the CPU and dilutes the offloaded-matmul share) and
    **larger on combined-projection architectures**: Phi-4-mini (single `qkv`/`gate_up` weights, so
    nothing to fuse) reads **+21% / +16% / +14%** at pp512 / 1024 / 2048. `auto`/`<N>`-MB/`1` modes and the
    reserve mirror `ROCKET_QUANT_RESIDENT`. For F16 models that fit ~2x in RAM, the resident tiles
    duplicate the still-mapped GGUF; a too-large model degrades to partial residency via a runtime
    `MemAvailable` floor (no OOM on a swapless board).
  - `ROCKET_FORCE_PREPACK=1` is the diagnostic force (default 2 GB budget unless `ROCKET_CACHE_MB`
    set) with an optional prefill-only `ROCKET_PREPACK_MADVISE` that reclaims the F16 source (breaks
    CPU decode).
- **int8 W8A8 + Hadamard:** coherent and char-identical to fp16, but a **net loss** in
  prefill speed (see below).
- **int4 W4A4 in-model (HW-validated):** native int4xint4 prefill for a Gemma-4-12B F16
  GGUF via `ROCKET_INT4=1`: weights and activations quantized to 4-bit (symmetric `[-7,7]`)
  with **group-wise scales** (per-128 K-group, the W4 quality lever) and a **Hadamard
  rotation**, both on by default. Generates **char-identically to fp16** ("Paris / Eiffel
  Tower / Louvre Museum"; primes + Jupiter). Per-matmul cosine vs fp16 ~ **0.98-0.99**
  (per-channel + Hadamard ~0.978, group-128 ~0.987, group-64 ~0.989); without Hadamard
  W4A4 collapses to ~0.79. Hadamard needs full-precision weights, so int4 quantizes from
  an **F16** GGUF (not a pre-quantized Q4 one). Knobs: `ROCKET_INT4_GROUP` (group size,
  default 128; 0 = per-channel), `ROCKET_INT4_HADAMARD` (default 1), `ROCKET_INT4_CACHE_MB`.
- **int4 W4A4 resident (HW-validated):** `ROCKET_INT4_RESIDENT=1` holds the group-wise
  int4 weights in **resident NPU nibble BOs** (the Hadamard rotation baked into the resident
  weight, only A rotated/quantized per call), so prefill pays **no per-call weight scatter**.
  Same quality as the non-resident path (cosine 0.987, char-identical to fp16). Gemma-4-12B
  pp512 @600 MHz: **resident int4 6.94 t/s vs non-resident int4 3.56 (~2x, the packB
  elimination) vs fp16 13.18 (0.53x)**, so it both **halves int4's prefill cost** *and*
  cuts the weights' NPU footprint to **¼** (2634 MB resident vs ~10.5 GB at fp16). vs fp16 it
  stays a RAM play: the group-wise int4's ~120 K-group readbacks per deep-K matmul keep it
  behind fp16+KACC's single readback. `ROCKET_INT4_CACHE_MB` caps the resident BO total
  (raise it: a full 12B is ~6 GB of nibble BOs; the offloaded subset here is ~2.6 GB). The
  one-time per-weight Hadamard build is parallelized across CPU cores (~14x, ~3 min for 12B).
  The resident weight is **M-independent** (the driver plans its tile layout at a canonical M),
  so the weight packed during warmup serves any later prefill M, including a short-prompt
  prefill, with **no re-pack** (the cache is keyed on weight name, not on the padded M). The
  same holds for the resident int8/fp16 weights, so warmup-M serves prefill-M across the board.

## Gemma-4-12B

The large-end model in this stack. RK3588, 8-core, NPU at 600 MHz, warm.

**Prefill (prompt processing), t/s, CPU -> NPU:**

| | pp512 | pp1024 | pp2048 |
|---|---|---|---|
| F16    | 4.82 -> **17.42** (3.6x) | 4.78 -> 16.09 (3.4x) | 4.63 -> 14.98 (3.2x) |
| Q8_0   | 4.52 -> 11.41 (2.5x) | 4.35 -> 13.53 (3.1x) | 4.16 -> 13.43 (3.2x) |
| Q4_K_M | 4.23 -> 11.20 (2.6x) | 4.15 -> 13.19 (3.2x) | 4.02 -> 13.28 (3.3x) |

F16 prefills fastest (17.4 t/s) but its win narrows with M (3.6x->3.2x, output readback grows); the
quants rise with M as their per-µbatch dequant amortizes, converging on F16 by pp2048 (~13-15 t/s),
a quantized GGUF fits smaller, it does not prefill faster at this operating point.

**Decode (generation), t/s, CPU ~ NPU** (decode is off the NPU), bandwidth-bound so it scales with
the quant: F16 **0.9** -> Q8_0 **1.7** -> Q4_K_M **2.5** t/s. At 12 B / F16 (22 GiB) decode is slow
(0.9 t/s), so a quant is effectively mandatory for interactive use on this model.

**Interactive. The NPU shortens the first-token wait; the quant speeds the stream:**

| turn | config | time-to-first-token | stream | total |
|---|---|---|---|---|
| 2048-tok prompt -> 200 out (RAG) | F16 CPU | 442 s | 0.9 t/s | 655 s |
| | F16 + NPU | **137 s** | 0.9 t/s | 350 s |
| | Q4_K_M + NPU | 154 s | 2.5 t/s | **234 s** |
| 128-tok prompt -> 400 out (chat) | F16 + NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M + NPU | 11 s | 2.5 t/s | **171 s** |

The NPU cuts first-token wait ~3.2x on a long prompt, but this model's slow decode means the stream
dominates the turn, so **Q4_K_M + NPU** is the practical pick: ~3.3x prefill at length and a
2.7x-faster stream, in 6.86 GB (fits an 8 GB board). F16 is the max-prefill option when RAM is ample
(the F16 GGUF is 22 GiB). NPU prefill is perplexity-faithful to the CPU (differential wikitext PPL Δ
within ±2% across F16/Q8_0/Q4_K_M, inside the ±10% per-run noise; the absolute PPL of this instruct
model on raw text is not meaningful).

## MoE and MLA models

The `MUL_MAT_ID` expert path (gpt-oss-20b, DeepSeek-V2-Lite).

Mixture-of-experts models route the expert FFNs through `GGML_OP_MUL_MAT_ID`. This backend has a
`MUL_MAT_ID` handler: it buckets the routed `(slot,token)` rows by expert and runs each active
expert's GEMM on the NPU, bit-faithfully. It is worth **~2.4x the CPU** at pp2048 on gpt-oss-20b, and
it wins at every prefill length there (table below), but eligibility is per architecture, not universal:
see the DeepSeek-V2-Lite section for a MoE whose routing puts it under the per-dispatch work floor at short
prefill.

**`ROCKET_MOE` has three states**, because the route's two regimes have opposite signs:

| `ROCKET_MOE` | placement |
|---|---|
| unset (**default**) | Claim an expert op only where the offload is measured to win: a **GGUF-quantized** stack, on the RK3588, whose whole `[K, N, n_expert]` stack the residency pre-flight can reserve before the first ingest, **and** whose per-expert GEMM clears both size floors: the tile granule (`ROCKET_MOE_M_BUCKET`) and the per-dispatch work floor (`ROCKET_MOE_MIN_WORK`). Anything else stays on the CPU. |
| `1` | Claim every `MUL_MAT_ID` the handler can compute, the fp16 streaming route included, with no reservation. The A/B arm; not the recommended setting. |
| `0` | Experts stay on the CPU whatever the shape. |

**The pre-flight is what makes the default safe.** An expert stack's resident cost is knowable from
its tensor alone (`[K, N, n_expert]` plus the GGUF source stride), so the decision is taken in
`supports_op`, *before* the first ingest, against both the RAM budget and the aggregate NPU IOVA
window (`n_threads x 4 GB`; the window is per fd). A stack that will not fit is left on the CPU
**whole**, rather than half-ingested into the partial-residency loss described under **Residency is
the route**.

**The budget is read once per process, and it moves.** `ROCKET_MOE_CACHE_MB`'s auto value is
`MemAvailable − 6 GiB`, resolved at the **first** `supports_op` and then frozen (a re-read per
context would shrink under its own success: the ingest it authorizes is what lowers
`MemAvailable`). Frozen per process still leaves it varying *between* processes: measured on one
board in one session, the same DeepSeek-V2-Lite run saw **24.6 GB** early and **20.7 GB** after
several hours of multi-GB allocation churn, which is the difference between admitting all 78 of its
expert stacks and 70. So a long-lived server that builds a fresh `llama_context` late in its life
can place fewer stacks on the NPU than the same process would have at startup. Nothing warns about
it; the teardown line from `ggml_backend_rocket_moe_stats` is the only place it shows.

Two things make it reproducible, and both are the caller's. **Pin `ROCKET_MOE_CACHE_MB`** to a value
the board can hold, or **drop the page cache before the run** (`echo 3 > /proc/sys/vm/drop_caches`):
with the cache dropped first, three processes requesting 24679 / 26000 / 28000 MB admitted
9539 / 10032 / 10810 experts, and a repeat of the auto arm landed within 13 experts of a separate
probe. That is a procedure measured to work rather than an explained one: `MemAvailable` already
counts most reclaimable page cache, so why dropping it steadies the number is not established.

**Two size floors are the second half of it, and residency implies neither.** The rows an expert
actually receives are `M_e = n_tokens · n_used / n_expert`, which is a property of the
**architecture**, not of the prompt: gpt-oss routes 4-of-32, so 512 tokens give each expert ~64 rows;
DeepSeek-V2-Lite routes 6-of-64, so the same 512 tokens give ~48. At **100% residency** DeepSeek at
pp512 reads **19.09 t/s against 26.04 with the experts left on the CPU**, a 27% regression that
residency cannot see.

- **The tile granule** (`ROCKET_MOE_M_BUCKET`, 64 rows). `M_e` is rounded up onto a fixed bucket
  ladder and the pad rows are computed and read back in full, so an expert GEMM smaller than the
  smallest tile the resident path can run is mostly padding. This one is tile geometry.
- **The work one dispatch carries** (`ROCKET_MOE_MIN_WORK`, 340 mega-MACs of `M_e · K · N`). The
  per-expert cost the granule exists to amortize (one gather, one activation quantize, one submit,
  one fence, one scatter) is paid **per dispatch**, while the work a dispatch carries is
  `M_e · K · N`. So the crossover is a threshold on that product, and `M_e` is a proxy for it only
  while `K · N` is constant. It is not: gpt-oss's expert GEMM is 2880×2880 and DeepSeek-V2-Lite's is
  2048×1408, **2.88x smaller at the same row count**.

Mapping `M_e` -> (default ÷ experts-on-CPU) over `-p 512…2048` on both models, 100% resident and
0 streamed throughout [HW sweep 2026-08-27, 600 MHz pinned, governor `performance`]:

| `M_e` | 64 | 72 | 96 | 128 | 144 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss (2880×2880) | 1.64x | n/a | **1.77x** | 1.91x | n/a | 2.05x | 2.08x |
| DeepSeek (2048×1408) | n/a | **0.948x** | 1.049x | n/a | 1.225x | 1.313x | n/a |

**gpt-oss wins 1.64x at `M_e` = 64 while DeepSeek loses 5-6% at `M_e` = 72**, so any row floor low
enough to admit the first admits the second, and no row floor separates them. Work per dispatch does:
the cells sort by it with no overlap, across both architectures at once. (DeepSeek's cells are means
of four adjacent pairs; its offloaded arm varies ~15% run to run, so a single pair cannot decide a
cell this close to 1.00: `M_e` = 72 and 144 are four pairs, `M_e` = 96 is eight, `M_e` = 192 is
two. gpt-oss's margins are an order of magnitude outside that spread, so one pair each suffices.)

**Which floor binds is per architecture.** On gpt-oss it is always the granule; the work floor would
first bind at `M_e` = 41, which the granule already excludes, so adding it left gpt-oss's placement
unchanged by construction, not only by measurement. On DeepSeek-V2-Lite the work floor binds at
`M_e` >= 118, i.e. past ~1250 tokens in a micro-batch.

The **form** is derived; the **number** is fitted on two architectures, and a third is where it gets
tested, and it changes placement there: a 128-expert/8-used model with a 2048×768 expert would be
declined at `M_e` = 128 (201 MMAC) where the row floor alone accepted it.

**The floor is a MATERIALITY bar, not `> 1.00`, and the ingest is why.** The one-time expert ingest
(~32 s per `llama_context` on DeepSeek, ~36 s on gpt-oss) is charged **only if the gate accepts**, so
the gate is the one place it can be avoided. An offload at ratio `r` saves `1 − 1/r` of prefill wall,
so the ingest breaks even after **~16 300 tokens** of prefill at a 1.05x cell, **~4 800** at 1.22x,
and **~2 100** at gpt-oss's 1.64x. A cell a few percent above parity costs a short session more than
it saves, which is what makes a bar above 1.00 the correct default rather than a cautious one.

The grain is **one weight stack** (one layer's `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`), not
the whole model. "Decline unless the entire model fits" would decline gpt-oss-20b on the very 32 GB
board its win was measured on: 19.1 GiB of int8 codes plus the 11.3 GiB GGUF they are decoded from
do not both fit, and the winning run was 99% resident, not 100%. Per-stack admission keeps that win
and still has no cliff, because a **declined** stack runs on the CPU rather than streaming: the
M-independent dequant is then never paid at all. What is left is a blend of the offloaded route and the
CPU baseline, bounded below by the baseline at every RAM size. A declined stack costs nothing at the
seam either, since this backend's buffer type *is* the CPU buffer type, so a CPU-placed op beside an
offloaded neighbour is a scheduler bookkeeping entry, not a copy.

**How an expert reaches the NPU decides everything.** There are two routes, and only one of them wins:

- **The fp16 dequant route** (`ROCKET_MOE_NATIVE=0`) dequantizes each expert weight to fp16 on the host
  *every micro-batch*. MoE has `n_expert`x more distinct weights per layer than a dense model, each
  amortized over only `M_e` rows, and the decode does not shrink with the row count, so it costs more
  than the GEMM saves, where the CPU's fused quantized kernel pays no dequant at all. Measured on
  gpt-oss-20b: **4.59 t/s at pp512, 10.18 at pp2048**, *worse than leaving the experts on the CPU*.
  It is the A/B baseline, and the default placement does not claim it.
- **The native-quant route** (`ROCKET_MOE_NATIVE`, **on by default**, and the only route the default
  placement claims) ingests a
  **GGUF-quantized** expert **once** into int8 codes that stay resident in NPU BOs, and each call
  quantizes only the activation, per `(row, K-group)`. That deletes the per-micro-batch dequant
  entirely, and it is what turns the loss into the win. An **F16** expert always takes the fp16
  route, since it has no dequant to delete.

### Native-quant experts

Three facts make this work, and they are worth stating because each is load-bearing:

- **The NPU cannot apply a K-blocked scale on chip.** At the output stage `K` is fully contracted, so
  nothing in the DPU is indexed by a K-block. But integer partials **already** leave the chip at every
  K-tile boundary (on-device integer K-accumulation is architecturally impossible), so a per-K-group
  scale rides along **free at a boundary already being paid for**: keep each K-tile inside one quant
  group and fold that group's scale into the readback loop that exists anyway.
- **An MXFP4 code already *is* an exact int8 value** (the codebook is declared `int8_t` and reaches only
  ±12), and MXFP4's block scale is E8M0, an exact power of two. So merging the native 32-element blocks
  of a K-group onto one exponent is an **integer shift, not a requantization**. Every other quantized
  type takes a dequantize-once-requantize route instead, which is equally fine: what costs is a dequant
  *per micro-batch*, not one at load.
- **The speed does not come from the quantization.** The int8 GEMM's int32 output reads back at 8 bytes
  per element, so it moves *more* bytes than the equivalent fp16 GEMM. The entire win is deleting the
  per-micro-batch dequant and weight scatter. Quantization here buys residency, and residency buys the
  speed, which also means the ceiling is what fp16 would do, not better.

Residency is **admission-only** (prefill touches every expert every micro-batch, so there is no hotness
for an eviction policy to exploit; only the resident *total* decides how much of the dequant tax is
removed). An expert that does not fit the budget or the NPU's per-fd IOVA window streams on the fp16
route instead, correctly; `ggml_backend_rocket_moe_stats` reports the split, and it is logged at
teardown. On a 31 GiB board gpt-oss reaches **99%** resident (~14 GB of int8 experts alongside the
11.3 GiB GGUF, which must stay mapped for CPU decode), but only once the resident weights stopped
wasting 35% of their bytes on tile padding. A smaller board will land short, and a partial blend is a
correct outcome rather than a failure; it is just a slower one (below).

**The ingest is not free; it is paid once, per context.** It happens lazily, inside the first
prefill, and it is **~70 s** on gpt-oss-20b: ~1750 experts at **42 ms each**, of which the NPU-BO
scatter is the larger half (50 s) and the MXFP4->int8 decode the smaller (21 s) [HW sweep, 600 MHz].
`ROCKET_LOG_STDERR=1` prints an `[moe-int8] ingesting experts` tick every 256 so the stall is visible
rather than looking like a hang, and the total is reported at teardown.

**It does not contaminate `llama-bench`.** That tool's warmup is a *full* prompt run, so the ingest
lands there and the reported t/s is clean. What it does inflate is wall clock: `llama-bench` builds a
fresh `llama_context` per test row, and the expert cache lives on the backend instance, so the ingest
is paid **once per row**, not once per process. A long-running host (`llama-cli`, `llama-server`) pays
it once.

The route is gated bit-faithful against the CPU backend at the primitive level (`test-rocket-moe`, with
outlier-channel activations: MXFP4 cosine 0.9999, Q4_K 0.9999, Q8_0 0.9999).

**Measured end to end on gpt-oss-20b (MXFP4), `-b 2048 -ub 2048`, 600 MHz, clock pinned, governor
`performance`, one board and one `.so` throughout:**

| test | CPU | NPU, `ROCKET_MOE=0` | **NPU, default** | NPU, `ROCKET_MOE=1` |
|---|---:|---:|---:|---:|
| pp512  | 12.17 | 14.08 | **22.0 (1.81x CPU)** | 24.81 (2.04x) |
| pp2048 | 11.80 | 13.62 | **28.1 (2.38x)** | 34.01 (2.88x) |

The default column is the **mean of six runs** spanning 20.5-22.8 (pp512) and 26.9-28.7 (pp2048),
at identical placement each time: 63 of 72 expert stacks, 1656 experts, **0 streamed**. It is quoted
as a mean because a single run of this arm landed at the bottom of that spread and briefly read as a
regression; the `ROCKET_MOE=0` control was stable to ±0.05 across the same rebuilds, which is what
identified the spread as this arm's own and not the board's.

At the llama.cpp **default `-ub 512`** the default reads **22.2 (21.7-22.7) / 22.5 (22.4-22.8)** against
14.13 / 13.53 with the experts on the CPU, about **1.6x / 1.7x**. The route does *not* collapse at a small
micro-batch: that tax belonged to the fp16 streaming route's per-micro-batch dequant, and
native-quant has none to multiply. It does give up ~20% at pp2048 versus `-ub 2048`, so
`-b 2048 -ub 2048` is still the setting to run it at.

**The default is deliberately not the peak.** The pre-flight reserves all `n_expert` experts of a
stack, where the lazy admission it front-runs charges only the ~82% the router actually exercises, so
the default takes 63 of 72 stacks (100% resident, **0 streamed**) and `ROCKET_MOE=1` takes all 72,
which is worth 8% at pp512 and 16% at pp2048. The trade is **sign for peak**, and on a board with RAM
to spare the peak comes back through `ROCKET_MOE_CACHE_MB`, which keeps the zero-streamed property, on a model whose experts are not most of its weight (raising it is a loss on one that is; see the knob's own row),
rather than through `ROCKET_MOE=1`, which does not.

**Residency is the route.** The win is conditional on nearly the whole expert stack fitting: at 99%
resident the numbers above hold, but at **82%** resident the same route reads 12.19 at pp512, *below*
the 14.11 you get by leaving the experts on the CPU. That is a cliff, not a gradient, and the reason is
the cost structure: a streamed expert keeps paying a weight dequant that is **independent of its row
count**, while a resident one pays a GEMM that shrinks with `M`, so the streamed remainder's share of
the wall clock *grows as the prefill shortens* (12% of pp2048, 43% of pp512). **A residency percentage
is not a cost percentage.**

**But "any streaming is a cliff" is too strong**, and it was the premise the pre-flight was designed
against. Measured on this board, `ROCKET_MOE=1` at **91% resident was the fastest arm of all**, at
high residency, streaming a few experts is *cheaper* than CPU-placing whole stacks. The crossover sits
between **52% and 82%**: at an induced 12 GB budget the same forced arm falls to 52% resident and
reads **13.63 at pp512 against a 14.08 experts-on-CPU baseline**, a net loss, where the default's 30
fully-resident stacks read **17.02 (1.21x)**. That is what the pre-flight buys, and it is the whole
reason the default is bounded below at every budget.

Under the default a stack is reserved before its op is claimed, so the streamed remainder does not
arise at all. Reaching a low residency under the default therefore means a limit no pre-flight can see
ahead of the ingest, most likely an exhausted NPU IOVA window, for which the exit is
`ROCKET_N_THREADS` (more fds, more per-fd window) rather than more RAM. The backend warns below 80%
and names which budget bound it; that threshold is conservative, not measured at its edge.

**DeepSeek-V2-Lite** is a **Multi-head Latent Attention (MLA) + MoE** model, with both an asymmetric
attention (DK=192 ≠ DV=128) and routed experts. The FA gate accepts DK≠DV (bit-faithful primitive),
so MLA is *engageable* with `-fa`, but it is dispatch-bound and pp-neutral at these lengths, so by default
it stays on the CPU. The routed experts take the native-quant route by default **at long prefill only**
(see below; this is the model that showed both size floors). Below them the NPU takes the
large MLA projections, the 2 always-on shared experts, and `lm_head` (ordinary `MUL_MAT`), a **modest
but real** win, *larger* than pure-MoE gpt-oss's dense-only case because those projections are
substantial (Q4_K_M, 15.71 B / ~2.4 B active, RK3588 @ 600 MHz, warm):

| | pp512 | pp1024 | pp2048 | tg64 |
|---|---|---|---|---|
| CPU | 20.37 | 19.92 | 19.01 | 7.73 |
| NPU (default) | 24.04 | 24.85 | 23.87 | 7.77 |

Prefill gains **1.18-1.26x** (NPU flat ~24 t/s, win rising with M as the CPU declines), faithful to the
CPU (differential wikitext PPL Δ −0.26%, a base model so the absolute ~8.2 is meaningful).

On the **fp16 expert route** the handler drops prefill to **0.25x->0.59x** the CPU (5.05 / 7.82 / 11.30 at
`-ub 2048`), a *bigger* loss than gpt-oss's, since DeepSeek's faster CPU and already-winning default NPU
leave more to give up, and its experts are smaller (`K=2048, N=1408`) and more numerous (64), so the
per-expert dequant and dispatch are a larger share of each one. That route is never taken by default.

**The native-quant route on DeepSeek, measured, and the reason the row floor exists.**
[HW sweep 2026-08-27, 600 MHz, governor `performance`.]

| test | CPU | NPU, `ROCKET_MOE=0` | NPU, experts on the NPU |
|---|---:|---:|---:|
| pp512  | 19.54 | **26.04** | 19.09, **0.73x**, a regression |
| pp2048 | 18.50 | 21.63 | **27.9 (1.29x)** |

The pp512 row is a 27% regression at **100% residency, 0 streamed**, so it is not the
partial-residency mechanism, and a residency check alone would never have caught it. DeepSeek routes
**6 of 64** experts, so 512 tokens give each one ~48 rows, below the 64-row granule its GEMM is padded
up to; gpt-oss routes 4 of 32 and gets 64 rows at the same token count. With the granule in place
DeepSeek's pp512 returns to **25.9**, the experts-on-CPU number, and pp2048 keeps its win. Same
model, offloaded at long prefill and declined at short.

**And the granule alone was not enough.** Mapping the whole range showed the default still reading
**0.96x at `M_e` = 72 and 0.94x at `M_e` = 96**, cells it accepted, while gpt-oss reads 1.77x at
that same `M_e` = 96. What separates them is the work per dispatch, not the row count, because
DeepSeek's expert GEMM is 2.88x smaller; see **Two size floors** above. **Do not read gpt-oss's ratio
across to another MoE architecture**: `n_used / n_expert` *and* `K · N` both decide whether the
offload is eligible, and the second one is the easy one to forget.

## The RK3576 (second target)

Everything above describes the RK3588. The **RK3576** runs a different route, selected by the
**detected part** (`rocket_hw_current()`) rather than by a knob, and it is the **only** matmul
route on that SoC: the RK3588 fp16/int8/int4/bf16 generators refuse there by construction, and
the RK3576 entries refuse on the RK3588. So an op this backend claims on that part is claimed by
the W8A8 handler or not at all.

**`ROCKET_INT8=1` is mandatory there.** `supports_op` declines every `MUL_MAT` without it, so a
run left at the defaults offloads nothing and says nothing about why. Set it, and confirm with
`ROCKET_LOG_STDERR=1`.

**Build it with `-DGGML_ROCKET_NATIVE_FP16=OFF`.** The default convert kernels target the
RK3588's `armv8.2-a+fp16` baseline.

### The route

The RK3588's int8 entry hands back a raw int32 accumulator and the host applies both scales. The
RK3576's writes **int8**: on that part any output element wider than one byte poisons the *next*
submit, across processes, so the int32 sibling is not a route a frontend can take. What the part
offers instead is a per-output-**column** requant in the DPU epilogue, and the whole route hangs
off supplying its scale. Four pieces, each load-bearing:

1. **Rotate A and B by an orthonormal Hadamard along K.** Mandatory, not a knob: the unrotated
   route is not merely less accurate, it is *chaotic* (a 2% per-column divisor change moved one
   model's perplexity nine orders of magnitude). `ROCKET_INT8_HADAMARD` does **not** gate it here.
2. **Quantize A per tensor and B per output channel.** Per-axis *input* scales are free at this
   interface and better on every per-GEMM norm, and 38x worse end to end on one model of two, so
   the per-tensor activation scale is the measured choice, not the lazy one (`ROCKET_RK3576_AROW`
   is the A/B).
3. **Per-column output requant**, its scale frozen from a short calibration pass and divided by a
   safety factor (`ROCKET_RK3576_NCAL` / `CALSAFE` / `BOOTMARGIN`).
4. **Host dequantize by the scale the entry was *asked* for**, not the gain its integer ramp
   delivered. The two differ by the entry's `worst_rel_err`; de-quantizing by the achieved gain is
   a measured wash and is deliberately not done.

Composed, that measures **1.008x / 1.020x wikitext-2 perplexity against fp32** on Qwen2.5-1.5B and
SmolLM2-1.7B, against 1.11x / 2.26x for the per-tensor output scale it replaces.
[host arithmetic over two models, 2026-08-10]

Taken off the device inside a model, the composed route reads **1.011x** on Qwen2.5-1.5B: 9.8969
against the F16 CPU arm's 9.7872 over eight wikitext-2 chunks. **That is not the same quantity as
the 1.008x above and the two must not be diffed** -- the simulator figure is against fp32 with an
exactly frozen colmax, this one against the same GGUF's F16 CPU arm with the shipped bootstrap at
`CALSAFE`=3.0, and the eight chunks carry their own +-0.59 error bar. What it settles is that
nothing on the composed device route costs a model an order of magnitude.
[HW sweep, H96 MAX M9, llama.cpp b10356, Qwen2.5-1.5B F16, 2026-08-11]

**Shape contract**, stricter than the RK3588's: `K%32`, `N%32` (not `N%16`), `K >= 64`, `N >= 32`,
2D static weight, `M >= ROCKET_MIN_M`, and a chunking of `K` the entry will take (see
`ROCKET_RK3576_KSPLIT`). **`M` carries no constraint** on this part, and `M=1` computes, which is the
opposite of the RK3588, where rows are the convolution's spatial height and a height below 4
mis-computes. There is deliberately no `M%4` padding here.

### Performance, and the denominator that matters

Qwen2.5-1.5B at `pp2048` under stock llama.cpp, `-ub` 512, four threads pinned to the A72s,
governor `performance`, attention on the CPU (see below), three processes an arm interleaved in
one boot: the port reads **13.36 t/s** against **10.18 t/s** for ggml's Q8_0 kernel (**1.31x**)
and **5.64 t/s** for the same F16 GGUF on the CPU (**2.37x**). Both numbers are real and they
answer different questions: the 2.37x is what a drop-in user reads, because the port consumes an
F16 GGUF; the 1.31x is against the strongest CPU arm on the same silicon. Quote them together.
Per-pair ratios are 1.30 / 1.34 / 1.30.
[HW sweep, H96 MAX M9, 7.1.7 / `rocket` 1.6.0, llama.cpp b10558, 2026-08-23]

**A `pp512` figure on this part prices the calibration path, not the route.** At that length the
route has not converged: a `pp512 -r 1` run reports 332 of 332 calls as calibration forwards. The
11.78 t/s it gives, and the 2.01x / 1.02x pair that follows from it, are a calibration-regime
measurement and are not the route's speed.
[HW sweep, H96 MAX M9, llama.cpp b10356, 2026-08-11]

Two operating notes. The governor is worth **16%**. And run-to-run spread is a function of prompt
length rather than a property of the route: at `pp2048` it is **±0.02 t/s within a process and 3.5%
across processes**, so one process is a measurement; at `pp512` the NPU arm spreads **±1.8 t/s
(15%)** against ±0.01 on both CPU arms, and one process is not.

### Attention stays on the CPU there

`FLASH_ATTN_EXT` is **declined** on the RK3576. The handler's only compute call goes through
`rocket_matmul_fp16`, which refuses on any profile that is not `rk3588`. The attention primitive
has no encoder for this part, so claiming the op would hand the scheduler something only the
single-threaded host reference can compute, slower than the CPU backend's own kernel. The gate
declines instead, which is what `ROCKET_FLASH_ATTN=0` did by hand and is the configuration every
number above was measured under. Nothing to set.

This is worth knowing because of how it used to fail: the gate had no RK3576 branch, a prompt of
512 kept `n_kv` under the offload floor so the whole W8A8 corpus ran, and anything longer returned
`llama_decode ... res = -3` with the explanation swallowed by `llama-bench`'s no-op `ggml` log
callback. Read a bare `res = -3` under a bench harness as "a message you are not being shown", and
reach for `ROCKET_LOG_STDERR=1`.

### RK3576 knobs

All twelve are RK3576-only and no-ops elsewhere. The defaults are the shipping route; the rest are
A/B and instrument knobs, and the last two of the accuracy group change what the model computes.

| var | default | meaning |
|---|---|---|
| `ROCKET_RK3576_NCAL` | 2 | calibration forwards before the per-column output scale is frozen. Each is two device passes: pass one estimates the per-column accumulator maxima from an analytic bound, pass two refines at that scale and returns the surface. Clamped to >=1, since a value below that would freeze an all-zero scale vector and divide by it |
| `ROCKET_RK3576_CALSAFE` | 3.0 | divisor applied to the frozen column maximum. 3.0 rather than 2.0 because the frozen maximum is *bootstrapped*: a bootstrap's 0.3% per-column error costs SmolLM2-1.7B 0.018 of perplexity ratio at 2.0 against 0.0021 at 3.0, while the exactly-frozen arm moves only 0.0017 between them |
| `ROCKET_RK3576_BOOTMARGIN` | 1.5 | slack on the calibration passes' own output scale. The returned surface is correct at a scale this much loose, which costs resolution and nothing else |
| `ROCKET_RK3576_AROW` | 0 | per-**row** activation scale instead of the per-tensor one. Free at this interface (the row scale never crosses the ioctl; it is applied in the dequantize) and better on every per-GEMM norm, and **38x worse end to end on one model of two**, which is why it is off. `=1` everywhere, `=2` only on the weights the K-split cut. Measure a candidate model; do not read a norm |
| `ROCKET_RK3576_KSPLIT` | **off** | cut a `K` past the library's `K >= 6176` refusal into chunks on the host, one whole W8A8 GEMM each, f32 partials summed. Fast and exact, and it costs Qwen2.5-1.5B **1.60x** wikitext-2 perplexity, because what it reaches is `ffn_down` and the route's per-tensor activation scale does not carry that input. More chunks is worse, not better (ten ways scores 18.22 against four ways' 15.84). Off, an FFN down-projection is declined and runs on the CPU |
| `ROCKET_RK3576_KCHUNK` | auto | move the head of the chunk-size preference list (2304, 2048, 1536, 1024, 512, 256, 128, 64, 32) so a candidate depth can be validated without a rebuild. Only 2304 / 2048 / 1536 were measured clean; a model whose chunking lands elsewhere should have that chunk timed before its numbers are quoted |
| `ROCKET_RK3576_FORCESPLIT` | off | cut a `K` the part would take whole. The arm that separates a defect in this code from a property of the route: at `K=1536` the rotation is bit-identical either way, so anything the output does under it is this code's doing. Use with `KCHUNK` |
| `ROCKET_RK3576_WSA` | on | hand the entry this weight's cached per-column `sum_k|qB|` instead of letting it recompute the O(K·N) pass every call. Bit-identical by construction, a cost knob rather than an accuracy one; `=0` prices the pass |
| `ROCKET_RK3576_WDEV` | on | once a weight's calibration converges, pack it into a device BO and hand the entry that, then drop the host `qB`. The cube is the same bytes, so this is memory-neutral; it deletes the per-call cube memset, copy and weight-BO churn. `=0` restores the per-call pack for pricing. Latches off for the whole context if one create fails |
| `ROCKET_RK3576_CALSCAN` | 2 | which form of the calibration column scan runs: `0` column-outer, `1` row-major with an `int` accumulator, `2` row-major with a byte one. Exists to price them: `2` is ~3.5x `0`, flat in `N`, because a byte lane processes 16 elements where an `int` lane processes 4 |
| `ROCKET_RK3576_CALROWS` | 0 (all) | cap the row count of calibration pass **one**. An instrument, not a policy: pass one's estimate is a maximum over `M` accumulator rows, so cutting rows undershoots it. Pass two always runs at full `M` and its surface is what the call returns |
| `ROCKET_RK3576_CALMAP` | off | write a per-column CSV readout of what the two calibration passes learn (`key,chunk,fwd,M,Kc,N,n,abound,est1,est2,sat,rmsA,rmsB`), enough to score a host-side predictor of the column maximum offline. Costs an O(M·Kc + N·Kc) sweep per forward; keep it out of a timed arm |

`ROCKET_MM_PROFILE` covers this route too, through a third accumulator: the library's
`sumabs / packA / packB / coeff / gen / alloc / stamp / submit / wait / read` plus the terms outside
the entry (`valloc / quant+rot / calibrate / entry / dequant`). Both sum to the **offloaded path**,
not to the prefill wall; attention, the norms, rope and the scheduler are in neither.

## Why quantization does not speed prefill

The dispatch floor. The fp16 prefill in the benchmarks is the product of a stack of
operating-point wins over the 200 MHz boot
clock: 600 MHz (×1.43) -> NPU-side fp16 K-accumulation (`ROCKET_KACC`, +19%) -> CBUF DATA_REUSE (+7%)
-> resident weights (+6%), which on Gemma-4-12B pp2048 reads 7.98 -> 11.40 -> 13.38 -> ~14.5 -> ~15.1 t/s
(KACC + DATA_REUSE is the default operating mode; resident weights are opt-in). What that stack does
*not* include is quantization:

**At the current operating point, quantization does not speed up prefill.** Resident
matmul is **~460 GOP/s across precisions** (fp16 461
/ int8 386 / int4 413 GOP/s); the NPU runs at ~15% of MAC peak, so it's
**DMA/dispatch-bound, not MAC-bound**, and int8's 2x / int4's 4x MAC don't express.
In-model **resident int8 prefill is 0.60x fp16** (its int32 readback can't be
K-accumulated, because the NPU's eltwise operand DMA is <=16-bit). So quantization's payoff
*today* is **RAM / model-size / decode-coexistence**, not throughput. Treat this
as **bottleneck-conditional, not permanent**: quant's MAC advantage is gated behind the
dtype-independent dispatch floor (not yet attacked), so it's a later-stage lever, not a
dead one.

So *today's* production path is **fp16 + KACC + DATA_REUSE** (~14.5 t/s), optionally
with resident weights for prefill (~15.1). int8/int4 are there for when a model must be
quantized to fit memory; whether they can also win on prefill speed is gated by the
dtype-independent dispatch floor, not by the datatype.

**Quantized prefill is micro-batch-sensitive.** A quantized GGUF dequantizes to fp16 **per
micro-batch**, so the llama.cpp default `-ub 512` halves it; **`-b 2048 -ub 2048` recovers ~2x**
(9B `Q4_K` 8->17 t/s). That per-µbatch dequant is itself **multi-threaded by default**
(`ROCKET_DEQUANT_THREADS`, auto = cores capped at 8; set 1 for the serial path), the weight
rows are independent, so fanning them across the CPU recovers **+33% @`-ub 512` / +19% @`-ub
2048`** over the serial decode, bit-exact, with no extra RAM (9B `Q4_K`, same-session HW A/B).
Quant *type* is then irrelevant to NPU throughput (`Q4_K` / `IQ4_XS` /
`Q8_0` converge); quant closes toward F16 with M: ~0.64x at pp512 rising to ~0.85x at pp2048 (F16 stays resident, no dequant). **`ROCKET_QUANT_RESIDENT=1`
closes that gap**: it dequant->packs each weight to resident fp16 NPU BOs **once**, so prefill then pays
neither the per-µbatch dequant nor the per-call packB (it reuses the F16 prepacked path, bit-identical PPL),
lifting quant prefill to **F16 parity** and recovering the `-ub 512` halving [HW, 0.8B `Q4_K` pp2048, same
session: resident **89** t/s = F16 **89**, vs streaming **81** @ub2048 and **51** @ub512, so **1.5x** at the
llama.cpp-default `-ub 512`, and the per-call `packB` collapses 12440->889 ms; 9B `Q4_K` resident **24.6**
vs streaming **15.8** = **1.56x**, ~0.92x F16; its ~18 GB fp16 fits the 31 GB RAM + IOVA window]. The cost is the full fp16
resident footprint (it negates the quant RAM saving), so it is opt-in for the RAM-to-spare case; over
`ROCKET_CACHE_MB` / once the NPU IOVA window fills, weights fall back to streaming. On a model larger
than the default 2 GB budget, use **`ROCKET_QUANT_RESIDENT=auto`** (budget sized from free RAM) so it
residents in full; blanket `1` at the default budget residents only part and can **net-lose to
streaming** (9B `Q4_K` pp2048 ub2048: 19.3 vs 22.1 t/s, `auto` 25.8 ~ F16 25.1; the win grows at the
llama.cpp-default `-ub 512`, where streaming re-dequants 4x: `auto` 26.1 vs streaming 15.3 = **1.7x**
[HW sweep, 600 MHz]). Short
quant prefills below `ROCKET_MIN_M_QUANT` (~ the measured ~360-row crossover) auto-route to the CPU.
Across models, the **F16 NPU prefill win grows with size then plateaus: 0.8B 1.44x -> 9-12B ~3.6x
CPU**, validated on Qwen3.5 / Qwen3.6 (incl. the 27B hybrid-DeltaNet) and Gemma-4-12B (3.6x at
pp512), all PPL-faithful.

Quantized int8 does not speed this backend's prefill, and that is about the datatype's role
here, not a hardware mode the backend is missing. int8 helps prefill only by cutting operand
*bandwidth*: a W8A8 scheme halves both operands' bytes, where this path quantizes weights and
keeps fp16 activations, and by batching more work per kernel submit to lower the dispatch floor.
Neither lifts the MAC ceiling: the matmul is DMA/dispatch-bound, not MAC-bound, so throughput
rises by attacking the dispatch floor (fewer, bigger, batched submits), not by narrowing the
datatype. On-device integer K-accumulation, which would let int8 keep its narrow weights *and*
avoid the int32 readback, does not exist on this silicon.

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

`ROCKET_INT8` / `ROCKET_INT4` / `ROCKET_BF16` are numerically faithful but tie the prefill throughput of
fp16; use them to fit a model in less RAM, never to speed prefill. Confirm the intended mode engaged by
running once with `ROCKET_LOG_STDERR=1` (llama-bench otherwise prints no backend lines).

## Runtime knobs

The backend reads a set of `ROCKET_*` env vars. `sudo` strips the environment, so always use `sudo -E`.
A set-but-empty var (e.g. `ROCKET_MIN_M=`, the shape a wrapper produces when it forwards an unset
`$VAR`) is treated as **unset** and takes the default; it does not parse as `0` and collapse a
routing threshold to its clamp floor, or a knob whose default is non-zero to a different arithmetic.
The one exception is the presence-only diagnostics (`ROCKET_DEBUG`, `ROCKET_MM_PROFILE`,
`ROCKET_AB`, `ROCKET_DUMP`, `ROCKET_FA_TIMING`), which are documented as "set to anything to arm"
and so are armed by an empty value; none of them changes a result.

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
| `ROCKET_MOE_CACHE_MB` | **auto** | resident native-quant **expert** budget in MB (`0` = unlimited). Auto = `MemAvailable` − reserve. **This is the knob that buys back what the pre-flight's conservatism costs**, and unlike `ROCKET_MOE=1` it keeps both the zero-streamed property and the sign guarantee. Measured ladder, gpt-oss-20b on a 31 GiB board, every arm 0 streamed [HW sweep 2026-08-27]: auto (24672 MB) takes 63 of 72 stacks; **26000 takes 66 (+4.3% / +5.1% at pp512 / pp2048); 28000 takes 71 (+8.3% / +14.1%)**, which is 79% of the `ROCKET_MOE=1` ceiling and above it at pp512. 28000 leaves only ~2.8 GB of the headroom the 6 GiB auto reserve exists for, so it is a setting for a **known working set**, not a new default; pin it explicitly if you also need placement to be reproducible across processes. **On an expert-dominated model the direction inverts, and raising the budget costs the whole offload.** On Qwen3-30B-A3B (29 of 30.5 B in the experts, 17.28 GiB GGUF) at `-ub` 4096 the placed-fraction curve is a plateau then a cliff, and the auto budget is over the edge: **18000-21000 MB** takes 58-67 stacks and reads **1.046x** the experts-on-CPU baseline (seven processes), auto (24425) takes 79 and reads **1.014x**, and 28000 takes 90 and reads **0.999x, no gain at all**, with per-rep spread rising from ±0.00 to ±1.02 as placement grows [HW sweep 2026-08-28, RK1, 600 MHz]. The cause is that the admission charge counts the GGUF source bytes of the experts it *places* but not of the ones it leaves on the CPU, which are read from the same mmap every micro-batch and are equally unreclaimable, so the true hot set crosses RAM before the budget says it has. **Read a `ROCKET_MOE_CACHE_MB` recommendation as scoped to the model class, not to the board size**: raise it on a model whose non-expert weights dominate, and on an expert-dominated one *lower* it, to roughly `MemTotal` minus the whole expert GGUF minus the reserve, and measure Charged per expert as *int8 codes + the expert's GGUF source bytes*, because the GGUF is mmapped and cannot be reclaimed (MoE **decode** reads the active experts from it on the CPU every token), so both copies must coexist. Read by the placement pre-flight as well as by admission, so the decision to claim an op and the charge taken when its experts are ingested are against the same number. Admission-only: an expert that does not fit streams on the fp16 route, correctly, and the split is logged at teardown (`ggml_backend_rocket_moe_stats`); under the default that split should read 100% resident, because an unreservable stack was never claimed |
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

The heavy recompute/trace diagnostics (`ROCKET_VERIFY` / `ROCKET_CPU_FORWARD` / `ROCKET_TRACE` /
`ROCKET_AB` / `ROCKET_DUMP`) are compiled out of the production `.so`; build with
`-DGGML_ROCKET_DIAGNOSTICS=ON` to enable them. `ROCKET_MM_PROFILE` / `ROCKET_DEBUG` /
`ROCKET_DEBUG_GRAPH` are always available.

K-accum is on by default (DATA_REUSE follows automatically); `ROCKET_KACC=0` opts out.

## Diagnostic logging

Every diagnostic the backend emits (errors, warnings, the `ROCKET_MM_PROFILE` /
`ROCKET_FA_TIMING` breakdowns, and the env-gated traces above) flows through ggml's log
channel, so the host (llama.cpp, whisper.cpp) redirects or silences it with the same
`ggml_log_set` callback it uses for its own output. Severity maps the obvious way: failures
to `ERROR`, recoverable fallbacks to `WARN`, the profile breakdowns and one-time hints to
`INFO`, the per-op traces to `DEBUG`. The env knobs above still gate whether a trace is
emitted at all; the channel only governs where it goes.

The driver it links (librocketnpu) has its own log channel, which the backend bridges into
ggml's at startup, so a driver message reaches the same host callback rather than landing
on raw stderr. The driver's own threshold applies first, so a driver `DEBUG` trace needs
both its driver-side gate (`ROCKET_DEBUG`) and a host logger that prints `DEBUG`.

One consequence: a host that *silences* its `ggml` logger silences the backend too. `llama-bench`
installs a no-op `ggml` callback unless run with `-v`, so its default run hides every backend
line, including one-shot decisions such as the `ROCKET_QUANT_RESIDENT=auto` resident-weight
budget, which changes the benchmarked prefill number and so is exactly what you want to see. For
that case the backend routes its user-facing config lines (the budget decision, the "raise `-ub`"
quant hint) through the driver channel, so `ROCKET_LOG_STDERR=1` tees them to stderr regardless of
the host's logger. Set it alongside a benchmark to confirm which budget/mode actually engaged.

## Implementation notes

- Uses ggml's private headers (`ggml-impl.h`, `ggml-backend-impl.h`), expected for an
  out-of-tree backend. The vtables (`ggml_backend_i` etc.) are initialized
  positionally, so the backend must be built against the **host's bundled ggml**
  (field order + `GGML_BACKEND_API_VERSION` must match). The concrete ABI this backend
  targets is **`GGML_BACKEND_API_VERSION 2`, with the device vtable that *includes* the
  `set_tensor_2d_async` / `get_tensor_2d_async` slots** (built/verified against the
  in-repo `ggml/`, tagged v0.14.0). Because the host apps (llama.cpp / whisper.cpp)
  clone their own ggml, build this backend against **that checkout's** headers and
  re-check on every bump: if the host's `ggml-backend-impl.h` has a different
  `GGML_BACKEND_API_VERSION` or 2d-field layout, the positional vtable drifts and you
  get a "ROCKET device not listed at startup" failure.
- `ggml_fp16_t` (uint16_t) shares the IEEE-half bit layout with `_Float16`, so the
  driver's `_Float16*` buffers are filled via `ggml_fp32_to_fp16_row` and cast.
- **Why offload happens:** ggml's scheduler only pulls a host-resident op onto an
  ACCEL device when **both** `supports_op` and `offload_op` return true (BLAS does the
  same). A NULL `offload_op` makes the device appear at startup but run zero matmuls.
</content>
</invoke>
