# ggml-rocket — API and tuning reference

The exhaustive per-model benchmarks, the full `ROCKET_*` runtime-knob table, the
diagnostic-logging channel, and the backend implementation notes. The [README](README.md)
is the guide; this is the reference.

## Models and precisions

- **Whisper encoder (HW-validated):** stock `whisper-cli` / `whisper-bench` load the `.so`
  via `GGML_BACKEND_PATH` and run the encoder `MUL_MAT`s on the NPU, transcripts
  byte-identical to the CPU (WER 0; encoder-output cosine 0.9998). Encoder latency beats
  the CPU at every size and the win **grows with the model** — `whisper-bench` encode,
  warm, RK3588 @ 600 MHz: tiny.en 1.18×, base.en 1.29×, small.en 1.54×, medium.en 1.84×,
  large-v3 2.14×, large-v3-turbo 2.12×. The decoder (autoregressive, M=1) stays on the CPU
  on both backends. **large-v3-turbo** keeps the full 32-layer encoder but a 4-layer decoder
  (~6× cheaper per step), so the NPU-accelerated encoder is the larger share of its pipeline.
- **Gemma-4-12B prefill (HW-validated):** coherent generation via the same drop-in on
  llama.cpp. Prefill GEMMs offload, decode stays on CPU.
- **Phi-4 (14B) prefill (HW-validated):** the 14.66 B Phi-4 (arch `llama`, dense GQA — a different
  architecture from Phi-4-mini's `phi3`) prefills on the NPU through the same drop-in, **PPL-faithful**
  to the CPU (Q8_0 / Q4_K_M wikitext Δ −0.09% / −0.31%). Its F16 GGUF (29.3 GB) does not fit a 31 GB
  board, so it runs quantized: **Q8_0 / Q4_K_M ~3.4–3.5× CPU at pp2048** (~12 t/s over a ~3.5 t/s 14 B
  CPU baseline), the win rising with prompt length as the per-micro-batch dequant amortizes. Q4_K_M
  (8.28 GB) is the practical pick — fits a ~9 GB budget and streams decode ~2.2 t/s. A mid-large point
  between Gemma-4-12B and Qwen3.6-27B.
- **Qwen3.5 / Qwen3.6 prefill (HW-validated):** Qwen3.5 (0.8B/9B, dense GQA) and
  **Qwen3.6-27B (hybrid Gated-DeltaNet linear attention)** prefill on the NPU through the same
  drop-in, **PPL-faithful** to the CPU. The DeltaNet / SSM-scan layers are CPU-only ops and stay
  there; the FFN + projections offload and the prefill win holds — the hybrid's linear-attention
  layers do not block it (**27B Q4_K_M 4.4× CPU at pp2048**, the largest prefill win measured, its
  quant lapping an especially slow 27B CPU baseline). The NPU prefill advantage **grows with model
  size** — F16 0.8B 1.44× → 9B 3.65× CPU. A quantized GGUF wants `-b 2048 -ub 2048` (see
  Performance / `ROCKET_MIN_M_QUANT`).
- **Small dense models — Llama-3.2-3B / Ministral-3-3B / Phi-4-mini (HW-validated):** the 3–4 B
  dense end (arch `llama` / `mistral3` / `phi3`, GQA) posts the **fastest prefill** through this
  backend — Llama-3.2-3B F16 **61.8 t/s at pp512 (3.4× CPU)**, Ministral-3-3B 57.6, Phi-4-mini 55.4
  — because fewer FLOPs per matmul. The F16 win narrows with prompt length (3.4×→2.4–2.6× by pp2048)
  as a small model's per-op readback grows with M, so it lands hardest on short-to-medium prompts; a
  Q4_K_M fits a 4 GB board (Llama 1.87 GB, Ministral 1.99 GB) and streams decode fastest (~7.7–7.9
  t/s). NPU prefill is PPL-faithful to the CPU (Llama / Ministral F16 wikitext Δ +0.02% / −0.04%).
  The NPU/CPU prefill *ratio* is modest on these because their CPU baseline is comparatively fast —
  the NPU's absolute prefill rate is the headline.
- **Vision-language / multimodal — SmolVLM2-2.2B (HW-validated):** the full pipeline runs on the
  NPU through llama.cpp's `mtmd` path — a **SigLIP-SO400M vision encoder** plus a **SmolLM2-class
  language model** (arch `llama`, 1.81 B; the "2.2B" is the combined count). The LLM half is the
  **fastest in this backend**: F16 **99 t/s at pp512 (3.5× CPU)** and the briskest decode (Q4_K_M
  **14.5 t/s** from a 1.03 GB file), PPL-faithful (F16 / Q4_K_M wikitext Δ −0.07% / −0.71%). The
  vision encoder runs through `clip.cpp`; attach the NPU with **`MTMD_BACKEND_DEVICE`** (clip's
  auto-path only probes GPU/IGPU device types, so the ACCEL device must be named explicitly, else
  the encoder stays on the CPU). Offloaded, the SigLIP encode is **~1.19× faster warm** and
  **faithful at cosine 0.99998** (projected image embeddings vs CPU) — a modest win: ggml-rocket
  offloads only the aligned static-weight GEMMs (q/k/v/o projections + fc1), while fc2 (K=4304 off
  the K%32 grid), the attention core, and all norm/softmax/GELU ops stay on the CPU, so the graph
  pays many CPU↔NPU handoffs. A resident whole-encoder path would widen this; the drop-in clip
  route is correct and modestly positive as-is.
- **Quantized GGUF prefill (HW-validated):** a `Q8_0` / `Q4_K` / `Q6_K` / … GGUF runs
  its prefill GEMMs on the NPU directly — each weight is dequantized to fp16 on the fly
  (one [K] row at a time, the exact values ggml's CPU backend decodes), with **no
  whole-model F16 copy in RAM**. So the model stays compact on disk and in memory (e.g.
  Gemma-4-12B `Q8_0` ≈ 12 GB vs the F16 GGUF's ≈ 24 GB) and prefill still takes the fp16
  NPU path; decode stays on the CPU on the quantized weights. Quantization buys
  RAM / model-fit here, not prefill speed (the matmul is dtype-independent — see
  Performance). The dequant is paid **per micro-batch**, so quantized prefill is
  micro-batch-sensitive: run with **`-b 2048 -ub 2048`** for ~2× over the llama.cpp
  default `-ub 512` (which re-dequantizes every 512-row chunk). Short prefills below
  `ROCKET_MIN_M_QUANT` (default 512 rows ≈ just past the measured ~360-row crossover)
  stay on the CPU, where they beat a dequant-bound NPU offload. Importance-matrix quants
  (`IQ4_XS`, …) take this same path — an imatrix is a *quantize-time* construct, so at inference
  they are ordinary dequant-to-fp16 types (`IQ4_XS` is 4.25 bpw with 256-element super-blocks,
  `K%32` holds) whose prefill GEMMs offload like any quant, not a CPU fallback (Qwen3.5-9B
  `IQ4_XS` prefill 3.1× CPU at pp2048, the fastest-decoding rung at 4.80 GB). Requires the host
  built `-DGGML_CPU_REPACK=OFF` (see *Drop-in use with llama.cpp* below).
- **Resident weights:** static F16 weight tensors are packed into resident NPU BOs and
  cached by **weight name** (not `src0->data`: the scheduler reuses pooled copy addresses
  across weights, so keying on the address aliases distinct weights and produces gibberish;
  `supports_buft = is_host`). +6% prefill (prefill-only `madvise` of the F16 source). The cache
  is keyed on the name alone — **not** the prefill M — so a weight packed at one prefill length is
  reused at any other (M ≥ the tile cap) with no re-pack (the resident layout is M-independent there).
  A weight is made resident only when first seen at a reuse-worthy M; a smaller one-shot M (a short
  prompt, or the tail ubatch of a long prompt) streams via the per-call mt path, leaving the resident
  weight intact for the big prefills (no re-pack thrash on mixed-length workloads). (Default-gated to
  K≤2048 weights; `ROCKET_FORCE_PREPACK=1` makes all weights resident.)
- **int8 W8A8 + Hadamard:** coherent — char-identical to fp16 — but a **net loss** in
  prefill speed (see below).
- **int4 W4A4 in-model (HW-validated):** native int4×int4 prefill for a Gemma-4-12B F16
  GGUF via `ROCKET_INT4=1` — weights and activations quantized to 4-bit (symmetric `[-7,7]`)
  with **group-wise scales** (per-128 K-group, the W4 quality lever) and a **Hadamard
  rotation**, both on by default. Generates **char-identically to fp16** ("Paris / Eiffel
  Tower / Louvre Museum"; primes + Jupiter). Per-matmul cosine vs fp16 ≈ **0.98–0.99**
  (per-channel + Hadamard ≈ 0.978, group-128 ≈ 0.987, group-64 ≈ 0.989); without Hadamard
  W4A4 collapses to ≈ 0.79. Hadamard needs full-precision weights, so int4 quantizes from
  an **F16** GGUF (not a pre-quantized Q4 one). Knobs: `ROCKET_INT4_GROUP` (group size,
  default 128; 0 = per-channel), `ROCKET_INT4_HADAMARD` (default 1), `ROCKET_INT4_CACHE_MB`.
- **int4 W4A4 resident (HW-validated):** `ROCKET_INT4_RESIDENT=1` holds the group-wise
  int4 weights in **resident NPU nibble BOs** (the Hadamard rotation baked into the resident
  weight, only A rotated/quantized per call), so prefill pays **no per-call weight scatter**.
  Same quality as the non-resident path (cosine 0.987, char-identical to fp16). Gemma-4-12B
  pp512 @600 MHz: **resident int4 6.94 t/s vs non-resident int4 3.56 (≈2×, the packB
  elimination) vs fp16 13.18 (0.53×)** — so it both **halves int4's prefill cost** *and*
  cuts the weights' NPU footprint to **¼** (2634 MB resident vs ~10.5 GB at fp16). vs fp16 it
  stays a RAM play: the group-wise int4's ~120 K-group readbacks per deep-K matmul keep it
  behind fp16+KACC's single readback. `ROCKET_INT4_CACHE_MB` caps the resident BO total
  (raise it — a full 12B is ~6 GB of nibble BOs; the offloaded subset here is ~2.6 GB). The
  one-time per-weight Hadamard build is parallelized across CPU cores (~14×, ~3 min for 12B).
  The resident weight is **M-independent** (the driver plans its tile layout at a canonical M),
  so the weight packed during warmup serves any later prefill M — including a short-prompt
  prefill — with **no re-pack** (the cache is keyed on weight name, not on the padded M). The
  same holds for the resident int8/fp16 weights, so warmup-M serves prefill-M across the board.

## Gemma-4-12B

The large-end model in this stack. RK3588, 8-core, NPU at 600 MHz, warm.

**Prefill (prompt processing), t/s — CPU → NPU:**

| | pp512 | pp1024 | pp2048 |
|---|---|---|---|
| F16    | 4.82 → **17.42** (3.6×) | 4.78 → 16.09 (3.4×) | 4.63 → 14.98 (3.2×) |
| Q8_0   | 4.52 → 11.41 (2.5×) | 4.35 → 13.53 (3.1×) | 4.16 → 13.43 (3.2×) |
| Q4_K_M | 4.23 → 11.20 (2.6×) | 4.15 → 13.19 (3.2×) | 4.02 → 13.28 (3.3×) |

F16 prefills fastest (17.4 t/s) but its win narrows with M (3.6×→3.2×, output readback grows); the
quants rise with M as their per-µbatch dequant amortizes, converging on F16 by pp2048 (~13–15 t/s) —
a quantized GGUF fits smaller, it does not prefill faster at this operating point.

**Decode (generation), t/s — CPU ≈ NPU** (decode is off the NPU), bandwidth-bound so it scales with
the quant: F16 **0.9** → Q8_0 **1.7** → Q4_K_M **2.5** t/s. At 12 B / F16 (22 GiB) decode is slow
(0.9 t/s), so a quant is effectively mandatory for interactive use on this model.

**Interactive — the NPU shortens the first-token wait; the quant speeds the stream:**

| turn | config | time-to-first-token | stream | total |
|---|---|---|---|---|
| 2048-tok prompt → 200 out (RAG) | F16 CPU | 442 s | 0.9 t/s | 655 s |
| | F16 + NPU | **137 s** | 0.9 t/s | 350 s |
| | Q4_K_M + NPU | 154 s | 2.5 t/s | **234 s** |
| 128-tok prompt → 400 out (chat) | F16 + NPU | 7 s | 0.9 t/s | 433 s |
| | Q4_K_M + NPU | 11 s | 2.5 t/s | **171 s** |

The NPU cuts first-token wait ~3.2× on a long prompt, but this model's slow decode means the stream
dominates the turn — so **Q4_K_M + NPU** is the practical pick: ~3.3× prefill at length and a
2.7×-faster stream, in 6.86 GB (fits an 8 GB board). F16 is the max-prefill option when RAM is ample
(the F16 GGUF is 22 GiB). NPU prefill is perplexity-faithful to the CPU (differential wikitext PPL Δ
within ±2% across F16/Q8_0/Q4_K_M, inside the ±10% per-run noise; the absolute PPL of this instruct
model on raw text is not meaningful).

## MoE and MLA models

The opt-in `MUL_MAT_ID` expert path (gpt-oss-20b, DeepSeek-V2-Lite).

Mixture-of-experts models route the expert FFNs through `GGML_OP_MUL_MAT_ID`. This backend has a
`MUL_MAT_ID` handler — it buckets the routed `(slot,token)` rows by expert and runs each active
expert's GEMM on the NPU, bit-faithfully. With `ROCKET_MOE=1` it is worth **2.16× the CPU** at pp2048
on gpt-oss-20b, and it wins at every prefill length (table below). It stays **opt-in** because the win
is conditional on how much RAM the host has, not because it is slow — see **Residency is the route**.

**How an expert reaches the NPU decides everything.** There are two routes, and only one of them wins:

- **The fp16 dequant route** (`ROCKET_MOE_NATIVE=0`) dequantizes each expert weight to fp16 on the host
  *every micro-batch*. MoE has `n_expert`× more distinct weights per layer than a dense model, each
  amortized over only `M_e` rows, and the decode does not shrink with the row count — so it costs more
  than the GEMM saves, where the CPU's fused quantized kernel pays no dequant at all. Measured on
  gpt-oss-20b: **4.59 t/s at pp512, 10.18 at pp2048** — *worse than leaving the experts on the CPU*.
  This is the route the handler originally shipped with, and why it was originally off by default.
- **The native-quant route** (`ROCKET_MOE_NATIVE`, **on by default** within `ROCKET_MOE=1`) ingests a
  **GGUF-quantized** expert **once** into int8 codes that stay resident in NPU BOs, and each call
  quantizes only the activation, per `(row, K-group)`. That deletes the per-micro-batch dequant
  entirely, and it is what turns the loss into the 2.16×. An **F16** expert always takes the fp16
  route — it has no dequant to delete.

### Native-quant experts

Three facts make this work, and they are worth stating because each is load-bearing:

- **The NPU cannot apply a K-blocked scale on chip** — at the output stage `K` is fully contracted, so
  nothing in the DPU is indexed by a K-block. But integer partials **already** leave the chip at every
  K-tile boundary (on-device integer K-accumulation is architecturally impossible), so a per-K-group
  scale rides along **free at a boundary already being paid for**: keep each K-tile inside one quant
  group and fold that group's scale into the readback loop that exists anyway.
- **An MXFP4 code already *is* an exact int8 value** (the codebook is declared `int8_t` and reaches only
  ±12), and MXFP4's block scale is E8M0 — an exact power of two. So merging the native 32-element blocks
  of a K-group onto one exponent is an **integer shift, not a requantization**. Every other quantized
  type takes a dequantize-once-requantize route instead, which is equally fine: what costs is a dequant
  *per micro-batch*, not one at load.
- **The speed does not come from the quantization.** The int8 GEMM's int32 output reads back at 8 bytes
  per element, so it moves *more* bytes than the equivalent fp16 GEMM. The entire win is deleting the
  per-micro-batch dequant and weight scatter. Quantization here buys residency, and residency buys the
  speed — which also means the ceiling is what fp16 would do, not better.

Residency is **admission-only** (prefill touches every expert every micro-batch, so there is no hotness
for an eviction policy to exploit — only the resident *total* decides how much of the dequant tax is
removed). An expert that does not fit the budget or the NPU's per-fd IOVA window streams on the fp16
route instead, correctly; `ggml_backend_rocket_moe_stats` reports the split, and it is logged at
teardown. On a 31 GiB board gpt-oss reaches **99%** resident (≈14 GB of int8 experts alongside the
11.3 GiB GGUF, which must stay mapped for CPU decode) — but only once the resident weights stopped
wasting 35% of their bytes on tile padding. A smaller board will land short, and a partial blend is a
correct outcome rather than a failure; it is just a slower one (below).

**The ingest is not free — it is just paid once, per context.** It happens lazily, inside the first
prefill, and it is **~70 s** on gpt-oss-20b: ~1750 experts at **42 ms each**, of which the NPU-BO
scatter is the larger half (50 s) and the MXFP4→int8 decode the smaller (21 s) [HW sweep, 600 MHz].
`ROCKET_LOG_STDERR=1` prints an `[moe-int8] ingesting experts` tick every 256 so the stall is visible
rather than looking like a hang, and the total is reported at teardown.

**It does not contaminate `llama-bench`.** That tool's warmup is a *full* prompt run, so the ingest
lands there and the reported t/s is clean. What it does inflate is wall clock: `llama-bench` builds a
fresh `llama_context` per test row, and the expert cache lives on the backend instance — so the ingest
is paid **once per row**, not once per process. A long-running host (`llama-cli`, `llama-server`) pays
it once.

The route is gated bit-faithful against the CPU backend at the primitive level (`test-rocket-moe`, with
outlier-channel activations: MXFP4 cosine 0.9999, Q4_K 0.9999, Q8_0 0.9999).

**Measured end to end — gpt-oss-20b (MXFP4), `-b 2048 -ub 2048`, 600 MHz, clock pinned:**

| test | CPU | NPU, experts on CPU | NPU, native-quant experts |
|---|---:|---:|---:|
| pp512  | 13.09 | 14.11 | **17.57 (1.34× CPU)** |
| pp1024 | 12.99 | 14.29 | **24.38 (1.88×)** |
| pp2048 | 12.40 | 13.89 | **26.78 (2.16×)** |

**Residency is the route.** The win is conditional on nearly the whole expert stack fitting: at 99%
resident the numbers above hold, but at **82%** resident the same route reads 12.19 at pp512 — *below*
the 14.11 you get by leaving the experts on the CPU. That is a cliff, not a gradient, and the reason is
the cost structure: a streamed expert keeps paying a weight dequant that is **independent of its row
count**, while a resident one pays a GEMM that shrinks with `M` — so the streamed remainder's share of
the wall clock *grows as the prefill shortens* (12% of pp2048, 43% of pp512). **A residency percentage
is not a cost percentage.** The backend warns when residency lands under ~95%; raise
`ROCKET_MOE_CACHE_MB` if the RAM is there, or set `ROCKET_MOE=0`.

`ROCKET_MOE` stays **opt-in** for exactly that reason: whether it helps depends on how much RAM the
machine has, and a default that silently regresses on a smaller board is not a default.

**DeepSeek-V2-Lite** is a **Multi-head Latent Attention (MLA) + MoE** model — it has both an asymmetric
attention (DK=192 ≠ DV=128) and routed experts. The FA gate accepts DK≠DV (bit-faithful primitive),
so MLA is *engageable* with `-fa`, but it is dispatch-bound and pp-neutral at these lengths, so by default
it stays on the CPU; the routed experts have a handler too (`ROCKET_MOE=1`), whose **fp16 route** loses
here (below). So by default the NPU takes the large MLA projections, the 2 always-on shared experts, and
`lm_head` (ordinary `MUL_MAT`) — enough for a **modest but real** win, *larger* than pure-MoE gpt-oss
because those dense projections are substantial (Q4_K_M, 15.71 B / ~2.4 B active, RK3588 @ 600 MHz, warm):

| | pp512 | pp1024 | pp2048 | tg64 |
|---|---|---|---|---|
| CPU | 20.37 | 19.92 | 19.01 | 7.73 |
| NPU (default) | 24.04 | 24.85 | 23.87 | 7.77 |

Prefill gains **1.18–1.26×** (NPU flat ~24 t/s, win rising with M as the CPU declines), faithful to the
CPU (differential wikitext PPL Δ −0.26%, a base model so the absolute ~8.2 is meaningful).

On the **fp16 expert route** the handler drops prefill to **0.25×→0.59×** the CPU (5.05 / 7.82 / 11.30 at
`-ub 2048`) — a *bigger* loss than gpt-oss's, since DeepSeek's faster CPU and already-winning default NPU
leave more to give up, and its experts are smaller (`K=2048, N=1408`) and more numerous (64), so the
per-expert dequant and dispatch are a larger share of each one. **The native-quant route has not been
measured on DeepSeek** — it is the route `ROCKET_MOE=1` now selects for a quantized expert, and the
mechanism that makes it win on gpt-oss (deleting the per-micro-batch dequant) applies here too, but
DeepSeek's Q4_K experts take the dequantize-once-requantize path rather than MXFP4's exact integer
shift, and its residency profile is its own question. Treat gpt-oss's 2.16× as measured and DeepSeek's
expert offload as **unmeasured**, not as a predicted win.

## Why quantization does not speed prefill

The dispatch floor. The fp16 prefill in the benchmarks is the product of a stack of
operating-point wins over the 200 MHz boot
clock: 600 MHz (×1.43) → NPU-side fp16 K-accumulation (`ROCKET_KACC`, +19%) → CBUF DATA_REUSE (+7%)
→ resident weights (+6%), which on Gemma-4-12B pp2048 reads 7.98 → 11.40 → 13.38 → ~14.5 → ~15.1 t/s
(KACC + DATA_REUSE is the default operating mode; resident weights are opt-in). What that stack does
*not* include is quantization —

**At the current operating point, quantization does not speed up prefill.** Resident
matmul is **~460 GOP/s across precisions** (fp16 461
/ int8 386 / int4 413 GOP/s); the NPU runs at ~15% of MAC peak, so it's
**DMA/dispatch-bound, not MAC-bound**, and int8's 2× / int4's 4× MAC don't express.
In-model **resident int8 prefill is 0.60× fp16** (its int32 readback can't be
K-accumulated — the NPU's eltwise operand DMA is ≤16-bit). So quantization's payoff
*today* is **RAM / model-size / decode-coexistence**, not throughput — but treat this
as **bottleneck-conditional, not permanent**: quant's MAC advantage is gated behind the
dtype-independent dispatch floor (not yet attacked), so it's a later-stage lever, not a
dead one.

So *today's* production path is **fp16 + KACC + DATA_REUSE** (≈14.5 t/s), optionally
with resident weights for prefill (≈15.1). int8/int4 are there for when a model must be
quantized to fit memory; whether they can also win on prefill speed is gated by the
dtype-independent dispatch floor, not by the datatype.

**Quantized prefill is micro-batch-sensitive.** A quantized GGUF dequantizes to fp16 **per
micro-batch**, so the llama.cpp default `-ub 512` halves it; **`-b 2048 -ub 2048` recovers ~2×**
(9B `Q4_K` 8→17 t/s). That per-µbatch dequant is itself **multi-threaded by default**
(`ROCKET_DEQUANT_THREADS`, auto = cores capped at 8; set 1 for the serial path) — the weight
rows are independent, so fanning them across the CPU recovers **+33% @`-ub 512` / +19% @`-ub
2048`** over the serial decode, bit-exact, with no extra RAM (9B `Q4_K`, same-session HW A/B).
Quant *type* is then irrelevant to NPU throughput (`Q4_K` / `IQ4_XS` /
`Q8_0` converge); quant closes toward F16 with M — ~0.64× at pp512 rising to ~0.85× at pp2048 (F16 stays resident, no dequant). **`ROCKET_QUANT_RESIDENT=1`
closes that gap** — it dequant→packs each weight to resident fp16 NPU BOs **once**, so prefill then pays
neither the per-µbatch dequant nor the per-call packB (it reuses the F16 prepacked path, bit-identical PPL),
lifting quant prefill to **F16 parity** and recovering the `-ub 512` halving [HW, 0.8B `Q4_K` pp2048, same
session: resident **89** t/s = F16 **89**, vs streaming **81** @ub2048 and **51** @ub512 — so **1.5×** at the
llama.cpp-default `-ub 512`, and the per-call `packB` collapses 12440→889 ms; 9B `Q4_K` resident **24.6**
vs streaming **15.8** = **1.56×**, ≈0.92× F16 — its ~18 GB fp16 fits the 31 GB RAM + IOVA window]. The cost is the full fp16
resident footprint (it negates the quant RAM saving), so it is opt-in for the RAM-to-spare case; over
`ROCKET_CACHE_MB` / once the NPU IOVA window fills, weights fall back to streaming. On a model larger
than the default 2 GB budget, use **`ROCKET_QUANT_RESIDENT=auto`** (budget sized from free RAM) so it
residents in full — blanket `1` at the default budget residents only part and can **net-lose to
streaming** (9B `Q4_K` pp2048 ub2048: 19.3 vs 22.1 t/s, `auto` 25.8 ≈ F16 25.1; the win grows at the
llama.cpp-default `-ub 512`, where streaming re-dequants 4×: `auto` 26.1 vs streaming 15.3 = **1.7×**
[HW sweep, 600 MHz]). Short
quant prefills below `ROCKET_MIN_M_QUANT` (≈ the measured ~360-row crossover) auto-route to the CPU.
Across models, the **F16 NPU prefill win grows with size then plateaus: 0.8B 1.44× → 9–12B ~3.6×
CPU** — validated on Qwen3.5 / Qwen3.6 (incl. the 27B hybrid-DeltaNet) and Gemma-4-12B (3.6× at
pp512), all PPL-faithful.

Quantized int8 does not speed this backend's prefill, and that is about the datatype's role
here, not a hardware mode the backend is missing. int8 helps prefill only by cutting operand
*bandwidth* — a W8A8 scheme halves both operands' bytes, where this path quantizes weights and
keeps fp16 activations — and by batching more work per kernel submit to lower the dispatch floor.
Neither lifts the MAC ceiling: the matmul is DMA/dispatch-bound, not MAC-bound, so throughput
rises by attacking the dispatch floor (fewer, bigger, batched submits), not by narrowing the
datatype. On-device integer K-accumulation — which would let int8 keep its narrow weights *and*
avoid the int32 readback — does not exist on this silicon.

## Runtime knobs

The backend reads a set of `ROCKET_*` env vars. `sudo` strips the environment — always use `sudo -E`.

| var | default | meaning |
|---|---|---|
| `GGML_BACKEND_PATH` | — | path to `libggml-rocket.so` (how the host loads it) |
| **`ROCKET_KACC`** | **on** | **the operating mode**: fp16 NPU-side K-accumulation (+19%). Default-on; `=0` (or `ROCKET_NO_KACC`) opts out to the byte-exact host fp64-accum path. Under it, `ROCKET_REUSE` defaults to 2 |
| **`ROCKET_REUSE`** | 2 (KACC on by default) | CBUF operand reuse: 0 off / 1 WEIGHT_REUSE / 2 DATA_REUSE (+7%). Defaults to 2 whenever K-accum is on, which is the default |
| `ROCKET_N_THREADS` | 5 | worker count (knee ~5; "one above #cores") |
| `ROCKET_MIN_M` | 128 | min M to offload. Below the crossover the per-call dispatch + weight packing (a weight only goes resident at `max_tile`=256, so below that it re-packs every call) outweigh the NPU's per-row advantage and the offload **loses to the CPU**. Measured F16 ratio NPU/CPU — 0.8B: pp16 0.36 / pp64 0.83 / pp96 1.04 / pp128 1.15; 3B: pp64 1.03 / pp128 1.60; 8B: pp48 0.93 / pp64 1.24 / pp128 1.89. The crossover is nearly model-independent (the packB you pay and the compute you gain both scale with K·N, so it cancels); the residual drift — ~86 rows at 0.8B down to ~55 at 8B — is the dispatch term, which does *not* scale with K·N and so weighs more when the weights are small. **128 is at or above every measured crossover, so no model regresses below CPU**; bigger models cross earlier still. Also covers **batched** decode: whisper.cpp's default beam search presents M=5 per step, which the old floor of 4 wrongly offloaded (2.3× slower; a 1.40× net loss end-to-end). Do **not** set it to 256 — llama streams below 256 and still wins, so 256 costs pp128 −46% |
| `ROCKET_MIN_M_QUANT` | 512 | min M to offload **quantized** weights (the dequant→fp16 path). Its per-microbatch dequant needs more rows than F16 to amortize — measured crossover ~360 on 9B/27B `Q4_K`, so short quant prefills below this stay on the CPU (avoids a net-loss offload). Floored at `ROCKET_MIN_M` |
| **`ROCKET_QUANT_RESIDENT`** | off | dequant a quantized GGUF weight to fp16 **once** and hold it in resident NPU BOs (reuses the F16 prepacked path) instead of re-dequantizing **and** re-packing it every micro-batch — lifting quant prefill to F16 parity (closes the per-microbatch dequant tax, and the `-ub 512` / short-follow-up cases `-ub 2048`'s amortization can't) at the cost of the **full fp16 resident footprint**. Modes: `1` = blanket, bounded by `ROCKET_CACHE_MB` (default 2 GB); **`auto`** = size the budget from free RAM (MemAvailable − a swap-safe reserve `max(6 GiB, 30 % of RAM)`, override `ROCKET_QUANT_RESIDENT_RESERVE_MB`); `N` = an explicit N-MB budget. Prefer `auto` for the "quantized-for-download, RAM-to-spare" case: on a model **larger than the default budget**, blanket `1` residents only part of it and is a **net loss vs streaming** — `auto` reaches parity (Qwen3.5-9B-`Q4_K` pp2048: `auto` 25.8 ≈ F16 25.1 vs streaming 22.1 t/s [HW sweep, 600 MHz]). Over-budget / IOVA-full weights fall back to streaming |
| **`ROCKET_FLASH_ATTN`** | on | offload the attention op (`FLASH_ATTN_EXT`) to the NPU: heads fanned across the worker fds + submit-chained (see `ROCKET_FA_CHAIN`), bit-faithful (PPL == CPU) **for the attention it implements** — `softmax(scale·QKᵀ + mask)·V`. An op carrying **attention sinks** (`src[4]`, a learned per-head logit in the softmax denominator — gpt-oss, `mimo2`, `deepseek4`) is **declined**: the handler has no sink term, so accepting it would compute a *different* attention, silently. **Parity at ≤1K, a growing win above — 1.07× at 4K, 1.50× at 8K, 1.25× at 16K** [HW sweep, `-r3`]. Context-gated (see `MIN_KV`); `=0` disables |
| `ROCKET_FA_CHAIN` | on | batch each worker's per-head QK (and AV) submits into one NPU job through a resident batched-matmul context — the dispatch-floor lever that pulls the crossover in to ~2K; `=0` forces the per-head path. Bit-identical |
| `ROCKET_FLASH_ATTN_MIN_KV` | 1024 | min `n_kv` (context length) to offload an attention op — 1024 ≈ the sliding-window length, so the windowed local layers offload while shorter prompts stay on CPU. Gates on `n_kv`, not `n_tokens` (every ubatch has `n_tokens≈512` regardless of total context) |
| `ROCKET_FLASH_ATTN_MIN_T` | 16 | min prefill `n_tokens` to offload attention (single-token decode stays on CPU) |
| `ROCKET_ATTN_HOST_SOFTMAX` | host (FA) | attention softmax placement; `=0` forces the on-NPU softmax (default host — scores are already host-side for the additive mask) |
| `ROCKET_FA_TIMING` | off | print the FA handler's host split at exit — gather / on-NPU compute / scatter ms (the `FLASH_ATTN_EXT` outer gather is host glue the driver's `ROCKET_MM_PROFILE` does not see). Diagnostic only; near-zero when off |
| `ROCKET_MOE` | **off (opt-in)** | `=1` offloads MoE routed-expert FFNs (`MUL_MAT_ID`) to the NPU: bucket the `(slot,token)` rows by expert id, run each active expert's `[M_e,K]×[N,K]ᵀ` GEMM fanned across the worker fds, scatter the rows back. A **quantized** expert takes the native-quant route by default (`ROCKET_MOE_NATIVE`), which holds it resident as int8 and is worth **2.16× the CPU** at pp2048 on gpt-oss-20b MXFP4 (1.34× at pp512 — it wins at every length). Run it at **`-b 2048 -ub 2048`** — not because the experts re-dequantize (native-quant is what stops that) but because the **dense** quantized weights still do, and because a smaller micro-batch gives each expert proportionally fewer rows (`n_tokens · n_used / n_expert` — 64 at `-ub 512` vs 256 at `-ub 2048`) while the per-expert dispatch/gather/scatter/padding stays flat; not measured at `-ub 512`. Opt-in **not** because it is slow but because the win is conditional on nearly the whole expert stack fitting RAM — 99% resident wins, 82% resident *loses* at pp512 — so its sign depends on the machine, and a default whose sign depends on the machine is not a default. Also costs a one-time ~70 s expert ingest per `llama_context`. See the MoE note above |
| `ROCKET_MOE_MIN_TOKENS` | 512 | with `ROCKET_MOE=1`, the min micro-batch `n_tokens` for a `MUL_MAT_ID` op to offload — `M_e ≈ n_tokens · n_expert_used / n_expert` sets the per-expert GEMM size; short prefills and decode stay on the CPU. Floored at `ROCKET_MIN_M` |
| `ROCKET_MOE_NATIVE` | **on** (within `ROCKET_MOE=1`) | route a **GGUF-quantized** expert through the resident int8 group-wise path: ingest its quant blocks **once** into int8 codes held in NPU BOs, then quantize only the activation per call. This is what removes the per-micro-batch host dequant that makes the fp16 expert route a loss. `=0` forces the fp16 dequant route (the A/B baseline). An **F16** expert always takes the fp16 route — it has no dequant to delete |
| `ROCKET_MOE_CACHE_MB` | **auto** | resident native-quant **expert** budget in MB (`0` = unlimited). Auto = `MemAvailable` − reserve. Charged per expert as *int8 codes + the expert's GGUF source bytes*, because the GGUF is mmapped and cannot be reclaimed (MoE **decode** reads the active experts from it on the CPU every token), so both copies must coexist. Admission-only: an expert that does not fit streams on the fp16 route, correctly, and the split is logged at teardown (`ggml_backend_rocket_moe_stats`) |
| `ROCKET_MOE_GROUP` | **auto** | the K-group the native-quant path quantizes on. Auto picks the largest divisor of `K` that is a multiple of 32 and that the CBUF can hold as one K-tile — the readback floor (gpt-oss `K=2880` → **576**, `nKt=5`). Readback scales as `K/group` and this path is readback-bound, so a finer group is more faithful and proportionally slower; the knob exists for that A/B |
| `ROCKET_MOE_M_BUCKET` | 64 | granule the ragged per-expert row count `M_e` is rounded up to (a power of two; ≤12% padded rows at `M_e ≈ 256`). Not a tuning knob so much as a requirement: `M%4` is a hardware contract, and the driver caches its resident scratch per `(M,K,N,group)` in a fixed-size table that a distinct `M` per expert would exhaust mid-prefill. The granule self-coarsens if that table starts to fill |
| `ROCKET_INT8` (+ `ROCKET_INT8_HADAMARD`) | off | W8A8 int8 path (coherent with Hadamard; net loss vs fp16+KACC) |
| `ROCKET_INT8_RESIDENT` | off | resident int8 weights (on top of `ROCKET_INT8`) |
| `ROCKET_INT8_CACHE_MB` | 4096 | resident rotated-int8 weight cache cap, measured against the actual resident NPU-BO tile footprint (over-budget weights fall back to fp16) |
| **`ROCKET_INT4`** | off | native W4A4 int4 path (from an F16 GGUF); group-wise + Hadamard on by default. Char-identical to fp16; a RAM play, not faster |
| **`ROCKET_INT4_RESIDENT`** | off | hold the int4 weights in **resident NPU nibble BOs** (~¼ the fp16 footprint, no per-call weight scatter) instead of re-packing per call. Group-wise only (`ROCKET_INT4_GROUP>0`, the default); per-channel / over-budget / IOVA-full weights fall back to the one-shot int4 path |
| `ROCKET_INT4_GROUP` | 128 | int4 K-group scale granularity (must divide K, %32; `0` = per-channel). Smaller = higher fidelity, more readback |
| `ROCKET_INT4_HADAMARD` | 1 | int4 outlier rotation (near-mandatory: cos ~0.79 → ~0.98 with it). `0` disables |
| `ROCKET_INT4_CACHE_MB` | 4096 | int4 weight budget in MB (caps the one-shot host cache AND the resident NPU-BO total; over-budget weights fall back). A full 12B's int4 resident BOs are ~6 GB, so raise this for full coverage |
| `ROCKET_FORCE_PREPACK` / `ROCKET_NO_PREPACK` | — | force / disable the resident-weights path |
| `ROCKET_PREPACK_MADVISE` | off | madvise the F16 source after packing (**prefill-only — breaks CPU decode**) |
| `ROCKET_CACHE_MB` | — | resident-weight byte budget in MB (the lower-level knob `ROCKET_QUANT_RESIDENT=auto`/`N` set; an explicit value here wins over both) |
| `ROCKET_NO_FUSE` | off | disable gate/up graph fusion |
| `ROCKET_VERIFY` | off | per-op NPU-vs-CPU correctness gate (`max_abs` is the signal; ignore `max_rel`; very slow) — needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_MM_PROFILE` | off | bucket breakdown at exit (adds noise → drop for headline t/s) |
| `ROCKET_TRACE` / `ROCKET_DEBUG` / `ROCKET_DEBUG_GRAPH` | off | per-op weight/ptr/amax trace; fusion groups; split census (`ROCKET_TRACE` needs `-DGGML_ROCKET_DIAGNOSTICS=ON`) |
| `ROCKET_WAIT_MS` | 8000 | output-fence wait deadline |
| `ROCKET_LOG_STDERR` | off | tee the driver/backend log channel to stderr even when the host silences its `ggml` logger (e.g. `llama-bench` without `-v`) — see Diagnostic logging |

The heavy recompute/trace diagnostics (`ROCKET_VERIFY` / `ROCKET_CPU_FORWARD` / `ROCKET_TRACE` /
`ROCKET_AB` / `ROCKET_DUMP`) are compiled out of the production `.so`; build with
`-DGGML_ROCKET_DIAGNOSTICS=ON` to enable them. `ROCKET_MM_PROFILE` / `ROCKET_DEBUG` /
`ROCKET_DEBUG_GRAPH` are always available.

K-accum is on by default (DATA_REUSE follows automatically); `ROCKET_KACC=0` opts out.

## Diagnostic logging

Every diagnostic the backend emits — errors, warnings, the `ROCKET_MM_PROFILE` /
`ROCKET_FA_TIMING` breakdowns, and the env-gated traces above — flows through ggml's log
channel, so the host (llama.cpp, whisper.cpp) redirects or silences it with the same
`ggml_log_set` callback it uses for its own output. Severity maps the obvious way: failures
to `ERROR`, recoverable fallbacks to `WARN`, the profile breakdowns and one-time hints to
`INFO`, the per-op traces to `DEBUG`. The env knobs above still gate whether a trace is
emitted at all; the channel only governs where it goes.

The driver it links (librocketnpu) has its own log channel, which the backend bridges into
ggml's at startup — so a driver message reaches the same host callback rather than landing
on raw stderr. The driver's own threshold applies first, so a driver `DEBUG` trace needs
both its driver-side gate (`ROCKET_DEBUG`) and a host logger that prints `DEBUG`.

One consequence: a host that *silences* its `ggml` logger silences the backend too. `llama-bench`
installs a no-op `ggml` callback unless run with `-v`, so its default run hides every backend
line — including one-shot decisions such as the `ROCKET_QUANT_RESIDENT=auto` resident-weight
budget, which changes the benchmarked prefill number and so is exactly what you want to see. For
that case the backend routes its user-facing config lines (the budget decision, the "raise `-ub`"
quant hint) through the driver channel, so `ROCKET_LOG_STDERR=1` tees them to stderr regardless of
the host's logger. Set it alongside a benchmark to confirm which budget/mode actually engaged.

## Implementation notes

- Uses ggml's private headers (`ggml-impl.h`, `ggml-backend-impl.h`) — expected for an
  out-of-tree backend. The vtables (`ggml_backend_i` etc.) are initialized
  positionally, so the backend must be built against the **host's bundled ggml**
  (field order + `GGML_BACKEND_API_VERSION` must match). The concrete ABI this backend
  targets is **`GGML_BACKEND_API_VERSION 2`, with the device vtable that *includes* the
  `set_tensor_2d_async` / `get_tensor_2d_async` slots** (built/verified against the
  in-repo `ggml/`, tagged v0.14.0). Because the host apps (llama.cpp / whisper.cpp)
  clone their own ggml, build this backend against **that checkout's** headers and
  re-check on every bump — if the host's `ggml-backend-impl.h` has a different
  `GGML_BACKEND_API_VERSION` or 2d-field layout, the positional vtable drifts and you
  get a "ROCKET device not listed at startup" failure.
- `ggml_fp16_t` (uint16_t) shares the IEEE-half bit layout with `_Float16`, so the
  driver's `_Float16*` buffers are filled via `ggml_fp32_to_fp16_row` and cast.
- **Why offload happens:** ggml's scheduler only pulls a host-resident op onto an
  ACCEL device when **both** `supports_op` and `offload_op` return true (BLAS does the
  same). A NULL `offload_op` makes the device appear at startup but run zero matmuls.
</content>
</invoke>
