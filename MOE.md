# ggml-rocket: MoE and MLA models

Mixture-of-experts models compute their expert FFNs with `MUL_MAT_ID`. The backend's `MUL_MAT_ID`
handler groups the routed `(slot,token)` rows by expert and runs each active expert's
`[M_e,K]x[N,K]ᵀ` GEMM on the NPU, spread across its workers. `M_e` is the number of rows routed to
that expert. The handler then scatters the rows back into the output.

Each comparison names its baseline. The experts-on-CPU baseline is the backend with
`ROCKET_MOE=0`, which offloads the dense matmuls and leaves the experts on the CPU. The models
measured are gpt-oss-20b
and DeepSeek-V2-Lite. Unless a Conditions line says otherwise, a figure here was measured on the
RK3588 with the NPU at ~816 MHz (the 600 MHz request).

A MoE model's architecture decides whether its experts offload. One model's ratio does not
transfer to another. The default needs no environment variable. Where the experts offload, add
`-b 2048 -ub 2048` and `ROCKET_N_THREADS=6`, as [Settings](#settings) explains.

## Results

### gpt-oss-20b

With the default placement, gpt-oss-20b (MXFP4) prefills at 37.8 t/s, 2.29x the CPU backend with
llama.cpp's weight repack on, at `pp2048`. The default places 62 of its 72
[expert stacks](README.md#terms), every placed expert resident.

**Conditions:** NPU 928 MHz (the 900 MHz request), CPU governor `performance`,
`-b 2048 -ub 2048`, llama.cpp b11539 [HW sweep 2026-11-01].

An earlier expert route compares the configurations as follows on one build:

| Test | CPU, unrepacked, t/s | NPU, `ROCKET_MOE=0`, t/s | NPU, default, t/s | NPU, `ROCKET_MOE=1`, t/s |
|---|---:|---:|---:|---:|
| `pp512` | 12.17 | 14.08 | 22.0 (1.81x) | 24.81 (2.04x) |
| `pp2048` | 11.80 | 13.62 | 28.1 (2.38x) | 34.01 (2.88x) |

**Conditions:** clock pinned, governor `performance`, `-b 2048 -ub 2048`, one board and one
backend `.so` throughout [HW sweep 2026-08-27]. Ratios are against the unrepacked CPU backend.

`ROCKET_MOE=1` is 6-13% faster than the default at `pp512` and 18-21% at `pp2048`, because it
reserves nothing and places all 72 stacks. The default gives up that peak for a guarantee: no
expert is streamed, and prefill stays at or above the experts-on-CPU baseline at every budget.
On a board with RAM to spare, raising `ROCKET_MOE_CACHE_MB` recovers most of the peak with the
guarantee kept, as [Raising the budget](#raising-the-budget) shows.

At `pp512` and the default `-ub 512`, the default placement prefills at 31.3-34.1 t/s. That is
1.97-2.00x the CPU backend with its weight repack on, and 2.17-2.21x the experts-on-CPU baseline,
which reads 0.90-0.91x that CPU backend. The NPU's repetitions rise within each run: an expert
the warm-up never selected ingests inside a timed repetition. The ratio is therefore a lower
bound.

**Conditions:** NPU 928 MHz (the 900 MHz request), CPU governor `performance`, `-t 8`,
`--repack 0` on the NPU, llama.cpp b11539, three passes with the order rotated
[HW sweep 2026-11-02].

### DeepSeek-V2-Lite

DeepSeek-V2-Lite combines Multi-head Latent Attention (MLA), with DK 192 and DV 128, with routed
experts. The backend claims a `FLASH_ATTN_EXT` whose DK and DV differ, so MLA attention offloads at
`n_kv` >= 1024 whenever the host runs flash attention (`-fa`). With the experts on the NPU at
`-b 2048 -ub 2048` and 928 MHz, `-fa on` prefills 1.04-1.05x as fast as `-fa off` at `pp2048`.
At `pp4096` it is 1.10-1.11x [HW sweep 2026-11-02]. Its output was not compared against the
CPU's on this model.

With the routed experts on the CPU, the backend offloads the large MLA projections, the two
always-on shared experts and `lm_head` as ordinary `MUL_MAT` ops. Prefill is then 1.18-1.26x the
unrepacked CPU backend:

| Configuration | `pp512` | `pp1024` | `pp2048` | `tg64` |
|---|---:|---:|---:|---:|
| CPU, unrepacked | 20.37 | 19.92 | 19.01 | 7.73 |
| NPU, experts on the CPU | 24.04 | 24.85 | 23.87 | 7.77 |

**Conditions:** `Q4_K_M` (15.71 B parameters, ~2.4 B active), `-b 2048 -ub 2048`, t/s
[HW sweep 2026-07-03].

With the experts on the NPU as well, DeepSeek-V2-Lite loses at `pp512` and gains at `pp2048`:

| Test | CPU, t/s | NPU, `ROCKET_MOE=0`, t/s | NPU, experts on the NPU, t/s |
|---|---:|---:|---:|
| `pp512` | 19.54 | 26.04 | 19.09 (0.73x) |
| `pp2048` | 18.50 | 21.63 | 27.9 (1.29x) |

**Conditions:** governor `performance` [HW sweep 2026-08-27]. Ratios are against
`ROCKET_MOE=0`.

The `pp512` loss is 27% at 100% residency, so the residency report does not show it.
DeepSeek-V2-Lite routes 6 of 64 experts, so 512 tokens give each one ~48 rows, below the 64-row
minimum bucket its GEMM is padded to. The default placement therefore declines the routed experts
at `pp512`, where it measures 25.9 t/s, the experts-on-CPU figure, and offloads them at `pp2048`.

At `-c 2048`, where `M_e` is 192, the routed experts on the NPU change perplexity by −0.12% ±
0.10% against `ROCKET_MOE=0` [HW sweep 2026-08-27]. That difference is within two standard
errors of zero. Against the CPU
backend the default measures −0.42%, which includes the dense offload's −0.30%. The absolute
perplexity, ~8.2, is meaningful because this is a base model.

With the default placement and every routed expert resident, DeepSeek-V2-Lite prefills at 43.4
t/s at `pp2048` [HW sweep 2026-11-01]. Against the CPU backend with llama.cpp's weight repack on,
under gpt-oss-20b's conditions above, the speedup is 1.83x.

## Settings

### Micro-batch size

Run a model whose experts offload at `-b 2048 -ub 2048`. At `pp2048`, it measures 1.254x the
default `-ub` on gpt-oss-20b and 1.323x on DeepSeek-V2-Lite [HW sweep 2026-08-29]. A smaller
micro-batch gives each expert fewer rows: 64 at `-ub 512` against 256 at `-ub 2048` on
gpt-oss-20b. The fixed cost of each dispatch stays the same. The dense quantized weights are
also dequantized on every micro-batch.

Keep the default `-ub` for a model whose experts stay on the CPU. The backend declines
Qwen3-30B-A3B's experts, and `-ub 2048` measures 0.908x the default there. The pre-flight's
startup line says which stacks the backend placed.

### Worker count

Set `ROCKET_N_THREADS=6`. On the native-quant route six workers measure 1.022x the default five on
gpt-oss-20b at `pp2048`, and seven and eight measure no more. Each worker past six adds
oversubscription cost without adding speed. The gain does not come from placement. The same
stacks are placed at every count, because RAM binds before the
[IOVA window](README.md#terms) on this route ([The IOVA window](#the-iova-window)).

### The budget

`ROCKET_MOE_CACHE_MB` sets the memory the expert stacks can take, and which way to move it
depends on the expert GGUF's size against RAM. Raise it where the GGUF is small against RAM, as
gpt-oss-20b's 11.3 GiB is on a 31 GiB board. Lower it where the GGUF is large, as
Qwen3-30B-A3B's 17.28 GiB is. [The expert budget](#the-expert-budget) gives the arithmetic.

## Placement

### Expert routes

An expert reaches the NPU by one of two [routes](README.md#terms), and only the
[native-quant](README.md#terms) route is faster than leaving the experts on the CPU:

- The fp16 dequant route (`ROCKET_MOE_NATIVE=0`) dequantizes each expert weight to fp16 on the
  host every micro-batch. A MoE layer has `n_expert` times as many distinct weights as a dense
  layer, each amortized over only the rows routed to it. The dequantization therefore costs more
  than the GEMM saves. On gpt-oss-20b it measures 4.59 t/s at `pp512` and 10.18 at `pp2048`, below the
  experts-on-CPU baseline. The default placement never claims it.
- The native-quant route (`ROCKET_MOE_NATIVE`, on by default) converts a GGUF-quantized expert
  once into int8 codes held [resident](README.md#terms) in NPU buffers. Each call then quantizes
  only the activation, per `(row, K-group)`. An F16 expert always takes the fp16 route, because it
  has nothing to dequantize.

### The `ROCKET_MOE` setting

`ROCKET_MOE` selects which `MUL_MAT_ID` ops the backend [claims](README.md#terms). In every
state, a micro-batch below `ROCKET_MOE_MIN_TOKENS` (512 tokens) stays on the CPU, which keeps
decode and short prefills there.

| `ROCKET_MOE` | Placement |
|---|---|
| unset (default) | On the RK3588, claims an expert op only where the offload is measured to be faster. The stack must be GGUF-quantized, the [pre-flight](README.md#terms) must reserve the whole stack, and each expert's GEMM must meet both size minimums, `ROCKET_MOE_M_BUCKET` and `ROCKET_MOE_MIN_WORK`. |
| `1` | Claims every `MUL_MAT_ID` the handler can compute, reserving nothing and applying neither size minimum. That includes the fp16 route for an F16 stack or one too large to hold resident, which measured 0.25-0.82x the CPU backend's prefill on DeepSeek-V2-Lite and gpt-oss-20b. It is a comparison configuration. |
| `0` | The experts stay on the CPU, whatever the shape. |

The unit of placement is one expert stack. A model whose whole expert set does not fit still
offloads the stacks that do. gpt-oss-20b runs 99% resident on a 31 GiB board, where its 19.1 GiB
of int8 codes and its 11.3 GiB GGUF do not both fit.

A declined stack runs on the CPU, so its weights are never dequantized per call. The result blends
the offloaded route and the experts-on-CPU baseline, and stays at or above the baseline at any
RAM size. A declined stack also costs no copy. The backend's buffer type is the CPU buffer type, so
a CPU-placed op beside an offloaded one is only a scheduler entry.

### Pre-flight

The pre-flight runs in `supports_op`, before the first ingest, because a stack's resident cost
follows from its tensor alone: `[K, N, n_expert]` and the GGUF source stride. It charges the stack
against RAM and against the IOVA window. Each worker has a 4 GB IOVA window, of which the
pre-flight budgets 3840 MB. A stack that does not fit either budget stays on the CPU whole.

Under the default, no expert of a claimed stack is [streamed](README.md#terms), so the teardown
report shows 100% resident. A low residency means the pre-flight declined stacks, and its
`resident budget reached` line names the budget that stopped it. Below 80% resident, the backend
warns at teardown.

### Size minimums

Residency alone does not make the offload faster. The rows an expert receives,
`M_e = n_tokens · n_used / n_expert`, depend on the architecture's routing. gpt-oss routes 4 of 32
experts, so 512 tokens give each expert ~64 rows. DeepSeek-V2-Lite routes 6 of 64, so the same 512
tokens give ~48. The default placement applies two minimums to each expert GEMM:

- The bucket minimum, `ROCKET_MOE_M_BUCKET` (64 rows). The backend rounds `M_e` up onto a ladder
  of buckets, 64, 96, 128, 192, 256 and on, and computes the pad rows in full. An expert GEMM below
  the smallest bucket is mostly padding.
- The [work per dispatch](README.md#terms), `ROCKET_MOE_MIN_WORK` (340 million MACs). Each
  dispatch has a fixed cost in gather, activation quantization, submit, fence and scatter. The
  point where the offload gains is therefore a threshold on the work it does.

The ladder steps twice per octave, so padding is at most ~33% of `M_e` and ~15% on average. A
model uses ~15 distinct row counts. `librocketnpu` keeps scratch for 32 `(M,K,N,group)` shapes per
context and reallocates the least recently used one past that.

The row count alone cannot separate the two models, because their expert GEMMs differ in size:

| `M_e` | 64 | 72 | 96 | 128 | 144 | 192 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss (2880×2880) | 1.64x | n/a | 1.77x | 1.91x | n/a | 2.05x | 2.08x |
| DeepSeek (2048×1408) | n/a | 0.948x | 1.049x | n/a | 1.225x | 1.313x | n/a |

**Conditions:** clock pinned, governor `performance`, `-p 512` to `-p 2048`, every expert resident.
Ratios are the default placement against the experts-on-CPU baseline [HW sweep 2026-08-27].

gpt-oss-20b gains 1.64x at `M_e` = 64, while DeepSeek-V2-Lite loses 5% at 72. DeepSeek-V2-Lite's
expert GEMM is 2.88x smaller than gpt-oss's at the same row count. Sorted by work per dispatch, the two
architectures' configurations separate with no overlap.

The architecture decides which minimum applies. On gpt-oss-20b the bucket minimum always decides,
because the work minimum there sits at `M_e` = 41. On DeepSeek-V2-Lite the work minimum decides,
and it admits from `M_e` >= 118, past ~1250 tokens in a micro-batch. The default of 340 is the
geometric midpoint between two measured DeepSeek-V2-Lite shapes, 1.049x at 277 million MACs and
1.22x at 415 million.

**Not characterized:** an architecture whose work per dispatch falls between 277 and 415 million
MACs.

Qwen3-30B-A3B, a third architecture, routes 8 of 128 experts of 2048×768. At `-ub 2048` its `M_e`
is 128 and its work 201 million MACs, so the bucket minimum accepts it and the work minimum
declines it. The default measures 1.000x its own `ROCKET_MOE=0` configuration and ingests nothing.
`ROCKET_MOE=1` measures 0.834x.

The work minimum sits above a 1.00x offload because of the [ingest](README.md#terms): ~18 s per
`llama_context` on DeepSeek-V2-Lite and ~21 s on gpt-oss-20b. An offload `r` times as fast as the
baseline saves `1 − 1/r` of prefill time. It therefore repays the ingest after ~7700 tokens of
prefill at 1.05x, ~2300 at 1.22x and ~1100 at 1.64x [host-computed].

### Partial residency

The native-quant route's gain needs nearly the whole expert stack resident, and below that the
loss is a cliff. A streamed expert costs a dequantization that does not shrink with its row
count, while a resident expert's GEMM does. The streamed remainder's share of prefill time
therefore grows as the prefill shortens: 12% of `pp2048` and 43% of `pp512`.

> [!NOTE]
> A residency percentage is not a cost percentage. On the build that first held experts resident
> (2026-07-14), 82% resident measured 12.19 t/s at `pp512`, below the 14.11 t/s with the experts
> on the CPU.

Only `ROCKET_MOE=1` reaches partial residency, because the pre-flight under the default reserves
whole stacks. At high residency, streaming a few experts costs less than leaving whole stacks on
the CPU. `ROCKET_MOE=1` at 91% resident was the fastest configuration measured on a 31 GiB board.
At an induced 12 GB budget it falls to 52% resident and measures 13.63 t/s at `pp512`, below the
14.08 with the experts on the CPU. The default places 30 whole stacks at that budget and measures
17.02 t/s, 1.21x the baseline.

**Not characterized:** the residency between 52% and 91% at which `ROCKET_MOE=1` falls below the
experts-on-CPU baseline.

## The expert budget

`ROCKET_MOE_CACHE_MB` sets the budget in MB that the pre-flight and the ingest both charge. `0` is
unlimited. The default, `auto`, is `MemAvailable` minus a flat 6 GiB reserve. Unlike the dense
residency routes, whose reserve is `max(6 GiB, 30% of RAM)`, it has no proportional term. It
therefore withholds 19% of 32 GB, 37.5% of 16 GB and 75% of 8 GB. At 8 GB, `auto` effectively disables the
route [expected].

> [!NOTE]
> `ROCKET_MOE_CACHE_MB` bounds the admission charge, not the RAM the experts occupy. A value that
> suits one quantization or one board does not transfer to another by RAM arithmetic alone.

### Charge factor

Per expert, the admission charge is `N·K` bytes of int8 codes, `N·(K/group)·4` bytes of scales,
and the expert's GGUF source stride. The source bytes are in the charge because the GGUF stays
mapped: MoE decode reads the active experts from it on the CPU every token. A budget therefore
holds less resident code than it names, by the charge factor `1 + 4/group +
source_bits_per_weight/8`.

For Qwen3-30B-A3B `Q4_K_M` (4.87 bits per weight, `K`=2048, group 512) the factor is 1.617. For
the same placement it is about 2.07 for a `Q8_0` MoE and 1.54 for `IQ4_XS` [expected].

When the budget stops admission, the pre-flight logs
`resident budget reached after N expert stacks (XMB RAM of which SMB is GGUF source, YMB IOVA)`.
Under the default charge, `X/Y` is the charge factor as the run applied it. It is 1.610 on
Qwen3-30B-A3B and 1.521 on gpt-oss-20b MXFP4, against 1.617 and 1.538 derived. The formula sizes a
budget to within ~1% before a run.

### Raising the budget

Raising the budget places stacks that the pre-flight's whole-stack reservation declines, and it
keeps the default's guarantees. On gpt-oss-20b, more budget places more stacks and makes prefill
faster:

| `ROCKET_MOE_CACHE_MB` | Stacks placed, of 72 | `pp512` | `pp2048` |
|---|---:|---:|---:|
| `auto` (24672) | 63 | baseline | baseline |
| 26000 | 66 | +4.3% | +5.1% |
| 28000 | 71 | +8.3% | +14.1% |

**Conditions:** 31 GiB board, gpt-oss-20b, no streamed experts in any configuration, gains against
`auto` [HW sweep 2026-08-27].

At 28000 MB the default reaches 79% of `ROCKET_MOE=1`'s gain at `pp2048`, and exceeds it at
`pp512`. It leaves ~2.8 GB of the headroom the 6 GiB reserve keeps, so it suits a known working
set rather than a default.

### An expert GGUF large against RAM

On a model whose expert GGUF is large against RAM, raising the budget can cost the whole gain. On
Qwen3-30B-A3B, whose experts hold 29 of its 30.5 B parameters in a 17.28 GiB GGUF, the gain holds
on a plateau and then drops. `auto` is already past the plateau:

| `ROCKET_MOE_CACHE_MB` | Stacks placed | Prefill against the experts-on-CPU baseline |
|---|---:|---:|
| 12000 | 38 | 1.032x |
| 18000-21000 | 58-67 | 1.046x |
| `auto` (24425) | 79 | 1.014x |
| 28000 | 90 | 0.999x |

**Conditions:** 31 GiB board, Qwen3-30B-A3B, `-ub 4096` [HW sweep 2026-08-28].

`auto` overshoots because the default charge omits the GGUF source of the stacks left on the CPU.
The CPU reads those experts from the same mapping, so the working set outgrows RAM before the
budget shows it. Size the budget for such a model through the charge factor:

    budget_MB ~ (1 + source_bits_per_weight/8) * (MemTotal − whole expert GGUF − runtime headroom)

On this board that is `1.610 × (31.7 − 17.3 − 1.2 GB)` = 21.3 GB, on the measured plateau, with the
headroom term chosen to match it.

**Not characterized:** the GGUF size, between gpt-oss-20b's and Qwen3-30B-A3B's, at which raising
the budget turns from a gain into a loss.

### Charging the whole mapping

`ROCKET_MOE_CHARGE_ALL_SOURCE=1` charges the whole mapped weight buffer once, up front, and leaves
only the codes and scales in each stack's charge. The pre-flight then also counts the source of
the stacks it declines. On the 31 GiB board it places about 46 Qwen3-30B-A3B stacks against the
default's 79, below the 58-67 plateau. On gpt-oss-20b it places about 58 of the 63 that `auto`
places [expected], which costs speed on a model that gains with every stack.
`test-rocket-placement` checks the contrast: at one budget over four stacks in one buffer, the
default admits 3 and the variable 2.

### The budget across processes

`auto` is resolved once per process, at the first `supports_op` or backend initialization, and
then frozen. It still varies between processes. On one board, the same DeepSeek-V2-Lite run saw
24.6 GB early and 20.7 GB after hours of multi-GB allocation churn. Those budgets admit all 78 of
its expert stacks and 70 of them. A long-lived server that builds a fresh `llama_context`
late in its life can therefore place fewer stacks than at startup, with no warning. The teardown
line from `ggml_backend_rocket_moe_stats` is the only place the difference shows.

To make placement reproducible, pin `ROCKET_MOE_CACHE_MB` to a value the board can hold, or drop
the page cache, as root, before the run:

```sh
echo 3 > /proc/sys/vm/drop_caches
```

### Several live backends

Each live backend holds its own resident copy of a placed stack, and the pre-flight charges each
copy against the same budget. Placement is decided once per process. A stack is identified by a
sampled fingerprint of its bytes, so two models with the same tensor names are two stacks. The
GGUF source is charged once per stack, because both backends read one mapping. A copy that does
not fit is refused, and that backend dequantizes the stack to fp16 on every call, correctly and
more slowly. A freed backend releases its copies after its resident buffers.

## The native-quant route

The native-quant route's speed comes from removing the per-micro-batch dequantization and weight
pack, not from int8 arithmetic. Its int32 output reads back at 8 bytes per element, more than the
equivalent fp16 GEMM moves. The route is therefore at most as fast as a resident fp16 expert.

The route applies a scale per K-group. The NPU contracts the whole of `K` at its output stage, and
the backend has no on-device integer [K-accumulation](README.md#terms). The integer partials
therefore leave the chip at every K-tile boundary. The backend keeps each K-tile inside one quant group and
applies the group's scale in the readback loop, at no added cost. `ROCKET_MOE_GROUP=0` picks the
largest group that fits the [CBUF](README.md#terms) as one K-tile: 576 for gpt-oss-20b's `K`=2880.

MXFP4 converts to int8 exactly. Its codebook reaches only ±12, and its E8M0 block scale is an
exact power of two. Merging a K-group's 32-element blocks onto one exponent is therefore an
integer shift. Every other quantized type is dequantized once and requantized. `test-rocket-moe` checks
the route against the CPU backend with outlier-channel activations at cosine 0.9999 for MXFP4,
`Q4_K` and `Q8_0`.

Residency is admission-only: a converted expert stays resident, and the backend evicts nothing.
`ggml_backend_rocket_moe_stats` reports the resident and streamed counts, and the backend logs
them at teardown.

### Expert ingest

The backend converts and packs the experts once per `llama_context`, lazily, inside the first
prefill:

| Model | Experts | Conversion to int8 | NPU buffer pack | Total |
|---|---:|---:|---:|---:|
| gpt-oss-20b | ~1575 | 11.2-11.8 s | 9.4-9.9 s | ~21 s, 13 ms each |
| DeepSeek-V2-Lite | 4746 | 6.4-6.6 s | 11.7-11.8 s | ~18 s |

**Conditions:** governor `ondemand`, `-b 2048 -ub 2048` [HW sweep 2026-10-09].

The pack runs at ~1.3 GB/s, so its time grows with the bytes held resident. With
`ROCKET_LOG_STDERR=1`, the backend prints an `[moe-int8] ingesting experts` line every 256 experts,
so the ingest does not look like a hang.

`llama-bench` builds a fresh `llama_context` for each test row, so it runs the ingest once per row,
and its prefill figure includes part of it. Its warm-up ingests every expert it routes to, and an
expert first routed in a timed repetition ingests inside that repetition. Compare MoE figures only
between runs whose ingest costs the same, or read the third and later repetitions. A long-running
host such as `llama-cli` or `llama-server` runs the ingest once.

### The IOVA window

Each worker's file descriptor has its own IOVA window, and the pre-flight charges each stack
`codes + scales + source` against RAM and `codes` alone against the window,
`ROCKET_N_THREADS x 3840 MB`. The window binds first only where `RAM budget / IOVA budget` exceeds
the charge factor. On gpt-oss-20b MXFP4 that ratio is 1.28 at five workers, against a factor of
1.538, so RAM binds, and more workers place no more stacks. At five workers the window binds only
with `MemAvailable` above 35.7 GB.

On this route, read the pre-flight's line, and raise `ROCKET_MOE_CACHE_MB` to place more stacks.
The F16 residency route behaves differently: its window can bind, and
[TUNING.md](TUNING.md#f16-weights) gives its worker count.
