# ggml-rocket: measurement record

This record states how the figures in the other documents were measured. It gives the comparison
method, the run and pass counts and the spread, and the figures a later measurement replaced. Each section follows the document it serves.

## Method

A comparison between two configurations on the RK1 runs interleaved passes with the
configuration order rotated, and pairs each ratio within a pass. The board's memory state moves a
single process by up to 8-13%, so one process per configuration does not settle a sign. A memory
reset (`drop_caches` and `compact_memory`) precedes each timed configuration. "Three
rotated passes" below means that protocol with three passes.

The CPU governor is pinned to `performance` unless a figure says `ondemand`. An offloading
process blocks with its threads off the run queue. An unpinned governor then parks the big cores
and biases a comparison against the NPU.

## README

The headline table, the interactive turn, and the language-model figures dated 2026-11-01 in
[MODELS.md](MODELS.md) and [MOE.md](MOE.md) are one measurement series on the RK1. It ran
llama.cpp b11539 with the backend built from the repository's 2026-10-29 state, the NPU at
928 MHz and the CPU governor at `performance`.

Each configuration of a GGUF ran in its own process from a memory reset. An NPU configuration first
ran a discarded warm-up process. `llama-bench` ran three repetitions on the NPU and one on the CPU.
The series made two passes, and the second reversed both the GGUF order and the configuration
order.

Each speedup is the mean of the two per-pass ratios. The two ratios of a table entry agree within
2% on 24 of the 27 entries, and within 5.6% on all of them. Every configuration completed.

On the MoE rows the mean of the three repetitions agrees with the last repetition within 0.3%. The
one-time expert conversion therefore does not reach the timed repetitions.

The 3-12 B F16 configurations stream their weights. By default the F16 residency admits only a
weight with `K` <= 2048, and these models have none. SmolVLM2's F16 configuration holds 130 of its 142
weights resident under the default 2048 MB budget. The quantized Ministral-3-8B configurations
hold all 235 weights resident, and the Gemma-4-12B ones 298-310 of 325.

The figures this series replaced were measured at ~816 MHz. The F16 rows were 2026-07 sessions
and the quantized ones 2026-09-29 on llama.cpp b11242. They gave 1.9-3.5x on the dense rows and
1.44-1.90x on the MoE rows. Their NPU rates were 0.45-0.69x this series' on the F16 rows and
0.76-0.83x on the quantized ones. The CPU rates the old ratios imply are within about 4% of this
series'. The series does not decompose the NPU's rise among the clock, CBUF weight streaming, the
host-cost cuts and the governor.

The decode column was 8-thread decode before. On b11539 the 8-thread CPU backend decodes slower
than 4 threads on the A76 cores on every GGUF measured. The column reports the 4-thread rate:

| GGUF | 8 threads, t/s | 4 threads on the A76s, t/s | Ratio |
|---|---:|---:|---:|
| SmolVLM2-2.2B `Q4_K_M` | 5.90 | 20.70 | 3.51x |
| Llama-3.2-3B `Q4_K_M` | 4.51 | 11.09 | 2.46x |
| Ministral-3-3B `Q4_K_M` | 5.02 | 10.47 | 2.08x |
| Phi-4-mini `Q4_K_M` | 4.15 | 8.77 | 2.11x |
| Ministral-3-8B F16 | 1.21 | 1.60 | 1.32x |
| Ministral-3-8B `Q8_0` | 1.97 | 3.01 | 1.53x |
| Ministral-3-8B `Q4_K_M` | 2.39 | 4.72 | 1.98x |
| Qwen3.5-9B `Q4_K_M` | 1.90 | 4.00 | 2.11x |
| Gemma-4-12B F16 | 0.80 | 1.07 | 1.33x |
| Gemma-4-12B `Q8_0` | 1.27 | 1.97 | 1.55x |
| Gemma-4-12B `Q4_K_M` | 1.50 | 3.05 | 2.03x |
| Phi-4 `Q4_K_M` | 1.70 | 2.74 | 1.62x |
| DeepSeek-V2-Lite `Q4_K_M` | 3.20 | 13.52 | 4.22x |
| gpt-oss-20b MXFP4 | 3.14 | 9.11 | 2.90x |
| Qwen3.6-27B `Q4_K_M` | 0.79 | 1.30 | 1.65x |

**Conditions:** `tg64`, three repetitions, the first pass only, each configuration from a memory
reset [HW sweep 2026-11-01]. The 4-thread configuration is `taskset 0xf0 -t 4`.

The F16 GGUFs gain least, which fits a decode near the memory-bandwidth bound at four threads.
Why eight threads lose 1.3-4.2x is not established here. The likely cause is that each decode
step waits on its A55 threads [hypothesis].

The interactive turns are computed from the series' rates, not timed as turns. The README's
turn they replaced was a 2026-07-01 session at ~816 MHz. It took 299 s, 96 s and 118 s to the
first token, and 445 s, 236 s and 170 s in total.

The MoE rows replaced the 2026-09-29 build. The expert route after that build measured 1.09-1.10x
its NPU prefill on gpt-oss-20b at `pp512`: 24.90-25.23 t/s. On DeepSeek-V2-Lite at `pp2048` it
measured 1.09-1.14x: 32.69-32.83 t/s. Both were NPU only, at governor `ondemand`
[HW sweep 2026-10-09]. That route also halves DeepSeek-V2-Lite's one-time expert conversion.

## TUNING

The `taskset 0xf0 -t 4` prefill figures are three rotated passes on three models, with the NPU
half identical in every configuration. They give 1.126x on Qwen3.5-0.8B F16, 1.062x on Qwen3.5-9B
`Q4_K_M` under quantized residency and 1.062x on Gemma-4-12B F16. Plain `taskset 0xf0` at 8 threads
gave 1.103x, 1.059x and 1.046x on the same three.

The decode figures are the README section's table above. An earlier record had pinned decode 34%
slower on Gemma-4-12B F16, with no protocol or raw data kept. On b11539 the same GGUF decodes 1.33x
faster pinned at `-t 4`. Whether that record ran 8 threads on the four A76 cores is not known.

The CPU-only prefill comparison is one process per thread count at `pp512`, governor
`performance`, before the series [HW 2026-11-01]. SmolVLM2-2.2B F16 ran at 30.94 t/s at 8 threads
and 29.63 at 4 on the A76s, 1.04x. Ministral-3-8B `Q4_K_M` ran at 10.29 and 9.34, 1.10x.

The `ROCKET_F16_RESIDENT` figures are three rotated passes per configuration, and each row's
speedup is the ratio of its two means. Before 2026-11-01 the table gave 1.078-1.084x on every row,
which is the range of those ratios over the three models. An
earlier build measured a larger gain at shorter prefills: Llama-3.2-3B +9.5% at `pp512` against
+5.9% at `pp2048`, with output token-identical to the streamed configuration. Phi-4-mini, whose
single `qkv` and `gate_up` weights leave nothing to fuse, measured +21% at `pp512` and +16% at
`pp1024` there. The likely cause of the smaller long-prefill gain is that attention stays on the
CPU at long prefill and dilutes the offloaded matmuls' share [hypothesis].

The residency cache's key includes the content fingerprint because the scheduler reuses pooled
addresses across weights. A cache keyed on the address served one weight's packed tiles for
another, and the model produced gibberish.

The quantized-residency comparison over seven models is three rotated passes per configuration.
The Qwen3.5-9B `Q4_K_M` table is three rotated passes, with the ratios paired within a pass. Its
`auto` at the default `-ub` row is a second session [HW sweep 2026-08-31], whose streamed default
measured 19.10 t/s. The residency table against F16 is three rotated passes per
configuration, each figure the mean of the three. Its `Q4_K_M` columns are one series
[HW sweep 2026-08-31], and its F16 column a second series on the same llama.cpp build
[HW sweep 2026-08-29]. The second series' streamed `Q4_K_M` figures agree with the first's within
0.7%.

An earlier build measured residency on Qwen3.5 at `pp2048` [HW sweep 2026-06-28]:

| Configuration | t/s |
|---|---|
| 0.8B `Q4_K` resident | 89, equal to F16's 89 |
| 0.8B `Q4_K` streamed, `-ub 2048` | 81 |
| 0.8B `Q4_K` streamed, `-ub 512` | 51 |
| 9B `Q4_K` resident | 24.6, ~0.92x F16 |
| 9B `Q4_K` streamed, `-ub 2048`, serial dequantization | 15.8 |

The 0.8B resident row's `-ub` was not recorded. On the 9B, residency was 1.56x, and the per-call
`packB` fell from 12440 to 889 ms. On an earlier build, a blanket `ROCKET_QUANT_RESIDENT=1` at the
default budget measured 19.3 t/s on the 9B `Q4_K`, against 22.1 t/s streamed, at `pp2048` and
`-ub 2048`.

The `ROCKET_DEQUANT_THREADS` gains compare the threaded and serial configurations in one session.

The defaults table's first three rows are steps on Gemma-4-12B F16 at `pp2048`. Each was added to
the one before it, on builds from 2026-06 and 2026-07:

| Step | t/s | Gain |
|---|---:|---:|
| 200 MHz boot clock | 7.98 | |
| NPU ~816 MHz (the 600 MHz request) | 11.40 | 1.43x |
| fp16 K-accumulation on the NPU | 13.38 | +19% |
| CBUF data reuse | ~14.5 | +7% |
| Resident weights | ~15.1 | +6% |

A later build measured Gemma-4-12B F16 at 19.96 t/s at `pp2048` over six rotated passes
[HW sweep 2026-09-02, llama.cpp b10558]. It ran at the default `-b` and `-ub` and governor
`performance`, without CBUF weight streaming.

The `ROCKET_MM_STREAM` speedups are three rotated passes per configuration.

The `ROCKET_MIN_M` crossover table divides the NPU's prefill rate by the faster of the 4- and
8-thread CPU backends at each length.

The attention context ratios are from a build that predates the current softmax and the row
handoff of the scores between the matmuls. The `ROCKET_FLASH_ATTN_UNMASKED` figures, on whisper
small (12 heads, `d`=64, `T` = `n_kv` = 1500), were measured on 2026-09-03, before the fused NEON
softmax and the direct score de-tile. The offload used as many CPU core-seconds as the CPU kernel
(16.1 against its 16.2 on a 20 s window), and with the on-NPU softmax encoding took 96% longer.

The 5.6% share in `ROCKET_FA_THREADS` is from a `ROCKET_FA_TIMING=1` readout: the walks took 5.44 s
of a 97.8 s pinned prefill, 3.46 s gathering and 1.98 s scattering. The thread-count sweep ran
three configurations in three rotated passes. A fit of `G_k = A/k + B` to the table leaves a
fixed 0.55 s, which bounds what workers past four can save.

The quantization table's four configurations ran on one build, selected by environment alone.
Each process followed a memory reset and a discarded warm-up process. Three passes ran with the
order rotated, and each ratio pairs two configurations within one pass [HW sweep 2026-11-06].
The whole-tile int8 figure is the same build under `ROCKET_MM_STREAM_INT8=0`: 69.4-70.5 t/s on
Llama-3.2-3B-Instruct at `pp512`. It reproduces the 0.60x of the earlier record, Gemma-4-12B F16
at `pp2048` at ~816 MHz. The int8 route's perplexity is the same on both plans at every chunk,
11.0729.

The table's 8-9 B rows ran the same way on a later build of the same sources [HW sweep
2026-11-07]. Ministral-3-8B-Instruct ran five configurations: the four, plus the int8 route at the
default 4096 MB budget. Qwen3.5-9B ran four, without the whole-tile plan. All 27 processes
completed. The whole-tile int8 plan ran Ministral-3-8B-Instruct at 31.7-31.8 t/s at `pp512`,
0.610-0.615x resident fp16. Resident fp16 places at these sizes because a prefill-only process
drops each F16 source page once its weight is resident.

The perplexity rows in [MODELS.md](MODELS.md#int8-w8a8) are 8 chunks of 512 of wikitext-2 test.
Each arm's error estimate is ~5%, so the sign of each difference rests on the per-chunk running
estimates. int8 is above fp16 at every chunk on both 8-9 B models.

## MODELS

The Ministral-3-8B and Gemma-4-12B tables replaced 2026-07-01 and 2026-07-02 sessions at ~816 MHz.
Their quantized rows ran at `-b 2048 -ub 2048`, against a CPU backend without the weight repack. On
those sessions the F16 speedup narrowed with `M`, from 3.8x to 3.1x on the 8B and from 3.6x to
3.2x on the 12B. In this series the 12B's holds at 5.1-5.2x and the 8B's narrows from 6.1x to
5.8x.

The 3-4 B models' figures replaced `pp512` figures from llama.cpp b9568 [HW sweep 2026-07-02/03]:
Llama-3.2-3B at 61.8 t/s, 3.4x, Ministral-3-3B at 57.6 t/s and Phi-4-mini at 55.4 t/s. Those
sessions measured the speedup narrowing to 2.4-2.6x by `pp2048`. SmolVLM2's replaced 99 t/s at
`pp512`, 3.5x, and a 14.5 t/s decode. This series measured the four at `pp2048` only.

The Qwen figures replaced the 27B's 9.6 t/s, 3.2x [HW sweep 2026-09-29]. They also replaced the
9B F16's 3.65x, the best run of a 2026-06 scaling sweep at `-ub 2048`. That sweep measured the
0.8B F16 at 1.44x the CPU backend at `pp512` [HW sweep 2026-06-27/28]. It is the 0.8B's only
speedup figure, from a build that measured the 3-4 B models at ~3.4x.

The `ROCKET_FLASH_ATTN` perplexity figures, 13.8994 with the offload and 13.9024 without, are
`llama-perplexity` runs on Ministral-3-3B F16. Two offloaded processes agreed to every digit
[HW sweep 2026-09-07].

## MOE

The gpt-oss-20b default column is the mean of six runs, which spanned 20.5-22.8 at `pp512` and
26.9-28.7 at `pp2048` with identical placement. The `ROCKET_MOE=0` configuration was stable to
±0.05 across the same rebuilds, which places the spread in the default configuration rather than
in the board.

The `ROCKET_MOE=1` gains of 6-13% at `pp512` and 18-21% at `pp2048` span two measurements, the
table's 13% and 21% among them. A third, over three rotated passes, measured 19.7% at `pp2048`
[HW sweep 2026-08-29]. The `pp512` half of the range is inside that configuration's own
run-to-run spread.

At llama.cpp's default `-ub 512`, the default placement on gpt-oss-20b measured 22.2 t/s at
`pp512` (range 21.7-22.7) and 22.5 at `pp2048` (range 22.4-22.8). With the experts on the CPU it
measured 14.13 and 13.53 [HW sweep 2026-08-27].

The current-route figures come from a 2026-10-09 comparison against the build before it, NPU
configuration only. On that earlier build, gpt-oss-20b measured 22.78-22.99 t/s at `pp512` at an
unchanged ingest. DeepSeek-V2-Lite measured 28.82-29.99 t/s at `pp2048`, and its ingest fell from
38.3-38.8 s to 18.2-18.4 s inside the same runs. The comparison measured no CPU ratio.

The three `llama-bench` ingest repetitions on gpt-oss-20b at `pp512` measured 22.60, 22.51 and
21.27 s in one run [HW sweep 2026-10-09]. That run exercised 1629 experts, where a one-repetition
run exercises 1575.

The 99% residency of gpt-oss-20b on a 31 GiB board holds for the current resident layout. An
earlier layout used 35% of the resident bytes on tile padding and did not reach it.

The gpt-oss-20b budget table is one `-r 3` process per budget, so each gain is a single
measurement. A single process on this board can sit ~10% off the level its configuration repeats
at. The `pp512` column also sits inside a ±1.4-2.1 t/s within-process spread. The table therefore
resolves only the direction: on this model, more stacks are faster. The stack counts are exact.

The Qwen3-30B-A3B plateau row is pooled over seven processes, `auto` over three (rep sd ±0.6), and
the 12000 MB and 28000 MB rows are one process each (28000 MB rep sd ±1.02). The 3.2% step from the
plateau to `auto` is the only difference deep enough to quote as a size. It is inside the board's
8-13% per-process spread, and resolving it at 3 sigma needs 27 rotated passes. At 28000 MB the
0.999x shows one process at parity. The sizing formula's headroom term is its one free parameter,
chosen to match the plateau. The formula is therefore consistent with the plateau, not an
independent prediction of it.

With the page cache dropped first, three processes requesting 24679, 26000 and 28000 MB admitted
9539, 10032 and 10810 experts. A repeat of the `auto` configuration came within 13 experts of a
separate probe. `MemAvailable` already counts most reclaimable page cache, so why dropping it
steadies the budget is not established.

DeepSeek-V2-Lite's ratios in the `M_e` table are means of adjacent pairs: four at `M_e` = 72 and
144, eight at 96, and two at 192. Its offloaded configuration varies ~15% run to run, so a single
pair cannot decide a ratio this close to 1.00. gpt-oss-20b's ratios are single pairs, whose margins
sit an order of magnitude outside that spread. The work minimum's form is derived, and its value
is fitted on two architectures. Qwen3-30B-A3B confirms its direction but not its position: halving
the work per dispatch at matched residency lowered `ROCKET_MOE=1` from 0.834x to 0.661x.

Qwen3-30B-A3B's `ROCKET_MOE=1` figure in [MOE.md](MOE.md), 0.800x at 62% resident, is three
rotated passes with a memory reset before every configuration [HW sweep 2026-08-30]. The 0.834x
above, at 62.4% resident, is three process pairs with the configurations alternated and the page
cache dropped first [HW sweep 2026-08-27]. It stays here because its series also ran `-ub` 1024,
which gives the matched-residency comparison.

The DeepSeek-V2-Lite perplexity differences at `-c 2048` are paired per-chunk differences over 8
chunks. The absolute error bar, ±2.5%, would resolve none of them, and pairing on the same chunks
cancels the variance both configurations share.

Each ingest break-even count is the 2026-08 count at that ratio, scaled by the model's ingest
ratio. That ratio compares the current build with the earlier pack in one session: 0.47x on
DeepSeek-V2-Lite and 0.51x on gpt-oss-20b.

On the build that first held experts resident, the fp16 dequant route dequantized each gpt-oss-20b
expert in 75 ms per micro-batch. At ~1580 expert dequantizations per prefill, that is ~119 s before
any arithmetic, which gives the route's 4.59 t/s at `pp512` [HW sweep 2026-07-14].

The worker-count ratios are paired within a pass over three rotated passes, and each is resolved
above 1.00. An A/B with the two configurations adjacent measured the 5-to-8 step higher, 1.027x.
The half hour the rotated ladder puts between a pair is the candidate cause [hypothesis].

On the native-quant route the little-cluster instruction share steps from 0.120 to 0.115 at six
workers and does not move again. Migrations and the A76 IPC keep degrading with every worker
added.

The worker-count runs admitted 63 stacks of gpt-oss-20b and the RAM-budget runs 62,
both on 2026-09-02. The `auto` budget is read from `MemAvailable` once per process, and the 63rd
stack brings the admitted total to 24529 MB. The 62-stack runs had budgets of 24517-24526 MB.

On the F16 residency route, each placement count was byte-identical across independent processes.
At `pp2048` eight workers measured 0.987x the prefill speed of five over three rotated passes
[HW sweep 2026-09-01]. They also placed fewer weights, 281 against 286, because the
`MemAvailable` minimum binds there rather than the window. The prefill-time difference between five and six workers is
below the measurement's resolution.

The dense fp16 route's worker sweep, at `pp1024` and `-ub 1024`, measured 7.15, 8.06, 8.24 and 8.27
t/s at three to six workers. That flat step from five to six is why the default stays at five.

## RK3576

The `pp2048` performance table interleaves three processes per configuration in one boot, and the
per-pair ratios behind the 1.31x are 1.30, 1.34 and 1.30. At `pp2048` the route's spread is
±0.02 t/s within a process and 0.3% across its three processes (the Q8_0 CPU configuration's is
3.5%). At `pp512` llama-bench reports ±1.8 t/s (15%) across one process's reps, against ±0.01 t/s
for both CPU configurations. A `pp512 -r 1` run reports 332 of 332 calls as calibration forwards.
In a longer run the first timed rep calibrates and the later reps run converged. The `pp512`
11.78 t/s, 2.01x and 1.02x against the two CPU configurations, averages the two regimes [HW sweep
2026-08-11].

## API

The `ROCKET_FA_CHECKSUM` hash agreed across `ROCKET_FA_THREADS` = 1, 2, 3, 4 and 5. Two runs at one
thread count on Gemma-4-12B F16 also disagreed, twice in seven runs on one build [HW sweep
2026-09-07]. They had the same graph, the same 288 ops and the same 2818572288 bytes.

A replay of
the dumped op gave the odd value in 7-91% of trials at 5 workers [HW sweep 2026-09-26/27]. At 1
worker it never did, 200 of 200. On the op traced, the disagreement is one fp16 ulp at one lane-0
element. The hash cannot separate the thread split from the core that ran each job. An equality across
thread counts is therefore no evidence about the split.

**Not characterized:** whether the attention output is bit-identical at every `ROCKET_FA_THREADS`
value.

The [Repacked weights](API.md#repacked-weights) figures were reported by the contributor of the
registration, from Ollama's `llama-server` registering the sources from its model loader. They
were not reproduced on the RK1. With gemma-4 E2B (Ollama's `gemma4:e2b`) and a 2,000-token prompt:

| Configuration | Prefill, t/s | Decode, t/s |
|---|---:|---:|
| `--no-repack` | 71-75 | 6.2-8.2 |
| Repack on, original bytes registered, `ROCKET_DEQUANT_CACHE_MB=2048` | 80.6 | 11.9-12.7 |

`ROCKET_DEQUANT_CACHE_MB` removes the per-micro-batch dequantization, and only the second
configuration set it. The prefill ratio therefore does not isolate the registration. The decode
gain is ggml-cpu's repacked kernels. Resident memory rose by about 3 GB with the repack on. The
output matched the `--no-repack` configuration at temperature 0.

**Conditions:** RK3588 with 32 GB of RAM, the vendor `rknpu` 0.9.7 driver through `rknpu-submit`,
Ollama's `llama-server` with 4 threads on the A76 cores, reported 2026-11-08. The NPU clock, the
`-b` and `-ub` settings and the run counts were not reported.

**Not characterized:** the registration on the RK1 with the `rocket` driver, and how much
residency `ROCKET_QUANT_RESIDENT=auto` loses to the repacked copy's RAM.
