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

Every RK3588 figure dated before 2026-11-21 ran behind a U-Boot whose SPL left SPLL unprogrammed,
which halves the NPU's DDR read rate. The series of 2026-11-21 and 2026-11-22 ran with SPLL
programmed, on kernel 7.3.3 with `rocket` 1.3.0 and the backend built from the repository's
2026-11-11 state. Its LLM figures ran on llama.cpp b11539 and its speech figures on whisper.cpp
v1.9.5. The figures that check the defaults of 2026-11-22 ran a build of the backend with those
defaults, and name it as the backend of 2026-11-22.

Two RK1 boards ran the series. The first runs the 900 MHz request at 927.5 MHz and ran most of the
earlier figures. The second runs it at 1028 MHz. Every process of the series completed, and the
kernel log showed no job timeout or IOMMU warning.

## README

The headline table, the interactive turn, and the language-model figures dated 2026-11-01 in
[MODELS.md](MODELS.md) and [MOE.md](MOE.md) are one measurement series on the RK1. It ran
llama.cpp b11539 with the backend built from the repository's 2026-10-29 state, the NPU at
928 MHz, SPLL unprogrammed and the CPU governor at `performance`.

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

The boot-loader figures in the README's Requirements are one RK1 at 1028 MHz under three boot
areas in sequence [HW sweep 2026-11-21]. A mainline U-Boot v2026.07 build that leaves SPLL
unprogrammed ran first and last, and Armbian's `2026.07_armbian` build, which programs it, ran
between. The two unprogrammed configurations agree within 2% on every row. Each configuration ran
two passes in rotated order, each process after a page-cache drop and a discarded warm-up process:

| Workload | SPLL programmed | SPLL unprogrammed | Ratio |
|---|---:|---:|---:|
| int8 weight read, 4×2048×4096, one task, one core | 0.796-0.802 ms | 1.578-1.586 ms | 1.98x |
| Llama-3.2-3B F16, `pp512`, fp16 route | 105.7-108.4 t/s | 97.8-101.4 t/s | 1.07-1.11x |
| Qwen3.5-9B F16, `pp512`, fp16 route | 40.1-40.5 t/s | 38.3-39.5 t/s | 1.01-1.05x |
| Qwen3.5-9B F16, `pp512`, rotated resident int8 | 45.0-45.9 t/s | 43.6-44.3 t/s | 1.02-1.05x |
| Whisper base encoder, 30 s window | 887-907 ms | 896-910 ms | 0.99-1.03x |
| Whisper small encoder, 30 s window | 2585-2593 ms | 2600-2625 ms | 1.00-1.02x |
| Llama-3.2-3B F16, `pp512`, CPU only | 20.7 t/s | 20.5-20.7 t/s | 1.00-1.01x |

**Conditions:** kernel 7.3.3, `rocket` 1.3.0, CPU governor `performance`, DDR at 2112 MHz.
`llama-bench -p 512 -t 8` unpinned on llama.cpp b11539, and `whisper-bench` on 4 threads on
whisper.cpp v1.9.5.

The decode comparison divides weight bytes by time. At M 4, three processes drive the NPU's three
cores to 30.0-30.2 GB/s at fp16 and 27.5-29.3 GB/s at int8. The NPU rates were measured on the RK1
at 1028 MHz with SPLL programmed [HW sweep 2026-11-21]. The CPU decoder's `tg64` rates of
2026-11-01 at `taskset 0xf0 -t 4` on llama.cpp b11539 imply ~22-27 GB/s on the four A76 cores
[host-computed].

The clock figure is fp16 `512×3840×4096` on one core with SPLL programmed: 37.55-37.58 ms at
471.9 MHz, 22.89-23.74 ms at 905.0 MHz and 18.97-21.42 ms at 1028 MHz [HW sweep 2026-11-21]. The
device time fits 14830/f + 6.2 ms, with f in MHz, so ~70% of it scales with the clock at 1028 MHz.
The 1.044x with SPLL unprogrammed is fp16 `512×3840×4096` on two cores, 25.07 ms at ~816 MHz
against 24.02 ms at 928 MHz [HW sweep 2026-10-19].

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

The `ROCKET_F16_RESIDENT` table pairs the default and resident fp16 configurations of the
quantization table's series, below, each ratio within one pass. The teardown report put every
weight of the 3-9 B models resident, and 307 of Gemma-4-12B-it's 325, where the IOVA window
filled. Gemma-4-12B-it's `pp512` ratios ran 0.981-1.137x, in the process modes that series
describes.

The table replaced a measurement on llama.cpp b10558 at ~816 MHz with SPLL unprogrammed [HW sweep
2026-08-29]. It ran `pp2048` at the default `-b` and `-ub`, with every weight resident, in three
rotated passes per configuration. Each speedup is the ratio of two means:

| Model, F16 | Streamed weights, t/s | `ROCKET_F16_RESIDENT=auto`, t/s | Speedup |
|---|---:|---:|---:|
| Llama-3.2-3B | 56.01 | 60.54 | 1.081x |
| Ministral-3-3B | 48.28 | 52.04 | 1.078x |
| Phi-4-mini | 50.19 | 54.40 | 1.084x |

An earlier build measured a larger gain at shorter prefills: Llama-3.2-3B +9.5% at `pp512`
against +5.9% at `pp2048`, with output token-identical to the streamed configuration.
Phi-4-mini, whose single `qkv` and `gate_up` weights leave nothing to fuse, measured +21% at
`pp512` and +16% at `pp1024` there. The likely cause of the smaller long-prefill gain is that
attention stays on the CPU at long prefill and dilutes the offloaded matmuls' share [hypothesis].

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
8-thread CPU backends at each length. It ran on the first RK1 [HW sweep 2026-11-21]. `llama-bench`
ran `-p 8,12,16,20,24,32,48,64,128 -n 0 -r 3` in two passes with the configurations rotated, and
every per-pass spread was under 1%. The 4-thread CPU configuration loads the backend and declines
every op (`ROCKET_MIN_M=1000000 ROCKET_FLASH_ATTN=0`).

The table replaced a 2026-10-09 measurement on llama.cpp a7fb71f and the backend of that date, at
~816 MHz with SPLL unprogrammed. It put the crossover at 21-25 rows:

| F16 model | `pp16` | `pp32` | `pp64` | `pp128` |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.75x | 1.20x | 1.71x | 1.96x |
| Llama-3.2-3B | 0.78x | 1.40x | 2.51x | 3.68x |
| Ministral-3-8B | 0.77x | 1.46x | 2.75x | 3.90x |

Most of the crossover's move comes from the newer backend. The 2026-10-09 build ran on
Llama-3.2-3B as a fourth configuration in the same passes. Its NPU rate rose 1.04-1.08x over its
own 2026-10-09 reading, a rise that includes both SPLL and the clock's move from ~816 to 927.5 MHz.
The current build measured 1.19x that build at `pp16`, and the CPU configurations were unchanged.
Interpolated on the earlier table's grid of 16 and 32 rows, the crossover on Llama-3.2-3B was 21.7
rows. The 2026-10-09 build moves it to about 19.7 rows, and the current one to 16.1.

The default of 24 was checked on the backend of 2026-11-22 [HW sweep 2026-11-22]. Llama-3.2-3B F16
prefilled a 24-row prompt at 23.81-23.86 t/s at the default. Under `ROCKET_MIN_M=32`, which keeps
it on the CPU, it ran at 16.52-16.55 t/s. The default ran 1.442x as fast in each of two rotated
passes. Both configurations ran `taskset -c 4-7 -t 4` and loaded the backend. The table's figure
for the same model and length is 1.444-1.447x the faster CPU configuration.

The Whisper beam figures ran `whisper-cli` at its defaults (beam 5, best-of 5, flash attention on)
on a 176 s clip, 4 threads pinned to the A76 cores [HW sweep 2026-11-22]. Three passes ran with the
thresholds rotated. The beam steps are whisper.cpp's `batchd` time, the M=5 calls. Thresholds of
8 and 32 measured the same, 0.988-1.011x, and the transcripts were md5-identical across the
thresholds of 4, 8 and 32. At a threshold of 4, `ROCKET_MM_PROFILE` put whisper-small's weight
pack at 13.0 s, against 1.0 s at a threshold of 8. With SPLL unprogrammed, the beam steps at 4
measured 2.3x their CPU time and the transcription 1.40x as long.

The attention context table ran two passes with the configurations rotated, on the first RK1
[HW sweep 2026-11-21]. The 8K and 16K tests ran one repetition with `--no-warmup`, after the
discarded warm-up process. A 512-token prompt with every op offloaded (`ROCKET_FLASH_ATTN_MIN_KV`
of 512) measured 0.992-0.998x on Gemma-4-12B and 0.976-0.990x on Llama-3.2-3B. At a 1K prompt
that threshold measured 1.082-1.083x and 1.074-1.087x, against the default's 1.069-1.086x and
1.073-1.083x. The pinned figure pairs `taskset 0xf0 -t 4` configurations with the offload on and
off at `pp2048`, in two passes.

The table replaced Gemma-4-12B F16 figures at ~816 MHz with SPLL unprogrammed, on the backend and
llama.cpp of that month [HW sweep 2026-06-26/28]. They were parity at 1K and below, 1.07x at 4K,
1.45-1.50x at 8K and 1.25x at 16K. At `pp2048` the same pinned and unpinned pairs measured 1.079x
and 1.045x [HW sweep 2026-09-03], against 1.240-1.252x and 1.199-1.205x now. Between those dates
the offload-off configurations gained 1.12-1.15x and the offload-on configurations 1.29-1.33x. The
two dates differ in the boot loader, the NPU clock, the backend's softmax, score de-tile and row
handoff, and the llama.cpp build. The measurements do not separate those causes.

The encoder-attention figures ran on the first RK1 with the backend of 2026-11-22 [HW sweep
2026-11-22]. Three configurations ran per model: the default, `ROCKET_FLASH_ATTN_UNMASKED=0`, and
the CPU backend without `libggml-rocket.so`. Every process ran `taskset -c 4-7 -t 4` after a memory
reset, and each backend configuration had a discarded warm-up process per model. The models are
whisper.cpp's multilingual `ggml-tiny`, `ggml-base` and `ggml-small`, F16, whose encoders attend
over `T` = `n_kv` = 1500 with `d`=64.

`whisper-bench` ran three passes with the configurations rotated. Its encode times, in ms:

| Model | Default | `ROCKET_FLASH_ATTN_UNMASKED=0` | CPU backend |
|---|---:|---:|---:|
| `tiny` | 438.2-442.8 | 475.5-478.3 | 666.0-668.2 |
| `base` | 749.7-761.6 | 941.7-943.7 | 1524.2-1533.2 |
| `small` | 1919.0-1923.9 | 2674.7-2686.5 | 5562.0-5569.1 |

The process core-seconds cover `whisper-bench`'s whole run, its decode benchmarks included. They
were 32.4-32.5 against 35.6-35.8 on `tiny`, 55.8-56.3 against 64.3-64.6 on `base`, and
132.2-133.9 against 161.2-163.2 on `small`. One untimed `ROCKET_FA_TIMING=1` process per
configuration counted 84, 126 and 252 offloaded attention ops at the default and none at `=0`. A
series on the backend of 2026-11-11 with `ROCKET_FLASH_ATTN_UNMASKED=1` measured `small` at
1913.7-1932.3 ms against 2673.4-2682.8 ms without it.

`whisper-cli` ran one process per configuration and model on a 176 s English clip, with `-nt` and
the default beam search.
Its wall-clock times were 10.03 s against 10.08 s on `tiny`, 17.06 s against 18.49 s on `base` and
39.53 s against 44.84 s on `small`. Its core-seconds were 30.5 against 33.3, 52.7 against 63.6 and
123.6 against 157.5. The transcripts' md5s matched between the two configurations, and on base and
small they matched the transcripts of the threshold comparison above. Each default process
offloaded `6 x n_audio_layer` ops for its six encodes and `5 x n_text_layer` more. Five decoder
prompt passes that offload their cross-attention give that remainder [hypothesis].

With SPLL unprogrammed, on 2026-09-03, the offload took 36% longer than the CPU kernel. Through
`whisper-cli` on a 20 s window, its encode took 4.31 s against 3.17 s, with an md5-identical
transcript. That build predates the fused NEON softmax, the direct score de-tile and the row
handoff. The offload used as many CPU core-seconds as the CPU kernel (16.1 against its 16.2), and
with the on-NPU softmax encoding took 96% longer. The two dates differ in both the boot loader and
the backend, and the measurements do not separate them.

The voice-command table ran `parakeet-cli` over four sets of ten utterances [HW sweep 2026-11-22].
Three repetitions rotated five configurations: the CPU only, and thresholds of 4, 8, 16 and 32. Each
per-utterance cost is a 10-file process less a 1-file process, divided by nine, in wall-clock time
and in user plus system core-seconds. Thresholds of 4, 8 and 16 admit the same GEMMs and measured
within 1.5% of each other. The transcripts were byte-identical across all five configurations.

The table replaced figures at ~816 MHz with SPLL unprogrammed and CPU governor `ondemand`
[HW sweep 2026-09-30]. A threshold of 4 then tied the CPU at 1.3-1.8 s and ran 1.18-1.73x its
speed at 2-6 s, freeing 34-61% of the CPU core-seconds.

The `ROCKET_MIN_M_QUANT` table ran one micro-batch of M rows on the first RK1 [HW sweep 2026-11-21].
Two passes ran eleven configurations, rotated by one each pass. The CPU configuration ran without
the backend. The resident configuration's teardown lines showed 0% resident at 128 and 192 rows, and
all 197 weights (12896 MB) resident from 256 rows, at 44.7-46.0 t/s. Qwen3.6-27B ran the streamed
and CPU configurations only, on 2026-11-22. The crossover interpolates linearly between 128 and 192
rows.

The default of 256 was checked on the backend of 2026-11-22 [HW sweep 2026-11-22]. Qwen3.5-9B
`Q4_K_M` ran a 256-row prompt at the default `-b` and `-ub` with `--repack 0`, at 15.83-15.93 t/s.
The CPU backend with its repack on ran 10.76-10.77 t/s, for 1.472x and 1.479x in two rotated
passes. A `ROCKET_MM_PROFILE` process confirmed the offload. The table's streamed 256-row
configuration measured 15.93-16.07 t/s.

The table replaced a crossover of ~360 rows on 9B and 27B `Q4_K` [HW sweep 2026-06-28]. That
measurement ran at ~816 MHz with a serial dequantization, against a CPU backend without the weight
repack. A 2026-11-02 measurement at 928 MHz with SPLL unprogrammed put the 9B's crossover at
~145 rows.

The 5.6% share in `ROCKET_FA_THREADS` is from a `ROCKET_FA_TIMING=1` readout: the walks took 5.44 s
of a 97.8 s pinned prefill, 3.46 s gathering and 1.98 s scattering. The thread-count sweep ran
three configurations in three rotated passes. A fit of `G_k = A/k + B` to the table leaves a
fixed 0.55 s, which bounds what workers past four can save.

The quantization table's configurations ran on one build, selected by environment alone, on the
second RK1 at 1028 MHz [HW sweep 2026-11-21]. Each process followed a memory reset and a discarded
warm-up process. Three passes ran with the order rotated, and each ratio pairs two configurations
within one pass. The 8-9 B models added the int8 route at a 4096 MB budget as a fifth
configuration in the same passes. The fifth configuration gives the 4096 MB budget's figures on
Qwen3.5-9B and at `pp2048`. One untimed `ROCKET_DEBUG=1` process per model read that budget's
placement on 2026-11-22.

Qwen3.5-9B's configurations spread more between processes than the other models' (resident fp16
at 44.71-47.24 t/s at `pp512`), so its per-pass ratios against resident fp16 straddle 1.00. Its
rotated route against the default, 1.090-1.145x at `pp512`, agrees with a run on the same board
the day before, 1.116x and 1.125x.

Gemma-4-12B-it's `pp512` rates ran in modes between processes. The unrotated int8 route measured
25.36 t/s in one process and 39.89 in another, while each process's repetitions agreed within
0.3 t/s. Its `pp512` ratios against resident fp16 were 0.804-1.477x unrotated and 1.142-1.319x
rotated, and the table omits that test.

The whole-tile int8 and resident int4 figures are one series on Gemma-4-12B-it at `pp512` [HW sweep
2026-11-22]. Resident fp16 and the streamed int8 plan ran in the same passes as references. The
whole-tile plan ran at 19.02-20.03 t/s and the streamed plan at 38.97-40.04, whose repetitions held
within 0.3 t/s. Resident fp16 ran at 26.70-27.25 t/s, with its repetitions spread ±7-9 t/s within
each process, so the ratios against it are approximate. One untimed `ROCKET_DEBUG=1` process
confirmed that the int4 route placed all 325 weights resident.

The table replaced these figures, measured on the first RK1 at 928 MHz with SPLL unprogrammed:

| Date | What was measured | Earlier value |
|---|---|---|
| 2026-11-06 | Resident int8 against resident fp16, Llama-3.2-3B-Instruct and Ministral-3-3B-Instruct | 0.948-0.967x at `pp512` and 0.964-0.978x at `pp2048` on Llama-3.2-3B-Instruct. 0.971-0.990x and 0.968-0.984x on Ministral-3-3B-Instruct |
| 2026-11-07 | Resident int8 against resident fp16, Ministral-3-8B-Instruct and Qwen3.5-9B | 0.893-0.899x and 0.908-0.921x on Ministral-3-8B-Instruct, 0.914-0.927x and 0.921-0.949x on Qwen3.5-9B, with the K passes summed in an int64 accumulator |
| 2026-11-07 | The int8 route with each K pass added straight into the output | 1.07-1.12x the int64 build on Ministral-3-8B-Instruct, 51.5-52.0 t/s at `pp512`, and 1.01-1.03x on Llama-3.2-3B-Instruct |
| 2026-11-06/07 | The whole-tile int8 plan against resident fp16 | 0.61-0.66x: 69.4-70.5 t/s on Llama-3.2-3B-Instruct and 31.7-31.8 t/s on Ministral-3-8B-Instruct at `pp512`. An earlier record measured 0.60x on Gemma-4-12B F16 at `pp2048` at ~816 MHz |
| 2026-11-07 | The 4096 MB int8 budget against `0` | It held 58-62% of the int8 weights, and cost 3-5% on Ministral-3-8B-Instruct and 1-3% on Qwen3.5-9B |
| 2026-11-11 | The rotated route against the unrotated one on Qwen3.5-9B at `pp512` | 0.951-0.972x, and 0.855-0.867x with the conversions on one thread |
| 2026-11-11 | Gemma-4-12B-it with every int8 weight resident | Unrotated, 1.35-1.37x the default fp16 route at `pp512` and 1.30-1.32x at `pp2048`. Rotated, 31.8-31.9 t/s at `pp512` against 25.5-25.9 for the fp16 route |

The int8 route's perplexity is the same on the whole-tile and streamed plans at every chunk,
11.0729 on Llama-3.2-3B-Instruct [HW sweep 2026-11-06]. Resident fp16 places every weight of the
3-9 B models because a prefill-only process drops each F16 source page once its weight is
resident.

The `ROCKET_INT8_CACHE_MB` default's figures ran on Ministral-3-8B-Instruct F16 at `pp512` on the
second RK1, with the backend of 2026-11-22 [HW sweep 2026-11-22]. Three configurations of
`ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1` ran three rotated passes: the default,
`ROCKET_INT8_CACHE_MB=0` and `=4096`. Each ratio pairs two of them within one pass. The default ran
at 58.87-59.64 t/s, `0` at 59.22-59.62 and 4096 at 53.39-53.70.

Every default process logged a budget of 21552-21580 MB, from a `MemAvailable` of ~31 GB less a
9535 MB reserve, and none logged the reserve stopping admission. One untimed `ROCKET_DEBUG=1`
process per configuration counted 235 of 235 int8 weights resident at the default and at `0`, and
141 at 4096. On the 2026-11-11 build, `0` ran at 60.04-60.19 t/s and 4096 at 0.896-0.903x it, in
the quantization series.

The `ROCKET_CONVERT_THREADS` figures are four configurations on one build, on Qwen3.5-9B F16 with
every int8 weight resident. Rotated and unrotated each ran with the host conversions spread over
the A76 cores and on one thread (`ROCKET_CONVERT_THREADS=1 ROCKET_NO_ACT_REUSE=1`). The governor
was `performance`. Three passes ran with the order rotated, and a memory reset and a discarded
warm-up process came before every configuration [HW sweep 2026-11-11]. Each ratio pairs two
configurations within one pass.

Rotated, the spread conversions ran at 41.38-42.03 t/s against 35.92-36.02 on one thread.
Unrotated, they ran at 42.94-43.65 t/s against 41.51-42.03. The one-thread unrotated rate matches
the serial build's of the day before, 41.7-42.0 t/s. The one-thread rotated rate is 2-3% under
that build's 36.6-37.0, because its rotation took 4.47 s against 4.11 over two `pp512` runs.
Against the previous build the rotated gain is 1.12-1.15x.

Perplexity over 4 chunks of 512 of wikitext-2 test was the same at every chunk within each pair
[HW sweep 2026-11-10]. The `ROCKET_MM_PROFILE` processes put the rotated activation conversions
at 5.31 s of caller time on one thread and 1.23 s spread, over two `pp512` runs. The unrotated
ones took 0.95 s and 0.32 s. The dequantization fell from 0.80-0.93 s to 0.53 s. The NPU wait and
the packing did not change.

The fp16 conversion figures are Ministral-3-8B-Instruct F16 at `pp512` on one build [HW sweep
2026-11-11]. They cover resident fp16 weights (`ROCKET_F16_RESIDENT=auto`) and the default route,
each against `ROCKET_CONVERT_THREADS=1`, under the protocol above. Resident, the spread
conversions ran at 52.70-52.88 t/s against 51.48-52.03. On the default route they ran at
44.28-44.44 against 43.79-43.84. In the `ROCKET_MM_PROFILE` processes the activation scaling fell from 733 to
331 ms over two runs and the output unscaling from 529 to 413 ms. The library's own activation
pack rose from 746 to 906 ms, taking back part of the saving.

On 2026-11-11, one series compared the Gemma-4-12B rotated rate on two builds [HW sweep
2026-11-11]. The later one restructured the 60-point rotation. The two builds'
rotations give the same bits. Over three passes in alternating order the newer one ran at
31.77-31.93 t/s against 29.09-29.24, 1.090-1.093x. Its `ROCKET_MM_PROFILE` process put the rotated
conversions at 3.50 s over two `pp512` runs, against 6.35 s for the earlier build. The fp16 route's
25.48-25.93 t/s comes from a series earlier that day under the same protocol.

The perplexity rows in [MODELS.md](MODELS.md#int8-w8a8) are 8 chunks of 512 of wikitext-2 test.
Each arm's error estimate is ~5%, so the sign of each difference rests on the per-chunk running
estimates. int8 is above fp16 at every chunk on both 8-9 B models.

## MODELS

The Whisper encoder speedups at the defaults, which the README also gives, divide the CPU
backend's encode time by the default configuration's within each pass. They come from the
encoder-attention series under [TUNING](#tuning).

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

The int4 table is the quantization series' Gemma-4-12B-it figures at `pp512`, described under
[TUNING](#tuning) [HW sweep 2026-11-22]. It replaced figures on an earlier build at ~816 MHz with
SPLL unprogrammed. The fp16 route ran at 13.18 t/s with ~10.5 GB of weights. Resident int4 ran at
6.94 t/s with 2634 MB, 0.53x fp16, and the per-call int4 route at 3.56 t/s.

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
