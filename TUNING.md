# ggml-rocket: tuning guide

This guide gives the configuration for each workload and what each setting is worth. Start with
the [README](README.md), which covers the build and a first run. [API.md](API.md) lists every
environment variable with its default.

Unless a Conditions line says otherwise, a figure here was measured on the RK3588 with the NPU at
~816 MHz (the 600 MHz request), warm. Its boot loader left [SPLL](README.md#terms) unprogrammed,
and its baseline is the 8-thread CPU backend on the same GGUF.

A figure dated 2026-11-21 or 2026-11-22 was measured with SPLL programmed, CPU governor
`performance`, kernel 7.3.3 and `rocket` 1.3.0. It ran the backend of 2026-11-11, unless its
Conditions line names the backend of 2026-11-22, which has the current defaults. It ran on an RK1
at 927.5 MHz (the 900 MHz request), unless its Conditions line names a second RK1 at 1028 MHz.

## Recommended configurations

The defaults enable fp16 [K-accumulation](README.md#terms), [CBUF](README.md#terms) operand reuse,
CBUF weight streaming, asymmetric tiling, threaded dequantization and attention offload. None
needs an environment variable. Add the settings the table gives for the workload:

| Workload | Precision | Add | Effect |
|---|---|---|---|
| Interactive chat, short prompts | Any | `taskset 0xf0 -t 4`, and a `Q4_K_M` GGUF for decode t/s | The turn is decode-bound, so the quantization and the thread placement set its speed. A quantized prompt under 256 rows stays on the CPU, and an F16 one under 24 |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form fits RAM, wholly or partly | `ROCKET_QUANT_RESIDENT=auto`, at the default `-ub` | 1.35-1.75x the default configuration's prefill on seven dense models of 1.8-11.9 B. Needs free RAM of about the fp16 model size |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form does not fit | `-b 2048 -ub 2048` | 0.91-1.53x the default `-ub 512`, by model, so measure it |
| Agentic or RAG, repeated prefill | F16, with RAM for ~2x the model | `ROCKET_F16_RESIDENT=auto`, and `ROCKET_N_THREADS=6` where the IOVA window fills | 1.13-1.21x the default configuration's prefill on four models of 3-9 B ([F16 weights](#f16-weights)) |
| Agentic or RAG, repeated prefill | F16, with RAM for ~1.5x the model | `ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1` | 1.15-1.21x the default configuration's prefill on four models of 3-9 B ([Quantization and prefill speed](#quantization-and-prefill-speed)), at a wikitext perplexity 0.7-2.8% above fp16. Where free RAM holds the int8 budget at its 4096 MB minimum, the weights past it run on the fp16 route, 4-8% slower at 8-9 B |
| Medium or long prompts | MoE whose experts offload (gpt-oss, DeepSeek-V2-Lite) | `-b 2048 -ub 2048` and `ROCKET_N_THREADS=6` | 1.25-1.32x the default `-ub` at `pp2048`. [MOE.md](MOE.md) has the rest |
| Model too large at F16 | Any | A `Q4_K_M` GGUF, or `ROCKET_INT4=1` with an F16 GGUF | Reduces the footprint at a slower prefill than fp16 |
| Transcription service (`whisper-server` fed fixed-length chunks) | F16 Whisper | `-nt -sns`, `-ac 50*(chunk_s+4)` capped at 1500, and `temperature_inc=0` as a request field | 0.70x the server defaults' CPU core-seconds per second of audio on 20 s chunks, at better WER |
| Voice commands, short utterances (Parakeet through whisper.cpp's `parakeet-cli`) | F16 | `ROCKET_MIN_M=16` in the STT process, and `OMP_WAIT_POLICY=PASSIVE` where another process shares its cores | Offloads the whole encoder of a command shorter than ~1.9 s, of which the default threshold offloads only part [expected] |

**Conditions:** the LLM rows at `pp2048`. The F16 residency and int8 rows on a second RK1 at
1028 MHz, at five NPU workers [HW sweep 2026-11-21]. The speech rows' conditions are in their
sections.

## Weight residency

A [resident weight](README.md#terms) is packed into NPU buffers once and reused by every later
micro-batch and prefill. Its pack (`packB`) therefore runs once instead of on every call. The backend
makes a static F16 weight resident by default when its `K` is at most 2048, which covers
Whisper's encoder. The two residency variables extend it to every weight.

A weight becomes resident the first time the backend sees it at `M` >= 256, the RK3588's tile
cap. Below that, a call [streams](README.md#terms) the weight and leaves any resident copy in
place. A weight packed during warm-up therefore serves every later micro-batch of 256 rows or
more, and no shorter one.

The cache is keyed on the weight's name and a fingerprint of its contents (sixteen 64-byte
windows), not on its address or on `M`. A weight that is reloaded or replaced under the same name
is packed again.

Residency trades RAM for prefill speed. A weight that does not fit the residency budget is
streamed instead, so requesting residency never fails a run. Admission stops at a `MemAvailable`
minimum, `ROCKET_QUANT_RESIDENT_RESERVE_MB`, so a board without swap does not run out of memory.

### F16 weights

`ROCKET_F16_RESIDENT=auto` makes every F16 weight resident within a budget sized from free RAM.
That includes the attention and FFN weights, at `K` of 3072, 8192 and above. It keeps QKV and gate-up
fusion: each fusable group becomes one resident weight, still with one activation pack and one
submit. It does not `madvise` the GGUF, so the file stays mapped for CPU decode. The full benefit
therefore needs RAM for about twice the F16 model.

| Model, F16 | `pp512` speedup | `pp2048` speedup | Weights resident |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct | 1.201-1.228x | 1.150-1.156x | 193 of 193 |
| Ministral-3-3B-Instruct | 1.207-1.211x | 1.151-1.164x | 179 of 179 |
| Ministral-3-8B-Instruct | 1.225-1.235x | 1.190-1.205x | 235 of 235 |
| Qwen3.5-9B | 1.119-1.142x | 1.127-1.164x | 197 of 197 |
| Gemma-4-12B-it | Varies by process | 1.133-1.170x | 307 of 325 |

**Conditions:** `ROCKET_F16_RESIDENT=auto` over the default configuration, which streams every
weight with `K` above 2048. A second RK1 at 1028 MHz (the 900 MHz request), five NPU workers,
llama.cpp b11539, `llama-bench -n 0 -r 3 -t 8` unpinned at the default `-b` and `-ub` [HW sweep
2026-11-21]. [Quantization and prefill speed](#quantization-and-prefill-speed) gives the rates.

On Gemma-4-12B-it the [IOVA window](README.md#terms) fills before every weight is resident.
`ROCKET_N_THREADS=6` places the most weights on this route where the window binds. On
Gemma-4-12B F16 at `pp512`, six workers placed 293 weights, against 286 at five and 274 at seven
[HW sweep 2026-09-02, llama.cpp b10558]. At `pp2048` the `MemAvailable` minimum bound before the
window there, and six workers placed 284 against five's 286 [HW sweep 2026-09-03]. Both ran the
backend of that date.

### Quantized weights

`ROCKET_QUANT_RESIDENT` dequantizes and packs each quantized weight into resident fp16 buffers
once, so prefill has neither the per-micro-batch dequantization nor the per-call `packB`.
Quantized prefill then runs at 0.95-1.00x the same model's F16 GGUF with resident weights, and its
perplexity is bit-identical to the streamed route's:

| Model | `Q4_K_M` streamed, t/s | `Q4_K_M` resident, t/s | F16 resident, t/s | `Q4_K_M` resident against F16 resident |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 39.20 | 57.72 | 60.54 | 0.95x |
| Ministral-3-3B | 34.60 | 50.00 | 52.04 | 0.96x |
| Phi-4-mini | 35.94 | 54.18 | 54.40 | 1.00x |

**Conditions:** `pp2048` at the default `-ub`, llama.cpp b10558, governor `performance`, every
weight resident [HW sweep 2026-08-29/31].

The cost is the full fp16 footprint in resident buffers, which cancels the RAM that quantization
saved. Use `auto`, which sizes its budget from free RAM. A blanket `1` uses the 2 GB
`ROCKET_CACHE_MB` budget, makes only part of a larger model resident, and can be slower than the
streamed default.

Keep the default `-ub` with `auto`. The table compares the configurations on Qwen3.5-9B `Q4_K_M`:

| Configuration | t/s | Speedup over the streamed default |
|---|---:|---:|
| Streamed, default `-ub` | 19.14 | |
| Streamed, `-b 2048 -ub 2048` | 27.26 | 1.424x |
| `auto` with `-b 2048 -ub 2048` | 31.78 | 1.661x |
| `auto`, default `-ub` | 33.45 | 1.752x |

**Conditions:** Qwen3.5-9B `Q4_K_M`, `pp2048`, governor `performance`, every resident
configuration 100% resident [HW sweep 2026-08-29/31]. The model's ~18 GB fp16 form fits the
board's 31 GiB.

On seven models of 1.8-11.9 B, `auto` at the default `-ub` measures 1.35-1.75x the streamed
default, and 1.00-1.66x combined with `-b 2048 -ub 2048`. The default `-ub` is faster on all
seven, partly resident models included, because `-ub 2048` uses RAM for compute buffers that
residency would otherwise use. Below 4 B, `-ub 2048` also costs 14-21% for reasons outside the
dequantization.

### The teardown report

Both residency variables report their outcome when the backend is freed:

    [f16-resident] weights offered to the resident route: N resident on the NPU (X MB),
    M streamed via the per-call pack -- P% resident

The next line names the limit that first declined a weight, with its numbers: the byte budget,
the `MemAvailable` minimum, or a full IOVA window. Below 80% resident, a warning follows. Each
member of a fused group counts once, so the percentage means the same with fusion on and off.

A matmul never offered to the route is in neither count, and its weight is still packed on every
call. Two kinds are never offered: a call below `ROCKET_MIN_M`, and a weight with no stable name,
which the report counts on a separate line.

> [!NOTE]
> A benchmark row does not show residency. A run with a quarter of its weights resident prints
> the same rows as a run with all of them resident. The two can also measure nearly the same t/s.
> Read the teardown report before concluding that residency had no effect.

## Quantized GGUFs

A quantized GGUF prefills on the fp16 route. The backend dequantizes each weight to fp16 as it
is used, one `[K]` row at a time, to the exact values ggml's CPU backend computes. It holds no
whole-model F16 copy. The model stays compact in RAM, and decode runs on the CPU from the
quantized weights. The host's weight repack must be off, as the README's
[Quick start](README.md#quick-start) cautions, unless the host registers the original weight
bytes ([Repacked weights](API.md#repacked-weights)).

The dequantization runs on every micro-batch, so a streamed quantized prefill depends on the
micro-batch size. Where the fp16 form fits RAM, residency removes it, as
[Quantized weights](#quantized-weights) describes. Where it does not, `-b 2048 -ub 2048` spreads
it over more rows. The setting measures 1.18-1.53x the default `-ub 512` on dense models above
8 B, and 0.94-1.13x on models under 4 B parameters. It is slower on two models of the eleven
measured, so measure it per model.

A quantized micro-batch shorter than `ROCKET_MIN_M_QUANT`, 256 rows by default, stays on the CPU.
The offload is faster than the CPU from 142-143 rows on Qwen3.5-9B `Q4_K_M` and from 137 rows on
Qwen3.6-27B `Q4_K_M`. The default is 1.8x that crossover, and is the row count at which quantized
residency starts. A smaller model's crossover is expected to be later, because its per-call
dispatch cost weighs more against its weights [expected]. Over the CPU, the offload measures:

| Micro-batch, rows | 128 | 192 | 256 | 384 | 512 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B, streamed, over the CPU | 0.93x | 1.24x | 1.48-1.50x | 1.82-1.86x | 2.12-2.13x |
| Qwen3.5-9B, `ROCKET_QUANT_RESIDENT=auto`, over the CPU | 0.92x | 1.24-1.25x | 4.19-4.22x | 4.11-4.16x | 4.07-4.17x |
| Qwen3.6-27B, streamed, over the CPU | 0.95x | Not measured | 1.59-1.61x | 2.00-2.04x | 2.41-2.43x |

**Conditions:** `Q4_K_M`, one micro-batch of M rows (`-b` and `-ub` at M) with
`ROCKET_MIN_M_QUANT=1` and `--repack 0`, against the CPU backend with its weight repack on. Both
run 8 threads unpinned on llama.cpp b11539 [HW sweep 2026-11-21/22].

Quantized residency serves no micro-batch under 256 rows, because a weight becomes resident only
at its first call of 256 rows or more. With every weight resident, a micro-batch of 256 rows or
more runs at 4.1-4.2x the CPU on the 9B. Between the crossover and 256 rows the default keeps a
micro-batch on the CPU. There the streamed offload measures 1.24x the CPU at 192 rows on the 9B.
Measure the crossover on your model before lowering `ROCKET_MIN_M_QUANT` below 256.

The dequantization is threaded across the A76 cores by default (`ROCKET_DEQUANT_THREADS`). It is
bit-exact and needs no extra RAM. On a 9B `Q4_K`, eight A76 and A55 threads gave
1.33x the serial prefill at `-ub 512` and 1.19x at `-ub 2048` [HW sweep 2026-06-28].
That run predates pinning the threads to the A76 cores. With the threaded dequantization, `Q4_K`,
`IQ4_XS` and `Q8_0` prefill at about the same rate, from ~0.64x F16 at `pp512` to ~0.85x at
`pp2048`.

## Quantization and prefill speed

Quantization saves RAM and lets a larger model fit. At equal residency it changes prefill speed
little: resident int8 prefills at 0.98-1.10x resident fp16 on four models of 3-9 B. The prefill
matmul is limited by host work and the weight read rather than by the MAC array. int8's 2x and
int4's 4x MAC rates therefore reach the prefill time only in part.

Resident int8 holds half the weight bytes of resident fp16. The default configuration converts
and scatters each weight on every call. Against it, resident int8 prefills at 1.13-1.33x at
`pp512` and 1.15-1.21x at `pp2048` on the 3-9 B models:

| Model, F16 GGUF | Test | Default, t/s | Resident fp16, t/s | Resident int8, t/s | Resident int8, rotated, t/s | int8 against resident fp16 | Rotated against resident fp16 |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | `pp512` | 104.32-105.32 | 126.54-128.13 | 135.74-136.73 | 131.06-131.16 | 1.065-1.073x | 1.023-1.037x |
| Llama-3.2-3B-Instruct | `pp2048` | 93.13-93.65 | 107.68-107.85 | 110.51-111.19 | 106.32-107.90 | 1.026-1.033x | 0.987-1.002x |
| Ministral-3-3B-Instruct | `pp512` | 94.54-94.67 | 114.28-114.61 | 125.35-125.54 | 122.90-123.34 | 1.094-1.097x | 1.075-1.077x |
| Ministral-3-3B-Instruct | `pp2048` | 81.82-82.33 | 94.16-95.26 | 97.40-99.38 | 96.86-97.29 | 1.034-1.047x | 1.017-1.033x |
| Ministral-3-8B-Instruct | `pp512` | 48.41-48.63 | 59.57-59.81 | 60.04-60.19 | 57.46-57.91 | 1.005-1.009x | 0.961-0.972x |
| Ministral-3-8B-Instruct | `pp2048` | 43.57-43.97 | 52.34-52.55 | 51.32-51.55 | 49.25-49.62 | 0.979-0.982x | 0.938-0.948x |
| Qwen3.5-9B | `pp512` | 39.95-41.37 | 44.71-47.24 | 46.56-47.08 | 44.40-45.74 | 0.986-1.049x | 0.955-1.023x |
| Qwen3.5-9B | `pp2048` | 39.03-40.09 | 43.99-46.30 | 45.66-46.15 | 43.94-45.05 | 0.990-1.038x | 0.952-1.023x |
| Gemma-4-12B-it | `pp2048` | 26.16-26.28 | 29.67-30.62 | 33.25-35.64 | 32.25-32.54 | 1.106-1.186x | 1.053-1.089x |

**Conditions:** a second RK1 at 1028 MHz (the 900 MHz request), llama.cpp b11539,
`llama-bench -n 0 -r 3 -t 8` unpinned at the default `-b` and `-ub` [HW sweep 2026-11-21].
Resident fp16 is `ROCKET_F16_RESIDENT=auto`. Resident int8 is
`ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1 ROCKET_INT8_CACHE_MB=0`, rotated by
`ROCKET_INT8_HADAMARD=1`, with every int8 weight resident. Resident fp16 places every weight
except on Gemma-4-12B-it, where the IOVA window fills at 307 of its 325.

> [!NOTE]
> Gemma-4-12B-it's `pp512` rate varies between processes by up to 1.57x, while each process's
> repetitions agree within 0.3 t/s. Its `pp2048` rate holds within 4%. Compare that model at
> `pp2048`.

Resident int8 reads back an int32 partial for each K pass, which fp16 accumulates on the NPU
instead, because the backend has no integer [K-accumulation](README.md#terms). `librocketnpu`
streams the int8 weights through the CBUF, which deepens each pass and removes most of them. The
passes left grow with `K`, which is the likely cause of int8's smaller lead at 8-9 B
[hypothesis].

`ROCKET_INT8_HADAMARD=1` costs prefill speed: the rotated route runs at 0.94-0.98x the unrotated
rate at `pp512` on the table's four 3-9 B models. The rotation and the int8 quantization and
dequantization run on the host for every matmul, spread over the A76 cores
(`ROCKET_CONVERT_THREADS`). Matmuls in one scheduler split that read the same activation convert it
once. On Qwen3.5-9B the threads raise the rotated route's rate 1.15-1.17x and the unrotated
route's 1.02-1.04x, with unchanged output. On that model the rotation also brings the perplexity
cost from +2.83% to +0.97% ([MODELS.md](MODELS.md#int8-w8a8)).

**Conditions:** the threads' gain on an RK1 at 928 MHz (the 900 MHz request), CPU governor
`performance`, against `ROCKET_CONVERT_THREADS=1 ROCKET_NO_ACT_REUSE=1` in the same build
[HW sweep 2026-11-11].

At 12 B the int8 route gains more. On Gemma-4-12B-it at `pp2048`, the unrotated route runs at
1.27-1.36x the default fp16 route and the rotated one at 1.23-1.24x. Unrotated output is far from
fp16's on that model, so set `ROCKET_INT8_HADAMARD=1` there ([MODELS.md](MODELS.md#int8-w8a8)).
Its `K` of 3840 and 15360 take the 60-point rotation. On an RK1 at 928 MHz (the 900 MHz request),
that rotation took about 1.6 s a 512-row micro-batch [HW sweep 2026-11-11].

`ROCKET_INT8_CACHE_MB` sizes the int8 weight budget from free RAM by default: `MemAvailable` at
init less the `ROCKET_QUANT_RESIDENT_RESERVE_MB` reserve, but never less than 4096 MB. Past
4096 MB, a weight is admitted only while `MemAvailable` stays above the reserve. The backend logs
the budget at init, in a `[rocket] ROCKET_INT8_CACHE_MB=auto -> int8 weight budget` line.

On the table's board, with a 9535 MB reserve, the budget came to 21552-21580 MB. It held all 235
of Ministral-3-8B-Instruct's int8 weights (6940 MB) and ran at 0.994-1.000x the `pp512` rate of an
unlimited budget (`0`). A 4096 MB budget holds 141 of those weights (4094 MB). The default ran at
1.103-1.111x the rate of that budget [HW sweep 2026-11-22, the backend of 2026-11-22].

A board whose free RAM is under 4096 MB plus the reserve keeps a 4096 MB budget, and the weights
past it take the fp16 route. In the table's series, a 4096 MB budget held 125 of Qwen3.5-9B's int8
weights and cost 5.6-9.0% of the `pp512` rate against `0`. At `pp2048` it cost 7-8% on
Ministral-3-8B-Instruct and 4-8% on Qwen3.5-9B.

On the same board at `pp512`, the int8 route's whole-tile plan (`ROCKET_MM_STREAM_INT8=0`) runs
Gemma-4-12B-it at 0.48-0.51x its streamed plan and 0.71-0.74x resident fp16. Resident int4 has no
streamed plan, and runs at 0.28x resident fp16 there with all 325 weights resident [HW sweep
2026-11-22].

Use the int8 route to hold a model's weights resident in half the RAM resident fp16 takes. Use
int4 and a quantized GGUF to fit a model in less RAM. [MODELS.md](MODELS.md#precisions) gives
each route's accuracy.

## Gains from the defaults

Each row is one default's gain over the configuration without it:

| Setting | Prefill gain |
|---|---|
| fp16 K-accumulation on the NPU (`ROCKET_KACC`) | 1.19x the host-summed partials |
| CBUF data reuse (`ROCKET_REUSE`) | 1.07x no reuse |
| Resident weights | 1.06x streamed weights |

**Conditions:** Gemma-4-12B F16, `pp2048`, each setting added to the one above it, on earlier
builds of the backend.

CBUF weight streaming (`ROCKET_MM_STREAM`) cuts the whole-tile program's task count. It is
bit-exact, and perplexity is unchanged:

| Model | GGUF | Speedup over `ROCKET_MM_STREAM=0` |
|---|---|---:|
| Llama-3.2-3B | F16 | 1.134-1.143x |
| Llama-3.2-3B | `Q4_K_M` under `ROCKET_QUANT_RESIDENT=auto` | 1.163-1.167x |
| Ministral-3-3B | F16 | 1.120-1.153x |
| Qwen3.5-9B | `Q4_K_M` under `ROCKET_QUANT_RESIDENT=auto` | 1.069-1.110x |

**Conditions:** `pp512`, NPU 928 MHz (the 900 MHz request), three cores, governor `performance`
[HW sweep 2026-10-21].

## Offload threshold

`ROCKET_MIN_M` sets the fewest rows an unquantized weight's matmul needs to offload. Below the
crossover, the per-call dispatch and weight pack cost more than the NPU's per-row speed saves.
The crossover is 16-19 rows on the three F16 models measured. The default of 24 is at least 5 rows
above it, so no model measured is slower than on the CPU. At 24 rows the offload measures
1.15-1.45x the CPU:

| F16 model | `pp16` | `pp20` | `pp24` | `pp32` | `pp64` | `pp128` | Crossover, rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.885-0.889x | 1.042-1.051x | 1.149-1.155x | 1.351-1.356x | 1.863-1.870x | 2.331-2.343x | 18.7-18.9 |
| Llama-3.2-3B | 0.995-0.999x | 1.243-1.246x | 1.444-1.447x | 1.786-1.790x | 2.625-2.655x | 3.765-3.767x | 16.0-16.1 |
| Ministral-3-8B | 0.977-0.986x | 1.069-1.075x | 1.282-1.286x | 1.627-1.629x | 2.912-2.917x | 4.233-4.243x | 16.7-16.9 |

**Conditions:** NPU prefill t/s at `ROCKET_MIN_M=4` on 4 threads pinned to the A76 cores. The
baseline is the faster of the CPU backend on those 4 threads and on 8 unpinned threads.
llama.cpp b11539 [HW sweep 2026-11-21]. The crossover interpolates linearly between the two
lengths around 1.00x.

The crossover depends on the backend build. In the same measurement, llama.cpp a7fb71f with the
backend of 2026-10-09 crosses at 18.9-19.1 rows on Llama-3.2-3B. Behind a boot loader that leaves
SPLL unprogrammed, the crossover is expected to stay below 24 [expected]. On Llama-3.2-3B, SPLL
and the faster clock together raised the 2026-10-09 build's rate 1.04-1.08x at these lengths.

The crossover barely depends on the model, because the weight pack the offload adds and the
compute it saves both scale with `K`·`N`. The dispatch cost does not scale, and it causes the
small drift between models. On whisper-small F16 a threshold of 32 runs at 0.995-1.006x the speed
of `ROCKET_MIN_M=128`, with a byte-identical transcript [HW sweep 2026-10-09]. On Whisper base and
small, the default of 24 gives the same transcript as thresholds of 4, 8 and 32
[HW sweep 2026-11-22].

> [!NOTE]
> `ROCKET_MIN_M` also keeps batched decode on the CPU. whisper.cpp's default beam search presents
> `M`=5 per step. At `ROCKET_MIN_M=4` those steps offload and take 1.68-1.86x as long as on the
> CPU. The whole transcription then takes 1.25-1.34x as long on Whisper base and small, with
> whisper.cpp v1.9.5 [HW sweep 2026-11-22].

For a CTC or transducer encoder on short utterances, set 16 in that process, as
[Voice commands](#voice-commands) describes. Leave an LLM process at the default rather than 16,
where the three F16 models measure 0.89-1.00x the CPU. Do not raise it to 256, the residency cap.
Below 256 rows a llama.cpp model streams its weights and still offloads faster than the CPU,
4.23-4.24x at `pp128` on Ministral-3-8B.

## Attention

The backend computes `softmax(scale·QKᵀ + mask)·V` on the NPU, with the heads spread across its
workers and their submits chained. On Ministral-3-3B F16 its perplexity is within 0.03% of the CPU
backend's attention, 13.8994 against 13.9024. It claims an op at `ROCKET_FLASH_ATTN_MIN_KV` keys
(1024) and `ROCKET_FLASH_ATTN_MIN_T` query tokens (16) or more. From a 1K-token prompt it is
faster than the CPU backend's flash attention, and the gain grows with the context:

| Prompt, tokens | Gemma-4-12B F16, speedup | Llama-3.2-3B F16, speedup |
|---|---:|---:|
| 512 | 0.997-1.001x | 0.993-1.009x |
| 1K | 1.069-1.086x | 1.073-1.083x |
| 2K | 1.199-1.205x | 1.239-1.240x |
| 4K | 1.337-1.339x | 1.494-1.499x |
| 8K | 1.543-1.548x | 1.855-1.865x |
| 16K | 1.936-1.937x | Not measured |

**Conditions:** prefill t/s with the offload over `ROCKET_FLASH_ATTN=0`, which runs the CPU
backend's flash attention with the matmuls still on the NPU. `llama-bench -fa 1` at the default
`-ub` of 512, 8 threads unpinned, llama.cpp b11539 [HW sweep 2026-11-21].

At a 512-token prompt no op reaches the default threshold, so the attention stays on the CPU.
Offloading those ops measures 0.98-1.00x the CPU. A threshold of 512 keys measures within 1.3% of
the default at a 1K prompt. With the process pinned at `taskset 0xf0 -t 4`, the offload measures
1.240-1.252x on Gemma-4-12B at `pp2048`.

The row handoff (`ROCKET_FA_ROWMAJOR`, the default) passes the scores and the softmax output as
plain rows instead of the tile layout. It measures 1.058-1.068x the tiled handoff at `pp2048` and
1.080-1.116x at `pp4096` on Llama-3.2-3B F16 with `-fa 1` [HW sweep 2026-10-11].

The backend declines an op with attention sinks (`src[4]`), which gpt-oss, `mimo2` and
`deepseek4` use, so the CPU computes it.

### Encoder attention

The backend also offloads an attention op that has no mask: an encoder's self-attention, and the
cross-attention of a decoder such as Whisper's. It claims one at the same `ROCKET_FLASH_ATTN_MIN_KV`
and `ROCKET_FLASH_ATTN_MIN_T` thresholds as a masked op, and `ROCKET_FLASH_ATTN_UNMASKED=0`
declines it. The offload shortens a Whisper encode, and the gain grows with the model:

| Whisper model | Encode time | `whisper-cli` wall-clock time | `whisper-cli` CPU core-seconds |
|---|---:|---:|---:|
| `tiny` | 0.916-0.930x | 0.995x | 0.92x |
| `base` | 0.796-0.807x | 0.92x | 0.83x |
| `small` | 0.715-0.718x | 0.88x | 0.78x |

**Conditions:** each figure with the ops offloaded over `ROCKET_FLASH_ATTN_UNMASKED=0`. The
multilingual F16 models, whisper.cpp v1.9.5 with flash attention on (its default), 4 threads
pinned to the A76 cores, the backend of 2026-11-22 [HW sweep 2026-11-22]. The encode time is
`whisper-bench`'s 1500-frame encode. The `whisper-cli` columns transcribe a 176 s English clip at
the default beam search, beam 5 and best of 5.

The transcripts are byte-identical with the ops offloaded and declined. In the `whisper-cli` run,
the decoder's prompt passes of 16 tokens or more also offload their cross-attention, whose keys
are the 1500 encoder frames [hypothesis]. An encoder whose head dimension is not a multiple of 32,
such as SigLIP's 72 in SmolVLM2, keeps its attention on the CPU.

### Host threads for attention

The attention handler's host walks (the Q conversion, the K copy, the V transpose, the mask copy
and the output scatter) run while the scheduler waits. `ROCKET_FA_THREADS=4` spreads them over
the four A76 cores:

| `ROCKET_FA_THREADS` | Host walk time per prefill |
|---:|---:|
| 1 | 5.80 s |
| 2 | 3.17 s |
| 4 | 1.87 s |

**Conditions:** Gemma-4-12B F16, `pp2048`, host pinned to the A76 cores [HW sweep 2026-09-07].

At 4 workers, prefill measured 1.039x as fast as at 1. The walks are 6.0% of a pinned prefill
there, so more workers can add at most ~0.6%. The default stays 1 because the gain is measured on
one model at one shape.

## CPU threads

Run the process on the four A76 big cores with `taskset 0xf0 -t 4`. Both halves of a turn are
faster there than at llama.cpp's 8 threads on all cores:

| Half | Speedup of `taskset 0xf0 -t 4` over `-t 8` |
|---|---|
| NPU prefill | 1.06-1.13x on three models, largest on the smallest |
| CPU decode | 1.32-4.22x on fifteen GGUFs, F16 and quantized, dense and MoE |

**Conditions:** CPU governor `performance`. Prefill at the 600 MHz request [HW sweep 2026-09-01].
Decode `tg64` on llama.cpp b11539 [HW sweep 2026-11-01].

The prefill gain holds with streamed weights: Gemma-4-12B F16 streams all of its weights and still
gains 1.06x. The NPU's share of the work is the same either way. Decode gains least on the F16
GGUFs, 1.32-1.33x, and most on the small and MoE ones, up to 4.22x on DeepSeek-V2-Lite.

A CPU-only prefill is the exception: it runs 4-10% faster at 8 threads than at 4 [HW 2026-11-01],
so the README's CPU baseline uses 8.

The backend's host conversions around each offloaded matmul run on a pool of A76 threads
(`ROCKET_CONVERT_THREADS`), between device calls while the NPU is idle. On the fp16 route these
are the activation's row scaling and the output's unscaling. Spread over the A76 cores they raise
Ministral-3-8B F16 at `pp512` 1.016-1.024x with resident fp16 weights and 1.010-1.014x on the
default route, with unchanged output. The output pass writes 4 bytes for each 2 it reads. It
already runs near the DRAM rate on one core, so more threads gain little there.

**Conditions:** CPU governor `performance`, a memory reset and a discarded warm-up process before
every configuration, three passes in rotated order. The baseline is `ROCKET_CONVERT_THREADS=1` in
the same build [HW sweep 2026-11-11].

## Speech

### Transcription service

`whisper-server` fed fixed-length chunks with no request fields, as OpenWebRX+ sends them, runs
on the server's own settings. Three of them set its CPU cost:

- `-nt -sns`
- An `-ac` of 50 times the chunk length plus 4 s, capped at 1500
- The temperature fallback off, through a `temperature_inc=0` request field, because the server's
  `-nf` flag is parsed and never applied

Together they cut the CPU cost to 0.70x the defaults' core-seconds per second of audio on 20 s
chunks, at better WER. On 30 s chunks with `-nt`, the server takes 0.65x. Without `-nt`, a 30 s
chunk is encoded twice.

**Conditions:** `ggml-small` F16, with the encoder's attention on the CPU [HW sweep].

**Not characterized:** these settings with the encoder's attention offloaded, which is the
default ([Encoder attention](#encoder-attention)).

> [!CAUTION]
> An `-ac` equal to the chunk length makes the decoder loop at the end of the audio. Size `-ac`
> as the list gives it.

> [!NOTE]
> A quantized Whisper model costs more CPU time than the F16 one. Its unnamed tensors keep the
> encoder off the resident route, so each request dequantizes the encoder.

### Voice commands

A 1.3-6 s command is 16-75 encoder rows. A threshold of 32 offloads every encoder GEMM of a
command from 2.6 s up, and 24 of the 265 at 1.3-2.5 s. At 16, every encoder GEMM offloads, and the
encoder is faster than the CPU at every length measured. It frees 37-64% of the CPU core-seconds,
with the transcript byte-identical:

| Audio | CPU only, per utterance | Threshold 32 | Threshold 16 | 16 over the CPU | 16 over 32 |
|---|---:|---:|---:|---:|---:|
| 1.3-1.8 s | 421 ms, 1.66 core-s | 409 ms, 1.57 core-s | 362 ms, 1.04 core-s | 1.146-1.196x | 1.116-1.161x |
| 2.1-2.5 s | 554 ms, 2.19 core-s | 523 ms, 1.99 core-s | 390 ms, 1.15 core-s | 1.422-1.428x | 1.340-1.342x |
| 2.6-3.9 s | 786 ms, 3.10 core-s | 462 ms, 1.33 core-s | 466 ms, 1.35 core-s | 1.586-1.708x | 0.935-1.005x |
| 4.4-5.9 s | 1202 ms, 4.77 core-s | 598 ms, 1.73 core-s | 592 ms, 1.71 core-s | 2.017-2.047x | 0.998-1.022x |

**Conditions:** Parakeet TDT 0.6B v3 F16 through whisper.cpp v1.9.5's `parakeet-cli`, 4 threads
pinned to the A76 cores, ten LibriSpeech test-clean utterances a length. Each time is a median
per-utterance cost without the model load [HW sweep 2026-11-22]. The CPU-only configuration loads
the backend and declines every op.

At the default threshold of 24, a command of ~1.9 s or more is expected to offload every encoder
GEMM, as at 16 [expected]. A shorter command is expected to measure as at 32 [expected].

Beside a CPU-bound process, the offloaded STT process slows more than a CPU-only one does.
`OMP_WAIT_POLICY=PASSIVE` recovers most of that slowdown.

## Measuring a configuration

The NPU clock drops at idle, so a cold run measures ~15% low. Discard a warm-up run before
timing, run at least three times, and compare warm runs.

The NPU's DDR read rate depends on the boot loader, so compare a rate against one here only under
the same boot-loader condition. [Boot loader](README.md#boot-loader) says which U-Boot builds
program SPLL.

Run once with `ROCKET_LOG_STDERR=1` to confirm that the intended route and budget engaged.
`llama-bench` silences the host's logger, and with it every backend line, unless it runs with
`-v`. [Diagnostic logging](API.md#diagnostic-logging) lists the lines that variable restores.
