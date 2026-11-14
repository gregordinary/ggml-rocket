# ggml-rocket: tuning guide

This guide gives the configuration for each workload and what each setting is worth. Start with
the [README](README.md), which covers the build and a first run. [API.md](API.md) lists every
environment variable with its default.

Unless a Conditions line says otherwise, a figure here was measured on the RK3588 with the NPU at
~816 MHz (the 600 MHz request), warm. Its baseline is the 8-thread CPU backend on the same GGUF.

## Recommended configurations

The defaults enable fp16 [K-accumulation](README.md#terms), [CBUF](README.md#terms) operand reuse,
CBUF weight streaming, asymmetric tiling, threaded dequantization and attention offload. None
needs an environment variable. Add the settings the table gives for the workload:

| Workload | Precision | Add | Effect |
|---|---|---|---|
| Interactive chat, short prompts | Any | `taskset 0xf0 -t 4`, and a `Q4_K_M` GGUF for decode t/s | The turn is decode-bound, so the quantization and the thread placement set its speed. A quantized prompt under 512 rows stays on the CPU, and an F16 one under 32 |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form fits RAM, wholly or partly | `ROCKET_QUANT_RESIDENT=auto`, at the default `-ub` | 1.35-1.75x the default configuration's prefill on seven dense models of 1.8-11.9 B. Needs free RAM of about the fp16 model size |
| Agentic, RAG or long prompts | Quantized GGUF whose fp16 form does not fit | `-b 2048 -ub 2048` | 0.91-1.53x the default `-ub 512`, by model, so measure it |
| Agentic or RAG, repeated prefill | F16, with RAM for ~2x the model | `ROCKET_F16_RESIDENT=auto` and `ROCKET_N_THREADS=6` | 1.08x the default configuration's prefill on three 3-4 B models |
| Agentic or RAG, repeated prefill | F16, with RAM for ~1.5x the model | `ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1`, and above 4 B a `ROCKET_INT8_CACHE_MB` that holds the int8 weights | 1.11-1.12x the default configuration's prefill on two 3 B models and at least 1.05-1.07x on two 8-9 B models ([Quantization and prefill speed](#quantization-and-prefill-speed)), at a wikitext perplexity 0.7-2.8% above fp16. At the default 4 GB budget the weights past it run on the fp16 route, 1-5% slower at 8-9 B |
| Medium or long prompts | MoE whose experts offload (gpt-oss, DeepSeek-V2-Lite) | `-b 2048 -ub 2048` and `ROCKET_N_THREADS=6` | 1.25-1.32x the default `-ub` at `pp2048`. [MOE.md](MOE.md) has the rest |
| Model too large at F16 | Any | A `Q4_K_M` GGUF, or `ROCKET_INT4=1` with an F16 GGUF | Reduces the footprint at a slower prefill than fp16 |
| Transcription service (`whisper-server` fed fixed-length chunks) | F16 Whisper | `-nt -sns`, `-ac 50*(chunk_s+4)` capped at 1500, and `temperature_inc=0` as a request field | 0.70x the server defaults' CPU core-seconds per second of audio on 20 s chunks, at better WER |
| Voice commands, short utterances (Parakeet through whisper.cpp's `parakeet-cli`) | F16 | `ROCKET_MIN_M=16` in the STT process, and `OMP_WAIT_POLICY=PASSIVE` where another process shares its cores | Offloads the whole encoder of a command shorter than 2.6 s, where the default threshold offloads 24 of its 265 GEMMs |

**Conditions:** the LLM rows at `pp2048`. The speech rows' conditions are in their sections.

## Weight residency

A [resident weight](README.md#terms) is packed into NPU buffers once and reused by every later
micro-batch and prefill. Its pack (`packB`) therefore runs once instead of on every call. The backend
makes a static F16 weight resident by default when its `K` is at most 2048, which covers
Whisper's encoder. The two residency variables extend it to every weight.

A weight becomes resident the first time the backend sees it at `M` >= 256, the RK3588's tile
cap. Below that, a call [streams](README.md#terms) the weight and leaves any resident copy in
place. A weight packed during warm-up therefore serves a later prefill of any length.

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

| Model, F16 | Streamed weights, t/s | `ROCKET_F16_RESIDENT=auto`, t/s | Speedup |
|---|---:|---:|---:|
| Llama-3.2-3B | 56.01 | 60.54 | 1.081x |
| Ministral-3-3B | 48.28 | 52.04 | 1.078x |
| Phi-4-mini | 50.19 | 54.40 | 1.084x |

**Conditions:** llama.cpp b10558, `pp2048` at the default `-b` and `-ub`, every weight resident
[HW sweep 2026-08-29].

`ROCKET_N_THREADS=6` places the most weights on this route where the [IOVA window](README.md#terms)
binds. On Gemma-4-12B F16 at `pp512`, six workers placed 293 weights, against 286 at the default
five and 274 at seven [HW sweep 2026-09-02]. At `pp2048` the `MemAvailable` minimum binds before
the window, and six workers placed 284 against five's 286 [HW sweep 2026-09-03].

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

A quantized prefill shorter than `ROCKET_MIN_M_QUANT`, 512 rows by default, stays on the CPU,
where it is faster than a dequantize-bound offload. The default sits just above the measured
crossover of ~360 rows on 9B and 27B `Q4_K`.

The dequantization is threaded across the A76 cores by default (`ROCKET_DEQUANT_THREADS`). It is
bit-exact and needs no extra RAM. On a 9B `Q4_K`, eight A76 and A55 threads gave
1.33x the serial prefill at `-ub 512` and 1.19x at `-ub 2048` [HW sweep 2026-06-28].
That run predates pinning the threads to the A76 cores. With the threaded dequantization, `Q4_K`,
`IQ4_XS` and `Q8_0` prefill at about the same rate, from ~0.64x F16 at `pp512` to ~0.85x at
`pp2048`.

## Quantization and prefill speed

Quantization saves RAM and lets a larger model fit. At equal residency it does not raise prefill
speed at this operating point. The prefill matmul is limited by DMA and dispatch rather than by
the MAC array. int8's 2x and int4's 4x MAC rates therefore do not reach the prefill time.

The resident int8 route prefills at 0.95-0.99x resident fp16 on 3 B models, with half the
resident weight bytes. On an 8 B model it prefills at about resident fp16's rate. Against the default configuration, which
converts and scatters each weight on every call, it is 1.11-1.16x at 3 B and 1.05-1.07x at
8-9 B:

| Model, F16 GGUF | Test | Default, t/s | `ROCKET_F16_RESIDENT=auto`, t/s | Resident int8, t/s | Resident int8 against resident fp16 |
|---|---|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct | `pp512` | 95.6-96.1 | 113.2-113.8 | 107.5-109.5 | 0.948-0.967x |
| Llama-3.2-3B-Instruct | `pp2048` | 84.4-84.7 | 97.0-97.6 | 94.1-94.8 | 0.964-0.978x |
| Ministral-3-3B-Instruct | `pp512` | 85.8-86.3 | 101.1-101.8 | 98.8-100.1 | 0.971-0.990x |
| Ministral-3-3B-Instruct | `pp2048` | 74.4-74.9 | 85.1-86.1 | 83.3-83.7 | 0.968-0.984x |
| Ministral-3-8B-Instruct | `pp512` | 43.8-44.0 | 51.7-52.0 | 46.2-46.8 | 0.893-0.899x |
| Ministral-3-8B-Instruct | `pp2048` | 39.5-39.6 | 45.7-46.2 | 42.0-42.1 | 0.908-0.921x |
| Qwen3.5-9B | `pp512` | 36.0-36.4 | 41.3-41.8 | 38.2-38.3 | 0.914-0.927x |
| Qwen3.5-9B | `pp2048` | 35.5-35.9 | 39.7-40.8 | 37.6-37.7 | 0.921-0.949x |

**Conditions:** RK3588, NPU at 928 MHz (the 900 MHz request), llama.cpp b11539,
`llama-bench -n 0 -r 3 -t 8` at the default `-b` and `-ub`, governor `performance`. Resident int8
is `ROCKET_INT8=1 ROCKET_INT8_RESIDENT=1` without the Hadamard rotation, with every weight resident.
The 8-9 B rows set `ROCKET_INT8_CACHE_MB=0` for that. The 3 B rows were measured 2026-11-06 and
the 8-9 B rows 2026-11-07.

Resident int8 reads back an int32 partial for each K pass, which fp16 accumulates on the NPU
instead, because the backend has no integer [K-accumulation](README.md#terms). `librocketnpu`
streams the int8 weights through the CBUF, which deepens each pass and removes most of them. The
passes left grow with `K`.

The table's 8-9 B int8 rows summed the passes in an int64 accumulator.
`librocketnpu` now adds each pass straight into the output, with unchanged output. The int8 route
then runs Ministral-3-8B-Instruct 1.07-1.12x faster than in the table, and Llama-3.2-3B-Instruct
1.01-1.03x [HW sweep 2026-11-07]. That is 51.5-52.0 t/s at `pp512` on the 8 B model, resident fp16's rate.

`ROCKET_INT8_HADAMARD=1` costs prefill speed. The rotation and the int8 quantization and
dequantization run on the host for every matmul, spread over the A76 cores
(`ROCKET_CONVERT_THREADS`). Matmuls in one scheduler split that read the same activation convert it
once. On Qwen3.5-9B the rotated route then runs at 41.4-42.0 t/s at `pp512`, 0.951-0.972x its
unrotated rate, against 0.855-0.867x with the conversions on one thread. The threads raise the
rotated route's rate 1.15-1.17x and the unrotated route's 1.02-1.04x, with unchanged output. On that
model the rotation also brings the perplexity cost from +2.83% to +0.97%
([MODELS.md](MODELS.md#int8-w8a8)).

**Conditions:** CPU governor `performance`, a memory reset and a discarded warm-up process before
every configuration, three passes in rotated order. One backend build, with one thread selected by
`ROCKET_CONVERT_THREADS=1 ROCKET_NO_ACT_REUSE=1` [HW sweep 2026-11-11].

At 12 B the int8 route gains more. On Gemma-4-12B-it F16, with every int8 weight resident
(`ROCKET_INT8_CACHE_MB=0`), it runs at 1.35-1.37x the default fp16 route at `pp512` and 1.30-1.32x
at `pp2048`. Unrotated output is far from fp16's on that model, so set `ROCKET_INT8_HADAMARD=1`
there ([MODELS.md](MODELS.md#int8-w8a8)). Rotated, the route runs at 31.8-31.9 t/s at `pp512`,
against 25.5-25.9 t/s for the fp16 route. The rotation is the difference: its `K` 3840 and 15360
take the 60-point construction, which costs about 1.6 s a 512-row micro-batch.

**Conditions:** CPU governor `performance`, a memory reset and a discarded warm-up process before
every configuration, three passes in rotated order [HW sweep 2026-11-11]. The rotated and fp16
rates come from two series run the same day.

**Not characterized:** the rotated route at `pp2048` with the current rotation.

The default `ROCKET_INT8_CACHE_MB` of 4096 holds 58-62% of an 8-9 B model's int8 weights, and the
rest take the fp16 route. That costs 3-5% of the prefill rate on Ministral-3-8B-Instruct and 1-3%
on Qwen3.5-9B against a budget that holds them all. With
its whole-tile plan (`ROCKET_MM_STREAM_INT8=0`) the same route measures 0.61-0.66x resident fp16.
Resident int4 has no streamed plan and measured 0.53x fp16 at ~816 MHz.

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
The crossover is 21-25 rows on all three models measured. The default of 32 is above it, so no
model measured is slower than on the CPU:

| F16 model | `pp16` | `pp32` | `pp64` | `pp128` |
|---|---:|---:|---:|---:|
| 0.8B | 0.75x | 1.20x | 1.71x | 1.96x |
| 3B | 0.78x | 1.40x | 2.51x | 3.68x |
| 8B | 0.77x | 1.46x | 2.75x | 3.90x |

**Conditions:** NPU prefill t/s over the faster of 4 and 8 CPU threads, CPU governor
`performance` [HW sweep 2026-10, llama.cpp a7fb71f].

The crossover barely depends on the model, because the weight pack the offload adds and the
compute it saves both scale with `K`·`N`. The dispatch cost does not scale, and it causes the
small drift between models. On whisper-small F16 the default runs at 0.995-1.006x the speed of
`ROCKET_MIN_M=128`, with a byte-identical transcript.

> [!NOTE]
> `ROCKET_MIN_M` also keeps batched decode on the CPU. whisper.cpp's default beam search presents
> `M`=5 per step. At `ROCKET_MIN_M=4` those steps offload and run 2.3x as slow as on the CPU,
> which makes the whole transcription 1.40x as slow.

For a CTC or transducer encoder on short utterances, set 16 in that process, as
[Voice commands](#voice-commands) describes. Leave an LLM process at the default. Do not raise it
to 256, the residency cap. Below 256 rows a llama.cpp model streams its weights and still
offloads faster than the CPU, 3.90x at `pp128` on the 8B.

## Attention

The backend computes `softmax(scale·QKᵀ + mask)·V` on the NPU, with the heads spread across its
workers and their submits chained. On Ministral-3-3B F16 its perplexity is within 0.03% of the CPU
backend's attention, 13.8994 against 13.9024. It claims an op at `ROCKET_FLASH_ATTN_MIN_KV` keys
(1024) and `ROCKET_FLASH_ATTN_MIN_T` query tokens (16) or more:

| Context | Speedup over the CPU backend's flash attention |
|---|---:|
| 1K or less | Parity |
| 4K | 1.07x |
| 8K | 1.50x |
| 16K | 1.25x |

**Conditions:** Gemma-4-12B F16, governor `performance` [HW sweep 2026-06-26/28].

**Not characterized:** these ratios with the current softmax and the row handoff of the scores.

The row handoff (`ROCKET_FA_ROWMAJOR`, the default) passes the scores and the softmax output as
plain rows instead of the tile layout. It measures 1.058-1.068x the tiled handoff at `pp2048` and
1.080-1.116x at `pp4096` on Llama-3.2-3B F16 with `-fa 1` [HW sweep 2026-10-11].

The backend declines an op with attention sinks (`src[4]`), which gpt-oss, `mimo2` and
`deepseek4` use, so the CPU computes it. An unmasked op, an encoder's self-attention, stays on the
CPU unless `ROCKET_FLASH_ATTN_UNMASKED` is set. On whisper small that offload took 36% longer than
the CPU kernel, because the `[T, n_kv]` scores make a round trip through the host softmax.

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

**Conditions:** `ggml-small` F16 [HW sweep].

> [!CAUTION]
> An `-ac` equal to the chunk length makes the decoder loop at the end of the audio. Size `-ac`
> as the list gives it.

> [!NOTE]
> A quantized Whisper model costs more CPU time than the F16 one. Its unnamed tensors keep the
> encoder off the resident route, so each request dequantizes the encoder.

### Voice commands

A 1.3-6 s command is 16-75 encoder rows. The default `ROCKET_MIN_M` of 32 offloads every encoder
GEMM of a command from 2.6 s up, and 24 of the 265 at 1.3-2.5 s. At 16, the encoder ties
the CPU at 1.3-1.8 s and runs 1.18-1.73x the CPU's speed at 2-6 s. It frees 34-61% of the CPU
core-seconds, with the transcript unchanged.

**Conditions:** Parakeet F16 through whisper.cpp's `parakeet-cli` [HW sweep].

Beside a CPU-bound process, the offloaded STT process slows more than a CPU-only one does.
`OMP_WAIT_POLICY=PASSIVE` recovers most of that slowdown.

## Measuring a configuration

The NPU clock drops at idle, so a cold run measures ~15% low. Discard a warm-up run before
timing, run at least three times, and compare warm runs.

Run once with `ROCKET_LOG_STDERR=1` to confirm that the intended route and budget engaged.
`llama-bench` silences the host's logger, and with it every backend line, unless it runs with
`-v`. [Diagnostic logging](API.md#diagnostic-logging) lists the lines that variable restores.
