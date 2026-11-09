# ggml-rocket: API and environment reference

This reference lists every `ROCKET_*` environment variable with its default and effect. It then
covers the diagnostic logging, the precision declarations the backend honors, and the
implementation notes.
[TUNING.md](TUNING.md) says which variables to set for a workload, and `ggml-rocket.h` documents
the exported functions.

## Environment variables

The tables in this section list the `ROCKET_*` environment variables the backend and `librocketnpu`
read, and each row marks a variable that `librocketnpu` reads. `sudo` drops the environment, so run under `sudo -E`.

A variable the backend reads that is set but empty is treated as unset, not as `0`, and takes its
default. A wrapper that forwards an unset `$VAR` produces that form, such as `ROCKET_MIN_M=`.
Two groups read an empty value differently:

- `librocketnpu` reads an empty `ROCKET_KACC`, `ROCKET_REUSE`, `ROCKET_MM_STREAM` or
  `ROCKET_MM_ASYM` as `0`, which turns that option off. An empty `ROCKET_WAIT_MS` takes its
  default, and `ROCKET_NO_KACC` set to any value turns K-accumulation off.
- The presence-only diagnostics are on whenever they are set, an empty value included. None of
  them changes a result: `ROCKET_DEBUG`, `ROCKET_DEBUG_GRAPH`, `ROCKET_MM_PROFILE`,
  `ROCKET_AB`, `ROCKET_DUMP` and `ROCKET_FA_TIMING`.

### Matmul offload

| Variable | Default | Effect |
|---|---|---|
| `GGML_BACKEND_PATH` | unset | The path from which the host's ggml loads `libggml-rocket.so`. Set it, as an absolute path, to load the backend |
| `ROCKET_KACC` | on | (read by `librocketnpu`) [K-accumulation](README.md#terms): the NPU sums the fp16 `K` partials. `=0`, or `ROCKET_NO_KACC` set to any value, sums them on the host in fp32 instead, which is byte-exact and slower ([TUNING.md](TUNING.md#gains-from-the-defaults)) |
| `ROCKET_REUSE` | auto with K-accumulation on, `0` with it off | (read by `librocketnpu`) Forces the [CBUF](README.md#terms) operand reuse mode: `0` off, `1` weight reuse, `2` data reuse. Auto picks, per shape, the operand with the longer reuse run, and data reuse on a tie |
| `ROCKET_MM_STREAM` | on | (read by `librocketnpu`) CBUF weight streaming: the fp16 planner streams each weight slice through the CBUF, so one NPU task covers up to 4096 output columns. `=0`, or any `ROCKET_MM_MT`, `ROCKET_MM_NT` or `ROCKET_MM_KT` override, selects the whole-tile plan |
| `ROCKET_MM_ASYM` | on | (read by `librocketnpu`) Asymmetric tiling in the whole-tile plan, for a shape that tiles both `N` and `K`: the tile's `N` halves to 128, and its `K` grows from 384 to 512. `=0` forces the symmetric plan, whose fp16 output can differ because the `K` partials round at different points |
| `ROCKET_N_THREADS` | 5 | The number of NPU workers, 1 to 8, each with its own [IOVA window](README.md#terms), and a value outside that range is ignored. Set 6 for the native-quant MoE route, and for the F16 residency route where the IOVA window binds ([TUNING.md](TUNING.md#f16-weights)). The default stays 5 because the dense fp16 route gains nothing past five workers ([MOE.md](MOE.md#worker-count)) |
| `ROCKET_MIN_M` | 32 | The fewest rows (`M`) an unquantized weight's `MUL_MAT` needs to offload, and a value below 4 is raised to 4. Set 16 for a CTC or transducer encoder on short utterances ([TUNING.md](TUNING.md#offload-threshold)) |
| `ROCKET_MIN_M_QUANT` | 512 | The fewest rows a quantized weight's `MUL_MAT` needs to offload, never below `ROCKET_MIN_M`. A shorter quantized prefill stays on the CPU, where it is faster ([TUNING.md](TUNING.md#quantized-ggufs)) |
| `ROCKET_DEQUANT_THREADS` | auto | The number of CPU threads that dequantize a quantized weight on each micro-batch. Auto, which `0` also selects, is the A76 big-core count, or the core count capped at 8 where no big cluster is found. `1` forces one thread |
| `ROCKET_NO_FUSE` | off | `1` turns off QKV and gate-up fusion, so each matmul is routed on its own. Fusion also turns off under `ROCKET_NO_STREAM`, `ROCKET_FORCE_PREPACK`, `ROCKET_INT8` and `ROCKET_INT4`, because the fused route is fp16 only |
| `ROCKET_NO_STREAM` | off | `1` turns off the persistent context that the per-call route uses for [streamed weights](README.md#terms), so each call opens and releases its own NPU file descriptors and buffers. It also turns fusion off |
| `ROCKET_WAIT_MS` | 8000 | (read by `librocketnpu`) The time in ms to wait for a job's output fence, and a value below 1 takes the default. The default covers a multi-second CPU-side stall, such as a large model faulting in under low RAM |

### Weight residency

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_F16_RESIDENT` | off | `auto` makes every static F16 weight resident at any `K`, within a budget sized from free RAM. Off, only weights with `K` <= 2048 are resident. It takes the same values as `ROCKET_QUANT_RESIDENT` ([TUNING.md](TUNING.md#f16-weights)) |
| `ROCKET_QUANT_RESIDENT` | off | Dequantizes each quantized weight to fp16 once and holds it resident. `auto` sizes the budget from free RAM less `ROCKET_QUANT_RESIDENT_RESERVE_MB`, `<N>` sets an `N`-MB budget, and `1` uses the `ROCKET_CACHE_MB` budget ([TUNING.md](TUNING.md#quantized-weights)) |
| `ROCKET_QUANT_RESIDENT_RESERVE_MB` | `max(6 GiB, 30 % of RAM)` | The RAM, in MB, that the `auto` budgets of both residency variables hold back from `MemAvailable`, and the `MemAvailable` minimum at which their admission stops. It also replaces the MoE expert budget's 6 GiB reserve. Set it where the KV cache and activations need a different headroom |
| `ROCKET_CACHE_MB` | 2048 | The resident fp16 weight budget in MB, `0` for unlimited. The residency variables set it from their `auto` or `<N>` value, and `ROCKET_F16_RESIDENT`'s wins where both are on. A valid `ROCKET_CACHE_MB` overrides both, and an empty or malformed one is ignored |
| `ROCKET_DEQUANT_CACHE_MB` | off | A host cache, in MB, of each quantized weight's fp16 form on the streamed route, `0` for unlimited, so a weight is dequantized once. It is a diagnostic that isolates the dequantization cost, and the teardown `[dq-cache]` line reports the weights held, skipped and still streamed |
| `ROCKET_FORCE_PREPACK` | off | `1` makes every F16 weight resident at any `K`, each on its own, within a 2 GB budget unless `ROCKET_CACHE_MB` sets one. It is the diagnostic form and turns fusion off. `ROCKET_F16_RESIDENT` is the setting for a production run |
| `ROCKET_NO_PREPACK` | off | `1` turns off resident F16 weights on the single-matmul route, the `K` <= 2048 default included, so those weights are streamed |
| `ROCKET_PREPACK_MADVISE` | off | `1` releases each F16 weight's GGUF pages with `madvise` once the weight is resident, which frees RAM. It breaks CPU decode, so set it only for a prefill-only run |

### Precision

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_INT8` | off | `1` selects the W8A8 int8 route. On the RK3588, with `ROCKET_INT8_RESIDENT=1`, it prefills at 0.95-0.99x resident fp16 ([TUNING.md](TUNING.md#quantization-and-prefill-speed)). On the RK3576 it is required, because [the RK3576 route](README.md#terms) is the only matmul route there |
| `ROCKET_INT8_HADAMARD` | off | `1` adds a Hadamard rotation to the RK3588 int8 route, and with it the route's output is coherent. The RK3576 route always rotates and ignores it |
| `ROCKET_INT8_RESIDENT` | off | With `ROCKET_INT8=1` on the RK3588, `1` holds the int8 weights resident, within `ROCKET_INT8_CACHE_MB` |
| `ROCKET_INT8_CACHE_MB` | 4096 | The resident int8 weight budget in MB, `0` for unlimited, counted against the weights' actual NPU buffer size. A weight over the budget takes the fp16 route |
| `ROCKET_INT4` | off | `1` selects the W4A4 int4 route from an F16 GGUF, with group-wise scales and a Hadamard rotation ([MODELS.md](MODELS.md#int4-w4a4)) |
| `ROCKET_INT4_RESIDENT` | off | With `ROCKET_INT4=1`, `1` holds the int4 weights resident at ~¼ of the fp16 footprint. It needs group-wise scales. A per-channel weight, or one over the budget or past a full IOVA window, takes the per-call int4 route |
| `ROCKET_INT4_GROUP` | 128 | The number of `K` elements that share one int4 scale, which must divide `K` and be a multiple of 32, and `0` selects per-channel scales. A smaller group is more accurate and reads back more partials |
| `ROCKET_INT4_HADAMARD` | 1 | The int4 route's Hadamard rotation, which raises per-matmul cosine from ~0.79 to ~0.98. `0` turns it off |
| `ROCKET_INT4_CACHE_MB` | 4096 | The int4 weight budget in MB, `0` for unlimited, which caps both the per-call route's host cache and the resident total. A weight over the budget falls back to fp16 |
| `ROCKET_BF16` | off | `1` selects the exact bf16 route, which writes its fp32 result straight to `dst`. Without it, a bf16 weight converts to fp16 and takes the fp16 route |
| `ROCKET_BF16_CACHE_MB` | off | A host cache, in MB, of the bf16 route's fp32 weights, `0` for unlimited. It removes the per-micro-batch bf16-to-fp32 conversion, the route's largest host cost, at twice the bf16 weight's size in RAM |

### MoE experts

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_MOE` | unset | Which MoE expert ops (`MUL_MAT_ID`) the backend [claims](README.md#terms): unset claims a quantized stack the [pre-flight](README.md#terms) can reserve and that meets both size minimums, `1` claims every one it can compute, and `0` none. Set `1` or `0` only to compare configurations ([MOE.md](MOE.md#placement)) |
| `ROCKET_MOE_MIN_TOKENS` | 512 | The smallest micro-batch, in tokens, at which the backend claims a `MUL_MAT_ID` op, under every `ROCKET_MOE` value, `1` included. It keeps short prefills and decode on the CPU, and is never below `ROCKET_MIN_M` |
| `ROCKET_MOE_NATIVE` | on | Selects the [native-quant](README.md#terms) route. `0` selects the fp16 dequant route, which the default placement never claims, so `0` takes effect only with `ROCKET_MOE=1` |
| `ROCKET_MOE_CACHE_MB` | auto | The resident expert budget in MB, which the pre-flight and the ingest both charge, `0` for unlimited. `auto` is `MemAvailable` minus 6 GiB, resolved once per process ([MOE.md](MOE.md#the-expert-budget)) |
| `ROCKET_MOE_CHARGE_ALL_SOURCE` | off | `1` charges the whole mapped GGUF weight buffer once, so the pre-flight also counts the source of the stacks it leaves on the CPU, and places fewer stacks |
| `ROCKET_MOE_GROUP` | 0 | The K-group the native-quant route quantizes on. `0` picks the largest divisor of `K` that is a multiple of 32 and fits the CBUF as one K-tile. A smaller group is more accurate and proportionally slower |
| `ROCKET_MOE_M_BUCKET` | 64 | The smallest per-expert row count the default placement claims, rounded up to a power of two. Raise it only to trade more padding for fewer distinct row counts |
| `ROCKET_MOE_MIN_WORK` | 340 | The least work one expert dispatch must do for the default placement to claim the op, in millions of MACs of `M_e · K · N`. `0` disables it. `240` also claims the shape just below it, which repays its ingest only after ~7700 tokens of prefill |
| `ROCKET_MOE_COSINE` | off | Reports, for each offloaded `MUL_MAT_ID`, the cosine between the NPU's output and a CPU reference |

### Attention

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_FLASH_ATTN` | on | Offloads `FLASH_ATTN_EXT`, prefill attention, to the NPU. `0` disables the offload ([TUNING.md](TUNING.md#attention)) |
| `ROCKET_FLASH_ATTN_MIN_KV` | 1024 | The minimum `n_kv`, the context length, at which the backend offloads an attention op. The threshold is on `n_kv` because every micro-batch has ~512 `n_tokens` at the default `-ub`, whatever the context |
| `ROCKET_FLASH_ATTN_MIN_T` | 16 | The minimum `n_tokens` at which the backend offloads an attention op, which keeps single-token decode on the CPU |
| `ROCKET_FLASH_ATTN_UNMASKED` | off | Claims `FLASH_ATTN_EXT` ops that have no mask, an encoder's self-attention. It is off because the one encoder measured was slower with it |
| `ROCKET_FA_CHAIN` | on | `librocketnpu` batches each worker's per-head QK submits into one NPU job, and its AV submits into a second. `0` forces the per-head route, with a bit-identical output |
| `ROCKET_FA_ROWMAJOR` | on | (read by `librocketnpu`) The NPU writes the scores and reads the softmax's output as plain rows, so the host converts neither to or from the tile layout. `0` restores the tiled handoff. A head alone in its chained group at long context differs by a few fp16 ulps between the two |
| `ROCKET_ATTN_HOST_SOFTMAX` | host | (read by `librocketnpu`) A nonzero value runs the attention softmax on the host, and `0` runs it on the NPU. Unset, flash attention runs it on the host, because its scores are already there for the additive mask |
| `ROCKET_FA_QSCALE` | 1 | Folds the op's scale into Q as the handler converts Q to fp16, so the fp16 scores overflow only past `65504/scale` ([`FLASH_ATTN_EXT` accumulator](#flash_attn_ext-accumulator)) |
| `ROCKET_FA_THREADS` | 1 | The number of host workers for the attention handler's gather and scatter, clamped to the `ROCKET_DEQUANT_THREADS` pool. Set 4 for the measured gain ([TUNING.md](TUNING.md#host-threads-for-attention)) |
| `ROCKET_FLASH_ATTN_NO_CTX` | off | Forces attention onto the per-call route, with its own file descriptors and scratch, instead of the persistent `rocket_fa_ctx` |

### RK3576

The thirteen `ROCKET_RK3576_*` variables apply only on the RK3576, and their defaults give the
route [RK3576.md](RK3576.md) describes. The first five set the route's accuracy, and two of them,
`ROCKET_RK3576_AROW` and `ROCKET_RK3576_KSPLIT`, change what the model computes:

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_RK3576_NCAL` | 2 | Calibration forwards before the per-column output scale is frozen, each one two device passes. A value below 1 is replaced by the default ([RK3576.md](RK3576.md#calibration)) |
| `ROCKET_RK3576_CALSAFE` | 3.0 | Divisor applied to the frozen column maximum, and a value of 0 or below is replaced by the default ([RK3576.md](RK3576.md#calibration)) |
| `ROCKET_RK3576_BOOTMARGIN` | 1.5 | Slack on the calibration passes' own output scale, and a value below 1 is replaced by the default. A calibration forward's output is correct at a scale this much looser, which costs resolution and nothing else |
| `ROCKET_RK3576_AROW` | 0 | `1` selects a per-row activation scale in place of the per-tensor one on every offloaded GEMM, and `2` only on the weights the K-split cut. A value outside 0-2 acts as `1`. Measure a candidate model end to end first ([RK3576.md](RK3576.md#the-route)) |
| `ROCKET_RK3576_KSPLIT` | off | Cuts a `K` of 6176 or more, which the entry refuses, into chunks on the host. Off, the backend declines that matmul ([RK3576.md](RK3576.md#splitting-k)) |

The other eight compare configurations or measure cost:

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_RK3576_KCHUNK` | unset | Moves a chunk size to the head of the preference list (2304, 2048, 1536, 1024, 512, 256, 128, 64, 32). Only 2304, 2048 and 1536 were measured clean, so time a model whose chunking uses another size before quoting its figures |
| `ROCKET_RK3576_FORCESPLIT` | off | Cuts a `K` the part would take whole, to separate a defect in the split code from a property of the route. At `K=1536` the rotation is bit-identical either way. Use it with `ROCKET_RK3576_KCHUNK` |
| `ROCKET_RK3576_WSA` | on | Passes the entry the weight's cached per-column `sum_k \|qB\|`, so the entry does not recompute it on every call. The output is bit-identical. `=0` measures the cost of the pass |
| `ROCKET_RK3576_WDEV` | on | Packs each converged weight into a device buffer and frees the host `qB`, removing the per-call cube memset, copy and weight-buffer allocation. `=0` restores them. One failed buffer creation turns it off for the whole context |
| `ROCKET_RK3576_CALSCAN` | 2 | The form of the calibration column scan: `0` column-outer, `1` row-major with an `int` accumulator, `2` row-major with a byte one, ~3.5x as fast as `0`. A value outside 0-2 is replaced by the default |
| `ROCKET_RK3576_CALROWS` | 0 | Caps the row count of calibration pass one, and `0` uses every row. Fewer rows underestimate the column maximum. Pass two always runs at full `M` |
| `ROCKET_RK3576_CALMAP` | unset | Names a file to which the backend writes a per-column CSV of what the calibration passes learn (`key,chunk,fwd,M,Kc,N,n,abound,est1,est2,sat,rmsA,rmsB`). It costs a sweep per forward, so keep it out of a timed run |
| `ROCKET_RK3576_PROF_CVT16` | off | With `ROCKET_MM_PROFILE`, also converts each converged call's activation to fp16 and times that pass into the `cvt16` column, the host pass an fp16 route would make. Keep it out of a timed run |

### Diagnostics

| Variable | Default | Effect |
|---|---|---|
| `ROCKET_LOG_STDERR` | off | Copies the `librocketnpu` log channel, which the backend also writes to, to stderr, so a host that silences its ggml logger still shows those lines ([Diagnostic logging](#diagnostic-logging)) |
| `ROCKET_MM_PROFILE` | off | Prints a per-phase breakdown of host and `librocketnpu` time at exit, including the dequantization bucket of a quantized GGUF's prefill. It adds noise, so leave it off for a timed run |
| `ROCKET_FA_TIMING` | off | Prints the attention handler's host split at exit: gather, NPU compute and scatter, in ms |
| `ROCKET_DEBUG` | off | Prints the fusion groups, among other debug lines |
| `ROCKET_DEBUG_GRAPH` | off | Prints one census line per scheduler split |
| `ROCKET_FA_CHECKSUM` | off | Prints a hash of the attention handler's output at exit, for comparing two runs ([Comparing two runs](#comparing-two-runs)) |
| `ROCKET_FA_DUMP_OP` | unset | With `ROCKET_FA_CHECKSUM` on, `N` writes op `N`'s four input tiles and its fp16 output to raw files |
| `ROCKET_FA_DUMP_DIR` | `/tmp` | The directory for `ROCKET_FA_DUMP_OP`'s files. Each file is created new with mode 0600, and a name that already exists, a symlink included, is refused and reported |
| `ROCKET_VERIFY` | off | Checks each offloaded op's output against a CPU recomputation. Read `max_abs`, and ignore `max_rel`. Very slow. Needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_TRACE` | off | Prints one line per offloaded matmul, with its weight name, data pointer and amax. Needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_AB` | off | Recomputes each resident F16 matmul through the per-call route and compares the two outputs. Needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_DUMP` | off | With `ROCKET_AB`, writes the inputs and both outputs of the first badly diverging op to a new mode-0600 file under `$TMPDIR`, for a standalone replay. Needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |
| `ROCKET_CPU_FORWARD` | off | Computes each claimed matmul with the CPU fp32 reference into the same output region, which separates the NPU's side effects from the output's content. Needs `-DGGML_ROCKET_DIAGNOSTICS=ON` |

#### Comparing two runs

`ROCKET_FA_CHECKSUM` hashes the attention handler's output, so two runs can be compared by one
number. Each level adds to the one below it:

- `1` prints, at exit, an FNV-1a hash over every byte the handler writes, accumulated over every
  offloaded op.
- `2` adds one hash per op with its shape beside it, which locates a hash that moves.
- `3` adds, on each op's line, the hashes of the four dense fp16 tiles the handler passes
  `librocketnpu`: `q`, `k`, `v` and the mask. An op whose output differs between two runs then
  shows whether its inputs already differed.

`ROCKET_FA_DUMP_OP` and `ROCKET_FA_DUMP_DIR` then write that op's tiles to files, for a
value-by-value comparison. Read the op count first, since a run that offloaded no attention
prints nothing.

> [!NOTE]
> Two runs of the same fp16 attention can differ, because of which NPU core ran each job. The
> RK3588's three cores differ only on output lane 0 (`n % 16 == 0`), where each rarely writes a
> result 2^-16 low on its own set of inputs. The `rocket` driver places jobs by load. Repeat
> one configuration and bound its own disagreement rate before reading a comparison across
> configurations.

> [!NOTE]
> The checksum and `ROCKET_FA_TIMING`'s summary print from an `atexit` handler. A tool built on
> llama.cpp's `common_init` drops both, because `common_log` is asynchronous and discards
> messages once its worker is paused. `llama-bench -v` prints them, and `llama-perplexity` prints
> neither.

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
threshold applies first, so a `librocketnpu` `DEBUG` trace needs both `ROCKET_DEBUG` and a host
logger that prints `DEBUG`.

A host that silences its ggml logger silences the backend too. `llama-bench` installs a no-op
ggml callback unless run with `-v`. That hides one-shot decisions such as the
`ROCKET_QUANT_RESIDENT=auto` budget, which changes the benchmarked prefill rate. For
that case the backend also sends the lines a benchmark needs through the `librocketnpu` channel:

- Its configuration lines: the budget decision, and the "raise `-ub`" quant hint
- Its residency outcome lines, `[f16-resident]` and `[moe-int8]`
- Its own `ROCKET_MM_PROFILE` breakdowns

`ROCKET_LOG_STDERR=1` copies them to stderr whatever the host's logger does.

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
in DeepSeek's lightning indexer.

### `FLASH_ATTN_EXT` accumulator

The backend claims a `FLASH_ATTN_EXT` that declares an F32 accumulator, and guards its range.
llama.cpp declares F32 on every attention op it builds, and every attention figure in these
documents was measured with it set.

The CPU backend is the declaration's reference implementation, and it runs F32 and the default
through one kernel. For F16 K and V, that kernel narrows Q to fp16 and accumulates P·V in fp16
(ggml-cpu's `ops.cpp`, the `VKQ16` path). The backend accumulates both contractions in fp32 on the
NPU, and runs the soft-cap, mask and softmax in fp32 on the host. It holds the scaled Q, the
scores, P and the output in fp16, so its rounding is the reference's.

The NPU writes the scores in fp16. A score from 65504 to 65519 is written as 65504, and one of
65520 or more as inf [HW sweep 2026-11-07]. That holds whether one product or a sum passes the
range. An overflowed logit therefore either ties with every other key at the limit or turns its
row's softmax into NaN.

`ROCKET_FA_QSCALE` moves that limit. The handler folds the op's scale into Q as it converts Q to
fp16, so the scores hold `scale*q.k`. They then overflow only past `65504/scale`, 8x further at
head_dim 64 and 16x at 256. The score is the same function either
way, `softcap*tanh(scale*q.k/softcap) + mask`, with one fp16 rounding of Q.

The backend also guards the range at both ends. A scaled Q element past 65504 is caught as Q is
gathered, and `librocketnpu`'s score pass flags a scaled logit past it (`ROCKET_E_RANGE`). Either
sends the op to a full-precision host reference. The reference reads Q as F32, computes q·k and
the softmax in double, and rounds the output once to F32. The op counts as a fallback, and the
reference is slower than both the NPU and the CPU kernel. No model has been measured reaching it
[expected].

`test-rocket-fa` scores an unscaled `q.k` of 60000-135000 exactly with the fold, and wrong by up to
1.0 without it. It scores random operands within 3.2e-5 and 3.4e-5 of a double reference either
way. On this ARM build, the CPU backend returns NaN on that case, because its fp16 dot product
accumulates in fp16.

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
- If an NPU job fails, the backend computes that op, or that slice of it, on its host reference
  and logs a warning. The graph continues. A matmul fallback computes in fp64.
  `ggml_backend_rocket_cpu_fallbacks` counts them.

### Repacked weights

ggml-cpu's weight repack moves a quantized weight into a non-host `CPU_REPACK` buffer, in an
interleaved layout for its own kernels. The backend declines an op on such a weight, so with the
repack on, prefill stays on the CPU. A host that still holds each weight's original bytes, such as
its range in the mapped GGUF, can register them. The NPU then prefills from those bytes, and
decode keeps the CPU's repacked kernels. llama.cpp and whisper.cpp do not register them.

The registration works as follows:

- The registry's `get_proc_address` returns `ggml_backend_rocket_register_weight_source` for that
  name. Its type is `void (*)(const struct ggml_tensor * t, const void * data, size_t nbytes)`,
  and `data == NULL` unregisters `t`.
- `supports_buft` accepts `CPU_REPACK`. `supports_op` claims an op on a repacked weight only while
  the weight has registered bytes whose size and tensor name match it.
- `graph_compute` points each such weight's `data` at its registered bytes for the duration of
  the split, and restores it on return. Every route reads the original layout.
- An op below the [offload threshold](TUNING.md#offload-threshold), decode included, is declined
  and runs on the CPU's repacked copy.
- The ggml scheduler places an op on a repacked weight by the weight's buffer, without asking
  `offload_op`. `--no-op-offload` therefore does not keep a registered weight's op on the CPU.

The repacked copy stays in RAM beside the mapped original. Leaving the repack on therefore costs
about the size of the repacked weights in RAM. [MEASUREMENTS.md](MEASUREMENTS.md#api) gives one
host's prefill, decode and memory figures.

> [!CAUTION]
> The host must keep the registered bytes mapped while the tensor lives, and must unregister
> the tensor before it frees it. The name and size check does not detect the same model
> reloaded into a reused tensor.
>
> The host must not compute another graph that reads the same weights while the backend's split
> runs. The backend points the shared tensor's `data` at the original bytes for that split, so a
> CPU kernel that reads it meanwhile computes wrong output. One ggml scheduler runs its splits one
> at a time.

### ggml version compatibility

The backend fills ggml's vtables, `ggml_backend_i` and the rest, with C++20 designated
initializers, so each function binds to its slot by name. Only the slots the backend implements
are named, and ggml reads every other slot as an unimplemented optional one. A slot ggml inserts
therefore builds unchanged, and a change to a slot the backend fills is a compile error.

The backend is audited against `GGML_BACKEND_API_VERSION` 2 and 3. Version 3 adds two slots to the
buffer-type vtable, which this backend does not implement, because its device uses the host's CPU
buffer type. Any other version builds with a warning, because a new slot that ggml requires is the
one change naming cannot see. `-DGGML_ROCKET_STRICT_ABI=ON` turns that warning into an error.

The README's [Troubleshooting](README.md#rocket-device-not-listed-at-startup) lists the host
versions it builds against. It builds with `-DGGML_ROCKET_STRICT_ABI=ON` against llama.cpp b11539 and whisper.cpp v1.9.5, both at version 3.
On both, the NPU's greedy completion and transcripts are byte-identical to the CPU's [HW sweep
2026-10-09].

### Op numbering check

At load, the backend checks the host's op numbering. ggml numbers its ops by position in an
enum, and inserting one shifts every op after it without moving `GGML_BACKEND_API_VERSION`. ggml
0.15 inserted `COL2IM_1D`, so `FLASH_ATTN_EXT` moved from 73 to 74 while `MUL_MAT` stayed at 29.

The load entry asks the host's `ggml_op_name` to name each op number the backend handles, and
returns `NULL` on a mismatch. The loader reports that as `ggml_backend_init returned NULL` and
unloads the `.so`. Before that line, the backend logs the op, for example `op ordinal 73 is
FLASH_ATTN_EXT in this .so and FILL in the host's ggml`. Loaded into whisper.cpp 1.9.4, a `.so`
built for whisper.cpp 1.8.6 logs that line.
