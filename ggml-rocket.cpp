// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * ggml-rocket.cpp — RK3588 NPU (rocket) ggml backend. ACCEL device modeled on
 * ggml's BLAS backend: host/CPU buffers, graph_compute offloads GGML_OP_MUL_MAT,
 * GGML_OP_MUL_MAT_ID (MoE routed experts), and GGML_OP_FLASH_ATTN_EXT (prefill
 * attention) to the NPU (fp16) via librocketnpu, CPU fallback for the rest.
 *
 * mul_mat mapping (ggml -> rocket C[M,N]=A[M,K]*B[N,K]^T):
 *   src0 = weights = B :  N = src0->ne[1], K = src0->ne[0]
 *   src1 = input   = A :  M = src1->ne[1], K = src1->ne[0]
 *   dst  = C[M,N] row-major (dst->ne[0]=N contiguous, dst->ne[1]=M)
 */
#include "ggml-rocket.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "rocket_raii.h"

// block_mxfp4 (the 17-byte { E8M0 exponent, 16 nibble-packed codes } block), for the
// native-quant MoE expert ingest, which reads MXFP4 blocks directly instead of going
// through ggml's dequantizer. DECL only -- we want the struct layout, not ggml-common's
// (large, mostly-unused) codebook tables; the 16-entry MXFP4 codebook is mirrored
// locally next to the ingest. Taking the struct from the real header means a future
// layout change breaks the BUILD rather than silently miscomputing.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <sys/mman.h>   // madvise(MADV_DONTNEED) on the GGUF source pages
#include <unistd.h>     // sysconf(_SC_PAGESIZE)

// NEON FP16 vectorization of the backend convert kernels. The A76 does the
// f32<->f16 conversion and the abs/max scan 4 lanes/instr; the scalar kernels were
// measured instruction-bound (~8.8 ns/elem pack, ~1 GB/s -- far under LPDDR5), so
// this is real headroom, not a memory wall. Gated on the FP16 *vector* arithmetic
// feature (RK3588 A76+A55 are armv8.2-a+fp16; CMake passes -march=armv8.2-a+fp16).
// Scalar fallback below keeps non-ARM dev-box builds (and any core without the
// feature) correct and compilable.
// -DROCKET_NO_NEON_CONVERT forces the scalar path (for the arch-flag-vs-SIMD A/B):
// it keeps -march=armv8.2-a+fp16 (so ggml_fp32_to_fp16 still uses the hardware vcvt,
// not the software table) but disables the NEON intrinsics. Isolates SIMD width from
// hardware-vs-software conversion. Default build leaves it undefined.
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(ROCKET_NO_NEON_CONVERT)
#include <arm_neon.h>
#define ROCKET_NEON_FP16 1
#pragma message("rocket: NEON FP16 convert kernels ENABLED")
#else
#pragma message("rocket: scalar convert kernels (NEON disabled or unavailable)")
#endif

// fp32 NEON for the Hadamard rotation (FWHT + the H_60 Paley block). The scalar
// rotation (act_rotate + wt_rotate) dominates the W8A8 CPU cost -- several times
// the NPU wait and the quant, so it is the first thing to vectorize. This is plain
// fp32 SIMD (float32x4), so it needs only __ARM_NEON, not the fp16-vector feature
// above. -DROCKET_NO_NEON_HADAMARD forces scalar (for the A/B).
#if defined(__ARM_NEON) && !defined(ROCKET_NO_NEON_HADAMARD)
#ifndef ROCKET_NEON_FP16
#include <arm_neon.h>
#endif
#define ROCKET_NEON_F32 1
#pragma message("rocket: NEON fp32 Hadamard rotation ENABLED")
#else
#pragma message("rocket: scalar Hadamard rotation")
#endif

// ===========================================================================
// Includes, weight-cache structs, and the backend context
// ===========================================================================
extern "C" {
#include "rocket_matmul.h"   // rocket_matmul_fp16_mt + pack-once (ctx/weights) API
#include "rocket_hw_profile.h" // rocket_hw_current()->max_tile (canonical resident-pack M)
#include "rocket_npu.h"      // rocket_open/close for the int8 one-shot path
#include "rocket_attn.h"     // rocket_flash_attn_fp16 (FLASH_ATTN_EXT handler)
#include "rocket_log.h"      // rocket_log_set_callback — bridge the driver's log channel into ggml's
}

// One packed-weights handle per static weight tensor, keyed by the STABLE logical
// weight name (rocket_weight_key) — NOT src0->data, which sched reuses across distinct
// weights (keying on the address would serve a stale weight for the wrong tensor; see the
// wcache member and rocket_weight_key).
struct rocket_weight_entry { rocket_weights * w; int M, K, N; size_t bytes; };

// One resident rotated-int8 weight: the per-channel-quantized (and, when
// Hadamard is on, pre-rotated) int8 weight + its per-channel scales. Built once
// per static weight and reused every forward pass — the weight prep is the
// dominant int8 cost, and weights are constant for the process lifetime.
struct rocket_int8_weight {
    std::vector<int8_t> qB;        // [N*K] int8, NPU-row-major (rotated if hadamard)
    std::vector<float>  b_scale;   // [N] per-output-channel dequant scale
    int N, K; bool hadamard; size_t bytes;
};

// One RK3576 W8A8 weight: the same per-output-channel rotated int8 weight as
// rocket_int8_weight, plus the two things that part's entry needs and the RK3588's does
// not. rocket_matmul_int8_rk3576_perc() writes int8 through a per-output-COLUMN requant,
// so the caller must hand it a per-column output scale — and the scale a column wants is
// set by that column's accumulator, which is what the call produces. `colmax` is the
// frozen estimate of that accumulator's per-column magnitude, accumulated over the first
// `cal_calls` forward passes and then read-only.
//
// WHAT THE FROZEN SCALE COSTS AND WHY THE SAFETY FACTOR IS HERE. Simulated on two real
// models' own prefill activations, freezing the colmax exactly costs Qwen2.5-1.5B 1.047x
// wikitext-2 perplexity and SmolLM2-1.7B 1.133x against fp32, where an oracle per-column
// scale costs 1.0025x / 1.008x. One global multiplier on the frozen colmax before it is
// inverted recovers 87% of that on both models (1.008x / 1.020x), and its floor is FLAT
// over 1.5-3.0 — so this is one constant, not a per-model tuning job. A larger
// calibration set buys the same thing more slowly (four times the windows recovers 71% /
// 66%), which is why `cal_done` stops at 2 and the factor carries the weight.
//
// AND THE FACTOR IS WHAT ABSORBS THE BOOTSTRAP'S OWN ERROR, which is why the default sits
// at the TOP of that flat floor rather than in the middle. `colmax` here is estimated from
// the part's int8 output, not read off an accumulator, and its 0.3% per-column error costs
// SmolLM2 0.018 of ratio at safety 2.0 and 0.0021 at 3.0 — while the exactly-frozen arm
// moves 0.0017 between the same two points. A dial that is flat for an exact scale is not
// flat for an estimated one. [host arithmetic over two models, 2026-08-11]
//
// AND THE WEIGHT IS HELD IN K-CHUNKS, one per device call, because the entry refuses
// K >= 6176 and every FFN down-projection is past it. A chunk is a whole independent W8A8
// GEMM — its own rotation, its own per-output-channel weight scale, its own frozen output
// scale — and the host sums the dequantized f32 partials. Two facts make that exact and
// nearly free rather than a compromise:
//
//   * the rotation is block-diagonal per chunk and each block is orthonormal, so
//     sum_c (A_c H_c)(B_c H_c)^T = sum_c A_c B_c^T = A B^T with no error at all;
//   * the entry's cost is 1.94 + 0.00983*K ms at M=512 N=1536 and its K-proportional part
//     is SPLIT-INVARIANT (sumabs, the cube memset, packB and packA are each O(N*K) or
//     O(M*K) per call and sum to the same total however K is cut), so the chunk COUNT
//     costs 1.94 ms apiece and Ktot is the whole of the rest.
//     [twelve cells 1536..2304, warm, first rep discarded, residuals under 0.6 ms, 2026-08-11]
//
// Summing partials that were each requantized to int8 INDEPENDENTLY does not compound:
// four chunks of a K=8960 contraction compose to 2.2385% RMS against the exact int32
// product where the single-chunk control is 2.2448% — a ratio of 0.997.
// [host arithmetic over the route's own arithmetic, 2026-08-11]
struct rocket_rk3576_chunk {
    std::vector<int8_t> qB;        // [N*Kc] int8, NPU-row-major, ALWAYS rotated
    std::vector<float>  b_scale;   // [N] per-output-channel weight dequant scale
    std::vector<float>  cbound;    // [N] 127/(128*sum_k|qB[n][k]|+1): the no-saturate scale
    // [N] sum_k|qB[n][k]| itself. cbound is derived from it, and the library's C-ramp
    // planner needs the same integer on every call — an O(N*K) pass over a weight that
    // does not change. Held here so the entry can be handed it instead
    // (rocket_matmul_int8_rk3576_perc_sa). N*8 bytes, 6.4 MB over this model's 197 sites.
    std::vector<int64_t> sumabs;
    std::vector<double> colmax;    // [N] frozen |accumulator| max, running over calibration
    std::vector<float>  scale_n;   // [N] what the entry is handed once frozen
    // The weight packed once into a DEVICE BO (rocket_rk3576_wbo_create), created at
    // this chunk's first CONVERGED call — calibration still reads qB — after which qB
    // is dropped: the cube is the same N*Kc bytes, so the swap is memory-neutral and
    // the entry's per-call cube memset + copy + cache maintenance and the weight BO's
    // per-tile allocate/free go away (rocket_matmul_int8_rk3576_perc_wbo). Freed with
    // ctx->rk76_fd wherever this chunk dies. ROCKET_RK3576_WDEV=0 keeps the qB path.
    struct rocket_rk3576_wbo * wbo = nullptr;
    int K0, Kc;                    // this chunk's offset into K, and its depth
};
struct rocket_rk3576_weight {
    std::vector<rocket_rk3576_chunk> ch;   // one entry for K < the library's refusal
    int cal_done;                  // calibration forwards recorded so far
    int N, K; size_t bytes;
    // The saturation the frozen scale did NOT anticipate, over every frozen call. A
    // nonzero rate is the calibration set failing to cover the scored activations, which
    // is the failure this route's tail arm measured; it is REPORTED rather than acted on,
    // because re-freezing mid-run makes a model's output depend on how many tokens
    // preceded it.
    uint64_t sat_elems, tot_elems;
    bool sat_warned;
};

// One resident rotated-int4 weight: per-channel-quantized (and, when Hadamard is
// on, pre-rotated) int4 weight stored one value-per-int8_t in [-7,7], + its scales.
// The int4 sibling of rocket_int8_weight. group==0 is per-channel (b_scale is [N]);
// group>0 reserves the group-wise layout (b_scale is [N*nG], nG=ceil(K/group)) for
// the W4A4 quality path -- group-wise scales are int4's main lever over per-channel.
struct rocket_int4_weight {
    std::vector<int8_t> qB;        // [N*K] int4 in [-7,7], one value per byte
    std::vector<float>  b_scale;   // [N] (group 0) or [N*nG] (group>0) dequant scale
    int N, K, group; bool hadamard; size_t bytes;
};

// One RESIDENT rotated-int8 weight: the int8 weight scattered into NPU
// tile BOs (rocket_i8_weights, held by the rocket_i8_ctx) + its per-channel
// scales. The host int8 vector is dropped after packing -- only the resident BOs
// persist. Keyed on weight NAME (the resident int8 tile plan is M-INDEPENDENT —
// canonical-tileM, so warmup-M serves prefill-M with no re-pack; Mp informational).
struct rocket_i8_resident {
    rocket_i8_weights * w;         // resident per-worker int8 tile BOs
    std::vector<float>  b_scale;   // [N] per-output-channel dequant scale
    int Mp, N, K; bool hadamard; size_t bytes;   // Mp = pack-time padded M (informational)
};

// One RESIDENT rotated-int4 weight: the group-wise-quantized (and, when Hadamard is
// on, pre-rotated) int4 weight SCATTERED into resident NPU nibble BOs (rocket_i4_weights,
// held by the rocket_i4_ctx) + its per-(channel,group) scales. The host int4 vector is
// dropped after packing -- only the resident BOs (~1/4 the fp16 footprint) persist. The
// int4 sibling of rocket_i8_resident; group-wise only (group>0, the saturation-safe Kt),
// so b_scale is [N*nG]. Keyed on weight NAME (the resident int4 tile plan is M-INDEPENDENT
// — canonical-tileM, so warmup-M serves prefill-M with no re-pack; Mp is informational only).
struct rocket_i4_resident {
    rocket_i4_weights * w;         // resident per-worker int4 nibble BOs
    std::vector<float>  b_scale;   // [N*nG] per-(output-channel,K-group) dequant scale
    int Mp, N, K, group; bool hadamard; size_t bytes;   // Mp = pack-time padded M (informational)
};

// One RESIDENT natively-quantized MoE expert weight: the expert's [N,K] GGUF-quant
// payload ingested ONCE into int8 codes + per-(output-channel, K-group) fp32 scales,
// scattered into resident NPU int8 tile BOs, and the host int8 copy dropped. Keyed on
// (weight name, expert index) -- the whole [K,N,n_expert] stack shares ONE tensor name,
// so a name-only key would alias every expert onto one entry.
//
// This is what removes the per-micro-batch host dequant that makes a quantized MoE
// offload a net loss today: the expert's codes never leave the NPU, and each forward
// pass quantizes only the activation. The layout is M-INDEPENDENT (canonical-tile
// planning), so one ingest serves every micro-batch's ragged per-expert row count.
struct rocket_moe_i8_expert {
    rocket_i8_weights * w;         // resident per-worker int8 tile BOs
    std::vector<float>  b_scale;   // [N*nG] per-(output-channel, K-group) dequant scale
    int K, N, group;
    size_t bytes;                  // resident NPU-BO bytes
    size_t charged;                // bytes + the GGUF source that must stay mapped (budget)
};

struct ggml_backend_rocket_context {
    int n_threads = 5;       // NPU worker fds. T-sweep (pp1024/ub=1024):
                             // T=3 7.15 / T=4 8.06 / T=5 8.24 / T=6 8.27 t/s — hard
                             // knee at 5 (3 cores + 2 to fill pack/read idle bubbles);
                             // 6+ is noise. ROCKET_N_THREADS overrides (1..8).

    // Pack-weights-once cache: a persistent device context (worker fds) +
    // one resident weight handle per static F16 weight tensor. Created lazily on
    // the first offloaded matmul; weights are static across forward passes, so we
    // pack each once and only re-pack A per call. Freed in the backend's free().
    rocket_ctx * dev = nullptr;
    bool dev_failed = false;   // ctx create failed once -> stop retrying, use fallback
    bool dev_resident_full = false;   // NPU 32-bit IOVA window full -> stop making new
                                      // weights resident; stream them instead.

    // Streaming context for the LLM (non-prepacked) path: persistent worker
    // fds + per-shape resident scratch/weight BOs, but B re-packed per call. Removes
    // the mt path's per-call fd open/close + BO alloc/free without holding weights
    // resident. Created lazily; ROCKET_NO_STREAM=1 forces the per-call mt path.
    rocket_stream * stream = nullptr;
    bool stream_failed = false;

    // Reusable host scratch for the streaming mul_mat path. Allocating the [N,K] dequant
    // buffer (B16) and the [Mp,N] output (C16) fresh per op made each call mmap + page-fault
    // a large buffer -- ~17-60ms for the big quantized B16 weight on the A76 [HW measured].
    // Holding them on the context (grow-only high-water mark) keeps the pages resident, so
    // a steady prefill reuses them with no re-fault. Both are fully overwritten before read
    // (B16 by the dequant's N*K rows; the per-op read region of C16 by the matmul), so
    // reuse is bit-identical to a fresh allocation. Only touched by graph_compute, which is
    // not re-entrant per backend, and each backend instance has its own context -> no
    // cross-thread sharing. A16 keeps its fresh value-init (its pad rows M..Mp-1 rely on
    // zero-init, so it is the one buffer not safe to blindly reuse) and `scales` is tiny.
    std::vector<ggml_fp16_t> scratch_B16;
    std::vector<ggml_fp16_t> scratch_C16;
    // Keyed on the STABLE logical weight name, NOT src0->data. ggml-backend-sched
    // copies our weights into a small POOLED buffer (the model's mmapped weights
    // live in a buffer our supports_buft rejects), and it REUSES those slots across
    // the graph -- so src0->data is shared by many different logical weights (the
    // trace shows ~12 copy addresses covering the whole model). Keying on the
    // address would make the cache serve a stale resident weight for the wrong tensor =>
    // gibberish, even though each matmul's math is correct. The weight name (e.g.
    // "blk.0.attn_q.weight") is unique and stable per pass, and the weights are
    // constant, so packing once per name is correct regardless of copy churn.
    std::unordered_map<std::string, rocket_weight_entry> wcache;

    // Resident-weight budget. The prepacked cache holds each packed weight in NPU
    // BOs for the life of the backend; for a 12B model that is ~20GB of NPU BOs on
    // top of the 24GB model -> real memory pressure (the rare cold-start fence-wait
    // timeout was seen under it). Cap total resident weight bytes; once exceeded,
    // weights fall back to the per-call mt path (which frees per call). NOTE: this
    // cap is NOT a correctness mechanism -- whole-model coherence comes from the
    // name-keyed wcache above, not from this budget. Default 2GB; ROCKET_CACHE_MB
    // overrides (0 = unlimited).
    size_t resident_bytes = 0;
    size_t cache_budget   = (size_t)2048 << 20;
    // Runtime OOM floor for the resident-weight admission (0 = disabled). Set to the
    // swap-safe reserve by the auto / N-MB budget modes: before committing a resident pack,
    // build_resident latches if MemAvailable has fallen below this, so residency stops
    // BEFORE an OOM even when the static byte budget was optimistic. Load-bearing for the F16
    // knob, whose resident tiles DUPLICATE the still-mapped GGUF (the init MemAvailable counts
    // the not-yet-faulted GGUF pages as free); the byte budget alone can over-commit a model
    // that does not fit ~2x. The board has no swap, so an over-commit is a hard kill.
    size_t resident_floor_bytes = 0;

    // Total source bytes reclaimed by ROCKET_PREPACK_MADVISE (the prefill-only
    // resident-weight reclaim; see rocket_prepack_madvise_on).
    size_t madvised_bytes = 0;

    // int8 W8A8 one-shot path, opt-in via ROCKET_INT8=1. A persistent
    // rocket fd drives rocket_matmul_int8 (which allocs/frees its own BOs per call).
    // Weights are re-quantized to int8 every call and it bypasses the fp16
    // streaming/prepacked machinery. -1 = unqueried; 0/1 = off/on.
    int  int8_mode     = -1;
    int  int8_hadamard = -1;   // ROCKET_INT8_HADAMARD: rotate A/B by H along K before quant
    int  int8_fd       = -1;
    bool int8_failed   = false;
    // Resident rotated-int8 weight cache (keyed on stable weight name, like wcache).
    // Budget ROCKET_INT8_CACHE_MB (default 4GB; 0 = unlimited). Over budget -> that
    // weight re-quantizes per call. Host vectors, freed with the context.
    std::unordered_map<std::string, rocket_int8_weight> int8_wcache;
    size_t int8_resident_bytes = 0;
    size_t int8_cache_budget   = (size_t)4096 << 20;

    // RK3576 W8A8 path, selected by the DETECTED PART (rocket_hw_current()) rather than
    // by an env knob, because the RK3588 int8 entry writes int32 with no output requant
    // and this one writes int8 through a per-column one — different arithmetic, different
    // cached record, different failure modes. It reuses ROCKET_INT8 as its opt-in and
    // rocket_weight_key() / rk_build_H60() / the budgeted-cache discipline as-is.
    // The rotation is NOT optional here (ROCKET_INT8_HADAMARD does not gate it): without
    // it one per-column output requant still costs 1.54x / 1.06x perplexity on the two
    // models measured, and the unrotated route is additionally CHAOTIC — a 2% per-column
    // divisor change moved one model's perplexity nine orders of magnitude.
    std::unordered_map<std::string, rocket_rk3576_weight> rk76_wcache;
    size_t rk76_resident_bytes = 0;
    int    rk76_ncal   = -1;   // ROCKET_RK3576_NCAL:   calibration forwards (default 2)
    float  rk76_calsafe = 0.0f; // ROCKET_RK3576_CALSAFE: colmax safety factor (default 2)
    float  rk76_bootmargin = 0.0f; // ROCKET_RK3576_BOOTMARGIN (default 1.5)
    int    rk76_arow   = -1;   // ROCKET_RK3576_AROW: per-ROW activation scale (default 0;
                               // 1 = every offloaded GEMM, 2 = only K-split ones)
    int    rk76_fd     = -1;
    bool   rk76_failed = false;
    // One device-weight-cache create failed (device memory, most likely): stop trying
    // for later weights too — on a board small enough to refuse one cube, growing the
    // resident set further is the wrong direction. The weights already resident stay.
    bool   rk76_wdev_off = false;

    // int4 W4A4 path, opt-in via ROCKET_INT4=1 (the int4 sibling of the int8 one-shot
    // path above). Native int4xint4->int16 NPU matmul; per-row activation + per-channel
    // weight symmetric quant to [-7,7]; the host applies the scales (rocket_dequant_int8
    // is reused verbatim -- int32 C * a_scale[m] * b_scale[n]). int4 is a RAM/bandwidth
    // play (1/4 fp16), NOT a speed play (the matmul is dispatch/readback-bound); the
    // host int4 cache is the same size as int8's (one value per byte) -- the 1/4 win is
    // on the NPU side (resident path). W4A4 is a far harder quality regime than W8A8, so
    // ROCKET_INT4_HADAMARD=1 (the outlier rotation) and group-wise scales are its levers.
    // ROCKET_INT4_GROUP=g sets the per-K-group scale granularity (0/unset = per-channel).
    int  int4_mode     = -1;
    int  int4_hadamard = -1;   // ROCKET_INT4_HADAMARD
    int  int4_group    = -1;   // ROCKET_INT4_GROUP (0 = per-channel; >0 = group size)
    int  int4_fd       = -1;
    bool int4_failed   = false;
    std::unordered_map<std::string, rocket_int4_weight> int4_wcache;
    size_t int4_resident_bytes = 0;
    size_t int4_cache_budget   = (size_t)4096 << 20;

    // RESIDENT int8 (W8A8) path, opt-in via ROCKET_INT8_RESIDENT=1 (on top of
    // ROCKET_INT8=1). The int8 weight is quantized+rotated to host int8 ONCE, then
    // SCATTERED INTO RESIDENT NPU int8 tile BOs (rocket_i8_ctx, the int8 sibling of
    // the fp16 dev ctx) and the host copy dropped -- so unlike the one-shot path
    // above it pays NO per-call weight scatter ("packB"). Fans N across worker fds.
    // Resident weights cached by stable name (re-packed on Mp/shape change); shares
    // int8_cache_budget. Over budget / IOVA full -> that weight streams on the fp16
    // path (safe: its F16 source is only madvise-dropped AFTER it goes resident).
    int  int8_resident = -1;        // ROCKET_INT8_RESIDENT (-1 = unqueried)
    rocket_i8_ctx * i8_dev = nullptr;
    bool i8_dev_failed = false;
    bool i8_resident_full = false;  // IOVA window full -> stream the rest on fp16
    std::unordered_map<std::string, rocket_i8_resident> i8_rwcache;
    size_t i8_resident_bytes = 0;

    // RESIDENT int4 (W4A4) path, opt-in via ROCKET_INT4_RESIDENT=1 (on top of
    // ROCKET_INT4=1). The int4 sibling of the resident int8 block above: the int4
    // weight is group-wise-quantized (+rotated by Hadamard) to host int4 ONCE, then
    // SCATTERED INTO RESIDENT NPU int4 nibble BOs (rocket_i4_ctx) and the host copy
    // dropped -- so the resident weight is ~1/4 the fp16 NPU-BO footprint (the int4
    // RAM win) AND it pays NO per-call weight scatter. Fans N across worker fds, like
    // the resident int8 path. Group-wise ONLY (group>0 = the saturation-safe Kt and
    // the W4 quality lever); per-channel (group==0) and over-budget / IOVA-full
    // weights fall back to the one-shot int4 path (safe). Cached by stable name;
    // shares int4_cache_budget. i4_resident_bytes is the resident NPU-BO total (NOT
    // int4_resident_bytes, which is the one-shot path's HOST int4 cache).
    int  int4_resident = -1;        // ROCKET_INT4_RESIDENT (-1 = unqueried)
    rocket_i4_ctx * i4_dev = nullptr;
    bool i4_dev_failed = false;
    bool i4_resident_full = false;  // IOVA window full -> one-shot int4 for the rest
    std::unordered_map<std::string, rocket_i4_resident> i4_rwcache;
    size_t i4_resident_bytes = 0;

    // Dedicated bf16 fp32-output datapath, selected by ROCKET_BF16=1. bf16 carries
    // fp32's 8-bit exponent, so activations need NO per-row amax-scan/scale (the
    // fp16-exponent hack rocket_pack_activations exists for): this packs A as bf16
    // UNSCALED and reads fp32 out directly -- no scale, no unscale -- staying
    // token-identical to the reference at fp32 accumulate range. The matmul runs
    // through the streaming context below (persistent fds + resident scratch +
    // multicore); weights convert to fp32 per call into a reused scratch and truncate
    // to bf16 on the scatter. With ROCKET_BF16 unset, bf16 weights instead decode to
    // fp16 and take the shared fp16 streaming route (fp16-faithful; similar speed).
    int  bf16_mode   = -1;          // ROCKET_BF16 (-1 = unqueried)
    int  bf16_fd     = -1;
    bool bf16_failed = false;
    std::vector<float> bf16_bscratch;   // reused F16->fp32 weight scratch
    // Streaming bf16 matmul context: persistent worker fds + per-shape resident scratch
    // BOs, re-packed A/B per call (rocket_matmul_bf16_stream). Created lazily on the first
    // offloaded bf16 op; turns the prior slow single-fd path into a multicore/resident one
    // (~3x on a prefill shape). Falls back to the single-fd bf16_fd if create fails.
    rocket_bf16_stream * bf16_stream = nullptr;
    bool bf16_stream_failed = false;

    // FLASH_ATTN_EXT (LLM prefill attention) path, ON by default, context-gated
    // (ROCKET_FLASH_ATTN=0 disables). The
    // handler fans the heads across n_threads worker fds (rocket_flash_attn_fp16_mt: per-head
    // QK -> mask -> softmax -> P*V on the NPU, the cores running head ranges in parallel); the
    // per-call gather/scatter of Q/K/V/mask/out are host glue. fa_fd is the availability probe
    // (and drives the nthreads==1 / host-reference paths); the mt workers open their own fds.
    // -1 = unqueried for the knob.
    int  fa_mode   = -1;
    int  fa_fd     = -1;
    bool fa_failed = false;
    // Persistent FA context: worker fds held open + per-worker score scratch kept resident
    // (grown to the largest shape seen), so a per-layer FA call pays neither the fd open/close
    // nor the per-call mmap of the 8-16 MB sc/P score matrices that bites at long context.
    // Created lazily on the first offloaded FA op; the _mt path is the fallback if create fails.
    rocket_fa_ctx * fa_ctx = nullptr;
    bool fa_ctx_failed = false;

    // MUL_MAT_ID (MoE routed-expert FFN) path, OPT-IN via ROCKET_MOE=1 (default off --
    // quant-MoE offload is dequant-bound, a net loss; see rocket_moe_on). llama.cpp
    // routes every expert gate/up/down GEMM through GGML_OP_MUL_MAT_ID; the handler
    // buckets (slot,token) rows by expert id, runs each expert's [M_e,K]x[N,K]^T on the
    // NPU (fanned across worker fds via rocket_matmul_fp16_mt, weight dequant->fp16 like
    // the dense path), and scatters the rows back. All buffers grow-only (high-water
    // mark), fully overwritten before read, so reuse is bit-identical to a fresh
    // allocation. Only touched by graph_compute (serial per backend). moe_mode = -1 unqueried.
    int moe_mode = -1;
    std::vector<float>       moe_Af32;    // gathered [M_e,K] f32 activations (per expert)
    std::vector<ggml_fp16_t> moe_A16;     // packed [Mp_e,K] fp16 activations
    std::vector<ggml_fp16_t> moe_B16;     // dequantized [N,K] fp16 expert weight
    std::vector<ggml_fp16_t> moe_C16;     // [Mp_e,N] fp16 expert output
    std::vector<float>       moe_scales;  // per-row activation scale [M_e]
    std::vector<float>       moe_Cf32;    // [M_e,N] f32 CPU-fallback output
    // Row buckets: for each expert, the (slot,token) pairs routed to it. Flattened
    // as moe_rows[expert_off[e] .. expert_off[e+1]) so one pass over ids fills them.
    std::vector<int32_t> moe_row_slot;    // id (0..n_expert_used) per bucketed row
    std::vector<int32_t> moe_row_tok;     // token (0..n_tokens)  per bucketed row
    std::vector<int32_t> moe_expert_off;  // [n_expert+1] prefix offsets into the buckets

    // NATIVE-QUANT MoE experts (the resident int8 group-wise path). A GGUF-quantized
    // expert weight (gpt-oss MXFP4, DeepSeek Q4_K/Q8_0) is ingested ONCE into int8 codes
    // + per-(channel, K-group) scales and left resident in NPU BOs, so the per-micro-batch
    // host dequant->fp16 and weight scatter both disappear -- the only source of speed on
    // this path (the int8 GEMM itself moves MORE bytes than the fp16 one). Shares the
    // resident int8 device context (i8_dev) and its worker fds with the dense W8A8 path.
    //
    // Keyed on (weight name, expert index): rocket_weight_key returns ONE name for the
    // whole [K,N,n_expert] stack, so a name-only key would alias all n_expert experts.
    // Admission-only, no eviction -- prefill touches every expert every micro-batch, so
    // there is no hotness to exploit; what does not fit streams on the fp16 dequant path.
    int moe_native = -1;                  // ROCKET_MOE_NATIVE (-1 = unqueried)
    std::unordered_map<std::string, rocket_moe_i8_expert> moe_i8_cache;
    size_t moe_i8_resident_bytes = 0;     // the resident NPU-BO bytes (what is on the NPU)
    size_t moe_charged_bytes     = 0;     // what the budget governs: BOs + the GGUF source
                                          // that has to stay mapped alongside them
    size_t moe_cache_budget      = 0;     // bytes; 0 = unlimited. Set in _init (default auto)
    bool   moe_i8_full           = false; // IOVA window full / budget hit -> stream the rest
    long   moe_n_resident        = 0;     // experts ingested (for the resident/streaming split)
    // What the residency COSTS: every resident expert is decoded from its GGUF blocks,
    // requantized to int8, and scattered into NPU BOs -- once. On a real MoE that is
    // thousands of experts and tens of GB, i.e. minutes, and because the ingest is lazy it
    // all lands inside the FIRST prefill. Timed unconditionally rather than behind
    // ROCKET_MM_PROFILE: a multi-minute startup stall is a cost the user pays and must be
    // told about, not a number only a developer wants. Two timers because the two halves
    // have different fixes -- the decode is CPU work that threads, the pack is a BO scatter
    // that does not.
    double moe_ingest_ms         = 0;     // GGUF quant blocks -> int8 codes + group scales
    double moe_pack_ms           = 0;     // int8 codes -> the tiled scatter into resident BOs
    // The DISTINCT experts that fell back to fp16 streaming. A per-call counter would count
    // the same expert once per micro-batch and read as a much worse split than it is; the
    // number that matters is how many of the model's experts never went resident.
    std::unordered_set<std::string> moe_streamed_keys;
    // Per-expert M bucketing (see rocket_moe_bucket_m): the router hands every expert a
    // different row count, and the driver's resident scratch is cached per (M,K,N,group) in a
    // fixed-size table that does NOT evict -- so the raw M_e values would exhaust it mid-prefill.
    // M_e is rounded up onto a fixed 2-rungs-per-octave ladder whose rung count is bounded by
    // construction; moe_slots then only *observes* how many slots that costs, so a shape mix we
    // did not anticipate can warn instead of silently degrading to the fp16 route.
    int moe_m_granule = 0;                // 0 = unqueried; the ladder's floor (ROCKET_MOE_M_BUCKET)
    std::unordered_set<uint64_t> moe_slots;   // diagnostic: distinct (M,K,N) slots asked for
    // Host scratch for the native-quant route (grow-only; fully written before read).
    std::vector<int8_t> moe_qA;           // [Mb,K] int8 activations
    std::vector<float>  moe_a_scale;      // [Mb*nG] per-(row, K-group) activation scale
    std::vector<float>  moe_Cgw;          // [Mb,N] fp32 group-wise matmul output
    // Ingest scratch, held on the context rather than allocated per expert. A fresh
    // [N*K] vector per expert would value-initialize (memset) a buffer we then overwrite
    // in full, and malloc hands back a fresh mmap at that size -- so a 2300-expert model
    // would memset and page-fault ~19 GB purely to throw it away. The ingest writes every
    // byte of both buffers before anything reads them, so reuse is identical to a fresh
    // allocation.
    std::vector<int8_t> moe_ingest_codes;  // [N*K] int8 codes for the expert being ingested
    std::vector<float>  moe_ingest_scales; // [N*nG] its per-(channel, K-group) scales
};

// ===========================================================================
// Configuration, environment knobs, and profiling counters
// ===========================================================================

// ---------------------------------------------------------------------------
// mul_mat
// ---------------------------------------------------------------------------

// Round M (rows) up to a multiple of 4. The driver's tiled matmul requires the
// total M be a multiple of 4 (rocket_matmul_plan rejects otherwise), but LLM
// prefill ubatches are arbitrary token counts. We pad A with zero rows up to
// Mp; the extra output rows are computed and discarded. Matmul rows are
// independent, so the first M rows are bit-identical to an unpadded run.
static inline int rocket_pad_m(int M) { return (M + 3) & ~3; }

// Minimum rows (M) worth offloading. Below it the offload LOSES to the CPU: the
// per-call fixed cost (dispatch, plus the full packB -- a weight only goes resident
// at M >= max_tile, so below that it is re-packed on EVERY call) outweighs the NPU's
// per-row throughput advantage.
//
// The crossover is where the NPU's cost meets the CPU's, per op:
//     NPU = dispatch + p*(K*N) + w*(M*K*N)      CPU = g*(M*K*N)
//     =>  M* = ( p + dispatch/(K*N) ) / (g - w)
// The K*N cancels out of the leading term, so M* is nearly MODEL-INDEPENDENT -- the
// packB you pay scales with the same K*N as the compute you gain. What remains is the
// dispatch term, which does NOT scale with K*N and so weighs more when the weights are
// small: a small model crosses LATER. Measured [HW sweep, 600 MHz, F16, warm, t/s
// NPU/CPU], and the drift is exactly that predicted shape:
//
//              pp16   pp32   pp48   pp64   pp96   pp128     M*
//     0.8B     0.36   0.58   0.72   0.83   1.04   1.15      ~86
//     3B       0.35   0.60   0.87   1.03   1.38   1.60      ~64
//     8B       0.35   0.65   0.93   1.24    -     1.89      ~55
//
// Default 128: at or above every measured crossover, so NO model regresses below the
// CPU, and larger models cross earlier still (so 12B+ is covered a fortiori). The cost
// is forgoing the 64..127 wins on the bigger models -- a forgone win, never a loss.
// Quantized weights and MoE experts have their own (higher) floors, so this governs the
// F16/BF16 dense path alone.
//
// A floor of 4 (the old default) is a TRAP for hosts whose *decode* is batched:
// whisper.cpp defaults to beam search with beam_size=5 and batches all active decoders
// into ONE decode call, so every decode step arrives as M=5. It cleared a floor of 4,
// landed on the NPU, and ran 2.3x SLOWER than the CPU -- making end-to-end whisper a
// 1.40x NET LOSS. M=1 GEMV decode (llama.cpp) was never the only small-M case.
//
// Do NOT set this to max_tile (256) on the theory that the residency pivot is the right
// floor: llama streams its weights below 256 too and still wins big there (pp128 =
// 1.89x), so 256 costs pp128 -46%. Tunable via ROCKET_MIN_M.
static int rocket_min_m(void) {
    static int m = 0;
    if (m == 0) {
        // getenv returns a non-NULL "" for a set-but-empty var (ROCKET_MIN_M=, or =$X with X
        // unset -- the common benchmark-wrapper shape); atoi("")==0 would then clamp to the
        // floor 4, silently re-arming the beam/small-M offload trap the 128 default exists to
        // prevent. Treat set-empty as UNSET here and at every routing knob below.
        const char * e = getenv("ROCKET_MIN_M");
        m = (e && *e) ? atoi(e) : 128;
        if (m < 4) m = 4;
    }
    return m;
}

// Quantized (GGUF) weights always take the per-call dequant->fp16 path, whose
// dequant is a fixed per-microbatch cost. Below a few hundred rows it doesn't
// amortize and the offload loses to the CPU -- measured crossover ~360 rows on
// 9B/27B Q4_K (NPU pp384 6.6 vs CPU 6.2 t/s; pp256 4.8 < 6.2 t/s). So quantized
// MUL_MATs use a higher floor than the F16 rocket_min_m(): default 512 = the
// default micro-batch, so full ubatches still offload (clearly winning, >=1.33x)
// while shorter prefills stay on the CPU. Tunable via ROCKET_MIN_M_QUANT; never
// below rocket_min_m(). (Native int4/int8 modes re-quantize F16 weights, not
// ggml_is_quantized ones, so this gate covers the dequant path alone.)
static int rocket_min_m_quant(void) {
    static int m = 0;
    if (m == 0) {
        const char * e = getenv("ROCKET_MIN_M_QUANT");
        m = (e && *e) ? atoi(e) : 512;
        const int base = rocket_min_m();
        if (m < base) m = base;
    }
    return m;
}

// MUL_MAT_ID (MoE routed-expert FFN) offload, OPT-IN via ROCKET_MOE=1 (default OFF).
// The handler is correct and bit-faithful (test-rocket-moe cos=1.000000), but for the
// QUANTIZED experts that every board-fitting MoE ships (gpt-oss MXFP4, DeepSeek Q4_K)
// it is a net LOSS: each expert weight is dequantized to fp16 on the host EVERY
// micro-batch (streaming), and MoE has ~n_expert times more distinct weights per layer
// than a dense model, each amortized over only M_e ~= n_tokens*n_expert_used/n_expert
// rows -- so the per-expert streaming dequant + dispatch dominates, where the CPU's
// fused quantized kernel pays no dequant. Measured gpt-oss-20b (MXFP4) pp2048: NPU 5.33
// vs CPU 12.56 t/s = 0.42x [HW sweep, 600 MHz]. So decode/dense-model behaviour is
// unchanged by default; ROCKET_MOE=1 opts a MoE model's experts onto the NPU (faithful,
// currently slower for quant). A win needs native-quant experts (no host dequant) or a
// resident-expert cache -- deferred. Same root cause as the quant-fused-group lever:
// quant prefill is dequant-bound.
static bool rocket_moe_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_MOE"); v = (e && atoi(e) > 0) ? 1 : 0; }
    return v > 0;
}

// Minimum n_tokens (the micro-batch's token count) for a MUL_MAT_ID op to offload when
// ROCKET_MOE=1. A MoE op's per-expert row count is M_e ~= n_tokens * n_expert_used /
// n_expert, so n_tokens is the direct handle on the per-expert GEMM size; the default
// 512 keeps decode and tiny ubatches on the CPU. Tunable via ROCKET_MOE_MIN_TOKENS;
// never below rocket_min_m().
static int rocket_moe_min_tokens(void) {
    static int m = 0;
    if (m == 0) {
        const char * e = getenv("ROCKET_MOE_MIN_TOKENS");
        m = (e && *e) ? atoi(e) : 512;
        const int base = rocket_min_m();
        if (m < base) m = base;
    }
    return m;
}

// NATIVE-QUANT MoE experts: within ROCKET_MOE=1, route a GGUF-QUANTIZED expert weight
// through the resident int8 group-wise path (ingest once -> int8 codes resident on the
// NPU) instead of dequantizing it to fp16 on the host every micro-batch. ON by default
// when ROCKET_MOE is on -- it is the reason the MoE offload can win at all on the models
// that ship quantized. ROCKET_MOE_NATIVE=0 forces the fp16 dequant route, which is the
// A/B baseline for the native path (and the only route for an F16 expert, which has no
// dequant to delete).
static bool rocket_moe_native_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_MOE_NATIVE"); v = (e && *e) ? (atoi(e) > 0) : 1; }
    return v > 0;
}

// ROCKET_MOE_GROUP=g pins the K-group the native-quant path quantizes on (0/unset = auto,
// see rocket_moe_pick_group). The group is the accuracy/speed dial: readback scales as
// K/group and these integer paths are readback-bound, so a fine group is more faithful
// and proportionally slower. Exists for the A/B; auto picks the readback floor.
static int rocket_moe_group_env(void) {
    static int g = -1;
    if (g < 0) { const char * e = getenv("ROCKET_MOE_GROUP"); g = e ? atoi(e) : 0; if (g < 0) g = 0; }
    return g;
}

// ROCKET_MOE_M_BUCKET: the FLOOR of the bucket ladder the ragged per-expert row count is rounded
// up onto (default 64 rows). See rocket_moe_bucket_m for why bucketing is mandatory rather than a
// tuning knob, and for the ladder itself. Rounded up to a power of two: the ladder puts its
// intermediate rung at 1.5x each power, so a power-of-two floor is what keeps every rung on the
// M%4 hardware contract.
static int rocket_moe_m_bucket_env(void) {
    const char * e = getenv("ROCKET_MOE_M_BUCKET");
    int g = (e && *e) ? atoi(e) : 64;
    if (g < 4) g = 4;
    int p = 4;
    while (p < g && p < (1 << 20)) p <<= 1;
    return p;
}

// ROCKET_QUANT_RESIDENT=1: hold a quantized GGUF weight's DEQUANTIZED fp16 form
// RESIDENT in NPU BOs (dequant once, then the F16 prepacked path) instead of
// re-dequantizing AND re-packing it every micro-batch (the streaming tax).
// Lifts quant prefill from ~0.64x toward F16 parity and fixes the -ub 512 /
// short-follow-up cases the -ub 2048 amortization can't, at the cost of the full
// fp16 resident footprint (negating the quant RAM saving) -- so it is opt-in, for
// the "quantized-for-download-size, RAM-to-spare" case. Cached like rocket_min_m so
// the no-ctx device supports_op path can read it without a context handle.
// On when ROCKET_QUANT_RESIDENT is a positive MB budget, "1" (blanket), or "auto"
// (budget sized from free RAM at init); off for "0" / unset. The budget VALUE is applied
// to ctx->cache_budget in ggml_backend_rocket_init (this no-ctx gate only reports on/off,
// so the device supports_op path can read it without a context handle).
// ROCKET_INT8, read without a context handle. The RK3576 branch of supports_op needs it:
// on that part the W8A8 route is the only matmul route, so whether the op can be claimed
// at all depends on whether int8 mode is on. ctx->int8_mode is the same knob and stays
// the one the dispatch reads.
static bool rocket_int8_mode_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_INT8"); v = (e && atoi(e) > 0) ? 1 : 0; }
    return v > 0;
}

static bool rocket_quant_resident_on(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("ROCKET_QUANT_RESIDENT");
        v = (e && (strcmp(e, "auto") == 0 || atoi(e) > 0)) ? 1 : 0;
    }
    return v > 0;
}

// ROCKET_F16_RESIDENT: the F16 sibling of ROCKET_QUANT_RESIDENT. Holds a static F16
// weight's scattered tiles RESIDENT in NPU BOs for ALL K -- not just the K<=2048 whisper
// default -- so the big LLM weights (attn/FFN, K in {3072,8192,...}) pay their weight
// scatter (packB) ONCE and reuse it across every later micro-batch / prefill instead of
// re-scattering per call (the streaming tax). Measured net win over the fused-streaming
// default [HW sweep, Llama-3.2-3B-F16, pp2048 -ub512, 600 MHz: resident 41.9 vs default 39.9
// t/s = +5%; the pure packB delta with fusion held off is +10.6%]. QKV/gate-up fusion is KEPT
// (see rocket_fuse_on): a fusable group routes through ggml_backend_rocket_mul_mat_group_resident
// (one resident combined-N weight), stacking the packB-once win with fusion's shared packA +
// single submit -- both levers, not a trade.
//
// Decode-safe by design: it does NOT madvise the F16 source (that stays behind the separate,
// prefill-only ROCKET_PREPACK_MADVISE), so the GGUF stays mapped for CPU decode. The resident
// tiles DUPLICATE the still-mapped GGUF, so this is the "F16-for-quality, RAM-to-spare" case
// (roughly, models that fit ~2x in RAM); the auto budget below caps residency and a runtime
// MemAvailable floor (build_resident) latches before an OOM, streaming the overflow.
//
// On for "auto" (budget sized from free RAM at init), a positive MB budget, or "1"; off for
// "0"/unset. Cached like rocket_quant_resident_on so the no-ctx supports_op path can read it.
static bool rocket_f16_resident_on(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("ROCKET_F16_RESIDENT");
        v = (e && (strcmp(e, "auto") == 0 || atoi(e) > 0)) ? 1 : 0;
    }
    return v > 0;
}

// FLASH_ATTN_EXT offload is ON by default, context-gated; ROCKET_FLASH_ATTN=0 disables it.
// It is numerically faithful (FA-NPU perplexity == FA-CPU, primitive cos=1.000000). The
// handler is multicored (heads fanned across worker fds), host-softmax, and submit-chained
// through a persistent batched context (rocket-userspace ROCKET_FA_CHAIN), so with the n_kv
// gate below it is parity-or-win across prefill depths: [HW sweep, F16, 600 MHz, -r3]
// 1.50x @8K, 1.25x @16K, with short prompts (n_kv < gate) and single-token decode kept on
// the CPU. Gating: the per-op n_kv gate (ROCKET_FLASH_ATTN_MIN_KV, default 1024) keeps short
// context on the CPU, and ROCKET_FLASH_ATTN_MIN_T (default 16) keeps decode on the CPU.
static bool rocket_flash_attn_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_FLASH_ATTN"); v = (e && *e) ? atoi(e) : 1; }
    return v > 0;
}
static int rocket_flash_attn_min_t(void) {
    static int t = 0;
    if (t == 0) { const char * e = getenv("ROCKET_FLASH_ATTN_MIN_T"); t = (e && *e) ? atoi(e) : 16; if (t < 1) t = 1; }
    return t;
}
// ROCKET_FLASH_ATTN_MIN_KV (default 1024): an FA op offloads only when n_kv >= this. With the
// per-worker QK/AV submit chaining default-on (rocket-userspace ROCKET_FA_CHAIN, persistent
// batched context), the FA-NPU-vs-CPU prefill crossover is ~2K: chained
// FA-NPU is parity at <=1K (1.00x @512, 0.97x @1024) and a growing win above (1.02x @2048,
// 1.07x @4096, 1.45x @8K). The gate sits at 1024 — the Gemma-4 sliding-window length — so the
// 40 windowed LOCAL layers (n_kv capped at 1024, where chaining helps most) offload at long
// context (the multi-rep win is 1.50x @8K / 1.25x @16K WITH them, vs 1.15x global-layers-only),
// while shorter prompts and decode stay on the CPU. [HW sweep 2026-06-28,
// F16, 600 MHz, -r3] Gate on n_kv (KV/context length), NOT n_tokens: under llama.cpp's
// 512-token ubatching every FLASH_ATTN_EXT op has n_tokens~512 regardless of total context,
// so only n_kv tells a short prompt from a deep ubatch in a long one; each ubatch
// independently picks the faster backend (both are correct).
static int rocket_flash_attn_min_kv(void) {
    static int kv = 0;
    if (kv == 0) { const char * e = getenv("ROCKET_FLASH_ATTN_MIN_KV"); kv = (e && *e) ? atoi(e) : 1024; if (kv < 1) kv = 1; }
    return kv;
}
// ROCKET_FLASH_ATTN_NO_CTX=1 forces the per-call mt path (fresh worker fds + per-call score
// scratch every call) instead of the persistent FA context. The persistent context is the
// default (it removes the per-layer fd open/close + the 8-16 MB sc/P mmap churn that bites at
// long context); this knob exists to A/B the two paths under identical thermal conditions, the
// FA analogue of ROCKET_NO_STREAM for the matmul path.
static bool rocket_flash_attn_no_ctx(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_FLASH_ATTN_NO_CTX"); v = e ? atoi(e) : 0; }
    return v > 0;
}

// The backend-side host kernels below
// (rocket_pack_activations f32->fp16, rocket_unpack_output(_seg) fp16->f32*scale, and the
// streaming weight_dequant bf16/quant->fp16) are NOT covered by the driver's
// ROCKET_MM_PROFILE (which times only the driver: packA/packB/wait/read). This sizes those
// host costs separately. The weight_dequant bucket matters most: the per-microbatch
// bf16/quant->fp16 decode is the dominant host cost of a quantized-GGUF prefill, so leaving
// it unattributed would understate host-bound work and flatter the driver's "wait" share --
// the exact host-vs-dispatch axis these probes exist to decide. Gated by the SAME
// ROCKET_MM_PROFILE knob so the breakdown prints alongside the driver's profile at exit.
// Single-threaded: these run on the backend's graph_compute dispatch thread (the
// driver's worker threads are downstream), so no mutex is needed.
static int rocket_convprof_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_MM_PROFILE") ? 1 : 0;
    return v;
}
// ROCKET_DEBUG / ROCKET_DEBUG_GRAPH gate per-split/per-matmul stderr in the hot
// graph_compute path; cache them once (presence-check, any value truthy) so the loop
// doesn't re-scan environ per node -- the same lazy-cache idiom as the knobs above,
// and always compiled (unlike the ROCKET_DIAGNOSTICS helpers). First-call race benign.
static bool rocket_debug_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_DEBUG") ? 1 : 0;
    return v > 0;
}
static bool rocket_debug_graph_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_DEBUG_GRAPH") ? 1 : 0;
    return v > 0;
}
static double rocket_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}
static struct { double pack_ms, unpack_ms, dequant_ms, pack_elems, unpack_elems, dequant_elems;
                long pack_calls, unpack_calls, dequant_calls; } g_convprof;
static int g_convprof_armed = 0;
// The profiler dumps below (this one, i8prof, moeprof, moecos) go out on the rocket_log
// channel, NOT GGML_LOG_*. They are measurement lines, and the tool they have to survive is
// llama-bench -- which installs a no-op ggml log callback and swallows everything sent
// through ggml. ROCKET_LOG_STDERR=1 tees the rocket channel to stderr past that. A profiler
// whose output is silenced by the profiling harness is worse than no profiler: it reads as
// "nothing to report".
static void rocket_convprof_dump(void) {
    ROCKET_LOGI(
        "ROCKET convert total(ms): pack_act=%.0f (%ld calls, %.0fM elems) "
        "weight_dequant=%.0f (%ld calls, %.0fM elems) "
        "unpack_out=%.0f (%ld calls, %.0fM elems)\n",
        g_convprof.pack_ms,    g_convprof.pack_calls,    g_convprof.pack_elems    / 1e6,
        g_convprof.dequant_ms, g_convprof.dequant_calls, g_convprof.dequant_elems / 1e6,
        g_convprof.unpack_ms,  g_convprof.unpack_calls,  g_convprof.unpack_elems  / 1e6);
}
static void rocket_convprof_add(double ms, double elems, bool is_unpack) {
    if (!g_convprof_armed) { atexit(rocket_convprof_dump); g_convprof_armed = 1; }
    if (is_unpack) { g_convprof.unpack_ms += ms; g_convprof.unpack_elems += elems; g_convprof.unpack_calls++; }
    else           { g_convprof.pack_ms   += ms; g_convprof.pack_elems   += elems; g_convprof.pack_calls++;   }
}
// The per-microbatch streaming weight dequant (bf16 / Q8_0 / Q4_K / MXFP4 / ... -> fp16) runs
// in the backend BEFORE the driver is entered, so neither ROCKET_MM_PROFILE nor the pack/unpack
// brackets above see it -- yet it is the dominant host cost of a quantized-GGUF prefill. Its own
// bucket keeps the breakdown honest (the MoE handler already times its equivalent in g_moeprof).
static void rocket_convprof_add_dequant(double ms, double elems) {
    if (!g_convprof_armed) { atexit(rocket_convprof_dump); g_convprof_armed = 1; }
    g_convprof.dequant_ms += ms; g_convprof.dequant_elems += elems; g_convprof.dequant_calls++;
}

// FLASH_ATTN_EXT handler timing probe (ROCKET_FA_TIMING=1, off by default; separate from
// ROCKET_MM_PROFILE so both can run together). The handler's outer strided-ggml -> dense-fp16
// gather (Q F32->F16, the strided K/V cache views, the mask) and the F32 scatter are single-
// threaded host loops the driver's ROCKET_MM_PROFILE does NOT see; this isolates them from the
// on-NPU compute so "is the gather worth threading?" is a measurement, not a guess. The gather
// is O(n_kv) per op, so its share is expected to grow with context. Near-zero when off (one
// cached env check + a branch; no clock reads). Single-threaded backend dispatch thread -> no mutex.
static bool rocket_fatiming_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_FA_TIMING") ? 1 : 0;
    return v > 0;
}
static struct { double gather_ms, compute_ms, scatter_ms; long calls; int min_kv, max_kv; } g_fatiming
    = { 0, 0, 0, 0, 1 << 30, 0 };
static int g_fatiming_armed = 0;
static void rocket_fatiming_dump(void) {
    double tot = g_fatiming.gather_ms + g_fatiming.compute_ms + g_fatiming.scatter_ms;
    if (tot <= 0) return;
    GGML_LOG_INFO(
        "ROCKET FA total(ms): gather=%.0f (%.0f%%) compute=%.0f (%.0f%%) scatter=%.0f (%.0f%%) "
        "over %ld ops, n_kv [%d..%d]\n",
        g_fatiming.gather_ms,  100.0 * g_fatiming.gather_ms  / tot,
        g_fatiming.compute_ms, 100.0 * g_fatiming.compute_ms / tot,
        g_fatiming.scatter_ms, 100.0 * g_fatiming.scatter_ms / tot,
        g_fatiming.calls, g_fatiming.min_kv, g_fatiming.max_kv);
}
static void rocket_fatiming_add(double gather_ms, double compute_ms, double scatter_ms, int n_kv) {
    if (!g_fatiming_armed) { atexit(rocket_fatiming_dump); g_fatiming_armed = 1; }
    g_fatiming.gather_ms += gather_ms; g_fatiming.compute_ms += compute_ms;
    g_fatiming.scatter_ms += scatter_ms; g_fatiming.calls++;
    if (n_kv < g_fatiming.min_kv) g_fatiming.min_kv = n_kv;
    if (n_kv > g_fatiming.max_kv) g_fatiming.max_kv = n_kv;
}

// ===========================================================================
// Activation packing, fp16 scaling, and output unpacking
// ===========================================================================
// "Sandwich Domain Shift": the NPU is fp16-only, but Gemma activations routinely
// exceed fp16's ~65504 (embedding scale ~sqrt(d), residual growth, outliers) ->
// a naive F32->fp16 cast yields Inf -> NaN -> garbage tokens. We convert A to
// fp16 DIVIDED by a scale so the scaled A lands in [-1,1] (also maximizing
// mantissa use), run the matmul, then multiply the fp16 result back by the same
// scale. Exact up to fp16 rounding: C = A*B = s*((A/s)*B).
//
// ROW-WISE scaling: each output row m is an independent dot product of input
// row A[m,:] with the weights, so we use ONE scale per row (max-abs of that row)
// instead of one global max-abs over all M*K. A single outlier token therefore no
// longer crushes the fp16 mantissa of every other token in the ubatch -- which
// matters more as prefill length grows. scales[] must hold M floats; the caller
// leaves any pad rows beyond M zero (value-initialized). Weights are left as is
// (bounded; the outlier problem is activations, per the quant findings).
static void rocket_pack_activations(const float * src, ggml_fp16_t * dst,
                                    int64_t M, int64_t K, float * scales) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t m = 0; m < M; m++) {
        const float * row = src + m * K;
        int64_t k = 0;
        float amax = 0.0f;
#ifdef ROCKET_NEON_FP16
        // Pass 1: row max-abs, 4 lanes/instr.
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (; k + 4 <= K; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(row + k)));
        amax = vmaxvq_f32(vmax);
#endif
        for (; k < K; k++) { const float v = fabsf(row[k]); if (v > amax) amax = v; }
        const float s   = (amax > 0.0f) ? amax : 1.0f;
        const float inv = 1.0f / s;
        ggml_fp16_t * d = dst + m * K;
        k = 0;
#ifdef ROCKET_NEON_FP16
        // Pass 2: scale by 1/s and convert f32->f16, 4 lanes/instr. ggml_fp16_t is a
        // 16-bit IEEE half; __fp16 store writes the same bit pattern (RNE, matching
        // ggml_fp32_to_fp16's hardware path).
        const float32x4_t vinv = vdupq_n_f32(inv);
        __fp16 * dh = (__fp16 *)d;
        for (; k + 4 <= K; k += 4)
            vst1_f16(dh + k, vcvt_f16_f32(vmulq_f32(vld1q_f32(row + k), vinv)));
#endif
        for (; k < K; k++) d[k] = ggml_fp32_to_fp16(row[k] * inv);
        scales[m] = s;
    }
    if (prof) rocket_convprof_add(rocket_now_ms() - t0, (double)M * K, false);
}

// Inverse of the per-row scale: fp16 result row m -> F32, multiplied back by
// scales[m]. src/dst each hold M*N (N contiguous).
static void rocket_unpack_output(const ggml_fp16_t * src, float * dst,
                                 int64_t M, int64_t N, const float * scales) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t m = 0; m < M; m++) {
        const float s = scales[m];
        const ggml_fp16_t * srow = src + m * N;
        float * drow = dst + m * N;
        int64_t n = 0;
#ifdef ROCKET_NEON_FP16
        const float32x4_t vs = vdupq_n_f32(s);
        const __fp16 * sh = (const __fp16 *)srow;
        for (; n + 4 <= N; n += 4)
            vst1q_f32(drow + n, vmulq_f32(vcvt_f32_f16(vld1_f16(sh + n)), vs));
#endif
        for (; n < N; n++) drow[n] = ggml_fp16_to_fp32(srow[n]) * s;
    }
    if (prof) rocket_convprof_add(rocket_now_ms() - t0, (double)M * N, true);
}

// Like rocket_unpack_output, but the source is a column slice [col0, col0+Ni) of a
// wider [M, src_stride] fp16 buffer (a fused matmul's combined-N output); the
// destination is contiguous [M, Ni]. The per-row scale is shared across the fused
// group (same activation A), so scales[] indexes the same rows.
static void rocket_unpack_output_seg(const ggml_fp16_t * src, int64_t src_stride,
                                     int64_t col0, float * dst,
                                     int64_t M, int64_t Ni, const float * scales) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t m = 0; m < M; m++) {
        const float s = scales[m];
        const ggml_fp16_t * srow = src + m * src_stride + col0;
        float * drow = dst + m * Ni;
        int64_t n = 0;
#ifdef ROCKET_NEON_FP16
        const float32x4_t vs = vdupq_n_f32(s);
        const __fp16 * sh = (const __fp16 *)srow;
        for (; n + 4 <= Ni; n += 4)
            vst1q_f32(drow + n, vmulq_f32(vcvt_f32_f16(vld1_f16(sh + n)), vs));
#endif
        for (; n < Ni; n++) drow[n] = ggml_fp16_to_fp32(srow[n]) * s;
    }
    if (prof) rocket_convprof_add(rocket_now_ms() - t0, (double)M * Ni, true);
}

// int8 convert profiler: split the int8 path's CPU-side cost into
// Hadamard ROTATE vs QUANT vs DEQUANT, for activations (every call) and weights
// (once, cached). The fp16 g_convprof above does NOT cover the int8 _had kernels,
// so the suspected-dominant per-call rotate+requant was invisible. Same
// ROCKET_MM_PROFILE knob; single-threaded (backend dispatch thread); own exit line.
static struct { double act_rot, act_q, wt_rot, wt_q, dq;
                long act_calls, wt_calls, dq_calls; } g_i8prof;
static int g_i8prof_armed = 0;
static void rocket_i8prof_dump(void) {
    ROCKET_LOGI(
        "ROCKET int8 convert total(ms): act_rotate=%.0f act_quant=%.0f "
        "wt_rotate=%.0f wt_quant=%.0f dequant=%.0f  (act %ld, wt %ld, dq %ld calls)\n",
        g_i8prof.act_rot, g_i8prof.act_q, g_i8prof.wt_rot, g_i8prof.wt_q, g_i8prof.dq,
        g_i8prof.act_calls, g_i8prof.wt_calls, g_i8prof.dq_calls);
}
static inline void rocket_i8prof_arm(void) {
    if (!g_i8prof_armed) { atexit(rocket_i8prof_dump); g_i8prof_armed = 1; }
}

// MoE handler profiler: the PER-MICRO-BATCH decomposition of the native-quant expert route
// (gather -> activation quant -> resident int8 GEMM -> scatter), plus whatever still falls
// through to the fp16 streaming route. This is the line that says where a MoE prefill's time
// actually goes, and it is what settles whether the route is GEMM-bound (nothing left to win
// on the host) or host-bound (the activation quant is the remaining lever).
//
// Disjoint from g_i8prof above, not overlapping it: the grouped activation quant is reached
// ONLY from this handler, so it is counted here and nowhere else. g_i8prof stays the DENSE
// W8A8 path's line.
//
// rows_used vs rows_computed prices the M-bucket padding directly (see rocket_moe_bucket_m):
// the router hands every expert a ragged row count, we round it up to a granule, and the pad
// rows are computed and thrown away. The ratio is the waste, measured rather than assumed.
//
// native_gemms vs fp16_gemms is the residency split as the COST is actually paid -- per
// (op, expert-with-rows), per micro-batch. The teardown line counts DISTINCT experts, which
// answers a different question (how much of the model went resident) and cannot be converted
// into this one without assuming every expert is hit equally often. This counter needs no
// such assumption.
//
// Same ROCKET_MM_PROFILE knob as the other two; own exit line. The one-time ingest is NOT
// here -- it is timed unconditionally on the context (moe_ingest_ms), because it is a cost
// the user pays and must be told about, not one only a developer wants.
static struct { double gather, quant, gemm, scatter, fp16;
                long   calls, gemms, fp16_gemms, rows_used, rows_computed; } g_moeprof;
static int g_moeprof_armed = 0;
static void rocket_moeprof_dump(void) {
    const double npu   = g_moeprof.gather + g_moeprof.quant + g_moeprof.gemm + g_moeprof.scatter;
    const long   total = g_moeprof.gemms + g_moeprof.fp16_gemms;
    ROCKET_LOGI(
        "ROCKET MoE native total(ms): gather=%.0f act_quant=%.0f gemm=%.0f scatter=%.0f "
        "| fp16_streamed=%.0f  (%ld ops; %ld/%ld expert GEMMs native = %.0f%%; %.1f%% padded "
        "rows; gemm=%.0f%% of the native route)\n",
        g_moeprof.gather, g_moeprof.quant, g_moeprof.gemm, g_moeprof.scatter, g_moeprof.fp16,
        g_moeprof.calls, g_moeprof.gemms, total,
        total ? 100.0 * (double)g_moeprof.gemms / (double)total : 0.0,
        g_moeprof.rows_computed ? 100.0 * (double)(g_moeprof.rows_computed - g_moeprof.rows_used)
                                        / (double)g_moeprof.rows_computed : 0.0,
        npu > 0 ? 100.0 * g_moeprof.gemm / npu : 0.0);
}
static inline void rocket_moeprof_arm(void) {
    if (!g_moeprof_armed) { atexit(rocket_moeprof_dump); g_moeprof_armed = 1; }
}

// ROCKET_MOE_COSINE=1: the per-matmul faithfulness probe for the native-quant expert route,
// on REAL weights and REAL activations.
//
// For ONE expert per MUL_MAT_ID op -- rotating, so the probe sweeps every layer, every
// projection and the whole expert set rather than camping on layer 0 -- recompute that exact
// GEMM on the CPU from the raw f32 activations and the UNDECODED GGUF weight (ggml's own
// dequantizer, fp64 accumulate: rocket_cpu_matmul_slice) and take the cosine against what the
// NPU's int8 route returned.
//
// This is the gate the synthetic test cannot be. test-rocket-moe pins the ROUTE (that the
// native path ran, and that its arithmetic is right on inputs we invented), but only a live
// forward pass carries the activation distribution -- outlier channels and all -- that decides
// whether int8 ACTIVATIONS survive without a Hadamard rotation. That was risk R3, and this is
// what retires it on real data instead of a synthetic proxy.
//
// Diagnostic only: the reference is a scalar fp64 triple loop, seconds per probe.
static int rocket_moe_cosine_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_MOE_COSINE"); v = (e && atoi(e) > 0) ? 1 : 0; }
    return v;
}
static struct { double sum, min; long n; } g_moecos = { 0.0, 2.0, 0 };
static int g_moecos_armed = 0;
static void rocket_moecos_dump(void) {
    if (!g_moecos.n) return;
    ROCKET_LOGI("ROCKET MoE native-quant cosine vs CPU fp64 reference (real weights, real "
                "activations): mean=%.6f min=%.6f over %ld expert GEMMs\n",
                g_moecos.sum / (double)g_moecos.n, g_moecos.min, g_moecos.n);
}
// cos(a,b) in double. Both vectors are a full [M_e,N] expert output, so a single cosine over
// the flattened tile is the right summary: it weights each output element equally, which is
// what a downstream layer sees.
static double rocket_cosine_f32(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na <= 0 || nb <= 0) return 1.0;   // an all-zero tile carries no angle
    return dot / (sqrt(na) * sqrt(nb));
}

// ===========================================================================
// int8 W8A8 quant/dequant. The int8 NPU path computes int8 x int8
// -> int32; the host applies scales. SYMMETRIC int8: q = round(x/s) clamped to
// [-127,127] (NOT -128, so +/- are symmetric and the scale is exact both ways),
// s = amax/127. Activations are scaled PER ROW (per token), matching the fp16
// path's row-wise scale -- this tames a row's magnitude but NOT per-CHANNEL
// activation outliers; that residual is the gibberish risk the Hadamard rotation
// addresses. Weights are scaled PER OUTPUT CHANNEL (per row of B[N,K]); weights
// are bounded so this is the easy half. The quant is scalar.
// ===========================================================================
static inline int8_t rocket_q8(float x, float inv) {
    long q = lrintf(x * inv);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

// NEON row kernels for the symmetric-int8 quant loops on the RK3576 route. The scalar
// loops pay one lrintf@plt CALL per element, and the L1d set-conflict class measured on
// this route rides on exactly that call's per-element stack traffic (the q8 phase moved
// 9.9 -> 21.9 s across binary epochs with the loop's instructions identical); a lane
// loop with no per-element call gives that congruence nothing to form on. The lane
// forms are rk_quant_act_i8_group's (the MoE path's, further down): amax by vabs/vmax
// -- max is exact, no rounding -- and quantize by fmul then vcvtnq_s32_f32, which
// rounds to nearest ties-to-even exactly as lrintf does under the default rounding
// mode. The float pre-clamp to +/-127 makes the saturating narrows exact and matches
// the scalar's integer clamp on every value including the +/-127.5 ties (both sides
// land on 127/-127). Bit-identical to the scalar tail by construction; the RK3576
// `--chunks 4` equality cell gates it on real data.
static inline float rk_amax_row(const float * src, int64_t n) {
    float amax = 0.0f;
    int64_t k = 0;
#ifdef ROCKET_NEON_F32
    float32x4_t vm = vdupq_n_f32(0.0f);
    for (; k + 4 <= n; k += 4) vm = vmaxq_f32(vm, vabsq_f32(vld1q_f32(src + k)));
    amax = vmaxvq_f32(vm);
#endif
    for (; k < n; k++) { const float v = fabsf(src[k]); if (v > amax) amax = v; }
    return amax;
}
static inline void rk_q8_row(const float * src, int8_t * dst, int64_t n, float inv) {
    int64_t k = 0;
#ifdef ROCKET_NEON_F32
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vhi  = vdupq_n_f32(127.0f), vlo = vdupq_n_f32(-127.0f);
    auto qv = [&](const float * p) {
        return vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p), vinv), vlo), vhi));
    };
    for (; k + 16 <= n; k += 16) {
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(qv(src + k     )), vqmovn_s32(qv(src + k +  4)));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(qv(src + k +  8)), vqmovn_s32(qv(src + k + 12)));
        vst1q_s8(dst + k, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
#endif
    for (; k < n; k++) dst[k] = rocket_q8(src[k], inv);
}

// A[M,K] f32 -> int8 [M,K] + per-row scale a_scale[m] (= amax_row/127).
static void rocket_quant_act_int8(const float * src, int8_t * dst,
                                  int64_t M, int64_t K, float * a_scale) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t m = 0; m < M; m++) {
        const float * row = src + m * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) { const float v = fabsf(row[k]); if (v > amax) amax = v; }
        const float s   = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        int8_t * d = dst + m * K;
        for (int64_t k = 0; k < K; k++) d[k] = rocket_q8(row[k], inv);
        a_scale[m] = s;
    }
    if (prof) { rocket_i8prof_arm(); g_i8prof.act_q += rocket_now_ms() - t0; g_i8prof.act_calls++; }
}

// B[N,K] (f16 or f32) -> int8 [N,K] + per-channel scale b_scale[n] (= amax_row/127).
static void rocket_quant_wt_int8(const void * src, bool f16, int8_t * dst,
                                 int64_t N, int64_t K, float * b_scale) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t n = 0; n < N; n++) {
        const ggml_fp16_t * brow16 = (const ggml_fp16_t *)src + n * K;
        const float       * brow32 = (const float       *)src + n * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            const float v = fabsf(f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k]);
            if (v > amax) amax = v;
        }
        const float s   = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        int8_t * d = dst + n * K;
        for (int64_t k = 0; k < K; k++)
            d[k] = rocket_q8(f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k], inv);
        b_scale[n] = s;
    }
    if (prof) { rocket_i8prof_arm(); g_i8prof.wt_q += rocket_now_ms() - t0; g_i8prof.wt_calls++; }
}

// int32 C[M,N] -> f32 dst, applying both scales: dst[m,n] = C[m,n]*a_scale[m]*b_scale[n].
static void rocket_dequant_int8(const int32_t * src, float * dst,
                                int64_t M, int64_t N,
                                const float * a_scale, const float * b_scale) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    for (int64_t m = 0; m < M; m++) {
        const float as = a_scale[m];
        const int32_t * srow = src + m * N;
        float * drow = dst + m * N;
        for (int64_t n = 0; n < N; n++) drow[n] = (float)srow[n] * as * b_scale[n];
    }
    if (prof) { rocket_i8prof_arm(); g_i8prof.dq += rocket_now_ms() - t0; g_i8prof.dq_calls++; }
}

// ===========================================================================
// Hadamard rotation for W8A8. Rotating A and B by an orthonormal
// Hadamard H along K BEFORE int8 quant spreads per-channel activation outliers
// across all channels, so the per-row scale is not pinned by one fat
// channel. Since H is orthonormal, (A H)(B H)^T = A B^T exactly -> the int32
// result and the host dequant are UNCHANGED; only the quantized values differ.
// Validated standalone in test-hadamard-quant.cpp (H60 orthogonality + the
// rotation preserving A*B^T to ~1e-15, incl. Gemma's 3840/15360).
//
// Gemma's K are 2^k (512/2048/4096/8192) or 60*2^k (3840=64*60, 15360=256*60),
// so H_K is a plain FWHT or the Kronecker H_{2^k} (x) H_60. H_60 is built once via
// Paley I (59 prime, =3 mod 4). Opt-in via ROCKET_INT8_HADAMARD=1 (needs ROCKET_INT8).
// ===========================================================================
static int  rk_modpow(long a, long e, long p) { long r = 1; a %= p; while (e) { if (e&1) r = r*a%p; a = a*a%p; e >>= 1; } return (int)r; }
static int  rk_legendre(int a, int p) { a %= p; if (a < 0) a += p; if (a == 0) return 0; return rk_modpow(a, (p-1)/2, p) == 1 ? 1 : -1; }
static float g_H60[60][60];
static bool  g_H60_ready = false;
// Builds the 60x60 Paley-Hadamard block once into g_H60. The init is idempotent (the
// table is a pure function of the constants), so a concurrent first-call race only
// rebuilds the same values -- benign. This (and the lazy getenv knob caches) assumes
// the documented usage of one backend context driven from a single thread; a future
// multi-context/multi-thread driver should fold this build into context creation.
static void rk_build_H60(void) {
    if (g_H60_ready) return;
    const int q = 59;
    for (int i = 0; i < 60; i++)
        for (int j = 0; j < 60; j++) {
            if      (i == 0) g_H60[i][j] =  1;
            else if (j == 0) g_H60[i][j] = -1;
            else if (i == j) g_H60[i][j] =  1;
            else             g_H60[i][j] = (float)rk_legendre((i-1)-(j-1), q);
        }
    g_H60_ready = true;
}
static bool rk_ispow2(int x) { return x > 0 && (x & (x-1)) == 0; }
// The non-pow2 path stages B2=K/60 lanes in a fixed 4096-float stack buffer (see
// rk_hadamard_rotate), so cap K/60 at 4096 (K<=245760). Gemma's max K=15360 -> B2=256,
// far under. The pow2(K) path rotates in place with no such buffer.
static bool rk_hadamard_ok(int K) {
    return rk_ispow2(K) || (K % 60 == 0 && rk_ispow2(K/60) && K/60 <= 4096);
}
static void rk_fwht(float *v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1) {
            int j = i;
#ifdef ROCKET_NEON_F32
            // len is a power of 2; for len>=4 the whole [i,i+len) range is 4-aligned
            // contiguous, so the scalar tail below runs 0 iterations.
            if (len >= 4)
                for (; j + 4 <= i + len; j += 4) {
                    float32x4_t a = vld1q_f32(v + j), b = vld1q_f32(v + j + len);
                    vst1q_f32(v + j,       vaddq_f32(a, b));
                    vst1q_f32(v + j + len, vsubq_f32(a, b));
                }
#endif
            for (; j < i + len; j++) { float a = v[j], b = v[j+len]; v[j] = a + b; v[j+len] = a - b; }
        }
}
// Rotate row[0..K) in place by H_K/sqrt(K). Caller guarantees rk_hadamard_ok(K).
// The H_60 block matvec, the FWHT, and the final scale are NEON-vectorized
// (fp32) under ROCKET_NEON_F32; the scalar fallback (below each #ifdef) is bit-for-bit
// the original. The H_60 entries are +/-1 floats, so the per-output dot is a plain
// fmla-reduction over 60 (=15*4) lanes. (gather/scatter of the cross-block columns
// stays scalar -- it's the smaller cost; the H_60 matvec is the FFN-K monster.)
// Returns 0, or <0 if K exceeds the col[] bound (rk_hadamard_ok guarantees it does
// not on the offload path; the check degrades to the caller's CPU fallback instead
// of GGML_ASSERT-aborting the host process from inside a loadable .so).
static int rk_hadamard_rotate(float *row, int K) {
    const float inv = 1.0f / sqrtf((float)K);
    if (rk_ispow2(K)) {
        rk_fwht(row, K);
    } else {
        const int B2 = K / 60;                       // power of 2, <= 256 for Gemma
        if (B2 > 4096) return -1;                     // col[] bound; rk_hadamard_ok guarantees it
        float col[4096];                             // B2 lanes; ample for K up to 245760
        for (int b = 0; b < B2; b++) {               // H_60 within each 60-block
            float *blk = row + b*60, out[60];
            for (int o = 0; o < 60; o++) {
                const float *h = g_H60[o];
#ifdef ROCKET_NEON_F32
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = 0; k < 60; k += 4)      // 60 = 15*4, exact
                    acc = vfmaq_f32(acc, vld1q_f32(h + k), vld1q_f32(blk + k));
                out[o] = vaddvq_f32(acc);
#else
                float s = 0; for (int k = 0; k < 60; k++) s += h[k]*blk[k]; out[o] = s;
#endif
            }
            for (int o = 0; o < 60; o++) blk[o] = out[o];
        }
        for (int lane = 0; lane < 60; lane++) {       // FWHT across blocks, per lane
            for (int b = 0; b < B2; b++) col[b] = row[b*60 + lane];
            rk_fwht(col, B2);
            for (int b = 0; b < B2; b++) row[b*60 + lane] = col[b];
        }
    }
    int k = 0;
#ifdef ROCKET_NEON_F32
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (; k + 4 <= K; k += 4) vst1q_f32(row + k, vmulq_f32(vld1q_f32(row + k), vinv));
#endif
    for (; k < K; k++) row[k] *= inv;
    return 0;
}
// BLOCK-DIAGONAL rotation, for a K that is neither a power of two nor 60*2^k.
//
// rk_hadamard_ok() covers exactly the two constructions above, and they do not cover a K
// like Qwen2.5-1.5B's 1536 (= 3*2^9): not a power of two, and 1536 % 60 != 0. That matters
// more than a missing shape, because 1536 is the contraction depth of EVERY projection in
// one of the two models the W8A8 accuracy numbers were measured on — so without this the
// RK3576 entry refuses that whole model and the port cannot run the arithmetic it was
// budgeted from.
//
// This is the construction those numbers were computed in: H_b applied to each aligned
// block of b = the largest power of two dividing K, which is still orthonormal (so
// (A H)(B H)^T = A B^T exactly) and smears an outlier over its block rather than over the
// whole contraction. That makes it the WEAKER of the two — it is the fallback, not a
// replacement — but it is always available and it is what the accuracy corpus used
// wherever K was not a power of two. A K whose largest power-of-two divisor is under 32
// is refused rather than rotated: at that block size the rotation stops smearing outliers
// and the route's whole premise goes with it.
static int rk_hadamard_rotate_blocks(float *row, int K) {
    const int b = K & -K;                     // largest power of two dividing K
    if (b < 32) return -1;
    const float inv = 1.0f / sqrtf((float)b);
    for (int i = 0; i < K; i += b) {
        rk_fwht(row + i, b);
        int k = 0;
#ifdef ROCKET_NEON_F32
        const float32x4_t vinv = vdupq_n_f32(inv);
        for (; k + 4 <= b; k += 4)
            vst1q_f32(row + i + k, vmulq_f32(vld1q_f32(row + i + k), vinv));
#endif
        for (; k < b; k++) row[i + k] *= inv;
    }
    return 0;
}

// Hadamard variants of the quant kernels: rotate each row into `tmp` (K floats,
// caller-owned, reused) then quantize. Same per-row / per-channel symmetric int8.
// Return 0, or <0 if a row's K exceeds the rotation bound (caller -> CPU fallback).
static int rocket_quant_act_int8_had(const float * src, int8_t * dst,
                                      int64_t M, int64_t K, float * a_scale, float * tmp) {
    const bool prof = rocket_convprof_on();
    double rot = 0, qt = 0, t0 = 0;
    for (int64_t m = 0; m < M; m++) {
        const float * row = src + m * K;
        if (prof) t0 = rocket_now_ms();
        for (int64_t k = 0; k < K; k++) tmp[k] = row[k];
        if (rk_hadamard_rotate(tmp, (int)K) < 0) return -1;
        if (prof) { rot += rocket_now_ms() - t0; t0 = rocket_now_ms(); }
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) { const float v = fabsf(tmp[k]); if (v > amax) amax = v; }
        const float s = (amax > 0.0f) ? amax / 127.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + m * K;
        for (int64_t k = 0; k < K; k++) d[k] = rocket_q8(tmp[k], inv);
        if (prof) qt += rocket_now_ms() - t0;
        a_scale[m] = s;
    }
    if (prof) { rocket_i8prof_arm(); g_i8prof.act_rot += rot; g_i8prof.act_q += qt; g_i8prof.act_calls++; }
    return 0;
}
// `blocks` selects the block-diagonal construction for a K the two exact ones do not
// cover (rk_hadamard_rotate_blocks); the RK3576 entry is the only caller that passes it.
// `src_stride` is the source row pitch in ELEMENTS and defaults to K. It is > K for
// exactly one caller: the RK3576 K-split, whose chunk c reads columns [K0, K0+Kc) out of a
// [N][Ktot] weight, so the sub-block is strided rather than contiguous and materializing it
// would cost a second N*Kc buffer per chunk for nothing.
static int rocket_quant_wt_int8_had(const void * src, bool f16, int8_t * dst,
                                     int64_t N, int64_t K, float * b_scale, float * tmp,
                                     bool blocks = false, int64_t src_stride = 0) {
    const bool prof = rocket_convprof_on();
    double rot = 0, qt = 0, t0 = 0;
    if (src_stride <= 0) src_stride = K;
    for (int64_t n = 0; n < N; n++) {
        const ggml_fp16_t * brow16 = (const ggml_fp16_t *)src + n * src_stride;
        const float       * brow32 = (const float       *)src + n * src_stride;
        if (prof) t0 = rocket_now_ms();
        for (int64_t k = 0; k < K; k++) tmp[k] = f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k];
        if ((blocks ? rk_hadamard_rotate_blocks(tmp, (int)K)
                    : rk_hadamard_rotate(tmp, (int)K)) < 0) return -1;
        if (prof) { rot += rocket_now_ms() - t0; t0 = rocket_now_ms(); }
        const float amax = rk_amax_row(tmp, K);
        const float s = (amax > 0.0f) ? amax / 127.0f : 1.0f, inv = 1.0f / s;
        rk_q8_row(tmp, dst + n * K, K, inv);
        if (prof) qt += rocket_now_ms() - t0;
        b_scale[n] = s;
    }
    if (prof) { rocket_i8prof_arm(); g_i8prof.wt_rot += rot; g_i8prof.wt_q += qt; g_i8prof.wt_calls++; }
    return 0;
}

// ===========================================================================
// int4 W4A4 quant. The int4 sibling of the int8 quant kernels above: SYMMETRIC
// int4, q = round(x/s) clamped to [-7,7] (NOT -8, so +/- are symmetric and the
// scale is exact both ways), s = amax/7. Activations PER ROW, weights PER OUTPUT
// CHANNEL -- identical structure to int8, only the clamp/divisor (7 vs 127) and
// the +/-7 range differ. The dequant (rocket_dequant_int8: int32 C * scales) and
// the Hadamard rotation (rk_hadamard_rotate) are dtype-independent and reused
// as-is. Each int4 value is stored one-per-int8_t (the rocket_matmul_int4 contract;
// the driver nibble-packs into the NPU BOs). NOTE the [-7,7] range bounds the
// int16-partial saturation in the matmul: |q*q| <= 49, so a K-tile of <=668 cannot
// overflow int16 -- the in-model caller caps Kt accordingly (see mul_mat_int4).
// ===========================================================================
static inline int8_t rocket_q4(float x, float inv) {
    long q = lrintf(x * inv);
    if (q >  7) q =  7;
    if (q < -7) q = -7;
    return (int8_t)q;   // one int4 value per int8_t
}
// A[M,K] f32 -> int4 [M,K] (one per byte) + per-row scale a_scale[m] (= amax_row/7).
static void rocket_quant_act_int4(const float * src, int8_t * dst,
                                  int64_t M, int64_t K, float * a_scale) {
    for (int64_t m = 0; m < M; m++) {
        const float * row = src + m * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) { const float v = fabsf(row[k]); if (v > amax) amax = v; }
        const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + m * K;
        for (int64_t k = 0; k < K; k++) d[k] = rocket_q4(row[k], inv);
        a_scale[m] = s;
    }
}
// B[N,K] (f16 or f32) -> int4 [N,K] (one per byte) + per-channel scale b_scale[n].
static void rocket_quant_wt_int4(const void * src, bool f16, int8_t * dst,
                                 int64_t N, int64_t K, float * b_scale) {
    for (int64_t n = 0; n < N; n++) {
        const ggml_fp16_t * brow16 = (const ggml_fp16_t *)src + n * K;
        const float       * brow32 = (const float       *)src + n * K;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            const float v = fabsf(f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k]);
            if (v > amax) amax = v;
        }
        const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + n * K;
        for (int64_t k = 0; k < K; k++)
            d[k] = rocket_q4(f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k], inv);
        b_scale[n] = s;
    }
}
// Hadamard variants: rotate each row into tmp (caller-owned, reused) then int4-quant.
// Return 0, or <0 if K exceeds the rotation bound (caller -> CPU/fp16 fallback).
static int rocket_quant_act_int4_had(const float * src, int8_t * dst,
                                     int64_t M, int64_t K, float * a_scale, float * tmp) {
    for (int64_t m = 0; m < M; m++) {
        const float * row = src + m * K;
        for (int64_t k = 0; k < K; k++) tmp[k] = row[k];
        if (rk_hadamard_rotate(tmp, (int)K) < 0) return -1;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) { const float v = fabsf(tmp[k]); if (v > amax) amax = v; }
        const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + m * K;
        for (int64_t k = 0; k < K; k++) d[k] = rocket_q4(tmp[k], inv);
        a_scale[m] = s;
    }
    return 0;
}
static int rocket_quant_wt_int4_had(const void * src, bool f16, int8_t * dst,
                                    int64_t N, int64_t K, float * b_scale, float * tmp) {
    for (int64_t n = 0; n < N; n++) {
        const ggml_fp16_t * brow16 = (const ggml_fp16_t *)src + n * K;
        const float       * brow32 = (const float       *)src + n * K;
        for (int64_t k = 0; k < K; k++) tmp[k] = f16 ? ggml_fp16_to_fp32(brow16[k]) : brow32[k];
        if (rk_hadamard_rotate(tmp, (int)K) < 0) return -1;
        float amax = 0.0f;
        for (int64_t k = 0; k < K; k++) { const float v = fabsf(tmp[k]); if (v > amax) amax = v; }
        const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + n * K;
        for (int64_t k = 0; k < K; k++) d[k] = rocket_q4(tmp[k], inv);
        b_scale[n] = s;
    }
    return 0;
}

// GROUP-WISE int4 quant: rotate the full K row (Hadamard) then quantize each K-group
// of `group` to int4 with its OWN scale -- the W4 quality lever (a separate dequant
// scale per K-slice bounds the int4 error, the GPTQ/AWQ regime). dst is [rows*K] int4;
// scale is [rows*nG] (nG=K/group). tmp is K floats (caller-owned). Hadamard rotates
// the FULL row BEFORE the per-group quant (H is orthogonal, so the rotated per-group
// scales reconstruct the original dot product exactly via rocket_matmul_int4_groupwise).
// Returns 0, or <0 if K exceeds the rotation bound (caller -> fp16 fallback).
// One row of the group-wise int4 quant: load -> (optional) Hadamard-rotate into the
// caller's per-row scratch `tmp` -> per-group amax/scale -> int4. Rows are independent.
// Returns 0, or <0 if the rotation rejects K (caller -> fp16 fallback).
static inline int rk_quant_int4_grouped_row(const float * srcf, const void * srcv, bool from_f32,
        bool f16, int8_t * dst, int64_t r, int64_t K, int group, int nG, bool had,
        float * scale, float * tmp) {
    if (from_f32) { const float * row = srcf + r * K;
                    for (int64_t k = 0; k < K; k++) tmp[k] = row[k]; }
    else { const ggml_fp16_t * b16 = (const ggml_fp16_t *)srcv + r * K;
           const float * b32 = (const float *)srcv + r * K;
           for (int64_t k = 0; k < K; k++) tmp[k] = f16 ? ggml_fp16_to_fp32(b16[k]) : b32[k]; }
    if (had && rk_hadamard_rotate(tmp, (int)K) < 0) return -1;
    for (int g = 0; g < nG; g++) {
        float amax = 0.0f;
        for (int k = 0; k < group; k++) { const float v = fabsf(tmp[(size_t)g*group + k]); if (v > amax) amax = v; }
        const float s = (amax > 0.0f) ? amax / 7.0f : 1.0f, inv = 1.0f / s;
        int8_t * d = dst + r * K + (size_t)g * group;
        for (int k = 0; k < group; k++) d[k] = rocket_q4(tmp[(size_t)g*group + k], inv);
        scale[(size_t)r * nG + g] = s;
    }
    return 0;
}

// GROUP-WISE int4 quant of `rows` weight/activation rows. The per-row Hadamard rotation
// + group-quant of a 12B model's weights is the dominant resident-build cost; rows are
// independent, so split them across CPU threads (each owns its own K-float scratch).
// The result is bit-identical to the serial path (no cross-row state). `tmp` is only the
// single-thread fallback's scratch. rk_hadamard_rotate is thread-safe (in-place on its
// `row`, local stack buffers, read-only g_H60 built by the caller before this call).
static int rocket_quant_int4_grouped(const float * srcf, const void * srcv, bool from_f32,
                                     bool f16, int8_t * dst, int64_t rows, int64_t K,
                                     int group, bool had, float * scale, float * tmp) {
    const int nG = (int)(K / group);
    unsigned hw = std::thread::hardware_concurrency();
    int nthr = (int)(hw ? hw : 1);
    if (nthr > 8) nthr = 8;
    if (rows < 64 || nthr <= 1) {                 // small: serial (use the caller's tmp)
        for (int64_t r = 0; r < rows; r++)
            if (rk_quant_int4_grouped_row(srcf, srcv, from_f32, f16, dst, r, K, group, nG, had, scale, tmp) < 0)
                return -1;
        return 0;
    }
    std::atomic<bool> failed(false);
    auto worker = [&](int64_t r0, int64_t r1) {
        std::vector<float> t((size_t)K);
        for (int64_t r = r0; r < r1 && !failed.load(std::memory_order_relaxed); r++)
            if (rk_quant_int4_grouped_row(srcf, srcv, from_f32, f16, dst, r, K, group, nG, had, scale, t.data()) < 0)
                failed.store(true, std::memory_order_relaxed);
    };
    std::vector<std::thread> th;
    const int64_t per = (rows + nthr - 1) / nthr;
    for (int i = 0; i < nthr; i++) {
        int64_t r0 = (int64_t)i * per, r1 = r0 + per > rows ? rows : r0 + per;
        if (r0 >= r1) break;
        th.emplace_back(worker, r0, r1);
    }
    for (auto & t : th) t.join();
    return failed.load() ? -1 : 0;
}

// ===========================================================================
// One-shot int8 / int4 / bf16 matmul generators
// ===========================================================================
static std::string rocket_weight_key(const ggml_tensor * t);   // defined below (fp16 wcache)
static size_t rocket_meminfo_bytes(const char * field);        // defined below (init); read in build_resident

// W8A8 int8 matmul for one plain 2D static-weight GEMM, via the one-shot tiled
// int8 driver. Returns 0 (dst written) or <0 to fall through to the fp16 path.
// Requires N%32==0 (int8 weight k-group); the caller checks that + 2D-ness.
static int ggml_backend_rocket_mul_mat_int8(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->int8_failed) return -1;
    if (ctx->int8_fd < 0) {
        ctx->int8_fd = rocket_open();
        if (ctx->int8_fd < 0) { ctx->int8_failed = true; return -1; }
    }

    const int Mp = rocket_pad_m(M);   // driver needs M%4; pad rows quantize to 0
    const bool b_f16 = (src0->type == GGML_TYPE_F16);

    if (ctx->int8_hadamard < 0) {
        const char * e = getenv("ROCKET_INT8_HADAMARD"); ctx->int8_hadamard = e ? atoi(e) : 0;
    }
    const bool had = ctx->int8_hadamard && rk_hadamard_ok(K);
    if (had) rk_build_H60();

    // ---- weights: reuse the resident rotated-int8 weight if cached, else build it
    // ONCE (per-channel quant, pre-rotated when hadamard) and cache. Weight prep is
    // the dominant int8 cost and weights are constant for the process lifetime, so
    // every forward pass after the first skips it. Keyed on the stable weight NAME.
    const int8_t * qB = nullptr;
    const float  * b_scale = nullptr;
    const std::string key = rocket_weight_key(src0);

    auto it = key.empty() ? ctx->int8_wcache.end() : ctx->int8_wcache.find(key);
    if (it != ctx->int8_wcache.end()
        && it->second.N == N && it->second.K == K && it->second.hadamard == had) {
        qB = it->second.qB.data();
        b_scale = it->second.b_scale.data();
    } else {
        // Build+cache ONLY if it fits the budget. If not (cache full, or the weight
        // has no stable name), DON'T re-quantize per call -- that thrashes a memory-
        // tight box (full int8 model ~10GB can't coexist with the 22GB F16 weights).
        // Return -1 so this weight runs on the fp16 path: int8-mode degrades to "int8
        // for what fits the budget, fp16 for the rest", usable and thrash-free.
        const size_t est = (size_t)N * K + (size_t)N * sizeof(float);
        if (key.empty() || (ctx->int8_cache_budget != 0
                            && ctx->int8_resident_bytes + est > ctx->int8_cache_budget))
            return -1;

        std::vector<int8_t> bq((size_t)N * K);
        std::vector<float>  bs((size_t)N);
        if (had) { std::vector<float> tmp((size_t)K);
                   if (rocket_quant_wt_int8_had(src0->data, b_f16, bq.data(), N, K, bs.data(), tmp.data()) < 0) return -1; }
        else     { rocket_quant_wt_int8(src0->data, b_f16, bq.data(), N, K, bs.data()); }

        rocket_int8_weight e;
        e.qB = std::move(bq); e.b_scale = std::move(bs);
        e.N = N; e.K = K; e.hadamard = had; e.bytes = est;
        auto & slot = (ctx->int8_wcache[key] = std::move(e));
        ctx->int8_resident_bytes += est;
        qB = slot.qB.data(); b_scale = slot.b_scale.data();
        if (rocket_debug_on())
            GGML_LOG_DEBUG("[int8-cache] +%-22s N=%6d K=%6d had=%d  resident=%zuMB\n",
                    key.c_str(), N, K, (int)had, ctx->int8_resident_bytes >> 20);
    }

    // ---- activations: always per call (cheap; M is small)
    std::vector<int8_t> qA((size_t)Mp * K, 0);    // pad rows = 0
    std::vector<float>  a_scale((size_t)Mp, 1.0f);
    if (had) { std::vector<float> tmp((size_t)K);
               if (rocket_quant_act_int8_had((const float *)src1->data, qA.data(), M, K, a_scale.data(), tmp.data()) < 0) return -1; }
    else     { rocket_quant_act_int8((const float *)src1->data, qA.data(), M, K, a_scale.data()); }

    std::vector<int32_t> C32((size_t)Mp * N);
    int rc = rocket_matmul_int8(ctx->int8_fd, Mp, K, N, qA.data(), qB, C32.data());
    if (rc != 0) return -1;

    rocket_dequant_int8(C32.data(), (float *)dst->data, M, N, a_scale.data(), b_scale);
    return 0;
}

// ===========================================================================
// RK3576 W8A8. A DIFFERENT ROUTE, not a parameterization of the one above.
//
// The RK3588 int8 entry hands back a raw int32 accumulator and the host applies both
// scales. The RK3576's writes int8: any output element wider than one byte poisons that
// part's NEXT submit across processes, so the int32 sibling is not a route a frontend
// takes. What the part gives instead is a per-output-COLUMN requant in the DPU epilogue,
// and the whole port hangs off supplying its scale.
//
// THE ROUTE, and every piece of it is load-bearing on the two models measured:
//
//   1. rotate A and B by an orthonormal Hadamard along K. MANDATORY, not a knob.
//   2. quantize A PER TENSOR and B PER OUTPUT CHANNEL. Per-axis INPUT scales are free at
//      this interface and better on every per-GEMM norm, and end to end they are 38x
//      WORSE on one model of two and 1.08x better on the other — so the per-tensor
//      activation scale is the measured choice, not the lazy one.
//   3. per-column output requant, whose scale is frozen from a calibration pass and
//      divided by a safety factor near 2 (see rocket_rk3576_weight).
//   4. host dequantize by the scale the entry was ASKED for, not the gain its integer
//      ramp delivered. The two differ by `worst_rel_err` (0.5-0.9% median on a real
//      model's shapes); de-quantizing by the achieved gain instead is a measured WASH and
//      is deliberately not done.
//
// Composed, that route measures 1.008x / 1.020x wikitext-2 perplexity against fp32 on
// Qwen2.5-1.5B and SmolLM2-1.7B, with the frozen scale and the safety factor — against
// 1.11x / 2.26x for the per-tensor output scale this replaces.
// [host arithmetic over two models, 2026-08-10; the DEVICE composition of the route is
//  gated separately — nothing here was measured on the part]
//
// M carries NO constraint on this part (M=1 computes), which is the opposite of the
// RK3588, where rows are the conv's spatial height and height < 4 mis-computes. So there
// is deliberately no rocket_pad_m() here.
//
// WHERE THIS CODE AND THAT INSTRUMENT KNOWINGLY DIFFER, so a device-vs-host comparison is
// read against the right reference: rocket_q8() clamps to [-127, 127] and the simulator
// clips to [-128, 127], which can differ by one code on the single most negative element
// of a tensor; and for a K of the form 60*2^k this code takes the stronger Kronecker
// H_64 (x) H_60 where the simulator would take the block-diagonal fallback. Neither model
// behind the accuracy numbers has such a K, so that second one is unscored either way.

static bool rocket_rk3576_selected(void) {
    const struct rocket_hw_profile * hw = rocket_hw_current();
    return hw && hw->name && strcmp(hw->name, "rk3576") == 0;
}

// ---------------------------------------------------------------------------------
// RK3576 W8A8 frontend phase profiling — ROCKET_MM_PROFILE, the same knob the library's
// three accumulators use, and its own exit line beside theirs.
//
// The library's buckets stop at the entry, and on this route the terms OUTSIDE it are not
// a rounding error: the activation rotation is O(M*K log K) every call, the dequantize is
// O(M*N) every call, and the three scratch vectors are value-initialized, so each call
// memsets M*K*4 + M*K + M*N bytes before it computes anything. None of those has a bucket
// in rocket_matmul.c, and a term with no bucket is charged to whichever neighbour spans it.
//
// WHAT IT CANNOT SEE: the rest of the model. Attention, the norms, rope, the scheduler and
// every CPU-placed matmul are outside this handler, so these buckets and the library's sum
// to the OFFLOADED path's time and not to the prefill wall. Reading a share of the wall off
// them is the error the caps here have made five times running.
static int rk76_prof_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("ROCKET_MM_PROFILE") != NULL;
    return v;
}
static struct {
    double valloc, quant, cal, entry, dequant;
    double calscan;   // SUBSET of `cal`: the bootstrap's per-column maximum scan alone
    long calls, cal_calls;
} g_rk76_fe;
static int g_rk76_fe_armed = 0;
// ROCKET_LOGI and NOT GGML_LOG_*, for the reason the convert/i8/moe dumps above give: the
// tool this has to survive is llama-bench, which installs a no-op ggml log callback and
// swallows everything sent through ggml. ROCKET_LOG_STDERR=1 tees the rocket channel past
// it. A profiler silenced by the profiling harness reads as "nothing to report".
static void rk76_fe_dump(void) {
    ROCKET_LOGI("ROCKET rk3576 frontend profile total(ms): valloc=%.0f quant+rot=%.0f "
                "calibrate=%.0f entry=%.0f dequant=%.0f calscan=%.0f  over %ld calls "
                "(%ld of them calibration forwards)\n",
                g_rk76_fe.valloc, g_rk76_fe.quant, g_rk76_fe.cal, g_rk76_fe.entry,
                g_rk76_fe.dequant, g_rk76_fe.calscan, g_rk76_fe.calls, g_rk76_fe.cal_calls);
}
static double rk76_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
// The handler runs on the backend's single graph_compute dispatch thread, so the
// accumulator needs no lock; arm the dump on first use, as the library's do.
#define RK76_FE_T0()       (fe_prof ? rk76_now_ms() : 0.0)
#define RK76_FE_ADD(f, t0) do { if (fe_prof) {                                   \
        if (!g_rk76_fe_armed) { atexit(rk76_fe_dump); g_rk76_fe_armed = 1; }     \
        g_rk76_fe.f += rk76_now_ms() - (t0); } } while (0)

// Which rotation this K takes. The two exact constructions where they exist, the
// block-diagonal fallback otherwise — and the fallback is what carries a K like Qwen's
// 1536. `blocks` is the same choice made once for the weight and every activation, since a
// weight rotated by one H and activations by another does not reconstruct the product.
static bool rk76_rot_blocks(int K)   { return !rk_hadamard_ok(K); }
static bool rk76_rot_ok(int K) {
    return rk_hadamard_ok(K) || (K & -K) >= 32;
}
static int rk76_rotate(float * row, int K) {
    return rk76_rot_blocks(K) ? rk_hadamard_rotate_blocks(row, K)
                              : rk_hadamard_rotate(row, K);
}

// ---------------------------------------------------------------------------------
// THE K SPLIT. Which contraction depths go to the part in one piece, and how the rest are
// cut.
//
// Two separate things put a K out of reach of one device call:
//
//   1. the library refuses K >= 6176 — past there its single-task planner cannot fit an
//      output tile — and the int32 K-split route that would run it wedges the part until
//      it is rebooted, so that is not a route a frontend takes;
//   2. a set of K at which an int8 matmul job raises no completion at all. The submit is
//      retired by the driver's 125 ms backstop, the surface is untouched, and the write
//      guard's redo (after a power-domain cycle) succeeds — so the result is CORRECT and
//      40-480x slower: 1028 ms at K=2240 against 24.6 ms at K=2304 on the same plan.
//      Mechanism unknown.
//      [HW sweep, M=512 N=1536, H96 MAX M9, rocket 1.6.0, 2026-08-11]
//
// The stall set below is what that sweep MEASURED, at one (M, N), on a step of 128 above
// 2304 — so it is a list of known-bad points and emphatically not a boundary. Steering off
// it is an optimization, not a correctness guard: a chunk that stalls anyway still returns
// the right answer, just slowly.
static bool rk76_k_stalls(int K) {
    static const int s[] = { 2208, 2240, 2272, 2432, 4416, 4448, 4480, 4512, 4544 };
    for (size_t i = 0; i < sizeof(s)/sizeof(s[0]); i++) if (s[i] == K) return true;
    return false;
}
static bool rk76_k_single_ok(int K) {
    return K > 0 && K % 32 == 0 && K < 6176 && rk76_rot_ok(K) && !rk76_k_stalls(K);
}
// The chunk-size preference, largest first. Every entry satisfies rk76_k_single_ok(), and
// 32 is on the list so the greedy walk always terminates EXACTLY: K%32==0 is a claim-time
// gate, so every remainder is a multiple of 32 and the last entry covers it.
//
// 2304 and 2048 are the two that were measured clean, 48 submits and 0 redos, and greedy
// returns 3*2304 + 2048 for the shape this exists for — Qwen2.5-1.5B's ffn_down at
// K=8960. 1536 is a real model's own K, clean on the same run. The rest are UNMEASURED and
// are the tail of a general K: a model whose chunking lands on them should have that chunk
// timed (automation/rk3576-ksplit-arm.c cost, one process a cell, read the profiler's redo
// count) before its numbers are quoted, because the stall map's step leaves gaps a clean
// sample says nothing about. ROCKET_RK3576_KCHUNK moves the head of this list so a
// candidate can be validated without a rebuild.
static const int RK76_KCHUNK_PREF[] = { 2304, 2048, 1536, 1024, 512, 256, 128, 64, 32 };
static const size_t RK76_KCHUNK_MAX = 16;   // a runaway guard; K=8960 takes 4

// Chunk Ktot into depths the entry will take. Returns false if no chunking exists, which
// is what the claim gate asks. A K the part takes whole is one chunk, so the single-call
// path and the split path are the same code and the shipping behaviour at K < 6176 is
// unchanged.
// `out` may be NULL, which is how the claim gate asks the question without allocating —
// supports_op runs per node per graph and the answer is the same either way.
//
// THE SPLIT IS OFF BY DEFAULT AND ROCKET_RK3576_KSPLIT=1 TURNS IT ON, because on the one
// model measured it costs the OUTPUT 1.60x wikitext-2 perplexity — 15.84 against the same
// build's 9.90 with the down-projections on the CPU, and the CPU F16 arm's 9.79. The
// mechanism is fast and it is exact; what it puts on the W8A8 route is an activation the
// route's per-tensor input scale does not carry.
// [8 windows, Qwen2.5-1.5B, H96 MAX M9, 2026-08-11]
//
// The split machinery itself is NOT the cost, and that is a separate measurement rather
// than an assumption: forcing a K=1536 projection to run as 512+512+512, with ffn_down
// left on the CPU so the offloaded set is identical, scores 9.82 against the unsplit
// 9.90 — slightly BETTER, since each chunk gets its own activation scale.
//
// And more chunks is WORSE, not better: the same ffn_down cut ten ways (ROCKET_RK3576_
// KCHUNK=1024) scores 18.22 against four ways' 15.84, so each additional independently
// requantized partial in the sum costs something even though a synthetic four-chunk
// composition measured 0.997x of its single-chunk control. Neither the entry's saturation
// report nor its worst_rel_err fires here — the frozen per-column OUTPUT scale is not the
// term.
//
// With it off, an FFN down-projection is declined and goes to the CPU exactly as the port
// shipped, so =0 is also the control arm for every number the split is quoted from.
//
// ROCKET_RK3576_FORCESPLIT=1 cuts a K the part WOULD take whole, which is the arm that
// separates a defect in this code from a property of the route: split a K=1536 projection
// into 512+512+512 and its rotation is bit-identical (the block-diagonal fallback at
// K=1536 is H_512 over three aligned blocks, which is what three 512-chunks each take), so
// anything the model's output does under it is this code's doing. Use it with
// ROCKET_RK3576_KCHUNK, or the preference list simply returns the whole K again.
static bool rk76_ksplit(int Ktot, std::vector<int> * out) {
    if (out) out->clear();
    if (Ktot <= 0 || Ktot % 32 != 0) return false;

    static int split_on = -1, force = -1;
    if (split_on < 0) {
        const char * e = getenv("ROCKET_RK3576_KSPLIT");
        split_on = (e && atoi(e) != 0) ? 1 : 0;
        e = getenv("ROCKET_RK3576_FORCESPLIT");
        force = (e && atoi(e) != 0) ? 1 : 0;
    }
    if (!force && rk76_k_single_ok(Ktot)) { if (out) out->push_back(Ktot); return true; }
    if (!split_on && !rk76_k_single_ok(Ktot)) return false;

    static int pref0 = -1;
    if (pref0 < 0) {
        const char * e = getenv("ROCKET_RK3576_KCHUNK");
        const int v = e ? atoi(e) : 0;
        pref0 = rk76_k_single_ok(v) ? v : 0;
    }
    int rem = Ktot;
    for (size_t n = 0; rem > 0; n++) {
        int pick = 0;
        if (pref0 && pref0 <= rem) pick = pref0;
        for (size_t i = 0; !pick && i < sizeof(RK76_KCHUNK_PREF)/sizeof(int); i++)
            if (RK76_KCHUNK_PREF[i] <= rem) pick = RK76_KCHUNK_PREF[i];
        if (!pick || n >= RK76_KCHUNK_MAX) { if (out) out->clear(); return false; }
        if (out) out->push_back(pick);
        rem -= pick;
    }
    return true;
}

// A[M,K] f32 -> rotated int8 [M,K] + ONE per-tensor scale. The per-tensor scale is the
// measured choice (see above), and it is why this cannot call
// rocket_quant_act_int8_had(): that one writes a scale per row. `rot` is caller-owned
// [M*K] scratch — the rotation is the expensive half (~2 ms at M=512 K=2048) and is done
// once, not once per pass.
//
// `src_stride` is the source row pitch in floats and defaults to K; the K-split passes the
// full Ktot so chunk c can read columns [K0, K0+Kc) straight out of the activation. Each
// chunk gets its OWN per-tensor scale, which is strictly better than one shared over Ktot:
// a chunk whose activations are small no longer spends its codes on another chunk's range.
// `a_scale` is written M entries long. In the shipping per-TENSOR mode every entry holds
// the same value, so the dequantize below indexes it unconditionally and costs nothing.
// ROCKET_RK3576_AROW=1 gives each ROW its own scale instead, which is free at this
// interface — the entry takes A already quantized and its `scale` parameter is the
// OUTPUT's, so the row scale never crosses the ioctl and is applied in the dequantize.
// ROCKET_RK3576_AROW=2 asks for it only on the weights the K-split cut; the caller resolves
// that per weight and passes the resulting `per_row` here, so this function is unaware of it.
// It is OFF by default: per-axis input scales at this entry are a coin flip, better on
// every per-GEMM norm and 38x worse end to end on one model of two, and that measurement
// was taken on K=1536 projections rather than on `ffn_down`.
static int rk76_quant_act(const float * src, int8_t * dst, int64_t M, int64_t K,
                          float * a_scale, float * rot, int64_t src_stride = 0,
                          bool per_row = false) {
    float amax = 0.0f;
    if (src_stride <= 0) src_stride = K;
    for (int64_t m = 0; m < M; m++) {
        float * r = rot + m * K;
        for (int64_t k = 0; k < K; k++) r[k] = src[m * src_stride + k];
        if (rk76_rotate(r, (int)K) < 0) return -1;
        const float rmax = rk_amax_row(r, K);
        if (rmax > amax) amax = rmax;
        if (per_row) {
            const float s = (rmax > 0.0f) ? rmax / 127.0f : 1.0f, inv = 1.0f / s;
            rk_q8_row(r, dst + m * K, K, inv);
            a_scale[m] = s;
        }
    }
    if (per_row) return 0;
    const float s = (amax > 0.0f) ? amax / 127.0f : 1.0f, inv = 1.0f / s;
    rk_q8_row(rot, dst, M * K, inv);
    for (int64_t m = 0; m < M; m++) a_scale[m] = s;
    return 0;
}

// int8 C[M,N] -> f32 dst: dst[m,n] = C[m,n]/scale_n[n] * a_scale * b_scale[n].
// scale_n is what the entry was ASKED for; see (4) above for why not the achieved gain.
//
// `acc` adds into dst instead of writing it, which is how the K-split's chunks compose:
// the first chunk writes and the rest accumulate, so dst is never separately zeroed and
// the f32 sum of the partials is the same pass as the dequantize. The sum is in f32 and
// the partials are the part's own int8 output, so nothing here can overflow.
static void rk76_dequant(const int8_t * src, float * dst, int64_t M, int64_t N,
                         const float * a_scale, const float * b_scale, const float * scale_n,
                         uint64_t * sat_elems, bool acc, bool per_row_scale) {
    // The per-TENSOR case folds `a_scale` into `f[]` and multiplies once, exactly as the
    // path shipped. Do NOT "simplify" this into the per-row form with a[m]==a: fp32 is not
    // associative, and re-associating this one product moved the split arm's perplexity
    // 15.8395 -> 15.9894 with the arithmetic otherwise untouched. That is 0.4 of a paired
    // standard error and so not a real accuracy change, but it is a gratuitous one, and it
    // costs the shipped path its bit-for-bit reproducibility across builds — which is the
    // property every arm on this route is read against.
    std::vector<float> f((size_t)N);
    uint64_t sat = 0;
    if (!per_row_scale) {
        // The shipped path, expression-for-expression. Do NOT fold this into the per-row
        // loop below with a[m] == a_scale[0]: fp32 is not associative and `-ffp-contract`
        // turns `d += v*f[n]` into one FMA where `d += v*f[n]*a` cannot be, so the two
        // differ in the last bits. Re-associating it moved the split arm's perplexity
        // 15.8395 -> 15.9894 and then 15.7996 across two attempts to "simplify" it. That is
        // under half a paired standard error and so not an accuracy change, but it costs
        // the shipped path its bit-for-bit reproducibility across builds, which is the
        // property every arm on this route is read against.
        for (int64_t n = 0; n < N; n++) f[n] = a_scale[0] * b_scale[n] / scale_n[n];
        for (int64_t m = 0; m < M; m++) {
            const int8_t * srow = src + m * N;
            float * drow = dst + m * N;
            if (acc) {
                for (int64_t n = 0; n < N; n++) {
                    const int v = srow[n];
                    if (v >= 127 || v <= -127) sat++;
                    drow[n] += (float)v * f[n];
                }
            } else {
                for (int64_t n = 0; n < N; n++) {
                    const int v = srow[n];
                    if (v >= 127 || v <= -127) sat++;
                    drow[n] = (float)v * f[n];
                }
            }
        }
        if (sat_elems) *sat_elems += sat;
        return;
    }
    for (int64_t n = 0; n < N; n++) f[n] = b_scale[n] / scale_n[n];
    for (int64_t m = 0; m < M; m++) {
        const int8_t * srow = src + m * N;
        float * drow = dst + m * N;
        const float a = a_scale[m];
        if (acc) {
            for (int64_t n = 0; n < N; n++) {
                const int v = srow[n];
                if (v >= 127 || v <= -127) sat++;
                drow[n] += (float)v * a * f[n];
            }
        } else {
            for (int64_t n = 0; n < N; n++) {
                const int v = srow[n];
                if (v >= 127 || v <= -127) sat++;
                drow[n] = (float)v * a * f[n];
            }
        }
    }
    if (sat_elems) *sat_elems += sat;
}

// One bootstrap pass: run the entry at `scale_n` and read each column's largest |C8| back
// as an estimate of that column's accumulator magnitude. Returns <0 if the entry refused.
//
// This is how a caller gets the frozen colmax WITHOUT a host int32 GEMM. The accumulator
// never leaves the part, so the estimate comes from the part's own int8 output: pass one
// runs at 127/(128*sum_k|B[n][k]|), the analytic accumulator bound, which overshoots a
// real column by ~60x and therefore cannot saturate, and pass two runs at the first
// pass's readback times a margin so the codes are actually spent. A column that saturates
// in pass two keeps pass one's margined value, which over-estimates — the safe direction,
// since a frozen scale that is too TIGHT is what the tail measurement says costs a model.
//
// THE SCAN IS OVER A ROW-MAJOR SURFACE, AND THAT IS WHAT IT COSTS. `C8` is `[M][N]`, so
// taking a column's maximum with the column OUTERMOST reads `M` bytes a stride of `N`
// apart, across the whole surface: at `M`=2048 / `N`=8960 one column touches 2048 lines
// spanning 18.3 MB = 4480 pages, past any level of this part's TLB, so the loop pays a
// table walk an element. Taking `m` outermost against an `N`-wide running max reads the
// surface once, sequentially, and vectorizes. The arithmetic is identical — a column
// saturates exactly when its largest |code| reaches 127, so the flag falls out of the
// maximum.
//
// THE RUNNING MAX IS A BYTE, AND THAT IS THE OTHER HALF. `if (v > mx[n]) mx[n] = v;` is a
// conditional store and does not vectorize at all — it measures ~1.3-1.7 ns an element,
// a scalar rate. Written unconditionally it vectorizes and reaches 0.35-0.43; held in a
// `uint8_t` rather than an `int` it reaches 0.11-0.14, because a byte lane processes 16
// elements where an `int` lane processes 4. That last step is exact rather than an
// approximation: |code| over an int8 surface is 0-128 and fits a byte, the -128 rail
// included, and 128 >= 127 saturates just as it did. The win is flat in `N` — 3.67x at
// `N`=256 against 3.27x at 8960 — which is what a lane-count argument predicts and what
// an accumulator-residency one does not. [HW sweep, standalone, both -O2 and -O3]
//
// ROCKET_RK3576_CALSCAN selects the form and exists to price it: 0 column-outer, 1
// row-major with an `int` accumulator, 2 (default) row-major with a byte one.
// ROCKET_RK3576_WSA: hand the entry this weight's cached per-column sum of |qB| instead
// of letting it recompute one (default 1; 0 restores the recomputing entry, for pricing
// the pass). The two are bit-identical by construction — the frontend's integer IS what
// the library would sum — so this is a cost knob, not an accuracy one.
static int rk76_wsa(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("ROCKET_RK3576_WSA"); on = e ? atoi(e) != 0 : 1; }
    return on;
}

// ROCKET_RK3576_WDEV: once a weight's calibration converges, pack it into a device BO
// and hand the entry that object (rocket_matmul_int8_rk3576_perc_wbo) instead of the
// row-major qB, then DROP qB — the cube is the same N*Kc bytes, so device memory goes
// up by what host memory comes down. =0 restores the per-call pack for pricing. Created
// lazily at the first converged call rather than at cache build, because calibration
// still reads qB (the bootstrap operand and the calmap instrument) and an early create
// would double-hold every weight through calibration instead of one at a time.
static int rk76_wdev(void) {
    static int on = -1;
    if (on < 0) { const char * e = getenv("ROCKET_RK3576_WDEV"); on = e ? atoi(e) != 0 : 1; }
    return on;
}

static int rk76_bootstrap_pass(int fd, int M, int K, int N, const int8_t * qA,
                               const int8_t * qB, const float * scale_n,
                               const int64_t * sumabs,
                               int8_t * C8, double * est, char * satcol) {
    static int form = -1;
    if (form < 0) {
        const char * e = getenv("ROCKET_RK3576_CALSCAN");
        form = e ? atoi(e) : 2;
        if (form < 0 || form > 2) form = 2;
    }
    const int fe_prof = rk76_prof_on();
    double werr = 0.0;
    if (rocket_matmul_int8_rk3576_perc_sa(fd, M, K, N, qA, qB, NULL, scale_n,
                                          rk76_wsa() ? sumabs : NULL, C8, &werr) != 0)
        return -1;
    double fet0 = RK76_FE_T0();
    if (form == 2) {
        std::vector<unsigned char> mx((size_t)N, 0);
        unsigned char * __restrict mxp = mx.data();
        for (int m = 0; m < M; m++) {
            const int8_t * __restrict row = C8 + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int v = row[n];
                v = v < 0 ? -v : v;
                unsigned char u = (unsigned char)v;
                mxp[n] = u > mxp[n] ? u : mxp[n];
            }
        }
        for (int n = 0; n < N; n++) {
            if (satcol) satcol[n] = mxp[n] >= 127 ? 1 : 0;
            // A column reading back all-zero is one whose accumulator is under half a code
            // at THIS scale; credit it the code it would have taken, so no column is later
            // handed a zero scale and pass two still has something to refine.
            est[n] = (double)(mxp[n] < 1 ? 1 : mxp[n]) / (double)scale_n[n];
        }
    } else if (form == 1) {
        std::vector<int> mx((size_t)N, 0);
        for (int m = 0; m < M; m++) {
            const int8_t * row = C8 + (size_t)m * N;
            for (int n = 0; n < N; n++) {
                int v = row[n];
                if (v < 0) v = -v;
                if (v > mx[n]) mx[n] = v;
            }
        }
        for (int n = 0; n < N; n++) {
            if (satcol) satcol[n] = mx[n] >= 127 ? 1 : 0;
            est[n] = (double)(mx[n] < 1 ? 1 : mx[n]) / (double)scale_n[n];
        }
    } else {
        for (int n = 0; n < N; n++) {
            int mx = 0; bool sat = false;
            for (int m = 0; m < M; m++) {
                int v = C8[(size_t)m * N + n];
                if (v < 0) v = -v;
                if (v > mx) mx = v;
                if (v >= 127) sat = true;
            }
            if (mx < 1) mx = 1;
            est[n] = (double)mx / (double)scale_n[n];
            if (satcol) satcol[n] = sat ? 1 : 0;
        }
    }
    RK76_FE_ADD(calscan, fet0);
    return 0;
}

// A per-column readout of what the two bootstrap passes actually learn, off unless
// ROCKET_RK3576_CALMAP names a file. One row per column per calibration forward: the
// analytic accumulator bound pass one is scaled by, pass one's estimate, pass two's
// refined one, the saturation flag, and the two operands' second moments. That is enough
// to score any HOST-side predictor of the column maximum offline — the question being
// whether pass one, a whole W8A8 GEMM, can be replaced by one — and enough to say what the
// second calibration forward adds over the first. It is an instrument, not a path: it
// costs an O(M*Kc + N*Kc) sweep a forward and belongs nowhere near a timed arm.
static int rk76_calrows(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_RK3576_CALROWS"); v = e ? atoi(e) : 0; }
    return v;
}

static FILE * rk76_calmap_file(void) {
    static FILE * f = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
        const char * p = getenv("ROCKET_RK3576_CALMAP");
        if (p && *p) {
            f = fopen(p, "w");
            if (f) fprintf(f, "key,chunk,fwd,M,Kc,N,n,abound,est1,est2,sat,rmsA,rmsB\n");
        }
    }
    return f;
}

// W8A8 matmul for one plain 2D static-weight GEMM on the RK3576. Returns 0 (dst written)
// or <0 to fall through to the fp16 path. Caller checks K%32, N%32, 2D-ness and that a
// chunking exists (rk76_ksplit). A K past the library's own K >= 6176 refusal is cut into
// chunks on the HOST and the dequantized f32 partials are summed — see rocket_rk3576_chunk
// for why that is exact and what the cut costs.
static int ggml_backend_rocket_mul_mat_rk3576(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->rk76_failed) return -1;
    if (ctx->rk76_fd < 0) {
        ctx->rk76_fd = rocket_open();
        if (ctx->rk76_fd < 0) { ctx->rk76_failed = true; return -1; }
    }
    if (ctx->rk76_ncal < 0) {
        const char * e = getenv("ROCKET_RK3576_NCAL");   ctx->rk76_ncal = e ? atoi(e) : 2;
        // 3.0, not 2.0: the frozen colmax here is BOOTSTRAPPED, and a bootstrap's 0.3%
        // per-column error costs SmolLM2-1.7B 0.018 of ratio at safety 2.0 against 0.0021
        // at 3.0 — an 88% reduction — while the exactly-frozen arm moves only 0.0017
        // between the two, so 3.0 is inside that model's floor. Qwen2.5-1.5B's exact arm
        // is 1.008x at both. [host arithmetic; Qwen's BOOTSTRAPPED arm at 3.0 is
        // [expected] from its 0.000055 gap at 2.0, not measured]
        // 2 keeps the per-row scale only where the K-split cut the weight; see the use site.
        e = getenv("ROCKET_RK3576_AROW");      ctx->rk76_arow = e ? atoi(e) : 0;
        if (ctx->rk76_arow < 0 || ctx->rk76_arow > 2) ctx->rk76_arow = 1;
        e = getenv("ROCKET_RK3576_CALSAFE");   ctx->rk76_calsafe = e ? (float)atof(e) : 3.0f;
        e = getenv("ROCKET_RK3576_BOOTMARGIN"); ctx->rk76_bootmargin = e ? (float)atof(e) : 1.5f;
        if (ctx->rk76_calsafe <= 0.0f)    ctx->rk76_calsafe = 3.0f;
        if (ctx->rk76_bootmargin < 1.0f)  ctx->rk76_bootmargin = 1.5f;
    }
    // The rotation is the route, so a K no construction can express is a decline, not a
    // fallback to an unrotated run — that route is chaotic, not merely less accurate.
    // With the split that condition is per CHUNK, and rk76_ksplit enforces it.
    std::vector<int> kc;
    if (!rk76_ksplit(K, &kc)) return -1;
    rk_build_H60();

    // ---- weights: quantize+rotate ONCE per stable name, and hold the calibration state
    // beside them. Keyed on the weight NAME, as every other cache here is: src0->data is
    // reused by the scheduler across distinct weights.
    const bool b_f16 = (src0->type == GGML_TYPE_F16);
    const std::string key = rocket_weight_key(src0);
    auto it = key.empty() ? ctx->rk76_wcache.end() : ctx->rk76_wcache.find(key);
    if (it != ctx->rk76_wcache.end() && (it->second.N != N || it->second.K != K)) {
        // Same name, different shape: the record's calibration belongs to the old shape.
        ctx->rk76_resident_bytes -= it->second.bytes;
        for (auto & q : it->second.ch)
            if (q.wbo) { rocket_rk3576_wbo_free(ctx->rk76_fd, q.wbo); q.wbo = nullptr; }
        ctx->rk76_wcache.erase(it);
        it = ctx->rk76_wcache.end();
    }
    if (it == ctx->rk76_wcache.end()) {
        // The int8 weight is N*Ktot however it is cut; only the per-column vectors are
        // paid per chunk.
        const size_t est = (size_t)N * K
                         + kc.size() * (size_t)N * (3 * sizeof(float) + sizeof(double)
                                                    + sizeof(int64_t));
        if (key.empty() || (ctx->int8_cache_budget != 0
                            && ctx->rk76_resident_bytes + est > ctx->int8_cache_budget))
            return -1;   // don't re-quantize per call; this weight runs on the fp16 path
        rocket_rk3576_weight e;
        e.ch.resize(kc.size());
        {
            int kmax = 0;
            for (size_t c = 0; c < kc.size(); c++) if (kc[c] > kmax) kmax = kc[c];
            std::vector<float> tmp((size_t)kmax);
            int k0 = 0;
            for (size_t c = 0; c < kc.size(); c++) {
                rocket_rk3576_chunk & q = e.ch[c];
                q.K0 = k0; q.Kc = kc[c];
                q.qB.resize((size_t)N * q.Kc);
                q.b_scale.resize((size_t)N);
                // Chunk c is columns [K0, K0+Kc) of a [N][Ktot] weight, so the source row
                // pitch is Ktot and the base is offset by K0 ELEMENTS of the source type.
                const void * base = b_f16
                    ? (const void *)((const ggml_fp16_t *)src0->data + q.K0)
                    : (const void *)((const float       *)src0->data + q.K0);
                if (rocket_quant_wt_int8_had(base, b_f16, q.qB.data(), N, q.Kc,
                                             q.b_scale.data(), tmp.data(),
                                             rk76_rot_blocks(q.Kc), K) < 0) return -1;
                q.cbound.resize((size_t)N);
                q.sumabs.resize((size_t)N);
                q.colmax.assign((size_t)N, 0.0);
                q.scale_n.resize((size_t)N);
                for (int n = 0; n < N; n++) {
                    // |acc| <= 128 * sum_k |B[n][k]| for int8 A, and this is the same term
                    // the library's own C-ramp planner uses to cap the per-column
                    // multiplier. Held as a SCALE so the bootstrap can hand it straight
                    // over. It is a per-CHUNK bound, and tighter than the whole-K one by
                    // exactly the contraction it no longer covers.
                    // Accumulated as the INTEGER the library's ramp planner wants, then
                    // widened for cbound. |qB| is 0-128 and Kc <= 8960, so the sum is
                    // under 2^21 and the double conversion is exact — cbound is the same
                    // float either way.
                    int64_t s = 0;
                    const int8_t * row = q.qB.data() + (size_t)n * q.Kc;
                    for (int k = 0; k < q.Kc; k++) s += row[k] < 0 ? -(int64_t)row[k] : (int64_t)row[k];
                    q.sumabs[n] = s;
                    q.cbound[n] = (float)(127.0 / (128.0 * (double)s + 1.0));
                }
                k0 += q.Kc;
            }
        }
        e.cal_done = 0; e.N = N; e.K = K; e.bytes = est;
        e.sat_elems = 0; e.tot_elems = 0; e.sat_warned = false;
        it = ctx->rk76_wcache.emplace(key, std::move(e)).first;
        ctx->rk76_resident_bytes += est;
        if (rocket_debug_on()) {
            std::string cuts;
            for (size_t c = 0; c < kc.size(); c++)
                cuts += (c ? "+" : "") + std::to_string(kc[c]);
            GGML_LOG_DEBUG("[rk3576-cache] +%-22s N=%6d K=%6d (%zu chunk(s): %s)  "
                           "resident=%zuMB\n", key.c_str(), N, K, kc.size(), cuts.c_str(),
                           ctx->rk76_resident_bytes >> 20);
        }
    }
    rocket_rk3576_weight & w = it->second;

    // ROCKET_RK3576_AROW=2 scopes the per-row activation scale to the weights the K-split
    // actually cut. At =1 the dial is GLOBAL: it reaches every offloaded GEMM, and the split
    // only decides which GEMMs are offloaded at all. On Qwen2.5-1.5B 169 of the 197 offloaded
    // sites are taken whole [readout, 2026-08-13], and on those a per-row scale is a measured
    // net LOSS — +0.571% perplexity over 29 windows with the split off, against +0.026% for
    // the split the dial exists to protect. With the split ON it is mandatory (per-tensor
    // gives 15.84 against 9.93), so the sign is regime-dependent and =2 is the dial that
    // follows the regime.
    //
    // The decision is a property of the WEIGHT, not of a chunk — every chunk in one call
    // belongs to one weight — so this needs no second activation quantization: the loop
    // below already quantizes each chunk's slab separately with its own scale, and the
    // calibration forwards take the same flag, so the frozen scales match the mode.
    const bool arow = (ctx->rk76_arow == 2) ? (w.ch.size() > 1) : (ctx->rk76_arow != 0);

    // ---- scratch, sized for the WIDEST chunk and reused by all of them. One allocation a
    // call however K is cut: the three vectors are value-initialized, so sizing them per
    // chunk would memset M*Ktot + M*Ktot + M*N bytes across the loop instead of
    // M*Kmax + M*Kmax + M*N once.
    const int fe_prof = rk76_prof_on();
    int kmax = 0;
    for (size_t c = 0; c < w.ch.size(); c++) if (w.ch[c].Kc > kmax) kmax = w.ch[c].Kc;
    double fet0 = RK76_FE_T0();
    std::vector<float>  rot((size_t)M * kmax);
    std::vector<int8_t> qA((size_t)M * kmax);
    std::vector<int8_t> C8((size_t)M * N);
    std::vector<float>  a_scale((size_t)M, 1.0f);
    RK76_FE_ADD(valloc, fet0);
    if (fe_prof) g_rk76_fe.calls++;

    const bool calibrating = w.cal_done < ctx->rk76_ncal;
    if (calibrating && fe_prof) g_rk76_fe.cal_calls++;
    std::vector<double> est1, est2;
    std::vector<char>   sat;
    std::vector<float>  sc2;
    if (calibrating) {
        est1.resize((size_t)N); est2.resize((size_t)N);
        sat.assign((size_t)N, 0);   // char, not bool: &v[0] must be writable
        sc2.resize((size_t)N);
    }

    // ---- one whole W8A8 GEMM per chunk, summed on the host. For an unsplit K this loop
    // runs once and is the path that shipped.
    for (size_t c = 0; c < w.ch.size(); c++) {
        rocket_rk3576_chunk & q = w.ch[c];

        // activations: chunk c's columns, rotated and quantized, every call
        fet0 = RK76_FE_T0();
        if (rk76_quant_act((const float *)src1->data + q.K0, qA.data(), M, q.Kc,
                           a_scale.data(), rot.data(), K, arow) < 0)
            return -1;
        RK76_FE_ADD(quant, fet0);

        if (calibrating) {
            // ---- calibration forward: two device passes to estimate this window's
            // per-column accumulator maxima, then accumulate the running max. The SECOND
            // pass's surface is the one handed back — it is a correct output at a scale
            // ~BOOTMARGIN loose, which costs resolution and nothing else. Each chunk
            // bootstraps its OWN columns: its accumulator is over its own contraction, so
            // a colmax frozen from another chunk's would be the wrong quantity.
            fet0 = RK76_FE_T0();
            // ROCKET_RK3576_CALROWS caps pass ONE's row count. It is an instrument, not a
            // shipped policy: pass one's estimate is a maximum over `M` accumulator rows,
            // so cutting the rows cuts the device work and undershoots the maximum, and
            // the arm is whether the undershoot composes with the readback quantum past
            // BOOTMARGIN. Pass two always runs at full `M` -- it is the accurate one and
            // its surface is what the call returns.
            int M1 = M;
            if (rk76_calrows() > 0 && rk76_calrows() < M) M1 = rk76_calrows();
            if (rk76_bootstrap_pass(ctx->rk76_fd, M1, q.Kc, N, qA.data(), q.qB.data(),
                                    q.cbound.data(), q.sumabs.data(),
                                    C8.data(), est1.data(), NULL) < 0)
                return -1;
            for (int n = 0; n < N; n++)
                sc2[n] = (float)(127.0 / (est1[n] * (double)ctx->rk76_bootmargin));
            if (rk76_bootstrap_pass(ctx->rk76_fd, M, q.Kc, N, qA.data(), q.qB.data(),
                                    sc2.data(), q.sumabs.data(),
                                    C8.data(), est2.data(), sat.data()) < 0)
                return -1;
            for (int n = 0; n < N; n++) {
                const double v = sat[n] ? est1[n] * (double)ctx->rk76_bootmargin : est2[n];
                if (v > q.colmax[n]) q.colmax[n] = v;
            }
            RK76_FE_ADD(cal, fet0);
            if (FILE * cf = rk76_calmap_file()) {
                double sa = 0.0;
                const size_t na = (size_t)M * (size_t)q.Kc;
                for (size_t i = 0; i < na; i++) { const double v = qA[i]; sa += v * v; }
                const double rmsA = sqrt(sa / (double)na);
                for (int n = 0; n < N; n++) {
                    double sb = 0.0;
                    const int8_t * row = q.qB.data() + (size_t)n * q.Kc;
                    for (int k = 0; k < q.Kc; k++) { const double v = row[k]; sb += v * v; }
                    fprintf(cf, "%s,%zu,%d,%d,%d,%d,%d,%.7g,%.7g,%.7g,%d,%.6g,%.6g\n",
                            key.c_str(), c, w.cal_done, M, q.Kc, N, n,
                            127.0 / (double)q.cbound[n], est1[n], est2[n], (int)sat[n],
                            rmsA, sqrt(sb / (double)q.Kc));
                }
                fflush(cf);
            }
            fet0 = RK76_FE_T0();
            rk76_dequant(C8.data(), (float *)dst->data, M, N, a_scale.data(),
                         q.b_scale.data(), sc2.data(), NULL, c != 0, arow);
            RK76_FE_ADD(dequant, fet0);
            continue;
        }

        // The device weight cache: create at the first converged call, then drop qB.
        // On failure, latch off for the whole context — a board that refused one cube
        // should not be asked to hold more — and stay on the per-call path, which is
        // still there because qB was not dropped.
        if (rk76_wdev() && !ctx->rk76_wdev_off && !q.wbo) {
            if (rocket_rk3576_wbo_create(ctx->rk76_fd, q.Kc, N, q.qB.data(),
                                         &q.wbo) == 0) {
                q.qB.clear();
                q.qB.shrink_to_fit();
            } else {
                ctx->rk76_wdev_off = true;
                ROCKET_LOGW("[rk3576] device weight cache create failed at N=%d Kc=%d "
                            "(device memory, most likely) -- staying on the per-call "
                            "weight pack from here on\n", N, q.Kc);
            }
        }

        double werr = 0.0;
        fet0 = RK76_FE_T0();
        int mmrc = q.wbo
            ? rocket_matmul_int8_rk3576_perc_wbo(ctx->rk76_fd, M, q.Kc, N, qA.data(),
                                                 q.wbo, NULL, q.scale_n.data(),
                                                 q.sumabs.data(), C8.data(), &werr)
            : rocket_matmul_int8_rk3576_perc_sa(ctx->rk76_fd, M, q.Kc, N, qA.data(),
                                                q.qB.data(), NULL, q.scale_n.data(),
                                                rk76_wsa() ? q.sumabs.data() : NULL,
                                                C8.data(), &werr);
        if (mmrc != 0)
            return -1;
        RK76_FE_ADD(entry, fet0);
        fet0 = RK76_FE_T0();
        rk76_dequant(C8.data(), (float *)dst->data, M, N, a_scale.data(),
                     q.b_scale.data(), q.scale_n.data(), &w.sat_elems, c != 0, arow);
        RK76_FE_ADD(dequant, fet0);
        w.tot_elems += (uint64_t)M * N;
    }

    if (calibrating) {
        w.cal_done++;
        if (w.cal_done == ctx->rk76_ncal) {
            for (size_t c = 0; c < w.ch.size(); c++) {
                rocket_rk3576_chunk & q = w.ch[c];
                for (int n = 0; n < N; n++) {
                    const double cm = q.colmax[n] > 0.0 ? q.colmax[n] : 1.0;
                    q.scale_n[n] = (float)(127.0 / (cm * (double)ctx->rk76_calsafe));
                }
            }
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[rk3576-cal] %-22s frozen over %d forward(s), "
                               "safety %.2f\n", key.c_str(), w.cal_done,
                               (double)ctx->rk76_calsafe);
        }
        return 0;
    }
    // The policy for a call whose accumulator exceeds the frozen scale: COUNT it and say
    // so once. Re-freezing here would make the model's output depend on how many tokens
    // preceded it, and widening only the offending columns re-runs the call. The
    // measurement behind the threshold: at safety 2.0 the saturating fraction of the
    // surface is 0.000-0.004% on both models, against 0.180% at safety 1.0 — so a rate
    // this side of a tenth of a percent means the calibration set did not cover the
    // activations it is being asked about, and the lever is a larger safety factor.
    if (!w.sat_warned && w.tot_elems >= (uint64_t)1 << 20
        && w.sat_elems * 1000 > w.tot_elems) {
        w.sat_warned = true;
        ROCKET_LOGI("[rocket] %s: %.3f%% of the RK3576 W8A8 output saturates against a "
                    "scale frozen over %d forward(s); raise ROCKET_RK3576_CALSAFE "
                    "(now %.2f) or ROCKET_RK3576_NCAL (now %d)\n",
                    key.c_str(), 100.0 * (double)w.sat_elems / (double)w.tot_elems,
                    ctx->rk76_ncal, (double)ctx->rk76_calsafe, ctx->rk76_ncal);
    }
    return 0;
}

// W4A4 int4 matmul for one plain 2D static-weight GEMM, via the native int4 driver.
// The int4 sibling of mul_mat_int8: per-channel-quantized (cached, rotated when
// hadamard) int4 weight + per-row int4 activations -> int4xint4->int16 NPU matmul
// (host-accumulated to int32) -> dequant. Returns 0 (dst written) or <0 to fall
// through to the fp16 path. Caller checks N%64 (int4 weight k-group) + K%32 + 2D-ness.
//
// SATURATION: the NPU int4 output is read back as int16 per K-tile; [-7,7] inputs make
// |partial| grow at <=49/elem, so an uncapped K-tile (the plan can pick K, single-pass)
// overflows int16 for large K. The driver entry rocket_matmul_int4_ex caps Kt to its
// kt_cap argument (rounded to %32); the call below passes 480 (49*480 < 32767), making the
// int16 partial saturation-safe for any K. This does NOT use ROCKET_MM_KT (that env knob
// can only lower Kt further, never raise it past this baked-in cap).
static int ggml_backend_rocket_mul_mat_int4(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->int4_failed) return -1;
    if (ctx->int4_fd < 0) {
        ctx->int4_fd = rocket_open();
        if (ctx->int4_fd < 0) { ctx->int4_failed = true; return -1; }
    }

    const int Mp = rocket_pad_m(M);   // driver needs M%4; pad rows quantize to 0
    const bool b_f16 = (src0->type == GGML_TYPE_F16);

    // Hadamard defaults ON for int4 (unlike int8's opt-in): W4A4 is unusable without
    // the outlier rotation (cos ~0.79 -> ~0.98). ROCKET_INT4_HADAMARD=0 disables it.
    if (ctx->int4_hadamard < 0) {
        const char * e = getenv("ROCKET_INT4_HADAMARD"); ctx->int4_hadamard = e ? atoi(e) : 1;
    }
    const bool had = ctx->int4_hadamard && rk_hadamard_ok(K);
    if (had) rk_build_H60();

    // ROCKET_INT4_GROUP=g: per-K-group dequant scales (the W4 quality lever). g must
    // divide K, be %32, and keep the int16 partial unsaturated (49*g < 32767). An
    // invalid g silently degrades to per-channel (group 0). nG scale slots per row.
    // Default 128 = group-wise (best W4 quality + the group forces a saturation-safe
    // Kt). ROCKET_INT4_GROUP=0 selects per-channel; an invalid g (doesn't divide K,
    // not %32, or would saturate int16) silently degrades to per-channel.
    if (ctx->int4_group < 0) { const char * e = getenv("ROCKET_INT4_GROUP"); ctx->int4_group = e ? atoi(e) : 128; }
    int group = ctx->int4_group;
    if (group > 0 && (group % 32 || K % group || 49 * group >= 32767)) group = 0;
    const int nG = group > 0 ? (int)(K / group) : 1;

    // ---- weights: reuse the resident rotated-int4 weight if cached, else build it
    // ONCE (per-channel OR per-group quant, pre-rotated when hadamard) and cache.
    // Keyed on the stable weight NAME (the wcache-aliasing reasoning applies here too).
    const int8_t * qB = nullptr;
    const float  * b_scale = nullptr;
    const std::string key = rocket_weight_key(src0);

    auto it = key.empty() ? ctx->int4_wcache.end() : ctx->int4_wcache.find(key);
    if (it != ctx->int4_wcache.end()
        && it->second.N == N && it->second.K == K && it->second.group == group
        && it->second.hadamard == had) {
        qB = it->second.qB.data();
        b_scale = it->second.b_scale.data();
    } else {
        // Build+cache only within budget; else this weight runs on the fp16 path
        // (don't re-quantize per call -- thrashes a memory-tight box). int4 mode then
        // degrades to "int4 for what fits, fp16 for the rest", usable and thrash-free.
        const size_t est = (size_t)N * K + (size_t)N * nG * sizeof(float);
        if (key.empty() || (ctx->int4_cache_budget != 0
                            && ctx->int4_resident_bytes + est > ctx->int4_cache_budget))
            return -1;

        std::vector<int8_t> bq((size_t)N * K);
        std::vector<float>  bs((size_t)N * nG);
        if (group > 0) { std::vector<float> tmp((size_t)K);
                   if (rocket_quant_int4_grouped(nullptr, src0->data, false, b_f16, bq.data(), N, K, group, had, bs.data(), tmp.data()) < 0) return -1; }
        else if (had) { std::vector<float> tmp((size_t)K);
                   if (rocket_quant_wt_int4_had(src0->data, b_f16, bq.data(), N, K, bs.data(), tmp.data()) < 0) return -1; }
        else     { rocket_quant_wt_int4(src0->data, b_f16, bq.data(), N, K, bs.data()); }

        rocket_int4_weight e;
        e.qB = std::move(bq); e.b_scale = std::move(bs);
        e.N = N; e.K = K; e.group = group; e.hadamard = had; e.bytes = est;
        auto & slot = (ctx->int4_wcache[key] = std::move(e));
        ctx->int4_resident_bytes += est;
        qB = slot.qB.data(); b_scale = slot.b_scale.data();
        if (rocket_debug_on())
            GGML_LOG_DEBUG("[int4-cache] +%-22s N=%6d K=%6d had=%d grp=%d  resident=%zuMB\n",
                    key.c_str(), N, K, (int)had, group, ctx->int4_resident_bytes >> 20);
    }

    // ---- activations: always per call (cheap; M is small), int4 [Mp,K] (pad rows = 0)
    std::vector<int8_t> qA((size_t)Mp * K, 0);
    if (group > 0) {
        // group-wise: per-(row,group) scales, fp32-accumulating matmul applies them.
        std::vector<float> a_scale((size_t)Mp * nG, 1.0f);
        std::vector<float> tmp((size_t)K);
        if (rocket_quant_int4_grouped((const float *)src1->data, nullptr, true, false,
                                      qA.data(), M, K, group, had, a_scale.data(), tmp.data()) < 0) return -1;
        std::vector<float> Cf((size_t)Mp * N);
        int rc = rocket_matmul_int4_groupwise(ctx->int4_fd, Mp, K, N, qA.data(), qB,
                                              a_scale.data(), b_scale, Cf.data(), group);
        if (rc != 0) return -1;
        memcpy((float *)dst->data, Cf.data(), (size_t)M * N * sizeof(float));  // first M rows
        return 0;
    }

    std::vector<float>  a_scale((size_t)Mp, 1.0f);
    if (had) { std::vector<float> tmp((size_t)K);
               if (rocket_quant_act_int4_had((const float *)src1->data, qA.data(), M, K, a_scale.data(), tmp.data()) < 0) return -1; }
    else     { rocket_quant_act_int4((const float *)src1->data, qA.data(), M, K, a_scale.data()); }

    std::vector<int32_t> C32((size_t)Mp * N);
    // kt_cap=480: [-7,7] int4 -> |K-tile partial| <= 49*480 < 32767, so the int16
    // output cannot saturate regardless of K (safe without ROCKET_MM_KT).
    int rc = rocket_matmul_int4_ex(ctx->int4_fd, Mp, K, N, qA.data(), qB, C32.data(), 480);
    if (rc != 0) return -1;

    rocket_dequant_int8(C32.data(), (float *)dst->data, M, N, a_scale.data(), b_scale);
    return 0;
}

// bf16 x bf16 -> fp32 path. The PAYOFF of the bf16 datatype: because
// bf16 carries fp32's exponent range (HW-proven: matmul_bf16_rocket passes at
// |values|>fp16's 65504), activations need NO per-row amax-scan/scale -- the hack
// rocket_pack_activations does only because fp16's 5-bit exponent overflows on
// Gemma activations. So this path hands the F32 activations straight to the bf16
// matmul (truncated to bf16 inside, UNSCALED) and writes the fp32 result directly
// to dst -- no scale, no unscale. Weights (F16) are converted to fp32 per call into
// a reused scratch (the bf16 matmul truncates them to bf16). The matmul runs on the
// streaming context (persistent worker fds + per-shape resident scratch BOs, multicore
// N-split), so a bf16 prefill is a multicore/resident path, not a single-fd one.
// Returns 0 (dst written) or <0 to fall through to fp16.
static int ggml_backend_rocket_mul_mat_bf16(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K] (BF16 / F16 / F32)
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K] (F32)

    if (ctx->bf16_failed) return -1;
    const int  Mp    = rocket_pad_m(M);         // driver needs M%4; pad rows = 0
    if (rocket_matmul_plan_bf16(Mp, K, N, nullptr, nullptr, nullptr) < 0)
        return -1;                              // unsupported shape (K%32/N%16) -> fp16 path
    // Prefer the streaming context (persistent fds + resident scratch + multicore); created
    // once on the first bf16 op. Fall back to the single-fd path if create fails.
    if (!ctx->bf16_stream && !ctx->bf16_stream_failed) {
        ctx->bf16_stream = rocket_bf16_stream_create(ctx->n_threads);
        if (!ctx->bf16_stream) ctx->bf16_stream_failed = true;
    }
    if (!ctx->bf16_stream && ctx->bf16_fd < 0) {
        ctx->bf16_fd = rocket_open();
        if (ctx->bf16_fd < 0) { ctx->bf16_failed = true; return -1; }
    }

    // weights -> fp32 scratch (reused); the bf16 matmul truncates fp32->bf16. Source
    // may be native BF16 (the real model weights -> exact bf16->fp32), F16 (a fp16
    // gguf, re-truncated to bf16), or F32.
    // Reused fp32 weight scratch. It grows to the high-water N*K and is never shrunk:
    // the bf16 path is a slow single-fd fallback, so holding the peak buffer (vs
    // realloc churn across calls) is the right trade. It frees with the ctx.
    std::vector<float> & Bf = ctx->bf16_bscratch;
    if (Bf.size() < (size_t)N * K) Bf.resize((size_t)N * K);
    if      (src0->type == GGML_TYPE_BF16) ggml_bf16_to_fp32_row((const ggml_bf16_t *)src0->data, Bf.data(), (size_t)N * K);
    else if (src0->type == GGML_TYPE_F16)  ggml_fp16_to_fp32_row((const ggml_fp16_t *)src0->data, Bf.data(), (size_t)N * K);
    else                                   memcpy(Bf.data(), src0->data, (size_t)N * K * sizeof(float));

    // activations: F32 straight through, NO scaling. Pad rows M..Mp-1 with zeros.
    const float * Af;
    std::vector<float> Apad;
    if (Mp == M) {
        Af = (const float *)src1->data;
    } else {
        Apad.assign((size_t)Mp * K, 0.0f);
        memcpy(Apad.data(), src1->data, (size_t)M * K * sizeof(float));
        Af = Apad.data();
    }

    // output fp32: write straight to dst when unpadded (dst is [M,N] contiguous F32,
    // guaranteed by the supports_op contiguity check); else via a padded buffer.
    float * Cf;
    std::vector<float> Cpad;
    if (Mp == M) {
        Cf = (float *)dst->data;
    } else {
        Cpad.resize((size_t)Mp * N);
        Cf = Cpad.data();
    }

    int rc = ctx->bf16_stream
        ? rocket_matmul_bf16_stream(ctx->bf16_stream, Mp, K, N, Af, Bf.data(), Cf)
        : rocket_matmul_bf16(ctx->bf16_fd, Mp, K, N, Af, Bf.data(), Cf);
    if (rc != 0) return -1;

    if (Mp != M) memcpy(dst->data, Cpad.data(), (size_t)M * N * sizeof(float));
    return 0;
}

// ===========================================================================
// Resident int8 / int4 matmul paths (weight held in NPU tile BOs)
// ===========================================================================
// RESIDENT int8 path. Defined later (the madvise reclaim, reused here to drop
// the F16 source once the int8 tiles are resident).
static bool   rocket_prepack_madvise_on(void);
static size_t rocket_madvise_dontneed(const void * ptr, size_t len);

// W8A8 int8 matmul with the weight held RESIDENT in NPU int8 tile BOs,
// the int8 sibling of ggml_backend_rocket_mul_mat_prepacked. The weight is
// quantized+rotated to host int8 ONCE, scattered into the resident rocket_i8_ctx
// BOs, and the host copy dropped; every later pass reuses the resident tiles and
// only quantizes A. Returns 0 (dst written) or <0 to fall through to the fp16
// path (over budget / IOVA full / uncacheable -- all SAFE: the F16 source of a
// weight that did NOT go resident is never madvise-dropped).
static int ggml_backend_rocket_mul_mat_int8_resident(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->i8_dev_failed) return -1;
    if (!ctx->i8_dev) {
        ctx->i8_dev = rocket_i8_ctx_create(ctx->n_threads);
        if (!ctx->i8_dev) { ctx->i8_dev_failed = true; return -1; }
    }

    const int  Mp    = rocket_pad_m(M);
    const bool b_f16 = (src0->type == GGML_TYPE_F16);
    if (ctx->int8_hadamard < 0) {
        const char * e = getenv("ROCKET_INT8_HADAMARD"); ctx->int8_hadamard = e ? atoi(e) : 0;
    }
    const bool had = ctx->int8_hadamard && rk_hadamard_ok(K);
    if (had) rk_build_H60();

    const std::string key = rocket_weight_key(src0);
    if (key.empty()) return -1;               // no stable identity -> fp16 path

    // ---- weights: reuse the resident rotated-int8 weight if cached, else build
    // it ONCE (quant + rotate -> host int8 -> scatter into resident NPU tiles ->
    // drop host copy). Keyed on NAME + hadamard. The resident int8 tile layout is
    // M-INDEPENDENT (canonical-tileM), so a weight packed at the warmup M is reused at
    // any prefill M with NO re-pack — killing the short-prompt re-pack stall.
    rocket_i8_weights * w = nullptr;
    const float * b_scale = nullptr;
    auto it = ctx->i8_rwcache.find(key);
    if (it != ctx->i8_rwcache.end()) {
        if (it->second.N == N && it->second.K == K
            && it->second.hadamard == had) {
            w = it->second.w; b_scale = it->second.b_scale.data();   // reuse across M
        } else {                              // genuine shape/hadamard change: re-pack
            rocket_i8_weights_free(ctx->i8_dev, it->second.w);
            ctx->i8_resident_bytes -= it->second.bytes;
            ctx->i8_rwcache.erase(it);
        }
    }
    if (!w && ctx->i8_resident_full) return -1;
    if (!w) {
        // Pre-check the budget against an ESTIMATE of the resident tile footprint
        // (~N*K int8 tiles + per-N scales). The real resident BO bytes are queried
        // from the packed weight after the fact (rocket_i8_weights_bytes) and used
        // for the running total, so the cap tracks the true NPU-BO footprint.
        const size_t est = (size_t)N * K + (size_t)N * sizeof(float);
        if (ctx->int8_cache_budget != 0
            && ctx->i8_resident_bytes + est > ctx->int8_cache_budget)
            return -1;                        // over budget -> fp16 streams this weight

        // quantize (+rotate) the weight to host int8 ONCE; the resident pack
        // scatters it into NPU tiles, after which the host vectors are dropped.
        std::vector<int8_t> bq((size_t)N * K);
        std::vector<float>  bs((size_t)N);
        if (had) { std::vector<float> tmp((size_t)K);
                   if (rocket_quant_wt_int8_had(src0->data, b_f16, bq.data(), N, K, bs.data(), tmp.data()) < 0) return -1; }
        else     { rocket_quant_wt_int8(src0->data, b_f16, bq.data(), N, K, bs.data()); }

        rocket_i8_weights * rw = rocket_i8_weights_pack(ctx->i8_dev, Mp, K, N, bq.data());
        if (!rw) {                            // IOVA/alloc exhausted: latch + stream the rest
            ctx->i8_resident_full = true;
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[int8-resident] IOVA window full at resident=%zuMB -> fp16 remainder\n",
                        ctx->i8_resident_bytes >> 20);
            return -1;
        }

        const size_t resident_bytes = rocket_i8_weights_bytes(rw);
        rocket_i8_resident e;
        e.w = rw; e.b_scale = std::move(bs);
        e.Mp = Mp; e.N = N; e.K = K; e.hadamard = had; e.bytes = resident_bytes;
        // rw holds resident NPU BOs (rocket_i8_resident stores it as a raw pointer with
        // no destructor); free it if the cache insert throws before the map owns it.
        rocketraii::scope_guard rw_guard([&] { rocket_i8_weights_free(ctx->i8_dev, rw); });
        auto & slot = (ctx->i8_rwcache[key] = std::move(e));
        rw_guard.dismiss();
        ctx->i8_resident_bytes += resident_bytes;
        w = slot.w; b_scale = slot.b_scale.data();
        if (rocket_debug_on())
            GGML_LOG_DEBUG("[int8-resident] +%-22s N=%6d K=%6d had=%d  resident=%zuMB\n",
                    key.c_str(), N, K, (int)had, ctx->i8_resident_bytes >> 20);

        // madvise reclaim (PREFILL-ONLY): the int8 tiles are now resident, so
        // reclaim the row-major F16 source. Done ONLY on the build path (a weight
        // that went resident), so a weight that fell back to fp16 keeps its source.
        if (rocket_prepack_madvise_on()) {
            const size_t dropped = rocket_madvise_dontneed(src0->data, ggml_nbytes(src0));
            ctx->madvised_bytes += dropped;
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[madvise] -%-24s %4zuMB dropped (reclaimed %zuMB, resident %zuMB)\n",
                        key.c_str(), dropped >> 20, ctx->madvised_bytes >> 20,
                        ctx->i8_resident_bytes >> 20);
        }
    }

    // ---- activations: per call (rotated when hadamard), int8 [Mp,K] (pad rows = 0)
    std::vector<int8_t> qA((size_t)Mp * K, 0);
    std::vector<float>  a_scale((size_t)Mp, 1.0f);
    if (had) { std::vector<float> tmp((size_t)K);
               if (rocket_quant_act_int8_had((const float *)src1->data, qA.data(), M, K, a_scale.data(), tmp.data()) < 0) return -1; }
    else     { rocket_quant_act_int8((const float *)src1->data, qA.data(), M, K, a_scale.data()); }

    std::vector<int32_t> C32((size_t)Mp * N);
    int rc = rocket_matmul_int8_prepacked(ctx->i8_dev, Mp, K, N, qA.data(), C32.data(), w);
    if (rc != 0) return -1;

    rocket_dequant_int8(C32.data(), (float *)dst->data, M, N, a_scale.data(), b_scale);
    return 0;
}

// W4A4 int4 matmul with the weight held RESIDENT in NPU int4 nibble BOs, the int4
// sibling of ggml_backend_rocket_mul_mat_int8_resident. The weight is group-wise
// quantized (+Hadamard-rotated) to host int4 ONCE, scattered into the resident
// rocket_i4_ctx BOs (~1/4 the fp16 footprint), and the host copy dropped; every later
// pass reuses the resident nibble tiles and only quantizes A. Group-wise ONLY (the
// K-tile is forced to `group`, which is both the W4 quality lever and the int16-
// saturation-safe tile size). Returns 0 (dst written) or <0 to fall through to the
// one-shot int4 path (per-channel / over budget / IOVA full / uncacheable -- all SAFE).
static int ggml_backend_rocket_mul_mat_int4_resident(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->i4_dev_failed) return -1;
    if (!ctx->i4_dev) {
        ctx->i4_dev = rocket_i4_ctx_create(ctx->n_threads);
        if (!ctx->i4_dev) { ctx->i4_dev_failed = true; return -1; }
    }

    const int  Mp    = rocket_pad_m(M);
    const bool b_f16 = (src0->type == GGML_TYPE_F16);

    // Hadamard defaults ON for int4 (W4A4 is unusable without the outlier rotation).
    if (ctx->int4_hadamard < 0) { const char * e = getenv("ROCKET_INT4_HADAMARD"); ctx->int4_hadamard = e ? atoi(e) : 1; }
    const bool had = ctx->int4_hadamard && rk_hadamard_ok(K);
    if (had) rk_build_H60();

    // Group-wise only: the resident path forces Kt=group (one tile == one quant group,
    // saturation-safe). group==0 / invalid -> one-shot int4 path (its kt_cap handles
    // per-channel). 49*group<32767 keeps the int16 partial unsaturated.
    if (ctx->int4_group < 0) { const char * e = getenv("ROCKET_INT4_GROUP"); ctx->int4_group = e ? atoi(e) : 128; }
    const int group = ctx->int4_group;
    if (group <= 0 || group % 32 || K % group || 49 * group >= 32767) return -1;
    const int nG = (int)(K / group);

    const std::string key = rocket_weight_key(src0);
    if (key.empty()) return -1;               // no stable identity -> one-shot path

    // ---- weights: reuse the resident rotated-int4 weight if cached, else build it
    // ONCE (group-wise quant + rotate -> host int4 -> scatter into resident nibble
    // BOs -> drop host copy). Keyed on NAME + group + hadamard. The resident int4 tile
    // layout is M-INDEPENDENT (canonical-tileM), so a weight packed at the warmup M is
    // reused at any prefill M with NO re-pack — this kills the short-prompt re-pack stall
    // (warmup M=512 then a small-M prefill no longer re-builds the whole int4 model).
    rocket_i4_weights * w = nullptr;
    const float * b_scale = nullptr;
    auto it = ctx->i4_rwcache.find(key);
    if (it != ctx->i4_rwcache.end()) {
        if (it->second.N == N && it->second.K == K
            && it->second.group == group && it->second.hadamard == had) {
            w = it->second.w; b_scale = it->second.b_scale.data();   // reuse across M
        } else {                              // genuine shape/group/hadamard change: re-pack
            rocket_i4_weights_free(ctx->i4_dev, it->second.w);
            ctx->i4_resident_bytes -= it->second.bytes;
            ctx->i4_rwcache.erase(it);
        }
    }
    if (!w && ctx->i4_resident_full) return -1;
    if (!w) {
        // Budget pre-check (nibble-packed ~ N*K/2 + per-group scales). The true
        // resident BO bytes come from rocket_i4_weights_bytes after packing.
        const size_t est = (size_t)N * K / 2 + (size_t)N * nG * sizeof(float);
        if (ctx->int4_cache_budget != 0 && ctx->i4_resident_bytes + est > ctx->int4_cache_budget)
            return -1;                        // over budget -> one-shot int4 for this weight

        std::vector<int8_t> bq((size_t)N * K);
        std::vector<float>  bs((size_t)N * nG);
        { std::vector<float> tmp((size_t)K);
          if (rocket_quant_int4_grouped(nullptr, src0->data, false, b_f16, bq.data(),
                                        N, K, group, had, bs.data(), tmp.data()) < 0) return -1; }

        rocket_i4_weights * rw = rocket_i4_weights_pack_gw(ctx->i4_dev, Mp, K, N, bq.data(), group);
        if (!rw) {                            // IOVA/alloc exhausted: latch + one-shot the rest
            ctx->i4_resident_full = true;
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[int4-resident] IOVA window full at resident=%zuMB -> one-shot remainder\n",
                        ctx->i4_resident_bytes >> 20);
            return -1;
        }

        const size_t resident_bytes = rocket_i4_weights_bytes(rw);
        rocket_i4_resident e;
        e.w = rw; e.b_scale = std::move(bs);
        e.Mp = Mp; e.N = N; e.K = K; e.group = group; e.hadamard = had; e.bytes = resident_bytes;
        // rw holds resident NPU nibble BOs (stored as a raw pointer, no destructor);
        // free it if the cache insert throws before the map takes ownership.
        rocketraii::scope_guard rw_guard([&] { rocket_i4_weights_free(ctx->i4_dev, rw); });
        auto & slot = (ctx->i4_rwcache[key] = std::move(e));
        rw_guard.dismiss();
        ctx->i4_resident_bytes += resident_bytes;
        w = slot.w; b_scale = slot.b_scale.data();
        if (rocket_debug_on())
            GGML_LOG_DEBUG("[int4-resident] +%-22s N=%6d K=%6d had=%d grp=%d  resident=%zuMB\n",
                    key.c_str(), N, K, (int)had, group, ctx->i4_resident_bytes >> 20);

        // madvise reclaim (PREFILL-ONLY, opt-in): the int4 tiles are resident, so the
        // row-major F16 source can be dropped -- but that breaks CPU decode (which reads
        // F16), so it stays behind ROCKET_PREPACK_MADVISE like the int8 path.
        if (rocket_prepack_madvise_on()) {
            const size_t dropped = rocket_madvise_dontneed(src0->data, ggml_nbytes(src0));
            ctx->madvised_bytes += dropped;
        }
    }

    // ---- activations: per call, group-wise int4 [Mp,K] (pad rows = 0), rotated when
    // hadamard. a_scale is [Mp*nG]; the matmul applies a_scale*b_scale per group in fp32.
    std::vector<int8_t> qA((size_t)Mp * K, 0);
    std::vector<float>  a_scale((size_t)Mp * nG, 1.0f);
    { std::vector<float> tmp((size_t)K);
      if (rocket_quant_int4_grouped((const float *)src1->data, nullptr, true, false,
                                    qA.data(), M, K, group, had, a_scale.data(), tmp.data()) < 0) return -1; }

    std::vector<float> Cf((size_t)Mp * N);
    int rc = rocket_matmul_int4_prepacked_gw(ctx->i4_dev, Mp, K, N, qA.data(),
                                             a_scale.data(), b_scale, Cf.data(), w);
    if (rc != 0) return -1;
    memcpy((float *)dst->data, Cf.data(), (size_t)M * N * sizeof(float));  // first M rows
    return 0;
}

// ===========================================================================
// Diagnostics, the dequant thread pool, and weight->fp16 dequantization
// ===========================================================================
// ROCKET_VERIFY=1: recompute row 0 of an offloaded matmul on the CPU in fp32 and
// compare to what the NPU produced (already in dst). Logs the shape, weight
// type, max abs/rel error, and how many outputs are non-finite. Diagnostic only.
// The heavy per-op recompute/trace diagnostics below compile out unless built with
// -DGGML_ROCKET_DIAGNOSTICS=ON (defines ROCKET_DIAGNOSTICS); the production .so omits them.
#ifdef ROCKET_DIAGNOSTICS
static bool rocket_verify_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_VERIFY"); v = e ? atoi(e) : 0; }
    return v > 0;
}
#endif // ROCKET_DIAGNOSTICS
// Dequantize weight element k of a [N,K] row to f32, honouring the weight tensor's
// type. The CPU reference/fallback paths share this so a BF16 weight is never
// reinterpreted as F16 or F32 -- a 2-byte bf16 element read as a 4-byte f32 both
// over-reads the buffer and decodes the wrong bits.
static inline float rocket_weight_elem_f32(const void * Brow, int64_t k, ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(((const ggml_fp16_t *)Brow)[k]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(((const ggml_bf16_t *)Brow)[k]);
        default:             return ((const float *)Brow)[k];
    }
}

// Worker count for the quantized-weight dequant fan-out below. ROCKET_DEQUANT_THREADS
// overrides it (set 1 to force the serial decode -- the A/B baseline; set N to pin the
// count). Unset/0 => auto: the BIG-core count. NOT hardware_concurrency: on RK3588 the A55
// little cores are a ~7x-slower straggler on this fp16 dequant, so an 8-thread equal-chunk
// fan-out that lands 4 workers on the A55s runs ~3.4x slower than a 4-thread big-core-only
// one (the A76 workers finish and idle-wait at the join) [HW measured]. The pool
// workers are pinned to the big cluster (rocket_dequant_pool::worker). Fallback to hw if no
// big cluster is detected. Lazy-cached (per-micro-batch hot path); idempotent first read.
static int rocket_dequant_threads(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("ROCKET_DEQUANT_THREADS");
        const int req = e ? atoi(e) : 0;
        if (req > 0) {
            v = req;
        } else {
            const int nb = rocket_num_big_cores();   // A76 count, or 0 if no big cluster
            if (nb >= 1) {
                v = nb;
            } else {
                const unsigned hw = std::thread::hardware_concurrency();
                const int n = (int)(hw ? hw : 1);
                v = n > 8 ? 8 : n;
            }
        }
    }
    return v;
}

// Persistent CPU worker pool for the streaming dequant fan-out. rocket_weight_to_fp16
// runs per quantized matmul per micro-batch; spawning fresh std::threads each call cost
// ~35-55% of the dequant wall time on the A76 [HW measured] versus reusing pooled
// workers. One process-wide pool (sized to rocket_dequant_threads()), shared across
// backend instances -- preferable to N pools oversubscribing the big cores. Each call
// hands the pool one closure run as size() independent chunk-jobs (worker i computes
// chunk i) and blocks until all finish, so the closure's captured buffers are safe to
// reuse on return (no async work outlives run()). run() is serialized, so concurrent
// callers (an in-process multi-instance pool) take turns -- each turn already uses every
// worker, so this is correct core usage, not contention. Intentionally never destroyed
// (workers reclaimed at process exit) to avoid static-destruction-order hazards.
class rocket_dequant_pool {
public:
    explicit rocket_dequant_pool(int nthreads) : n_(nthreads < 1 ? 1 : nthreads) {
        for (int i = 0; i < n_; i++) workers_.emplace_back([this, i] { worker(i); });
    }
    int size() const { return n_; }
    void run(const std::function<void(int)> & job) {
        std::lock_guard<std::mutex> run_lk(run_mutex_);   // one parallel_for at a time
        {
            std::unique_lock<std::mutex> lk(m_);
            job_ = &job; done_ = 0; gen_++;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return done_ == n_; });
        job_ = nullptr;
    }
private:
    void worker(int i) {
        rocket_pin_worker(i);   // keep this persistent dequant worker on an A76, never an A55
        long mygen = 0;
        for (;;) {
            const std::function<void(int)> * job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ > mygen; });
                mygen = gen_;
                job = job_;
            }
            (*job)(i);
            {
                std::unique_lock<std::mutex> lk(m_);
                if (++done_ == n_) cv_done_.notify_one();
            }
        }
    }
    const int n_;
    std::vector<std::thread> workers_;
    const std::function<void(int)> * job_ = nullptr;
    std::mutex m_, run_mutex_;
    std::condition_variable cv_, cv_done_;
    long gen_ = 0;
    int done_ = 0;
};

// Lazily-created, leaked-at-exit process-wide dequant pool (see rocket_dequant_pool).
// First touched by the first quantized matmul that fans out; an F16-only model never
// creates it.
static rocket_dequant_pool & rocket_get_dequant_pool(void) {
    static rocket_dequant_pool * pool = new rocket_dequant_pool(rocket_dequant_threads());
    return *pool;
}

// Convert a static weight tensor's row-major [N,K] payload (any ggml type other than
// F16, which the caller takes zero-copy) into a row-major fp16 B buffer the NPU packer
// consumes. This is the streaming dequant path: a quantized GGUF (Q8_0 / Q4_K / Q6_K / ...) is
// dequantized straight to fp16 here -- per weight and transiently, never a whole-model
// F16 copy in RAM. So the model stays compact (fast quantized CPU decode, no 22 GB-F16
// pressure) while prefill runs on the NPU's fp16 matmul. Quantized rows are decoded with
// the type's own dequantizer (bit-identical to ggml's CPU backend) one [K] row at a time,
// so the K-float scratch is the only extra allocation. Returns false for an undecodable
// type (caller then leaves the op on the CPU). Requires K % ggml_blck_size(t) == 0 for
// quantized types (supports_op enforces it).
static bool rocket_weight_to_fp16(const void * B_src, ggml_type t,
                                  int64_t N, int64_t K, ggml_fp16_t * B16) {
    if (t == GGML_TYPE_F32) {
        ggml_fp32_to_fp16_row((const float *)B_src, B16, N * K);
        return true;
    }
    if (t == GGML_TYPE_BF16) {
        const ggml_bf16_t * b = (const ggml_bf16_t *)B_src;
        // Each [N] row is independent. A bf16 weight is re-decoded to fp16 every
        // micro-batch on the prefill critical path (bf16 cannot be zero-copied like
        // F16), so on a large model the serial decode dominates; fan the rows across
        // the persistent dequant pool. Bit-identical to the serial loop -- disjoint
        // output rows, no cross-row state -- and mirrors the quantized fan-out below.
        auto cvt_rows = [&](int64_t n0, int64_t n1) {
            for (int64_t i = n0 * K; i < n1 * K; i++)
                B16[i] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(b[i]));
        };
        const int nthr = rocket_dequant_threads();
        if (N < 64 || nthr <= 1) { cvt_rows(0, N); return true; }
        rocket_dequant_pool & pool = rocket_get_dequant_pool();
        const int64_t per = (N + pool.size() - 1) / pool.size();
        pool.run([&](int i) {
            int64_t n0 = (int64_t)i * per, n1 = n0 + per > N ? N : n0 + per;
            if (n0 < n1) cvt_rows(n0, n1);
        });
        return true;
    }
    if (ggml_is_quantized(t)) {
        const ggml_type_traits * tr = ggml_get_type_traits(t);
        if (!tr || !tr->to_float) return false;
        const size_t rbytes = ggml_row_size(t, K);   // bytes for one [K] quant row
        // Each [N] row is independent -- a private K-float scratch in, a disjoint B16
        // slice out -- so fan the rows across CPU workers. For a streaming quantized
        // GGUF this dequant runs on the host every micro-batch and sits on the prefill
        // critical path; splitting the rows is bit-identical to the serial decode (no
        // cross-row state). Mirrors rocket_quant_int4_grouped's fan-out. Per-row decode
        // is the same scalar ggml dequantizer either way.
        auto dq_rows = [&](int64_t n0, int64_t n1) {
            std::vector<float> frow((size_t)K);
            for (int64_t n = n0; n < n1; n++) {
                tr->to_float((const char *)B_src + (size_t)n * rbytes, frow.data(), K);
                ggml_fp32_to_fp16_row(frow.data(), B16 + n * K, K);
            }
        };
        const int nthr = rocket_dequant_threads();
        if (N < 64 || nthr <= 1) {                    // small / disabled: serial
            dq_rows(0, N);
            return true;
        }
        // Fan the rows across the PERSISTENT dequant pool (worker i decodes chunk i).
        // Reusing pooled workers instead of spawning std::threads per call saves the
        // ~35-55% of the dequant wall time that per-call create/join cost [HW measured].
        // run() blocks until every chunk finishes, so B16 is fully written on return.
        // Bit-identical to the serial decode (same rows, disjoint output slices, no
        // cross-row state). The pool's worker count is rocket_dequant_threads(), so the
        // chunking matches a per-call fan-out exactly.
        rocket_dequant_pool & pool = rocket_get_dequant_pool();
        const int64_t per = (N + pool.size() - 1) / pool.size();
        pool.run([&](int i) {
            int64_t n0 = (int64_t)i * per, n1 = n0 + per > N ? N : n0 + per;
            if (n0 < n1) dq_rows(n0, n1);
        });
        return true;
    }
    return false;
}
#ifdef ROCKET_DIAGNOSTICS
static void rocket_verify(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // weights B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   A[M,K]
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    const ggml_type wt = src0->type;
    const float * A = (const float *)src1->data;   // row 0
    const float * C = (const float *)dst->data;    // row 0 (NPU result)
    double maxabs = 0, maxrel = 0; int nonfinite = 0; double amax = 0;
    for (int64_t k = 0; k < K; k++) { double a = fabs((double)A[k]); if (a > amax) amax = a; }
    for (int64_t n = 0; n < N; n++) {
        const char * Brow = (const char *)src0->data + (size_t)n * src0->nb[1];
        double ref = 0;
        for (int64_t k = 0; k < K; k++) ref += (double)A[k] * (double)rocket_weight_elem_f32(Brow, k, wt);
        const double got = C[n];
        if (!std::isfinite(got)) nonfinite++;
        const double ad = fabs(got - ref);
        const double rd = ad / (fabs(ref) + 1e-6);
        if (ad > maxabs) maxabs = ad;
        if (rd > maxrel) maxrel = rd;
    }
    GGML_LOG_DEBUG("[VERIFY] M=%lld K=%lld N=%lld wtype=%d amaxA=%.1f row0 max_abs=%.4f max_rel=%.4f nonfinite=%d/%lld\n",
            (long long)M, (long long)K, (long long)N, (int)wt, amax,
            maxabs, maxrel, nonfinite, (long long)N);
}
#endif // ROCKET_DIAGNOSTICS

// ---------------------------------------------------------------------------
// Per-op diagnostics for the prepacked path (env-gated; default path untouched).
// These diagnose a class of resident-weight corruption rooted in CACHE ALIASING:
// keying the cache on src0->data is unsafe because ggml-backend-sched copies weights
// into a pooled, address-reused buffer, so one address backs many logical weights
// and the cache would serve the wrong one (avoided by keying on weight name +
// supports_buft=is_host; see wcache / rocket_weight_key). General-purpose tools.
//
//  ROCKET_TRACE=1  -> one line per offloaded matmul: seq, weight name + data ptr
//    (`grep wptr | uniq -d` reveals one ptr shared across many
//    distinct weight names; such sharing is invisible to ROCKET_AB, which recomputes
//    mt from the SAME src0->data), dst contiguity/ne/nb (supports_op never checks
//    dst contiguity; a strided/viewed dst + our M*N contiguous write would smear a
//    neighbour tensor), and amax/nonfinite/mean of the INPUT activation and OUTPUT.
//    With a coherent baseline (NO_PREPACK) and a suspect run (FORCE_PREPACK) on the
//    same fixed prompt + greedy decode, the op-set/order is identical so traces
//    align 1:1; the first op whose INPUT amax/nonfinite diverges qualitatively
//    (orders of magnitude / NaN, not fp16-rounding ulps) localizes corruption to a
//    layer.
//
//  ROCKET_CPU_FORWARD=1 -> compute on the NPU (so every NPU/driver side effect
//    still happens), then OVERWRITE dst with a CPU fp32 reference. Splits "wrong dst
//    content/stride" (output turns coherent) from "side effect on other memory"
//    (output stays wrong). O(M*N*K); use a short prompt (-n 16) so it's cheap.
// ---------------------------------------------------------------------------
// Lazy getenv cache. The function-local `static int v = -1` callers below share this
// idiom. The first-use read is NOT synchronized, so concurrent first calls can race;
// it is benign because the cached value is idempotent (every racer computes the same
// getenv result and writes the same int — no torn value, no policy divergence). If a
// knob ever becomes non-idempotent or load-bearing for first-call ordering, switch to
// std::call_once / a ctx-time read instead.
#ifdef ROCKET_DIAGNOSTICS
static int rocket_env_int(const char * name, int * cache) {
    if (*cache < 0) { const char * e = getenv(name); *cache = e ? atoi(e) : 0; }
    return *cache;
}

// CPU fp32 reference matmul: dst[M,N] = src1[M,K] * src0[N,K]^T, written as M*N
// contiguous F32 -- bit-for-bit the same destination region rocket_unpack_output
// writes, so swapping it in isolates NPU side effects from the dst content.
static void rocket_cpu_matmul(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // weights B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   A[M,K]
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    const ggml_type wt = src0->type;
    float * C = (float *)dst->data;
    for (int64_t m = 0; m < M; m++) {
        const float * A = (const float *)src1->data + m * K;
        for (int64_t n = 0; n < N; n++) {
            const char * Brow = (const char *)src0->data + (size_t)n * src0->nb[1];
            double acc = 0;
            for (int64_t k = 0; k < K; k++) acc += (double)A[k] * (double)rocket_weight_elem_f32(Brow, k, wt);
            C[m * N + n] = (float)acc;
        }
    }
}
static bool rocket_cpu_forward_on(void) {
    static int v = -1; return rocket_env_int("ROCKET_CPU_FORWARD", &v) > 0;
}
#endif // ROCKET_DIAGNOSTICS

// CPU fp32 fallback for a single matmul slice, computed from the ORIGINAL inputs
// (no activation scaling, no fp16 rounding of A). Used when an NPU job fails at
// runtime so the graph still produces a correct result instead of aborting the
// host process. A = src1 [M,K] f32 contiguous; B = src0 [N,K] (f16/bf16/
// f32, per `wt`) contiguous; C = dst [M,N] f32 contiguous.
static void rocket_cpu_matmul_slice(const float * A, const void * B, ggml_type wt,
                                    float * C, int64_t M, int64_t N, int64_t K) {
    if (ggml_is_quantized(wt)) {
        // Quantized weight: per-element striding is invalid (block layout), so
        // decode each [K] row with the type's dequantizer -- the same values ggml's CPU
        // backend uses -- then the plain f32 dot. Rare safety-net path (the NPU matmul
        // declined), so clarity over speed.
        const ggml_type_traits * tr = ggml_get_type_traits(wt);
        const size_t rbytes = ggml_row_size(wt, K);
        std::vector<float> Brow((size_t)K);
        for (int64_t n = 0; n < N; n++) {
            tr->to_float((const char *)B + (size_t)n * rbytes, Brow.data(), K);
            for (int64_t m = 0; m < M; m++) {
                const float * Arow = A + m * K;
                double acc = 0;
                for (int64_t k = 0; k < K; k++) acc += (double)Arow[k] * (double)Brow[k];
                C[m * N + n] = (float)acc;
            }
        }
        return;
    }
    const size_t bstride = ggml_type_size(wt);   // bytes per weight element
    for (int64_t m = 0; m < M; m++) {
        const float * Arow = A + m * K;
        for (int64_t n = 0; n < N; n++) {
            const char * Brow = (const char *)B + (size_t)n * K * bstride;
            double acc = 0;
            for (int64_t k = 0; k < K; k++) acc += (double)Arow[k] * (double)rocket_weight_elem_f32(Brow, k, wt);
            C[m * N + n] = (float)acc;
        }
    }
}

#ifdef ROCKET_DIAGNOSTICS
static bool rocket_trace_on(void) {
    static int v = -1; return rocket_env_int("ROCKET_TRACE", &v) > 0;
}
static void rocket_tensor_stats(const ggml_tensor * t, double * amax, double * mean, int64_t * nonfin) {
    const int64_t n = ggml_nelements(t);
    const float * p = (const float *)t->data;
    double am = 0, sum = 0; int64_t nf = 0;
    for (int64_t i = 0; i < n; i++) {
        const float v = p[i];
        if (!std::isfinite(v)) { nf++; continue; }
        const double a = fabs((double)v);
        if (a > am) am = a;
        sum += a;
    }
    *amax = am; *mean = n ? sum / (double)n : 0; *nonfin = nf;
}
static void rocket_trace(const ggml_tensor * dst, const char * path) {
    static int seq = 0;
    const ggml_tensor * src0 = dst->src[0];   // weights
    const ggml_tensor * src1 = dst->src[1];   // input
    double in_amax, in_mean, out_amax, out_mean; int64_t in_nf, out_nf;
    rocket_tensor_stats(src1, &in_amax, &in_mean, &in_nf);
    rocket_tensor_stats(dst,  &out_amax, &out_mean, &out_nf);
    GGML_LOG_DEBUG(
        "[TRACE] #%04d %-8s w=%-24s wptr=%p wop=%d dst=%s contig=%d "
        "ne=[%lld,%lld] nb=[%zu,%zu] vsrc=%p | "
        "in amax=%.3g mean=%.3g nan=%lld | out amax=%.3g mean=%.3g nan=%lld\n",
        seq++, path,
        src0->name[0] ? src0->name : "(noname)", src0->data, (int)src0->op,
        dst->name[0] ? dst->name : "(noname)", ggml_is_contiguous(dst),
        (long long)dst->ne[0], (long long)dst->ne[1], dst->nb[0], dst->nb[1],
        (void *)dst->view_src,
        in_amax, in_mean, (long long)in_nf,
        out_amax, out_mean, (long long)out_nf);
}
#endif // ROCKET_DIAGNOSTICS

// Stable cache key for a weight tensor. ggml-backend-sched names its input copies
// "<backend>#<original_name>#<copy_idx>" (e.g. "ROCKET#blk.0.attn_q.weight#0");
// normalize to the original name so the copy and the original map to one entry,
// and so two copies that happen to share a pooled address stay distinct. Empty
// string => uncacheable (caller falls back to the per-call mt path).
static std::string rocket_weight_key(const ggml_tensor * t) {
    const char * n = t->name;
    if (!n || !n[0]) return std::string();
    // The scheduler's copy names are <backend>#<name>#<idx>; extract <name> (the part
    // between the first and last '#') so a copy maps to its original's cache entry.
    std::string key(n);
    const char * first = strchr(n, '#');
    if (first) {
        const char * last = strrchr(n, '#');
        if (last > first) key.assign(first + 1, last - first - 1);
        // Exactly one '#' (not the copy format): fall through and key on the whole
        // name verbatim — still stable and unique, just not stripped.
    }
    // ggml auto-names any UNNAMED tensor "leaf_%d" / "node_%d" by its position in the
    // graph being built. Those indices are NOT a stable per-weight identity across a
    // host's several sub-graphs: whisper.cpp never names its weight tensors and builds
    // separate conv/encode/cross/decode graphs, so distinct weights (even different
    // shapes) reuse the same "leaf_N" string — and the resident-weight cache, keyed on
    // this name, would then serve one weight's packed tiles for another's matmul
    // (garbage output). Reject the ggml-default names: an empty key makes every caller
    // (fp16/int8/int4 wcache) stream the weight per call instead — correct, and
    // timing-neutral for a single-pass encoder (each weight is packed once regardless).
    // Models whose weights carry real names (llama.cpp: "blk.N.attn_q.weight", ...) never
    // match, so their resident-weight caching is unchanged.
    auto is_ggml_autoname = [](const std::string & s) -> bool {
        const char * p;
        if      (s.compare(0, 5, "leaf_") == 0) p = s.c_str() + 5;
        else if (s.compare(0, 5, "node_") == 0) p = s.c_str() + 5;
        else return false;
        if (!*p) return false;                 // "leaf_" with no index -> not the pattern
        for (; *p; ++p) if (*p < '0' || *p > '9') return false;
        return true;                           // leaf_<digits> / node_<digits>
    };
    if (is_ggml_autoname(key)) return std::string();
    return key;
}

// ROCKET_PREPACK_MADVISE=1: after a weight is scattered
// into its resident NPU BO, MADV_DONTNEED the row-major GGUF source pages to
// reclaim RAM. The GGUF is mmap'd PROT_READ,
// so the pages are clean and the kernel drops them with no data loss (a later read
// re-faults from the file).
//
// WARNING: PREFILL / llama-bench `-n 0` ONLY. This BREAKS CPU decode. Our hybrid runs
// prefill on the NPU but DECODE on the CPU, which reads these SAME row-major weights
// every token -- dropping them forces a per-token re-fault from NVMe (or, under
// --no-mmap, anonymous pages re-fault to ZERO -> garbage). It is safe ONLY when the
// row-major source is never read again. Hence a distinct opt-in flag, separate from
// ROCKET_FORCE_PREPACK: the madvise is purely the RAM lever that lets the resident
// (~22GB tiled) + GGUF (~22GB) coexistence fit under 32GB for prefill-only runs.
static bool rocket_prepack_madvise_on(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("ROCKET_PREPACK_MADVISE"); v = e ? atoi(e) : 0; }
    return v > 0;
}
// True if `addr` lies in a FILE-BACKED mapping (a real backing file with a non-zero
// inode in /proc/self/maps), as opposed to anonymous memory. MADV_DONTNEED on a
// file-backed clean page drops it harmlessly (a later read re-faults from the file);
// on an ANONYMOUS page (e.g. a --no-mmap model copied into a malloc'd buffer) it
// zero-fills on next touch -> silent weight corruption. Conservative: any parse
// failure or "addr not found" returns false (skip the madvise).
static bool rocket_addr_is_file_backed(const void * addr) {
    const uintptr_t a = (uintptr_t)addr;
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) return false;
    char line[512];
    bool file_backed = false;
    while (fgets(line, sizeof(line), f)) {
        uintptr_t lo = 0, hi = 0; unsigned long inode = 0;
        // format: lo-hi perms offset dev:dev inode  pathname
        if (sscanf(line, "%lx-%lx %*s %*s %*s %lu", &lo, &hi, &inode) == 3
            && a >= lo && a < hi) {
            file_backed = (inode != 0);   // anonymous mappings report inode 0
            break;
        }
    }
    fclose(f);
    return file_backed;
}

// Drop only whole pages fully inside [ptr, ptr+len) -- round the start UP and the end
// DOWN to page boundaries -- so a page shared with an adjacent tensor is never
// discarded. Returns the bytes actually dropped (0 on a sub-page weight or failure).
// Refuses to discard a non-file-backed (anonymous) mapping: under --no-mmap the model
// is copied into anonymous RAM, where MADV_DONTNEED would zero-fill the weights.
static size_t rocket_madvise_dontneed(const void * ptr, size_t len) {
    if (!rocket_addr_is_file_backed(ptr)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: ROCKET_PREPACK_MADVISE is set but the weight source is not "
                          "file-backed (e.g. --no-mmap) -- skipping reclaim; dropping these "
                          "anonymous pages would zero-fill the weights.\n", __func__);
        }
        return 0;
    }
    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t a   = (uintptr_t)ptr;
    const uintptr_t beg = (a + pg - 1) & ~(uintptr_t)(pg - 1);   // first full page
    const uintptr_t end = (a + len)    & ~(uintptr_t)(pg - 1);   // last full-page bound
    if (end <= beg) return 0;
    if (madvise((void *)beg, (size_t)(end - beg), MADV_DONTNEED) != 0) return 0;
    return (size_t)(end - beg);
}
// (Deliberately NO MADV_WILLNEED/SEQUENTIAL prefetch before the pack scatter. Measured on
// RK3588 + NVMe: the kernel's default readahead already streams the cold GGUF mmap in as
// MINOR faults [HW sweep], so the resident-weight load is NVMe-bandwidth-bound, not
// fault-latency-bound. An explicit MADV_WILLNEED is then neutral, and MADV_SEQUENTIAL +
// MADV_WILLNEED is actively HARMFUL -- SEQUENTIAL's drop-behind eviction re-faults the
// pages the tiled scatter revisits across K-blocks, ~+33% wall. Leave the prefetch to the
// kernel.)

// ===========================================================================
// fp16 prepacked (pack-weights-once) matmul
// ===========================================================================
// Pack-weights-once fast path for a plain 2D matmul against a STATIC F16 weight
// (a model parameter: leaf tensor, op == GGML_OP_NONE). The weights are packed
// into resident NPU BOs on first use and cached by weight name (see
// rocket_weight_key); every later forward pass reuses them and only re-packs A.
// Returns 0 on success, <0 to tell
// the caller to fall back to the per-call (rocket_matmul_fp16_mt) path.
static int ggml_backend_rocket_mul_mat_prepacked(
        ggml_backend_rocket_context * ctx, ggml_tensor * dst,
        int M, int K, int N) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    if (ctx->dev_failed) return -1;
    if (!ctx->dev) {
        ctx->dev = rocket_ctx_create(ctx->n_threads);
        if (!ctx->dev) { ctx->dev_failed = true; return -1; }
    }

    // The driver plan (and hence the resident weight layout) is keyed on the
    // padded M, so pack/cache/matmul all use Mp; only the copy-back uses M.
    const int Mp = rocket_pad_m(M);

    const std::string key = rocket_weight_key(src0);
    if (key.empty()) return -1;               // no stable identity -> mt path
    // Build (pack + cache + madvise) the resident weight, planning its tile layout at pack_m.
    // Returns the weight, or nullptr if the NPU's 32-bit IOVA window is full / over budget --
    // in which case the caller streams this weight via the per-call mt path. Once the window
    // fills (a prior pack failed), dev_resident_full latches so new weights stop trying (those
    // allocs would fail too and could starve the mt fallback); already-resident weights keep working.
    // Packs from b_src (fp16 [N,K]) -- src0->data for an F16 weight (zero-copy), or a
    // transient dequant buffer for a quantized one (ROCKET_QUANT_RESIDENT). The resident
    // BO becomes the only persistent fp16 copy, so the dequant buffer can be freed after.
    auto build_resident = [&](int pack_m, const ggml_fp16_t * b_src) -> rocket_weights * {
        if (ctx->dev_resident_full) return nullptr;
        const size_t est = (size_t)N * K * sizeof(ggml_fp16_t);   // resident weight bytes (M-independent)
        if (ctx->cache_budget && ctx->resident_bytes + est > ctx->cache_budget)
            return nullptr;                   // over budget -> per-call mt path (frees per call)
        // Runtime OOM guard (no swap): if free RAM has fallen to the reserve floor, stop making
        // weights resident and stream the rest. The static byte budget is sized from init-time
        // MemAvailable, which counts the not-yet-faulted (reclaimable) GGUF pages as free -- but an
        // F16 resident tile DUPLICATES its still-mapped GGUF weight, so a model that does not fit
        // ~2x can approach the ceiling before the byte budget trips. This latch stops it first.
        if (ctx->resident_floor_bytes) {
            const size_t avail = rocket_meminfo_bytes("MemAvailable");
            if (avail && avail < ctx->resident_floor_bytes) {
                ctx->dev_resident_full = true;   // latch: the rest streams (per-call mt)
                if (rocket_debug_on())
                    GGML_LOG_DEBUG("[prepack] MemAvailable %zuMB < floor %zuMB at resident=%zuMB"
                            " -> streaming remaining weights\n",
                            avail >> 20, ctx->resident_floor_bytes >> 20, ctx->resident_bytes >> 20);
                return nullptr;
            }
        }
        rocket_weights * nw = rocket_weights_pack(ctx->dev, pack_m, K, N,
                                reinterpret_cast<const _Float16 *>(b_src));
        if (!nw) {                            // IOVA/alloc exhausted: latch + stream the rest
            ctx->dev_resident_full = true;
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[prepack] IOVA window full at resident=%zuMB -> streaming remaining weights\n",
                        ctx->resident_bytes >> 20);
            return nullptr;
        }
        // nw holds NPU BOs; free it if the cache insert (rehash / key copy) throws
        // before the map takes ownership. Dismissed once the insert has succeeded.
        rocketraii::scope_guard nw_guard([&] { rocket_weights_free(ctx->dev, nw); });
        ctx->wcache[key] = { nw, pack_m, K, N, est };
        nw_guard.dismiss();
        ctx->resident_bytes += est;

        // The tiled copy is now resident, so reclaim the row-major source.
        // PREFILL-ONLY -- breaks CPU decode (see rocket_prepack_madvise_on). The compute
        // below reads the resident BO, never src0->data, so dropping it here is safe.
        if (rocket_prepack_madvise_on()) {
            const size_t dropped = rocket_madvise_dontneed(src0->data, ggml_nbytes(src0));
            ctx->madvised_bytes += dropped;
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[madvise] -%-24s %4zuMB dropped (reclaimed %zuMB, resident BOs %zuMB)\n",
                        key.c_str(), dropped >> 20, ctx->madvised_bytes >> 20,
                        ctx->resident_bytes >> 20);
        }
        return nw;
    };

    // Reuse the resident weight ACROSS M: the scatter layout is M-independent for M>=MAX_TILE,
    // so the cache keys on the weight name/K/N only (NOT the padded M) — a weight packed at one
    // prefill length serves any other M>=MAX_TILE with no re-pack. A weight is made resident only
    // when first seen at a reuse-worthy M (>= max_tile); a small one-shot M (a short-prompt
    // prefill) streams via the per-call mt path instead, so there is no wasted resident pack and
    // no re-pack thrash on mixed-length workloads — a later big prefill makes it resident on its
    // first >= max_tile call, and a smaller M against an already-resident weight returns -2 and
    // streams (leaving the resident weight intact for the big prefills).
    const int max_tile = rocket_hw_current()->max_tile;
    rocket_weights * w = nullptr;
    auto it = ctx->wcache.find(key);
    if (it != ctx->wcache.end()) {
        if (it->second.K == K && it->second.N == N) {
            w = it->second.w;
        } else {                              // same buffer, different shape: re-pack
            rocket_weights_free(ctx->dev, it->second.w);
            ctx->resident_bytes -= it->second.bytes;
            ctx->wcache.erase(it);
        }
    }
    if (!w) {
        if (Mp < max_tile) return -1;                     // small one-shot M -> mt path (no resident pack)
        const ggml_fp16_t * b_src;
        std::vector<ggml_fp16_t> b_dq;                    // transient dequant buffer (quant only)
        if (ggml_is_quantized(src0->type)) {
            // ROCKET_QUANT_RESIDENT: dequant the quantized weight to fp16 ONCE here (cache
            // miss). rocket_weights_pack copies it into the resident BOs, after which b_dq is
            // freed -- so prefill pays neither the per-call dequant nor the per-call packB the
            // streaming path pays every micro-batch.
            b_dq.resize((size_t)N * K);
            if (!rocket_weight_to_fp16(src0->data, src0->type, N, K, b_dq.data()))
                return -1;                                // undecodable type -> caller falls back
            b_src = b_dq.data();
        } else {
            b_src = (const ggml_fp16_t *)src0->data;      // F16 zero-copy (the original path)
        }
        if (!(w = build_resident(Mp, b_src))) return -1;  // window full / over budget -> mt path
    }

    std::vector<ggml_fp16_t> A16((size_t)Mp * K);   // value-initialized: pad rows = 0
    std::vector<ggml_fp16_t> C16((size_t)Mp * N);
    std::vector<float> scales((size_t)M);
    rocket_pack_activations((const float *)src1->data, A16.data(), M, K, scales.data());

    int rc = rocket_matmul_fp16_prepacked(ctx->dev, Mp, K, N,
                reinterpret_cast<const _Float16 *>(A16.data()),
                reinterpret_cast<_Float16 *>(C16.data()), w);

    if (rc != 0) {
        // rc == -2: this compute M is smaller than the canonical resident tiling (a short-prompt
        // prefill) -> stream it via the per-call mt path; expected, debug-only, and the resident
        // weight stays put for the big prefills. Any other rc is a real driver failure (e.g. a
        // cold-start fence-wait timeout) -> warn once, since the path self-heals via mt.
        if (rc == -2) {
            if (rocket_debug_on())
                GGML_LOG_DEBUG("[prepack] w=%s M=%d below resident tiling -> mt path\n", key.c_str(), M);
        } else {
            static bool warned = false;
            if (!warned || rocket_debug_on()) {
                GGML_LOG_WARN("[PREPACK-FAIL] w=%s M=%d K=%d N=%d rc=%d -> mt fallback\n",
                        key.c_str(), M, K, N, rc);
                warned = true;
            }
        }
        return -1;
    }

#ifdef ROCKET_DIAGNOSTICS
    // ROCKET_AB=1: recompute the SAME (scaled A16, F16 weights) via the mt path
    // in-context and compare to the prepacked result, to see if the prepacked
    // driver call diverges from the validated mt call on real inputs. F16-only:
    // a quantized resident weight (ROCKET_QUANT_RESIDENT) has no fp16 src0->data to
    // re-pack from (the dequant buffer was freed after the resident pack).
    if (getenv("ROCKET_AB") && !ggml_is_quantized(src0->type)) {
        std::vector<ggml_fp16_t> Cmt((size_t)Mp * N);
        rocket_matmul_fp16_mt(Mp, (int)K, (int)N,
            reinterpret_cast<const _Float16 *>(A16.data()),
            reinterpret_cast<const _Float16 *>(src0->data),
            reinterpret_cast<_Float16 *>(Cmt.data()), ctx->n_threads);
        double md = 0;
        for (int64_t i = 0; i < (int64_t)M * N; i++) {
            // DECODE fp16 -> fp32 before diffing (ggml_fp16_t is uint16_t; a raw
            // integer cast diffs bit patterns, not values -- that was a metric bug).
            double d = fabs((double)ggml_fp16_to_fp32(C16[i]) - (double)ggml_fp16_to_fp32(Cmt[i]));
            if (d > md) md = d;
        }
        GGML_LOG_DEBUG("[AB] M=%d K=%d N=%d prepacked-vs-mt max_abs(scaled)=%.5f\n",
                (int)M, (int)K, (int)N, md);

        // ROCKET_DUMP=1: on the FIRST badly-diverging op, write the EXACT inputs
        // (scaled A16, F16 weights) + both outputs so we can replay this precise call
        // standalone (the driver's replay harness) and learn whether the garbage is
        // data-specific or live-process-state-specific. Created via mkstemp under
        // $TMPDIR (default /tmp): a unique, O_EXCL, mode-0600 file, so a hostile symlink
        // or a predictable name can't redirect or expose the model data.
        static bool dumped = false;
        if (!dumped && md > 100.0 && getenv("ROCKET_DUMP")) {
            const char * tmpdir = getenv("TMPDIR");
            if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
            char path[512];
            snprintf(path, sizeof(path), "%s/rk_dump_XXXXXX", tmpdir);
            int dfd = mkstemp(path);   // O_CREAT|O_EXCL, mode 0600, unique name
            FILE * f = (dfd >= 0) ? fdopen(dfd, "wb") : nullptr;
            if (f) {
                int hdr[4] = { (int)M, (int)K, (int)N, Mp };
                fwrite(hdr, sizeof(int), 4, f);
                fwrite(A16.data(),  sizeof(ggml_fp16_t), (size_t)Mp * K, f);
                fwrite(src0->data,  sizeof(ggml_fp16_t), (size_t)N  * K, f);
                fwrite(C16.data(),  sizeof(ggml_fp16_t), (size_t)Mp * N, f);  // in-context prepacked
                fwrite(Cmt.data(),  sizeof(ggml_fp16_t), (size_t)Mp * N, f);  // in-context mt
                fclose(f);
                GGML_LOG_DEBUG("[DUMP] wrote %s M=%d K=%d N=%d Mp=%d md=%.1f\n",
                        path, (int)M, (int)K, (int)N, Mp, md);
                dumped = true;
            } else if (dfd >= 0) {
                close(dfd);
            }
        }
    }
#endif // ROCKET_DIAGNOSTICS

    rocket_unpack_output(C16.data(), (float *)dst->data, M, N, scales.data());
    return 0;
}

// ===========================================================================
// Matmul dispatch (path selection) and node fusion
// ===========================================================================
// Per-op diagnostics, run after the NPU has produced dst (so every NPU/driver
// side effect has already happened). Order matters: trace + verify must read the
// NPU result BEFORE CPU_FORWARD overwrites it with the reference.
static void rocket_mul_mat_post(ggml_tensor * dst, const char * path) {
#ifdef ROCKET_DIAGNOSTICS
    if (rocket_trace_on())       rocket_trace(dst, path);
    if (rocket_verify_on())      rocket_verify(dst);
    if (rocket_cpu_forward_on()) rocket_cpu_matmul(dst);
#else
    (void)dst; (void)path;
#endif
}

static void ggml_backend_rocket_mul_mat(ggml_backend_rocket_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // weights -> B[N,K]
    const ggml_tensor * src1 = dst->src[1];   // input   -> A[M,K]

    const int64_t K  = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t M  = src1->ne[1];

    // One-time hint: quantized weights run the per-microbatch dequant path, so a
    // small micro-batch makes the dequant dominate. If we're offloading a quantized
    // prefill below a full 2048 ubatch, nudge the user to raise it (~2x prefill).
    if (ggml_is_quantized(src0->type) && M < 2048) {
        static bool ub_hinted = false;
        if (!ub_hinted) {
            ub_hinted = true;
            // Emitted on the rocket_log channel (not GGML_LOG_*) so ROCKET_LOG_STDERR
            // can surface it even when the host silences ggml (e.g. llama-bench).
            ROCKET_LOGI("[rocket] quantized prefill is dequant-bound at this micro-batch; "
                        "run with -b 2048 -ub 2048 for ~2x (the default -ub 512 ~halves it)\n");
        }
    }

    // Dedicated bf16 fp32-output datapath, selected by ROCKET_BF16=1: packs
    // activations as bf16 with NO scaling and reads fp32 out directly (bf16's
    // fp32-range exponent makes the per-row amax-scan/scale unnecessary), so it is
    // token-identical to the reference at fp32 accumulate range. 2D src1 only. With
    // ROCKET_BF16 unset -- and for batched src1, or on a decline here -- bf16 weights
    // fall through to the fp16 streaming route below, which decodes bf16 to fp16 like
    // an F32 or quantized weight.
    if (ctx->bf16_mode < 0) { const char * e = getenv("ROCKET_BF16"); ctx->bf16_mode = e ? atoi(e) : 0; }
    if (ctx->bf16_mode
        && !ggml_is_quantized(src0->type)   // quantized weights take the fp16 dequant path
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->ne[2] == 1 && src1->ne[3] == 1) {
        if (ggml_backend_rocket_mul_mat_bf16(ctx, dst, (int)M, (int)K, (int)N) == 0) {
            rocket_mul_mat_post(dst, "bf16");
            return;
        }
        // Declined (device open failed / unsupported shape): fall through to the fp16
        // streaming route, which decodes the bf16 weight to fp16.
    }

    // int8 W8A8 path, opt-in via ROCKET_INT8=1. Routes a plain 2D
    // static-weight GEMM through quantize -> one-shot int8 NPU matmul -> dequantize,
    // bypassing the fp16 prepacked/streaming machinery. Requires N%32 (int8 weight
    // k-group, stricter than fp16's N%16) and 2D shapes; anything else falls through
    // to the fp16 path below. Correctness-first bring-up (re-quants weights per call);
    // ROCKET_VERIFY=1 reports the per-op W8A8-vs-fp32 error (high max_rel localizes
    // the activation-outlier layers the Hadamard rotation must fix).
    if (ctx->int8_mode < 0) { const char * e = getenv("ROCKET_INT8"); ctx->int8_mode = e ? atoi(e) : 0; }
    if (ctx->int8_resident < 0) { const char * e = getenv("ROCKET_INT8_RESIDENT"); ctx->int8_resident = e ? atoi(e) : 0; }
    // N%32 (int8 weight k-group) and K%32 (driver's int8 K-group) are the int8
    // shape contract. supports_op already gates K%32 for offloaded ops, but the
    // dispatch self-guards so a direct/edge caller can't reach rocket_matmul_int8
    // with K%32!=0 (the bf16 branch self-guards via rocket_matmul_plan_bf16).
    if (ctx->int8_mode
        && !ggml_is_quantized(src0->type)   // quantized weights take the fp16 dequant path
        && (N % 32 == 0) && (K % 32 == 0)
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->ne[2] == 1 && src1->ne[3] == 1) {
        // ROCKET_INT8_RESIDENT=1 routes through the resident-weight path;
        // otherwise the one-shot per-call path. Either returning 0 means done; <0
        // falls through to the fp16 path below (the un-resident weight's F16 source
        // is intact -- madvise only drops sources of weights that went resident).
        // On the RK3576 the W8A8 route is a different one — int8 out through a per-column
        // requant, not int32 out — so the branch is chosen off the DETECTED PART, not off
        // a knob. A mis-selected branch here is a REFUSAL and not a wrong surface: the
        // RK3588 generators refuse on this part by construction and the RK3576 entries
        // refuse there, which is the one place this port is protected by the library
        // rather than by review.
        const int r8 = rocket_rk3576_selected()
            ? ggml_backend_rocket_mul_mat_rk3576(ctx, dst, (int)M, (int)K, (int)N)
            : (ctx->int8_resident
                ? ggml_backend_rocket_mul_mat_int8_resident(ctx, dst, (int)M, (int)K, (int)N)
                : ggml_backend_rocket_mul_mat_int8(ctx, dst, (int)M, (int)K, (int)N));
        if (r8 == 0) {
            rocket_mul_mat_post(dst, rocket_rk3576_selected() ? "i8-76"
                                     : (ctx->int8_resident ? "int8r" : "int8"));
            return;
        }
    }

    // int4 W4A4 path, opt-in via ROCKET_INT4=1. The int4 sibling of the int8 block:
    // quantize -> native int4xint4->int16 NPU matmul -> dequantize. int4's weight
    // k-group is 64 (stricter than int8's N%32); K%32 as int8. Quantized src0 takes
    // the fp16 dequant path. A RAM/bandwidth play, not MAC-bound; W4A4 is a
    // hard quality regime, so it leans on ROCKET_INT4_HADAMARD + group-wise scales.
    if (ctx->int4_mode < 0) { const char * e = getenv("ROCKET_INT4"); ctx->int4_mode = e ? atoi(e) : 0; }
    if (ctx->int4_resident < 0) { const char * e = getenv("ROCKET_INT4_RESIDENT"); ctx->int4_resident = e ? atoi(e) : 0; }
    if (ctx->int4_mode
        && !ggml_is_quantized(src0->type)   // quantized weights take the fp16 dequant path
        && (N % 64 == 0) && (K % 32 == 0)
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->ne[2] == 1 && src1->ne[3] == 1) {
        // Resident int4 first (group-wise weights held in NPU nibble BOs, ~1/4 RAM, no
        // per-call weight scatter); on a miss (per-channel / over budget / IOVA full)
        // fall through to the one-shot int4 path.
        if (ctx->int4_resident
            && ggml_backend_rocket_mul_mat_int4_resident(ctx, dst, (int)M, (int)K, (int)N) == 0) {
            rocket_mul_mat_post(dst, "int4r");
            return;
        }
        if (ggml_backend_rocket_mul_mat_int4(ctx, dst, (int)M, (int)K, (int)N) == 0) {
            rocket_mul_mat_post(dst, "int4");
            return;
        }
    }

    // Resident dequant cache for quantized GGUF weights, opt-in via ROCKET_QUANT_RESIDENT=1.
    // A quantized weight is dequantized to fp16 ONCE and scattered into resident NPU BOs
    // (reusing the F16 prepacked path), so prefill pays neither the per-microbatch dequant
    // NOR the per-call packB -- lifting quant prefill from ~0.64x toward F16 parity and fixing
    // the -ub 512 / short-follow-up cases the -ub 2048 amortization can't. Routed for all K
    // (the point is to stop re-dequanting deep-K FFN weights too), 2D static shapes only; on a
    // miss (M below the resident tiling, IOVA window full, over cache_budget) the prepacked call
    // returns <0 and execution falls through to the per-call dequant->fp16 streaming path
    // below -- so an un-resident weight stays correct, just at the streaming cost.
    if (rocket_quant_resident_on()
        && ggml_is_quantized(src0->type)
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->ne[2] == 1 && src1->ne[3] == 1
        && ggml_backend_rocket_mul_mat_prepacked(ctx, dst, (int)M, (int)K, (int)N) == 0) {
        rocket_mul_mat_post(dst, "prepacked-q");
        return;
    }

    // Fast path: static F16 weight, plain 2D (no batch broadcast). Pack once,
    // reuse across forward passes. Falls through to the per-call mt path on miss.
    // ROCKET_NO_PREPACK=1 disables it entirely (force the mt path).
    //
    // Prepacked is CORRECT for all K. Coherence rests on two choices that close the
    // cache-aliasing class: the cache keys on the weight NAME [rocket_weight_key], not
    // src0->data (ggml-backend-sched copies weights into pooled, address-reused buffers,
    // so one address backs many logical weights); and supports_buft=is_host stops sched
    // copying weights at all. Address-keying would serve the wrong resident weight --
    // correct matmul math on the wrong weights.
    //
    // The K<=2048 cutoff is a PERF choice, NOT a correctness one: pack-once pays when a
    // weight is reused across forward passes -- whisper's repeated encoder passes (K<=2048),
    // OR a multi-micro-batch / repeated LLM prefill (any K) once residency is enabled. A
    // single-micro-batch one-shot prefill uses each weight once, so there the mt path -- which
    // frees its BOs per call rather than holding them resident -- is just as fast and avoids
    // the resident-BO footprint. So all-K residency is opt-in:
    //   ROCKET_F16_RESIDENT=auto|N|1 -- the production knob: budget-managed, decode-safe (no
    //     madvise), sized from free RAM (see rocket_f16_resident_on / ggml_backend_rocket_init).
    //   ROCKET_FORCE_PREPACK=1       -- the diagnostic force (default 2GB budget unless
    //     ROCKET_CACHE_MB set; +prefill-only ROCKET_PREPACK_MADVISE). Both HW-validated coherent.
    // This is the per-node resident path; a fusable group (QKV / gate-up) under
    // ROCKET_F16_RESIDENT instead goes to ggml_backend_rocket_mul_mat_group_resident (one
    // resident combined-N weight = residency + fusion stacked). ROCKET_FORCE_PREPACK disables
    // fusion, so it residents every weight here individually.
    static int no_prepack = -1;
    if (no_prepack < 0) { const char * e = getenv("ROCKET_NO_PREPACK"); no_prepack = e ? atoi(e) : 0; }
    static int force_prepack = -1;
    if (force_prepack < 0) { const char * e = getenv("ROCKET_FORCE_PREPACK"); force_prepack = e ? atoi(e) : 0; }
    const bool cacheable = !no_prepack
        && (K <= 2048 || force_prepack || rocket_f16_resident_on())
        && (src0->type == GGML_TYPE_F16)
        && src0->op == GGML_OP_NONE
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->ne[2] == 1 && src1->ne[3] == 1;
    if (cacheable &&
        ggml_backend_rocket_mul_mat_prepacked(ctx, dst, (int)M, (int)K, (int)N) == 0) {
        rocket_mul_mat_post(dst, "prepacked");
        return;
    }

    const int64_t ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t  nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    // broadcast ratios (src0/weights broadcast over src1/input batch dims)
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const int64_t Mp = rocket_pad_m((int)M);   // pad rows to a multiple of 4
    std::vector<ggml_fp16_t> A16((size_t)Mp * K);   // value-initialized: pad rows = 0
    std::vector<float> scales((size_t)M);    // per-row activation scale
    // C16 / B16 are reused context scratch (resize is a no-op once the high-water mark is
    // reached, so the pages stay resident across the prefill -- no per-op re-fault). Both
    // are fully written before read, so reuse is bit-identical to a fresh buffer.
    ctx->scratch_C16.resize((size_t)Mp * N);
    ggml_fp16_t * C16 = ctx->scratch_C16.data();
    const bool b_is_f16 = (src0->type == GGML_TYPE_F16);
    ggml_fp16_t * B16 = nullptr;             // used when weights are F32 / BF16 / quantized
    if (!b_is_f16) { ctx->scratch_B16.resize((size_t)N * K); B16 = ctx->scratch_B16.data(); }

    for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const char * A_src = (const char *)src1->data + i3 * nb13 + i2 * nb12;
            const char * B_src = (const char *)src0->data + (i3 / r3) * nb03 + (i2 / r2) * nb02;
            char       * C_dst = (char *)dst->data + i3 * nb3 + i2 * nb2;

            // A (input, F32) -> per-row scaled fp16 [M,K] (pad rows M..Mp-1 stay zero)
            rocket_pack_activations((const float *)A_src, A16.data(), M, K, scales.data());

            // B (weights) -> fp16 [N,K]. F16 is zero-copy; F32 / BF16 / quantized
            // (Q8_0 / Q4_K / ...) are dequantized transiently into B16 -- the
            // dequant-into-fp16 path, so a compact quantized GGUF prefills on the NPU
            // with no whole-model F16 intermediate. rocket_weight_to_fp16 never
            // reinterprets a bf16/quant payload as f32 (over-read); supports_op already
            // gated the decodable types, so a false return here is a defensive CPU
            // fallback rather than the expected path.
            const ggml_fp16_t * Bp;
            if (b_is_f16) {
                Bp = (const ggml_fp16_t *)B_src;
            } else {
                // Time the streaming dequant into its own ROCKET_MM_PROFILE bucket (see
                // rocket_convprof_add_dequant): it is the dominant host cost here and is
                // invisible to every other profiler.
                const bool wprof = rocket_convprof_on();
                const double wt0 = wprof ? rocket_now_ms() : 0.0;
                const bool ok = rocket_weight_to_fp16(B_src, src0->type, N, K, B16);
                if (wprof) rocket_convprof_add_dequant(rocket_now_ms() - wt0, (double)N * K);
                if (ok) {
                    Bp = B16;
                } else {
                    rocket_cpu_matmul_slice((const float *)A_src, (const void *)B_src,
                                            src0->type, (float *)C_dst, M, N, K);
                    continue;
                }
            }

            // Prefer the streaming context (persistent fds + per-shape resident
            // scratch BOs, B re-packed per call). It also shares the A-pack across
            // workers. On miss (disabled, cache full, or driver error) drop to the
            // per-call mt path; on its failure, to a CPU matmul.
            static int no_stream = -1;
            if (no_stream < 0) { const char * e = getenv("ROCKET_NO_STREAM"); no_stream = e ? atoi(e) : 0; }

            int rc = -1;
            if (!no_stream && !ctx->stream_failed) {
                if (!ctx->stream) {
                    ctx->stream = rocket_stream_create(ctx->n_threads);
                    if (!ctx->stream) ctx->stream_failed = true;
                }
                if (ctx->stream)
                    rc = rocket_matmul_fp16_stream(ctx->stream, (int)Mp, (int)K, (int)N,
                            reinterpret_cast<const _Float16 *>(A16.data()),
                            reinterpret_cast<const _Float16 *>(Bp),
                            reinterpret_cast<_Float16 *>(C16));
            }
            if (rc != 0)
                rc = rocket_matmul_fp16_mt((int)Mp, (int)K, (int)N,
                        reinterpret_cast<const _Float16 *>(A16.data()),
                        reinterpret_cast<const _Float16 *>(Bp),
                        reinterpret_cast<_Float16 *>(C16),
                        ctx->n_threads);
            if (rc != 0) {
                // Don't abort the process: degrade this slice to a correct CPU
                // matmul so llama.cpp/whisper.cpp keep running. Computed
                // from the original f32 inputs, so it ignores the scaled A16 above.
                GGML_LOG_ERROR("%s: rocket_matmul_fp16_mt failed (%d) for M=%lld K=%lld N=%lld"
                               " -> CPU fallback for this slice\n",
                               __func__, rc, (long long)M, (long long)K, (long long)N);
                rocket_cpu_matmul_slice((const float *)A_src, (const void *)B_src,
                                        src0->type, (float *)C_dst, M, N, K);
                continue;
            }

            // C fp16 [M,N] -> dst F32, undoing the per-row activation scale
            rocket_unpack_output(C16, (float *)C_dst, M, N, scales.data());
        }
    }
    rocket_mul_mat_post(dst, "mt");
}

// ---------------------------------------------------------------------------
// Graph-level matmul fusion (Q/K/V, gate/up).
//
// Static-weight projections that share one activation source -- Q/K/V read the
// attention-norm output; gate/up read the FFN-norm output -- can be run as ONE
// matmul whose weights are concatenated along N, then the [M, sum(N)] output is
// split back into each projection's own dst. This collapses several NPU jobs +
// A-packs into one and grows N (better core utilization). It does NOT cut the
// weight-scatter bytes (the weights are distinct, each scattered once) -- that
// floor is hardware-mandatory. We do it at EXECUTION time (not via graph_optimize)
// so the graph and the scheduler's dst allocations are untouched: ggml-backend-
// sched hands graph_compute one rocket-only split at a time, with the shared
// activation as a constant split input, so we just bucket the split's MUL_MATs by
// src[1] and write each member's own dst -- no consumer rewiring, no hazards.
//
// Runs on the STREAMING path only (whisper's prepacked path would need a composite
// per-group cache key -- deferred). ROCKET_NO_FUSE=1 disables for clean A/B.
// ---------------------------------------------------------------------------
static bool rocket_fuse_on(void) {
    static int v = -1;
    if (v < 0) {
        const char * nf = getenv("ROCKET_NO_FUSE");
        const char * ns = getenv("ROCKET_NO_STREAM");      // fusion needs the stream path
        const char * fp = getenv("ROCKET_FORCE_PREPACK");  // prepacked fusion is deferred
        const char * i8 = getenv("ROCKET_INT8");           // int8 routes via the single-node path
        // ROCKET_F16_RESIDENT keeps fusion ON: the group runner routes each fusable group
        // through ggml_backend_rocket_mul_mat_group_resident (one resident combined-N weight),
        // which stacks residency's pack-B-once win with fusion's shared pack-A + single submit.
        // (ROCKET_FORCE_PREPACK -- the 2GB diagnostic -- still turns fusion off via `fp`, staying
        // on the per-node resident path.) ROCKET_NO_FUSE=1 forces per-node for a clean A/B.
        v = ((nf ? atoi(nf) : 0) == 0 && (ns ? atoi(ns) : 0) == 0
             && (fp ? atoi(fp) : 0) == 0 && (i8 ? atoi(i8) : 0) == 0) ? 1 : 0;
    }
    return v > 0;
}

// A MUL_MAT we can fold into the combined stream matmul: a static F16 2D-weight
// GEMM we already offload, in the streaming regime (K>2048, so it isn't on the
// prepacked path). Mirrors the supports_op check; the shared-src1 grouping below
// pairs these up. F32 weights are excluded (the fused driver entry takes _Float16).
//
// Quantized weights are deliberately NOT fused: a quant-aware fused path (per-member
// dequant into fp16 scratch, shared activation pack) is a consistent ~2.8% net LOSS on
// quant pp2048 [HW sweep 2026-06-30, Qwen3.5-9B-Q4_K/Q8_0, interleaved clock-fair].
// Quant prefill is dequant-bound, so fusion's only savings (one shared packA + one fewer
// submit per group) are negligible next to dequanting each member's weight -- which
// fusion does not reduce -- while the combined-N path adds host scratch overhead.
static bool rocket_node_fusable(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * a = op->src[0];   // weights
    const ggml_tensor * b = op->src[1];   // input (the shared activation)
    const int64_t K = a->ne[0], N = a->ne[1], M = b->ne[1];
    return a->op == GGML_OP_NONE
        && a->type == GGML_TYPE_F16
        && b->type == GGML_TYPE_F32
        && op->type == GGML_TYPE_F32
        && ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op)
        && a->ne[0] == b->ne[0]
        && a->ne[2] == 1 && a->ne[3] == 1
        && b->ne[2] == 1 && b->ne[3] == 1
        && (K % 32 == 0) && (N % 16 == 0)
        && K >= 64 && N >= 64 && M >= rocket_min_m()
        && K > 2048;                       // streaming regime (not prepacked)
}

// Resident-fused sibling (defined after this function); tried first when ROCKET_F16_RESIDENT
// is on, with the streaming-fused body below as the fallback.
static int ggml_backend_rocket_mul_mat_group_resident(ggml_backend_rocket_context * ctx,
                                                      ggml_tensor ** nodes, int ng);

// Run a group (>=2) of static-F16 MUL_MATs that share src1 as one fused matmul.
// Returns 0 on success (every member's dst written); <0 to tell the caller to run
// the members individually (NOTHING written, so fallback is clean). Members all
// share src1 => identical M and K; only N (src0->ne[1]) differs, which is exactly
// what we concatenate.
static int ggml_backend_rocket_mul_mat_group(ggml_backend_rocket_context * ctx,
                                             ggml_tensor ** nodes, int ng) {
    if (ng < 2 || ng > ROCKET_MAX_FUSE) return -1;

    // bf16 has no fused path yet -> decline so each member runs individually through
    // ggml_backend_rocket_mul_mat (the bf16 branch). Keeps ROCKET_BF16 all-bf16
    // without the caller needing ROCKET_NO_FUSE.
    if (ctx->bf16_mode < 0) { const char * e = getenv("ROCKET_BF16"); ctx->bf16_mode = e ? atoi(e) : 0; }
    if (ctx->bf16_mode) return -1;

    // Resident-fused: with ROCKET_F16_RESIDENT on, pack the group into one resident
    // combined-N weight (pack-once, reused across micro-batches/prefills). On decline (small
    // one-shot M, over budget, IOVA/floor full) fall through to the streaming-fused path below.
    if (rocket_f16_resident_on()) {
        if (ggml_backend_rocket_mul_mat_group_resident(ctx, nodes, ng) == 0) return 0;
    }

    // The streaming context drives the fused matmul; create it lazily (mirrors the
    // single-node path). On failure, fall back to per-node.
    if (ctx->stream_failed) return -1;
    if (!ctx->stream) {
        ctx->stream = rocket_stream_create(ctx->n_threads);
        if (!ctx->stream) { ctx->stream_failed = true; return -1; }
    }

    const ggml_tensor * src1 = nodes[0]->src[1];     // shared input A[M,K]
    const int64_t K  = src1->ne[0];
    const int64_t M  = src1->ne[1];
    const int     Mp = rocket_pad_m((int)M);

    // Pack the shared activation ONCE (per-row scaled fp16; pad rows stay zero).
    std::vector<ggml_fp16_t> A16((size_t)Mp * K);
    std::vector<float>       scales((size_t)M);
    rocket_pack_activations((const float *)src1->data, A16.data(), M, K, scales.data());

    // Lay out the group's weights along the combined N.
    const _Float16 * Bs[ROCKET_MAX_FUSE];
    int              Ns[ROCKET_MAX_FUSE];
    int64_t          Ntot = 0;
    for (int i = 0; i < ng; i++) {
        const ggml_tensor * w = nodes[i]->src[0];
        Bs[i] = reinterpret_cast<const _Float16 *>(w->data);
        Ns[i] = (int)w->ne[1];
        Ntot += w->ne[1];
    }

    std::vector<ggml_fp16_t> C16((size_t)Mp * Ntot);
    int rc = rocket_matmul_fp16_stream_fused(ctx->stream, Mp, (int)K, Bs, Ns, ng,
                reinterpret_cast<const _Float16 *>(A16.data()),
                reinterpret_cast<_Float16 *>(C16.data()));
    if (rc != 0) return rc;                            // nothing written -> caller falls back

    // Split the combined [M, Ntot] output into each member's own dst (per-row
    // unscale), honouring each dst separately so its consumers see what they expect.
    int64_t col0 = 0;
    for (int i = 0; i < ng; i++) {
        rocket_unpack_output_seg(C16.data(), Ntot, col0,
                                 (float *)nodes[i]->data, M, Ns[i], scales.data());
        col0 += Ns[i];
        rocket_mul_mat_post(nodes[i], "fused");
    }
    return 0;
}

// Resident sibling of ggml_backend_rocket_mul_mat_group: pack a fusable group's weights
// ONCE into a single RESIDENT combined-N weight (Q|K|V or gate|up laid contiguously along
// N), cached under a COMPOSITE key (the members' stable names joined by '|'), and reuse it
// across every micro-batch / prefill. Combines the two independent levers: residency's
// pack-B-once (eliminates the per-call weight scatter) AND fusion's shared activation pack +
// single submit over the combined N. Engaged only when ROCKET_F16_RESIDENT is on (the
// group runner tries it first, then falls back to the streaming-fused path on decline).
//
// The combined resident weight is BYTE-IDENTICAL to the streaming-fused layout: both scatter
// the concatenated [Ntot,K] weight into the driver's (N/16,K/32,16,32) tiles
// (rocket_weights_pack on a host concat == mm_pack_weights_seg on the segments), so greedy
// output matches the streaming path. The concat buffer is transient (copied into the resident
// BOs by rocket_weights_pack, freed on return); the resident weight bytes equal the sum of the
// members' bytes, i.e. no extra RAM over residenting them individually.
//
// Returns 0 (all members' dst written) or <0 to fall back cleanly (nothing written): small
// one-shot M (< max_tile), over budget / IOVA full / MemAvailable floor, no stable weight
// identity, or a driver failure.
static int ggml_backend_rocket_mul_mat_group_resident(
        ggml_backend_rocket_context * ctx, ggml_tensor ** nodes, int ng) {
    if (ctx->dev_failed) return -1;
    if (!ctx->dev) {
        ctx->dev = rocket_ctx_create(ctx->n_threads);
        if (!ctx->dev) { ctx->dev_failed = true; return -1; }
    }

    const ggml_tensor * src1 = nodes[0]->src[1];      // shared input A[M,K]
    const int64_t K  = src1->ne[0];
    const int64_t M  = src1->ne[1];
    const int     Mp = rocket_pad_m((int)M);

    // Composite key = the members' stable weight names joined by '|'. Any member without a
    // stable identity (empty key -- e.g. a ggml auto-name) makes the whole group uncacheable.
    std::string key;
    int64_t Ntot = 0;
    for (int i = 0; i < ng; i++) {
        const std::string mk = rocket_weight_key(nodes[i]->src[0]);
        if (mk.empty()) return -1;
        if (i) key += '|';
        key += mk;
        Ntot += nodes[i]->src[0]->ne[1];
    }

    rocket_weights * w = nullptr;
    auto it = ctx->wcache.find(key);
    if (it != ctx->wcache.end()) {
        if (it->second.K == (int)K && it->second.N == (int)Ntot) {
            w = it->second.w;
        } else {                                      // composite shape drift: re-pack
            rocket_weights_free(ctx->dev, it->second.w);
            ctx->resident_bytes -= it->second.bytes;
            ctx->wcache.erase(it);
        }
    }
    if (!w) {
        // Small one-shot M streams via the caller's streaming-fused fallback (no wasted
        // resident pack); mirrors the single-weight prepacked path's max_tile pivot.
        if (Mp < rocket_hw_current()->max_tile) return -1;
        // Budget / OOM-floor admission -- identical policy to build_resident (single path).
        if (ctx->dev_resident_full) return -1;
        const size_t est = (size_t)Ntot * K * sizeof(ggml_fp16_t);
        if (ctx->cache_budget && ctx->resident_bytes + est > ctx->cache_budget) return -1;
        if (ctx->resident_floor_bytes) {
            const size_t avail = rocket_meminfo_bytes("MemAvailable");
            if (avail && avail < ctx->resident_floor_bytes) { ctx->dev_resident_full = true; return -1; }
        }
        // Concatenate the members' fp16 weights into one [Ntot,K] buffer (member i's ne[1]*K
        // fp16 rows, back to back). All members are F16 + contiguous (rocket_node_fusable).
        std::vector<ggml_fp16_t> comb((size_t)Ntot * K);
        size_t off = 0;
        for (int i = 0; i < ng; i++) {
            const ggml_tensor * a = nodes[i]->src[0];
            const size_t nb = (size_t)a->ne[1] * K;
            memcpy(comb.data() + off, a->data, nb * sizeof(ggml_fp16_t));
            off += nb;
        }
        w = rocket_weights_pack(ctx->dev, Mp, (int)K, (int)Ntot,
                                reinterpret_cast<const _Float16 *>(comb.data()));
        if (!w) { ctx->dev_resident_full = true; return -1; }  // IOVA/alloc full -> stream the rest
        rocketraii::scope_guard w_guard([&] { rocket_weights_free(ctx->dev, w); });
        ctx->wcache[key] = { w, Mp, (int)K, (int)Ntot, est };
        w_guard.dismiss();
        ctx->resident_bytes += est;
    }

    // One shared activation pack (per-row scaled fp16, pad rows zero) + one combined compute.
    std::vector<ggml_fp16_t> A16((size_t)Mp * K);
    std::vector<ggml_fp16_t> C16((size_t)Mp * Ntot);
    std::vector<float>       scales((size_t)M);
    rocket_pack_activations((const float *)src1->data, A16.data(), (int)M, (int)K, scales.data());

    int rc = rocket_matmul_fp16_prepacked(ctx->dev, Mp, (int)K, (int)Ntot,
                reinterpret_cast<const _Float16 *>(A16.data()),
                reinterpret_cast<_Float16 *>(C16.data()), w);
    if (rc != 0) return -1;   // rc==-2 (M below resident tiling) or driver fail -> stream-fused

    // Split the combined [M, Ntot] output back into each member's own dst (per-row unscale).
    int64_t col0 = 0;
    for (int i = 0; i < ng; i++) {
        const int64_t Ni = nodes[i]->src[0]->ne[1];
        rocket_unpack_output_seg(C16.data(), Ntot, col0,
                                 (float *)nodes[i]->data, M, Ni, scales.data());
        col0 += Ni;
        rocket_mul_mat_post(nodes[i], "fused-resident");
    }
    return 0;
}

// ===========================================================================
// Backend / device / registry vtables, op support, and graph compute
// ===========================================================================
// ---------------------------------------------------------------------------
// backend interface
// ---------------------------------------------------------------------------
static const char * ggml_backend_rocket_get_name(ggml_backend_t backend) {
    (void)backend; return "ROCKET";
}

static void ggml_backend_rocket_free(ggml_backend_t backend) {
    ggml_backend_rocket_context * ctx = (ggml_backend_rocket_context *)backend->context;
    if (ctx->stream) rocket_stream_free(ctx->stream);
    if (ctx->int8_fd >= 0) rocket_close(ctx->int8_fd);
    if (ctx->int4_fd >= 0) rocket_close(ctx->int4_fd);
    if (ctx->bf16_stream) rocket_bf16_stream_free(ctx->bf16_stream);
    if (ctx->bf16_fd >= 0) rocket_close(ctx->bf16_fd);
    if (ctx->fa_ctx) rocket_fa_ctx_free(ctx->fa_ctx);
    if (ctx->fa_fd   >= 0) rocket_close(ctx->fa_fd);
    if (ctx->dev) {
        for (auto & kv : ctx->wcache) rocket_weights_free(ctx->dev, kv.second.w);
        rocket_ctx_free(ctx->dev);
    }
    // The resident/streaming split of the native-quant MoE experts, stated rather than
    // left to be inferred: a partly-resident model is a partial win by design (admission
    // only, see rocket_moe_expert_resident), and a run that quietly ingested a third of
    // its experts should not look like a run that ingested all of them.
    if (ctx->moe_n_resident || !ctx->moe_streamed_keys.empty()) {
        const long streamed = (long)ctx->moe_streamed_keys.size();
        const long total    = ctx->moe_n_resident + streamed;
        // The denominator is the experts ACTUALLY EXERCISED (an expert the router never sent a
        // row to is never looked up), not the model's expert count -- and that is the right
        // denominator for this ratio: the dequant tax is paid per (op, expert-with-rows) and
        // costs a full [N,K] decode regardless of how many rows that expert got. So this
        // percentage is the fraction of the dequant actually removed, not a residency fraction.
        //
        // rocket_log, not GGML_LOG_*: llama-bench silences ggml, and this line is the one that
        // explains the number llama-bench just printed.
        const double res_pct = total ? 100.0 * (double)ctx->moe_n_resident / (double)total : 0.0;
        ROCKET_LOGI("[moe-int8] experts exercised: %ld resident on the NPU (%zuMB), %ld streamed "
                    "via dequant->fp16 -- %.0f%% of the per-micro-batch dequant removed\n",
                    ctx->moe_n_resident, ctx->moe_i8_resident_bytes >> 20, streamed, res_pct);
        // Say what that number MEANS, because the relationship is not the linear one it looks
        // like. A streamed expert pays a weight dequant that is INDEPENDENT of its row count
        // (it decodes the whole [N,K] whatever the router gave it), while a resident one pays a
        // GEMM that shrinks with M. So the streamed remainder's share of the wall clock grows as
        // the prefill shortens, and the route falls off a cliff rather than degrading smoothly:
        // measured on gpt-oss at pp512, 99% resident is 17.6 t/s and 82% resident is 12.2 --
        // below the 14.1 you get by simply leaving the experts on the CPU. Residency is not a
        // nice-to-have for this route; it IS the route.
        if (res_pct < 95.0)
            ROCKET_LOGW("[moe-int8] only %.0f%% resident -- below ~95%% this route is typically a "
                        "net LOSS at short prefill (a streamed expert's dequant does not shrink "
                        "with the row count). Raise ROCKET_MOE_CACHE_MB if the RAM is there, or "
                        "set ROCKET_MOE=0 to leave the experts on the CPU.\n", res_pct);
        // What that residency cost, once: the price of admission to the route above. It is
        // paid inside the first prefill and it is minutes on a large MoE, so it is reported
        // next to the win rather than left to be discovered as a startup hang. It is also
        // paid PER CONTEXT, which is why a tool that builds a fresh context per measurement
        // (llama-bench does) pays it per row -- see the ingest note in the README.
        if (ctx->moe_ingest_ms + ctx->moe_pack_ms > 0)
            ROCKET_LOGI("[moe-int8] one-time ingest: %.1fs total (%.1fs GGUF->int8 decode, "
                        "%.1fs NPU-BO pack) for %ld experts = %.0fms each\n",
                        (ctx->moe_ingest_ms + ctx->moe_pack_ms) / 1000.0,
                        ctx->moe_ingest_ms / 1000.0, ctx->moe_pack_ms / 1000.0,
                        ctx->moe_n_resident,
                        ctx->moe_n_resident
                            ? (ctx->moe_ingest_ms + ctx->moe_pack_ms) / (double)ctx->moe_n_resident
                            : 0.0);
    }
    if (ctx->i8_dev) {   // resident int8 weights hold BOs on the ctx fds -> free first
        for (auto & kv : ctx->i8_rwcache)   rocket_i8_weights_free(ctx->i8_dev, kv.second.w);
        for (auto & kv : ctx->moe_i8_cache) rocket_i8_weights_free(ctx->i8_dev, kv.second.w);
        rocket_i8_ctx_free(ctx->i8_dev);
    }
    if (ctx->i4_dev) {   // resident int4 weights hold BOs on the ctx fds -> free first
        for (auto & kv : ctx->i4_rwcache) rocket_i4_weights_free(ctx->i4_dev, kv.second.w);
        rocket_i4_ctx_free(ctx->i4_dev);
    }
    if (ctx->rk76_fd >= 0) {   // rk3576 device weight cubes + any pooled transient BOs
        for (auto & kv : ctx->rk76_wcache)
            for (auto & q : kv.second.ch)
                if (q.wbo) { rocket_rk3576_wbo_free(ctx->rk76_fd, q.wbo); q.wbo = nullptr; }
        rocket_rk3576_bo_pool_drain(ctx->rk76_fd);
    }
    delete ctx;
    delete backend;
}

// ---------------------------------------------------------------------------
// FLASH_ATTN_EXT (LLM prefill attention)
//
// llama.cpp fuses attention into one GGML_OP_FLASH_ATTN_EXT op (the QK/softmax/AV
// matmuls never appear as separate nodes), so the LLM attention offload is this op,
// not a mul_mat interception. The on-NPU per-head QK -> mask -> softmax -> P*V lives in
// librocketnpu (rocket_flash_attn_fp16); here we gather the op's strided/permuted
// ggml operands into the dense head-major fp16 layout that primitive expects, run it,
// and scatter the F32 result. The causal + sliding-window mask is an INPUT (src3) we
// apply additively -- the backend does not synthesise it.
//
// op layout (ggml_flash_attn_ext): src0=Q F32 [head_dim, n_tokens, n_head, 1] (permuted),
// src1=K / src2=V F16 [head_dim, n_kv, n_kv_heads, 1] (strided cache views), src3=mask F16
// [n_kv, n_tokens(padded), 1, 1]; dst F32 [head_dim, n_head, n_tokens, 1] contiguous.
// op_params = {scale, max_bias, logit_softcap}. Returns 0, or <0 to fall back to CPU.
static int ggml_backend_rocket_flash_attn(ggml_backend_rocket_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    const int head_dim   = (int)q->ne[0];   // dk: Q/K head dim (QK contraction)
    const int dv         = (int)v->ne[0];    // dv: V/output head dim (== head_dim for GQA; MLA differs)
    const int n_tokens   = (int)q->ne[1];
    const int n_head     = (int)q->ne[2];
    const int n_kv       = (int)k->ne[1];
    const int n_kv_heads = (int)k->ne[2];

    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale,    (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *)dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) return -1;            // ALiBi unsupported (rope models pass 0)
    if (!m) return -1;                          // require an explicit mask (the causal path)

    // Lazily create the persistent FA context (worker fds + resident scratch); if it can't
    // open its fds, fall through to the lazy single fa_fd (and, failing that, the host
    // reference) — supports_op already accepted the op, so it must produce a result.
    if (rocket_flash_attn_no_ctx()) ctx->fa_ctx_failed = true;   // A/B: force per-call mt path
    if (!ctx->fa_ctx_failed && !ctx->fa_ctx) {
        ctx->fa_ctx = rocket_fa_ctx_create(ctx->n_threads);
        if (!ctx->fa_ctx) ctx->fa_ctx_failed = true;
    }
    if (rocket_debug_on()) {
        static int once = 0;
        if (!once) { once = 1;
            GGML_LOG_DEBUG("rocket FA: path=%s (n_threads=%d)\n",
                    ctx->fa_ctx ? "persistent-ctx" : "per-call-mt", ctx->n_threads); }
    }
    if (!ctx->fa_ctx && !ctx->fa_failed && ctx->fa_fd < 0) {
        ctx->fa_fd = rocket_open();
        if (ctx->fa_fd < 0) ctx->fa_failed = true;
    }
    const int use_fd = ctx->fa_failed ? -1 : ctx->fa_fd;

    // Gather the op's strided/permuted operands into dense head-major fp16 tiles.
    const bool fatiming = rocket_fatiming_on();
    const double t_g0 = fatiming ? rocket_now_ms() : 0.0;
    std::vector<ggml_fp16_t> Qd((size_t)n_head     * n_tokens * head_dim);   // [n_head][n_tokens][dk]
    std::vector<ggml_fp16_t> Kd((size_t)n_kv_heads * n_kv     * head_dim);   // [n_kv_heads][n_kv][dk]
    std::vector<ggml_fp16_t> Vd((size_t)n_kv_heads * dv       * n_kv);       // [n_kv_heads][dv][n_kv]
    std::vector<ggml_fp16_t> Md((size_t)n_tokens   * n_kv);
    std::vector<ggml_fp16_t> Od((size_t)n_head     * n_tokens * dv);         // [n_head][n_tokens][dv]

    const char * qb = (const char *)q->data;
    for (int h = 0; h < n_head; h++)
        for (int t = 0; t < n_tokens; t++) {
            ggml_fp16_t * dstrow = Qd.data() + ((size_t)h * n_tokens + t) * head_dim;
            const char * src = qb + (size_t)t*q->nb[1] + (size_t)h*q->nb[2];
            for (int c = 0; c < head_dim; c++)   // Q is F32
                dstrow[c] = ggml_fp32_to_fp16(*(const float *)(src + (size_t)c*q->nb[0]));
        }
    const char * kb = (const char *)k->data;   // K F16 -> [n_kv_heads][n_kv][head_dim]
    for (int hk = 0; hk < n_kv_heads; hk++)
        for (int j = 0; j < n_kv; j++) {
            ggml_fp16_t * dstrow = Kd.data() + ((size_t)hk * n_kv + j) * head_dim;
            const char * src = kb + (size_t)j*k->nb[1] + (size_t)hk*k->nb[2];
            for (int c = 0; c < head_dim; c++)
                dstrow[c] = *(const ggml_fp16_t *)(src + (size_t)c*k->nb[0]);
        }
    const char * vb = (const char *)v->data;   // V F16 -> [n_kv_heads][dv][n_kv]
    for (int hk = 0; hk < n_kv_heads; hk++)
        for (int c = 0; c < dv; c++) {
            ggml_fp16_t * dstrow = Vd.data() + ((size_t)hk * dv + c) * n_kv;
            const char * src = vb + (size_t)c*v->nb[0] + (size_t)hk*v->nb[2];
            for (int j = 0; j < n_kv; j++)
                dstrow[j] = *(const ggml_fp16_t *)(src + (size_t)j*v->nb[1]);
        }
    const char * mb = (const char *)m->data;   // mask F16 [n_kv, n_tokens] -> [n_tokens][n_kv]
    for (int t = 0; t < n_tokens; t++) {
        ggml_fp16_t * dstrow = Md.data() + (size_t)t * n_kv;
        const char * src = mb + (size_t)t*m->nb[1];
        for (int j = 0; j < n_kv; j++)
            dstrow[j] = *(const ggml_fp16_t *)(src + (size_t)j*m->nb[0]);
    }
    const double t_g1 = fatiming ? rocket_now_ms() : 0.0;

    // Fan the heads across the worker fds (one drm entity per fd -> the NPU cores run
    // the head ranges in parallel). Running all heads on one fd serializes the per-head
    // submits (the dispatch-bound bottleneck); ctx->n_threads is the same worker count the
    // weight-GEMM path tuned to (5: 3 cores + 2 to fill the gather/softmax bubbles). The
    // persistent fa_ctx reuses its worker fds + resident scratch across calls; only if its
    // create failed do we fall to the per-call _mt path (own fds, per-call scratch).
    int rc = ctx->fa_ctx
        ? rocket_flash_attn_fp16_ctx(ctx->fa_ctx, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                                    scale, softcap,
                                    reinterpret_cast<const _Float16 *>(Qd.data()),
                                    reinterpret_cast<const _Float16 *>(Kd.data()),
                                    reinterpret_cast<const _Float16 *>(Vd.data()),
                                    reinterpret_cast<const _Float16 *>(Md.data()),
                                    reinterpret_cast<_Float16 *>(Od.data()))
        : rocket_flash_attn_fp16_mt(use_fd, n_tokens, n_kv, head_dim, dv, n_head, n_kv_heads,
                                    scale, softcap,
                                    reinterpret_cast<const _Float16 *>(Qd.data()),
                                    reinterpret_cast<const _Float16 *>(Kd.data()),
                                    reinterpret_cast<const _Float16 *>(Vd.data()),
                                    reinterpret_cast<const _Float16 *>(Md.data()),
                                    reinterpret_cast<_Float16 *>(Od.data()), ctx->n_threads);
    const double t_c1 = fatiming ? rocket_now_ms() : 0.0;
    if (rc != 0) return rc;

    // scatter Od [n_head][n_tokens][dv] -> dst F32 [dv, n_head, n_tokens]
    char * db = (char *)dst->data;
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < n_head; h++) {
            const ggml_fp16_t * srcrow = Od.data() + ((size_t)h * n_tokens + t) * dv;
            char * dr = db + (size_t)h*dst->nb[1] + (size_t)t*dst->nb[2];
            for (int c = 0; c < dv; c++)
                *(float *)(dr + (size_t)c*dst->nb[0]) = ggml_fp16_to_fp32(srcrow[c]);
        }
    if (fatiming) rocket_fatiming_add(t_g1 - t_g0, t_c1 - t_g1, rocket_now_ms() - t_c1, n_kv);
    return 0;
}

// ===========================================================================
// Native-quant MoE experts: GGUF quant blocks -> resident int8 on the NPU
//
// A quantized MoE expert costs a full host dequant->fp16 of its [N,K] weight EVERY
// micro-batch (rocket_weight_to_fp16), and a MoE layer has n_expert times more distinct
// weights than a dense one -- which is why the fp16 MUL_MAT_ID offload is a net loss on
// every MoE that actually ships quantized. The fix is to stop dequantizing: ingest each
// expert ONCE into int8 codes, leave them resident in NPU BOs, and quantize only the
// activation per call.
//
// The NPU cannot apply a K-blocked scale on chip -- at the output stage K is fully
// contracted, so no register or operand cube is indexed by a K-block, for any dtype. But
// integer partials ALREADY leave the chip at every K-tile boundary (on-device integer
// K-accumulation is architecturally impossible: the DPU eltwise operand DMA is <=16-bit
// and an int32 partial does not fit), so a per-K-group scale rides along for free at a
// boundary that is already being paid for. Keep each K-tile inside one quant group,
// multiply its int32 partial by that group's scale, accumulate in fp32 on the host --
// which is exactly rocket_matmul_int8_prepacked_gw's contract, and why the ingest emits
// per-(output-channel, K-group) scales rather than one scale per channel.
//
// WHERE THE SPEED COMES FROM, precisely: NOT from the quantization. The int8 GEMM's int32
// output is read back at 8 B/element (the HW output-cube stride), so at the group the CBUF
// allows it moves MORE bytes than the equivalent fp16 GEMM. The entire win is deleting the
// per-micro-batch dequant and weight scatter. Quantization here buys residency, and
// residency buys the speed. A change that reintroduces a per-call pass over the weight --
// or a separate scale-and-convert pass over the int32 output -- hands the win straight back.
// ===========================================================================

// The MXFP4 codebook, mirroring kvalues_mxfp4 (ggml-common.h). The mantissa is DECLARED
// int8_t and reaches only +/-12, so an MXFP4 code IS an exact int8 value with 3 bits of
// headroom under int8's +/-127: the weight side of a native-int8 expert path is a
// shift-and-copy, not a quantization -- and strictly MORE faithful than the fp16->int8
// rounding rocket_quant_wt_int8 does. Mirrored rather than pulled in via
// GGML_COMMON_IMPL_CPP so we don't drag ggml-common's other (large, unused) codebooks
// into the .so; the block STRUCT still comes from the real header, so a layout change is
// a build error. rk_moe_ingest_mxfp4_row asserts the block geometry at compile time.
static const int8_t rk_mxfp4_kvalues[16] = { 0, 1, 2, 3, 4, 6, 8, 12,
                                             0, -1, -2, -3, -4, -6, -8, -12 };

// Move one MXFP4 code from its own block's exponent onto the merged group's reference
// exponent. sh = e_block - e_ref.
//
// A LEFT shift is exact and cannot overflow int8: e_ref >= e_max - 3, so sh <= 3 for any
// code that is nonzero (a nonzero code implies a nonzero block, and only nonzero blocks
// enter e_max), and |code| <= 12 gives 12 << 3 = 96 <= 127.
// A RIGHT shift rounds to nearest (half away from zero) and can only shrink the value.
// A zero code stays zero for ANY sh -- which is what keeps an all-zero block, whose
// exponent is meaningless and is therefore excluded from the group's band, from shifting
// by an out-of-range amount.
static inline int8_t rk_mxfp4_shift(int code, int sh) {
    if (code == 0) return 0;
    if (sh >= 0) {
        // sh <= 3 here (see above). Clamp anyway: a future change to the e_ref rule must
        // not be able to turn this into an undefined shift.
        return (int8_t)(code << (sh < 3 ? sh : 3));
    }
    const int s = -sh;
    if (s >= 5) return 0;                 // |code| <= 12 < 2^4, so code/2^5 rounds to 0
    const int half = 1 << (s - 1);
    const int mag  = ((code < 0 ? -code : code) + half) >> s;
    return (int8_t)(code < 0 ? -mag : mag);
}

// Ingest ONE row of an MXFP4 expert weight: merge each K-group's `group/32` native blocks
// onto a single exponent and emit the group's int8 codes + its fp32 scale.
//
// MXFP4's block scale is E8M0 -- an EXACT power of two, 2^(e-128) -- so the merge is an
// integer shift of the codes, not a requantization. The group's reference exponent is
//
//     e_ref = max(e_min, e_max - 3)
//
// and NOT "clamp everything onto e_min". The choice is load-bearing for the ~0.1% of
// groups whose exponent band is wider than 3 octaves, because it decides WHERE their
// error lands. Under e_ref every block at or above e_ref left-shifts exactly, and a block
// further down right-shifts (rounding) instead -- so the error falls on the group's
// SMALLEST weights. Clamping to e_min would instead have to clip the group's LARGEST
// weights (a 12<<9 saturated to 127 is a 48x underestimate), and measures 19x less
// faithful: cosine 0.9958 (clamp) vs 0.9998 (e_ref) at group=576, pooled over all 72
// gpt-oss expert tensors.
//
// The merge is EXACTLY lossless whenever the group's spread is <= 3 octaves, which holds
// for 99.9% of groups at group=576 [offline sweep, 597M native blocks]. No other GGUF
// format offers this lever -- Q4_K's d/dmin are fp16, not powers of two -- which is why
// every other type takes the dequantize-once-then-requantize route below.
static void rk_moe_ingest_mxfp4_row(const void * W_src, int64_t n, int64_t K, int group,
                                    int nG, int8_t * qB, float * b_scale) {
    static_assert(sizeof(block_mxfp4) == 17, "MXFP4 block layout changed");
    const int nblk_g = group / 32;                   // native blocks per merged group
    const block_mxfp4 * row = (const block_mxfp4 *)W_src + (size_t)n * (K / 32);
    int8_t * qrow = qB      + (size_t)n * K;
    float  * srow = b_scale + (size_t)n * nG;

    for (int g = 0; g < nG; g++) {
        const block_mxfp4 * blk = row + (size_t)g * nblk_g;

        // Pass 1: the group's exponent band, over the NONZERO blocks only. An all-zero
        // block constrains nothing (0 shifts to 0) and its exponent is arbitrary, so
        // letting it into the band would widen it for no reason. A code is zero iff its
        // nibble's low 3 bits are zero (kvalues[0] == kvalues[8] == 0).
        int e_min = 255, e_max = -1;
        for (int b = 0; b < nblk_g; b++) {
            bool nz = false;
            for (int j = 0; j < 16; j++)
                if (blk[b].qs[j] & 0x77) { nz = true; break; }
            if (!nz) continue;
            const int e = blk[b].e;
            if (e < e_min) e_min = e;
            if (e > e_max) e_max = e;
        }
        if (e_max < 0) {                             // the whole group is zero
            srow[g] = 1.0f;
            memset(qrow + (size_t)g * group, 0, (size_t)group);
            continue;
        }

        int e_ref = e_max - 3;
        if (e_ref < e_min) e_ref = e_min;
        // The SAME conversion ggml uses for a block scale, so the merged scale is exactly
        // the power of two the source blocks were built on -- no fp32 re-derivation.
        srow[g] = GGML_E8M0_TO_FP32_HALF((uint8_t)e_ref);

        // Pass 2: shift each block's codes onto e_ref. Byte j of a block holds element j
        // in its low nibble and element j+16 in its high nibble (ggml's MXFP4 packing).
        for (int b = 0; b < nblk_g; b++) {
            const int sh = (int)blk[b].e - e_ref;
            int8_t * out = qrow + ((size_t)g * nblk_g + b) * 32;
            for (int j = 0; j < 16; j++) {
                const uint8_t byte = blk[b].qs[j];
                out[j]      = rk_mxfp4_shift(rk_mxfp4_kvalues[byte & 0x0F], sh);
                out[j + 16] = rk_mxfp4_shift(rk_mxfp4_kvalues[byte >> 4],   sh);
            }
        }
    }
}

// Ingest ONE row of a NON-MXFP4 quantized expert weight: dequantize it with the type's own
// ggml dequantizer (bit-identical to what the CPU backend reads), then symmetric-int8
// quantize it per K-group.
//
// The MXFP4 merge lever is a BONUS, not a requirement. What costs is a dequant per
// MICRO-BATCH; a one-time one at ingest does not, so ANY format can be dequantized once
// and requantized to whatever group the CBUF wants. int8 at group ~512 carries 255 levels
// where Q4_K has 16 per 32-element sub-block, so the requant is FINER than the source
// quantization (cosine 0.99997 on DeepSeek's Q4_K experts, 0.999976 on its Q8_0 ones).
// The asymmetric min of a K-quant needs no special algebra here either: dequantizing folds
// the offset in before the requant. This also covers DeepSeek-V2-Lite's MIXED-format expert
// stacks (Q4_K gate/up, Q8_0 *and* Q5_0 down, by layer), which a format-specific ingest
// would not.
//
// `frow` is caller-owned scratch of K floats (one per worker thread).
static bool rk_moe_ingest_generic_row(const void * W_src, ggml_type wt, int64_t n, int64_t K,
                                      int group, int nG, int8_t * qB, float * b_scale,
                                      float * frow) {
    const ggml_type_traits * tr = ggml_get_type_traits(wt);
    if (!tr || !tr->to_float) return false;
    const size_t rbytes = ggml_row_size(wt, K);      // bytes of one [K] quantized row
    tr->to_float((const char *)W_src + (size_t)n * rbytes, frow, K);

    int8_t * qrow = qB      + (size_t)n * K;
    float  * srow = b_scale + (size_t)n * nG;
    for (int g = 0; g < nG; g++) {
        const float * src = frow + (size_t)g * group;
        float amax = 0.0f;
        for (int k = 0; k < group; k++) { const float v = fabsf(src[k]); if (v > amax) amax = v; }
        const float s   = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        int8_t * d = qrow + (size_t)g * group;
        for (int k = 0; k < group; k++) d[k] = rocket_q8(src[k], inv);
        srow[g] = s;
    }
    return true;
}

// Ingest a whole [N,K] quantized expert weight into int8 codes + [N*nG] fp32 group scales.
// Rows are independent (a private scratch in, a disjoint output slice out), so they fan
// across the persistent dequant pool -- this runs once per expert at model load, over
// n_expert * 3 * n_layer weights, so the serial cost would be minutes. Bit-identical to
// the serial loop (no cross-row state). Returns false for a type ggml cannot decode, and
// the caller then leaves that expert on the fp16 route.
static bool rocket_moe_ingest_int8(const void * W_src, ggml_type wt, int64_t N, int64_t K,
                                   int group, int8_t * qB, float * b_scale) {
    if (!ggml_is_quantized(wt) || group <= 0 || K % group) return false;
    const int nG      = (int)(K / group);
    const bool mxfp4  = (wt == GGML_TYPE_MXFP4) && (ggml_blck_size(wt) == 32)
                                                && (ggml_type_size(wt) == sizeof(block_mxfp4));
    std::atomic<bool> ok(true);

    auto ingest_rows = [&](int64_t n0, int64_t n1) {
        if (mxfp4) {
            for (int64_t n = n0; n < n1; n++)
                rk_moe_ingest_mxfp4_row(W_src, n, K, group, nG, qB, b_scale);
            return;
        }
        std::vector<float> frow((size_t)K);
        for (int64_t n = n0; n < n1 && ok.load(std::memory_order_relaxed); n++)
            if (!rk_moe_ingest_generic_row(W_src, wt, n, K, group, nG, qB, b_scale, frow.data()))
                ok.store(false, std::memory_order_relaxed);
    };

    const int nthr = rocket_dequant_threads();
    if (N < 64 || nthr <= 1) { ingest_rows(0, N); return ok.load(); }
    rocket_dequant_pool & pool = rocket_get_dequant_pool();
    const int64_t per = (N + pool.size() - 1) / pool.size();
    pool.run([&](int i) {
        const int64_t n0 = (int64_t)i * per, n1 = n0 + per > N ? N : n0 + per;
        if (n0 < n1) ingest_rows(n0, n1);
    });
    return ok.load();
}

// Quantize one activation row's K-group to symmetric int8 + its scale (= amax/127).
// [-127,127], NOT -128, so +/- are symmetric and the scale is exact both ways -- the same
// arithmetic as rocket_quant_act_int8, only the scale granularity differs (the activation
// must be quantized on the SAME K-group grid as the weight, since the matmul applies
// a_scale[m,g] * b_scale[n,g] to K-group g's int32 partial as it reads it back).
//
// This is the LAST host cost on the native-quant path: the weight ingest is one-time and
// the per-K-group weight scale fuses into the readback loop the integer partials already
// force (+0.6%, measured). Hence the NEON. The float pre-clamp to +/-127 makes the
// saturating narrows exact and keeps this bit-identical to the scalar tail: values already
// satisfy |x*inv| <= 127 by construction (inv = 127/amax), and vcvtnq_s32_f32 rounds
// ties-to-even exactly as lrintf does under the default rounding mode.
static inline void rk_quant_act_i8_group(const float * src, int8_t * dst, int group,
                                         float * pscale) {
    float amax = 0.0f;
    int k = 0;
#ifdef ROCKET_NEON_F32
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (; k + 4 <= group; k += 4) vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(src + k)));
    amax = vmaxvq_f32(vmax);
#endif
    for (; k < group; k++) { const float v = fabsf(src[k]); if (v > amax) amax = v; }

    const float s   = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    *pscale = s;

    k = 0;
#ifdef ROCKET_NEON_F32
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vhi  = vdupq_n_f32(127.0f), vlo = vdupq_n_f32(-127.0f);
    auto q4 = [&](const float * p) {
        return vcvtnq_s32_f32(vminq_f32(vmaxq_f32(vmulq_f32(vld1q_f32(p), vinv), vlo), vhi));
    };
    for (; k + 16 <= group; k += 16) {
        const int16x8_t s0 = vcombine_s16(vqmovn_s32(q4(src + k     )), vqmovn_s32(q4(src + k +  4)));
        const int16x8_t s1 = vcombine_s16(vqmovn_s32(q4(src + k +  8)), vqmovn_s32(q4(src + k + 12)));
        vst1q_s8(dst + k, vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)));
    }
#endif
    for (; k < group; k++) dst[k] = rocket_q8(src[k], inv);
}

// A[M,K] f32 -> int8 [M,K] + per-(row, K-group) scale a_scale[m*nG + g]. Rows are
// independent, so they fan across the persistent dequant pool for a prefill-sized M
// (bit-identical to the serial loop). Pad rows M..Mp are the caller's business.
static void rocket_quant_act_int8_grouped(const float * src, int8_t * dst, int64_t M,
                                          int64_t K, int group, float * a_scale) {
    const bool prof = rocket_convprof_on();
    const double t0 = prof ? rocket_now_ms() : 0.0;
    const int nG = (int)(K / group);

    auto quant_rows = [&](int64_t m0, int64_t m1) {
        for (int64_t m = m0; m < m1; m++)
            for (int g = 0; g < nG; g++)
                rk_quant_act_i8_group(src + m * K + (size_t)g * group,
                                      dst + m * K + (size_t)g * group,
                                      group, a_scale + m * nG + g);
    };

    const int nthr = rocket_dequant_threads();
    if (M < 64 || nthr <= 1) {
        quant_rows(0, M);
    } else {
        rocket_dequant_pool & pool = rocket_get_dequant_pool();
        const int64_t per = (M + pool.size() - 1) / pool.size();
        pool.run([&](int i) {
            const int64_t m0 = (int64_t)i * per, m1 = m0 + per > M ? M : m0 + per;
            if (m0 < m1) quant_rows(m0, m1);
        });
    }
    // Counted on the MoE line, not the dense W8A8 one: this quant is reached only from
    // ggml_backend_rocket_mul_mat_id, and it is the host lever that route is judged on.
    if (prof) { rocket_moeprof_arm(); g_moeprof.quant += rocket_now_ms() - t0; }
}

// The K-group the native-quant expert path quantizes on, for a given (K,N). 0 = none legal
// (the caller then leaves this weight on the fp16 route).
//
// Readback (~ M*N*nKt, nKt = K/Kt) is what these integer paths are bound by. A K-tile must
// lie wholly inside one quant group, so the group UPPER-BOUNDS the driver's K-tile and the
// CBUF caps it. The best group is therefore the LARGEST divisor of K that is a multiple of
// 32 and that the group-wise planner can still serve with Kt == group: that reaches the
// readback floor, and no larger group can beat it (a group too wide for the CBUF is split
// into several K-tiles, which costs readback without buying accuracy). For gpt-oss's
// K=2880 that is 576 (nKt=5); for K=2048 it is 512 (nKt=4).
//
// Probe the SHIPPED planner rather than re-deriving its CBUF arithmetic here -- the cap is
// a machine parameter (CBUF banks x tile geometry), not a constant to copy. Pure and
// cheap: a few dozen calls to a pure function, once per MUL_MAT_ID op.
//
// The native MXFP4 block size (group=32) is NOT the operating point and never can be: it
// would mean nKt=90 and a readback that swamps the whole win.
static int rocket_moe_pick_group(int K, int N) {
    const int max_tile = rocket_hw_current()->max_tile;   // the canonical resident tile M
    auto legal = [&](int g) -> bool {
        if (g < 32 || g % 32 || K % g) return false;
        int Mt, Kt, Nt;
        if (rocket_matmul_plan_int8_gw(max_tile, K, N, g, &Mt, &Kt, &Nt) < 0) return false;
        return Kt == g;                                   // group kept whole -> readback floor
    };
    const int forced = rocket_moe_group_env();
    if (forced > 0) {
        // An explicit group is honoured even when the CBUF must split it across K-tiles
        // (that is legal -- just slower); only a shape-illegal group is rejected.
        int Mt, Kt, Nt;
        if (forced % 32 == 0 && K % forced == 0
            && rocket_matmul_plan_int8_gw(max_tile, K, N, forced, &Mt, &Kt, &Nt) >= 0)
            return forced;
        GGML_LOG_WARN("%s: ROCKET_MOE_GROUP=%d is not legal for K=%d N=%d "
                      "(need %%32 and to divide K) -> auto\n", __func__, forced, K, N);
    }
    for (int g = (K / 32) * 32; g >= 32; g -= 32)
        if (legal(g)) return g;
    return 0;
}

// Round an expert's ragged row count up to a coarse bucket. TWO hard constraints meet here,
// and one bucket satisfies both:
//
//  - M%4 is a HW contract. The resident paths reject an unaligned M outright, because an
//    unaligned M does not merely tile badly -- it MISCOMPUTES (the matmul's rows are the
//    conv's spatial height, and a height below 4 is broken silicon geometry). A router
//    hands out whatever row count it likes, so this is the shape that actually occurs.
//    Pad rows quantize to zero and contribute nothing.
//  - The driver caches its resident scratch per (M,K,N,group) in a FIXED-SIZE table (32
//    slots), each slot holding its own NPU BOs. A distinct M per (expert, layer,
//    micro-batch) would exhaust that table partway through a single forward pass, after
//    which every remaining expert's matmul returns -1 and degrades to the CPU -- which
//    reads as "the win didn't materialize", not as a bug.
//
// M_e is rounded up to a rung of a FIXED ladder with two steps per octave, starting at the
// granule (default 64):
//
//     64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, ...
//
// Two properties, and both are the whole point:
//
//   BOUNDED BY CONSTRUCTION. Two rungs per octave means every M a router can produce
//   (M_e <= n_tokens * n_expert_used, so a few thousand at most) lands on one of ~15 values.
//   With room for a couple of distinct (K,N) expert shapes that stays comfortably inside the
//   driver's 32-slot table -- and it is bounded with no state that can be got wrong.
//
//   PADDING IS CAPPED. A rung is at most 1.5x the one below, so a bucket never exceeds M by
//   more than ~33%, and is ~15% over on average. Pad rows quantize to zero and contribute
//   nothing to the result, but the GEMM still computes and reads them back, so they are paid
//   in full and the cap matters.
//
// This REPLACES an adaptive granule that doubled whenever the distinct-slot count neared the
// cap, and that design had a fatal RATCHET. The slot set never shrinks -- the driver's scratch
// slots are permanent, it has no eviction -- so once the set passed the cap, the "do we still
// have headroom" test could never become true again. Every subsequent new bucket therefore
// doubled the granule, in a loop, until it hit its 4096 ceiling on the very first overflow.
// After that a 356-row expert computed 4096 rows: 88% of every expert GEMM was padding, and the
// native route measured 6.11 t/s at pp2048 where it should be several times that. The bug was
// invisible without an explicit padded-row counter, because nothing failed -- it just quietly
// did 8x the arithmetic. A fixed ladder cannot ratchet, because there is nothing to adapt.
static inline uint64_t rk_moe_slot_key(int Mb, int K, int N) {
    return ((uint64_t)(uint32_t)Mb << 42) | ((uint64_t)(uint32_t)K << 21) | (uint32_t)N;
}
static int rocket_moe_bucket_m(ggml_backend_rocket_context * ctx, int M, int K, int N) {
    if (ctx->moe_m_granule == 0) ctx->moe_m_granule = rocket_moe_m_bucket_env();
    const int g = ctx->moe_m_granule;              // a power of two >= 4 (see the env reader)

    int Mb = g;
    if (M > g) {
        Mb = (M + 3) & ~3;                          // absurd M: fall through, still M%4-legal
        for (int p = g; p <= (1 << 22); p *= 2) {
            // The 1.5x rung between p and 2p, rounded up onto the M%4 HW contract. That
            // rounding is load-bearing at a small granule: at g=4 the raw rung would be 6,
            // which the driver rejects (-1) and the expert would silently fall back to fp16.
            const int mid = (p + p / 2 + 3) & ~3;
            if (M <= mid)   { Mb = mid;   break; }
            if (M <= p * 2) { Mb = p * 2; break; }
        }
    }

    // Track the distinct (M,K,N) slots we have asked the driver for. Purely diagnostic now that
    // the ladder bounds them -- but warn once if we ever near the table, because the failure it
    // would cause (rki_ctx_scratch returns NULL, the expert degrades to fp16) reads as "the win
    // didn't materialize" rather than as a fault. Shapes past 2^21 cannot be packed into the
    // key; they are far outside any real model, so skip the accounting rather than mis-key it.
    if (K < (1 << 21) && N < (1 << 21)) {
        ctx->moe_slots.insert(rk_moe_slot_key(Mb, K, N));
        if (ctx->moe_slots.size() == 28)
            GGML_LOG_WARN("[moe-int8] %zu distinct (M,K,N) scratch slots, near the driver's 32 "
                          "-- further shapes will fall back to the fp16 route. Raise "
                          "ROCKET_MOE_M_BUCKET to collapse them.\n", ctx->moe_slots.size());
    }
    return Mb;
}

// Fetch (or build) expert `e`'s resident int8 weight. Returns nullptr to tell the caller to
// stream this expert on the fp16 dequant path -- over budget, IOVA window full, no stable
// weight name, or an undecodable type. All of those are SAFE: the fp16 route reads the
// untouched GGUF source (the native path never madvises it away; see below).
//
// ADMISSION ONLY, no eviction, and that is the correct policy rather than a missing feature:
// prefill touches EVERY expert EVERY micro-batch, so there is no hotness for an eviction
// policy to exploit -- the only thing that decides how much of the dequant tax is removed
// is the total resident bytes. A 60%-resident model removes 60% of the tax; the blend is a
// win, not a cliff.
static const rocket_moe_i8_expert * rocket_moe_expert_resident(
        ggml_backend_rocket_context * ctx, const ggml_tensor * as, int64_t e,
        int K, int N, int group) {
    if (ctx->i8_dev_failed) return nullptr;
    if (!ctx->i8_dev) {
        // Shared with the dense W8A8 resident path (same device, same worker fds). More
        // worker fds is also more IOVA: the 4GB window is PER FD, and the expert stack of
        // a real MoE is tens of GB, so the fan-out is what makes any of it resident at all.
        ctx->i8_dev = rocket_i8_ctx_create(ctx->n_threads);
        if (!ctx->i8_dev) { ctx->i8_dev_failed = true; return nullptr; }
    }

    // Key on (weight name, expert index). rocket_weight_key returns ONE name for the whole
    // [K,N,n_expert] stack, so a name-only key would serve expert 0's tiles for every
    // expert's matmul -- correct arithmetic on the wrong weights.
    const std::string base = rocket_weight_key(as);
    if (base.empty()) return nullptr;                  // no stable identity -> fp16 route
    const std::string key = base + "/e" + std::to_string((long long)e);

    auto it = ctx->moe_i8_cache.find(key);
    if (it != ctx->moe_i8_cache.end()) {
        if (it->second.K == K && it->second.N == N && it->second.group == group)
            return &it->second;                        // reused across every M (M-independent)
        rocket_i8_weights_free(ctx->i8_dev, it->second.w);   // genuine shape/group change
        ctx->moe_i8_resident_bytes -= it->second.bytes;
        ctx->moe_charged_bytes     -= it->second.charged;
        ctx->moe_n_resident--;
        ctx->moe_i8_cache.erase(it);
    }
    if (ctx->moe_i8_full) { ctx->moe_streamed_keys.insert(key); return nullptr; }

    // Charge the budget for the int8 codes, the group scales, AND the expert's GGUF source
    // bytes (as->nb[2] is exactly one expert's payload).
    //
    // The source is NOT reclaimable here the way a dense resident weight's is. llama.cpp
    // MMAPS the GGUF, so it lives in page cache -- which a large anonymous allocation will
    // happily evict -- and MoE DECODE reads the active experts from it on the CPU every
    // token, so evicting it means faulting the expert back from NVMe per token. Both copies
    // have to coexist, so both are charged.
    //
    // This is a PROXY, and it under-charges when residency is partial: a streamed expert's
    // source still occupies page cache but is charged to nobody. It is deliberately the
    // conservative direction to be wrong in, and it is what keeps a full-residency model
    // honest -- when every expert is resident the charge is exact. Charging the int8 bytes
    // alone is what turns "19.1 GiB of gpt-oss int8 experts fits a 31 GiB board" into a
    // thrash: it does fit, but only by evicting the 11 GiB GGUF it was made from.
    const int    nG  = (int)(K / group);
    const size_t est = (size_t)N * K + (size_t)N * nG * sizeof(float) + (size_t)as->nb[2];
    if (ctx->moe_cache_budget != 0
        && ctx->moe_charged_bytes + est > ctx->moe_cache_budget) {
        ctx->moe_i8_full = true;                       // latch: the rest streams on fp16
        ctx->moe_streamed_keys.insert(key);
        // On the rocket_log channel, NOT GGML_LOG_*: llama-bench SILENCES ggml's logger, and
        // llama-bench is precisely the tool this path is measured with -- a split that only
        // prints under llama-cli is a split nobody sees when it matters.
        ROCKET_LOGI("[moe-int8] resident budget reached at %ld experts (%zuMB on the NPU, "
                    "%zuMB charged incl. the GGUF source) -- the remaining experts stream via "
                    "dequant->fp16 (raise ROCKET_MOE_CACHE_MB, or ROCKET_N_THREADS for more "
                    "per-fd IOVA)\n",
                    ctx->moe_n_resident, ctx->moe_i8_resident_bytes >> 20,
                    ctx->moe_charged_bytes >> 20);
        return nullptr;
    }

    if (ctx->moe_n_resident == 0) {
        char budget[32];
        if (ctx->moe_cache_budget) snprintf(budget, sizeof(budget), "%zuMB", ctx->moe_cache_budget >> 20);
        else                       snprintf(budget, sizeof(budget), "unlimited");
        ROCKET_LOGI("[rocket] MoE native-quant experts ON: %s -> int8, group=%d (nKt=%d), "
                    "resident budget %s (ROCKET_MOE_NATIVE=0 for the fp16 dequant route)\n",
                    ggml_type_name(as->type), group, nG, budget);
    }

    // Ingest ONCE: GGUF quant blocks -> int8 codes + per-(channel, group) scales. Into the
    // context's grow-only scratch: the codes are consumed by the pack below (which scatters
    // them into the resident NPU BOs) and then dropped, so only the scales are kept.
    const char * W_src = (const char *)as->data + (size_t)e * as->nb[2];
    ctx->moe_ingest_codes.resize((size_t)N * K);
    ctx->moe_ingest_scales.resize((size_t)N * nG);
    const double t_ingest = rocket_now_ms();
    if (!rocket_moe_ingest_int8(W_src, as->type, N, K, group,
                                ctx->moe_ingest_codes.data(), ctx->moe_ingest_scales.data())) {
        ctx->moe_streamed_keys.insert(key);            // undecodable type -> fp16 route
        return nullptr;
    }
    const double t_pack = rocket_now_ms();
    ctx->moe_ingest_ms += t_pack - t_ingest;

    rocket_i8_weights * rw = rocket_i8_weights_pack_gw(
            ctx->i8_dev, rocket_hw_current()->max_tile, K, N,
            ctx->moe_ingest_codes.data(), group);
    ctx->moe_pack_ms += rocket_now_ms() - t_pack;
    if (!rw) {                                         // IOVA/alloc exhausted
        ctx->moe_i8_full = true;
        ctx->moe_streamed_keys.insert(key);
        ROCKET_LOGI("[moe-int8] NPU IOVA window full at %ld experts / %zuMB -- the remaining "
                    "experts stream via dequant->fp16 (more workers = more per-fd IOVA: "
                    "ROCKET_N_THREADS)\n",
                    ctx->moe_n_resident, ctx->moe_i8_resident_bytes >> 20);
        return nullptr;
    }
    // The pack planned its scratch at the canonical tile M; register that slot so the
    // bucket accounting sees it (it competes for the same 32-slot table).
    if (K < (1 << 21) && N < (1 << 21))
        ctx->moe_slots.insert(rk_moe_slot_key(rocket_hw_current()->max_tile, K, N));

    // The BUDGET pre-check used an estimate; the running total uses the weight's TRUE
    // resident NPU-BO footprint (queried from the packed weight) plus the same source
    // charge, so the cap tracks what memory is actually committed.
    const size_t bytes   = rocket_i8_weights_bytes(rw);
    const size_t charged = bytes + (size_t)as->nb[2];
    rocket_moe_i8_expert ent;
    ent.w = rw;
    // Copied out of the shared ingest scratch (it is reused by the next expert); the scales
    // are the only host-side state a resident expert keeps -- the codes now live on the NPU.
    ent.b_scale.assign(ctx->moe_ingest_scales.begin(),
                       ctx->moe_ingest_scales.begin() + (size_t)N * nG);
    ent.K = K; ent.N = N; ent.group = group; ent.bytes = bytes; ent.charged = charged;
    // rw holds resident NPU BOs and the entry stores it as a raw pointer with no
    // destructor; free it if the cache insert throws before the map takes ownership.
    rocketraii::scope_guard rw_guard([&] { rocket_i8_weights_free(ctx->i8_dev, rw); });
    auto & slot = (ctx->moe_i8_cache[key] = std::move(ent));
    rw_guard.dismiss();
    ctx->moe_i8_resident_bytes += bytes;
    ctx->moe_charged_bytes     += charged;
    ctx->moe_n_resident++;

    // NOTE: deliberately NO ROCKET_PREPACK_MADVISE reclaim of the GGUF source here, unlike
    // the dense resident paths. Two reasons, either alone sufficient: MoE DECODE reads the
    // active experts from that source on the CPU every token, and under partial residency
    // the experts that did NOT go resident need it for their fp16 route -- and the source
    // is one [K,N,n_expert] mapping shared by all of them.

    if (rocket_debug_on())
        GGML_LOG_DEBUG("[moe-int8] +%-30s K=%5d N=%5d group=%3d  resident=%zuMB (%ld experts)\n",
                       key.c_str(), K, N, group, ctx->moe_i8_resident_bytes >> 20,
                       ctx->moe_n_resident);
    // The ingest is lazy, so it lands inside the FIRST prefill -- and on a real MoE that is
    // thousands of experts and tens of GB of decode + NPU-BO scatter, i.e. minutes. A silent
    // multi-minute stall at the first token reads as a hang, so tick every 256 experts, and
    // carry the elapsed time so the tick is a progress RATE and not just a sign of life.
    else if ((ctx->moe_n_resident % 256) == 0)
        ROCKET_LOGI("[moe-int8] ingesting experts to int8: %ld done, %zuMB resident, %.0fs elapsed\n",
                    ctx->moe_n_resident, ctx->moe_i8_resident_bytes >> 20,
                    (ctx->moe_ingest_ms + ctx->moe_pack_ms) / 1000.0);
    return &slot;
}

// ---------------------------------------------------------------------------
// MUL_MAT_ID (MoE routed-expert FFN)
//
// llama.cpp lowers every mixture-of-experts gate/up/down projection to one
// GGML_OP_MUL_MAT_ID op:
//     as  (src0) -> [K, N, n_expert]                 the per-expert weight matrices
//     b   (src1) -> [K, ne11, n_tokens]              the input activations (ne11 = 1 or n_used)
//     ids (src2) -> [n_expert_used, n_tokens] (i32)  which expert each token-slot picks
//     dst        -> [N, n_expert_used, n_tokens]
// and, for every (slot e in 0..n_used, token t):
//     x         = ids[e, t]                          (the selected expert)
//     dst[:,e,t] = as[:,:,x] @ b[:, e % ne11, t]
//
// The dense MUL_MAT handler can't run this: the weight is selected per row by a
// runtime index, so the op is n_expert independent small GEMMs whose row counts
// depend on the routing. This handler reproduces the CPU reference's row-grouping
// (ggml_compute_forward_mul_mat_id): bucket the (slot,token) rows by their expert
// id, then for each expert with a nonzero bucket gather its rows into a dense
// [M_e,K] activation tile, run [M_e,K] x [N,K]^T on the NPU, and scatter the [M_e,N]
// result back to each row's dst slot.
//
// TWO weight routes, chosen per expert:
//
//   NATIVE-QUANT (a GGUF-quantized expert, the default under ROCKET_MOE=1). The expert's
//   quant blocks were ingested ONCE into int8 codes that live in NPU BOs, so the call
//   quantizes only the activation (per row, per K-group) and runs the resident group-wise
//   int8 matmul, which returns fp32 with every scale already applied. No host dequant, no
//   weight scatter -- which IS the win; see the native-quant section above.
//
//   fp16 (an F16/F32/BF16 expert, an expert that did not fit the resident budget, or
//   ROCKET_MOE_NATIVE=0). The weight is dequantized to fp16 per micro-batch and run
//   through the multicore fp16 GEMM -- the original route, kept as the fallback and as the
//   A/B baseline.
//
// Faithful to the reference within each route's numeric envelope -- proven by
// test-rocket-moe (cosine vs the CPU backend, on both routes) and the per-model
// differential-PPL / greedy-match gates.
//
// Returns 0 on success. A failed NPU job degrades that ONE expert (native -> fp16 -> CPU),
// never aborting the graph; nonzero is returned only on a malformed graph (an id outside
// [0,n_expert), which supports_op cannot see because ids is data).
static int ggml_backend_rocket_mul_mat_id(ggml_backend_rocket_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];   // experts [K, N, n_expert]
    const ggml_tensor * b   = dst->src[1];   // input   [K, ne11, n_tokens]
    const ggml_tensor * ids = dst->src[2];   // ids     [n_used, n_tokens] i32

    const int64_t K        = as->ne[0];
    const int64_t N        = as->ne[1];
    const int64_t n_expert = as->ne[2];
    const int64_t ne11     = b->ne[1];       // b's expert-slot dim (1 broadcasts, or n_used)
    const int64_t n_used   = ids->ne[0];     // n_expert_used
    const int64_t n_tokens = ids->ne[1];
    const ggml_type wt     = as->type;
    const bool  w_is_f16   = (wt == GGML_TYPE_F16);

    // --- bucket (slot,token) rows by expert id (the reference's matrix_rows) ---
    // Counting sort: pass 1 counts rows per expert into off[e+1]; prefix-sum gives
    // each expert's contiguous span; pass 2 fills the flat (slot,tok) row lists.
    std::vector<int32_t> & off  = ctx->moe_expert_off;
    std::vector<int32_t> & slot = ctx->moe_row_slot;
    std::vector<int32_t> & tok  = ctx->moe_row_tok;
    off.assign((size_t)n_expert + 1, 0);
    const int64_t total_rows = n_used * n_tokens;
    slot.resize((size_t)total_rows);
    tok.resize((size_t)total_rows);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_used; e++) {
            const int32_t x = *(const int32_t *)((const char *)ids->data + t * ids->nb[1] + e * ids->nb[0]);
            if (x < 0 || x >= n_expert) return 1;   // malformed graph
            off[x + 1]++;
        }
    }
    for (int64_t e = 0; e < n_expert; e++) off[e + 1] += off[e];
    std::vector<int32_t> cur(off.begin(), off.end() - 1);   // per-expert write cursor
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_used; e++) {
            const int32_t x = *(const int32_t *)((const char *)ids->data + t * ids->nb[1] + e * ids->nb[0]);
            const int32_t p = cur[x]++;
            slot[p] = (int32_t)e;
            tok[p]  = (int32_t)t;
        }
    }

    const size_t nb_b_slot = b->nb[1];    // stride over b's expert-slot dim
    const size_t nb_b_tok  = b->nb[2];    // stride over b's token dim
    const size_t nb_d_slot = dst->nb[1];  // stride over dst's slot dim
    const size_t nb_d_tok  = dst->nb[2];  // stride over dst's token dim
    const int    nth       = ctx->n_threads;

    // Scatter one expert's [M_e,N] fp16 result (undoing the per-row activation
    // scale) into each row's dst[:, slot, tok]. Shared by the NPU and fallback paths.
    auto scatter_fp16 = [&](const ggml_fp16_t * C16, int64_t r0, int64_t M_e) {
        for (int64_t r = 0; r < M_e; r++) {
            const float s = ctx->moe_scales[r];
            const ggml_fp16_t * srow = C16 + (size_t)r * N;
            float * drow = (float *)((char *)dst->data + tok[r0 + r] * nb_d_tok + slot[r0 + r] * nb_d_slot);
            int64_t n = 0;
#ifdef ROCKET_NEON_FP16
            const float32x4_t vs = vdupq_n_f32(s);
            const __fp16 * sh = (const __fp16 *)srow;
            for (; n + 4 <= N; n += 4)
                vst1q_f32(drow + n, vmulq_f32(vcvt_f32_f16(vld1_f16(sh + n)), vs));
#endif
            for (; n < N; n++) drow[n] = ggml_fp16_to_fp32(srow[n]) * s;
        }
    };
    // Scatter a [M_e,N] f32 CPU-fallback result (no scale; computed from raw inputs).
    auto scatter_f32 = [&](const float * Cf, int64_t r0, int64_t M_e) {
        for (int64_t r = 0; r < M_e; r++) {
            float * drow = (float *)((char *)dst->data + tok[r0 + r] * nb_d_tok + slot[r0 + r] * nb_d_slot);
            memcpy(drow, Cf + (size_t)r * N, (size_t)N * sizeof(float));
        }
    };

    // --- native-quant route: is it available for THIS op? ---
    // Only a GGUF-quantized expert has a per-micro-batch dequant to delete, so an F16
    // expert stays on the fp16 route (int8 would cost accuracy and buy nothing). int8's
    // weight k-group is 32, stricter than the N%16 supports_op gate, so self-guard it.
    // rocket_moe_pick_group returns 0 when no legal K-group exists, which also disables it.
    const bool native_type = rocket_moe_native_on() && ggml_is_quantized(wt)
                          && (K % 32 == 0) && (N % 32 == 0);
    const int  group = native_type ? rocket_moe_pick_group((int)K, (int)N) : 0;
    const int  nG    = group > 0 ? (int)(K / group) : 0;

    // ROCKET_MM_PROFILE: split this op into gather / quant / GEMM / scatter, and count the
    // rows the M-bucket padding makes us compute and throw away. See g_moeprof.
    const bool prof = rocket_convprof_on();
    if (prof) { rocket_moeprof_arm(); g_moeprof.calls++; }

    // ROCKET_MOE_COSINE: probe ONE expert per op against the fp64 CPU reference, rotating the
    // chosen expert across ops so the sample sweeps every layer, every projection and the
    // whole expert set instead of camping on expert 0. See rocket_moe_cosine_on.
    const bool cos_probe = rocket_moe_cosine_on() && group > 0;
    static long   cos_op     = 0;
    const int64_t cos_expert = cos_probe ? (cos_op++ % n_expert) : -1;

    // --- per-expert GEMM ---
    for (int64_t e = 0; e < n_expert; e++) {
        const int64_t r0 = off[e], r1 = off[e + 1];
        const int64_t M_e = r1 - r0;
        if (M_e == 0) continue;   // this expert got no tokens this micro-batch

        const char * W_src = (const char *)as->data + e * as->nb[2];   // expert weight [N,K]

        // gather this expert's M_e input rows into a dense [M_e,K] f32 tile (both routes
        // consume it: the native one quantizes it, the fp16 one packs it)
        const double t_gather = prof ? rocket_now_ms() : 0.0;
        ctx->moe_Af32.resize((size_t)M_e * K);
        for (int64_t r = 0; r < M_e; r++) {
            const int64_t i11 = slot[r0 + r] % ne11;   // b's slot (broadcast when ne11==1)
            const int64_t i12 = tok[r0 + r];
            const float * src = (const float *)((const char *)b->data + i12 * nb_b_tok + i11 * nb_b_slot);
            memcpy(ctx->moe_Af32.data() + (size_t)r * K, src, (size_t)K * sizeof(float));
        }
        if (prof) g_moeprof.gather += rocket_now_ms() - t_gather;

        // ---- route 1: NATIVE-QUANT (resident int8 codes, group-wise scales) ----
        if (group > 0) {
            // NB: the one-time ingest happens inside this call on an expert's first touch.
            // It is timed separately (ctx->moe_ingest_ms) and is deliberately NOT part of
            // any per-call bucket below -- folding a minutes-long startup cost into a
            // per-micro-batch line would make every phase unreadable.
            const rocket_moe_i8_expert * rw =
                rocket_moe_expert_resident(ctx, as, e, (int)K, (int)N, group);
            if (rw) {
                // Bucket M (>= M_e, %4) -- both a HW contract and what keeps the driver's
                // fixed-size resident-scratch table from filling. Pad rows quantize to zero
                // and contribute nothing; their output rows are computed and discarded.
                const int Mb = rocket_moe_bucket_m(ctx, (int)M_e, (int)K, (int)N);
                ctx->moe_qA.resize((size_t)Mb * K);
                ctx->moe_a_scale.resize((size_t)Mb * nG);
                ctx->moe_Cgw.resize((size_t)Mb * N);
                if (Mb > M_e) {   // the scratch is reused across experts -> clear the pad
                    memset(ctx->moe_qA.data() + (size_t)M_e * K, 0, (size_t)(Mb - M_e) * K);
                    std::fill(ctx->moe_a_scale.begin() + (size_t)M_e * nG,
                              ctx->moe_a_scale.begin() + (size_t)Mb  * nG, 1.0f);
                }
                rocket_quant_act_int8_grouped(ctx->moe_Af32.data(), ctx->moe_qA.data(),
                                              M_e, K, group, ctx->moe_a_scale.data());

                // The primitive applies a_scale[m,g]*b_scale[n,g] to each K-group's int32
                // partial INSIDE the readback loop the integer partials already force, and
                // hands back fp32 -- so there is nothing left to convert or rescale here.
                // Do not add a pass over the output: that is exactly the cost this design
                // avoids (+0.6% as fused, vs the ~2.8ms a separate pass was projected at).
                const double t_gemm = prof ? rocket_now_ms() : 0.0;
                const int rc = rocket_matmul_int8_prepacked_gw(
                        ctx->i8_dev, Mb, (int)K, (int)N, ctx->moe_qA.data(),
                        ctx->moe_a_scale.data(), rw->b_scale.data(), ctx->moe_Cgw.data(), rw->w);
                if (prof) {
                    g_moeprof.gemm += rocket_now_ms() - t_gemm;
                    g_moeprof.gemms++;
                    g_moeprof.rows_used     += M_e;   // the rows the router actually asked for
                    g_moeprof.rows_computed += Mb;    // the rows the bucketed GEMM ran
                }
                if (rc == 0) {
                    // Faithfulness probe (ROCKET_MOE_COSINE=1): one expert per op, rotating,
                    // against the fp64 CPU reference. See rocket_moe_cosine_on.
                    if (cos_probe && e == cos_expert) {
                        ctx->moe_Cf32.resize((size_t)M_e * N);
                        rocket_cpu_matmul_slice(ctx->moe_Af32.data(), (const void *)W_src, wt,
                                                ctx->moe_Cf32.data(), M_e, N, K);
                        const double c = rocket_cosine_f32(ctx->moe_Cgw.data(),
                                                           ctx->moe_Cf32.data(),
                                                           (size_t)M_e * N);
                        if (!g_moecos_armed) { atexit(rocket_moecos_dump); g_moecos_armed = 1; }
                        g_moecos.sum += c;
                        if (c < g_moecos.min) g_moecos.min = c;
                        g_moecos.n++;
                        // Report as we go, not only at exit. atexit is not enough: the host may
                        // never exit cleanly (llama-cli drops into an interactive loop after
                        // generating, so a harness has to kill it), and a diagnostic that only
                        // prints on a graceful shutdown is a diagnostic you do not get.
                        if ((g_moecos.n % 24) == 0) rocket_moecos_dump();
                    }
                    const double t_sc = prof ? rocket_now_ms() : 0.0;
                    scatter_f32(ctx->moe_Cgw.data(), r0, M_e);   // first M_e rows; pad discarded
                    if (prof) g_moeprof.scatter += rocket_now_ms() - t_sc;
                    continue;
                }
                // A driver decline here (a re-pack request, or an exhausted scratch table)
                // is not a wrong answer -- it fails closed. Degrade THIS expert to the fp16
                // route below, and say so once: silently reverting mid-prefill is exactly
                // how a lost win gets mistaken for a bug.
                static bool warned_gw = false;
                if (!warned_gw || rocket_debug_on()) {
                    warned_gw = true;
                    GGML_LOG_WARN("[moe-int8] resident group-wise matmul declined (rc=%d) for "
                                  "expert=%lld M=%d K=%lld N=%lld group=%d -> fp16 route for it\n",
                                  rc, (long long)e, Mb, (long long)K, (long long)N, group);
                }
            }
            // rw == nullptr: admission declined this expert (budget / IOVA / no stable
            // name). It streams on the fp16 route below -- correct, just at the streaming
            // cost. rocket_moe_expert_resident has already recorded and logged the split.
        }

        // ---- route 2: fp16 (dequant per micro-batch) ----
        // One timer for the whole route, charged on every way out of it (the two CPU
        // fallbacks `continue`, the normal path falls through), because what matters is the
        // total that streaming still costs -- not which of its steps it was spent in.
        const double t_fp16 = prof ? rocket_now_ms() : 0.0;
        rocketraii::scope_guard fp16_timer([&] {
            if (prof) { g_moeprof.fp16 += rocket_now_ms() - t_fp16; g_moeprof.fp16_gemms++; }
        });

        const int64_t Mp = rocket_pad_m((int)M_e);   // driver needs M%4; pad rows = 0
        ctx->moe_A16.resize((size_t)Mp * K);
        if (Mp > M_e)   // clear the pad rows (moe_A16 is reused, so they may be stale)
            memset(ctx->moe_A16.data() + (size_t)M_e * K, 0, (size_t)(Mp - M_e) * K * sizeof(ggml_fp16_t));
        ctx->moe_scales.resize((size_t)M_e);
        rocket_pack_activations(ctx->moe_Af32.data(), ctx->moe_A16.data(), M_e, K, ctx->moe_scales.data());

        // expert weight slice [N,K]: F16 zero-copy, else dequant->fp16 (F32/BF16/quant)
        const ggml_fp16_t * Bp;
        if (w_is_f16) {
            Bp = (const ggml_fp16_t *)W_src;
        } else {
            ctx->moe_B16.resize((size_t)N * K);
            if (rocket_weight_to_fp16(W_src, wt, N, K, ctx->moe_B16.data())) {
                Bp = ctx->moe_B16.data();
            } else {
                // undecodable weight (supports_op gates the types, so this is defensive):
                // CPU-fallback this expert straight from the raw weight.
                ctx->moe_Cf32.resize((size_t)M_e * N);
                rocket_cpu_matmul_slice(ctx->moe_Af32.data(), (const void *)W_src, wt,
                                        ctx->moe_Cf32.data(), M_e, N, K);
                scatter_f32(ctx->moe_Cf32.data(), r0, M_e);
                continue;
            }
        }

        // NPU GEMM: C[Mp,N] = A[Mp,K] * Bp[N,K]^T, N fanned across the worker fds.
        ctx->moe_C16.resize((size_t)Mp * N);
        int rc = rocket_matmul_fp16_mt((int)Mp, (int)K, (int)N,
                     reinterpret_cast<const _Float16 *>(ctx->moe_A16.data()),
                     reinterpret_cast<const _Float16 *>(Bp),
                     reinterpret_cast<_Float16 *>(ctx->moe_C16.data()),
                     nth);
        if (rc != 0) {
            // Degrade this expert to a correct CPU matmul (from the original f32
            // inputs, ignoring the scaled A16) rather than aborting the graph.
            GGML_LOG_ERROR("%s: rocket_matmul_fp16_mt failed (%d) expert=%lld M_e=%lld K=%lld N=%lld"
                           " -> CPU fallback for this expert\n",
                           __func__, rc, (long long)e, (long long)M_e, (long long)K, (long long)N);
            ctx->moe_Cf32.resize((size_t)M_e * N);
            rocket_cpu_matmul_slice(ctx->moe_Af32.data(), (const void *)W_src, wt,
                                    ctx->moe_Cf32.data(), M_e, N, K);
            scatter_f32(ctx->moe_Cf32.data(), r0, M_e);
            continue;
        }
        scatter_fp16(ctx->moe_C16.data(), r0, M_e);
    }
    return 0;
}

static enum ggml_status ggml_backend_rocket_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rocket_context * ctx = (ggml_backend_rocket_context *)backend->context;
    const int  n    = cgraph->n_nodes;
    const bool fuse = rocket_fuse_on();

    // ROCKET_DEBUG_GRAPH=1: one line per graph_compute call (per scheduler split)
    // listing its MUL_MATs + src1 -- to see whether Q/K/V (or other projections)
    // co-occur in a single split (fusable) or are isolated by interleaved CPU ops
    // (reshape/norm/rope) into separate splits (NOT fusable from a backend). Separate
    // env from ROCKET_DEBUG so it isn't drowned by the per-worker affinity logging.
    if (rocket_debug_graph_on()) {
        int nmm = 0;
        for (int i = 0; i < n; i++) if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) nmm++;
        // Assemble the whole census as one string and emit it in a single log call:
        // a host callback may prefix each call (timestamp/level), which would otherwise
        // fragment a line built from many GGML_LOG_* calls.
        char buf[80];
        snprintf(buf, sizeof(buf), "[GRAPH] split: %d nodes, %d matmuls:", n, nmm);
        std::string line = buf;
        for (int i = 0; i < n; i++) {
            const ggml_tensor * nd = cgraph->nodes[i];
            if (nd->op == GGML_OP_MUL_MAT) {
                line += ' ';
                line += nd->src[0]->name[0] ? nd->src[0]->name : "(noname)";
                snprintf(buf, sizeof(buf), "(src1=%p)", (void *)nd->src[1]);
                line += buf;
            }
        }
        GGML_LOG_DEBUG("%s\n", line.c_str());
    }

    // A fused group computes several MUL_MAT nodes at once; mark them so the
    // main loop skips them when it reaches their indices.
    std::vector<bool> fused_done(fuse ? (size_t)n : 0, false);

    for (int i = 0; i < n; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (fuse && fused_done[i]) continue;
        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                // Gather the later MUL_MATs in this split that share src1 and are
                // fusable into one combined-N matmul. The split is rocket-only and
                // src1 is a constant split input, so computing several members at
                // this position (reading the shared src1, writing each member's own
                // dst) is order-independent and hazard-free: nothing in the split
                // rewrites src1, and a member's output is only consumed after its
                // original index. (See the fusion note above.)
                if (fuse && rocket_node_fusable(node)) {
                    ggml_tensor * group[ROCKET_MAX_FUSE];
                    int           gidx[ROCKET_MAX_FUSE];
                    int           ng = 0;
                    group[ng] = node; gidx[ng] = i; ng++;
                    for (int j = i + 1; j < n && ng < ROCKET_MAX_FUSE; j++) {
                        ggml_tensor * nj = cgraph->nodes[j];
                        if (!fused_done[j] && nj->op == GGML_OP_MUL_MAT
                            && nj->src[1] == node->src[1] && rocket_node_fusable(nj)) {
                            group[ng] = nj; gidx[ng] = j; ng++;
                        }
                    }
                    if (ng >= 2) {
                        if (rocket_debug_on()) {
                            // One string, one log call (see the [GRAPH] census note above).
                            char buf[80];
                            snprintf(buf, sizeof(buf), "[FUSE] %d matmuls share src1=%p:",
                                     ng, (void *)node->src[1]);
                            std::string line = buf;
                            for (int g = 0; g < ng; g++) {
                                line += ' ';
                                line += group[g]->src[0]->name[0] ? group[g]->src[0]->name : "(noname)";
                                snprintf(buf, sizeof(buf), "(N=%lld)", (long long)group[g]->src[0]->ne[1]);
                                line += buf;
                            }
                            GGML_LOG_DEBUG("%s\n", line.c_str());
                        }
                        if (ggml_backend_rocket_mul_mat_group(ctx, group, ng) == 0) {
                            for (int g = 0; g < ng; g++) fused_done[gidx[g]] = true;
                            break;   // group handled; other members skipped via fused_done
                        }
                        // fusion failed -> run `node` per-node below; the other
                        // members aren't marked done, so they run individually too.
                    }
                }
                ggml_backend_rocket_mul_mat(ctx, node);
                break;
            }
            case GGML_OP_MUL_MAT_ID:
                if (ggml_backend_rocket_mul_mat_id(ctx, node) != 0) {
                    GGML_LOG_ERROR("%s: MUL_MAT_ID handler failed\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                if (ggml_backend_rocket_flash_attn(ctx, node) != 0) {
                    GGML_LOG_ERROR("%s: FLASH_ATTN_EXT handler failed\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;   // no-ops on a view of already-computed data
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// The vtables below (ggml_backend_i / ggml_backend_device_i / ggml_backend_reg_i) are
// POSITIONAL designated initializers against the host app's ggml ABI. ggml's loader
// rejects an api_version MISMATCH cleanly, but a struct field-ORDER drift WITHOUT a
// version bump would silently bind the wrong slots (crash / wrong call, no diagnostic).
// Pin the version we were compiled against so any such drift fails at COMPILE time;
// when this fires, re-audit every vtable literal against the new ggml-backend-impl.h
// before bumping the constant.
static_assert(GGML_BACKEND_API_VERSION == 2,
              "ggml backend ABI changed: re-audit the rocket vtable literals, then bump this guard");

static const ggml_backend_i rocket_backend_i = {
    /* .get_name           = */ ggml_backend_rocket_get_name,
    /* .free               = */ ggml_backend_rocket_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .set_tensor_2d_async= */ NULL,
    /* .get_tensor_2d_async= */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ ggml_backend_rocket_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
    /* .graph_optimize     = */ NULL,
};

static ggml_guid_t ggml_backend_rocket_guid(void) {
    // any fixed 16-byte GUID ("ROCK" + random)
    static ggml_guid guid = { 0x52,0x4f,0x43,0x4b, 0x88,0x12,0x4f,0x3a,
                              0x9b,0x55,0x21,0xee, 0x73,0x0c,0x6d,0x42 };
    return &guid;
}

// ---------------------------------------------------------------------------
// device interface
// ---------------------------------------------------------------------------
static const char * ggml_backend_rocket_device_get_name(ggml_backend_dev_t dev) {
    (void)dev; return "ROCKET";
}
static const char * ggml_backend_rocket_device_get_description(ggml_backend_dev_t dev) {
    (void)dev; return "RK3588 NPU (mainline rocket driver)";
}
static void ggml_backend_rocket_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev; *free = 0; *total = 0;   // NPU uses system DRAM; not separately tracked
}
static enum ggml_backend_dev_type ggml_backend_rocket_device_get_type(ggml_backend_dev_t dev) {
    (void)dev; return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
static void ggml_backend_rocket_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rocket_device_get_name(dev);
    props->description  = ggml_backend_rocket_device_get_description(dev);
    props->type         = ggml_backend_rocket_device_get_type(dev);
    ggml_backend_rocket_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}
static ggml_backend_t ggml_backend_rocket_device_init(ggml_backend_dev_t dev, const char * params) {
    (void)dev; (void)params; return ggml_backend_rocket_init();
}
static ggml_backend_buffer_type_t ggml_backend_rocket_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev; return ggml_backend_cpu_buffer_type();
}
static ggml_backend_buffer_t ggml_backend_rocket_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)dev; (void)max_tensor_size; return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_rocket_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    (void)dev;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * a = op->src[0];   // weights
            const ggml_tensor * b = op->src[1];   // input
            const int64_t K = a->ne[0];
            const int64_t N = a->ne[1];
            const int64_t M = b->ne[1];
            // Only offload big STATIC-WEIGHT GEMMs (q/k/v/o/gate/up/down/lm_head:
            // src0 is a model parameter -> leaf, op == GGML_OP_NONE). This skips
            // the attention QK/AV matmuls (src0 is a computed/view tensor): they
            // are small-K, dynamic, and would churn fds per call for no win.
            // M is padded to a multiple of 4 internally, so M%4==0 is not required.
            // We DO require M >= rocket_min_m() so decode (M=1 GEMV) and tiny
            // ubatches stay on the CPU.
            // Quantized weights (Q8_0 / Q4_K / Q6_K / ...) are dequantized to fp16 on
            // the streaming/mt path (rocket_weight_to_fp16). They need K aligned
            // to the type's block size so each [K] row is a whole number of quant blocks
            // (K%32 below already covers Q8_0's block 32; K-quants use 256).
            const bool a_quant = ggml_is_quantized(a->type);
            // On the RK3576 the W8A8 route is the ONLY matmul route: the RK3588 fp16
            // generators refuse on that part by construction, so an op accepted here and
            // then declined by the W8A8 entry does not fall back — it fails at compute
            // time. So the gate is exactly what that entry can run, and nothing wider.
            // The library refuses K >= 6176 (no single-task plan) and the int32 K-split
            // route that would run it wedges the part until it is rebooted. The handler can
            // instead split K on the HOST, one whole W8A8 GEMM per chunk with the f32
            // partials summed, so the claim is "a chunking exists" rather than a depth
            // bound. That is OFF by default and ROCKET_RK3576_KSPLIT=1 turns it on: it is
            // fast and exact, and it costs the one model measured 1.60x perplexity, because
            // what it reaches is the FFN down-projection and the route's per-tensor
            // activation scale does not carry that input. See rk76_ksplit.
            if (rocket_rk3576_selected()) {
                return rocket_int8_mode_on()
                    && a->op == GGML_OP_NONE
                    && (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32)
                    && b->type == GGML_TYPE_F32
                    && op->type == GGML_TYPE_F32
                    && ggml_is_contiguous(a) && ggml_is_contiguous(b)
                    && ggml_is_contiguous(op)
                    && a->ne[0] == b->ne[0]
                    && a->ne[2] == 1 && a->ne[3] == 1
                    && b->ne[2] == 1 && b->ne[3] == 1
                    && (K % 32 == 0) && (N % 32 == 0)   // int8 weight k-group AND N-group
                    && K >= 64 && N >= 32
                    && rk76_ksplit((int)K, NULL)        // a chunking the entry will take,
                                                        // each chunk rotatable and under
                                                        // the library's own refusal
                    && M >= rocket_min_m();
            }
            return a->op == GGML_OP_NONE
                && (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32
                    || a->type == GGML_TYPE_BF16   // decoded to fp16; ROCKET_BF16=1 selects the fp32-out datapath
                    || a_quant)                    // dequant->fp16
                && b->type == GGML_TYPE_F32
                && op->type == GGML_TYPE_F32
                && ggml_is_contiguous(a) && ggml_is_contiguous(b)
                // Both matmul paths write M*N contiguous floats into dst->data; a
                // viewed/strided dst would smear a neighbouring tensor. Require a
                // contiguous output until copyback honours dst->nb.
                && ggml_is_contiguous(op)
                && a->ne[0] == b->ne[0]
                && a->ne[2] == 1 && a->ne[3] == 1   // plain 2D weight
                && (!a_quant || (K % ggml_blck_size(a->type) == 0))
                && (K % 32 == 0) && (N % 16 == 0)
                && K >= 64 && N >= 64
                // Quantized weights take the dequant->fp16 path (a fixed per-microbatch
                // dequant cost); it needs more rows than the F16 path to amortize, so gate
                // it on a higher floor (rocket_min_m_quant). Native int4/int8 modes
                // re-quantize F16 weights, not ggml_is_quantized ones, so a_quant here is
                // exactly the dequant path.
                && M >= (a_quant ? rocket_min_m_quant() : rocket_min_m());
        }
        case GGML_OP_MUL_MAT_ID: {
            // MoE routed-expert FFN. The handler (ggml_backend_rocket_mul_mat_id)
            // buckets the (slot,token) rows by expert id, then runs one dense
            // [M_e,K] x [N,K]^T NPU GEMM per active expert. Gate exactly what it
            // can compute so it never fails at compute time.
            //   as  (src0) [K, N, n_expert]                 static expert weights
            //   b   (src1) [K, ne11, n_tokens] F32          input activations
            //   ids (src2) [n_expert_used, n_tokens] I32    routing
            //   dst        [N, n_expert_used, n_tokens] F32
            if (!rocket_moe_on()) return false;
            const ggml_tensor * a  = op->src[0];
            const ggml_tensor * b  = op->src[1];
            const ggml_tensor * id = op->src[2];
            if (!a || !b || !id) return false;
            const int64_t K = a->ne[0];
            const int64_t N = a->ne[1];
            const int64_t n_tokens = id->ne[1];
            // Weight types mirror the dense MUL_MAT gate: F16 zero-copy, F32/BF16/
            // quantized (MXFP4/Q4_K/...) dequant->fp16 via rocket_weight_to_fp16.
            const bool a_quant = ggml_is_quantized(a->type);
            return a->op == GGML_OP_NONE
                && (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32
                    || a->type == GGML_TYPE_BF16 || a_quant)
                && b->type == GGML_TYPE_F32
                && id->type == GGML_TYPE_I32
                && op->type == GGML_TYPE_F32
                // The gather/scatter read b rows / write dst rows as contiguous
                // [K]/[N] spans indexed by the per-token/per-slot strides; require
                // both contiguous (the fresh dst always is; b is in MoE graphs).
                && ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(op)
                && a->ne[0] == b->ne[0]           // can_mul_mat (K matches)
                && a->ne[3] == 1                  // as is 3D (one matrix per expert)
                && (!a_quant || (K % ggml_blck_size(a->type) == 0))
                && (K % 32 == 0) && (N % 16 == 0)
                && K >= 64 && N >= 64
                // Prefill-gate on the micro-batch token count (M_e ~ n_tokens *
                // n_used / n_expert): short prefills and decode stay on the CPU.
                && n_tokens >= rocket_moe_min_tokens();
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // Offload the fused attention op (LLM prefill). Gate exactly what the handler
            // can compute so it never fails at compute time: F32 Q / F16 K,V / F16
            // contiguous mask / F32 contiguous dst, no batch dim, ALiBi off (max_bias==0),
            // GQA divides, head_dim%32 (the QK contraction alignment), and a prefill-sized
            // n_tokens (decode's single token stays on the CPU). Caller passes scale/softcap
            // in op_params; both are handled.
            //
            // The handler (ggml_backend_rocket_flash_attn) gathers strided operands into
            // dense tiles BEFORE touching the NPU, indexing K/V/mask by extents INFERRED
            // from k (n_kv = k->ne[1], n_kv_heads = k->ne[2]) and q (n_tokens, n_head).
            // ggml's own op constructor does not assert that V's KV extents match K's, nor
            // that the mask covers [n_kv, n_tokens] (ggml.c: ggml_flash_attn_ext only checks
            // head_dim/batch/GQA divisibility, with a standing TODO on the V check). So the
            // gather would read out of bounds on a malformed graph unless we validate those
            // extents here -- this gate is the handler's only contract check. Real llama.cpp
            // graphs always satisfy them; the explicit checks make a bad graph fall to the
            // CPU instead of reading past K/V/mask.
            const ggml_tensor * q = op->src[0];
            const ggml_tensor * k = op->src[1];
            const ggml_tensor * v = op->src[2];
            const ggml_tensor * m = op->src[3];
            if (!rocket_flash_attn_on() || !q || !k || !v || !m) return false;
            // ATTENTION SINKS (src[4]) — DECLINE. A sink is a learned per-head logit that joins
            // the softmax denominator: softmax over [scale*QK^T + mask, sink] rather than over
            // the scores alone. The handler computes softmax(scale*QK^T + mask) and carries no
            // sink term, so accepting such an op does not merely lose accuracy — it computes a
            // DIFFERENT attention, silently, and the graph has no way to notice.
            //
            // This is not hypothetical: gpt-oss ships sinks on every layer
            // (llama.cpp openai-moe: attn_sinks -> ggml_flash_attn_ext_add_sinks -> src[4]), and
            // the gate below accepts everything else about its FA ops. The bug was invisible
            // because it only fires past the n_kv floor (1024), so short-prompt tests never
            // reached it, and because a wrong-but-plausible attention still produces fluent text
            // — the differential-PPL check read it as +1.0%, i.e. "within noise".
            //
            // Declining is also FASTER on this model, which is why the loss looked like a
            // performance regression: the offload was costing time to compute the wrong answer.
            // Implementing the sink is easy in principle (the softmax is host-side, so it is one
            // extra term in the denominator) — but do that only if the offload is a WIN on a
            // sink-carrying model, which on gpt-oss it is not.
            if (op->src[4]) return false;
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *)op->op_params + 1, sizeof(float));
            // head_dim (= DK) is the QK contraction dim; dv (= DV) the value/output dim.
            // They are equal for GQA/MHA and DIFFER for MLA (DeepSeek: DK=192, DV=128) --
            // the handler threads both, so the gate only requires K to match Q on DK and V
            // to carry a DV that clears the AV output alignment.
            const int64_t head_dim = q->ne[0], n_tokens = q->ne[1], n_head = q->ne[2];
            const int64_t dv = v->ne[0];
            const int64_t n_kv = k->ne[1], n_kv_heads = k->ne[2];
            return max_bias == 0.0f
                && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16
                && v->type == GGML_TYPE_F16 && m->type == GGML_TYPE_F16
                && op->type == GGML_TYPE_F32
                && ggml_is_contiguous(m) && ggml_is_contiguous(op)
                && q->ne[3] == 1 && k->ne[3] == 1   // no batch dim (LLM prefill)
                && k->ne[0] == head_dim             // K matches Q on the QK contraction (DK)
                && op->ne[0] == dv                  // dst head dim is the value dim (DV)
                // V's KV extents must match K's: the handler reads V as
                // [n_kv_heads][dv][n_kv] (v->nb[1] over n_kv, v->nb[2] over n_kv_heads).
                && v->ne[1] == n_kv && v->ne[2] == n_kv_heads
                && head_dim % 32 == 0                // QK contraction alignment (DK)
                && dv % 16 == 0                      // AV output alignment (DV)
                && n_kv_heads > 0                    // guard the GQA modulo (div-by-zero)
                && n_head % n_kv_heads == 0          // GQA group divides cleanly
                // The mask gather reads m[t in 0..n_tokens, j in 0..n_kv]; require it to
                // span at least that (ggml pads ne[1] up to GGML_KQ_MASK_PAD, so >=).
                && m->ne[0] >= n_kv && m->ne[1] >= n_tokens
                && n_tokens >= rocket_flash_attn_min_t()    // decode (1 token) stays on CPU
                && n_kv >= rocket_flash_attn_min_kv();       // short context stays on CPU (parity below ~1K)
        }
        default:
            return false;
    }
}
// Accept any HOST-accessible buffer (mirrors the BLAS backend), not just the
// exact cpu buffer type. The model's mmapped weights live in a CPU_Mapped buft;
// rejecting it forced ggml-backend-sched to COPY every weight into a pooled,
// address-reused buffer -- which is what let the prepacked cache alias distinct
// weights onto one address (see rocket_weight_key). Accepting host bufts lets
// sched use the weights in place: src0->data becomes the real, unique, stable
// weight address (the alias can't even form), and a per-op weight copy is
// dropped. We still only run MUL_MAT (supports_op) and still hand out cpu buffers
// (get_buffer_type), so this changes whether inputs are copied, not op placement.
//
// CONSEQUENCE FOR QUANTIZED WEIGHTS: the scheduler only assigns a weight's
// matmul to this backend when the weight's buffer is host (it picks the highest-prio
// backend whose supports_buft + supports_op both accept it, then falls back to the CPU's
// op_offload, which also gates on is_host). With ggml-cpu's repack extra-buffer-type
// enabled (the GGML_CPU_REPACK build option, on by default), quantized weights are loaded
// into a non-host "CPU_REPACK" buffer for the CPU's fast repacked kernels, so this returns
// false for them and their prefill stays on the CPU. To prefill a quantized GGUF on the NPU
// (dequant->fp16, see rocket_weight_to_fp16), build the host's ggml with GGML_CPU_REPACK=OFF
// so the weights stay in a host (CPU_Mapped) buffer. F16 weights are never repacked.
static bool ggml_backend_rocket_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev; return ggml_backend_buft_is_host(buft);
}

// The scheduler only pulls a CPU/host-resident op onto an accel device when the
// device's offload_op ALSO says yes (ggml-backend.cpp: supports_op && offload_op).
// Our weights live in CPU buffers, so without this every matmul stays on the CPU
// even though supports_op accepts it. Offload exactly the weight ops we can run.
//
// Why this is MUL_MAT / MUL_MAT_ID (and FLASH_ATTN_EXT is not here): the scheduler
// consults offload_op ONLY inside its weight-source branch -- for an op one of whose
// srcs lives in a GGML_BACKEND_BUFFER_USAGE_WEIGHTS host buffer (ggml-backend.cpp
// ggml_backend_sched_backend_id_from_cur). A MUL_MAT's src0 (the projection weight)
// and a MUL_MAT_ID's src0 (the per-expert weight stack) are both such weights, so this
// hook decides their placement. FLASH_ATTN_EXT has NO weight src (Q/K/V/mask are computed
// tensors and KV-cache views, not WEIGHTS buffers), so offload_op is never evaluated for
// it -- adding it here would be dead code. Attention instead reaches the NPU via
// supports_op + the scheduler's "expand" passes: the surrounding Q/K/V/O projection
// matmuls are placed on the NPU by this hook, and expand-up/down propagate that NPU
// assignment to the adjacent FLASH_ATTN_EXT through set_if_supported (which checks
// supports_op only). So FA placement rides on its neighbourhood being NPU-resident --
// which holds for LLM prefill graphs, where attention sits between NPU projections. The
// trade-off: FA placement is neighbourhood-driven, not policy-pinned, so a graph that
// isolated the attention op between CPU-only ops would silently leave it on the CPU.
// test-rocket-placement pins the supports_op contract that this expansion relies on.
static bool ggml_backend_rocket_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // MUL_MAT and MUL_MAT_ID both have a static-weight src0 (the projection weight /
    // the per-expert weight stack) in a WEIGHTS host buffer, so the scheduler's
    // weight-source branch consults offload_op for them. FLASH_ATTN_EXT has no
    // weight src (see the note above) -- it rides its neighbourhood, not this hook.
    return (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID)
        && ggml_backend_rocket_device_supports_op(dev, op);
}

static const ggml_backend_device_i rocket_device_i = {
    /* .get_name             = */ ggml_backend_rocket_device_get_name,
    /* .get_description      = */ ggml_backend_rocket_device_get_description,
    /* .get_memory           = */ ggml_backend_rocket_device_get_memory,
    /* .get_type             = */ ggml_backend_rocket_device_get_type,
    /* .get_props            = */ ggml_backend_rocket_device_get_props,
    /* .init_backend         = */ ggml_backend_rocket_device_init,
    /* .get_buffer_type      = */ ggml_backend_rocket_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rocket_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rocket_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rocket_device_supports_buft,
    /* .offload_op           = */ ggml_backend_rocket_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------
static const char * ggml_backend_rocket_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg; return "ROCKET";
}
static size_t ggml_backend_rocket_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg; return 1;
}
static ggml_backend_dev_t ggml_backend_rocket_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ rocket_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static const ggml_backend_reg_i rocket_reg_i = {
    /* .get_name         = */ ggml_backend_rocket_reg_get_name,
    /* .get_device_count = */ ggml_backend_rocket_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rocket_reg_get_device,
    /* .get_proc_address = */ NULL,
};

// ===========================================================================
// Log bridge, registration, and the .so entry points
// ===========================================================================
// ---------------------------------------------------------------------------
// Unified diagnostic channel. Everything this backend stack emits — the backend's
// own messages (the GGML_LOG_* calls plus the profile/trace lines routed below)
// AND the librocketnpu driver's messages — flows through ggml's logger, so the
// host (llama.cpp / whisper.cpp) can redirect or silence all of it via
// ggml_log_set instead of raw stderr writes leaking past the host's log settings.
//
// The driver keeps its own channel (rocket_log), with its ROCKET_LOG_LEVEL /
// ROCKET_DEBUG threshold applied FIRST; we install a forwarding sink that maps a
// rocket_log_level onto the matching ggml_log_level and re-emits the already-
// formatted, newline-terminated line. A driver DEBUG trace therefore needs both
// its driver-side gate (ROCKET_DEBUG) and a host logger that prints DEBUG.
static void rocket_log_forward_to_ggml(rocket_log_level level, const char * text, void * user_data) {
    (void) user_data;
    enum ggml_log_level g;
    switch (level) {
        case ROCKET_LOG_ERROR: g = GGML_LOG_LEVEL_ERROR; break;
        case ROCKET_LOG_WARN:  g = GGML_LOG_LEVEL_WARN;  break;
        case ROCKET_LOG_DEBUG: g = GGML_LOG_LEVEL_DEBUG; break;
        case ROCKET_LOG_INFO:
        default:               g = GGML_LOG_LEVEL_INFO;  break;
    }
    ggml_log_internal(g, "%s", text);   // text already formatted by rocket_log; %s avoids re-parsing it
}
static void rocket_install_log_bridge(void) {
    // Installed once, from the registry accessor below — before any offload, on the
    // single-threaded backend-discovery path. rocket_log_set_callback is set-once-
    // before-first-use, so call_once matches its contract.
    static std::once_flag once;
    std::call_once(once, [] { rocket_log_set_callback(rocket_log_forward_to_ggml, nullptr); });
}

ggml_backend_reg_t ggml_backend_rocket_reg(void) {
    rocket_install_log_bridge();
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rocket_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

// Parse an MB-budget env knob into bytes. Rejects garbage and negative input (a
// negative atoll would become an enormous size_t after <<20), keeping the caller's
// default; caps the value so the MB->bytes shift can't overflow size_t. 0 stays the
// documented "unlimited".
static bool rocket_parse_mb_budget(const char * e, size_t * out_bytes) {
    char * end = nullptr;
    long long mb = strtoll(e, &end, 10);
    if (end == e || mb < 0) return false;          // not a number / negative -> keep default
    const long long MB_MAX = 1LL << 40;            // far past any real RAM; keeps <<20 in range
    if (mb > MB_MAX) mb = MB_MAX;
    *out_bytes = (size_t)mb << 20;
    return true;
}

// Read one "Field: N kB" line from /proc/meminfo into bytes. 0 if unreadable / absent,
// which the caller treats as "signal unavailable" (falls back to the fixed default budget).
static size_t rocket_meminfo_bytes(const char * field) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    const size_t flen = strlen(field);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, field, flen) == 0 && line[flen] == ':') {
            kb = strtoull(line + flen + 1, nullptr, 10);   // " <N> kB"
            break;
        }
    }
    fclose(f);
    return kb << 10;
}

ggml_backend_t ggml_backend_rocket_init(void) {
    // Own the context until the backend takes it, so a std::bad_alloc from the
    // backend allocation below cannot leak it (released on the success path).
    auto ctx = std::make_unique<ggml_backend_rocket_context>();
    // ROCKET_N_THREADS overrides the NPU worker/fd count (1..8) without a rebuild,
    // for sweeping multicore vs per-job fence overhead on a given workload.
    if (const char * e = getenv("ROCKET_N_THREADS")) {
        int v = atoi(e);
        if (v >= 1 && v <= 8) ctx->n_threads = v;
    }
    // ROCKET_CACHE_MB: resident packed-weight budget in MB (0 = unlimited).
    if (const char * e = getenv("ROCKET_CACHE_MB")) {
        size_t b; if (rocket_parse_mb_budget(e, &b)) ctx->cache_budget = b;
    }
    // ROCKET_QUANT_RESIDENT / ROCKET_F16_RESIDENT budget modes (on/off gates are
    // rocket_quant_resident_on / rocket_f16_resident_on). Both size the SAME resident-weight
    // budget (cache_budget) -- QUANT_RESIDENT holds a quant GGUF's dequant->fp16 form resident,
    // F16_RESIDENT holds an F16 GGUF's scattered tiles resident:
    //   auto    -> size the budget from free RAM: MemAvailable minus a reserve (KV cache /
    //              activations / general headroom, and -- for F16 -- the still-mapped GGUF the
    //              resident tiles duplicate). Promotes as many weights as SAFELY fit, so the
    //              "RAM to spare" case needs no hand-computed ROCKET_CACHE_MB. The RK1 has no
    //              swap, so the reserve is deliberately generous (default max(6GiB, 30% of RAM);
    //              override ROCKET_QUANT_RESIDENT_RESERVE_MB) AND a runtime MemAvailable floor
    //              (resident_floor_bytes, checked in build_resident) latches before an OOM.
    //   <N>>=2  -> an explicit N-MB budget (same effect as ROCKET_CACHE_MB=N, co-located here).
    //   1       -> blanket, bounded by the existing cache_budget default.
    // An explicit ROCKET_CACHE_MB always wins -- it is the lower-level knob. Promotion order
    // is weight-encounter (~layer) order, which is optimal here: LLM prefill uses every weight
    // once per micro-batch, so per-layer "hotness" is uniform and only total resident bytes
    // (hence the budget) sets the fraction of the packB / dequant tax removed.
    const char * rez = getenv("ROCKET_F16_RESIDENT");
    const char * rez_name = "ROCKET_F16_RESIDENT";
    if (!rez) { rez = getenv("ROCKET_QUANT_RESIDENT"); rez_name = "ROCKET_QUANT_RESIDENT"; }
    if (rez) {
        if (!getenv("ROCKET_CACHE_MB")) {                 // explicit budget overrides
            const size_t total = rocket_meminfo_bytes("MemTotal");
            size_t reserve = (size_t)6144 << 20;              // 6 GiB floor
            if (total / 10 * 3 > reserve) reserve = total / 10 * 3;   // or 30% of RAM
            if (const char * r = getenv("ROCKET_QUANT_RESIDENT_RESERVE_MB")) {
                size_t rb; if (rocket_parse_mb_budget(r, &rb)) reserve = rb;
            }
            if (strcmp(rez, "auto") == 0) {
                const size_t avail = rocket_meminfo_bytes("MemAvailable");
                const size_t budget = (avail > reserve) ? (avail - reserve)
                                                        : ((size_t)256 << 20);  // minimal floor
                if (avail) ctx->cache_budget = budget;            // 0 avail -> keep default
                ctx->resident_floor_bytes = reserve;              // runtime OOM guard
                // rocket_log channel (not GGML_LOG_*) so ROCKET_LOG_STDERR shows this
                // budget decision even under a host that silences ggml (llama-bench).
                ROCKET_LOGI("[rocket] %s=auto -> resident budget %zuMB "
                            "(MemAvailable %zuMB - reserve %zuMB, no swap)\n",
                            rez_name, ctx->cache_budget >> 20, avail >> 20, reserve >> 20);
            } else {
                const long long n = atoll(rez);
                if (n >= 2) {
                    ctx->cache_budget = (size_t)n << 20;
                    ctx->resident_floor_bytes = reserve;          // guard explicit budgets too
                    ROCKET_LOGI("[rocket] %s=%lld -> resident budget %lldMB\n", rez_name, n, n);
                }
            }
        }
    }
    // ROCKET_INT8_CACHE_MB: resident rotated-int8 weight budget in MB (0 = unlimited).
    if (const char * e = getenv("ROCKET_INT8_CACHE_MB")) {
        size_t b; if (rocket_parse_mb_budget(e, &b)) ctx->int8_cache_budget = b;
    }
    // ROCKET_INT4_CACHE_MB: int4 weight budget in MB (0 = unlimited). Caps BOTH the
    // one-shot host int4 cache and the resident int4 NPU-BO total (the nibble-packed
    // resident weights are ~1/4 fp16, so a full 12B fits a few GB).
    if (const char * e = getenv("ROCKET_INT4_CACHE_MB")) {
        size_t b; if (rocket_parse_mb_budget(e, &b)) ctx->int4_cache_budget = b;
    }
    // ROCKET_MOE_CACHE_MB: the resident native-quant EXPERT budget in MB (0 = unlimited).
    //
    // This is a much bigger appetite than the other caches and needs its own budget rather
    // than the dense path's fixed 4GB default: a real MoE's expert stack is the bulk of the
    // model (gpt-oss-20b: 19.1 GiB of int8 expert codes), and the GGUF it was ingested from
    // must stay mapped -- MoE DECODE reads the active experts from it on the CPU every
    // token, so it cannot be reclaimed. Full residency therefore does NOT fit a 32GB board
    // for gpt-oss, and partial residency is the design, not a failure mode.
    //
    // So the default is AUTO: size the budget from MemAvailable minus a reserve, exactly as
    // ROCKET_QUANT_RESIDENT=auto does, and let admission fill it and then degrade. A blanket
    // "unlimited" on a memory-tight board is a trap (the board has no swap); a fixed default
    // would be wrong on every board but one.
    {
        const char * e = getenv("ROCKET_MOE_CACHE_MB");
        size_t b;
        if (e && rocket_parse_mb_budget(e, &b)) {
            ctx->moe_cache_budget = b;                      // explicit (0 = unlimited)
            ROCKET_LOGI("[rocket] ROCKET_MOE_CACHE_MB=%s -> resident expert budget %s\n",
                        e, b ? "set" : "unlimited");
        } else {
            const size_t avail = rocket_meminfo_bytes("MemAvailable");
            const size_t total = rocket_meminfo_bytes("MemTotal");
            size_t reserve = (size_t)6144 << 20;             // 6 GiB floor
            if (total / 10 * 3 > reserve) reserve = total / 10 * 3;   // or 30% of RAM
            if (const char * r = getenv("ROCKET_QUANT_RESIDENT_RESERVE_MB")) {
                size_t rb; if (rocket_parse_mb_budget(r, &rb)) reserve = rb;
            }
            ctx->moe_cache_budget = (avail > reserve) ? (avail - reserve)
                                                      : ((size_t)256 << 20);   // minimal floor
            if (!avail) ctx->moe_cache_budget = (size_t)4096 << 20;   // no signal -> conservative
        }
    }
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rocket_guid(),
        /* .iface   = */ rocket_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rocket_reg(), 0),
        /* .context = */ ctx.get(),
    };
    ctx.release();   // ownership transferred to backend->context; freed in ggml_backend_rocket_free
    return backend;
}

bool ggml_backend_is_rocket(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rocket_guid());
}

void ggml_backend_rocket_moe_stats(ggml_backend_t backend, long * n_resident, long * n_streamed) {
    GGML_ASSERT(ggml_backend_is_rocket(backend));
    const ggml_backend_rocket_context * ctx = (const ggml_backend_rocket_context *)backend->context;
    if (n_resident) *n_resident = ctx->moe_n_resident;
    if (n_streamed) *n_streamed = (long)ctx->moe_streamed_keys.size();
}

void ggml_backend_rocket_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_rocket(backend));
    ggml_backend_rocket_context * ctx = (ggml_backend_rocket_context *)backend->context;
    // Clamp to the same 1..8 worker range the ROCKET_N_THREADS env path enforces, so
    // the public API can't install a 0/negative/huge worker count that the lazy
    // rocket_*_create paths would pass to the driver. Out-of-range -> clamp + warn
    // rather than silently store a bad value.
    int v = n_threads;
    if (v < 1) v = 1;
    if (v > 8) v = 8;
    if (v != n_threads)
        GGML_LOG_WARN("%s: n_threads=%d out of range, clamped to %d (valid 1..8)\n",
                      __func__, n_threads, v);
    // Effective only BEFORE the first offload: dev/stream/i8_dev/i4_dev/bf16/fa contexts
    // capture the worker count at lazy-create time, so a call after the first offload
    // updates the field but not the live worker pools. This matches the header contract
    // ("call before graph compute"); warn if the resources already exist so the no-op is
    // not silent.
    const bool workers_live = ctx->dev || ctx->stream || ctx->i8_dev || ctx->i4_dev
        || ctx->int8_fd >= 0 || ctx->int4_fd >= 0 || ctx->bf16_fd >= 0
        || ctx->fa_ctx || ctx->fa_fd >= 0;
    if (workers_live)
        GGML_LOG_WARN("%s: called after the first offload; worker pools already sized to "
                      "%d -> the new value (%d) does not reconfigure them\n",
                      __func__, ctx->n_threads, v);
    ctx->n_threads = v;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rocket_reg)
