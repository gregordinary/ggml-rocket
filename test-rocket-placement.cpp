// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-placement.cpp — placement-gate regression guard for the rocket backend's
 * device supports_op / offload_op contract.
 *
 * The device-level supports_op is a pure function of the op's shape/type (no data, no
 * device handle), so this gate is metadata-only and runs ANYWHERE (CI included) — it
 * needs the backend registry, not an open NPU. It pins down exactly which MUL_MATs the
 * scheduler may pull onto the NPU and, just as importantly, which it must leave on the
 * CPU: the shape contract (K%32, N%16, K>=64, N>=64, M>=min_m), static-leaf-weight only,
 * and "MUL_MAT only". Complements test-rocket-matmul (the offloaded math) and
 * test-rocket-bf16 (the bf16 decode route).
 *
 * THE ACCEPT-SET IS PER-PART. On the RK3588 a weight reaches the NPU as fp16, so BF16 and
 * the ggml-quantized types are accepted (decoded / dequantized on the way) and attention
 * offloads. The RK3576 has no fp16 matmul encoder at all: its only matmul route is the
 * W8A8 one, which ROCKET_INT8 selects and which takes an F16/F32 leaf on a K%32/N%32 shape
 * -- so bf16, the quantized types and FLASH_ATTN_EXT are declined there, and declining is
 * correct (claiming them would hand the scheduler work only the host reference can do).
 * Each expectation below therefore names the part it holds for.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rocket.h"
#include "test-common.h"   // rk_is_rk3576()

#include <cstdio>
#include <cstdlib>
#include <unistd.h>      // fork/_exit — one child per cached-getenv MoE mode
#include <sys/wait.h>

// supports_op for a 2D mul_mat: W[K,N] (type wt, leaf) x X[K,M] (f32) -> f32 [N,M].
static bool mm_supported(ggml_backend_dev_t dev, int K, int N, int M, ggml_type wt) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);
    bool ok = ggml_backend_dev_supports_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

// supports_op for a mul_mat whose src0 is a COMPUTED tensor (op != GGML_OP_NONE), the
// attention QK/AV case. We only offload static leaf weights, so this must be rejected.
static bool mm_computed_w_supported(ggml_backend_dev_t dev, int K, int N, int M) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * W  = ggml_scale(ctx, W0, 1.0f);   // W->op == GGML_OP_SCALE (non-leaf)
    ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);
    bool ok = ggml_backend_dev_supports_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

// offload_op (= MUL_MAT && supports_op) for a non-MUL_MAT compute op: a plain ADD must
// never be pulled onto the NPU.
static bool add_offloaded(ggml_backend_dev_t dev, int K, int M) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst = ggml_add(ctx, A, B);
    bool ok = ggml_backend_dev_offload_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

// offload_op for the canonical offloadable GEMM; must agree with supports_op.
static bool mm_offloaded(ggml_backend_dev_t dev, int K, int N, int M) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);
    bool ok = ggml_backend_dev_offload_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

// supports_op for a FLASH_ATTN_EXT op (the LLM-prefill attention offload). Q is
// [head_dim, n_tokens, n_head] F32; K/V are [head_dim, n_kv, n_kv_heads] (type kv_t);
// the mask is [n_kv, n_tokens_pad] F16. The ggml constructor (ggml_flash_attn_ext)
// already enforces head_dim / GQA / batch consistency, so this exercises the dtype and
// size floors our gate adds: n_tokens>=min_t, n_kv>=min_kv, head_dim%32, K/V F16,
// ALiBi off. (The V/mask-extent guards in supports_op defend against malformed graphs
// the public constructor can't build, so they aren't reachable from here.)
static bool fa_supported(ggml_backend_dev_t dev, int head_dim, int n_tokens, int n_head,
                         int n_kv, int n_kv_heads, ggml_type kv_t, float max_bias) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, kv_t, head_dim, n_kv, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, kv_t, head_dim, n_kv, n_kv_heads, 1);
    const int n_tok_pad = ((n_tokens + 31) / 32) * 32;             // mask rows padded (>= n_tokens)
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tok_pad);
    ggml_tensor * dst = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, max_bias, 0.0f);
    bool ok = ggml_backend_dev_supports_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

// supports_op for a MoE routed-expert op: As[K,N,n_expert] x B[K,1,n_tokens] with
// Ids[n_used,n_tokens] -> [N,n_used,n_tokens]. `name` is load-bearing, not decoration: the
// residency pre-flight keys its ledger on the weight's name, and an unnamed stack has no
// identity the resident expert cache could hold it under.
static bool moe_supported(ggml_backend_dev_t dev, int K, int N, int n_expert, int n_tokens,
                          ggml_type wt, const char * name, int n_used_in = 4) {
    const int n_used = n_used_in;
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, wt, K, N, n_expert);
    if (name) ggml_set_name(as, name);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, as, b, ids);
    bool ok = ggml_backend_dev_supports_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

#define CHECK(cond, msg) do { bool _c = (cond); \
    printf("  [%s] %s\n", _c ? "PASS" : "FAIL", msg); if (!_c) fails++; } while (0)

// ---------------------------------------------------------------------------
// MoE expert placement (MUL_MAT_ID) — the residency pre-flight
// ---------------------------------------------------------------------------
// ROCKET_MOE's three states and the pre-flight's budget are read ONCE per process (cached
// getenv, and the budget is deliberately frozen so it cannot shrink under its own ingest),
// so each mode gets its own forked child rather than a setenv the running process would
// not see. Each child returns its own failure count.
//
// The pre-flight's ledger is also cumulative across the stacks a child asks about, which is
// the point of it -- so a child asserts at most one "does not fit" case, and asks about the
// fitting stacks first.
static int moe_child(const char * moe, const char * budget_mb) {
    if (moe)       setenv("ROCKET_MOE", moe, 1); else unsetenv("ROCKET_MOE");
    if (budget_mb) setenv("ROCKET_MOE_CACHE_MB", budget_mb, 1);

    ggml_backend_reg_t reg = ggml_backend_rocket_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    const bool rk76 = rk_is_rk3576();
    int fails = 0;

    // K/N are the int8 route's contract (K%32, N%32, K>=64, N>=64) and K picks a legal
    // quant K-group; n_tokens clears the prefill floor (ROCKET_MOE_MIN_TOKENS, 512).
    //
    // These are gpt-oss-20b's REAL expert dimensions, not a token 256x256, because the
    // placement gate now has a per-dispatch WORK floor (M_e * K * N) and a toy shape sits
    // three orders of magnitude under it -- every "should offload" case would pass for the
    // wrong reason, or rather fail for one. The DeepSeek-V2-Lite cases below use that
    // model's real 2048x1408 for the same reason: the whole point of those cells is that
    // K * N differs 2.88x between the two architectures.
    const int EK = 2880, EN = 2880, NE = 8, NT = 512;
    const int DK = 2048, DN = 1408;                  // DeepSeek-V2-Lite's expert GEMM

    if (!moe) {   // AUTO — the default
        printf("  -- ROCKET_MOE unset (AUTO) --\n");
        // A quantized expert stack that fits: the native-quant route, which is the only MoE
        // route measured to WIN. RK3576 declines it -- there is no encoder for this
        // datapath on that part, so claiming it would hand the scheduler host-reference work.
        CHECK( moe_supported(dev, EK, EN, NE, NT, GGML_TYPE_Q8_0, "blk.0.ffn_gate_exps.weight") == !rk76,
               rk76 ? "MoE: Q8_0 experts, fits budget  -> CPU (no expert encoder on rk3576)"
                    : "MoE: Q8_0 experts, fits budget  -> offload (native-quant, reserved)" );
        // An F16/BF16/F32 stack has no per-micro-batch host dequant to delete, so the only
        // route left for it is the streaming one that measured 0.42-0.90x. Not a default.
        CHECK( !moe_supported(dev, EK, EN, NE, NT, GGML_TYPE_F16, "blk.1.ffn_gate_exps.weight"),
               "MoE: F16 experts                -> CPU (no dequant to delete)" );
        // No stable name -> nothing the resident cache could key the ingest on, so the
        // pre-flight cannot reserve it and the runtime would stream it.
        CHECK( !moe_supported(dev, EK, EN, NE, NT, GGML_TYPE_Q8_0, nullptr),
               "MoE: unnamed expert stack       -> CPU (nothing to reserve)" );
        // Decode and short micro-batches: M_e ~ n_tokens*n_used/n_expert is too small.
        CHECK( !moe_supported(dev, EK, EN, NE, 1, GGML_TYPE_Q8_0, "blk.2.ffn_gate_exps.weight"),
               "MoE: n_tokens=1 (decode)        -> CPU" );
        // The shape contract, same int8 k-group/N-group the dense W8A8 route uses.
        CHECK( !moe_supported(dev, 80, EN, NE, NT, GGML_TYPE_Q8_0, "blk.3.ffn_gate_exps.weight"),
               "MoE: K%32!=0 (K=80)             -> CPU" );
        // THE PER-EXPERT ROW FLOOR, which is an architecture property and not a prompt one.
        // Both cases below clear the n_tokens gate; what separates them is n_used/n_expert.
        // gpt-oss's 4-of-32 gives 512 tokens -> 64 rows an expert, at the granule, and it
        // wins. DeepSeek-V2-Lite's 6-of-64 gives 48 -- under the granule its GEMM is padded
        // up to -- and it measured 19.09 t/s against 26.04 with the experts on the CPU, a 27%
        // regression at 100% residency. Declining it is what keeps the default safe on a
        // model nobody re-benched before flipping the flag.
        CHECK( moe_supported(dev, EK, EN, 32, 512, GGML_TYPE_Q8_0,
                             "blk.4.ffn_gate_exps.weight", 4) == !rk76,
               rk76 ? "MoE: 4-of-32, M_e=64 (at granule) -> CPU (no expert encoder on rk3576)"
                    : "MoE: 4-of-32, M_e=64 (at granule) -> offload" );
        CHECK( !moe_supported(dev, DK, DN, 64, 512, GGML_TYPE_Q8_0,
                              "blk.5.ffn_gate_exps.weight", 6),
               "MoE: 6-of-64, M_e=48 (under granule) -> CPU (the GEMM would be padding)" );
        CHECK( moe_supported(dev, DK, DN, 64, 2048, GGML_TYPE_Q8_0,
                             "blk.6.ffn_gate_exps.weight", 6) == !rk76,
               rk76 ? "MoE: 6-of-64, M_e=192 at pp2048      -> CPU (rk3576)"
                    : "MoE: 6-of-64, M_e=192 at pp2048      -> offload (same model, longer prefill)" );

        // THE PER-DISPATCH WORK FLOOR, which is the floor the row floor cannot be. These
        // three cells are the measurement that put it there: the SAME per-expert row count
        // is a win on one architecture and a loss on the other, because the work a dispatch
        // carries is M_e * K * N and gpt-oss's expert GEMM is 2.88x DeepSeek-V2-Lite's.
        // Measured 2026-08-27 (default / experts-on-CPU, 100% resident, 0 streamed):
        // gpt-oss M_e=96 reads 1.77x, DeepSeek M_e=96 reads 0.94x, DeepSeek M_e=144 1.25x.
        // A row floor has no value that separates the first two. See supports_op.
        CHECK( moe_supported(dev, EK, EN, 32, 768, GGML_TYPE_Q8_0,
                             "blk.7.ffn_gate_exps.weight", 4) == !rk76,
               rk76 ? "MoE: 4-of-32 M_e=96, 796 MMAC/dispatch -> CPU (rk3576)"
                    : "MoE: 4-of-32 M_e=96, 796 MMAC/dispatch -> offload" );
        // M_e=72 is the measured LOSER and the cell that proves a row floor cannot work: it
        // clears the granule (72 >= 64) and still reads 0.94x over four adjacent pairs, none
        // above 0.970, while gpt-oss at a LOWER row count (64) reads 1.64x.
        CHECK( !moe_supported(dev, DK, DN, 64, 768, GGML_TYPE_Q8_0,
                              "blk.10.ffn_gate_exps.weight", 6),
               "MoE: 6-of-64 M_e=72, 208 MMAC/dispatch -> CPU (over the granule, under the work floor)" );
        CHECK( !moe_supported(dev, DK, DN, 64, 1024, GGML_TYPE_Q8_0,
                              "blk.8.ffn_gate_exps.weight", 6),
               "MoE: 6-of-64 M_e=96, 277 MMAC/dispatch -> CPU (the boundary cell)" );
        CHECK( moe_supported(dev, DK, DN, 64, 1536, GGML_TYPE_Q8_0,
                             "blk.9.ffn_gate_exps.weight", 6) == !rk76,
               rk76 ? "MoE: 6-of-64 M_e=144, 415 MMAC/dispatch -> CPU (rk3576)"
                    : "MoE: 6-of-64 M_e=144, 415 MMAC/dispatch -> offload (over the floor)" );
        return fails;
    }
    if (moe[0] == '0') {
        printf("  -- ROCKET_MOE=0 (OFF) --\n");
        CHECK( !moe_supported(dev, EK, EN, NE, NT, GGML_TYPE_Q8_0, "blk.0.ffn_gate_exps.weight"),
               "MoE: Q8_0 experts, ROCKET_MOE=0 -> CPU" );
        return fails;
    }
    printf("  -- ROCKET_MOE=1 (FORCED) --\n");
    // FORCED claims everything the handler can compute, which is what the archived MoE
    // measurements were taken under: the fp16 streaming route included, and with no
    // residency reservation. It is the A/B arm, not the recommended setting.
    CHECK( moe_supported(dev, EK, EN, NE, NT, GGML_TYPE_F16, "blk.0.ffn_gate_exps.weight") == !rk76,
           rk76 ? "MoE: F16 experts, ROCKET_MOE=1  -> CPU (rk3576 has no MoE route at all)"
                : "MoE: F16 experts, ROCKET_MOE=1  -> offload (the streaming A/B arm)" );
    CHECK( !moe_supported(dev, EK, EN, NE, 1, GGML_TYPE_Q8_0, "blk.1.ffn_gate_exps.weight"),
           "MoE: n_tokens=1, ROCKET_MOE=1   -> CPU (the prefill floor still holds)" );
    return fails;
}

// The budget-exhaustion case gets its own child: it needs a budget small enough that a
// stack cannot fit, and the budget is frozen for the life of the process.
static int moe_budget_child(void) {
    setenv("ROCKET_MOE_CACHE_MB", "1", 1);   // 1 MB: smaller than any real expert stack
    unsetenv("ROCKET_MOE");
    ggml_backend_reg_t reg = ggml_backend_rocket_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    int fails = 0;
    printf("  -- ROCKET_MOE unset (AUTO), ROCKET_MOE_CACHE_MB=1 --\n");
    // The whole point of the pre-flight: a stack that will not fit is declined OUTRIGHT,
    // before any ingest, so the experts run on the CPU instead of half-ingesting into the
    // streamed-remainder loss that partial residency is.
    // Real expert dimensions, deliberately: a toy 256x256 stack is declined by the
    // per-dispatch work floor long before the budget is consulted, so the case would pass
    // while testing nothing about the pre-flight.
    CHECK( !moe_supported(dev, 2880, 2880, 8, 512, GGML_TYPE_Q8_0, "blk.0.ffn_gate_exps.weight"),
           "MoE: stack over the RAM budget  -> CPU (pre-flight declines, no half-ingest)" );
    return fails;
}

// Run one child's cases in a forked process and return its failure count. The knobs the
// child sets are cached-getenv statics; a fork is what keeps them independent.
static int run_moe_child(int (*fn)(void)) {
    fflush(stdout);
    pid_t pid = fork();
    if (pid < 0) { printf("  [FAIL] fork() for the MoE placement cases\n"); return 1; }
    if (pid == 0) { int f = fn(); fflush(stdout); _exit(f > 100 ? 100 : f); }
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st)) { printf("  [FAIL] MoE placement child did not exit cleanly\n"); return 1; }
    return WEXITSTATUS(st);
}
static int moe_auto_child(void)   { return moe_child(nullptr, nullptr); }
static int moe_off_child(void)    { return moe_child("0", nullptr); }
static int moe_forced_child(void) { return moe_child("1", nullptr); }

int main() {
    // The RK3576's only matmul route is the W8A8 one, and ROCKET_INT8 selects it. Set it
    // BEFORE the first supports_op call: rocket_int8_mode_on() caches the knob on first
    // read, so a later setenv would not be seen. With it unset the part offloads no matmul
    // at all, which is correct but asserts nothing.
    const bool rk76 = rk_is_rk3576();
    if (rk76) setenv("ROCKET_INT8", "1", 1);

    ggml_backend_reg_t reg = ggml_backend_rocket_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        fprintf(stderr, "rocket backend registry/device unavailable\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);

    int fails = 0;
    const int K = 256, N = 128, M = 256;   // valid F16/F32 shape (floor = rocket_min_m, default 4)
    const int Mq = 512;                    // quantized floor = rocket_min_m_quant (default 512)

    // ACCEPT — the canonical static-weight GEMM we offload (F16, F32, and BF16 weights;
    // a BF16 weight is decoded to fp16 on the streaming/mt path, same as F32).
    CHECK( mm_supported(dev, K, N, M, GGML_TYPE_F16),  "F16 leaf weight, valid shape  -> offload" );
    CHECK( mm_supported(dev, K, N, M, GGML_TYPE_F32),  "F32 leaf weight, valid shape  -> offload" );
    // BF16 reaches the NPU by being decoded to fp16 — an RK3588-only route. The RK3576's
    // W8A8 entry takes an F16/F32 leaf and quantizes it itself; bf16 is not on its list.
    CHECK( mm_supported(dev, K, N, M, GGML_TYPE_BF16) == !rk76,
           rk76 ? "BF16 leaf weight              -> CPU (no bf16 decode route on rk3576)"
                : "BF16 leaf weight, valid shape -> offload (decoded to fp16)" );

    // ACCEPT — quantized leaf weights: dequantized to fp16 on the streaming/mt path, but
    // only at M >= rocket_min_m_quant (the per-microbatch dequant needs more rows to
    // amortize than the F16 path). K=256 is a whole number of blocks for Q8_0 (block 32)
    // and the K-quants (256).
    // On the RK3576 there is no dequant->fp16 path to reach, so a ggml-quantized weight
    // stays on the CPU whatever M is.
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q8_0) == !rk76,
           rk76 ? "Q8_0 leaf                            -> CPU (no dequant->fp16 on rk3576)"
                : "Q8_0 leaf, K%32==0,  M>=quant floor -> offload" );
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q4_K) == !rk76,
           rk76 ? "Q4_K leaf                            -> CPU (no dequant->fp16 on rk3576)"
                : "Q4_K leaf, K%256==0, M>=quant floor -> offload" );
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q6_K) == !rk76,
           rk76 ? "Q6_K leaf                            -> CPU (no dequant->fp16 on rk3576)"
                : "Q6_K leaf, K%256==0, M>=quant floor -> offload" );
    // REJECT — a quantized weight below the higher quant floor (but above the F16 floor):
    // the dequant doesn't amortize, so a short quant prefill stays on the CPU even at an M
    // that an F16 weight offloads at. Pins the rocket_min_m_quant > rocket_min_m gap.
    CHECK( !mm_supported(dev, K, N, M, GGML_TYPE_Q4_K), "Q4_K leaf, M(256)<quant floor   -> CPU" );
    // (A K-quant weight with K not a multiple of its 256-block is rejected by the
    // K%ggml_blck_size guard in supports_op, but ggml itself can't construct such a
    // tensor -- row_size truncates to a transposed-looking layout that aborts in
    // ggml_mul_mat -- and a real GGUF never has one, so there's nothing to assert here.)

    // REJECT — shape-contract violations stay on the CPU.
    CHECK( !mm_supported(dev, 80,  N, M, GGML_TYPE_F16), "K%32!=0 (K=80)   -> CPU" );
    CHECK( !mm_supported(dev, K,  70, M, GGML_TYPE_F16), "N%16!=0 (N=70)   -> CPU" );
    // The two parts round N differently: the RK3588 fp16 route wants N%16, the RK3576 int8
    // weight N-group is 32. N=112 is a multiple of 16 and not of 32, so it is the shape
    // that separates them — offloadable on one part, CPU on the other.
    CHECK( mm_supported(dev, K, 112, M, GGML_TYPE_F16) == !rk76,
           rk76 ? "N%32!=0 (N=112)  -> CPU (rk3576 int8 N-group is 32)"
                : "N=112 (N%16==0) -> offload" );
    CHECK( !mm_supported(dev, 32,  N, M, GGML_TYPE_F16), "K<64 (K=32)      -> CPU" );
    CHECK( !mm_supported(dev, K,  48, M, GGML_TYPE_F16), "N<64 (N=48)      -> CPU" );
    CHECK( !mm_supported(dev, K,   N, 1, GGML_TYPE_F16), "M=1 decode GEMV  -> CPU" );

    // REJECT — not a static leaf weight (computed src0 = attention QK/AV).
    CHECK( !mm_computed_w_supported(dev, K, N, M), "computed (non-leaf) src0 -> CPU" );

    // offload_op agrees with supports_op for MUL_MAT, and rejects a non-MUL_MAT op.
    CHECK(  mm_offloaded(dev, K, N, M), "offload_op: valid GEMM   -> offload" );
    CHECK( !add_offloaded(dev, K, M),   "offload_op: GGML_OP_ADD  -> CPU" );

    // FLASH_ATTN_EXT placement (the LLM-prefill attention offload, default-on). FA has no
    // weight src, so the scheduler never consults offload_op for it -- it reaches the NPU
    // via supports_op + the expand passes (see ggml_backend_rocket_device_offload_op). So
    // pinning supports_op is exactly what guards FA placement; assert its accept/reject set.
    const int HD = 64, NH = 8, NKVH = 2;   // head_dim%32==0, GQA 8/2 divides cleanly
    // Every compute path in the FA handler goes through rocket_matmul_fp16, which refuses
    // on any profile that is not rk3588 — so the RK3576 declines the op rather than claim
    // work only the single-threaded host reference could do. The remaining FA cases below
    // are rejected on both parts, for the reasons their labels give.
    CHECK(  fa_supported(dev, HD, 256, NH, 1024, NKVH, GGML_TYPE_F16, 0.0f) == !rk76,
            rk76 ? "FA: F16 KV                                -> CPU (no fp16 attention encoder on rk3576)"
                 : "FA: F16 KV, n_tokens>=min_t, n_kv>=min_kv -> offload" );
    CHECK( !fa_supported(dev, HD,   1, NH, 1024, NKVH, GGML_TYPE_F16, 0.0f),
            "FA: decode (n_tokens=1 < min_t)           -> CPU" );
    CHECK( !fa_supported(dev, HD, 256, NH,  512, NKVH, GGML_TYPE_F16, 0.0f),
            "FA: short context (n_kv=512 < min_kv)     -> CPU" );
    CHECK( !fa_supported(dev, 48, 256, NH, 1024, NKVH, GGML_TYPE_F16, 0.0f),
            "FA: head_dim%32!=0 (48)                   -> CPU" );
    CHECK( !fa_supported(dev, HD, 256, NH, 1024, NKVH, GGML_TYPE_F32, 0.0f),
            "FA: K/V not F16                           -> CPU" );
    CHECK( !fa_supported(dev, HD, 256, NH, 1024, NKVH, GGML_TYPE_F16, 8.0f),
            "FA: ALiBi (max_bias>0)                    -> CPU" );

    // MoE routed-expert placement. Each mode is a forked child (see run_moe_child).
    fails += run_moe_child(moe_auto_child);
    fails += run_moe_child(moe_budget_child);
    fails += run_moe_child(moe_off_child);
    fails += run_moe_child(moe_forced_child);

    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
