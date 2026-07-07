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
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rocket.h"

#include <cstdio>
#include <cstdlib>

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

#define CHECK(cond, msg) do { bool _c = (cond); \
    printf("  [%s] %s\n", _c ? "PASS" : "FAIL", msg); if (!_c) fails++; } while (0)

int main() {
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
    CHECK( mm_supported(dev, K, N, M, GGML_TYPE_BF16), "BF16 leaf weight, valid shape -> offload (decoded to fp16)" );

    // ACCEPT — quantized leaf weights: dequantized to fp16 on the streaming/mt path, but
    // only at M >= rocket_min_m_quant (the per-microbatch dequant needs more rows to
    // amortize than the F16 path). K=256 is a whole number of blocks for Q8_0 (block 32)
    // and the K-quants (256).
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q8_0), "Q8_0 leaf, K%32==0,  M>=quant floor -> offload" );
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q4_K), "Q4_K leaf, K%256==0, M>=quant floor -> offload" );
    CHECK( mm_supported(dev, K, N, Mq, GGML_TYPE_Q6_K), "Q6_K leaf, K%256==0, M>=quant floor -> offload" );
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
    CHECK(  fa_supported(dev, HD, 256, NH, 1024, NKVH, GGML_TYPE_F16, 0.0f),
            "FA: F16 KV, n_tokens>=min_t, n_kv>=min_kv -> offload" );
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

    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
