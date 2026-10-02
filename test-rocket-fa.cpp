// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-fa.cpp — the FLASH_ATTN_EXT handler against the CPU backend, including a q.k
 * that fp16 cannot hold.
 *
 * The handler hands the NPU fp16 Q and K and gets back an fp16 score surface, while
 * llama.cpp declares F32 on every attention op it builds. The CPU backend computes q.k in
 * fp32 and scales it after. So a raw q.k past 65504 is exact on the CPU and overflows an
 * fp16 surface. The handler folds the op's scale into Q (ROCKET_FA_QSCALE, default on), which
 * puts scale*q.k on the surface instead.
 *
 *   overflow   Q and K are scaled one-hot directions. Each query matches 16 of the 1024 keys,
 *              with raw q.k from 60000 to 135000 across them, so the scaled logits run from
 *              7500 to 16875 at head_dim 64 and the softmax is one-hot on the largest. The
 *              output is that key's V row. An overflowed surface either reads inf, which
 *              makes the softmax NaN, or saturates at 65504, which ties the matching keys and
 *              averages their V rows. Both are far off the one-hot answer, so the gate cannot
 *              pass on an unfolded surface whichever way the converter overflows.
 *   typical    random Q, K and V in [-1, 1] under a causal mask: the fold must not cost
 *              accuracy where nothing overflows.
 *
 * THE ORACLE IS A HOST REFERENCE IN DOUBLE, NOT THE CPU BACKEND. On an ARMv8.2 build the
 * CPU backend's fp16 dot product accumulates q.k in fp16, so its own flash attention
 * overflows on the first case and returns NaN; it is printed beside each case for
 * information and scored on nothing. A NaN on either side fails a case outright, since a
 * NaN compares false against any tolerance.
 *
 * Every case also reads the backend's "fa" route counter, so a case the host reference
 * computed cannot pass as an NPU result. RK3588 only: the attention route does not exist on
 * the RK3576, and the gate SKIPs there.
 *
 * Run with ROCKET_FA_QSCALE=0 to see the overflow case fail on the unfolded surface.
 */
#include "ggml-cpu.h"
#include "ggml-rocket.h"
#include "test-common.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdint>

struct fa_case {
    const char * name;
    int D, T, H, KV, HK;
    bool overflow;          // the one-hot overflow construction, else typical random
    double tol;             // max abs error against the CPU backend
};

static bool run_fa(ggml_backend_t be, const fa_case & c, float scale,
                   const std::vector<float> & Q, const std::vector<float> & K,
                   const std::vector<float> & V, const std::vector<float> & M,
                   std::vector<float> & out)
{
    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                            /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.D, c.T, c.H);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.D, c.KV, c.HK);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.D, c.KV, c.HK);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.KV, c.T);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(m);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);   // [D, H, T]
    ggml_set_output(o);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }
    std::vector<ggml_fp16_t> K16(K.size()), V16(V.size()), M16(M.size());
    ggml_fp32_to_fp16_row(K.data(), K16.data(), (int64_t)K.size());
    ggml_fp32_to_fp16_row(V.data(), V16.data(), (int64_t)V.size());
    ggml_fp32_to_fp16_row(M.data(), M16.data(), (int64_t)M.size());
    ggml_backend_tensor_set(q, Q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, K16.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, V16.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(m, M16.data(), 0, ggml_nbytes(m));
    bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize((size_t)c.D * c.H * c.T);
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// softmax(scale * q.k + mask) . v in double, with Q rounded to fp16 as both backends round
// it and K, V, mask exactly the fp16 values the op carries. Output [D, H, T] like the op's.
static void ref_fa(const fa_case & c, float scale, const std::vector<float> & Q,
                   const std::vector<float> & K, const std::vector<float> & V,
                   const std::vector<float> & M, std::vector<double> & out)
{
    out.assign((size_t)c.D * c.H * c.T, 0.0);
    std::vector<double> sc(c.KV);
    const int g = c.H / c.HK;
    auto h16 = [](float x) { return (double)ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); };
    for (int h = 0; h < c.H; h++) {
        const int hk = h / g;
        for (int t = 0; t < c.T; t++) {
            double mx = -INFINITY;
            for (int j = 0; j < c.KV; j++) {
                double acc = 0.0;
                for (int d = 0; d < c.D; d++)
                    acc += h16(Q[((size_t)h * c.T + t) * c.D + d]) *
                           h16(K[((size_t)hk * c.KV + j) * c.D + d]);
                sc[j] = scale * acc + h16(M[(size_t)t * c.KV + j]);
                if (sc[j] > mx) mx = sc[j];
            }
            double sum = 0.0;
            for (int j = 0; j < c.KV; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int d = 0; d < c.D; d++) {
                double a = 0.0;
                for (int j = 0; j < c.KV; j++) a += sc[j] * h16(V[((size_t)hk * c.KV + j) * c.D + d]);
                out[((size_t)t * c.H + h) * c.D + d] = a / sum;
            }
        }
    }
}

static void fill_case(const fa_case & c, std::vector<float> & Q, std::vector<float> & K,
                      std::vector<float> & V, std::vector<float> & M)
{
    Q.assign((size_t)c.D * c.T * c.H, 0.0f);
    K.assign((size_t)c.D * c.KV * c.HK, 0.0f);
    V.resize((size_t)c.D * c.KV * c.HK);
    M.assign((size_t)c.KV * c.T, 0.0f);
    uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(c.D * 131 + c.KV);
    auto rnd = [&s]() {                      // uniform in [-1, 1), period-free
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((double)(s >> 11) / (double)(1ull << 52)) - 1.0f;
    };
    for (auto & x : V) x = rnd();
    if (c.overflow) {
        // Query (t, h) points along direction (t + h) % D with amplitude 1000; key j points
        // along j % D with amplitude 60 + 5 * (j / D). Every magnitude is exact in fp16.
        for (int h = 0; h < c.H; h++)
            for (int t = 0; t < c.T; t++)
                Q[((size_t)h * c.T + t) * c.D + (size_t)((t + h) % c.D)] = 1000.0f;
        for (int hk = 0; hk < c.HK; hk++)
            for (int j = 0; j < c.KV; j++)
                K[((size_t)hk * c.KV + j) * c.D + (size_t)(j % c.D)] = 60.0f + 5.0f * (float)(j / c.D);
    } else {
        for (auto & x : Q) x = rnd();
        for (auto & x : K) x = rnd();
        // causal: token t (the last T of the KV positions) sees keys up to KV - T + t
        for (int t = 0; t < c.T; t++)
            for (int j = c.KV - c.T + t + 1; j < c.KV; j++) M[(size_t)t * c.KV + j] = -INFINITY;
    }
}

int main() {
    const fa_case cases[] = {
        { "overflow d64",  64, 32, 4, 1024, 2, true,  1e-3 },
        { "overflow d128", 128, 32, 4, 1024, 2, true,  1e-3 },
        { "typical d64",   64, 64, 8, 1024, 2, false, 2e-2 },
        { "typical d128", 128, 64, 4, 1536, 4, false, 2e-2 },
    };

    if (rk_is_rk3576()) { fprintf(stderr, "RK3576: no attention route to test -> SKIP\n"); return 77; }
    if (!rk_device_opens()) return 77;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    int fails = 0;
    for (const auto & c : cases) {
        std::vector<float> Q, K, V, M, oc, orr;
        fill_case(c, Q, K, V, M);
        const float scale = 1.0f / sqrtf((float)c.D);
        std::vector<double> want;
        ref_fa(c, scale, Q, K, V, M, want);
        const long fa0 = ggml_backend_rocket_route_ops(rocket, "fa");
        const long fb0 = ggml_backend_rocket_cpu_fallbacks(rocket);
        bool ok = run_fa(cpu, c, scale, Q, K, V, M, oc) && run_fa(rocket, c, scale, Q, K, V, M, orr);
        const long fa = ggml_backend_rocket_route_ops(rocket, "fa") - fa0;
        const long fb = ggml_backend_rocket_cpu_fallbacks(rocket) - fb0;
        // The worst error, where a NaN or inf on the scored side counts as an infinite one.
        auto worst = [&](const std::vector<float> & got, size_t & nf) {
            double m = 0.0;
            nf = 0;
            for (size_t i = 0; i < want.size(); i++) {
                if (!std::isfinite(got[i])) { nf++; m = INFINITY; continue; }
                double d = fabs((double)got[i] - want[i]);
                if (d > m) m = d;
            }
            return m;
        };
        size_t nf_npu = 0, nf_cpu = 0;
        const double e_npu = ok ? worst(orr, nf_npu) : INFINITY;
        const double e_cpu = ok ? worst(oc, nf_cpu) : INFINITY;
        const bool pass = ok && fa == 1 && fb == 0 && nf_npu == 0 && e_npu <= c.tol;
        printf("%-14s D %3d T %2d H %d KV %4d HK %d: npu max err %.3g (%zu non-finite, tol %.0e), "
               "cpu backend %.3g (%zu non-finite), fa route %ld, fallbacks %ld -> %s\n",
               c.name, c.D, c.T, c.H, c.KV, c.HK, e_npu, nf_npu, c.tol, e_cpu, nf_cpu, fa, fb,
               pass ? "PASS" : "FAIL");
        fails += !pass;
    }
    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
