// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-int4.cpp — measure native int4 (W4A4) matmul quality in the rocket
 * backend against an fp16 CPU golden, on Gemma-scale shapes. A FAST proxy for the
 * in-model int4 quality (cosine per matmul) that drives the quant-scheme iteration
 * (per-channel / Hadamard / group-wise) without the slow 12B perplexity run.
 *
 * The native NPU int4 matmul is SYMMETRIC W4A4 (int4 weights x int4 activations);
 * this forces ROCKET_INT4=1 so the rocket backend routes MUL_MAT through it, then
 * compares cosine(int4, fp16) per shape. W4A4 is far lossier than W8A8, so the bar
 * is cosine (not the fp16-tolerance the F16/quant gates use). Sweep the quality lever
 * from the caller env: ROCKET_INT4_HADAMARD=1 (the outlier rotation). The int16-partial
 * saturation cap (Kt<=480) is baked into the in-model int4 path (rocket_matmul_int4_ex),
 * so the test does not need to set ROCKET_MM_KT.
 *
 * Off-device (no NPU) -> SKIP (77). On-device it PASSES if every shape clears the
 * cosine floor (default 0.90; tighten via ROCKET_INT4_COS_MIN) and is finite.
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rocket.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>

// dst = mul_mat(W,X); W is F16 weights [K,N], X is F32 input [K,M] -> out [N*M] f32.
// (Identical to test-rocket-matmul's run(): the rocket backend's path is chosen by
// the ROCKET_* env, so with ROCKET_INT4=1 this exercises the native int4 matmul.)
static bool run(ggml_backend_t backend, int K, int N, int M,
                const std::vector<float> & Wf, const std::vector<float> & Xf,
                std::vector<float> & out)
{
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(W); ggml_set_input(X);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }
    std::vector<ggml_fp16_t> W16((size_t)K*N);
    ggml_fp32_to_fp16_row(Wf.data(), W16.data(), (int64_t)K*N);
    ggml_backend_tensor_set(W, W16.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, Xf.data(), 0, ggml_nbytes(X));
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (ok) { out.resize((size_t)N*M); ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst)); }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return -2.0;  // NaN/Inf -> fail
        dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (sqrt(na)*sqrt(nb));
}

int main() {
    // Force the int4 path in the rocket backend; the caller can still set
    // ROCKET_INT4_HADAMARD to sweep the quality lever. The int16-saturation Kt cap is
    // baked into rocket_matmul_int4_ex (480), so no ROCKET_MM_KT is needed here.
    setenv("ROCKET_INT4", "1", 1);

    const char * cm = getenv("ROCKET_INT4_COS_MIN");
    const double cos_min = cm ? atof(cm) : 0.90;

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    // Gemma-4-12B matmul shapes (K=feature in, N=feature out): attn proj, FFN
    // gate/up/down. K%32, N%64 (int4 weight k-group), M%4-padded internally.
    struct { int K, N, M; const char * name; } shapes[] = {
        { 3840, 4096, 512, "attn_qkv-ish" },
        { 4096, 3840, 512, "attn_out-ish" },
        { 3840,15360, 256, "ffn_gate/up"  },
        {15360, 3840, 256, "ffn_down"     },
        { 3840, 4096, 128, "prefill-128"  },
    };

    std::mt19937 rng(1234);
    std::normal_distribution<float> wdist(0.0f, 0.02f);   // weights ~ N(0, 0.02) (Gemma-ish)
    std::normal_distribution<float> xdist(0.0f, 1.0f);    // activations ~ N(0,1)

    int fails = 0;
    printf("int4 W4A4 quality vs fp16 (cos_min=%.3f, HADAMARD=%s, Kt cap=480 baked-in)\n",
           cos_min, getenv("ROCKET_INT4_HADAMARD") ? getenv("ROCKET_INT4_HADAMARD") : "0");
    for (auto s : shapes) {
        std::vector<float> Wf((size_t)s.K*s.N), Xf((size_t)s.K*s.M), golden, got;
        for (auto & w : Wf) w = wdist(rng);
        for (auto & x : Xf) x = xdist(rng);
        // Inject per-channel activation outliers (a few feature channels ~30x): the
        // realistic LLM regime the Hadamard rotation exists to tame.
        for (int c = 0; c < s.K; c += 512)
            for (int m = 0; m < s.M; m++) Xf[(size_t)m*s.K + c] *= 30.0f;

        if (!run(cpu, s.K, s.N, s.M, Wf, Xf, golden)) { fprintf(stderr, "cpu run failed\n"); fails++; continue; }
        if (!run(rocket, s.K, s.N, s.M, Wf, Xf, got))  { fprintf(stderr, "rocket run failed\n"); fails++; continue; }

        double cos = cosine(golden, got);
        bool pass = cos >= cos_min;
        printf("  %-14s K=%5d N=%5d M=%4d  cos=%.5f -> %s\n",
               s.name, s.K, s.N, s.M, cos, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "SOME INT4 SHAPES BELOW COS FLOOR" : "ALL INT4 SHAPES PASS");
    return fails ? 1 : 0;
}
