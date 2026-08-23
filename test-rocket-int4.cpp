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
#include "ggml-cpu.h"
#include "ggml-rocket.h"
#include "test-common.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>

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

        if (!rk_run_mul_mat(cpu, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xf, golden, s.name)) { fprintf(stderr, "cpu run failed\n"); fails++; continue; }
        if (!rk_run_mul_mat(rocket, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xf, got, s.name))  { fprintf(stderr, "rocket run failed\n"); fails++; continue; }

        double cos = rk_cosine(golden, got);
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
