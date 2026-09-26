// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rk3576-w8a8.cpp — the RK3576 W8A8 route against the CPU backend.
 *
 * That route is the only matmul route on the RK3576 and it carries the most bespoke
 * arithmetic in this backend: a mandatory Hadamard rotation along K, a per-tensor
 * activation scale with a per-output-channel weight scale, a per-column output requant
 * whose scale is FROZEN from a two-pass calibration bootstrap and divided by a safety
 * factor, and an optional host K-split whose f32 partials are summed. None of the other
 * gates reach it: test-rocket-matmul / -int4 / -bf16 all exercise RK3588 paths that
 * refuse on this part by construction, and test-rocket-placement is metadata only.
 *
 * WHAT IT CHECKS, and why the bar is a cosine rather than a tolerance. The route is
 * int8 end to end, so it is not bit-faithful to fp32 and never claims to be; what it
 * claims is that composed over a model it costs 1.008x-1.020x wikitext-2 perplexity.
 * A per-matmul cosine is the fast proxy for that, the same proxy test-rocket-int4 uses
 * for W4A4. The floor is deliberately loose enough that ordinary int8 error passes and
 * tight enough that the failures this route actually produces do not: a frozen scale
 * that never got calibrated, a rotation that did not happen, a K-split partial summed
 * at the wrong scale. Each of those collapses the cosine, because each computes a full,
 * correctly sized, entirely plausible WRONG surface -- which is the signature of nearly
 * every defect found on this part.
 *
 * THE CALIBRATION IS STATEFUL, so each shape runs ROCKET_RK3576_NCAL calibration forwards
 * first and only the forward after them is scored. Those passes return a surface correct at
 * a scale ~BOOTMARGIN loose; the frozen scale only takes effect afterwards. The scored
 * forward reads a fresh input, so a stale surface cannot repeat a calibration answer, and
 * it must finish on the W8A8 route with no CPU fallback.
 *
 * Each shape is also gated on its worst output column, since a whole-tensor cosine averages
 * one wrong column into the rest. The host K-split is on unless the caller sets
 * ROCKET_RK3576_KSPLIT, so ffn_down's K=8960 runs on the part rather than reaching the CPU.
 *
 * SKIPS (77) off-device and on any part that is not an RK3576, so it is safe in the
 * default ctest run on either board.
 *
 * Optional env: ROCKET_RK3576_COS_MIN overrides the cosine floor and
 * ROCKET_RK3576_COL_COS_MIN the worst-column one.
 */
#include "ggml-cpu.h"
#include "ggml-rocket.h"
#include "test-common.h"

extern "C" {
#include "rocket_hw_profile.h"
}

#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <random>

int main() {
    const struct rocket_hw_profile * hw = rocket_hw_current();
    if (!hw || !hw->name || strcmp(hw->name, "rk3576") != 0) {
        fprintf(stderr, "not an RK3576 (detected %s) -> SKIP\n",
                (hw && hw->name) ? hw->name : "unknown");
        return 77;
    }
    // Mandatory on this part: supports_op declines every MUL_MAT without it, so an unset
    // knob would score the CPU fallback and pass for the wrong reason.
    setenv("ROCKET_INT8", "1", 1);
    // The K-split too, unless the caller chose: without it ffn_down's K=8960 is declined, a
    // declined shape reaches the CPU fallback and scores an exact 1.0, and the route check
    // below then fails it. A caller who sets 0 asks for exactly that failure.
    setenv("ROCKET_RK3576_KSPLIT", "1", 0);

    const char * cm = getenv("ROCKET_RK3576_COS_MIN");
    const double cos_min = cm ? atof(cm) : 0.995;
    // The worst single output column, gated separately: a whole-tensor cosine averages one
    // wrong column into the rest. The measured worst columns read 0.99916-0.99943 (H96,
    // 2026-09-24); a wrong or dropped column reads near 0, and a dropped K-split partial
    // near 0.87. The floor allows twelve times the worst measured error power.
    const char * ccm = getenv("ROCKET_RK3576_COL_COS_MIN");
    const double col_cos_min = ccm ? atof(ccm) : 0.99;

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    if (!rk_device_opens()) return 77;
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    // Qwen2.5-1.5B's own shapes, which is what the route's accuracy numbers were taken on.
    // K=1536 is the projection K the part takes whole; K=8960 is ffn_down, which runs on the
    // host K-split.
    struct { int K, N, M; const char * name; } shapes[] = {
        { 1536, 1536, 512, "attn_qkv"  },
        { 1536, 8960, 512, "ffn_gate"  },
        { 8960, 1536, 512, "ffn_down"  },
        { 1536, 1536, 128, "prefill-128" },
    };

    std::mt19937 rng(1234);
    std::normal_distribution<float> wdist(0.0f, 0.02f);
    std::normal_distribution<float> xdist(0.0f, 1.0f);

    int fails = 0;
    printf("RK3576 W8A8 vs CPU backend (cos_min=%.4f, col_cos_min=%.4f, KSPLIT=%s)\n",
           cos_min, col_cos_min,
           getenv("ROCKET_RK3576_KSPLIT") ? getenv("ROCKET_RK3576_KSPLIT") : "0");
    // One input drawn per pass. Per-channel activation outliers (a few feature channels
    // ~30x) are the LLM regime this route's mandatory rotation exists to tame, and the one
    // that separates a rotated run from an unrotated one.
    auto draw_input = [&](std::vector<float> & X, int K, int M) {
        for (auto & x : X) x = xdist(rng);
        for (int c = 0; c < K; c += 512)
            for (int m = 0; m < M; m++) X[(size_t)m*K + c] *= 30.0f;
    };
    for (auto s : shapes) {
        std::vector<float> Wf((size_t)s.K*s.N), Xcal((size_t)s.K*s.M), Xs((size_t)s.K*s.M);
        std::vector<float> golden, got;
        for (auto & w : Wf) w = wdist(rng);
        draw_input(Xcal, s.K, s.M);
        // The scored pass reads a FRESH input from the same distribution: the calibration
        // freezes its scales on Xcal, and a stale surface would repeat Xcal's answer, which
        // scoring on Xcal itself could not tell from a right one.
        draw_input(Xs, s.K, s.M);

        // A distinct weight NAME per shape: the W8A8 weight cache -- and the calibration
        // state held beside it -- is keyed on the name, so one name across four shapes
        // would evict and re-calibrate rather than give each shape its own entry.
        if (!rk_run_mul_mat(cpu, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xs, golden, s.name)) {
            fprintf(stderr, "cpu run failed\n"); fails++; continue;
        }
        // NCAL calibration forwards, then the scored one. The calibration passes return a
        // correct surface at a scale ~BOOTMARGIN loose; the frozen scale only takes effect
        // after them, so scoring the first forward would score the bootstrap.
        const char * nc = getenv("ROCKET_RK3576_NCAL");
        int ncal = (nc && *nc) ? atoi(nc) : 2;
        if (ncal < 1) ncal = 2;
        bool ok = true;
        for (int i = 0; i < ncal && ok; i++)
            ok = rk_run_mul_mat(rocket, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xcal, got, s.name);
        const rk_route_mark mk = rk_route_mark_take(rocket, "i8-76");
        ok = ok && rk_run_mul_mat(rocket, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xs, got, s.name);
        if (!ok) { fprintf(stderr, "rocket run failed\n"); fails++; continue; }
        if (!rk_route_check(rocket, "i8-76", mk, s.name)) fails++;

        const double cos = rk_cosine(golden, got);
        int wcol = -1;
        const double wcos = rk_worst_col_cosine(golden, got, s.M, s.N, &wcol);
        const bool pass = cos >= cos_min && wcos >= col_cos_min;
        printf("  %-12s K=%5d N=%5d M=%4d  cos=%.6f worst column %d cos=%.6f -> %s\n",
               s.name, s.K, s.N, s.M, cos, wcol, wcos, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "SOME RK3576 W8A8 SHAPES BELOW COS FLOOR" : "ALL RK3576 W8A8 SHAPES PASS");
    return fails ? 1 : 0;
}
