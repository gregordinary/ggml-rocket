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
 * THE CALIBRATION IS STATEFUL, so each shape is run TWICE and only the second run is
 * scored. The first ROCKET_RK3576_NCAL forwards of a weight are calibration passes whose
 * output is correct at a scale ~BOOTMARGIN loose; the frozen scale only takes effect
 * afterwards. Scoring the first forward would score the bootstrap, not the route.
 *
 * SKIPS (77) off-device and on any part that is not an RK3576, so it is safe in the
 * default ctest run on either board.
 *
 * Optional env: ROCKET_RK3576_COS_MIN overrides the cosine floor.
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

    const char * cm = getenv("ROCKET_RK3576_COS_MIN");
    const double cos_min = cm ? atof(cm) : 0.995;

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    // Qwen2.5-1.5B's own shapes, which is what the route's accuracy numbers were taken on.
    // K=1536 is the projection K the part takes whole; K=8960 is ffn_down, which needs the
    // host K-split and is therefore DECLINED by default -- included so a run with
    // ROCKET_RK3576_KSPLIT=1 scores the split path and a default run scores the decline
    // (which routes to the CPU and so scores an exact 1.0).
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
    printf("RK3576 W8A8 vs CPU backend (cos_min=%.4f, KSPLIT=%s)\n", cos_min,
           getenv("ROCKET_RK3576_KSPLIT") ? getenv("ROCKET_RK3576_KSPLIT") : "0");
    for (auto s : shapes) {
        std::vector<float> Wf((size_t)s.K*s.N), Xf((size_t)s.K*s.M), golden, got;
        for (auto & w : Wf) w = wdist(rng);
        for (auto & x : Xf) x = xdist(rng);
        // Per-channel activation outliers (a few feature channels ~30x): the LLM regime
        // this route's mandatory rotation exists to tame, and the one that separates a
        // rotated run from an unrotated one.
        for (int c = 0; c < s.K; c += 512)
            for (int m = 0; m < s.M; m++) Xf[(size_t)m*s.K + c] *= 30.0f;

        // A distinct weight NAME per shape: the W8A8 weight cache -- and the calibration
        // state held beside it -- is keyed on the name, so one name across four shapes
        // would evict and re-calibrate rather than give each shape its own entry.
        if (!rk_run_mul_mat(cpu, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xf, golden, s.name)) {
            fprintf(stderr, "cpu run failed\n"); fails++; continue;
        }
        // NCAL calibration forwards, then the scored one. The calibration passes return a
        // correct surface at a scale ~BOOTMARGIN loose; the frozen scale only takes effect
        // after them, so scoring the first forward would score the bootstrap.
        const char * nc = getenv("ROCKET_RK3576_NCAL");
        int ncal = (nc && *nc) ? atoi(nc) : 2;
        if (ncal < 1) ncal = 2;
        bool ok = true;
        for (int i = 0; i <= ncal && ok; i++)
            ok = rk_run_mul_mat(rocket, GGML_TYPE_F16, s.K, s.N, s.M, 1, Wf, Xf, got, s.name);
        if (!ok) { fprintf(stderr, "rocket run failed\n"); fails++; continue; }

        const double cos = rk_cosine(golden, got);
        const bool pass = cos >= cos_min;
        printf("  %-12s K=%5d N=%5d M=%4d  cos=%.6f -> %s\n",
               s.name, s.K, s.N, s.M, cos, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "SOME RK3576 W8A8 SHAPES BELOW COS FLOOR" : "ALL RK3576 W8A8 SHAPES PASS");
    return fails ? 1 : 0;
}
