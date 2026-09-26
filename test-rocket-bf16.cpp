// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-bf16.cpp — regression guard for native BF16 weights.
 *
 * By default (ROCKET_BF16 unset) a BF16 weight is decoded to fp16 and offloaded on
 * the shared fp16 streaming route, exactly like an F32 or quantized weight. This
 * gate pins that default route down:
 *
 *   1. supports_op accepts a BF16 matmul for both plain 2D and batched (ne[2]>1)
 *      src1 — the fp16 streaming/mt path decodes bf16 for either.
 *
 *   2. A BF16 matmul run on the rocket backend matches the CPU reference within fp16
 *      tolerance, for both 2D and batched src1.
 *
 * ROCKET_BF16=1 instead selects the dedicated fp32-output bf16 datapath (token-
 * identical, fp32 accumulate range); its matmul is gated by the rocket-userspace
 * bf16 CTest, not here.
 *
 * ON THE RK3576 THERE IS NO SUCH ROUTE. Both bf16 datapaths are built on the fp16
 * matmul, whose RK3588 geometry-register encoding that part does not run, so supports_op
 * declines a bf16 weight there and the scheduler leaves it with the CPU backend. The
 * arithmetic below still has to agree — it just agrees via that fallback rather than on
 * the NPU — so case 2 is asserted identically on both parts and only the placement
 * expectation in case 1 forks.
 *
 * Case 2 also reads the backend's route counters, because the CPU fallback computes the
 * golden answer's own arithmetic and the numbers alone cannot say where it ran. On the
 * RK3588 an NPU route must finish the op with no fallback; on the RK3576 the fallback must
 * be what computed it.
 */
#include "ggml-cpu.h"
#include "ggml-rocket.h"
#include "test-common.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Query supports_op for a BF16 mul_mat with the given src1 batch B (no data needed).
static bool bf16_supported(ggml_backend_dev_t dev, int K, int N, int M, int B) {
    ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, N);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, B);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);
    bool ok = ggml_backend_dev_supports_op(dev, dst);
    ggml_free(ctx);
    return ok;
}

int main() {
    setenv("ROCKET_BF16", "0", 1);   // default route: decode bf16 -> fp16 streaming

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    if (!rk_device_opens()) return 77;
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    int fails = 0;

    // 1. supports_op placement: K%32, N%16, K>=64, N>=64, M>=min_m all satisfied so
    //    only the src1-batch dimension is in question. On the rk3588 BF16 is accepted
    //    for both plain 2D and batched src1 (the fp16 route decodes bf16 for either);
    //    on the rk3576 there is no fp16 route to decode onto, so both are declined.
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(rocket);
        const bool rk76 = rk_is_rk3576();
        const int K = 256, N = 128, M = 256;
        bool sup2d  = bf16_supported(dev, K, N, M, 1);
        bool supbat = bf16_supported(dev, K, N, M, 4);
        // Accepted for both plain 2D and batched src1 on the rk3588; declined for both on
        // the rk3576. Asserting each of the two separately keeps a half-answer (one shape
        // accepted, the other not) a failure on either part rather than an average.
        bool pass = rk76 ? (!sup2d && !supbat) : (sup2d && supbat);
        printf("supports_op bf16  2D=%d batched=%d (%s) -> %s\n",
               sup2d, supbat, rk76 ? "rk3576: expect declined" : "rk3588: expect accepted",
               pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // 2. BF16 matmul on the rocket backend matches the CPU reference within fp16
    //    tolerance, for 2D (B=1) and batched (B=3) src1. The weights are decoded
    //    bf16 -> fp16, so the bar is fp16-level, not the looser bf16 accumulate bar.
    for (int B : {1, 3}) {
        const int K = 256, N = 128, M = 64;
        std::vector<float> Wf((size_t)K*N), Xf((size_t)K*M*B), oc, orr;
        // Period-free, and sized so the outputs sit near 10: the element bar below ignores
        // an error under 0.5, so outputs below it would let a zeroed element through.
        rk_fill(Wf, 11u + (uint64_t)B, 1.0f);
        rk_fill(Xf, 22u + (uint64_t)B, rk_amp_for(K, 1.0f));

        bool ok = rk_run_mul_mat(cpu, GGML_TYPE_BF16, K, N, M, B, Wf, Xf, oc);
        const rk_route_mark mk = rk_route_mark_take(rocket, nullptr);
        ok = ok && rk_run_mul_mat(rocket, GGML_TYPE_BF16, K, N, M, B, Wf, Xf, orr);
        if (!ok) { fprintf(stderr, "bf16 B=%d: backend run failed\n", B); fails++; continue; }
        if (rk_is_rk3576()) {
            // No bf16 route on this part: the op reaches the CPU fallback by design, and the
            // count says it did rather than leaving the numbers to imply it.
            const long fb = ggml_backend_rocket_cpu_fallbacks(rocket) - mk.fallbacks;
            printf("  rk3576: %ld CPU fallback(s), as a part with no bf16 route computes -> %s\n",
                   fb, fb > 0 ? "PASS" : "FAIL");
            if (fb <= 0) fails++;
        } else if (!rk_route_check(rocket, nullptr, mk, "on the NPU")) {
            fails++;
        }

        float max_abs = 0, max_rel = 0; long nbad = 0;
        for (size_t i = 0; i < oc.size(); i++) {
            if (!std::isfinite(orr[i])) { nbad++; continue; }
            float ad = fabsf(orr[i]-oc[i]);
            float rd = ad / (fabsf(oc[i]) + 1e-6f);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (ad >= 0.5f && rd >= 0.05f) nbad++;
        }
        bool pass = (nbad == 0);
        printf("bf16 matmul K=%d N=%d M=%d B=%d  max_abs=%.4f max_rel=%.4f nbad=%ld -> %s\n",
               K, N, M, B, max_abs, max_rel, nbad, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
