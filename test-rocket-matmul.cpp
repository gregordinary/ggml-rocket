// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-matmul.cpp — validate the rocket ggml backend's MUL_MAT against the
 * CPU backend on a standalone ggml graph (no whisper.cpp dependency).
 *
 * For each shape: build dst = ggml_mul_mat(W[K,N], X[K,M]) -> [N,M], run it on the
 * CPU backend and on the rocket backend with identical inputs, compare.
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rocket.h"

#include <vector>
#include <cstdio>
#include <cmath>

// run dst = mul_mat(W,X) on `backend`; W is F16 weights [K,N], X is F32 input [K,M].
// fills `out` with the [N*M] f32 result.
static bool run(ggml_backend_t backend, int K, int N, int M,
                const std::vector<float> & Wf, const std::vector<float> & Xf,
                std::vector<float> & out)
{
    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
                            /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);  // weights
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // input
    ggml_set_input(W); ggml_set_input(X);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);                     // -> [N, M]
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }

    // set inputs (convert W to f16)
    std::vector<ggml_fp16_t> W16((size_t)K*N);
    ggml_fp32_to_fp16_row(Wf.data(), W16.data(), (int64_t)K*N);
    ggml_backend_tensor_set(W, W16.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, Xf.data(), 0, ggml_nbytes(X));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out.resize((size_t)N*M);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Run ng matmuls D[i] = mul_mat(W[i], X) that all SHARE input X, on `backend`.
// W[i] is F16 [K, Ns[i]], X is F32 [K, M]; fills outs[i] with the [Ns[i]*M] f32
// result. On the rocket backend this exercises the shared-src1 fusion path
// (one combined-N matmul split back into per-member dsts); on CPU it's ng plain
// matmuls -> the reference. (Needs K>2048 for rocket to take the fused path.)
static bool run_group(ggml_backend_t backend, int K, const int * Ns, int ng, int M,
                      const std::vector<std::vector<float>> & Wf,
                      const std::vector<float> & Xf,
                      std::vector<std::vector<float>> & outs)
{
    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead()*(size_t)(2*ng+4) + ggml_graph_overhead(),
                            /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // shared input
    ggml_set_input(X);
    std::vector<ggml_tensor *> W(ng), D(ng);
    for (int i = 0; i < ng; i++) {
        W[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, Ns[i]);
        ggml_set_input(W[i]);
        D[i] = ggml_mul_mat(ctx, W[i], X);                          // -> [Ns[i], M]
        ggml_set_output(D[i]);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int i = 0; i < ng; i++) ggml_build_forward_expand(gf, D[i]);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "group alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }

    ggml_backend_tensor_set(X, Xf.data(), 0, ggml_nbytes(X));
    for (int i = 0; i < ng; i++) {
        std::vector<ggml_fp16_t> W16((size_t)K*Ns[i]);
        ggml_fp32_to_fp16_row(Wf[i].data(), W16.data(), (int64_t)K*Ns[i]);
        ggml_backend_tensor_set(W[i], W16.data(), 0, ggml_nbytes(W[i]));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "group graph_compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    outs.resize(ng);
    for (int i = 0; i < ng; i++) {
        outs[i].resize((size_t)Ns[i]*M);
        ggml_backend_tensor_get(D[i], outs[i].data(), 0, ggml_nbytes(D[i]));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Like run(), but the weight is stored QUANTIZED (Q8_0 / Q4_K / ...). Wf (f32 [K*N])
// is quantized into the tensor with ggml_quantize_chunk -- the exact payload a real
// GGUF holds -- so this exercises the rocket backend's dequant->fp16->NPU path
// (supports_op accepts the quant weight; the streaming/mt path dequantizes it). Returns
// the dequantized f32 weight (what the NPU effectively sees, modulo the final fp16
// rounding) in `Wdq` so the caller can form a tight fp16-tolerance golden.
static bool run_q(ggml_backend_t backend, int K, int N, int M, ggml_type wt,
                  const std::vector<float> & Wf, const std::vector<float> & Xf,
                  std::vector<float> & out, std::vector<float> & Wdq)
{
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, K, N);            // quantized weights
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M); // input
    ggml_set_input(W); ggml_set_input(X);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);                    // -> [N, M]
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "quant alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }

    // Quantize Wf [N rows x K] into the tensor's payload, then read the EFFECTIVE
    // (dequantized) weight back so the golden uses identical values.
    std::vector<char> q(ggml_nbytes(W));
    ggml_quantize_chunk(wt, Wf.data(), q.data(), 0, N, K, nullptr);
    Wdq.resize((size_t)N*K);
    const ggml_type_traits * tr = ggml_get_type_traits(wt);
    const size_t rbytes = ggml_row_size(wt, K);
    for (int n = 0; n < N; n++)
        tr->to_float(q.data() + (size_t)n*rbytes, Wdq.data() + (size_t)n*K, K);

    ggml_backend_tensor_set(W, q.data(),   0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, Xf.data(),  0, ggml_nbytes(X));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "quant graph_compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t)N*M);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    struct { int K, N, M; } shapes[] = {
        {  64,  32,   8 },     // tiny
        { 384, 256, 256 },     // multicore tile
        {3840,4096, 512 },     // Gemma FFN-ish
        // M not a multiple of 4 -> exercises rocket_pad_m (pad rows must be
        // computed-and-discarded; the first M rows stay bit-identical).
        {  64,  64,   7 },     // pad 7 -> 8
        { 384, 256, 130 },     // pad 130 -> 132
        {3840,4096, 250 },     // pad 250 -> 252, prefill-scale
    };

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    // No NPU -> the rocket backend won't init. SKIP (77) rather than FAIL, so this
    // gate skips cleanly off-device under CTest (SKIP_RETURN_CODE 77).
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    int fails = 0;
    for (auto s : shapes) {
        std::vector<float> Wf((size_t)s.K*s.N), Xf((size_t)s.K*s.M), oc, orr;
        for (size_t i = 0; i < Wf.size(); i++) Wf[i] = ((int)(i*7)%13-6)*0.05f;
        for (size_t i = 0; i < Xf.size(); i++) Xf[i] = ((int)(i*5)%11-5)*0.05f;

        bool ok = run(cpu, s.K, s.N, s.M, Wf, Xf, oc)
               && run(rocket, s.K, s.N, s.M, Wf, Xf, orr);
        if (!ok) { fails++; continue; }

        // An element is bad only if it fails BOTH tolerances at once (large abs alone
        // = fp16 rounding on a big value; large rel alone = a near-zero reference).
        // The old "max_abs<0.5 OR max_rel<0.05" on the global maxes could pass a wrong
        // result that stayed small in just one metric.
        float max_abs = 0, max_rel = 0; long nbad = 0;
        for (size_t i = 0; i < oc.size(); i++) {
            // A NaN/Inf result must FAIL: fabsf(NaN-x) is NaN and NaN>=t is false,
            // so a non-finite element would otherwise slip past both tolerances.
            if (!std::isfinite(orr[i])) { nbad++; continue; }
            float ad = fabsf(orr[i]-oc[i]);
            float rd = ad / (fabsf(oc[i]) + 1e-6f);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (ad >= 0.5f && rd >= 0.05f) nbad++;
        }
        bool pass = (nbad == 0);
        printf("K=%4d N=%4d M=%4d  max_abs=%.4f max_rel=%.4f nbad=%ld -> %s\n",
               s.K, s.N, s.M, max_abs, max_rel, nbad, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    // Row-wise activation scaling. One outlier token (row) with values
    // ~3000x the others must NOT crush the fp16 precision of the small rows. Under
    // the OLD global max-abs scaling, the small rows would scale by ~1/3000 and
    // round to near-zero in fp16 -> large rel error; per-row scaling keeps each
    // row in [-1,1] independently, so the small rows stay accurate.
    {
        const int K = 256, N = 64, M = 8;
        std::vector<float> Wf((size_t)K*N), Xf((size_t)K*M), oc, orr;
        for (size_t i = 0; i < Wf.size(); i++) Wf[i] = ((int)(i*7)%13-6)*0.05f;
        for (size_t i = 0; i < Xf.size(); i++) Xf[i] = ((int)(i*5)%11-5)*0.02f;   // small (~[-0.08,0.08])
        const int outlier_row = 3;
        for (int k = 0; k < K; k++) Xf[(size_t)outlier_row*K + k] = ((k%7)-3)*300.0f;  // |.|<=900

        bool ok = run(cpu, K, N, M, Wf, Xf, oc) && run(rocket, K, N, M, Wf, Xf, orr);
        if (!ok) { fprintf(stderr, "outlier test: backend run failed\n"); fails++; }
        else {
            // Check the SMALL (non-outlier) rows specifically.
            float small_max_rel = 0; int small_nonfinite = 0;
            for (int m = 0; m < M; m++) {
                if (m == outlier_row) continue;
                for (int n = 0; n < N; n++) {
                    size_t i = (size_t)m*N + n;
                    if (!std::isfinite(orr[i])) small_nonfinite++;
                    float rd = fabsf(orr[i]-oc[i]) / (fabsf(oc[i]) + 1e-6f);
                    if (rd > small_max_rel) small_max_rel = rd;
                }
            }
            // The outlier row may carry more abs error (large magnitude) but must be finite.
            int outlier_nonfinite = 0;
            for (int n = 0; n < N; n++)
                if (!std::isfinite(orr[(size_t)outlier_row*N + n])) outlier_nonfinite++;

            bool pass = small_max_rel < 0.05f && small_nonfinite == 0 && outlier_nonfinite == 0;
            printf("outlier-row scaling  small_rows max_rel=%.4f nonfinite(small/outlier)=%d/%d -> %s\n",
                   small_max_rel, small_nonfinite, outlier_nonfinite, pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    // Fused shared-src1 matmuls. Two patterns, both K>2048 so the rocket
    // backend takes the streaming + fusion path: gate/up (equal N) and Q/K/V
    // (GQA-style UNEQUAL N). Each member's output, sliced from the fused [M,sumN]
    // result, must match the CPU reference of running them separately.
    {
        const int K = 3840, M = 128;
        struct { int Ns[3]; int ng; const char * name; } cases[] = {
            { {4096, 4096,    0}, 2, "gate/up" },        // equal-N concat
            { {4096, 2048, 2048}, 3, "q/k/v(GQA)" },     // unequal-N concat
        };
        for (auto c : cases) {
            std::vector<std::vector<float>> Wf(c.ng);
            std::vector<float> Xf((size_t)K*M);
            for (size_t i = 0; i < Xf.size(); i++) Xf[i] = ((int)(i*5)%11-5)*0.05f;
            for (int g = 0; g < c.ng; g++) {
                Wf[g].resize((size_t)K*c.Ns[g]);
                for (size_t i = 0; i < Wf[g].size(); i++)   // distinct per-weight pattern
                    Wf[g][i] = ((int)((i + (size_t)g*131)*7)%13-6)*0.05f;
            }
            std::vector<std::vector<float>> oc, orr;
            bool ok = run_group(cpu,    K, c.Ns, c.ng, M, Wf, Xf, oc)
                   && run_group(rocket, K, c.Ns, c.ng, M, Wf, Xf, orr);
            if (!ok) { fprintf(stderr, "fused %s: backend run failed\n", c.name); fails++; continue; }
            float max_abs = 0, max_rel = 0; long nbad = 0;
            for (int g = 0; g < c.ng; g++)
                for (size_t i = 0; i < oc[g].size(); i++) {
                    if (!std::isfinite(orr[g][i])) { nbad++; continue; }  // NaN/Inf must FAIL
                    float ad = fabsf(orr[g][i]-oc[g][i]);
                    float rd = ad / (fabsf(oc[g][i]) + 1e-6f);
                    if (ad > max_abs) max_abs = ad;
                    if (rd > max_rel) max_rel = rd;
                    if (ad >= 0.5f && rd >= 0.05f) nbad++;   // bad only if BOTH fail
                }
            bool pass = (nbad == 0);
            printf("fused %-12s K=%d M=%d ng=%d  max_abs=%.4f max_rel=%.4f nbad=%ld -> %s\n",
                   c.name, K, M, c.ng, max_abs, max_rel, nbad, pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    // Quantized weights. A Q8_0 / Q4_K GGUF weight prefills on the NPU by
    // dequantizing to fp16 on the fly (no whole-model F16 copy). The golden is a pure
    // f32 matmul of the EFFECTIVE dequantized weight (run_q returns it) with the f32
    // input, so the residual is just fp16 rounding (weight + activation + accumulation)
    // -- the same regime as the F32-weight path. Q4_K needs K % 256 == 0.
    {
        struct { int K, N, M; ggml_type wt; const char * name; } qshapes[] = {
            {  256, 128,  64, GGML_TYPE_Q8_0, "Q8_0" },
            { 3840, 256, 128, GGML_TYPE_Q8_0, "Q8_0" },   // Gemma-scale K
            {  256, 128,  64, GGML_TYPE_Q4_K, "Q4_K" },
            { 3840, 256, 128, GGML_TYPE_Q4_K, "Q4_K" },
            {  512, 256, 128, GGML_TYPE_Q6_K, "Q6_K" },
        };
        for (auto s : qshapes) {
            std::vector<float> Wf((size_t)s.K*s.N), Xf((size_t)s.K*s.M), orr, Wdq;
            // Smooth-ish weights so quantization is well-behaved (random hash patterns
            // make Q4_K's per-block scales noisy and the test about quant error, not the
            // NPU path). Magnitudes ~[-0.5,0.5].
            for (size_t i = 0; i < Wf.size(); i++) Wf[i] = sinf((float)i*0.013f)*0.5f;
            for (size_t i = 0; i < Xf.size(); i++) Xf[i] = ((int)(i*5)%11-5)*0.05f;

            if (!run_q(rocket, s.K, s.N, s.M, s.wt, Wf, Xf, orr, Wdq)) { fails++; continue; }

            // golden[m*N+n] = sum_k Wdq[n,k]*Xf[m,k]  (matches ggml mul_mat [N,M] layout)
            std::vector<double> golden((size_t)s.N*s.M, 0.0);
            for (int m = 0; m < s.M; m++)
                for (int n = 0; n < s.N; n++) {
                    double acc = 0;
                    for (int k = 0; k < s.K; k++)
                        acc += (double)Wdq[(size_t)n*s.K+k] * (double)Xf[(size_t)m*s.K+k];
                    golden[(size_t)m*s.N+n] = acc;
                }
            float max_abs = 0, max_rel = 0; long nbad = 0;
            for (size_t i = 0; i < golden.size(); i++) {
                if (!std::isfinite(orr[i])) { nbad++; continue; }
                float ad = fabsf(orr[i]-(float)golden[i]);
                float rd = ad / (fabsf((float)golden[i]) + 1e-6f);
                if (ad > max_abs) max_abs = ad;
                if (rd > max_rel) max_rel = rd;
                if (ad >= 0.5f && rd >= 0.05f) nbad++;
            }
            bool pass = (nbad == 0);
            printf("quant %-4s K=%4d N=%4d M=%4d  max_abs=%.4f max_rel=%.4f nbad=%ld -> %s\n",
                   s.name, s.K, s.N, s.M, max_abs, max_rel, nbad, pass ? "PASS" : "FAIL");
            if (!pass) fails++;
        }
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf("%s\n", fails ? "SOME TESTS FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
