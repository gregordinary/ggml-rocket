// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-rocket-moe.cpp — validate the rocket backend's MUL_MAT_ID (MoE routed-expert
 * FFN) handler against the CPU backend on a standalone ggml graph.
 *
 * For each config: build dst = ggml_mul_mat_id(As[K,N,n_expert], B[K,ne11,n_tokens],
 * Ids[n_used,n_tokens]) -> [N, n_used, n_tokens], run it on the CPU backend and on
 * the rocket backend with identical inputs + routing, and compare. Running the graph
 * directly on each backend (not via the scheduler) drives the rocket MUL_MAT_ID
 * handler unconditionally, so this gate exercises the bucket/gather/GEMM/scatter path
 * regardless of the prefill placement gate (ROCKET_MOE_MIN_TOKENS).
 *
 * Covers: both b-forms (ne11==1, the gate/up input broadcast over slots; and
 * ne11==n_used, the down input with a distinct row per slot); balanced and imbalanced
 * routing (some experts get zero rows); an M_e not a multiple of 4 (pad rows); and
 * the F16 zero-copy weight path plus the MXFP4 (gpt-oss) and Q4_K (DeepSeek) dequant
 * paths. The tolerance mirrors test-rocket-matmul: an element fails only if it misses
 * BOTH abs and rel, and the aggregate cosine similarity must be >= 0.9999.
 */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rocket.h"

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

// One MoE config + its routing. Wf[e] is expert e's weight [N rows x K] (f32, to be
// converted / quantized into the tensor); Xf is B [n_tokens][ne11][K] f32 row-major;
// ids is [n_tokens][n_used] i32 row-major.
struct moe_case {
    int K, N, n_expert, ne11, n_used, n_tokens;
    ggml_type wt;               // GGML_TYPE_F16 / _MXFP4 / _Q4_K / ...
    const char * name;
};

// Build + run dst = mul_mat_id(As, B, Ids) on `backend`; fill out[N*n_used*n_tokens].
static bool run(ggml_backend_t backend, const moe_case & c,
                const std::vector<std::vector<float>> & Wf,
                const std::vector<float> & Xf,
                const std::vector<int32_t> & ids,
                std::vector<float> & out)
{
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * As  = ggml_new_tensor_3d(ctx, c.wt,           c.K, c.N,     c.n_expert);
    ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  c.K, c.ne11,  c.n_tokens);
    ggml_tensor * Ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  c.n_used, c.n_tokens);
    ggml_set_input(As); ggml_set_input(B); ggml_set_input(Ids);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, As, B, Ids);     // -> [N, n_used, n_tokens]
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }

    // As: quantize / convert each expert's [N,K] into its slice of the tensor payload.
    const size_t rbytes = ggml_row_size(c.wt, c.K);            // bytes per [K] weight row
    std::vector<char> aq((size_t)c.n_expert * c.N * rbytes);
    for (int e = 0; e < c.n_expert; e++) {
        char * dstp = aq.data() + (size_t)e * c.N * rbytes;
        if (c.wt == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(Wf[e].data(), (ggml_fp16_t *)dstp, (int64_t)c.N * c.K);
        } else {
            ggml_quantize_chunk(c.wt, Wf[e].data(), dstp, 0, c.N, c.K, nullptr);
        }
    }
    ggml_backend_tensor_set(As,  aq.data(),  0, ggml_nbytes(As));
    ggml_backend_tensor_set(B,   Xf.data(),  0, ggml_nbytes(B));
    ggml_backend_tensor_set(Ids, ids.data(), 0, ggml_nbytes(Ids));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t)c.N * c.n_used * c.n_tokens);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    // (K, N, n_expert, ne11, n_used, n_tokens, wt, name)
    moe_case cases[] = {
        {  256, 128,  8, 1, 2,  64, GGML_TYPE_F16,   "F16 gate/up (ne11=1)        " },
        {  256, 128,  8, 2, 2,  64, GGML_TYPE_F16,   "F16 down    (ne11=n_used)   " },
        {  256, 128,  8, 1, 2,  66, GGML_TYPE_F16,   "F16 pad-rows (n_tokens=66)  " },
        {  256, 128, 16, 1, 4,  96, GGML_TYPE_F16,   "F16 16 experts / 4 used     " },
        {  256, 128,  8, 1, 2,  64, GGML_TYPE_MXFP4, "MXFP4 gate/up (gpt-oss)     " },
        {  256, 128,  8, 2, 2,  64, GGML_TYPE_MXFP4, "MXFP4 down                  " },
        {  256, 128,  8, 1, 2,  64, GGML_TYPE_Q4_K,  "Q4_K gate/up (DeepSeek)     " },
    };

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    // No NPU -> the rocket backend won't init. SKIP (77) rather than FAIL under CTest.
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    int fails = 0;
    for (const auto & c : cases) {
        // Deterministic weights per expert, input, and routing.
        std::vector<std::vector<float>> Wf(c.n_expert, std::vector<float>((size_t)c.N * c.K));
        for (int e = 0; e < c.n_expert; e++)
            for (size_t i = 0; i < Wf[e].size(); i++)
                Wf[e][i] = ((int)((i + e*131)*7)%13 - 6) * 0.05f;
        std::vector<float> Xf((size_t)c.K * c.ne11 * c.n_tokens);
        for (size_t i = 0; i < Xf.size(); i++) Xf[i] = ((int)(i*5)%11 - 5) * 0.05f;

        // Routing: hash (token,slot) into [0,n_expert). Skewed so some experts stay
        // empty (imbalance) while others are hot -- exercises the zero-bucket skip and
        // the varying-M_e per-expert GEMM. Distinct experts within a token are not
        // required by the op (the CPU reference handles any ids), so no de-dup.
        std::vector<int32_t> ids((size_t)c.n_used * c.n_tokens);
        for (int t = 0; t < c.n_tokens; t++)
            for (int s = 0; s < c.n_used; s++) {
                int h = (t*3 + s*7 + t/5) % (c.n_expert - 1);   // biased away from the last expert
                ids[(size_t)t*c.n_used + s] = h;
            }

        std::vector<float> oc, orr;
        bool ok = run(cpu, c, Wf, Xf, ids, oc) && run(rocket, c, Wf, Xf, ids, orr);
        if (!ok) { fprintf(stderr, "%s: backend run failed\n", c.name); fails++; continue; }

        // Aggregate cosine + per-element abs/rel (an element is bad only if it misses
        // BOTH tolerances at once -- large abs alone = fp16 rounding of a big value;
        // large rel alone = a near-zero reference).
        double dot = 0, na = 0, nb = 0;
        float max_abs = 0, max_rel = 0; long nbad = 0;
        for (size_t i = 0; i < oc.size(); i++) {
            if (!std::isfinite(orr[i])) { nbad++; continue; }
            dot += (double)oc[i]*orr[i]; na += (double)oc[i]*oc[i]; nb += (double)orr[i]*orr[i];
            float ad = fabsf(orr[i]-oc[i]);
            float rd = ad / (fabsf(oc[i]) + 1e-6f);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (ad >= 0.5f && rd >= 0.05f) nbad++;
        }
        double cos = (na > 0 && nb > 0) ? dot / (sqrt(na)*sqrt(nb)) : 0.0;
        bool pass = (nbad == 0) && (cos >= 0.9999);
        printf("%s cos=%.6f max_abs=%.4f max_rel=%.4f nbad=%ld -> %s\n",
               c.name, cos, max_abs, max_rel, nbad, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
