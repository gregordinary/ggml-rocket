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
 * Covers BOTH weight routes the handler chooses between:
 *
 *   fp16 (an F16 expert): both b-forms (ne11==1, the gate/up input broadcast over slots;
 *   and ne11==n_used, the down input with a distinct row per slot); balanced and imbalanced
 *   routing (some experts get zero rows); an M_e not a multiple of 4 (pad rows).
 *
 *   NATIVE-QUANT (a GGUF-quantized expert): the expert's quant blocks are ingested once
 *   into int8 codes resident on the NPU, and each call quantizes only the activation, per
 *   (row, K-group). Cases span nG = 1 / 2 / 4 K-groups (K picks the group, see
 *   rocket_moe_pick_group), the MXFP4 e_ref block merge (including a WIDE-DYNAMIC-RANGE
 *   weight, which is what forces the merge's rounding right-shift rather than its exact
 *   left-shift), and the dequantize-once-requantize route for Q4_K and Q8_0.
 *
 * Two properties that are not visible in the output values get their own gates:
 *
 *   THE NATIVE ROUTE ACTUALLY RAN. It declines silently (to a correct fp16 result) on a
 *   weight with no stable name, an unsupported shape, or an exhausted budget -- so a gate
 *   that only checked the numbers would pass while testing nothing. Assert the resident
 *   expert count via ggml_backend_rocket_moe_stats.
 *
 *   ONE INGEST SERVES EVERY M. The resident layout is M-independent, which is what lets a
 *   single ingest serve the ragged per-micro-batch row count a router produces. Drive one
 *   named weight at several n_tokens and assert the expert count does not grow.
 *
 * The tolerance mirrors test-rocket-matmul: an element fails only if it misses BOTH abs and
 * rel, and the aggregate cosine must clear the case's floor. The int8 route's floor is
 * lower than the fp16 route's by construction -- it re-quantizes the activation to 8 bits
 * on the K-group grid -- and each floor sits ~2-3x under the measured cosine, so the gate
 * catches a broken route rather than tracking quantization noise. Measured on HW at
 * 600 MHz: MXFP4 0.99986 (nG=1) / 0.99985 (nG=2) / 0.99956 (nG=4), the wide-range merge
 * 0.99934, Q4_K 0.99993, Q8_0 0.99987 -- all far above the 0.98 that already proved
 * token-identical for the int4+Hadamard path.
 *
 * These inputs are synthetic. This gate pins the ROUTE (that it runs, that it is
 * bit-faithful to the reference within the int8 envelope, that one ingest serves every M);
 * the model-level accuracy gate is a greedy-match / differential-PPL run on a real model.
 */
#include "ggml-cpu.h"
#include "ggml-rocket.h"
#include "test-common.h"

extern "C" {
#include "rocket_npu.h"
}

#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

// How an expert weight's magnitudes are laid out along K. The rocket backend's MXFP4
// ingest merges the native 32-element blocks of a K-group onto one exponent, and the merge
// is an EXACT integer shift only while the group's exponent spread is <= 3 octaves. WIDE
// deliberately blows past that so the merge has to take its rounding right-shift branch --
// the path a real model reaches in its later layers (spread up to 9-10 octaves) and the one
// whose error placement the e_ref rule exists to control.
enum wrange { FLAT, WIDE };

// One MoE config + its routing. Wf[e] is expert e's weight [N rows x K] (f32, to be
// converted / quantized into the tensor); Xf is B [n_tokens][ne11][K] f32 row-major;
// ids is [n_tokens][n_used] i32 row-major.
struct moe_case {
    int K, N, n_expert, ne11, n_used, n_tokens;
    ggml_type wt;               // GGML_TYPE_F16 / _MXFP4 / _Q4_K / _Q8_0 / ...
    wrange    range;            // weight dynamic range along K
    double    cos_min;          // aggregate cosine floor for this route
    bool      native;           // does the handler's native-quant route apply here?
    const char * name;
};

// Deterministic expert weights. `range == WIDE` scales each 32-element K-block by a power
// of two spanning ~9 octaves, so a merged K-group's blocks cannot share one exponent
// exactly. The pattern is a fixed function of (e, block), not random, so CPU and NPU see
// byte-identical tensors.
static void fill_weights(std::vector<float> & w, int e, int K, int N, wrange range) {
    for (size_t i = 0; i < w.size(); i++) {
        float v = ((int)((i + e * 131) * 7) % 13 - 6) * 0.05f;
        if (range == WIDE) {
            const int blk = (int)((i % (size_t)K) / 32);       // K-block index within the row
            v *= ldexpf(1.0f, -(blk % 10));                    // 1 .. 1/512
        }
        w[i] = v;
    }
    (void)N;
}

// Build + run dst = mul_mat_id(As, B, Ids) on `backend`; fill out[N*n_used*n_tokens].
// `wname` is the expert-stack tensor name. It is LOAD-BEARING, not cosmetic: the backend's
// resident-expert cache is keyed on (weight name, expert index), and it REJECTS ggml's
// auto-assigned "leaf_%d" names (those indices are a graph position, not a weight identity,
// so caching on them would serve one weight's tiles for another's matmul). An unnamed
// tensor therefore silently declines the native route -- so every caller here names it, and
// names it UNIQUELY, since one backend instance serves every case below.
static bool run(ggml_backend_t backend, const moe_case & c, const char * wname,
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
    ggml_set_name(As, wname);
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

// Routing: hash (token,slot) into [0,n_expert). Skewed so some experts stay empty
// (imbalance) while others are hot -- exercises the zero-bucket skip and the varying-M_e
// per-expert GEMM (and, on the native route, the M bucketing). Distinct experts within a
// token are not required by the op (the CPU reference handles any ids), so no de-dup.
static std::vector<int32_t> make_ids(const moe_case & c) {
    std::vector<int32_t> ids((size_t)c.n_used * c.n_tokens);
    for (int t = 0; t < c.n_tokens; t++)
        for (int s = 0; s < c.n_used; s++)
            ids[(size_t)t*c.n_used + s] = (t*3 + s*7 + t/5) % (c.n_expert - 1);  // last expert stays empty
    return ids;
}

// Input activations, with OUTLIER CHANNELS. ~1% of the K channels carry 20x the magnitude
// of the rest -- the well-documented LLM activation profile, and the one thing that makes
// int8 activation quantization hard: a per-row scale is pinned by the fattest channel, and
// every other channel then quantizes into a fraction of the int8 range. It is the reason
// the int4 path needs a Hadamard rotation at all.
//
// The native-quant expert path answers it with granularity instead of rotation: the
// activation is quantized per (row, K-GROUP), so an outlier only degrades the group it
// sits in, not the whole row. That is the property this gate exists to pin -- a flat input
// would pass just as happily against a per-ROW scale and prove nothing.
static std::vector<float> make_input(const moe_case & c) {
    std::vector<float> Xf((size_t)c.K * c.ne11 * c.n_tokens);
    for (size_t i = 0; i < Xf.size(); i++) {
        float v = ((int)(i*5)%11 - 5) * 0.05f;
        const size_t k = i % (size_t)c.K;                 // channel index within the row
        if (k % 97 == 13) v *= 20.0f;                     // ~1% of channels, 20x
        Xf[i] = v;
    }
    return Xf;
}


int main() {
    // K picks the quant group (the largest divisor of K that is a multiple of 32 and that
    // the CBUF can hold whole), so K is how these cases span the K-group count:
    //   K=256  -> group 256, nG=1        K=1152 -> group 576, nG=2 (18 MXFP4 blocks merged,
    //   K=1024 -> group 512, nG=2                                   the gpt-oss operating point)
    //   K=2048 -> group 512, nG=4
    const moe_case cases[] = {
        // --- fp16 route (an F16 expert has no per-micro-batch dequant to delete) ---
        {  256, 128,  8, 1, 2,  64, GGML_TYPE_F16,   FLAT, 0.9999, false, "F16   gate/up (ne11=1)       " },
        {  256, 128,  8, 2, 2,  64, GGML_TYPE_F16,   FLAT, 0.9999, false, "F16   down    (ne11=n_used)  " },
        {  256, 128,  8, 1, 2,  66, GGML_TYPE_F16,   FLAT, 0.9999, false, "F16   pad-rows (n_tokens=66) " },
        {  256, 128, 16, 1, 4,  96, GGML_TYPE_F16,   FLAT, 0.9999, false, "F16   16 experts / 4 used    " },
        // --- native-quant route: MXFP4 (gpt-oss), the e_ref block merge ---
        {  256, 128,  8, 1, 2,  64, GGML_TYPE_MXFP4, FLAT, 0.9990, true,  "MXFP4 nG=1  gate/up          " },
        {  256, 128,  8, 2, 2,  64, GGML_TYPE_MXFP4, FLAT, 0.9990, true,  "MXFP4 nG=1  down             " },
        { 1152, 128,  8, 1, 2,  64, GGML_TYPE_MXFP4, FLAT, 0.9990, true,  "MXFP4 nG=2  (group=576)      " },
        { 2048, 128,  8, 1, 2,  66, GGML_TYPE_MXFP4, FLAT, 0.9990, true,  "MXFP4 nG=4  + pad rows       " },
        // The merge's rounding right-shift: a 9-octave spread inside every merged group, so
        // no single exponent represents it exactly. e_ref puts the error on the group's
        // SMALLEST weights (which contribute least to the dot product), which is why the
        // cosine survives at all -- clamping to e_min would clip the LARGEST instead.
        { 1152, 128,  8, 1, 2,  64, GGML_TYPE_MXFP4, WIDE, 0.9985, true,  "MXFP4 nG=2  wide-range merge " },
        // --- native-quant route: dequantize-once-requantize (no merge lever) ---
        { 1024, 128,  8, 1, 2,  64, GGML_TYPE_Q4_K,  FLAT, 0.9990, true,  "Q4_K  nG=2  (DeepSeek)       " },
        { 1152, 128,  8, 2, 2,  64, GGML_TYPE_Q8_0,  FLAT, 0.9990, true,  "Q8_0  nG=2  down             " },
    };

    ggml_backend_t cpu    = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "cpu backend init failed\n"); return 1; }
    // Probe the accel node BEFORE the backend, and SKIP (77) if it will not open.
    // ggml_backend_rocket_init() opens nothing -- every fd is lazy -- so it succeeds on a
    // machine with no usable NPU and the matmuls then degrade to the CPU fallback. The
    // other gates read that as a pass (they only compare against the CPU), but this one
    // additionally asserts that experts were INGESTED onto the device, so a missing or
    // unreadable /dev/accel would report FAIL for what is an environment, not a defect.
    {
        const int probe = rocket_open();
        if (probe < 0) {
            fprintf(stderr, "cannot open the accel device (absent, or no permission -- "
                            "run with sudo -E) -> SKIP\n");
            return 77;
        }
        rocket_close(probe);
    }
    ggml_backend_t rocket = ggml_backend_rocket_init();
    if (!rocket) { fprintf(stderr, "rocket backend unavailable (no NPU?) -> SKIP\n"); return 77; }

    int fails = 0;
    int n_native_cases = 0;

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        const moe_case & c = cases[ci];
        std::vector<std::vector<float>> Wf(c.n_expert, std::vector<float>((size_t)c.N * c.K));
        for (int e = 0; e < c.n_expert; e++) fill_weights(Wf[e], e, c.K, c.N, c.range);
        const std::vector<float>   Xf  = make_input(c);
        const std::vector<int32_t> ids = make_ids(c);

        // A UNIQUE name per case: one rocket backend serves them all, and its resident
        // expert cache is name-keyed -- two cases sharing a name and a shape would have the
        // second silently compute against the first's ingested weights.
        char wname[64];
        snprintf(wname, sizeof(wname), "blk.%zu.ffn_gate_exps.weight", ci);

        long res_before = 0;
        ggml_backend_rocket_moe_stats(rocket, &res_before, nullptr);

        std::vector<float> oc, orr;
        if (!run(cpu, c, wname, Wf, Xf, ids, oc) || !run(rocket, c, wname, Wf, Xf, ids, orr)) {
            fprintf(stderr, "%s: backend run failed\n", c.name); fails++; continue;
        }

        float max_abs, max_rel; long nbad;
        const double cos = rk_compare(oc, orr, &max_abs, &max_rel, &nbad);

        // Did the native route actually run? It declines to a CORRECT fp16 result, so the
        // numbers alone cannot tell -- without this the quant cases would pass vacuously.
        long res_after = 0;
        ggml_backend_rocket_moe_stats(rocket, &res_after, nullptr);
        const long ingested = res_after - res_before;
        const bool route_ok = c.native ? (ingested > 0) : (ingested == 0);
        if (c.native) n_native_cases++;

        const bool pass = (nbad == 0) && (cos >= c.cos_min) && route_ok;
        printf("%s cos=%.6f max_abs=%.4f max_rel=%.4f nbad=%3ld ingested=%2ld -> %s%s\n",
               c.name, cos, max_abs, max_rel, nbad, ingested, pass ? "PASS" : "FAIL",
               route_ok ? "" : (c.native ? "  (native route did NOT run)"
                                         : "  (fp16 case took the native route)"));
        if (!pass) fails++;
    }

    // --- cross-M: ONE ingested expert weight, several micro-batch sizes ---
    //
    // The resident int8 expert layout is M-INDEPENDENT (the driver plans it at a canonical
    // tile M), which is the whole reason one ingest can serve the ragged, per-expert,
    // per-micro-batch row count a router hands out. Drive the SAME named weight at several
    // n_tokens and assert two things: every M still matches the CPU, and the expert count
    // does NOT grow after the first pass. A re-ingest per M would show up as n_expert more
    // experts each round -- and would mean the handler was paying the cost this whole design
    // exists to remove.
    //
    // The row counts are deliberately awkward: n_tokens*n_used/n_expert lands each expert on
    // a row count that is not a multiple of 4, which the driver REJECTS (an unaligned M does
    // not tile badly, it miscomputes), so this also gates the handler's M bucketing.
    {
        const int n_tokens_seq[] = { 64, 130, 96, 258 };
        moe_case c = { 1152, 128, 8, 1, 2, 0, GGML_TYPE_MXFP4, FLAT, 0.9990, true,
                       "MXFP4 cross-M                " };
        std::vector<std::vector<float>> Wf(c.n_expert, std::vector<float>((size_t)c.N * c.K));
        for (int e = 0; e < c.n_expert; e++) fill_weights(Wf[e], e, c.K, c.N, c.range);

        long res_before = 0;
        ggml_backend_rocket_moe_stats(rocket, &res_before, nullptr);
        long res_last = 0;
        bool ok = true;
        for (size_t i = 0; i < sizeof(n_tokens_seq)/sizeof(n_tokens_seq[0]); i++) {
            c.n_tokens = n_tokens_seq[i];
            const std::vector<float>   Xf  = make_input(c);
            const std::vector<int32_t> ids = make_ids(c);
            std::vector<float> oc, orr;
            if (!run(cpu, c, "blk.99.ffn_up_exps.weight", Wf, Xf, ids, oc) ||
                !run(rocket, c, "blk.99.ffn_up_exps.weight", Wf, Xf, ids, orr)) {
                fprintf(stderr, "cross-M: backend run failed at n_tokens=%d\n", c.n_tokens);
                ok = false; break;
            }
            float max_abs, max_rel; long nbad;
            const double cos = rk_compare(oc, orr, &max_abs, &max_rel, &nbad);
            if (nbad != 0 || cos < c.cos_min) {
                printf("%s n_tokens=%3d cos=%.6f nbad=%ld -> FAIL\n", c.name, c.n_tokens, cos, nbad);
                ok = false;
            }
            ggml_backend_rocket_moe_stats(rocket, &res_last, nullptr);
        }
        // At most ONE ingest per expert, not one per (expert, M): a re-pack per micro-batch
        // size would show up here as up to 4x n_expert.
        const long ingested = res_last - res_before;
        if (ingested <= 0 || ingested > c.n_expert) {
            printf("%s ingested %ld experts over 4 micro-batch sizes (expected 1..%d) -> FAIL "
                   "(the resident layout is not M-independent)\n", c.name, ingested, c.n_expert);
            ok = false;
        }
        printf("%s 4 micro-batch sizes (n_tokens 64/130/96/258), %ld experts ingested -> %s\n",
               c.name, ingested, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
        n_native_cases++;
    }

    // A run in which the native route never engaged would report ALL PASS while testing only
    // the fp16 path. Fail loudly instead.
    long n_res = 0, n_str = 0;
    ggml_backend_rocket_moe_stats(rocket, &n_res, &n_str);
    printf("\nnative-quant experts: %ld resident, %ld streamed (over %d native cases)\n",
           n_res, n_str, n_native_cases);
    if (n_res == 0) {
        fprintf(stderr, "FAIL: no expert was ever ingested -- the native-quant route never ran\n");
        fails++;
    }

    ggml_backend_free(rocket);
    ggml_backend_free(cpu);
    printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
