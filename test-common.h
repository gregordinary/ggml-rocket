// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
//
// Scaffolding shared by the standalone gates.
//
// Every gate here asks the same question in the same shape -- build a one-matmul graph,
// run it on a backend, read the result back, compare against the CPU backend's -- and the
// ggml sequence for that (init -> new_tensor -> mul_mat -> alloc_ctx_tensors -> tensor_set
// -> graph_compute -> tensor_get -> free) had been written out once per gate. This is that
// sequence, once. A gate that needs a different graph (a fused shared-src1 group, a
// MUL_MAT_ID, a supports_op probe with no data) still writes its own; what lives here is
// what more than one of them actually shares.
#ifndef GGML_ROCKET_TEST_COMMON_H
#define GGML_ROCKET_TEST_COMMON_H

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "rocket_hw_profile.h"   // rocket_hw_current(): which part the driver selected
#include "rocket_npu.h"          // rocket_open(): whether the accel node opens at all
#include "ggml-rocket.h"        // the route and fallback counters

#include <cstdint>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// Which part the backend is running on. The two do not accept the same ops -- the RK3576's
// only matmul route is the W8A8 one, and the bf16 / dequant-to-fp16 / fp16-attention routes
// exist solely on the RK3588 -- so a placement expectation has to say which part it is for.
// A gate that asserts one part's accept-set on the other reports a failure the library is
// right about, which is what this exists to stop.
static inline bool rk_is_rk3576(void) {
    const struct rocket_hw_profile * hw = rocket_hw_current();
    return hw && hw->name && strcmp(hw->name, "rk3576") == 0;
}

// dst = mul_mat(W, X) on `backend`. W is [K,N] of `wtype` (F16 / BF16 / F32), X is F32
// [K,M,B] (B==1 gives a plain 2D src1, B>1 a batched one); `out` receives the [N*M*B] f32
// result. `Wf` is always f32 and is converted to `wtype` here.
//
// `wname`, when given, names the weight tensor. That is not cosmetic on this backend: the
// resident-weight caches -- and, on the RK3576, the calibration state beside them -- are
// keyed on the weight NAME, so an unnamed weight is re-packed (and re-calibrated) on every
// call and a gate that meant to exercise the cached path would not.
static inline bool rk_run_mul_mat(ggml_backend_t backend, ggml_type wtype,
                                  int K, int N, int M, int B,
                                  const std::vector<float> & Wf, const std::vector<float> & Xf,
                                  std::vector<float> & out, const char * wname = nullptr)
{
    ggml_init_params ip = { /*.mem_size=*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
                            /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, B);
    ggml_set_input(W); ggml_set_input(X);
    if (wname) ggml_set_name(W, wname);
    ggml_tensor * dst = ggml_mul_mat(ctx, W, X);        // -> [N, M, B]
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); ggml_free(ctx); return false; }

    const size_t nw = (size_t)K * N;
    std::vector<ggml_fp16_t> W16;
    std::vector<ggml_bf16_t> Wb;
    switch (wtype) {
        case GGML_TYPE_F16:
            W16.resize(nw); ggml_fp32_to_fp16_row(Wf.data(), W16.data(), (int64_t)nw);
            ggml_backend_tensor_set(W, W16.data(), 0, ggml_nbytes(W));
            break;
        case GGML_TYPE_BF16:
            Wb.resize(nw); ggml_fp32_to_bf16_row(Wf.data(), Wb.data(), (int64_t)nw);
            ggml_backend_tensor_set(W, Wb.data(), 0, ggml_nbytes(W));
            break;
        case GGML_TYPE_F32:
            ggml_backend_tensor_set(W, Wf.data(), 0, ggml_nbytes(W));
            break;
        default:
            fprintf(stderr, "rk_run_mul_mat: unsupported weight type %d\n", (int)wtype);
            ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    ggml_backend_tensor_set(X, Xf.data(), 0, ggml_nbytes(X));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n");
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t)N*M*B);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Aggregate cosine + per-element abs/rel over a whole result tile. An element counts as
// bad only if it misses BOTH tolerances at once -- a large absolute error alone is the
// rounding of a big value, a large relative error alone is a near-zero reference. Any
// non-finite element is bad outright. `max_abs` / `max_rel` / `nbad` may be null.
//
// The cosine over the flattened tile is the summary because it weights every output
// element equally, which is what the next layer sees. It is also the bar the integer
// routes are read against: they are not bit-faithful to fp32 and never claim to be.
static inline double rk_compare(const std::vector<float> & ref, const std::vector<float> & got,
                                float * max_abs = nullptr, float * max_rel = nullptr,
                                long * nbad = nullptr)
{
    double dot = 0, na = 0, nb = 0;
    float ma = 0, mr = 0; long bad = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i])) { bad++; continue; }
        dot += (double)ref[i]*got[i]; na += (double)ref[i]*ref[i]; nb += (double)got[i]*got[i];
        const float ad = fabsf(got[i]-ref[i]);
        const float rd = ad / (fabsf(ref[i]) + 1e-6f);
        if (ad > ma) ma = ad;
        if (rd > mr) mr = rd;
        if (ad >= 0.5f && rd >= 0.05f) bad++;
    }
    if (max_abs) *max_abs = ma;
    if (max_rel) *max_rel = mr;
    if (nbad)    *nbad    = bad;
    return (na > 0 && nb > 0) ? dot / (sqrt(na)*sqrt(nb)) : 0.0;
}

// The cosine alone, with a non-finite element failing the comparison outright rather
// than being counted. -2.0 is unreachable for a real cosine, so a caller comparing
// against a floor rejects it without a separate finiteness check.
static inline double rk_cosine(const std::vector<float> & a, const std::vector<float> & b) {
    for (size_t i = 0; i < a.size(); i++)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return -2.0;
    return rk_compare(a, b);
}

// ---- is the device there, and did it compute? -------------------------------------------
//
// ggml_backend_rocket_init() opens nothing -- every fd is lazy -- so it succeeds on a machine
// with no usable NPU, and each matmul then degrades to the backend's CPU fallback. That
// fallback computes in fp64, as the CPU backend computes a gate's golden answer, so the
// numbers alone pass on a device that computed nothing. Two checks close it: probe the node
// before the backend, and assert after a run which route computed.

// True when the accel node opens. Otherwise says why, and the gate SKIPs (77).
static inline bool rk_device_opens(void) {
    const int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "cannot open the accel device (absent, or no permission -- run with "
                        "sudo -E) -> SKIP\n");
        return false;
    }
    rocket_close(fd);
    return true;
}

// The backend's counters at one point, so a gate can assert over the ops it then ran.
struct rk_route_mark { long ops; long fallbacks; };

static inline rk_route_mark rk_route_mark_take(ggml_backend_t b, const char * route) {
    return { ggml_backend_rocket_route_ops(b, route), ggml_backend_rocket_cpu_fallbacks(b) };
}

// Since `m`: at least one op finished on `route` (NULL: any route) and nothing fell back to
// the CPU. Prints both counts either way, so a failing run says which half failed.
static inline bool rk_route_check(ggml_backend_t b, const char * route, const rk_route_mark & m,
                                  const char * what) {
    const long ops = ggml_backend_rocket_route_ops(b, route) - m.ops;
    const long fb  = ggml_backend_rocket_cpu_fallbacks(b) - m.fallbacks;
    const bool ok  = ops > 0 && fb == 0;
    printf("  %s: %ld op(s) on route %s, %ld CPU fallback(s) -> %s\n", what, ops,
           route ? route : "(any)", fb, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- fills and comparisons ----------------------------------------------------------------

// A period-free fill, uniform in [-amp, amp]: splitmix64 over (seed, index). A fill with a
// period that divides a row or a tile gives a transposed or shifted surface the same values
// as the right one, which is what the ((i*7)%13) patterns these gates used could do.
static inline uint64_t rk_hash(uint64_t seed, uint64_t i) {
    uint64_t z = seed * 0x9E3779B97F4A7C15ull + i + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline void rk_fill(std::vector<float> & v, uint64_t seed, float amp) {
    for (size_t i = 0; i < v.size(); i++)
        v[i] = amp * (float)((double)(rk_hash(seed, i) >> 11) * (2.0 / 9007199254740992.0) - 1.0);
}

// The amplitude that puts a K-deep dot of two uniform fills near `target` rms, given the
// other operand's amplitude: rms = sqrt(K) * a * b / 3. The fp16 gates' element bar ignores
// an error under 0.5, so outputs near 10 are what make a zeroed element visible to it.
static inline float rk_amp_for(int K, float other_amp, float target = 10.0f) {
    return 3.0f * target / (sqrtf((float)K) * other_amp);
}

// The worst per-output-column cosine over a [M rows, N columns] result laid out out[m*N+n].
// A whole-tensor cosine averages one wrong column into thousands of right ones; this is the
// number a single bad column moves. `worst_col` (may be null) receives the column. A column
// whose reference is all zero is skipped, and a non-finite element returns -2.0.
static inline double rk_worst_col_cosine(const std::vector<float> & ref, const std::vector<float> & got,
                                         int M, int N, int * worst_col = nullptr) {
    double worst = 2.0; int wc = -1;
    for (int n = 0; n < N; n++) {
        double dot = 0, na = 0, nb = 0;
        for (int m = 0; m < M; m++) {
            const size_t i = (size_t)m * N + n;
            if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { if (worst_col) *worst_col = n; return -2.0; }
            dot += (double)ref[i] * got[i]; na += (double)ref[i] * ref[i]; nb += (double)got[i] * got[i];
        }
        if (na <= 0) continue;
        const double c = nb > 0 ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
        if (c < worst) { worst = c; wc = n; }
    }
    if (worst_col) *worst_col = wc;
    return worst > 1.5 ? 1.0 : worst;
}

// The worst per-row cosine over the same layout: one row is one output vector over N, the
// unit a mis-routed or mis-scattered row replaces whole. Same conventions as the column form.
static inline double rk_worst_row_cosine(const std::vector<float> & ref, const std::vector<float> & got,
                                         int M, int N, int * worst_row = nullptr) {
    double worst = 2.0; int wr = -1;
    for (int m = 0; m < M; m++) {
        double dot = 0, na = 0, nb = 0;
        for (int n = 0; n < N; n++) {
            const size_t i = (size_t)m * N + n;
            if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) { if (worst_row) *worst_row = m; return -2.0; }
            dot += (double)ref[i] * got[i]; na += (double)ref[i] * ref[i]; nb += (double)got[i] * got[i];
        }
        if (na <= 0) continue;
        const double c = nb > 0 ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
        if (c < worst) { worst = c; wr = m; }
    }
    if (worst_row) *worst_row = wr;
    return worst > 1.5 ? 1.0 : worst;
}

#endif // GGML_ROCKET_TEST_COMMON_H
