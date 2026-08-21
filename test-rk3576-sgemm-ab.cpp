/* test-rk3576-sgemm-ab — does this part's int8 GEMM beat ggml's own CPU MUL_MAT?
 *
 * The question a port has to answer BEFORE it is written: at LLM PREFILL shapes (M at
 * the ubatch, never 1), is `rocket_matmul_int8_rk3576()` end to end faster than the
 * thing it would displace, which is ggml's CPU backend running the same MUL_MAT?
 *
 * WHY THE STORED COST TABLE CANNOT ANSWER IT. `chips/rk3576.md` puts the int8 matmul at
 * 2.4-13.7x, but its CPU column is a single-threaded scalar loop at about 5 GOP/s and
 * the table says so — "read the ratio as an upper bound against that baseline". A port
 * is not competing with that. So the CPU arm is rebuilt here three ways and the FASTEST
 * of the three is the denominator, which is the only reading a port cannot argue with:
 *
 *   - ggml F32   MUL_MAT, the backend's own kernel
 *   - ggml Q8_0  MUL_MAT, weights quantized, activations quantized per call by ggml —
 *                llama.cpp's actual quantized prefill path, and on this board it gets
 *                ggml's ARM repack kernels
 *   - OpenBLAS   cblas_sgemm, a tuned NEON GEMM, which is what the notes' caveat means
 *                by "a good NEON GEMM across the four A72s"
 *
 * WHAT THE TIMED REGIONS ARE, AND THE ONE ASYMMETRY THAT MATTERS. The NPU arm times the
 * WHOLE public entry — host pack of A and of B into the native cubes, the submits, the
 * de-scatter back to row-major — because that is what a frontend pays. The ggml arms
 * time `ggml_backend_graph_compute` only, so their weights are already in their final
 * layout and cost nothing per call. That is not a rigged comparison, it is the real
 * asymmetry: this part has NO resident-weight matmul path, so the entry re-packs B on
 * every call. To price what building one would buy, the (K, N) pair at 2048x2048 is run
 * at four values of M. B's pack is a function of (K, N) alone, so a least-squares fit of
 * T(M) = alpha + beta*M puts the M-independent work — B's pack plus the fixed per-call
 * overhead — in alpha, with no library change and no second copy of the arithmetic.
 *
 * Arms are INTERLEAVED inside each rep, one rep of every arm before the next rep of any,
 * because the arms differ enormously in host work. Pin the governor and taskset the
 * process to the A72s (cpu 4-7); both arms then get the same four cores.
 *
 *   sudo -E taskset -c 4-7 ./test-rk3576-sgemm-ab <shape-index|all> [reps]
 *
 * ONE SHAPE PER PROCESS, and the result appended to a file on DISK before the next one
 * starts. The first run of this harness took the board down hard enough to need a
 * physical power cycle, and everything it had measured was in a tmpfs, so it was all
 * lost. Two consequences are built in here: the driver runs the shapes as separate
 * processes smallest-first, so a hang names the shape that caused it instead of the
 * whole list; and every line is fsynced before the next shape starts. `all` is still
 * available for a board known to survive it.
 *
 * Discard the first PROCESS as well as the first rep: this is host-bound on both sides.
 * With one shape per process the driver gets that by running the whole list twice.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

extern "C" {
#include "rocket_npu.h"
#include "rocket_matmul.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <vector>
#include <algorithm>

/* Declared rather than included so the build does not have to find Debian's cblas.h. */
extern "C" {
void cblas_sgemm(int order, int transa, int transb, int m, int n, int k,
                 float alpha, const float *a, int lda, const float *b, int ldb,
                 float beta, float *c, int ldc);
void openblas_set_num_threads(int n);
}
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111
#define CBLAS_TRANS     112

#define N_THREADS 4

struct shape {
    int M, K, N;
    const char *note;
};

/* The 2048x2048 pair at four M is the intercept fit; the rest are one shape each.
 * K=8192 is the only cell past one task's 4608-element contraction, so it is the only
 * one that routes through the int32 writer's K split. */
static const shape SHAPES[] = {
    {  256, 2048, 2048, "d2048 attn qo   (M fit)" },
    {  512, 2048, 2048, "d2048 attn qo   (M fit)" },
    { 1024, 2048, 2048, "d2048 attn qo   (M fit)" },
    { 2048, 2048, 2048, "d2048 attn qo   (M fit)" },
    {  512, 2048,  512, "d2048 GQA kv"            },
    {  512, 2048, 8192, "d2048 FFN gate/up"       },
    {  512, 8192, 2048, "d2048 FFN down  (K SPLIT)" },
    {  512, 4096, 4096, "d4096 attn qo"           },
    {  512, 4096,11008, "d4096 FFN gate/up"       },
};
static const int N_SHAPES = (int)(sizeof SHAPES / sizeof *SHAPES);

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ---- the ggml arm ------------------------------------------------------------------
 *
 * One context per (shape, weight type). The weights are written once, outside the timed
 * region, exactly as a real model loads them; the activations are rewritten every rep so
 * no rep can be served from a cached quantization of the last one.
 */
struct ggml_arm {
    ggml_backend_t          backend = nullptr;
    ggml_context           *ctx_w   = nullptr;
    ggml_context           *ctx_g   = nullptr;
    ggml_backend_buffer_t   buf_w   = nullptr;
    ggml_gallocr_t          galloc  = nullptr;
    ggml_cgraph            *gf      = nullptr;
    ggml_tensor            *w = nullptr, *a = nullptr, *r = nullptr;
    std::vector<uint8_t>    gbuf;
    bool                    ok = false;

    bool init(ggml_backend_t be, int M, int K, int N, ggml_type wtype,
              const std::vector<float> &wf32)
    {
        backend = be;

        ggml_init_params pw = { (size_t)2 * ggml_tensor_overhead(), nullptr, true };
        ctx_w = ggml_init(pw);
        if (!ctx_w) return false;
        w = ggml_new_tensor_2d(ctx_w, wtype,          K, N);
        a = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32,  K, M);
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        if (!buf_w) return false;

        /* The weights, quantized once — a model's load-time cost, not a per-call one. */
        if (wtype == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, wf32.data(), 0, ggml_nbytes(w));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(w));
            size_t got = ggml_quantize_chunk(wtype, wf32.data(), q.data(), 0, N, K, nullptr);
            if (got != q.size()) return false;
            ggml_backend_tensor_set(w, q.data(), 0, q.size());
        }

        size_t gsz = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        gbuf.resize(gsz);
        ggml_init_params pg = { gsz, gbuf.data(), true };
        ctx_g = ggml_init(pg);
        if (!ctx_g) return false;
        gf = ggml_new_graph(ctx_g);
        r  = ggml_mul_mat(ctx_g, w, a);
        ggml_build_forward_expand(gf, r);

        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) return false;

        ok = true;
        return true;
    }

    /* Returns the wall of one compute, activations rewritten first and NOT timed. */
    double run(const std::vector<float> &af32)
    {
        ggml_backend_tensor_set(a, af32.data(), 0, ggml_nbytes(a));
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        return now_ms() - t0;
    }

    ~ggml_arm()
    {
        if (galloc) ggml_gallocr_free(galloc);
        if (buf_w)  ggml_backend_buffer_free(buf_w);
        if (ctx_g)  ggml_free(ctx_g);
        if (ctx_w)  ggml_free(ctx_w);
    }
};

struct row {
    shape  s;
    double npu = 0.0, gf32 = 0.0, gq8 = 0.0, blas = 0.0;
    int    npu_rc = 0;
    bool   npu_ran = false;
    long   bad = 0, worst = 0;      /* elements off by more than one count, and the max */
};

/* THE ARM THAT MUST SUCCEED. A timing table over a datapath nobody scored is worthless
 * here: on this part a wrong value comes back as a full, correctly sized, entirely
 * plausible surface. The float GEMM the BLAS arm already computed IS the reference —
 * the NPU entry's contract is C = sat8(round(scale * sum_k A*B)), so requantizing the
 * exact float sum gives the expected int8 surface for free.
 *
 * A count of +-1 is NOT a defect: the DPU's requant rounds ties to EVEN where the host's
 * round() goes half-away-from-zero, and the OUT_CVT multiplier approximates `scale`. So
 * the assertion is elements off by MORE than one, which no rounding rule can explain. */
static void score_int8(const shape &s, const std::vector<float> &ref, float scale,
                       const std::vector<int8_t> &got, long *bad, long *worst)
{
    *bad = 0; *worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double v = (double)ref[i] * scale;
        long r = (long)(v < 0 ? v - 0.5 : v + 0.5);
        if (r < -128) r = -128; else if (r > 127) r = 127;
        long d = labs(r - (long)got[i]);
        if (d > *worst) *worst = d;
        if (d > 1) (*bad)++;
    }
    (void)s;
}

/* The SoC's own temperature, so a run that ends in a silent death leaves a trend behind
 * it rather than nothing. The first version of this harness took the board down with no
 * kernel message of any kind, which is what a brownout looks like and what a driver
 * fault does not. */
static double zone_temp(const char *type)
{
    for (int z = 0; z < 12; z++) {
        char p[128], buf[64];
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/type", z);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        if (!fgets(buf, sizeof buf, f)) { fclose(f); continue; }
        fclose(f);
        buf[strcspn(buf, "\n")] = 0;
        if (strcmp(buf, type)) continue;
        snprintf(p, sizeof p, "/sys/class/thermal/thermal_zone%d/temp", z);
        f = fopen(p, "r");
        if (!f) return -1.0;
        double t = -1.0;
        if (fscanf(f, "%lf", &t) != 1) t = -1.0;
        fclose(f);
        return t / 1000.0;
    }
    return -1.0;
}

/* Appended and FSYNCED before the next shape starts: a hang must not cost the shapes
 * that already ran, which is exactly what it cost the first time. */
static void record(const char *line)
{
    const char *path = getenv("SGEMM_AB_LOG");
    if (!path || !*path) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fputs(line, f);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
}

int main(int argc, char **argv)
{
    /* `<index>` runs one shape and exits, which is how the driver isolates a hang.
     * `all` keeps the original behaviour for a board known to survive it. */
    int only = -1;
    if (argc > 1 && strcmp(argv[1], "all") != 0) only = atoi(argv[1]);
    int reps = (argc > 2) ? atoi(argv[2]) : 3;
    if (reps < 1) reps = 3;

    int fd = rocket_open();
    if (fd < 0) {
        fprintf(stderr, "rocket_open failed — /dev/accel/accel0 needs privilege "
                        "(run under sudo -E)\n");
        return 2;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { fprintf(stderr, "ggml_backend_cpu_init failed\n"); return 2; }
    ggml_backend_cpu_set_n_threads(be, N_THREADS);
    openblas_set_num_threads(N_THREADS);

    printf("RK3576 sgemm A/B — the NPU's int8 GEMM against ggml's own CPU MUL_MAT.\n");
    printf("%d threads, %d timed reps after a discarded one, arms interleaved.\n", N_THREADS, reps);
    printf("The CPU column is the FASTEST of ggml-F32, ggml-Q8_0 and OpenBLAS sgemm.\n\n");

    std::vector<row> rows;

    for (int si = 0; si < N_SHAPES; si++) {
        if (only >= 0 && si != only) continue;
        const shape s = SHAPES[si];
        row out; out.s = s;
        double t_before = zone_temp("npu-thermal"), tb_big = zone_temp("bigcore-thermal");

        /* One operand set, shared by every arm, so no arm sees different data. */
        std::vector<float>  Wf((size_t)s.K * s.N), Af((size_t)s.M * s.K);
        std::vector<int8_t> Wi((size_t)s.K * s.N), Ai((size_t)s.M * s.K);
        std::vector<int8_t> Ci((size_t)s.M * s.N);
        std::vector<float>  Cf((size_t)s.M * s.N);
        srand(1234 + si);
        for (size_t i = 0; i < Wf.size(); i++) { int v = (rand() & 15) - 8; Wi[i] = (int8_t)v; Wf[i] = (float)v; }
        for (size_t i = 0; i < Af.size(); i++) { int v = (rand() & 15) - 8; Ai[i] = (int8_t)v; Af[i] = (float)v; }

        /* Keeps the requant off its rails: E|sum| ~ sqrt(K)*sigma^2 with sigma ~ 4.6. */
        float scale = 1.0f / (float)(4.0 * sqrt((double)s.K) * 21.0 / 8.0);

        ggml_arm g32, gq8;
        bool have32 = g32.init(be, s.M, s.K, s.N, GGML_TYPE_F32,  Wf);
        bool haveq8 = gq8.init(be, s.M, s.K, s.N, GGML_TYPE_Q8_0, Wf);

        std::vector<double> v_npu, v_g32, v_gq8, v_bl;

        for (int r = 0; r <= reps; r++) {          /* r == 0 is the discarded rep */
            double t;

            t = now_ms();
            int rc = rocket_matmul_int8_rk3576(fd, s.M, s.K, s.N, Ai.data(), Wi.data(),
                                               nullptr, scale, Ci.data());
            t = now_ms() - t;
            if (r) { if (rc == ROCKET_OK) v_npu.push_back(t); }
            out.npu_rc = rc;

            if (have32) { t = g32.run(Af); if (r) v_g32.push_back(t); }
            if (haveq8) { t = gq8.run(Af); if (r) v_gq8.push_back(t); }

            t = now_ms();
            cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                        s.M, s.N, s.K, 1.0f, Af.data(), s.K, Wf.data(), s.K,
                        0.0f, Cf.data(), s.N);
            t = now_ms() - t;
            if (r) v_bl.push_back(t);
        }

        /* Cf holds the last BLAS result, which is the exact float sum for this shape. */
        if (out.npu_rc == ROCKET_OK) score_int8(s, Cf, scale, Ci, &out.bad, &out.worst);

        out.npu     = median(v_npu);
        out.npu_ran = !v_npu.empty();
        out.gf32    = have32 ? median(v_g32) : 0.0;
        out.gq8     = haveq8 ? median(v_gq8) : 0.0;
        out.blas    = median(v_bl);
        rows.push_back(out);

        double gops = 2.0 * s.M * s.K * s.N / 1e9;
        printf("  %-26s M%-5d K%-5d N%-6d  %6.2f GOP", s.note, s.M, s.K, s.N, gops);
        if (out.npu_ran) printf("   NPU %8.2f ms", out.npu);
        else             printf("   NPU  REFUSED rc=%d", out.npu_rc);
        printf("   ggmlF32 %8.2f   ggmlQ8_0 %8.2f   BLAS %8.2f   npuC %.1f->%.1f bigC %.1f->%.1f\n",
               out.gf32, out.gq8, out.blas,
               t_before, zone_temp("npu-thermal"), tb_big, zone_temp("bigcore-thermal"));
        fflush(stdout);

        printf("      correctness vs the exact float sum requantized: %ld of %zu elements "
               "off by more than one count, worst %ld\n",
               out.bad, (size_t)s.M * s.N, out.worst);
        fflush(stdout);

        char line[360];
        snprintf(line, sizeof line,
                 "SHAPE %d M %d K %d N %d npu %.3f gf32 %.3f gq8 %.3f blas %.3f rc %d "
                 "bad %ld of %zu worst %ld npuC %.1f bigC %.1f\n",
                 si, s.M, s.K, s.N, out.npu_ran ? out.npu : -1.0,
                 out.gf32, out.gq8, out.blas, out.npu_rc,
                 out.bad, (size_t)s.M * s.N, out.worst,
                 zone_temp("npu-thermal"), zone_temp("bigcore-thermal"));
        record(line);
    }

    /* ---- the verdict ----------------------------------------------------------
     * Only meaningful over the whole list; per-shape runs are aggregated from the log. */
    if (rows.size() < 2) { ggml_backend_free(be); rocket_close(fd); return 0; }
    printf("\n== the ratio, against the FASTEST CPU arm at each shape ==\n");
    printf("  %-26s %-20s %10s %10s %8s %s\n",
           "shape", "M / K / N", "NPU ms", "bestCPU", "ratio", "which CPU arm");
    std::vector<double> ratios, ratios_fit;
    for (const row &o : rows) {
        double best = 1e30; const char *who = "-";
        if (o.gf32 > 0 && o.gf32 < best) { best = o.gf32; who = "ggml F32";  }
        if (o.gq8  > 0 && o.gq8  < best) { best = o.gq8;  who = "ggml Q8_0"; }
        if (o.blas > 0 && o.blas < best) { best = o.blas; who = "OpenBLAS";  }
        char mkn[40];
        snprintf(mkn, sizeof mkn, "%d / %d / %d", o.s.M, o.s.K, o.s.N);
        if (!o.npu_ran) {
            printf("  %-26s %-20s %10s %10.2f %8s %s\n", o.s.note, mkn, "REFUSED", best, "-", who);
            continue;
        }
        double ratio = best / o.npu;
        ratios.push_back(ratio);
        printf("  %-26s %-20s %10.2f %10.2f %8.2fx %s\n", o.s.note, mkn, o.npu, best, ratio, who);
    }
    printf("\n  MEDIAN ratio over %zu shapes that ran: %.2fx   (STEP 5's bar is 1.30x)\n",
           ratios.size(), median(ratios));
    int over = 0;
    for (double x : ratios) if (x >= 1.30) over++;
    printf("  shapes at or above the bar: %d of %zu\n", over, ratios.size());

    /* ---- the intercept: what a resident-weight path could remove ---------------- */
    printf("\n== T(M) = alpha + beta*M at K=2048 N=2048, four M ==\n");
    {
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
        for (const row &o : rows) {
            if (o.s.K != 2048 || o.s.N != 2048 || !o.npu_ran) continue;
            double x = o.s.M, y = o.npu;
            sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
        }
        if (n >= 2) {
            double den = n * sxx - sx * sx;
            double beta  = (n * sxy - sx * sy) / den;
            double alpha = (sy - beta * sx) / n;
            printf("  alpha = %.2f ms (M-independent: B's cube pack + the fixed per-call\n"
                   "          overhead — the term a resident-weight path would remove)\n", alpha);
            printf("  beta  = %.4f ms per row of M\n", beta);
            for (const row &o : rows) {
                if (o.s.K != 2048 || o.s.N != 2048 || !o.npu_ran) continue;
                printf("    M=%-5d measured %7.2f ms, alpha is %5.1f%% of it\n",
                       o.s.M, o.npu, 100.0 * alpha / o.npu);
            }
        } else {
            printf("  fewer than two M cells ran; no fit\n");
        }
    }

    ggml_backend_free(be);
    rocket_close(fd);
    return 0;
}
