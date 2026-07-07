// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The ggml-rocket authors
/*
 * test-hadamard-quant.cpp — standalone, model-free validation of the Hadamard
 * rotation for int8 W8A8. Pure CPU; no ggml, no NPU. Build+run:
 *     g++ -O2 -std=c++17 -o /tmp/hq test-hadamard-quant.cpp && /tmp/hq
 *
 * Claim under test: rotating A and B by an orthonormal Hadamard H along K BEFORE
 * int8 quantization smears per-channel activation outliers across all channels,
 * so per-row/per-channel symmetric int8 quant loses far less precision. Since H is
 * orthonormal, (A·H)(B·H)^T = A·B^T exactly (fp error aside) — the rotation is free
 * arithmetically; only the quantization sees the (now benign) rotated values.
 *
 * THE OBSTACLE this solves: Gemma's dominant K are 3840=64*60 and 15360=256*60 —
 * 15*2^k, NOT powers of 2, so a plain fast Walsh-Hadamard (FWHT, power-of-2 only)
 * can't rotate them. We use a Kronecker Hadamard H_K = H_{2^k} (x) H_60: FWHT
 * across 60-element blocks + a fixed 60x60 Hadamard within each block. H_60 exists
 * via the Paley-I construction (59 prime, 59 = 3 mod 4 => skew-Hadamard of order 60).
 *
 * Two correctness checks are built in and printed: (1) H_60 is orthogonal
 * (H_60 H_60^T = 60 I); (2) the full rotation preserves A·B^T (rel err ~1e-15).
 * Then the payoff: W8A8 Frobenius error plain vs Hadamard, per K.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
using std::vector;

static int modpow(long a, long e, long p) { long r = 1; a %= p; while (e) { if (e & 1) r = r*a%p; a = a*a%p; e >>= 1; } return (int)r; }
// Legendre symbol (a/p): +1 if a is a nonzero quadratic residue mod p, -1 if not, 0 if a==0.
static int legendre(int a, int p) { a %= p; if (a < 0) a += p; if (a == 0) return 0; return modpow(a, (p-1)/2, p) == 1 ? 1 : -1; }

// H_60 via Paley I (q=59 prime, q=3 mod 4): H = I + S, S = [[0, 1^T],[-1, Q]],
// Q the Jacobsthal matrix Q[a][b]=chi(a-b). Skew-Hadamard => H H^T = 60 I.
static double H60[60][60];
static void build_H60(void) {
    const int q = 59;
    for (int i = 0; i < 60; i++)
        for (int j = 0; j < 60; j++) {
            if (j == 0 && i == 0)      H60[i][j] =  1;   // corner
            else if (i == 0)           H60[i][j] =  1;   // top row
            else if (j == 0)           H60[i][j] = -1;   // left col (skew)
            else if (i == j)           H60[i][j] =  1;   // diag = I (chi(0)=0 -> +1)
            else                       H60[i][j] = legendre((i-1) - (j-1), q);
        }
}
static double check_H60(void) {
    double maxoff = 0, maxdiag = 0;
    for (int i = 0; i < 60; i++)
        for (int j = 0; j < 60; j++) {
            double s = 0; for (int k = 0; k < 60; k++) s += H60[i][k]*H60[j][k];
            s /= 60.0;
            if (i == j) maxdiag = fmax(maxdiag, fabs(s - 1));
            else        maxoff  = fmax(maxoff, fabs(s));
        }
    printf("[gate1] H60 orthogonality: max_offdiag=%.2e max_diagerr=%.2e %s\n",
           maxoff, maxdiag, (maxoff < 1e-9 && maxdiag < 1e-9) ? "OK" : "FAIL");
    return fmax(maxoff, maxdiag);
}

static bool ispow2(int x) { return x > 0 && (x & (x-1)) == 0; }

// in-place unnormalized FWHT on contiguous v[0..n), n a power of 2 (returns H_n v).
static void fwht(double *v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) { double a = v[j], b = v[j+len]; v[j] = a + b; v[j+len] = a - b; }
}

// Apply the normalized orthonormal rotation H_K/sqrt(K) to each row of X[rows][K].
// K=2^k -> plain FWHT. K=60*2^k -> H_60 within each 60-block, FWHT across blocks.
static void rotate_rows(double *X, int rows, int K) {
    bool kron = !ispow2(K);
    int B2 = 0;
    if (kron) { if (K % 60 || !ispow2(K/60)) { fprintf(stderr, "unsupported K=%d\n", K); exit(1); } B2 = K/60; }
    const double inv = 1.0 / sqrt((double)K);
    vector<double> col(kron ? B2 : 0);
    for (int r = 0; r < rows; r++) {
        double *row = X + (size_t)r * K;
        if (!kron) {
            fwht(row, K);
        } else {
            for (int b = 0; b < B2; b++) {                 // H_60 within each block (matrix*col)
                double *blk = row + b*60, out[60];
                for (int o = 0; o < 60; o++) { double s = 0; for (int k = 0; k < 60; k++) s += H60[o][k]*blk[k]; out[o] = s; }
                for (int o = 0; o < 60; o++) blk[o] = out[o];
            }
            for (int lane = 0; lane < 60; lane++) {         // FWHT across blocks, per lane (stride 60)
                for (int b = 0; b < B2; b++) col[b] = row[b*60 + lane];
                fwht(col.data(), B2);
                for (int b = 0; b < B2; b++) row[b*60 + lane] = col[b];
            }
        }
        for (int k = 0; k < K; k++) row[k] *= inv;
    }
}

static int8_t q8(double x, double inv) { long q = lrint(x * inv); if (q > 127) q = 127; if (q < -127) q = -127; return (int8_t)q; }

// W8A8: A per-row scale, B per-channel scale, int8xint8->int32, dequant. Returns
// the relative Frobenius error ||C_q - C_ref|| / ||C_ref|| (an honest aggregate;
// NOT the near-zero-denominator per-element rel that inflates ROCKET_VERIFY).
static double w8a8_relerr(const double *A, const double *B, const double *Cref, int M, int K, int N) {
    vector<int8_t> qA((size_t)M*K), qB((size_t)N*K);
    vector<double> as(M), bs(N);
    for (int m = 0; m < M; m++) { double am = 0; for (int k = 0; k < K; k++) am = fmax(am, fabs(A[(size_t)m*K+k]));
        double s = am > 0 ? am/127.0 : 1.0; as[m] = s; double iv = 1.0/s;
        for (int k = 0; k < K; k++) qA[(size_t)m*K+k] = q8(A[(size_t)m*K+k], iv); }
    for (int n = 0; n < N; n++) { double am = 0; for (int k = 0; k < K; k++) am = fmax(am, fabs(B[(size_t)n*K+k]));
        double s = am > 0 ? am/127.0 : 1.0; bs[n] = s; double iv = 1.0/s;
        for (int k = 0; k < K; k++) qB[(size_t)n*K+k] = q8(B[(size_t)n*K+k], iv); }
    double num = 0, den = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            long acc = 0; for (int k = 0; k < K; k++) acc += (int)qA[(size_t)m*K+k] * (int)qB[(size_t)n*K+k];
            double cq = (double)acc * as[m] * bs[n], d = cq - Cref[(size_t)m*N+n];
            num += d*d; den += Cref[(size_t)m*N+n]*Cref[(size_t)m*N+n];
        }
    return sqrt(num/den);
}

int main(void) {
    build_H60();
    if (check_H60() > 1e-9) return 1;

    struct { int M, K, N; const char *tag; } shapes[] = {
        { 8,  4096, 256, "2^12 (FWHT)" },     // power-of-2: plain FWHT
        { 8,  3840, 256, "64x60 (kron)" },     // Gemma hidden: q/k/v/gate/up read this K
        { 8, 15360, 256, "256x60 (kron)" },    // Gemma FFN: down_proj reads this K (worst outliers)
    };
    srand(1);
    for (auto s : shapes) {
        const int M = s.M, K = s.K, N = s.N;
        vector<double> A((size_t)M*K), B((size_t)N*K);
        for (auto &x : B) x = ((rand()/(double)RAND_MAX) - 0.5) * 0.2;          // bounded weights
        for (int m = 0; m < M; m++) for (int k = 0; k < K; k++)
            A[(size_t)m*K+k] = ((rand()/(double)RAND_MAX) - 0.5) * 2.0;          // ~U(-1,1) activations
        int nout = 0;                                                            // per-CHANNEL outliers: ~1% of K cols x100
        for (int k = 0; k < K; k++) if (k % 97 == 0) { nout++; for (int m = 0; m < M; m++) A[(size_t)m*K+k] *= 100.0; }

        vector<double> C((size_t)M*N);
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            double acc = 0; for (int k = 0; k < K; k++) acc += A[(size_t)m*K+k]*B[(size_t)n*K+k];
            C[(size_t)m*N+n] = acc;
        }
        const double e_plain = w8a8_relerr(A.data(), B.data(), C.data(), M, K, N);

        vector<double> Ar = A, Br = B;
        rotate_rows(Ar.data(), M, K);
        rotate_rows(Br.data(), N, K);
        double num = 0, den = 0;                                                 // gate2: rotation preserves A*B^T
        for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
            double acc = 0; for (int k = 0; k < K; k++) acc += Ar[(size_t)m*K+k]*Br[(size_t)n*K+k];
            double d = acc - C[(size_t)m*N+n]; num += d*d; den += C[(size_t)m*N+n]*C[(size_t)m*N+n];
        }
        const double preserve = sqrt(num/den);
        const double e_had = w8a8_relerr(Ar.data(), Br.data(), C.data(), M, K, N);

        printf("K=%-6d %-14s outliers=%3d/%d | [gate2] rot preserves A*B^T rel=%.1e | "
               "W8A8 relerr plain=%.4f hadamard=%.4f -> %.1fx better\n",
               K, s.tag, nout, K, preserve, e_plain, e_had, e_plain/e_had);
    }

    // Outlier-severity sweep (K=3840). As the per-channel outlier magnitude grows,
    // the per-row int8 scale (= row_amax/127) is pinned by the outlier, so normal
    // channels round toward ZERO and the matmul collapses -> plain W8A8 falls off a
    // cliff. Hadamard spreads each outlier's energy across all K channels, so no
    // single channel dominates the scale and the normals survive. The real Gemma
    // down_proj had outliers ~2000x, i.e. well past the cliff.
    printf("\noutlier-severity sweep (K=3840, ~1%% outlier channels, identical base data):\n");
    {
        const int M = 8, K = 3840, N = 256;
        for (double factor : { 1.0, 10.0, 100.0, 300.0, 1000.0, 10000.0 }) {
            srand(1);   // reset => identical base A,B every factor; only outlier scale varies
            vector<double> A((size_t)M*K), B((size_t)N*K);
            for (auto &x : B) x = ((rand()/(double)RAND_MAX) - 0.5) * 0.2;
            for (int m = 0; m < M; m++) for (int k = 0; k < K; k++)
                A[(size_t)m*K+k] = ((rand()/(double)RAND_MAX) - 0.5) * 2.0;
            for (int k = 0; k < K; k++) if (k % 97 == 0) for (int m = 0; m < M; m++) A[(size_t)m*K+k] *= factor;
            vector<double> C((size_t)M*N);
            for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
                double acc = 0; for (int k = 0; k < K; k++) acc += A[(size_t)m*K+k]*B[(size_t)n*K+k];
                C[(size_t)m*N+n] = acc;
            }
            const double ep = w8a8_relerr(A.data(), B.data(), C.data(), M, K, N);
            vector<double> Ar = A, Br = B; rotate_rows(Ar.data(), M, K); rotate_rows(Br.data(), N, K);
            const double eh = w8a8_relerr(Ar.data(), Br.data(), C.data(), M, K, N);
            printf("  outlier x%-7.0f | plain=%.4f  hadamard=%.4f  -> %.1fx better\n", factor, ep, eh, ep/eh);
        }
    }
    return 0;
}
