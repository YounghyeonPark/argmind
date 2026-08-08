/* SPDX-License-Identifier: Apache-2.0 */
/* Tuned CPU implementation of the reduced-form condensed ADMM.
 * =============================================================
 *
 * WHY THIS EXISTS
 *
 * Every CPU number in the performance model rests on an assumed "small-GEMM
 * efficiency" of 25-35% of SIMD peak. PHASE0_DECISION.md calls that assumption
 * the largest single error source in every ratio the project quotes, and it has
 * never been measured. This measures it, on the same algorithm the code
 * generator emits, so the comparison is apples to apples:
 *
 *   OSQP          sparse LDL^T on the augmented KKT system, the industry reference
 *   numpy/OpenBLAS the same reduced-form loop, library-tuned dense kernels
 *   THIS FILE      the same reduced-form loop, hand-laid-out and auto-vectorised
 *
 * It also tests the founder's thesis directly: how much of the gap between a
 * CPU's SIMD peak and its achieved throughput is recoverable in software?
 *
 * The algorithm is deliberately identical to argmind/codegen/admm.py, including
 * two rules from our own codegen contract:
 *   - Gamma^T y is accumulated ROW-WISE (contract rule 2). A column walk would
 *     stride the matrix and defeat the cache the same way it defeats our banks.
 *   - rho is a power of two, so rho and 1/rho are exact reciprocals (ruling R5).
 *
 * Build: gcc -O3 -march=native -ffast-math -o admm_tuned admm_tuned.c -lm
 * (-ffast-math is reported separately; it changes rounding, so the accuracy
 *  check in the harness is what decides whether it is admissible.)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef double real; /* fp64: matches the validated reference exactly */

/* ---------------------------------------------------------------- timing */
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ------------------------------------------------------------- problem */
typedef struct {
    int n;      /* decision variables                        */
    int mg;     /* rows of Gamma (state-box constraints)     */
    int m;      /* total constraint rows = n + mg            */
    real rho, sigma, rho_inv;
    real *L;    /* n x n lower triangular, row-major         */
    real *G;    /* mg x n Gamma, row-major                   */
    real *q;    /* n                                          */
    real *l, *u;/* m each: [input box (n) ; state box (mg)]  */
} Problem;

/* -------------------------------------------------------------- kernels */

/* y = G^T v, accumulated row-wise. One pass over G in memory order; the
 * accumulator stays in cache. This is codegen contract rule 2. */
static inline void gemv_T_rowwise(const real *restrict G, const real *restrict v,
                                  real *restrict y, int mg, int n) {
    for (int j = 0; j < n; ++j) y[j] = 0.0;
    for (int i = 0; i < mg; ++i) {
        const real vi = v[i];
        const real *restrict Gi = G + (size_t)i * n;
        for (int j = 0; j < n; ++j) y[j] += Gi[j] * vi;
    }
}

/* out = G x */
static inline void gemv_rowwise(const real *restrict G, const real *restrict x,
                                real *restrict out, int mg, int n) {
    for (int i = 0; i < mg; ++i) {
        const real *restrict Gi = G + (size_t)i * n;
        real acc = 0.0;
        for (int j = 0; j < n; ++j) acc += Gi[j] * x[j];
        out[i] = acc;
    }
}

/* Forward substitution L w = b. Sequential in i -- the loop-carried
 * dependency the whole architecture argument is about -- with the inner dot
 * product vectorisable. */
static inline void trsv_lower(const real *restrict L, const real *restrict b,
                              real *restrict w, int n) {
    for (int i = 0; i < n; ++i) {
        const real *restrict Li = L + (size_t)i * n;
        real acc = b[i];
        for (int j = 0; j < i; ++j) acc -= Li[j] * w[j];
        w[i] = acc / Li[i];
    }
}

/* Back substitution L^T x = w. Walks L by rows from the bottom so the memory
 * access stays sequential rather than striding down columns. */
static inline void trsv_upper_from_lower(const real *restrict L,
                                         const real *restrict w,
                                         real *restrict x, int n) {
    for (int i = 0; i < n; ++i) x[i] = w[i];
    for (int i = n - 1; i >= 0; --i) {
        const real *restrict Li = L + (size_t)i * n;
        x[i] /= Li[i];
        const real xi = x[i];
        for (int j = 0; j < i; ++j) x[j] -= Li[j] * xi;
    }
}

/* ------------------------------------------------------------------ ADMM */
/* Returns iterations run. Fixed iteration count: we are measuring throughput,
 * and the harness verifies the answer against OSQP separately. */
static int admm_solve(const Problem *p, int iters,
                      real *restrict x, real *restrict z, real *restrict y,
                      real *restrict work) {
    const int n = p->n, mg = p->mg;
    real *rhs = work;              /* n  */
    real *w   = rhs + n;           /* n  */
    real *Ax  = w + n;             /* m = n + mg */
    real *v   = Ax + p->m;         /* m  */

    for (int k = 0; k < iters; ++k) {
        /* v = rho*z - y   over both blocks */
        for (int i = 0; i < p->m; ++i) v[i] = p->rho * z[i] - y[i];

        /* rhs = sigma*x - q + A^T v, where A = [I ; Gamma].
         * The identity block contributes v[0..n) at zero MAC cost -- the
         * structural exploit the code generator also uses. */
        gemv_T_rowwise(p->G, v + n, rhs, mg, n);
        for (int j = 0; j < n; ++j) rhs[j] += p->sigma * x[j] - p->q[j] + v[j];

        /* K x = rhs via the cached Cholesky factor */
        trsv_lower(p->L, rhs, w, n);
        trsv_upper_from_lower(p->L, w, x, n);

        /* Ax = [x ; Gamma x] */
        memcpy(Ax, x, (size_t)n * sizeof(real));
        gemv_rowwise(p->G, x, Ax + n, mg, n);

        /* z = clip(Ax + y/rho, l, u);  y += rho*(Ax - z) */
        for (int i = 0; i < p->m; ++i) {
            const real t = Ax[i] + y[i] * p->rho_inv;
            real zi = t < p->l[i] ? p->l[i] : (t > p->u[i] ? p->u[i] : t);
            z[i] = zi;
            y[i] += p->rho * (Ax[i] - zi);
        }
    }
    return iters;
}

/* MACs per iteration, for the achieved-GFLOP/s figure. Counts only the
 * multiply-accumulates; the elementwise passes are reported separately by the
 * harness so neither inflates the other. */
static double macs_per_iter(const Problem *p) {
    const double n = p->n, mg = p->mg;
    return mg * n            /* Gamma^T v      */
         + n * (n - 1) / 2.0 /* forward solve  */
         + n * (n - 1) / 2.0 /* back solve     */
         + mg * n;           /* Gamma x        */
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s problem.bin iters [reps]\n", argv[0]);
        return 2;
    }
    const int iters = atoi(argv[2]);
    const int reps = argc > 3 ? atoi(argv[3]) : 2000;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }

    int hdr[3];
    if (fread(hdr, sizeof(int), 3, f) != 3) return 1;
    Problem p = {0};
    p.n = hdr[0]; p.mg = hdr[1]; p.m = p.n + p.mg;
    double scal[2];
    if (fread(scal, sizeof(double), 2, f) != 2) return 1;
    p.rho = scal[0]; p.sigma = scal[1]; p.rho_inv = 1.0 / p.rho;

    p.L = malloc(sizeof(real) * (size_t)p.n * p.n);
    p.G = malloc(sizeof(real) * (size_t)p.mg * p.n);
    p.q = malloc(sizeof(real) * p.n);
    p.l = malloc(sizeof(real) * p.m);
    p.u = malloc(sizeof(real) * p.m);
    if (fread(p.L, sizeof(real), (size_t)p.n * p.n, f) != (size_t)p.n * p.n) return 1;
    if (fread(p.G, sizeof(real), (size_t)p.mg * p.n, f) != (size_t)p.mg * p.n) return 1;
    if (fread(p.q, sizeof(real), p.n, f) != (size_t)p.n) return 1;
    if (fread(p.l, sizeof(real), p.m, f) != (size_t)p.m) return 1;
    if (fread(p.u, sizeof(real), p.m, f) != (size_t)p.m) return 1;
    fclose(f);

    real *x = calloc(p.n, sizeof(real));
    real *z = calloc(p.m, sizeof(real));
    real *y = calloc(p.m, sizeof(real));
    real *work = calloc((size_t)2 * p.n + 2 * p.m, sizeof(real));

    /* warm up caches and the branch predictor */
    for (int r = 0; r < 50; ++r) {
        memset(x, 0, p.n * sizeof(real));
        memset(z, 0, p.m * sizeof(real));
        memset(y, 0, p.m * sizeof(real));
        admm_solve(&p, iters, x, z, y, work);
    }

    /* time cold-start solves: zeroed state each time, as a first solve would be */
    double best = 1e30, total = 0.0;
    for (int r = 0; r < reps; ++r) {
        memset(x, 0, p.n * sizeof(real));
        memset(z, 0, p.m * sizeof(real));
        memset(y, 0, p.m * sizeof(real));
        const double t0 = now_seconds();
        admm_solve(&p, iters, x, z, y, work);
        const double dt = now_seconds() - t0;
        if (dt < best) best = dt;
        total += dt;
    }

    const double mac = macs_per_iter(&p) * iters;
    printf("{\"impl\":\"tuned_c\",\"n\":%d,\"mg\":%d,\"m\":%d,\"iters\":%d,"
           "\"reps\":%d,\"best_us\":%.4f,\"mean_us\":%.4f,\"macs\":%.0f,"
           "\"gflops_best\":%.3f,\"x\":[",
           p.n, p.mg, p.m, iters, reps, best * 1e6, total / reps * 1e6, mac,
           2.0 * mac / best / 1e9);
    for (int j = 0; j < p.n; ++j)
        printf("%s%.17g", j ? "," : "", (double)x[j]);
    printf("]}\n");
    return 0;
}
