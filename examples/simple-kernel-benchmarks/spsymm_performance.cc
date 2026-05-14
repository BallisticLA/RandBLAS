// Copyright, 2026. See LICENSE for copyright holder information.
//
// ============================================================================
// SPSYMM / SKETCH_SYMMETRIC PERFORMANCE BENCHMARK
// ============================================================================
//
// This benchmark answers two performance questions about the symm-kernels
// added in https://github.com/BallisticLA/RandBLAS/pull/163 :
//
//   1. Sparse: How much does the new RandBLAS::spsymm (one-triangle storage,
//      MKL fast path via SPARSE_MATRIX_TYPE_SYMMETRIC) speed up over the
//      "both-triangles + right_spmm" workaround that downstream code had to
//      use before this PR?
//
//   2. Dense: How much does the rewritten RandBLAS::sketch_symmetric (now
//      backed by blas::symm) speed up over its pre-PR behaviour, which
//      silently forwarded to sketch_general -> lskge3 -> blas::gemm with no
//      symmetry exploitation?
//
// NOTATION:
//   A_symm  - symmetric matrix of order n_A (dense or sparse, one triangle)
//   B       - dense matrix (n_A x d for side=Left; d x n_A for side=Right)
//   Y       - dense result matrix (same shape as B)
//   density - fraction of upper-triangle entries of A_symm that are nonzero
//             (the implied lower-triangle entries follow by symmetry)
//
// USAGE:
//   ./spsymm_performance                                # default sweep
//   ./spsymm_performance n_A d density [num_trials]     # single config
//
//   Defaults: num_trials=10, density=0.05.
//
// EXAMPLES:
//   env OMP_NUM_THREADS=8 ./spsymm_performance
//   env OMP_NUM_THREADS=8 ./spsymm_performance 2000 200 0.01
//
// ============================================================================

#include <RandBLAS.hh>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using blas::Layout;
using blas::Uplo;
using blas::Side;
using blas::Op;

using T = double;
using sint_t = int64_t;
using SpMat = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;


// ============================================================================
// Helpers: random dense symmetric matrix + sparse counterparts.
// ============================================================================

void make_dense_symmetric(int64_t n, std::vector<T>& A, uint64_t seed) {
    A.assign(static_cast<size_t>(n) * n, T(0));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> uni(-1.0, 1.0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i <= j; ++i) {
            T v = uni(rng);
            A[i + j * n] = v;
            if (i != j) A[j + i * n] = v;  // mirror
        }
    }
}

// Sparsify the upper triangle of A (including diagonal) at the given density,
// then mirror into the lower triangle.
void sparsify_upper_then_mirror(int64_t n, std::vector<T>& A, double density, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i <= j; ++i) {
            if (uni01(rng) >= density) {
                A[i + j * n] = T(0);
                if (i != j) A[j + i * n] = T(0);
            }
        }
    }
}

// Build a one-triangle (Upper) CSR view by walking the upper triangle of A_dense.
SpMat build_csr_upper_only(int64_t n, const std::vector<T>& A_dense) {
    std::vector<sint_t> rowptr(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j)
            if (A_dense[i + j * n] != T(0)) ++rowptr[i + 1];
    }
    for (int64_t i = 0; i < n; ++i) rowptr[i + 1] += rowptr[i];
    int64_t nnz = rowptr[n];

    SpMat A_sparse(n, n);
    A_sparse.reserve(nnz);
    std::vector<sint_t> tmp_rp(rowptr);  // running cursor per row
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            T v = A_dense[i + j * n];
            if (v != T(0)) {
                int64_t pos = tmp_rp[i]++;
                A_sparse.colidxs[pos] = static_cast<sint_t>(j);
                A_sparse.vals[pos] = v;
            }
        }
    }
    for (int64_t i = 0; i <= n; ++i) A_sparse.rowptr[i] = rowptr[i];
    return A_sparse;
}

// Build a full (both-triangles-stored) CSR for the pre-PR workaround comparison.
SpMat build_csr_both_triangles(int64_t n, const std::vector<T>& A_dense) {
    std::vector<sint_t> rowptr(n + 1, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            if (A_dense[i + j * n] != T(0)) ++rowptr[i + 1];
    for (int64_t i = 0; i < n; ++i) rowptr[i + 1] += rowptr[i];
    int64_t nnz = rowptr[n];

    SpMat A_sparse(n, n);
    A_sparse.reserve(nnz);
    std::vector<sint_t> tmp_rp(rowptr);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            T v = A_dense[i + j * n];
            if (v != T(0)) {
                int64_t pos = tmp_rp[i]++;
                A_sparse.colidxs[pos] = static_cast<sint_t>(j);
                A_sparse.vals[pos] = v;
            }
        }
    }
    for (int64_t i = 0; i <= n; ++i) A_sparse.rowptr[i] = rowptr[i];
    return A_sparse;
}


// ============================================================================
// Timing helper: median + min over num_trials.
// ============================================================================
template <typename Func>
std::pair<long, long> run_trials(Func&& func, int num_trials) {
    std::vector<long> times;
    times.reserve(num_trials);
    for (int t = 0; t < num_trials; ++t) {
        auto start = steady_clock::now();
        func();
        auto end = steady_clock::now();
        times.push_back(duration_cast<microseconds>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return {times[0], times[num_trials / 2]};
}

void print_row(const std::string& name, long min_us, long med_us, long baseline) {
    double ratio = (baseline > 0) ? (double)min_us / baseline : 1.0;
    std::cout << "  " << std::setw(38) << std::left << name
              << std::setw(10) << std::right << min_us
              << std::setw(10) << med_us
              << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x\n";
}


// ============================================================================
// run_config: one (n_A, d, density) point. Compares the symm-aware path
// against the workaround / pre-PR baselines.
// ============================================================================
void run_config(int64_t n_A, int64_t d, double density, int num_trials) {
    uint64_t seed = 12345;
    Layout layout = Layout::ColMajor;
    T alpha = T(1.0), beta = T(0.0);

    std::cout << "--- A is " << n_A << "x" << n_A
              << " symmetric, B is " << n_A << "x" << d
              << ", density=" << std::setprecision(4) << density
              << " (median + min over " << num_trials << " trials) ---\n";

    std::vector<T> A_full(static_cast<size_t>(n_A) * n_A);
    make_dense_symmetric(n_A, A_full, seed);
    sparsify_upper_then_mirror(n_A, A_full, density, seed + 1);

    SpMat A_csr_upper = build_csr_upper_only(n_A, A_full);
    SpMat A_csr_full  = build_csr_both_triangles(n_A, A_full);

    int64_t actual_nnz_upper = A_csr_upper.rowptr[n_A];
    int64_t actual_nnz_full  = A_csr_full.rowptr[n_A];
    std::cout << "  upper-triangle nnz = " << actual_nnz_upper
              << " (full = " << actual_nnz_full << ")\n";

    std::vector<T> B(static_cast<size_t>(n_A) * d);
    {
        std::mt19937_64 rng(seed + 2);
        std::uniform_real_distribution<T> uni(-1.0, 1.0);
        for (auto& x : B) x = uni(rng);
    }
    std::vector<T> Y(static_cast<size_t>(n_A) * d, T(0));

    std::cout << "\n  SPARSE (side=Left, Y = A*B):\n";
    std::cout << "  " << std::setw(38) << std::left << "kernel"
              << std::setw(10) << std::right << "min(us)"
              << std::setw(10) << "med(us)"
              << std::setw(10) << "ratio\n";

    // 1. RandBLAS::spsymm on one-triangle CSR (PR #163's new path).
    auto [t1_min, t1_med] = run_trials([&] {
        RandBLAS::spsymm(layout, Uplo::Upper, n_A, d,
                         alpha, A_csr_upper, B.data(), n_A,
                         beta, Y.data(), n_A);
    }, num_trials);

    // 2. RandBLAS::spmm with full storage (the pre-PR workaround:
    //    both triangles materialized into a general-sparse CSR).
    auto [t2_min, t2_med] = run_trials([&] {
        RandBLAS::spmm(layout, Op::NoTrans, Op::NoTrans,
                       n_A, d, n_A,
                       alpha, A_csr_full, B.data(), n_A,
                       beta, Y.data(), n_A);
    }, num_trials);

    // 3. Dense blas::symm on the fully populated dense A (the "ideal"
    //    BLAS-symm baseline -- doesn't exploit sparsity, but is the
    //    fastest possible dense SYMM).
    auto [t3_min, t3_med] = run_trials([&] {
        blas::symm(layout, Side::Left, Uplo::Upper, n_A, d,
                   alpha, A_full.data(), n_A, B.data(), n_A,
                   beta, Y.data(), n_A);
    }, num_trials);

    print_row("RandBLAS::spsymm (one tri + MKL)", t1_min, t1_med, t1_min);
    print_row("RandBLAS::spmm (full + MKL)",      t2_min, t2_med, t1_min);
    print_row("blas::symm (dense ref)",           t3_min, t3_med, t1_min);

    // Dense-symmetric sketching comparison: new SYMM-backed vs the
    // pre-PR GEMM-forwarding behaviour.
    std::cout << "\n  DENSE (sketch_symmetric vs sketch_general on dense-symm A):\n";
    std::cout << "  " << std::setw(38) << std::left << "kernel"
              << std::setw(10) << std::right << "min(us)"
              << std::setw(10) << "med(us)"
              << std::setw(10) << "ratio\n";

    RandBLAS::DenseDist DS(n_A, d, RandBLAS::ScalarDist::Uniform);
    RandBLAS::DenseSkOp<T> S(DS, static_cast<uint32_t>(seed + 3));
    RandBLAS::fill_dense(S);

    // 1. New: RandBLAS::sketch_symmetric (SYMM-backed via blas::symm).
    auto [s1_min, s1_med] = run_trials([&] {
        RandBLAS::sketch_symmetric(layout, Uplo::Upper, n_A, d,
                                   alpha, A_full.data(), n_A, S, 0, 0,
                                   beta, Y.data(), n_A);
    }, num_trials);

    // 2. Old: equivalent call via sketch_general (the pre-PR behaviour
    //    when sketch_symmetric silently forwarded to GEMM).
    auto [s2_min, s2_med] = run_trials([&] {
        RandBLAS::sketch_general(layout, Op::NoTrans, Op::NoTrans,
                                 n_A, d, n_A,
                                 alpha, A_full.data(), n_A, S, 0, 0,
                                 beta, Y.data(), n_A);
    }, num_trials);

    print_row("sketch_symmetric (new SYMM path)", s1_min, s1_med, s1_min);
    print_row("sketch_general (pre-PR GEMM path)", s2_min, s2_med, s1_min);

    std::cout << "\n";
}


// ============================================================================
// main: default 4-point sweep, or single config from argv.
// ============================================================================
int main(int argc, char** argv) {
    if (argc >= 4) {
        int64_t n_A = std::stoll(argv[1]);
        int64_t d   = std::stoll(argv[2]);
        double density = std::stod(argv[3]);
        int num_trials = (argc >= 5) ? std::stoi(argv[4]) : 10;
        run_config(n_A, d, density, num_trials);
    } else {
        int num_trials = 10;
        std::cout << "Default sweep (n_A in {500, 1000, 2000}, d=200, density=0.05)\n\n";
        for (int64_t n_A : {500, 1000, 2000}) {
            run_config(n_A, /*d=*/200, /*density=*/0.05, num_trials);
        }
    }
    return 0;
}
