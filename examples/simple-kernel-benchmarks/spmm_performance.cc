// Copyright, 2026. See LICENSE for copyright holder information.
//
// ============================================================================
// SPARSE-DENSE MATRIX MULTIPLICATION PERFORMANCE BENCHMARK
// ============================================================================
//
// This benchmark answers four questions about RandBLAS SpMM performance:
//
//   1. Which sparse format is best for left SpMM (B = S*A)?
//      -> CSR, CSC, and COO are compared; CSR and COO (via MKL) tend to win.
//
//   2. Which sparse format is best for right SpMM (B = A*S)?
//      -> CSC wins, because the dispatch transposes CSC to CSR and uses MKL
//         with SPARSE_OPERATION_NON_TRANSPOSE — the fastest MKL path.
//
//   3. How much does MKL accelerate SpMM over hand-rolled kernels?
//      -> For left SpMM, the benchmark runs CSR through both MKL and the
//         hand-rolled kernel on the same data, giving a direct speedup ratio.
//
//   4. Are left and right SpMM comparable in performance?
//      -> The "SQUARE PROBLEMS" section uses d = m = n so both directions
//         have identical FLOP counts and output sizes. A summary line after
//         each config reports the best-of-each comparison.
//
// NOTE: This benchmarks the SpMM kernel directly (left_spmm / right_spmm),
// not sketch_general. Any overhead from SKOP generation or sketch_general's
// internal dispatch is not captured here. In practice SpMM dominates the
// cost, so these results are a good proxy for sketching performance.
//
// NOTATION:
//   S  - Sparse matrix (m x n)
//   A  - Dense input matrix
//   B  - Dense result matrix
//
//   Left SpMM:  B(m x d) = S(m x n) * A(n x d)
//   Right SpMM: B(d x n) = A(d x m) * S(m x n)
//
//   A "densify + GEMM" reference converts S to dense and calls BLAS GEMM,
//   reporting the densify and GEMM times separately.
//
// USAGE:
//   ./spmm_performance                             # default sweep
//   ./spmm_performance m n d density [num_trials]  # single config
//
//   Default sweep (no arguments):
//     Two sections, 10 trials each:
//       1. SQUARE (d = m = n): sizes 100..2000, fair left-vs-right comparison
//       2. RECTANGULAR: square S with small d, tall S, wide S
//
//   Single config (4+ arguments):
//     One (m, n, d, density) configuration, default 20 trials.
//
// EXAMPLES:
//   env OMP_NUM_THREADS=8 ./spmm_performance
//   env OMP_NUM_THREADS=8 ./spmm_performance 2000 2000 200 0.01
//   env OMP_NUM_THREADS=8 ./spmm_performance 5000 500 200 0.01 10
//
// ============================================================================

#include <RandBLAS.hh>
#include <blas.hh>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Internal headers for sparse dispatch, format conversions, and generation
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"
#include "RandBLAS/sparse_data/random_matrix.hh"

#include "RandBLAS/config.h"

using namespace std::chrono;
using blas::Layout;
using blas::Op;

// Calls the hand-rolled CSR left-multiply kernel directly, bypassing MKL.
// Used to answer question 3 (MKL vs hand-rolled speedup). Replicates the
// beta-scaling and kernel selection logic from spmm_dispatch.hh.
template <typename T, typename sint_t>
void handrolled_left_spmm_csr(
    blas::Layout layout, int64_t d, int64_t n, int64_t m,
    T alpha, const RandBLAS::sparse_data::CSRMatrix<T, sint_t> &A,
    const T *B, int64_t ldb, T beta, T *C, int64_t ldc
) {
    // Apply beta to C (same as dispatch)
    if (layout == Layout::ColMajor) {
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &C[i*ldc]);
    } else {
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &C[i*ldc]);
    }
    if (alpha == (T)0) return;

    // Call the hand-rolled CSR kernel directly (same as dispatch fallback)
    Layout layout_opB = layout;  // opB == NoTrans
    Layout layout_C = layout;
    if (layout_opB == Layout::RowMajor && layout_C == Layout::RowMajor) {
        RandBLAS::sparse_data::csr::apply_csr_left_ikb_p1b_rowmajor(
            alpha, d, n, m, A, B, ldb, C, ldc);
    } else {
        RandBLAS::sparse_data::csr::apply_csr_left_jik_p11(
            alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
    }
}

// Run num_trials repetitions, return {min, median} times in microseconds.
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

// Like run_trials, but times densify and compute phases separately.
// Returns {min_densify, min_compute}.
template <typename DensifyFunc, typename ComputeFunc>
std::pair<long, long> run_split_trials(
    DensifyFunc&& densify, ComputeFunc&& compute, int num_trials
) {
    std::vector<long> t_dens, t_comp;
    t_dens.reserve(num_trials);
    t_comp.reserve(num_trials);

    for (int t = 0; t < num_trials; ++t) {
        auto s1 = steady_clock::now();
        densify();
        auto s2 = steady_clock::now();
        compute();
        auto s3 = steady_clock::now();
        t_dens.push_back(duration_cast<microseconds>(s2 - s1).count());
        t_comp.push_back(duration_cast<microseconds>(s3 - s2).count());
    }

    std::sort(t_dens.begin(), t_dens.end());
    std::sort(t_comp.begin(), t_comp.end());
    return {t_dens[0], t_comp[0]};
}

// Output formatting helpers.
void print_row(const std::string& name, long min_us, long med_us, long baseline) {
    double ratio = (double)min_us / baseline;
    std::cout << "  " << std::setw(24) << std::left << name
              << std::setw(10) << std::right << min_us
              << std::setw(10) << med_us
              << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x\n";
}

void print_densify_row(const std::string& name, long total, long dens, long gemm, long baseline) {
    double ratio = (double)total / baseline;
    std::cout << "  " << std::setw(24) << std::left << name
              << std::setw(10) << std::right << total
              << std::setw(10) << total
              << std::setw(10) << std::fixed << std::setprecision(2) << ratio << "x"
              << "  (densify " << dens << " + GEMM " << gemm << ")\n";
}

// Run one complete benchmark configuration: generate matrices, time all
// format x direction combinations, verify correctness, and print results.
void run_config(int64_t m, int64_t n, int64_t d, double density, int num_trials) {
    using T = double;
    uint64_t seed = 12345;

    // Header
    std::string shape;
    if (m == n && d == m) shape = "all square";
    else if (m == n) shape = "square S";
    else if (m > n) shape = "tall";
    else shape = "wide";

    std::cout << "--- S(" << m << "x" << n << "), d=" << d
              << ", density=" << std::setprecision(4) << density << " (" << shape << ") ---\n";

    // Generate sparse matrices (O(nnz) expected time via geometric skips)
    RandBLAS::sparse_data::CSRMatrix<T> S_csr(m, n);
    RandBLAS::sparse_data::CSCMatrix<T> S_csc(m, n);
    RandBLAS::sparse_data::COOMatrix<T> S_coo(m, n);
    RandBLAS::sparse_data::random_csr(density, S_csr, RandBLAS::RNGState<>(seed));
    RandBLAS::sparse_data::random_csc(density, S_csc, RandBLAS::RNGState<>(seed + 1));
    RandBLAS::sparse_data::random_coo(density, S_coo, RandBLAS::RNGState<>(seed + 2));

    std::cout << "  nnz: CSR=" << S_csr.nnz << " CSC=" << S_csc.nnz
              << " COO=" << S_coo.nnz << ", trials=" << num_trials << "\n\n";

    // Generate dense matrices
    auto state = RandBLAS::RNGState<>(seed + 100);
    std::vector<T> A_left(n * d);
    RandBLAS::DenseDist D_left(n, d);
    state = RandBLAS::fill_dense(D_left, A_left.data(), state);

    std::vector<T> A_right(d * m);
    RandBLAS::DenseDist D_right(d, m);
    state = RandBLAS::fill_dense(D_right, A_right.data(), state);

    // Result and workspace buffers
    std::vector<T> B_left(m * d);
    std::vector<T> B_right(d * n);
    std::vector<T> S_dense(m * n);

    // ---- Left SpMM: B = S * A ----
    auto [min_l_csr, med_l_csr] = run_trials([&]() {
        std::fill(B_left.begin(), B_left.end(), 0.0);
        RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, d, n, 1.0, S_csr, 0, 0, A_left.data(), n, 0.0, B_left.data(), m);
    }, num_trials);

    // Hand-rolled CSR: same data as above, but bypasses MKL (question 3).
    long min_l_csr_hr = 0, med_l_csr_hr = 0;
    #if defined(RandBLAS_HAS_MKL)
    {
        auto [mn, md] = run_trials([&]() {
            std::fill(B_left.begin(), B_left.end(), 0.0);
            handrolled_left_spmm_csr(Layout::ColMajor, m, d, n, 1.0,
                S_csr, A_left.data(), n, 0.0, B_left.data(), m);
        }, num_trials);
        min_l_csr_hr = mn;
        med_l_csr_hr = md;
    }
    #endif

    auto [min_l_csc, med_l_csc] = run_trials([&]() {
        std::fill(B_left.begin(), B_left.end(), 0.0);
        RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, d, n, 1.0, S_csc, 0, 0, A_left.data(), n, 0.0, B_left.data(), m);
    }, num_trials);

    auto [min_l_coo, med_l_coo] = run_trials([&]() {
        std::fill(B_left.begin(), B_left.end(), 0.0);
        RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, d, n, 1.0, S_coo, 0, 0, A_left.data(), n, 0.0, B_left.data(), m);
    }, num_trials);

    // ---- Right SpMM: B = A * S ----
    auto [min_r_csr, med_r_csr] = run_trials([&]() {
        std::fill(B_right.begin(), B_right.end(), 0.0);
        RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, 1.0, A_right.data(), d, S_csr, 0, 0, 0.0, B_right.data(), d);
    }, num_trials);

    auto [min_r_csc, med_r_csc] = run_trials([&]() {
        std::fill(B_right.begin(), B_right.end(), 0.0);
        RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, 1.0, A_right.data(), d, S_csc, 0, 0, 0.0, B_right.data(), d);
    }, num_trials);

    auto [min_r_coo, med_r_coo] = run_trials([&]() {
        std::fill(B_right.begin(), B_right.end(), 0.0);
        RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, 1.0, A_right.data(), d, S_coo, 0, 0, 0.0, B_right.data(), d);
    }, num_trials);

    // ---- Reference: densify + GEMM ----
    auto [min_dens, min_gemm_left] = run_split_trials(
        [&]() { RandBLAS::sparse_data::csr::csr_to_dense(S_csr, Layout::ColMajor, S_dense.data()); },
        [&]() {
            std::fill(B_left.begin(), B_left.end(), 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       m, d, n, 1.0, S_dense.data(), m, A_left.data(), n, 0.0, B_left.data(), m);
        }, num_trials);
    long min_ref_left = min_dens + min_gemm_left;

    auto [min_dens2, min_gemm_right] = run_split_trials(
        [&]() { RandBLAS::sparse_data::csr::csr_to_dense(S_csr, Layout::ColMajor, S_dense.data()); },
        [&]() {
            std::fill(B_right.begin(), B_right.end(), 0.0);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       d, n, m, 1.0, A_right.data(), d, S_dense.data(), m, 0.0, B_right.data(), d);
        }, num_trials);
    long min_ref_right = min_dens + min_gemm_right;

    // ---- Correctness: compare each SpMM path against densify + GEMM ----
    bool all_pass = true;
    int num_checks = 0;

    auto check = [&](const std::string& label, auto& sparse, auto& dense_in, auto& result,
                     int64_t r, int64_t c, int64_t k, bool is_left) {
        // Densify this sparse matrix and compute reference via GEMM
        std::vector<T> S_dens(m * n);
        if constexpr (std::is_same_v<std::decay_t<decltype(sparse)>,
                      RandBLAS::sparse_data::CSRMatrix<T, int64_t>>) {
            RandBLAS::sparse_data::csr::csr_to_dense(sparse, Layout::ColMajor, S_dens.data());
        } else if constexpr (std::is_same_v<std::decay_t<decltype(sparse)>,
                             RandBLAS::sparse_data::CSCMatrix<T, int64_t>>) {
            RandBLAS::sparse_data::csc::csc_to_dense(sparse, Layout::ColMajor, S_dens.data());
        } else {
            RandBLAS::sparse_data::coo::coo_to_dense(sparse, Layout::ColMajor, S_dens.data());
        }

        std::vector<T> ref(r * c, 0.0);
        if (is_left) {
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       r, c, k, 1.0, S_dens.data(), m, dense_in.data(), n, 0.0, ref.data(), m);
        } else {
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       r, c, k, 1.0, dense_in.data(), d, S_dens.data(), m, 0.0, ref.data(), d);
        }

        // Compute via SpMM and compare
        std::fill(result.begin(), result.end(), 0.0);
        if (is_left) {
            RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                r, c, k, 1.0, sparse, 0, 0, dense_in.data(), n, 0.0, result.data(), m);
        } else {
            RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                r, c, k, 1.0, dense_in.data(), d, sparse, 0, 0, 0.0, result.data(), d);
        }

        double maxdiff = 0;
        for (int64_t i = 0; i < r * c; ++i)
            maxdiff = std::max(maxdiff, std::abs(result[i] - ref[i]));
        num_checks++;
        if (maxdiff > 1e-10) {
            std::cout << "  FAIL: " << label << " max|diff|=" << std::scientific << maxdiff << "\n";
            all_pass = false;
        }
    };

    check("left+CSR",  S_csr, A_left,  B_left,  m, d, n, true);
    check("left+CSC",  S_csc, A_left,  B_left,  m, d, n, true);
    check("left+COO",  S_coo, A_left,  B_left,  m, d, n, true);
    check("right+CSR", S_csr, A_right, B_right, d, n, m, false);
    check("right+CSC", S_csc, A_right, B_right, d, n, m, false);
    check("right+COO", S_coo, A_right, B_right, d, n, m, false);

    // Also verify hand-rolled CSR correctness
    #if defined(RandBLAS_HAS_MKL)
    {
        std::vector<T> S_dens(m * n);
        RandBLAS::sparse_data::csr::csr_to_dense(S_csr, Layout::ColMajor, S_dens.data());
        std::vector<T> ref(m * d, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   m, d, n, 1.0, S_dens.data(), m, A_left.data(), n, 0.0, ref.data(), m);

        std::fill(B_left.begin(), B_left.end(), 0.0);
        handrolled_left_spmm_csr(Layout::ColMajor, m, d, n, 1.0,
            S_csr, A_left.data(), n, 0.0, B_left.data(), m);

        double maxdiff = 0;
        for (int64_t i = 0; i < m * d; ++i)
            maxdiff = std::max(maxdiff, std::abs(B_left[i] - ref[i]));
        num_checks++;
        if (maxdiff > 1e-10) {
            std::cout << "  FAIL: left+CSR(hand-rolled) max|diff|=" << std::scientific << maxdiff << "\n";
            all_pass = false;
        }
    }
    #endif

    if (all_pass)
        std::cout << "  Correctness: all " << num_checks << " checks PASS\n";
    std::cout << "\n";

    // ---- Results tables ----
    std::cout << std::fixed << std::setprecision(2);

    // Left SpMM table
    #if defined(RandBLAS_HAS_MKL)
    long bl = std::min({min_l_csr, min_l_csr_hr, min_l_csc, min_l_coo});
    #else
    long bl = std::min({min_l_csr, min_l_csc, min_l_coo});
    #endif

    std::cout << "  LEFT SPMM: B(" << m << "x" << d << ") = S * A\n";
    std::cout << "  " << std::setw(24) << std::left << "Kernel"
              << std::setw(10) << std::right << "Min (us)"
              << std::setw(10) << "Med (us)"
              << std::setw(11) << "vs best" << "\n";
    std::cout << "  " << std::string(55, '-') << "\n";

    #if defined(RandBLAS_HAS_MKL)
    print_row("CSR (MKL)",          min_l_csr,    med_l_csr,    bl);
    print_row("CSR (hand-rolled)",  min_l_csr_hr, med_l_csr_hr, bl);
    print_row("CSC (hand-rolled)",  min_l_csc,    med_l_csc,    bl);
    print_row("COO (MKL)",          min_l_coo,    med_l_coo,    bl);
    #else
    print_row("CSR (hand-rolled)",  min_l_csr, med_l_csr, bl);
    print_row("CSC (hand-rolled)",  min_l_csc, med_l_csc, bl);
    print_row("COO (hand-rolled)",  min_l_coo, med_l_coo, bl);
    #endif
    print_densify_row("densify+GEMM", min_ref_left, min_dens, min_gemm_left, bl);
    std::cout << "\n";

    // Right SpMM table
    long br = std::min({min_r_csr, min_r_csc, min_r_coo});

    std::cout << "  RIGHT SPMM: B(" << d << "x" << n << ") = A * S\n";
    std::cout << "  " << std::setw(24) << std::left << "Kernel"
              << std::setw(10) << std::right << "Min (us)"
              << std::setw(10) << "Med (us)"
              << std::setw(11) << "vs best" << "\n";
    std::cout << "  " << std::string(55, '-') << "\n";

    #if defined(RandBLAS_HAS_MKL)
    print_row("CSR (MKL)",          min_r_csr, med_r_csr, br);
    print_row("CSC (MKL via CSR)",  min_r_csc, med_r_csc, br);
    print_row("COO (MKL)",          min_r_coo, med_r_coo, br);
    #else
    print_row("CSR (hand-rolled)",  min_r_csr, med_r_csr, br);
    print_row("CSC (hand-rolled)",  min_r_csc, med_r_csc, br);
    print_row("COO (hand-rolled)",  min_r_coo, med_r_coo, br);
    #endif
    print_densify_row("densify+GEMM", min_ref_right, min_dens, min_gemm_right, br);
    std::cout << "\n";

    // Summary: best left vs best right
    std::cout << "  SUMMARY: best left " << bl << " us  |  best right " << br << " us  |  ";
    if (bl <= br)
        std::cout << "left " << std::fixed << std::setprecision(2) << (double)br / bl << "x faster\n";
    else
        std::cout << "right " << std::fixed << std::setprecision(2) << (double)bl / br << "x faster\n";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "============================================================================\n";
    std::cout << "SPMM PERFORMANCE BENCHMARK\n";
    std::cout << "============================================================================\n";
#if defined(RandBLAS_HAS_MKL)
    std::cout << "MKL support: ENABLED\n";
#else
    std::cout << "MKL support: DISABLED\n";
#endif
    std::cout << "\n";
    std::cout << "  S is m-by-n (sparse), A and B are dense.\n";
    std::cout << "  Left SpMM:  B(m x d) = S(m x n) * A(n x d)\n";
    std::cout << "  Right SpMM: B(d x n) = A(d x m) * S(m x n)\n\n";

    if (argc >= 5) {
        // Single config mode
        int64_t m = std::atoll(argv[1]);
        int64_t n = std::atoll(argv[2]);
        int64_t d = std::atoll(argv[3]);
        double density = std::atof(argv[4]);
        int num_trials = (argc > 5) ? std::atoi(argv[5]) : 20;
        run_config(m, n, d, density, num_trials);
    } else {
        // Default sweep
        struct Config { int64_t m, n, d; double density; };
        int num_trials = 10;

        // Section 1: Square problems (d = m = n)
        std::vector<Config> square_configs = {
            {  100,   100,   100, 0.01},
            {  200,   200,   200, 0.01},
            {  500,   500,   500, 0.01},
            { 1000,  1000,  1000, 0.01},
            { 2000,  2000,  2000, 0.01},
        };

        std::cout << "=== SQUARE PROBLEMS (d = m = n) ===\n";
        std::cout << num_trials << " trials per config\n\n";
        for (auto& c : square_configs) {
            run_config(c.m, c.n, c.d, c.density, num_trials);
        }

        // Section 2: Rectangular problems
        std::vector<Config> rect_configs = {
            { 2000,  2000,   200, 0.01},
            { 5000,  5000,   500, 0.001},
            { 5000,   500,   500, 0.01},
            {  500,  5000,   500, 0.01},
        };

        std::cout << "=== RECTANGULAR PROBLEMS ===\n";
        std::cout << num_trials << " trials per config\n\n";
        for (auto& c : rect_configs) {
            run_config(c.m, c.n, c.d, c.density, num_trials);
        }
    }

    return 0;
}
