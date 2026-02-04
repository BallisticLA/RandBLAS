// Copyright, 2026. See LICENSE for copyright holder information.
//
// ============================================================================
// SPMM IMPLEMENTATION COMPARISON BENCHMARK: Hand-Rolled vs MKL
// ============================================================================
//
// PURPOSE:
// This benchmark compares the performance of RandBLAS's hand-rolled sparse-dense
// matrix multiplication kernels against Intel MKL's optimized sparse BLAS routines.
// It tests ALL combinations of sparse format (CSR/CSC) and operation direction
// (left_spmm/right_spmm) to show the impact of format choice on MKL acceleration.
//
// BACKGROUND:
// RandBLAS originally used hand-rolled kernels for sparse-dense multiplication.
// These kernels were found to be significantly slower than densify+GEMM for
// moderate densities. MKL sparse BLAS support was added to address this.
//
// MKL FORMAT SUPPORT:
// - MKL supports CSR and COO formats directly
// - MKL does NOT support CSC format (returns SPARSE_STATUS_NOT_SUPPORTED)
//
// COMBINATIONS TESTED:
// 1. left_spmm  + CSR → MKL handles directly          (FAST)
// 2. left_spmm  + CSC → MKL can't handle, hand-rolled (SLOW)
// 3. right_spmm + CSR → transpose→CSC, MKL can't handle (SLOW)
// 4. right_spmm + CSC → transpose→CSR, MKL handles    (FAST)
//
// USAGE:
//   ./spmm_mkl_comparison [m] [n] [d] [density] [num_trials]
//
// PARAMETERS:
//   m          - Rows of sparse matrix A (default: 5000)
//   n          - Columns of sparse matrix A (default: 500)
//   d          - Columns of dense matrix B / result dimension (default: 500)
//   density    - Sparsity density (0.0 to 1.0, default: 0.01 = 1%)
//   num_trials - Number of benchmark trials (default: 20, reports minimum time)
//
// EXAMPLE COMMANDS:
//   # Default (5000x500 sparse, 1% density)
//   env OMP_NUM_THREADS=8 ./spmm_mkl_comparison
//
//   # Scaling study
//   for size in 1000 2000 5000 10000; do
//     env OMP_NUM_THREADS=8 ./spmm_mkl_comparison $size $((size/10)) $((size/10)) 0.01 10
//   done
//
// ============================================================================

#include <RandBLAS.hh>
#include <blas.hh>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

// Include internal headers for direct kernel access
#include "RandBLAS/sparse_data/csr_spmm_impl.hh"
#include "RandBLAS/sparse_data/csc_spmm_impl.hh"
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/spmm_dispatch.hh"

#include "RandBLAS/config.h"
#if defined(RandBLAS_HAS_MKL)
#include "RandBLAS/sparse_data/mkl_spmm_impl.hh"
#endif

using namespace std::chrono;
using blas::Layout;
using blas::Op;

// ============================================================================
// Helper: Generate sparse Gaussian matrix in CSR format
// ============================================================================
template <typename T>
RandBLAS::sparse_data::CSRMatrix<T, int64_t> generate_sparse_csr(
    int64_t m, int64_t n, double density, uint64_t seed
) {
    using RandBLAS::sparse_data::CSRMatrix;

    std::vector<T> vals;
    std::vector<int64_t> colidxs;
    std::vector<int64_t> rowptr(m + 1, 0);

    std::mt19937 rng(seed);
    std::bernoulli_distribution coin(density);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (int64_t i = 0; i < m; ++i) {
        rowptr[i] = vals.size();
        for (int64_t j = 0; j < n; ++j) {
            if (coin(rng)) {
                vals.push_back(static_cast<T>(gauss(rng)));
                colidxs.push_back(j);
            }
        }
    }
    rowptr[m] = vals.size();

    CSRMatrix<T, int64_t> A(m, n);
    A.reserve(vals.size());
    std::copy(vals.begin(), vals.end(), A.vals);
    std::copy(colidxs.begin(), colidxs.end(), A.colidxs);
    std::copy(rowptr.begin(), rowptr.end(), A.rowptr);

    return A;
}

// ============================================================================
// Helper: Generate sparse Gaussian matrix in CSC format
// ============================================================================
template <typename T>
RandBLAS::sparse_data::CSCMatrix<T, int64_t> generate_sparse_csc(
    int64_t m, int64_t n, double density, uint64_t seed
) {
    using RandBLAS::sparse_data::CSCMatrix;

    std::vector<T> vals;
    std::vector<int64_t> rowidxs;
    std::vector<int64_t> colptr(n + 1, 0);

    std::mt19937 rng(seed);
    std::bernoulli_distribution coin(density);
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (int64_t j = 0; j < n; ++j) {
        colptr[j] = vals.size();
        for (int64_t i = 0; i < m; ++i) {
            if (coin(rng)) {
                vals.push_back(static_cast<T>(gauss(rng)));
                rowidxs.push_back(i);
            }
        }
    }
    colptr[n] = vals.size();

    CSCMatrix<T, int64_t> A(m, n);
    A.reserve(vals.size());
    std::copy(vals.begin(), vals.end(), A.vals);
    std::copy(rowidxs.begin(), rowidxs.end(), A.rowidxs);
    std::copy(colptr.begin(), colptr.end(), A.colptr);

    return A;
}

// ============================================================================
// Benchmark result structure
// ============================================================================
struct BenchResult {
    std::string name;
    std::string mkl_status;  // "MKL", "hand-rolled", or "N/A"
    long min_us;
    long median_us;
    int64_t flops;           // Theoretical FLOPs for this operation
    int group;               // 0 = left_spmm, 1 = right_spmm, 2 = reference
};

// ============================================================================
// Run benchmark trials and return min/median times
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
        std::cout << "." << std::flush;
    }
    std::cout << "\n";

    std::sort(times.begin(), times.end());
    return {times[0], times[num_trials / 2]};
}

// ============================================================================
// Main benchmark
// ============================================================================
int main(int argc, char** argv) {
    using T = double;

    // Parse command-line arguments
    int64_t m = (argc > 1) ? std::atoll(argv[1]) : 5000;
    int64_t n = (argc > 2) ? std::atoll(argv[2]) : 500;
    int64_t d = (argc > 3) ? std::atoll(argv[3]) : 500;
    double density = (argc > 4) ? std::atof(argv[4]) : 0.01;
    int num_trials = (argc > 5) ? std::atoi(argv[5]) : 20;

    std::cout << "\n";
    std::cout << "============================================================================\n";
    std::cout << "SPMM FORMAT COMPARISON: Impact of CSR vs CSC on MKL Acceleration\n";
    std::cout << "============================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Sparse matrix dimensions: " << m << " x " << n << "\n";
    std::cout << "  Density: " << density << " (" << (density * 100) << "%)\n";
    std::cout << "  Dense matrix columns (d): " << d << "\n";
    std::cout << "  Number of trials: " << num_trials << "\n";
#if defined(RandBLAS_HAS_MKL)
    std::cout << "  MKL support: ENABLED\n";
#else
    std::cout << "  MKL support: DISABLED\n";
#endif
    std::cout << "\n";

    std::cout << "MKL Acceleration Rules:\n";
    std::cout << "  - MKL supports CSR format directly\n";
    std::cout << "  - MKL does NOT support CSC format\n";
    std::cout << "  - right_spmm internally transposes the sparse matrix\n";
    std::cout << "    (CSR→CSC or CSC→CSR before calling left_spmm)\n";
    std::cout << "\n";

    // Generate test matrices (use same seed for fair comparison)
    uint64_t seed = 12345;

    std::cout << "Generating sparse matrices...\n";
    auto A_csr = generate_sparse_csr<T>(m, n, density, seed);
    auto A_csc = generate_sparse_csc<T>(m, n, density, seed + 1);
    std::cout << "  CSR: " << m << "x" << n << ", nnz=" << A_csr.nnz << "\n";
    std::cout << "  CSC: " << m << "x" << n << ", nnz=" << A_csc.nnz << "\n\n";

    // Generate dense matrices
    auto state = RandBLAS::RNGState<>(seed + 100);

    // For left_spmm: C = A * B, where A is m×n sparse, B is n×d dense, C is m×d
    std::vector<T> B_left(n * d);
    RandBLAS::DenseDist D_left(n, d);
    state = RandBLAS::fill_dense(D_left, B_left.data(), state);

    // For right_spmm: C = B * A, where B is d×m dense, A is m×n sparse, C is d×n
    std::vector<T> B_right(d * m);
    RandBLAS::DenseDist D_right(d, m);
    state = RandBLAS::fill_dense(D_right, B_right.data(), state);

    // Result matrices
    std::vector<T> C_left(m * d);
    std::vector<T> C_right(d * n);

    std::vector<BenchResult> results;

    // Theoretical FLOPs for sparse-dense multiply:
    // Each non-zero in sparse matrix contributes to 'd' multiply-adds
    // FLOPs = 2 * nnz * (dense matrix columns in output direction)
    //
    // For left_spmm:  C(m×d) = A(m×n) × B(n×d), FLOPs = 2 * nnz * d
    // For right_spmm: C(d×n) = B(d×m) × A(m×n), FLOPs = 2 * nnz * d
    //
    // Note: Both operations have the same theoretical FLOPs when d is the same,
    // but different memory access patterns affect actual performance.
    int64_t flops_left  = 2 * A_csr.nnz * d;  // For left_spmm
    int64_t flops_right = 2 * A_csr.nnz * d;  // For right_spmm
    int64_t flops_gemm_left  = 2 * m * n * d; // Dense GEMM for left
    int64_t flops_gemm_right = 2 * d * m * n; // Dense GEMM for right

    std::cout << "Theoretical FLOPs:\n";
    std::cout << "  left_spmm:  2 * nnz * d = 2 * " << A_csr.nnz << " * " << d
              << " = " << flops_left / 1e6 << " MFLOPs\n";
    std::cout << "  right_spmm: 2 * nnz * d = 2 * " << A_csr.nnz << " * " << d
              << " = " << flops_right / 1e6 << " MFLOPs\n";
    std::cout << "  dense GEMM: 2 * m * n * d = 2 * " << m << " * " << n << " * " << d
              << " = " << flops_gemm_left / 1e6 << " MFLOPs\n\n";

    // =========================================================================
    // PART 1: left_spmm (C = A * B)
    // =========================================================================
    std::cout << "============================================================================\n";
    std::cout << "PART 1: left_spmm (C = A * B)\n";
    std::cout << "  A is " << m << "x" << n << " sparse, B is " << n << "x" << d << " dense\n";
    std::cout << "============================================================================\n\n";

    // 1a. left_spmm + CSR (MKL handles directly)
    std::cout << "1a. left_spmm + CSR format\n";
    std::cout << "    Expected: MKL handles directly (FAST)\n    ";
    auto [min_left_csr, med_left_csr] = run_trials([&]() {
        std::fill(C_left.begin(), C_left.end(), 0.0);
        RandBLAS::sparse_data::left_spmm(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, d, n, 1.0, A_csr, 0, 0, B_left.data(), n, 0.0, C_left.data(), m
        );
    }, num_trials);
    results.push_back({"left_spmm + CSR", "MKL", min_left_csr, med_left_csr, flops_left, 0});
    std::cout << "    Min: " << min_left_csr << " us, Median: " << med_left_csr << " us\n\n";

    // 1b. left_spmm + CSC (MKL can't handle, falls back to hand-rolled)
    std::cout << "1b. left_spmm + CSC format\n";
    std::cout << "    Expected: MKL can't handle CSC, uses hand-rolled (SLOW)\n    ";
    auto [min_left_csc, med_left_csc] = run_trials([&]() {
        std::fill(C_left.begin(), C_left.end(), 0.0);
        RandBLAS::sparse_data::left_spmm(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, d, n, 1.0, A_csc, 0, 0, B_left.data(), n, 0.0, C_left.data(), m
        );
    }, num_trials);
    results.push_back({"left_spmm + CSC", "hand-rolled", min_left_csc, med_left_csc, flops_left, 0});
    std::cout << "    Min: " << min_left_csc << " us, Median: " << med_left_csc << " us\n\n";

    // =========================================================================
    // PART 2: right_spmm (C = B * A)
    // =========================================================================
    std::cout << "============================================================================\n";
    std::cout << "PART 2: right_spmm (C = B * A)\n";
    std::cout << "  B is " << d << "x" << m << " dense, A is " << m << "x" << n << " sparse\n";
    std::cout << "============================================================================\n\n";

    // 2a. right_spmm + CSR (internally transposes to CSC, MKL can't handle)
    std::cout << "2a. right_spmm + CSR format\n";
    std::cout << "    Expected: Transpose CSR→CSC, MKL can't handle CSC (SLOW)\n    ";
    auto [min_right_csr, med_right_csr] = run_trials([&]() {
        std::fill(C_right.begin(), C_right.end(), 0.0);
        RandBLAS::sparse_data::right_spmm(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, 1.0, B_right.data(), d, A_csr, 0, 0, 0.0, C_right.data(), d
        );
    }, num_trials);
    results.push_back({"right_spmm + CSR", "hand-rolled", min_right_csr, med_right_csr, flops_right, 1});
    std::cout << "    Min: " << min_right_csr << " us, Median: " << med_right_csr << " us\n\n";

    // 2b. right_spmm + CSC (internally transposes to CSR, MKL handles)
    std::cout << "2b. right_spmm + CSC format\n";
    std::cout << "    Expected: Transpose CSC→CSR, MKL handles CSR (FAST)\n    ";
    auto [min_right_csc, med_right_csc] = run_trials([&]() {
        std::fill(C_right.begin(), C_right.end(), 0.0);
        RandBLAS::sparse_data::right_spmm(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, 1.0, B_right.data(), d, A_csc, 0, 0, 0.0, C_right.data(), d
        );
    }, num_trials);
    results.push_back({"right_spmm + CSC", "MKL", min_right_csc, med_right_csc, flops_right, 1});
    std::cout << "    Min: " << min_right_csc << " us, Median: " << med_right_csc << " us\n\n";

    // =========================================================================
    // PART 3: Reference (densify + GEMM)
    // =========================================================================
    std::cout << "============================================================================\n";
    std::cout << "PART 3: Reference (densify + GEMM)\n";
    std::cout << "============================================================================\n\n";

    std::vector<T> A_dense(m * n);

    std::cout << "3a. left_spmm reference (densify CSR + GEMM)\n    ";
    auto [min_ref_left, med_ref_left] = run_trials([&]() {
        RandBLAS::sparse_data::csr::csr_to_dense(A_csr, Layout::ColMajor, A_dense.data());
        std::fill(C_left.begin(), C_left.end(), 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   m, d, n, 1.0, A_dense.data(), m, B_left.data(), n, 0.0, C_left.data(), m);
    }, num_trials);
    results.push_back({"densify + GEMM (left)", "BLAS", min_ref_left, med_ref_left, flops_gemm_left, 2});
    std::cout << "    Min: " << min_ref_left << " us, Median: " << med_ref_left << " us\n\n";

    std::cout << "3b. right_spmm reference (densify CSR + GEMM)\n    ";
    auto [min_ref_right, med_ref_right] = run_trials([&]() {
        RandBLAS::sparse_data::csr::csr_to_dense(A_csr, Layout::ColMajor, A_dense.data());
        std::fill(C_right.begin(), C_right.end(), 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   d, n, m, 1.0, B_right.data(), d, A_dense.data(), m, 0.0, C_right.data(), d);
    }, num_trials);
    results.push_back({"densify + GEMM (right)", "BLAS", min_ref_right, med_ref_right, flops_gemm_right, 2});
    std::cout << "    Min: " << min_ref_right << " us, Median: " << med_ref_right << " us\n\n";

    // =========================================================================
    // CORRECTNESS VERIFICATION
    // =========================================================================
    std::cout << "============================================================================\n";
    std::cout << "CORRECTNESS VERIFICATION\n";
    std::cout << "============================================================================\n\n";

    // For CSR-based tests, use CSR dense reference
    RandBLAS::sparse_data::csr::csr_to_dense(A_csr, Layout::ColMajor, A_dense.data());
    std::vector<T> C_ref_csr_left(m * d, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, d, n, 1.0, A_dense.data(), m, B_left.data(), n, 0.0, C_ref_csr_left.data(), m);

    // Test left_spmm + CSR
    std::fill(C_left.begin(), C_left.end(), 0.0);
    RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                      m, d, n, 1.0, A_csr, 0, 0, B_left.data(), n, 0.0, C_left.data(), m);
    double diff1 = 0;
    for (size_t i = 0; i < C_left.size(); ++i) diff1 = std::max(diff1, std::abs(C_left[i] - C_ref_csr_left[i]));
    std::cout << "left_spmm + CSR:  max|diff| = " << std::scientific << diff1 << (diff1 < 1e-10 ? "  PASS\n" : "  FAIL\n");

    // For CSC-based tests, use CSC dense reference
    std::vector<T> A_csc_dense(m * n);
    RandBLAS::sparse_data::csc::csc_to_dense(A_csc, Layout::ColMajor, A_csc_dense.data());
    std::vector<T> C_ref_csc_left(m * d, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, d, n, 1.0, A_csc_dense.data(), m, B_left.data(), n, 0.0, C_ref_csc_left.data(), m);

    // Test left_spmm + CSC
    std::fill(C_left.begin(), C_left.end(), 0.0);
    RandBLAS::sparse_data::left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                      m, d, n, 1.0, A_csc, 0, 0, B_left.data(), n, 0.0, C_left.data(), m);
    double diff2 = 0;
    for (size_t i = 0; i < C_left.size(); ++i) diff2 = std::max(diff2, std::abs(C_left[i] - C_ref_csc_left[i]));
    std::cout << "left_spmm + CSC:  max|diff| = " << std::scientific << diff2 << (diff2 < 1e-10 ? "  PASS\n" : "  FAIL\n");

    // Compute reference for right_spmm with CSR
    std::vector<T> C_ref_csr_right(d * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               d, n, m, 1.0, B_right.data(), d, A_dense.data(), m, 0.0, C_ref_csr_right.data(), d);

    // Test right_spmm + CSR
    std::fill(C_right.begin(), C_right.end(), 0.0);
    RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                       d, n, m, 1.0, B_right.data(), d, A_csr, 0, 0, 0.0, C_right.data(), d);
    double diff3 = 0;
    for (size_t i = 0; i < C_right.size(); ++i) diff3 = std::max(diff3, std::abs(C_right[i] - C_ref_csr_right[i]));
    std::cout << "right_spmm + CSR: max|diff| = " << std::scientific << diff3 << (diff3 < 1e-10 ? "  PASS\n" : "  FAIL\n");

    // Compute reference for right_spmm with CSC
    std::vector<T> C_ref_csc_right(d * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               d, n, m, 1.0, B_right.data(), d, A_csc_dense.data(), m, 0.0, C_ref_csc_right.data(), d);

    // Test right_spmm + CSC
    std::fill(C_right.begin(), C_right.end(), 0.0);
    RandBLAS::sparse_data::right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                       d, n, m, 1.0, B_right.data(), d, A_csc, 0, 0, 0.0, C_right.data(), d);
    double diff4 = 0;
    for (size_t i = 0; i < C_right.size(); ++i) diff4 = std::max(diff4, std::abs(C_right[i] - C_ref_csc_right[i]));
    std::cout << "right_spmm + CSC: max|diff| = " << std::scientific << diff4 << (diff4 < 1e-10 ? "  PASS\n" : "  FAIL\n");

    // =========================================================================
    // RESULTS SUMMARY
    // =========================================================================
    std::cout << "\n";
    std::cout << "============================================================================\n";
    std::cout << "RESULTS SUMMARY\n";
    std::cout << "============================================================================\n\n";

    std::cout << std::fixed << std::setprecision(2);

    // Helper lambda for printing a result row
    auto print_row = [](const std::string& format, const std::string& mkl_status,
                        const std::string& backend, long min_us, long med_us, long baseline) {
        double speedup = (double)baseline / min_us;
        std::cout << std::setw(10) << std::left << format
                  << std::setw(18) << std::right << mkl_status
                  << std::setw(14) << backend
                  << std::setw(12) << min_us
                  << std::setw(12) << med_us
                  << std::setw(11) << std::setprecision(2) << speedup << "x\n";
    };

    // -------------------------------------------------------------------------
    // LEFT SPMM GROUP: C(m×d) = A(m×n) × B(n×d)
    // -------------------------------------------------------------------------
    std::cout << "LEFT SPMM: C(" << m << "x" << d << ") = A(" << m << "x" << n
              << ") * B(" << n << "x" << d << ")\n";
    std::cout << "  Sparse FLOPs: 2*nnz*d = 2*" << A_csr.nnz << "*" << d << "\n";
    std::cout << "  Dense FLOPs:  2*m*n*d = 2*" << m << "*" << n << "*" << d
              << " (" << std::setprecision(0) << (1.0/density) << "x more)\n\n";

    long baseline_left = std::min({min_left_csr, min_left_csc, min_ref_left});

    std::cout << std::setw(10) << std::left << "Format"
              << std::setw(18) << std::right << "MKL Applicable?"
              << std::setw(14) << "Backend"
              << std::setw(12) << "Min (us)"
              << std::setw(12) << "Median (us)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(88, '-') << "\n";

    print_row("CSR", "Yes", "MKL", min_left_csr, med_left_csr, baseline_left);
    print_row("CSC", "No (unsupported)", "hand-rolled", min_left_csc, med_left_csc, baseline_left);
    print_row("dense", "N/A", "BLAS GEMM", min_ref_left, med_ref_left, baseline_left);

    std::cout << "\n";

    // -------------------------------------------------------------------------
    // RIGHT SPMM GROUP: C(d×n) = B(d×m) × A(m×n)
    // -------------------------------------------------------------------------
    std::cout << "RIGHT SPMM: C(" << d << "x" << n << ") = B(" << d << "x" << m
              << ") * A(" << m << "x" << n << ")\n";
    std::cout << "  Sparse FLOPs: 2*nnz*d = 2*" << A_csc.nnz << "*" << d << "\n";
    std::cout << "  Dense FLOPs:  2*d*m*n = 2*" << d << "*" << m << "*" << n
              << " (" << std::setprecision(0) << (1.0/density) << "x more)\n";
    std::cout << "  Note: right_spmm internally transposes sparse matrix (CSR<->CSC)\n\n";

    long baseline_right = std::min({min_right_csr, min_right_csc, min_ref_right});

    std::cout << std::setw(10) << std::left << "Format"
              << std::setw(18) << std::right << "MKL Applicable?"
              << std::setw(14) << "Backend"
              << std::setw(12) << "Min (us)"
              << std::setw(12) << "Median (us)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(88, '-') << "\n";

    print_row("CSR", "No (becomes CSC)", "hand-rolled", min_right_csr, med_right_csr, baseline_right);
    print_row("CSC", "Yes (becomes CSR)", "MKL", min_right_csc, med_right_csc, baseline_right);
    print_row("dense", "N/A", "BLAS GEMM", min_ref_right, med_ref_right, baseline_right);

    std::cout << "\n";

    // -------------------------------------------------------------------------
    // KEY INSIGHTS
    // -------------------------------------------------------------------------
    std::cout << "============================================================================\n";
    std::cout << "KEY INSIGHTS\n";
    std::cout << "============================================================================\n\n";

    std::cout << "MKL sparse BLAS only supports CSR format (not CSC).\n";
    std::cout << "right_spmm internally transposes the sparse matrix before computing.\n\n";

    double mkl_speedup_left = (double)min_left_csc / min_left_csr;
    double mkl_speedup_right = (double)min_right_csr / min_right_csc;

    std::cout << "MKL vs hand-rolled speedup:\n";
    std::cout << "  - left_spmm:  MKL is " << std::setprecision(1) << mkl_speedup_left
              << "x faster than hand-rolled\n";
    std::cout << "  - right_spmm: MKL is " << std::setprecision(1) << mkl_speedup_right
              << "x faster than hand-rolled\n\n";

    std::cout << "RECOMMENDATION:\n";
    std::cout << "  Use MKL when possible. To enable MKL:\n";
    std::cout << "  - left_spmm:  store sparse matrix in CSR format\n";
    std::cout << "  - right_spmm: store sparse matrix in CSC format\n\n";

    return 0;
}
