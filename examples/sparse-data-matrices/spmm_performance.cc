// Copyright, 2024. See LICENSE for copyright holder information.
//
// ============================================================================
// SPARSE-DENSE MATRIX MULTIPLICATION PERFORMANCE BENCHMARK
// ============================================================================
//
// PURPOSE:
// This benchmark demonstrates a performance issue with RandBLAS sparse-dense
// matrix multiplication. It compares two approaches:
//
// 1. Direct sparse-dense multiplication using right_spmm
// 2. Densification followed by dense-dense BLAS multiplication
//
// OBSERVED BEHAVIOR:
// For moderate sparsity (density ~0.1), approach (2) is significantly faster
// than approach (1), even though it includes the overhead of densification.
//
// Example results (with OMP_NUM_THREADS=8):
//   - Small matrices (1000×100 @ 1000×100): 11× slower for right_spmm
//   - Medium matrices (1000×100 @ 1000×200): Similar performance issues
//   - Large matrices (5000×500 @ 5000×500): 1.5× slower for right_spmm
//
// USAGE:
//   ./spmm_performance [m] [n] [d] [density] [num_trials]
//
// PARAMETERS:
//   m          - Number of rows in sparse matrix A (default: 1000)
//   n          - Number of columns in sparse matrix A (default: 100)
//   d          - Number of columns in dense matrices B and C (default: 100)
//   density    - Sparsity density (0.0 to 1.0, default: 0.1)
//   num_trials - Number of benchmark trials (default: 5)
//
// OPERATION BENCHMARKED:
//   C = B^T * A
//   where A is sparse (m×n, CSR format), B is dense (m×d), C is result (n×d)
//
// EXAMPLE COMMANDS:
//   # Default parameters (shows ~11× slowdown)
//   OMP_NUM_THREADS=8 ./spmm_performance
//
//   # Medium problem (demonstrates issue)
//   OMP_NUM_THREADS=8 ./spmm_performance 1000 100 200 0.1 3
//
//   # Larger problem (still shows ~1.5× slowdown)
//   OMP_NUM_THREADS=8 ./spmm_performance 5000 500 500 0.1 5
//
//   # Dense case (density 0.5)
//   OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.5 3
//
// PERFORMANCE ANALYSIS:
// The current sparse-dense multiplication implementation needs optimization.
// Potential improvements:
// - Using vectorized operations for dense matrix access
// - Leveraging BLAS routines (dgemv) for sparse row × dense column products
// - Using vendor sparse BLAS libraries (Intel MKL mkl_sparse_d_mm, cuSPARSE)
// - Improving cache utilization through blocking strategies
//
// ROOT CAUSE:
// The bottleneck is in apply_csr_to_vector_from_left_ik() which uses manual
// element-wise loops instead of optimized BLAS routines. See csr_spmm_impl.hh.
//
// ============================================================================

#include <RandBLAS.hh>
#include <blas.hh>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std::chrono;
using blas::Layout;
using blas::Op;

template <typename T>
void sparse_to_dense(const RandBLAS::sparse_data::CSRMatrix<T, int64_t>& A_sp,
                     blas::Layout layout, T* A_dense) {
    int64_t m = A_sp.n_rows;
    int64_t n = A_sp.n_cols;

    // Initialize to zero
    std::fill(A_dense, A_dense + m * n, (T)0.0);

    // Fill from CSR format
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t idx = A_sp.rowptr[i]; idx < A_sp.rowptr[i + 1]; ++idx) {
            int64_t j = A_sp.colidxs[idx];
            T val = A_sp.vals[idx];
            if (layout == Layout::ColMajor) {
                A_dense[j * m + i] = val;
            } else {
                A_dense[i * n + j] = val;
            }
        }
    }
}

template <typename T>
RandBLAS::sparse_data::CSRMatrix<T, int64_t> generate_sparse_gaussian(
    int64_t m, int64_t n, double density, RandBLAS::RNGState<>& state
) {
    using RandBLAS::sparse_data::CSRMatrix;

    // Estimate number of nonzeros
    int64_t nnz_estimate = (int64_t)(m * n * density * 1.2); // 20% buffer

    std::vector<T> vals;
    std::vector<int64_t> colidxs;
    std::vector<int64_t> rowptr(m + 1, 0);

    vals.reserve(nnz_estimate);
    colidxs.reserve(nnz_estimate);

    // Generate sparse matrix row by row
    std::mt19937 rng(static_cast<unsigned>(state.counter[0]));
    std::bernoulli_distribution coin(density);
    std::normal_distribution<double> gauss_dist(0.0, 1.0);

    for (int64_t i = 0; i < m; ++i) {
        rowptr[i] = vals.size();
        for (int64_t j = 0; j < n; ++j) {
            if (coin(rng)) {
                vals.push_back(static_cast<T>(gauss_dist(rng)));
                colidxs.push_back(j);
            }
        }
    }
    rowptr[m] = vals.size();

    int64_t actual_nnz = vals.size();

    // Allocate arrays for CSR matrix
    T* vals_arr = new T[actual_nnz];
    int64_t* colidxs_arr = new int64_t[actual_nnz];
    int64_t* rowptr_arr = new int64_t[m + 1];

    std::copy(vals.begin(), vals.end(), vals_arr);
    std::copy(colidxs.begin(), colidxs.end(), colidxs_arr);
    std::copy(rowptr.begin(), rowptr.end(), rowptr_arr);

    // Create CSR matrix using expert constructor
    CSRMatrix<T, int64_t> A(
        m, n, actual_nnz,
        vals_arr, rowptr_arr, colidxs_arr,
        RandBLAS::sparse_data::IndexBase::Zero
    );
    // Set own_memory to true so the matrix will clean up arrays on destruction
    A.own_memory = true;

    return A;
}

int main(int argc, char** argv) {
    using T = double;

    // Parse command-line arguments or use defaults
    int64_t m = (argc > 1) ? std::atoll(argv[1]) : 1000;
    int64_t n = (argc > 2) ? std::atoll(argv[2]) : 100;
    int64_t d = (argc > 3) ? std::atoll(argv[3]) : 100;
    double density = (argc > 4) ? std::atof(argv[4]) : 0.1;
    int num_trials = (argc > 5) ? std::atoi(argv[5]) : 5;

    std::cout << "\n=== RandBLAS Sparse-Dense Multiplication Performance Benchmark ===\n\n";
    std::cout << "Matrix dimensions:\n";
    std::cout << "  Sparse matrix A: " << m << " × " << n << " (density " << density << ")\n";
    std::cout << "  Dense matrix B:  " << m << " × " << d << "\n";
    std::cout << "  Result matrix C: " << n << " × " << d << "\n";
    std::cout << "  Operation: C = B^T * A  (using right_spmm)\n";
    std::cout << "  Number of trials: " << num_trials << "\n\n";

    // Initialize RNG
    auto state = RandBLAS::RNGState<>();

    // Generate sparse matrix A (m × n, CSR format)
    std::cout << "Generating sparse matrix A ... " << std::flush;
    auto A_sp = generate_sparse_gaussian<T>(m, n, density, state);
    std::cout << "done (actual nnz: " << A_sp.nnz
              << ", actual density: " << (double)A_sp.nnz / (m * n) << ")\n";

    // Generate dense matrix B (m × d)
    std::cout << "Generating dense matrix B ... " << std::flush;
    std::vector<T> B(m * d);
    RandBLAS::DenseDist D(m, d);
    state = RandBLAS::fill_dense(D, B.data(), state);
    std::cout << "done\n\n";

    // Allocate result matrices
    std::vector<T> C1(n * d, 0.0);  // For sparse-dense multiplication
    std::vector<T> C2(n * d, 0.0);  // For densify-then-multiply
    std::vector<T> A_dense(m * n);  // For densification approach

    // =========================================================================
    // Benchmark 1: Direct sparse-dense multiplication (right_spmm)
    // =========================================================================

    std::cout << "Approach 1: Direct sparse-dense multiplication (right_spmm)\n";
    std::cout << "  Operation: C = alpha * B^T * A + beta * C\n";

    std::vector<long> times_spmm;
    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C1.begin(), C1.end(), 0.0);

        auto start = steady_clock::now();

        // C = B^T * A  where B is d×m (transposed), A is m×n (sparse)
        // Result C is d×n
        RandBLAS::sparse_data::right_spmm(
            Layout::ColMajor,  // Layout of B and C
            Op::Trans,         // op(B) = B^T
            Op::NoTrans,       // op(A) = A
            d, n, m,           // C is d×n, op(B) is d×m, op(A) is m×n
            1.0,               // alpha
            B.data(), m,       // B is m×d stored col-major
            A_sp, 0, 0,        // Sparse matrix A
            0.0,               // beta
            C1.data(), d       // C is d×n stored col-major
        );

        auto end = steady_clock::now();
        long duration = duration_cast<microseconds>(end - start).count();
        times_spmm.push_back(duration);

        std::cout << "  Trial " << (trial + 1) << ": " << duration << " μs\n";
    }

    // Compute median time
    std::sort(times_spmm.begin(), times_spmm.end());
    long median_spmm = times_spmm[num_trials / 2];
    std::cout << "  Median time: " << median_spmm << " μs\n\n";

    // =========================================================================
    // Benchmark 2: Densify + BLAS GEMM
    // =========================================================================

    std::cout << "Approach 2: Densify sparse matrix + BLAS GEMM\n";
    std::cout << "  Step 1: Convert sparse A to dense\n";
    std::cout << "  Step 2: C = alpha * B^T * A_dense + beta * C  (using BLAS gemm)\n";

    std::vector<long> times_densify;
    std::vector<long> times_gemm;

    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C2.begin(), C2.end(), 0.0);

        // Step 1: Densification
        auto start_densify = steady_clock::now();
        sparse_to_dense(A_sp, Layout::ColMajor, A_dense.data());
        auto end_densify = steady_clock::now();
        long duration_densify = duration_cast<microseconds>(end_densify - start_densify).count();
        times_densify.push_back(duration_densify);

        // Step 2: Dense GEMM
        auto start_gemm = steady_clock::now();
        blas::gemm(
            Layout::ColMajor,
            Op::Trans,         // op(B) = B^T
            Op::NoTrans,       // op(A) = A
            d, n, m,           // C is d×n, B^T is d×m, A is m×n
            1.0,               // alpha
            B.data(), m,       // B is m×d
            A_dense.data(), m, // A_dense is m×n
            0.0,               // beta
            C2.data(), d       // C is d×n
        );
        auto end_gemm = steady_clock::now();
        long duration_gemm = duration_cast<microseconds>(end_gemm - start_gemm).count();
        times_gemm.push_back(duration_gemm);

        long total = duration_densify + duration_gemm;
        std::cout << "  Trial " << (trial + 1) << ": "
                  << "densify=" << duration_densify << " μs, "
                  << "gemm=" << duration_gemm << " μs, "
                  << "total=" << total << " μs\n";
    }

    // Compute medians
    std::sort(times_densify.begin(), times_densify.end());
    std::sort(times_gemm.begin(), times_gemm.end());
    long median_densify = times_densify[num_trials / 2];
    long median_gemm = times_gemm[num_trials / 2];
    long median_total = median_densify + median_gemm;

    std::cout << "  Median times: "
              << "densify=" << median_densify << " μs, "
              << "gemm=" << median_gemm << " μs, "
              << "total=" << median_total << " μs\n\n";

    // =========================================================================
    // Results summary and comparison
    // =========================================================================

    std::cout << "====================================================================\n";
    std::cout << "RESULTS SUMMARY (median times):\n";
    std::cout << "====================================================================\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Approach 1 (sparse-dense right_spmm):  "
              << std::setw(8) << median_spmm << " μs\n";
    std::cout << "Approach 2 (densify + gemm):           "
              << std::setw(8) << median_total << " μs\n";
    std::cout << "  - Densification overhead:            "
              << std::setw(8) << median_densify << " μs  ("
              << (100.0 * median_densify / median_total) << "%)\n";
    std::cout << "  - BLAS GEMM (dense × dense):         "
              << std::setw(8) << median_gemm << " μs  ("
              << (100.0 * median_gemm / median_total) << "%)\n\n";

    if (median_spmm > median_total) {
        double slowdown = (double)median_spmm / median_total;
        std::cout << "⚠️  PERFORMANCE ISSUE DETECTED:\n";
        std::cout << "    Sparse-dense multiplication is " << slowdown << "× SLOWER\n";
        std::cout << "    than densify+gemm approach!\n\n";
        std::cout << "    Even accounting for densification overhead (" << median_densify
                  << " μs),\n";
        std::cout << "    it is faster to:\n";
        std::cout << "      1. Convert sparse matrix to dense (" << median_densify << " μs)\n";
        std::cout << "      2. Use optimized BLAS gemm (" << median_gemm << " μs)\n";
        std::cout << "      Total: " << median_total << " μs\n";
        std::cout << "    vs. direct sparse-dense multiplication (" << median_spmm << " μs)\n\n";

        std::cout << "RECOMMENDED FIXES:\n";
        std::cout << "  1. Use vectorized operations for dense matrix access\n";
        std::cout << "  2. Leverage BLAS routines (dgemv) per sparse row\n";
        std::cout << "  3. Use vendor sparse BLAS (MKL mkl_sparse_d_mm, cuSPARSE)\n";
        std::cout << "  4. Implement cache-efficient blocking strategies\n";
    } else {
        double speedup = (double)median_total / median_spmm;
        std::cout << "✓  Sparse-dense multiplication is " << speedup << "× faster\n";
        std::cout << "   than densify+gemm (as expected).\n";
    }

    std::cout << "\n====================================================================\n\n";

    // Verify correctness
    double max_diff = 0.0;
    for (size_t i = 0; i < C1.size(); ++i) {
        double diff = std::abs(C1[i] - C2[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Correctness check: max|C1 - C2| = " << max_diff << "\n";
    if (max_diff < 1e-10) {
        std::cout << "✓  Results match (both methods produce the same output)\n\n";
    } else {
        std::cout << "⚠️  Results differ! Check implementation.\n\n";
    }

    return 0;
}
