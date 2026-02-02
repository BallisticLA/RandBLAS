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
// For high density (density ~0.1 = 10%), approach (2) is significantly faster
// than approach (1), even though it includes the overhead of densification.
// Note: 0.1 is very dense for sparse matrices. True sparse matrices typically
// have densities of 1e-2 (1%), 1e-3 (0.1%), or 1e-4 (0.01%).
//
// Example results (with OMP_NUM_THREADS=8, density=0.1):
//   - Small matrices (1000×100): 11× slower for right_spmm
//   - Medium matrices (10000×1000): Performance gap varies with density
//   - Large matrices (100000×5000): 1.5× slower for right_spmm
//
// USAGE:
//   ./spmm_performance [m] [n] [d] [density] [num_trials]
//
// PARAMETERS:
//   m          - Number of rows in sparse matrix A (default: 1000)
//   n          - Number of columns in sparse matrix A (default: 100)
//   d          - Number of columns in dense matrices B and C (default: 100)
//   density    - Sparsity density (0.0 to 1.0, default: 0.01 = 1%)
//   num_trials - Number of benchmark trials (default: 50, reports minimum time)
//
// OPERATION BENCHMARKED:
//   C = B^T * A
//   where A is sparse (m×n, CSR format), B is dense (m×d), C is result (n×d)
//
// EXAMPLE COMMANDS:
//   # Small matrix, realistic sparse density (1%)
//   env OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.01
//
//   # Medium matrix, very sparse (0.1%)
//   env OMP_NUM_THREADS=8 ./spmm_performance 10000 1000 1000 0.001
//
//   # Large matrix, extremely sparse (0.01%)
//   env OMP_NUM_THREADS=8 ./spmm_performance 100000 5000 5000 0.0001
//
//   # Density sweep on small matrix (find crossover point)
//   env OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.1    # 10% (very dense)
//   env OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.01   # 1% (sparse)
//   env OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.001  # 0.1% (very sparse)
//   env OMP_NUM_THREADS=8 ./spmm_performance 1000 100 100 0.0001 # 0.01% (extremely sparse)
//
// ROOT CAUSE:
// right_spmm transposes the CSR matrix, creating a CSC view, then dispatches
// to CSC left-spmm kernel apply_csc_left_jki_p11(). The bottleneck is in
// apply_csc_to_vector_ki() which uses manual element-wise loops with indirect
// indexing instead of optimized BLAS routines. See csc_spmm_impl.hh lines 50-72.
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

// Note: RandBLAS provides csr_to_dense() in sparse_data/csr_matrix.hh

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

    // Create CSR matrix using memory-owning constructor
    CSRMatrix<T, int64_t> A(m, n);
    A.reserve(actual_nnz);
    std::copy(vals.begin(), vals.end(), A.vals);
    std::copy(colidxs.begin(), colidxs.end(), A.colidxs);
    std::copy(rowptr.begin(), rowptr.end(), A.rowptr);

    return A;
}

int main(int argc, char** argv) {
    using T = double;

    // Parse command-line arguments or use defaults
    int64_t m = (argc > 1) ? std::atoll(argv[1]) : 1000;
    int64_t n = (argc > 2) ? std::atoll(argv[2]) : 100;
    int64_t d = (argc > 3) ? std::atoll(argv[3]) : 100;
    double density = (argc > 4) ? std::atof(argv[4]) : 0.01;  // 1% - realistic for sparse matrices
    int num_trials = (argc > 5) ? std::atoi(argv[5]) : 50;

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

    // Compute minimum time (best performance, standard for CPU benchmarks)
    std::sort(times_spmm.begin(), times_spmm.end());
    long min_spmm = times_spmm[0];
    std::cout << "  Minimum time: " << min_spmm << " μs (best of " << num_trials << " trials)\n\n";

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
        RandBLAS::sparse_data::csr::csr_to_dense(A_sp, Layout::ColMajor, A_dense.data());
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

    // Compute minimum times (best performance)
    std::sort(times_densify.begin(), times_densify.end());
    std::sort(times_gemm.begin(), times_gemm.end());
    long min_densify = times_densify[0];
    long min_gemm = times_gemm[0];
    long min_total = min_densify + min_gemm;

    std::cout << "  Minimum times: "
              << "densify=" << min_densify << " μs, "
              << "gemm=" << min_gemm << " μs, "
              << "total=" << min_total << " μs (best of " << num_trials << " trials)\n\n";

    // =========================================================================
    // Results summary and comparison
    // =========================================================================

    std::cout << "====================================================================\n";
    std::cout << "RESULTS SUMMARY (minimum times - best of " << num_trials << " trials):\n";
    std::cout << "====================================================================\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Approach 1 (sparse-dense right_spmm):  "
              << std::setw(8) << min_spmm << " μs\n";
    std::cout << "Approach 2 (densify + gemm):           "
              << std::setw(8) << min_total << " μs\n";
    std::cout << "  - Densification overhead:            "
              << std::setw(8) << min_densify << " μs  ("
              << (100.0 * min_densify / min_total) << "%)\n";
    std::cout << "  - BLAS GEMM (dense × dense):         "
              << std::setw(8) << min_gemm << " μs  ("
              << (100.0 * min_gemm / min_total) << "%)\n\n";

    if (min_spmm > min_total) {
        double slowdown = (double)min_spmm / min_total;
        std::cout << "⚠️  PERFORMANCE ISSUE DETECTED:\n";
        std::cout << "    Sparse-dense multiplication is " << slowdown << "× SLOWER\n";
        std::cout << "    than densify+gemm approach!\n\n";
        std::cout << "    Even accounting for densification overhead (" << min_densify
                  << " μs),\n";
        std::cout << "    it is faster to:\n";
        std::cout << "      1. Convert sparse matrix to dense (" << min_densify << " μs)\n";
        std::cout << "      2. Use optimized BLAS gemm (" << min_gemm << " μs)\n";
        std::cout << "      Total: " << min_total << " μs\n";
        std::cout << "    vs. direct sparse-dense multiplication (" << min_spmm << " μs)\n";
    } else {
        double speedup = (double)min_total / min_spmm;
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
