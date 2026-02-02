// Copyright, 2024. See LICENSE for copyright holder information.
//
// ============================================================================
// SPARSE-DENSE MATRIX MULTIPLICATION PERFORMANCE BENCHMARK
// ============================================================================
//
// PURPOSE:
// This benchmark demonstrates a performance issue with RandBLAS sparse-dense
// matrix multiplication. It tests both multiplication directions:
//
// DENSE × SPARSE (C = B^T * A):
//   1a. Direct multiplication using right_spmm
//   1b. Densification followed by BLAS GEMM
//
// SPARSE × DENSE (C = A * B):
//   2a. Direct multiplication using left_spmm
//   2b. Densification followed by BLAS GEMM
//
// OBSERVED BEHAVIOR:
// For certain densities and matrix sizes, densification + GEMM can be faster
// than direct sparse-dense multiplication, even with densification overhead.
// Note: True sparse matrices typically have densities of 1e-2 (1%), 1e-3 (0.1%),
// or 1e-4 (0.01%). Density of 0.1 (10%) is very dense for sparse matrices.
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
// OPERATIONS BENCHMARKED:
//   1. Dense × Sparse: C = B^T * A
//      where A is sparse (m×n, CSR), B is dense (m×d), C is result (d×n)
//   2. Sparse × Dense: C = A * B2
//      where A is sparse (m×n, CSR), B2 is dense (n×d), C is result (m×d)
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
// Both right_spmm (dense×sparse) and left_spmm (sparse×dense) can dispatch to
// CSC kernels. right_spmm transposes the CSR matrix to CSC, then dispatches to
// apply_csc_left_jki_p11(). The bottleneck is apply_csc_to_vector_ki() which
// uses manual element-wise loops with indirect indexing instead of optimized
// BLAS routines. See csc_spmm_impl.hh lines 50-72.
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
    std::cout << "  Number of trials: " << num_trials << "\n\n";
    std::cout << "Testing both directions:\n";
    std::cout << "  1. Dense × Sparse: C1 = B^T * A  (B is " << m << "×" << d << ", result is " << d << "×" << n << ")\n";
    std::cout << "  2. Sparse × Dense: C2 = A * B2   (B2 is " << n << "×" << d << ", result is " << m << "×" << d << ")\n\n";

    // Initialize RNG
    auto state = RandBLAS::RNGState<>();

    // Generate sparse matrix A (m × n, CSR format)
    std::cout << "Generating sparse matrix A ... " << std::flush;
    auto A_sp = generate_sparse_gaussian<T>(m, n, density, state);
    std::cout << "done (actual nnz: " << A_sp.nnz
              << ", actual density: " << (double)A_sp.nnz / (m * n) << ")\n";

    // Generate dense matrix B (m × d) for dense × sparse (B^T * A)
    std::cout << "Generating dense matrix B (m×d) for dense×sparse ... " << std::flush;
    std::vector<T> B(m * d);
    RandBLAS::DenseDist D(m, d);
    state = RandBLAS::fill_dense(D, B.data(), state);
    std::cout << "done\n";

    // Generate dense matrix B2 (n × d) for sparse × dense (A * B2)
    std::cout << "Generating dense matrix B2 (n×d) for sparse×dense ... " << std::flush;
    std::vector<T> B2(n * d);
    RandBLAS::DenseDist D2(n, d);
    state = RandBLAS::fill_dense(D2, B2.data(), state);
    std::cout << "done\n\n";

    // Allocate result matrices
    std::vector<T> C_dense_sparse_spmm(d * n, 0.0);      // For dense×sparse via right_spmm
    std::vector<T> C_dense_sparse_densify(d * n, 0.0);   // For dense×sparse via densify+gemm
    std::vector<T> C_sparse_dense_spmm(m * d, 0.0);      // For sparse×dense via left_spmm
    std::vector<T> C_sparse_dense_densify(m * d, 0.0);   // For sparse×dense via densify+gemm
    std::vector<T> A_dense(m * n);                       // For densification approach

    // =========================================================================
    // Benchmark 1: Dense × Sparse (right_spmm)
    // =========================================================================

    std::cout << "=== DENSE × SPARSE BENCHMARKS ===\n\n";
    std::cout << "Approach 1a: Direct dense×sparse multiplication (right_spmm)\n";
    std::cout << "  Operation: C = B^T * A  (B is dense m×d, A is sparse m×n)\n";

    std::vector<long> times_dense_sparse_spmm;
    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C_dense_sparse_spmm.begin(), C_dense_sparse_spmm.end(), 0.0);

        auto start = steady_clock::now();

        // C = B^T * A  where B is m×d, A is m×n (sparse)
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
            C_dense_sparse_spmm.data(), d  // C is d×n stored col-major
        );

        auto end = steady_clock::now();
        long duration = duration_cast<microseconds>(end - start).count();
        times_dense_sparse_spmm.push_back(duration);

        std::cout << "  Trial " << (trial + 1) << ": " << duration << " μs\n";
    }

    // Compute minimum time (best performance, standard for CPU benchmarks)
    std::sort(times_dense_sparse_spmm.begin(), times_dense_sparse_spmm.end());
    long min_dense_sparse_spmm = times_dense_sparse_spmm[0];
    std::cout << "  Minimum time: " << min_dense_sparse_spmm << " μs (best of " << num_trials << " trials)\n\n";

    // =========================================================================
    // Benchmark 2: Dense × Sparse via Densify + BLAS GEMM
    // =========================================================================

    std::cout << "Approach 1b: Densify sparse matrix + BLAS GEMM\n";
    std::cout << "  Step 1: Convert sparse A to dense\n";
    std::cout << "  Step 2: C = B^T * A_dense  (using BLAS gemm)\n";

    std::vector<long> times_dense_sparse_densify;
    std::vector<long> times_dense_sparse_gemm;

    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C_dense_sparse_densify.begin(), C_dense_sparse_densify.end(), 0.0);

        // Step 1: Densification
        auto start_densify = steady_clock::now();
        RandBLAS::sparse_data::csr::csr_to_dense(A_sp, Layout::ColMajor, A_dense.data());
        auto end_densify = steady_clock::now();
        long duration_densify = duration_cast<microseconds>(end_densify - start_densify).count();
        times_dense_sparse_densify.push_back(duration_densify);

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
            C_dense_sparse_densify.data(), d  // C is d×n
        );
        auto end_gemm = steady_clock::now();
        long duration_gemm = duration_cast<microseconds>(end_gemm - start_gemm).count();
        times_dense_sparse_gemm.push_back(duration_gemm);

        long total = duration_densify + duration_gemm;
        std::cout << "  Trial " << (trial + 1) << ": "
                  << "densify=" << duration_densify << " μs, "
                  << "gemm=" << duration_gemm << " μs, "
                  << "total=" << total << " μs\n";
    }

    // Compute minimum times (best performance)
    std::sort(times_dense_sparse_densify.begin(), times_dense_sparse_densify.end());
    std::sort(times_dense_sparse_gemm.begin(), times_dense_sparse_gemm.end());
    long min_dense_sparse_densify = times_dense_sparse_densify[0];
    long min_dense_sparse_gemm = times_dense_sparse_gemm[0];
    long min_dense_sparse_total = min_dense_sparse_densify + min_dense_sparse_gemm;

    std::cout << "  Minimum times: "
              << "densify=" << min_dense_sparse_densify << " μs, "
              << "gemm=" << min_dense_sparse_gemm << " μs, "
              << "total=" << min_dense_sparse_total << " μs (best of " << num_trials << " trials)\n\n";

    // =========================================================================
    // Benchmark 3: Sparse × Dense (left_spmm)
    // =========================================================================

    std::cout << "=== SPARSE × DENSE BENCHMARKS ===\n\n";
    std::cout << "Approach 2a: Direct sparse×dense multiplication (left_spmm)\n";
    std::cout << "  Operation: C = A * B2  (A is sparse m×n, B2 is dense n×d)\n";

    std::vector<long> times_sparse_dense_spmm;
    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C_sparse_dense_spmm.begin(), C_sparse_dense_spmm.end(), 0.0);

        auto start = steady_clock::now();

        // C = A * B2  where A is m×n (sparse), B2 is n×d
        // Result C is m×d
        RandBLAS::sparse_data::left_spmm(
            Layout::ColMajor,  // Layout of B2 and C
            Op::NoTrans,       // op(A) = A
            Op::NoTrans,       // op(B2) = B2
            m, d, n,           // C is m×d, op(A) is m×n, op(B2) is n×d
            1.0,               // alpha
            A_sp, 0, 0,        // Sparse matrix A
            B2.data(), n,      // B2 is n×d stored col-major
            0.0,               // beta
            C_sparse_dense_spmm.data(), m  // C is m×d stored col-major
        );

        auto end = steady_clock::now();
        long duration = duration_cast<microseconds>(end - start).count();
        times_sparse_dense_spmm.push_back(duration);

        std::cout << "  Trial " << (trial + 1) << ": " << duration << " μs\n";
    }

    // Compute minimum time
    std::sort(times_sparse_dense_spmm.begin(), times_sparse_dense_spmm.end());
    long min_sparse_dense_spmm = times_sparse_dense_spmm[0];
    std::cout << "  Minimum time: " << min_sparse_dense_spmm << " μs (best of " << num_trials << " trials)\n\n";

    // =========================================================================
    // Benchmark 4: Sparse × Dense via Densify + BLAS GEMM
    // =========================================================================

    std::cout << "Approach 2b: Densify sparse matrix + BLAS GEMM\n";
    std::cout << "  Step 1: Convert sparse A to dense (reuse from earlier)\n";
    std::cout << "  Step 2: C = A_dense * B2  (using BLAS gemm)\n";

    std::vector<long> times_sparse_dense_gemm;

    for (int trial = 0; trial < num_trials; ++trial) {
        std::fill(C_sparse_dense_densify.begin(), C_sparse_dense_densify.end(), 0.0);

        // Note: We already timed densification in the dense×sparse benchmark
        // For this comparison, we use the same densified matrix
        // In practice, densification cost is the same regardless of multiplication direction

        // Dense GEMM: C = A_dense * B2
        auto start_gemm = steady_clock::now();
        blas::gemm(
            Layout::ColMajor,
            Op::NoTrans,       // op(A) = A
            Op::NoTrans,       // op(B2) = B2
            m, d, n,           // C is m×d, A is m×n, B2 is n×d
            1.0,               // alpha
            A_dense.data(), m, // A_dense is m×n
            B2.data(), n,      // B2 is n×d
            0.0,               // beta
            C_sparse_dense_densify.data(), m  // C is m×d
        );
        auto end_gemm = steady_clock::now();
        long duration_gemm = duration_cast<microseconds>(end_gemm - start_gemm).count();
        times_sparse_dense_gemm.push_back(duration_gemm);

        long total = min_dense_sparse_densify + duration_gemm;  // Use minimum densify time from earlier
        std::cout << "  Trial " << (trial + 1) << ": "
                  << "densify=" << min_dense_sparse_densify << " μs (from earlier), "
                  << "gemm=" << duration_gemm << " μs, "
                  << "total=" << total << " μs\n";
    }

    // Compute minimum times
    std::sort(times_sparse_dense_gemm.begin(), times_sparse_dense_gemm.end());
    long min_sparse_dense_gemm = times_sparse_dense_gemm[0];
    long min_sparse_dense_total = min_dense_sparse_densify + min_sparse_dense_gemm;

    std::cout << "  Minimum times: "
              << "densify=" << min_dense_sparse_densify << " μs (from earlier), "
              << "gemm=" << min_sparse_dense_gemm << " μs, "
              << "total=" << min_sparse_dense_total << " μs (best of " << num_trials << " trials)\n\n";

    // =========================================================================
    // Results summary and comparison
    // =========================================================================

    std::cout << "====================================================================\n";
    std::cout << "RESULTS SUMMARY (minimum times - best of " << num_trials << " trials):\n";
    std::cout << "====================================================================\n\n";

    std::cout << std::fixed << std::setprecision(1);

    std::cout << "DENSE × SPARSE (C = B^T * A):\n";
    std::cout << "  Approach 1a (right_spmm):         " << std::setw(8) << min_dense_sparse_spmm << " μs\n";
    std::cout << "  Approach 1b (densify + gemm):     " << std::setw(8) << min_dense_sparse_total << " μs\n";
    std::cout << "    - Densification:                " << std::setw(8) << min_dense_sparse_densify << " μs  ("
              << (100.0 * min_dense_sparse_densify / min_dense_sparse_total) << "%)\n";
    std::cout << "    - BLAS GEMM:                    " << std::setw(8) << min_dense_sparse_gemm << " μs  ("
              << (100.0 * min_dense_sparse_gemm / min_dense_sparse_total) << "%)\n\n";

    std::cout << "SPARSE × DENSE (C = A * B2):\n";
    std::cout << "  Approach 2a (left_spmm):          " << std::setw(8) << min_sparse_dense_spmm << " μs\n";
    std::cout << "  Approach 2b (densify + gemm):     " << std::setw(8) << min_sparse_dense_total << " μs\n";
    std::cout << "    - Densification:                " << std::setw(8) << min_dense_sparse_densify << " μs  ("
              << (100.0 * min_dense_sparse_densify / min_sparse_dense_total) << "%)\n";
    std::cout << "    - BLAS GEMM:                    " << std::setw(8) << min_sparse_dense_gemm << " μs  ("
              << (100.0 * min_sparse_dense_gemm / min_sparse_dense_total) << "%)\n\n";

    std::cout << "====================================================================\n";
    std::cout << "PERFORMANCE ANALYSIS:\n";
    std::cout << "====================================================================\n\n";

    // Analyze dense × sparse
    if (min_dense_sparse_spmm > min_dense_sparse_total) {
        double slowdown = (double)min_dense_sparse_spmm / min_dense_sparse_total;
        std::cout << "⚠️  DENSE × SPARSE PERFORMANCE ISSUE:\n";
        std::cout << "    right_spmm is " << slowdown << "× SLOWER than densify+gemm\n";
        std::cout << "    (" << min_dense_sparse_spmm << " μs vs " << min_dense_sparse_total << " μs)\n\n";
    } else {
        double speedup = (double)min_dense_sparse_total / min_dense_sparse_spmm;
        std::cout << "✓  Dense × sparse: right_spmm is " << speedup << "× faster than densify+gemm\n\n";
    }

    // Analyze sparse × dense
    if (min_sparse_dense_spmm > min_sparse_dense_total) {
        double slowdown = (double)min_sparse_dense_spmm / min_sparse_dense_total;
        std::cout << "⚠️  SPARSE × DENSE PERFORMANCE ISSUE:\n";
        std::cout << "    left_spmm is " << slowdown << "× SLOWER than densify+gemm\n";
        std::cout << "    (" << min_sparse_dense_spmm << " μs vs " << min_sparse_dense_total << " μs)\n\n";
    } else {
        double speedup = (double)min_sparse_dense_total / min_sparse_dense_spmm;
        std::cout << "✓  Sparse × dense: left_spmm is " << speedup << "× faster than densify+gemm\n\n";
    }

    std::cout << "====================================================================\n\n";

    // Verify correctness
    std::cout << "CORRECTNESS CHECKS:\n";
    std::cout << "====================================================================\n\n";

    // Check dense × sparse results match
    double max_diff_dense_sparse = 0.0;
    for (size_t i = 0; i < C_dense_sparse_spmm.size(); ++i) {
        double diff = std::abs(C_dense_sparse_spmm[i] - C_dense_sparse_densify[i]);
        max_diff_dense_sparse = std::max(max_diff_dense_sparse, diff);
    }

    std::cout << "Dense × Sparse: max|right_spmm - densify+gemm| = " << max_diff_dense_sparse << "\n";
    if (max_diff_dense_sparse < 1e-10) {
        std::cout << "✓  Results match (both approaches produce the same output)\n\n";
    } else {
        std::cout << "⚠️  Results differ! Check implementation.\n\n";
    }

    // Check sparse × dense results match
    double max_diff_sparse_dense = 0.0;
    for (size_t i = 0; i < C_sparse_dense_spmm.size(); ++i) {
        double diff = std::abs(C_sparse_dense_spmm[i] - C_sparse_dense_densify[i]);
        max_diff_sparse_dense = std::max(max_diff_sparse_dense, diff);
    }

    std::cout << "Sparse × Dense: max|left_spmm - densify+gemm| = " << max_diff_sparse_dense << "\n";
    if (max_diff_sparse_dense < 1e-10) {
        std::cout << "✓  Results match (both approaches produce the same output)\n\n";
    } else {
        std::cout << "⚠️  Results differ! Check implementation.\n\n";
    }

    return 0;
}
