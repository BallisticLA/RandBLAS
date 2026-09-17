// Copyright, 2026. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

// ============================================================================
// SPSYMM / SKETCH_SYMMETRIC PERFORMANCE BENCHMARK
// ============================================================================
//
// This benchmark answers two performance questions about symmetric-aware
// multiplication kernels:
//
//   1. Sparse: how much does the one-triangle RandBLAS::spsymm path (MKL
//      SPARSE_MATRIX_TYPE_SYMMETRIC fast path where available) gain over the
//      both-triangles workaround (materialize both triangles into a general
//      sparse matrix and call RandBLAS::spmm)?
//
//   2. Dense: how much does the SYMM-backed RandBLAS::sketch_symmetric gain
//      over the equivalent sketch_general call (a GEMM that reads the full
//      matrix and exploits no symmetry)?
//
// Every timed method is verified against a dense blas::symm reference before
// its timing rows are printed; a FAIL note marks any row whose output
// diverged.
//
// NOTATION:
//   A_symm  - symmetric matrix of order n_A (dense or sparse, one triangle)
//   B       - dense matrix (n_A x d for side=Left)
//   C       - dense result matrix (same shape as B)
//   density - fraction of upper-triangle entries of A_symm that are nonzero
//             (the implied lower-triangle entries follow by symmetry)
//
// USAGE:
//   ./spsymm_performance [--help] [--threads T]                 # default sweep
//   ./spsymm_performance [--threads T] n_A d density [trials]   # single config
//
//   Defaults: trials=10, density=0.05, ambient OpenMP thread count.
//
// ============================================================================

#include <RandBLAS.hh>
#include "RandBLAS/testing/benchmarking.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

namespace bench = RandBLAS::testing;


// ============================================================================
// Helpers: random dense symmetric matrix + sparse counterparts.
// ============================================================================

void make_dense_symmetric(int64_t n, std::vector<T>& A, uint64_t seed) {
    A.assign(static_cast<size_t>(n) * n, T(0));
    // std::mt19937_64 rather than a RandBLAS sampler: A is arbitrary fixed
    // input data here, not a sketching operator.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> uni(-1.0, 1.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i <= j; ++i)
            A[i + j * n] = uni(rng);
    RandBLAS::symmetrize(blas::Layout::ColMajor, blas::Uplo::Upper, n, A.data(), n);
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

// Build a one-triangle (Upper) CSR by zeroing the strict lower triangle of a
// copy of A_dense and converting with the library's dense_to_csr.
SpMat build_csr_upper_only(int64_t n, const std::vector<T>& A_dense) {
    std::vector<T> upper(A_dense);
    RandBLAS::overwrite_triangle(blas::Layout::ColMajor, blas::Uplo::Lower, n, 1, upper.data(), n);
    SpMat A_sparse(n, n);
    RandBLAS::sparse_data::csr::dense_to_csr(blas::Layout::ColMajor, upper.data(), T(0), A_sparse);
    return A_sparse;
}

// Build a full (both-triangles-stored) CSR for the general-spmm workaround.
SpMat build_csr_both_triangles(int64_t n, const std::vector<T>& A_dense) {
    std::vector<T> full(A_dense);
    SpMat A_sparse(n, n);
    RandBLAS::sparse_data::csr::dense_to_csr(blas::Layout::ColMajor, full.data(), T(0), A_sparse);
    return A_sparse;
}


// ============================================================================
// Timing helper: (min, median) microseconds over num_trials.
// ============================================================================
template <typename Func>
std::pair<int64_t, int64_t> run_trials(Func&& func, int num_trials) {
    std::vector<int64_t> times;
    times.reserve(num_trials);
    for (int t = 0; t < num_trials; ++t) {
        auto start = steady_clock::now();
        func();
        auto end = steady_clock::now();
        times.push_back(static_cast<int64_t>(duration_cast<microseconds>(end - start).count()));
    }
    std::sort(times.begin(), times.end());
    return {times[0], times[num_trials / 2]};
}

// Max relative elementwise deviation between two equally-shaped buffers.
double max_rel_error(const std::vector<T>& actual, const std::vector<T>& expect) {
    double err = 0.0;
    for (size_t i = 0; i < expect.size(); ++i) {
        double scale = std::max(1.0, std::abs(static_cast<double>(expect[i])));
        err = std::max(err, std::abs(static_cast<double>(actual[i] - expect[i])) / scale);
    }
    return err;
}

void print_row(const std::string& name, int64_t min_us, int64_t med_us, int64_t baseline, bool pass) {
    double ratio = (baseline > 0) ? static_cast<double>(min_us) / static_cast<double>(baseline) : 1.0;
    std::cout << "  " << std::setw(40) << std::left << name
              << std::setw(10) << std::right << min_us
              << std::setw(10) << med_us
              << std::setw(9) << std::fixed << std::setprecision(2) << ratio << "x"
              << "  " << (pass ? "PASS" : "FAIL") << "\n";
}


// ============================================================================
// run_config: one (n_A, d, density) point. Every method writes its own copy
// of C, is checked against the dense blas::symm reference, and is then timed.
// ============================================================================
void run_config(int64_t n_A, int64_t d, double density, int num_trials) {
    uint64_t seed = 12345;
    Layout layout = Layout::ColMajor;
    T alpha = T(1.0), beta = T(0.0);
    // Verification tolerance: the sparse and dense paths accumulate in
    // different orders; a loose 1e-10 relative bound catches routing bugs
    // without tripping on roundoff.
    const double check_tol = 1e-10;

    std::cout << "--- A is " << n_A << "x" << n_A
              << " symmetric, B is " << n_A << "x" << d
              << ", density=" << std::setprecision(4) << density
              << " (min + median over " << num_trials << " trials) ---\n";

    std::vector<T> A_full(static_cast<size_t>(n_A) * n_A);
    make_dense_symmetric(n_A, A_full, seed);
    sparsify_upper_then_mirror(n_A, A_full, density, seed + 1);

    SpMat A_csr_upper = build_csr_upper_only(n_A, A_full);
    SpMat A_csr_full  = build_csr_both_triangles(n_A, A_full);

    std::cout << "  upper-triangle nnz = " << A_csr_upper.rowptr[n_A]
              << " (full = " << A_csr_full.rowptr[n_A] << ")\n";

    std::vector<T> B(static_cast<size_t>(n_A) * d);
    {
        std::mt19937_64 rng(seed + 2);
        std::uniform_real_distribution<T> uni(-1.0, 1.0);
        for (auto& x : B) x = uni(rng);
    }

    // Dense blas::symm reference output (also the timed dense baseline).
    std::vector<T> C_ref(static_cast<size_t>(n_A) * d, T(0));
    blas::symm(layout, Side::Left, Uplo::Upper, n_A, d,
               alpha, A_full.data(), n_A, B.data(), n_A, beta, C_ref.data(), n_A);

    std::cout << "\n  SPARSE (side=Left, C = A*B):\n";
    std::cout << "  " << std::setw(40) << std::left << "kernel"
              << std::setw(10) << std::right << "min(us)"
              << std::setw(10) << "med(us)"
              << std::setw(10) << "ratio" << "  check\n";

    std::vector<T> C(static_cast<size_t>(n_A) * d, T(0));

    // 1. One-triangle storage through RandBLAS::spsymm.
    RandBLAS::spsymm(layout, Uplo::Upper, d,
                     alpha, A_csr_upper, B.data(), n_A, beta, C.data(), n_A);
    bool ok1 = max_rel_error(C, C_ref) <= check_tol;
    auto [t1_min, t1_med] = run_trials([&] {
        RandBLAS::spsymm(layout, Uplo::Upper, d,
                         alpha, A_csr_upper, B.data(), n_A,
                         beta, C.data(), n_A);
    }, num_trials);

    // 2. Both-triangles workaround through general RandBLAS::spmm.
    RandBLAS::spmm(layout, Op::NoTrans, Op::NoTrans, n_A, d, n_A,
                   alpha, A_csr_full, B.data(), n_A, beta, C.data(), n_A);
    bool ok2 = max_rel_error(C, C_ref) <= check_tol;
    auto [t2_min, t2_med] = run_trials([&] {
        RandBLAS::spmm(layout, Op::NoTrans, Op::NoTrans,
                       n_A, d, n_A,
                       alpha, A_csr_full, B.data(), n_A,
                       beta, C.data(), n_A);
    }, num_trials);

    // 3. Dense blas::symm on the fully populated dense A: does not exploit
    //    sparsity, but is the fastest possible dense SYMM.
    auto [t3_min, t3_med] = run_trials([&] {
        blas::symm(layout, Side::Left, Uplo::Upper, n_A, d,
                   alpha, A_full.data(), n_A, B.data(), n_A,
                   beta, C.data(), n_A);
    }, num_trials);

    print_row("RandBLAS::spsymm (one triangle)",   t1_min, t1_med, t1_min, ok1);
    print_row("RandBLAS::spmm (both triangles)",   t2_min, t2_med, t1_min, ok2);
    print_row("blas::symm (dense reference)",      t3_min, t3_med, t1_min, true);

    // Dense-symmetric sketching comparison: SYMM-backed sketch_symmetric vs
    // the GEMM-forwarding sketch_general equivalent.
    std::cout << "\n  DENSE (sketch_symmetric vs sketch_general on dense-symm A):\n";
    std::cout << "  " << std::setw(40) << std::left << "kernel"
              << std::setw(10) << std::right << "min(us)"
              << std::setw(10) << "med(us)"
              << std::setw(10) << "ratio" << "  check\n";

    RandBLAS::DenseDist DS(n_A, d, RandBLAS::ScalarDist::Uniform);
    RandBLAS::DenseSkOp<T> S(DS, static_cast<uint32_t>(seed + 3));
    RandBLAS::fill_dense(S);

    // Reference for the sketching comparison: the GEMM-forwarding
    // sketch_general result (reads the full symmetric matrix, so it is the
    // trusted baseline; the check below is SYMM-path vs GEMM-path agreement).
    std::vector<T> C_sk_ref(static_cast<size_t>(n_A) * d, T(0));
    RandBLAS::sketch_general(layout, Op::NoTrans, Op::NoTrans, n_A, d, n_A,
                             alpha, A_full.data(), n_A, S, 0, 0, beta, C_sk_ref.data(), n_A);

    // 1. SYMM-backed sketch_symmetric.
    RandBLAS::sketch_symmetric(layout, Uplo::Upper, n_A, d,
                               alpha, A_full.data(), n_A, S, 0, 0, beta, C.data(), n_A);
    bool ok3 = max_rel_error(C, C_sk_ref) <= check_tol;
    auto [s1_min, s1_med] = run_trials([&] {
        RandBLAS::sketch_symmetric(layout, Uplo::Upper, n_A, d,
                                   alpha, A_full.data(), n_A, S, 0, 0,
                                   beta, C.data(), n_A);
    }, num_trials);

    // 2. Equivalent call via sketch_general (GEMM, no symmetry exploited;
    //    this is the reference, so its check is definitionally PASS).
    auto [s2_min, s2_med] = run_trials([&] {
        RandBLAS::sketch_general(layout, Op::NoTrans, Op::NoTrans,
                                 n_A, d, n_A,
                                 alpha, A_full.data(), n_A, S, 0, 0,
                                 beta, C.data(), n_A);
    }, num_trials);

    print_row("sketch_symmetric (SYMM path)",      s1_min, s1_med, s1_min, ok3);
    print_row("sketch_general (GEMM path)",        s2_min, s2_med, s1_min, true);

    std::cout << "\n";
}


// ============================================================================
// main: default 3-point sweep, or single config from positional arguments.
// ============================================================================
void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " [--help] [--threads T]                 default sweep\n"
              << "  " << prog << " [--threads T] n_A d density [trials]  single configuration\n"
              << "n_A, d, trials are positive integers; density is in (0, 1].\n";
}

int main(int argc, char** argv) {
    bench::OpenMPSettingsGuard omp_guard;

    std::vector<std::string> args(argv + 1, argv + argc);
    int requested_threads = 0;
    for (size_t i = 0; i < args.size(); ) {
        if (args[i] == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (args[i] == "--threads" && i + 1 < args.size()) {
            requested_threads = std::atoi(args[i + 1].c_str());
            if (requested_threads <= 0) {
                std::cerr << "Invalid configuration. Expected a positive --threads value.\n";
                return 1;
            }
            args.erase(args.begin() + i, args.begin() + i + 2);
        } else {
            ++i;
        }
    }
    if (requested_threads > 0)
        bench::set_threads(requested_threads);
    std::cout << "OpenMP threads: " << bench::current_threads();
    if (requested_threads > 0)
        std::cout << " (requested " << requested_threads
                  << ", effective " << bench::effective_threads(requested_threads) << ")";
    std::cout << "\n\n";

    if (args.size() >= 3) {
        int64_t n_A = std::atoll(args[0].c_str());
        int64_t d   = std::atoll(args[1].c_str());
        double density = std::atof(args[2].c_str());
        int num_trials = (args.size() >= 4) ? std::atoi(args[3].c_str()) : 10;
        if (n_A <= 0 || d <= 0 || density <= 0.0 || density > 1.0 || num_trials <= 0) {
            std::cerr << "Invalid configuration. Expected positive n_A, d, trials and density in (0, 1].\n";
            print_usage(argv[0]);
            return 1;
        }
        run_config(n_A, d, density, num_trials);
    } else if (args.empty()) {
        int num_trials = 10;
        std::cout << "Default sweep (n_A in {500, 1000, 2000}, d=200, density=0.05)\n\n";
        for (int64_t n_A : {500, 1000, 2000}) {
            run_config(n_A, /*d=*/200, /*density=*/0.05, num_trials);
        }
    } else {
        std::cerr << "Invalid configuration. Expected zero or 3-4 positional arguments.\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
