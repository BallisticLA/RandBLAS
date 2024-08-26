// Copyright, 2024. See LICENSE for copyright holder information.
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
#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <unordered_map>
#include <iomanip> 
#include <limits> 
#include <numbers>
#include <chrono>
#include <fstream>
#include <stdexcept>


using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

#define DOUT(_d) std::setprecision(8) << _d

auto parse_args(int argc, char** argv) {
    std::string mat{"../sparse-data-matrices/N_reactome/N_reactome.mtx"};
    int k = 4;
    if (argc > 1)
        k = atoi(argv[1]);
    if (argc > 2)
        mat = argv[2];
    return std::make_tuple(mat, k);
}

template <typename T>
COOMatrix<T> from_matrix_market(std::string fn) {

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows,  cols, vals
    );

    COOMatrix<T> out(n_rows, n_cols);
    out.reserve(vals.size());
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

template <typename T>
void col_swap(int64_t m, int64_t n, int64_t k, T* A, int64_t lda, const int64_t* idx) {
    // Adapted from RandLAPACK code.
    if(k > n) 
        throw std::runtime_error("Invalid rank parameter.");
    int64_t *idx_copy = new int64_t[n]{};
    for (int i = 0; i < n; ++i)
        idx_copy[i] = idx[i];

    int64_t i, j;
    for (i = 0, j = 0; i < k; ++i) {
        j = idx_copy[i] - 1;
        blas::swap(m, &A[i * lda], 1, &A[j * lda], 1);
        auto it = std::find(idx_copy + i, idx_copy + k, i + 1);
        idx_copy[it - idx_copy] = j + 1;
    }
    delete [] idx_copy;
}

template <typename T>
int qr_row_stabilize(int64_t m, int64_t n, T* mat, T* vec_work) {
    if(lapack::gelqf(m, n, mat, m, vec_work))
        return 1;
    randblas_require(m < n);
    // The signature of UNGLQ is weird. LAPACK++ provides it as a wrapper to ORMLQ. See
    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-1/orglq.html
    // for why we're using these particular arguments to get an orthonormal basis
    // for the rowspace of "mat" (where the basis is given by a column-major representation
    // of the transposed basis vectors).
    lapack::unglq(m, n, m, mat, m, vec_work);
    return 0;
}

template <typename T>
int sketch_orthogonalize_rows(int64_t m, int64_t n, T* A, T* work, int64_t d, int32_t key) {
    RandBLAS::RNGState state(key);
    // A is wide, m-by-n in column-major format.
    // work has at least m*d space.
    randblas_require(d >= 2);
    randblas_require(d >= m);
    std::vector<T> tau(d, 0.0);
    int64_t vec_nnz = std::min(d/2, (int64_t) 4);
    RandBLAS::SparseDist D{n, d, vec_nnz};
    RandBLAS::SparseSkOp<T> S(D, state);
    // Simple option (shown here):
    //      Sketch A in column-major format, then do LQ on the sketch.
    //      If the sketch looks singular after we decompose it, then we bite the bullet and do LQ on A.
    //
    // Fancy option (for a later implementation):
    //      Compute the sketch in row-major format (this is lying about A's format, but we resolve that by setting transA=T).
    //      Look at the row-major sketch, interpret it as a transposed column-major sketch, factor the "implicitly de-transposed" version by GEQP3,
    //      then implicitly transpose back to get the factors for row-pivoted LQ of A_sk:
    //          A_sk = P R^* Q^*
    //      Apply M = inv(P R^*) = inv(R^*) P^* to the left of A by TRSM.
    //      If rank(A) = rank(A_sk) = m, then in exact arithmetic the conditioning of MA should be independent from
    //      that of A. However, there can be a positive correlation between cond(MA) and cond(A) in finite-precision.
    //      This happens when cond(A) is very large, and may warrant truncating rows of MA.
    //      There are many ways to select the size of that row-block. Two options jump out:
    //          1. Compute (or estimate) the numerical rank of R by a reliable method of your choosing.
    //          2. Proceed in a similar vein as CQRRPT: estimate the condition number of leading row blocks of MA
    //             by forming the Gram matrix (MA)(MA)^*, doing Cholesky on it, and computing or estimating the 
    //             condition numbers of leading submatrices of the Cholesky factor.
    //
    //
    RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, d, n, 1.0, A, m, S, 0.0, work, m);
    lapack::gelqf(m, d, work, m, tau.data());
    T tol = std::numeric_limits<T>::epsilon()*100;
    for (int i = 0; i < m; ++i) {
        if (std::abs(work[i*m + i]) < tol) {
            // We can't safely invert. Fall back on LQ of A.
            qr_row_stabilize(m, n, A, tau.data());
            std::cout << "\n----> Could not safely sketch-orthogonalize. Falling back on GELQF instead.\n\n";
            return 1;
        }
    }
    // L is in the lower triangle of work.
    // Need to transform
    //      A <- inv(L)A
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, blas::Diag::NonUnit, m, n, 1.0, work, m, A, m);
    return 0;
}

template <typename T>
int lu_row_stabilize(int64_t m, int64_t n, T* mat, int64_t* piv_work) {
    randblas_require(m < n);
    for (int64_t i = 0; i < m; ++i)
        piv_work[i] = 0;
    lapack::getrf(m, n, mat, m, piv_work);
    // above: the permutation applied to the rows of mat doesn't matter in our context.
    // below: Need to zero-out the strict lower triangle of mat and scale each row.
    T tol = std::numeric_limits<T>::epsilon()*10;
    bool nonzero_diag_U = true;
    for (int64_t j = 0; (j < m-1) & nonzero_diag_U; ++j) {
        nonzero_diag_U = abs(mat[j + j*m]) > tol;
        for (int64_t i = j + 1; i < m; ++i) {
            mat[i + j*m] = 0.0;
        }
    }
    if (!nonzero_diag_U) {
        throw std::runtime_error("LU stabilization failed. Matrix has been overwritten, so we cannot recover.");
    }
    for (int64_t i = 0; i < m; ++i) {
        T scale = 1.0 / mat[i + i*m];
        blas::scal(n, scale, mat + i, m);
    }
    return 0;
}

#ifdef FINE_GRAINED
#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << DOUT(dtime / 1e6) << std::endl; \
        }
#else
#define TIMED_LINE(_op, _name) _op;
#endif

enum class StabilizationMethod : char {
    LU = 'L',
    LQ = 'H',  // householder
    sketch = 'S',
    None = 'N'
};

template <typename T, typename SpMat, typename STATE>
void power_iter_col_sketch(SpMat &A, int64_t k, T* Y, int64_t p_data_aware, STATE state, T* work, StabilizationMethod sm) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    using RandBLAS::sparse_data::right_spmm;
    using blas::Op;
    using blas::Layout;
    // Want k-by-n matrix Y = SA, where S has p_data_aware passes over A to build up data-aware geometry.
    // Run ...
    //   p_done = 0
    //   if p_data_aware is even:
    //      S = oblivious k-by-m.
    //   if p_data_aware is odd:
    //      T = oblivious k-by-n.
    //      S = row_orth(T A')
    //      p_done += 1
    //   while (p_data_aware - p_done > 0) 
    //      T = row_orth(S A)
    //      S = row_orth(T A')
    //      p_done += 2
    //    Y = S A
    T* mat_work1 = Y;
    T* mat_work2 = work;

    // Messy code to allow for different stabilization methods
    T* tau_work = new T[std::max(n, m)];
    int64_t* piv_work = new int64_t[k];
    int64_t sketch_dim = (int64_t) (1.25*k + 1);
    T* sketch_orth_work = new T[sketch_dim * m]{0.0};
    auto stab_func = [sm, k, piv_work, tau_work, sketch_orth_work, sketch_dim](T* mat_to_stab, int64_t num_mat_cols, int64_t key) {
        if (sm == StabilizationMethod::LU) {
            lu_row_stabilize(k, num_mat_cols, mat_to_stab, piv_work);
        } else if (sm == StabilizationMethod::LQ) {
            qr_row_stabilize(k, num_mat_cols, mat_to_stab, tau_work);
        } else if (sm == StabilizationMethod::sketch) {
            sketch_orthogonalize_rows(k, num_mat_cols, mat_to_stab, sketch_orth_work, sketch_dim, key);
        } else if (sm == StabilizationMethod::None) {
            // do nothing
        }
        return;
    };

    int64_t p_done = 0;
    if (p_data_aware % 2 == 0) {
        RandBLAS::DenseDist D(k, m, RandBLAS::ScalarDist::Gaussian);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work2, state), "sampling : ")
    } else {
        RandBLAS::DenseDist D(k, n, RandBLAS::ScalarDist::Gaussian);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work1, state), "sampling : ")
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::Trans, k, m, n, 1.0, mat_work1, k, A, 0, 0, 0.0, mat_work2, k), "spmm : ")
        p_done += 1;
        TIMED_LINE(
        stab_func(mat_work2, m, p_done), "stabilization   : ")
    }

    while (p_data_aware - p_done > 0) {
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m, 1.0, mat_work2, k, A, 0, 0, 0.0, mat_work1, k), "right_spmm      : ")
        p_done += 1;
        TIMED_LINE(
        stab_func(mat_work1, n, p_done), "stabilization   : ")
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::Trans, k, m, n, 1.0, mat_work1, k, A, 0, 0, 0.0, mat_work2, k), "right_spmm      : ")
        p_done += 1;
        TIMED_LINE(
        stab_func(mat_work2, m, p_done), "stabilization   : ")
    }
    TIMED_LINE(
    right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m, 1.0, mat_work2, k, A, 0, 0, 0.0, Y, k),  "spmm : ")

    delete [] tau_work;
    delete [] piv_work;
    delete [] sketch_orth_work;
    return;
}

template <typename T>
void print_row_norms(T* mat, int64_t m, int64_t n, std::string s) {
    std::cout << "Row norms for " << s << " : [ ";
    int i;
    for (i = 0; i < m-1; ++i) {
        std::cout << DOUT(blas::nrm2(n, mat + i, m)) << ", ";
    }
    std::cout << DOUT(blas::nrm2(n, mat + i, m)) << " ] " << std::endl;
    return;
}

void print_pivots(int64_t *piv, int64_t k) {
    std::cout << "Leading pivots   : [ ";
    int i;
    for (i = 0; i < k-1; ++i) {
        std::cout << piv[i]-1 << ", ";
    }
    std::cout << piv[i]-1 << " ]" << std::endl;
}

template <typename T, typename SpMat>
void sketch_to_tqrcp(SpMat &A, int64_t k, T* Q, int64_t ldq,  T* Y, int64_t ldy, int64_t *piv) {
    // On input, Y is a left-sketch of A.
    // On exit, Q, Y, piv are overwritten so that ...
    //      The columns of Q are an orthonormal basis for A(:, piv(:k))
    //      Y = Q' A(:, piv) is upper-triangular.
    using sint_t = typename SpMat::index_t;
    constexpr bool valid_type = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    randblas_require(valid_type);
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    using blas::Layout;
    using blas::Op;
    using blas::Side;
    using blas::Uplo;
    for (int64_t i = 0; i < n; ++i)
        piv[i] = 0;
    T* tau = new T[n]{};
    T* precond = new T[k * k]{};

    // ================================================================
    // Step 1: get the pivots
    TIMED_LINE(
    lapack::geqp3(k, n, Y, ldy, piv, tau), "GEQP3 : ")

    // ================================================================
    // Step 2: copy A(:, piv(0)-1), ..., A(:, piv(k)-1) into dense Q
    for (int64_t j = 0; j < k; ++j) {
        RandBLAS::util::safe_scal(m, 0.0, Q + j*ldq, 1);
        for (int64_t ell = A.colptr[piv[j]-1]; ell < A.colptr[piv[j]]; ++ell) {
            int64_t i = A.rowidxs[ell];
            Q[i + ldq*j] = A.vals[ell];
        }
    }

    // ================================================================
    // Step 3: get explicit representation of orth(Q).
    TIMED_LINE(
    //      Extract a preconditioner from the column-pivoted QR decomposition of Y.
    for (int64_t j = 0; j < k; j++) {
        for (int64_t i = 0; i < k; ++i) {
            precond[i + k*j] = Y[i + k*j];
        }
    }
    //      Apply the preconditioner: Q = Q / precond.
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, blas::Diag::NonUnit, m, k, 1.0, precond, k, Q, ldq);
    //      Cholesky-orthogonalize the preconditioned matrix:
    //          precond = chol(Q' * Q, "upper")
    //          Q = Q / precond.
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q, ldq, 0.0, precond, k);
    lapack::potrf(Uplo::Upper, k, precond, k);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, blas::Diag::NonUnit, m, k, 1.0, precond, k, Q, ldq), "getQ : ")
    
    // ================================================================
    // Step 4: multiply Y = Q'A and pivot Y = Y(:, piv)
    TIMED_LINE(
    RandBLAS::right_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, 1.0, Q, ldq, A, 0, 0, 0.0, Y, ldy);
    col_swap(k, n, k, Y, ldy, piv), "getR : ")

    delete [] tau;
    delete [] precond;
    return;
}

template <typename SpMat>
int run(SpMat &A, int64_t k, int64_t power_iteration_steps, StabilizationMethod sm, bool extra_verbose) {
    auto m = A.n_rows;
    auto n = A.n_cols;

    using T = typename SpMat::scalar_t;
    T *Q  = new T[m*k]{};
    T *R = new T[k*n]{};
    int64_t *piv = new int64_t[n]{};
    RandBLAS::RNGState<r123::Philox4x32> state(0);

    auto start_timer = std_clock::now();
    TIMED_LINE(
    power_iter_col_sketch(A, k, R, power_iteration_steps, state, Q, sm), "\n\tpower iter sketch  : ")
    if (extra_verbose)
        print_row_norms(R, k, n, "Yf");
    TIMED_LINE(
    sketch_to_tqrcp(A, k, Q, m, R, k, piv), "\n\tsketch to QRCP     : ")
    auto stop_timer = std_clock::now();
    if (extra_verbose)
        print_row_norms(R, k, n, "R ");
    print_pivots(piv, k);

    T runtime = (T) duration_cast<microseconds>(stop_timer - start_timer).count();
    std::cout << "Runtime in μs    : " << DOUT(runtime) << std::endl << std::endl;

    delete [] Q;
    delete [] R;
    delete [] piv;
    return 0;
}

int main(int argc, char** argv) {
    /*
    This program should be called from a "build" folder that's one level below RandBLAS/examples.

    If called with two arguments, then the first argument will be the approximation rank,
    and the second argument will be a path (relative or absolute) to a MatrixMarket file.

    If called with zero or one arguments, we'll assume that there's a file located at 
        ../sparse-data-matrices/N_reactome/N_reactome.mtx.
    
    If called with zero arguments, we'll automatically set the approximation rank to 4.
    */
    auto [fn, _k] = parse_args(argc, argv);
    auto mat_coo = from_matrix_market<double>(fn);
    auto m = mat_coo.n_rows;
    auto n = mat_coo.n_cols;
    int64_t k = (int64_t) _k;

    std::cout << "\nProcessing matrix in " << fn << std::endl;
    std::cout << "n_rows  : " << mat_coo.n_rows << std::endl;
    std::cout << "n_cols  : " << mat_coo.n_cols << std::endl;
    double density = ((double) mat_coo.nnz) / ((double) (mat_coo.n_rows * mat_coo.n_cols));
    std::cout << "density : " << DOUT(density) << std::endl << std::endl;

    RandBLAS::CSCMatrix<double> mat_csc(m, n);
    RandBLAS::conversions::coo_to_csc(mat_coo, mat_csc);
    int64_t power_iter_steps = 2;
    bool extra_verbose = true;

    std::cout << "Computing rank-" << k << " truncated QRCP.\n";
    std::cout << "Internally use " << power_iter_steps << " steps of power iteration.\n";
    std::cout << "Consider four runs, each stabilizing power iteration in a different way.\n\n";
    std::cout << "Take Q from LQ\n";
    run(mat_csc, k, power_iter_steps, StabilizationMethod::LQ, extra_verbose);
    std::cout << "Sketch-orthogonalize\n";
    run(mat_csc, k, power_iter_steps, StabilizationMethod::sketch, extra_verbose);
    std::cout << "Do nothing. This is numerically dangerous unless power_iter_steps is extremely small.\n";
    run(mat_csc, k, power_iter_steps, StabilizationMethod::None, extra_verbose);
    std::cout << "Take (scaled) U from row-pivoted LU. This may exit with an error!\n";
    run(mat_csc, k, power_iter_steps, StabilizationMethod::LU, extra_verbose);
    return 0;
}
