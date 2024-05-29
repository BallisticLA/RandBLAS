

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


using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

#define FOUT(_d) std::setprecision(6) << _d


std::string parse_args(int argc, char** argv) {
    if (argc > 1) {
        return std::string{argv[1]};
    } else {
        return "../sparse-data-matrices/Binaryalphadigs_10NN/Binaryalphadigs_10NN.mtx";
    }
}

template <typename T>
COOMatrix<T> adjacency_mat_from_matrix_market(std::string fn) {
    int64_t n, n_ = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};
    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n, n_, rows,  cols, vals
    );
    randblas_require(n == n_); // we must be square.
    int64_t m = vals.size();
    COOMatrix<T> out(n, n);
    out.reserve(m);
    for (int64_t i = 0; i < m; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }
    return out;
}

template <typename T>
COOMatrix<T> regularized_laplacian_from_matrix_market(std::string fn, T regularization) {
    randblas_require(regularization >= 0);

    int64_t n, n_ = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n, n_, rows,  cols, vals
    );
    randblas_require(n == n_); // we must be square.

    int64_t m = vals.size();
    int64_t nnz = m + n;
    COOMatrix<T> out(n, n);
    out.reserve(nnz);

    T* diag = new T[n]{regularization};
    for (int64_t i = 0; i < m; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
        diag[cols[i]] += std::abs(vals[i]);
    }
    for (int64_t i = m; i < nnz; ++i) {
        out.rows[i] = i - m;
        out.cols[i] = i - m;
        out.vals[i] = diag[i - m];
    }

    delete [] diag;
    return out;
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
int lu_row_stabilize(int64_t m, int64_t n, T* mat, int64_t* piv_work) {
    randblas_require(m < n);
    for (int64_t i = 0; i < m; ++i)
        piv_work[i] = 0;
    int status = lapack::getrf(m, n, mat, m, piv_work);
    // above: the permutation applied to the rows of mat doesn't matter in our context.
    // below: Need to zero-out the strict lower triangle of mat and scale each row.
    for (int64_t j = 0; j < m-1; ++j) {
        for (int64_t i = j + 1; i < m; ++i) {
            mat[i + j*m] = 0.0;
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        T scale = 1.0 / mat[i + i*m];
        blas::scal(n, scale, mat + i, m);
    }
    return status;
}

#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        float ftime = (float) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << FOUT(ftime / 1e6) << std::endl; \
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

template <typename T, typename SpMat, typename STATE>
void power_iter_col_sketch(SpMat &A, int64_t k, T* Y, int64_t p_data_aware, STATE state, T* work) {
    T ONE = 1.0;
    T ZERO = 0.0;
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
    int64_t* piv_work = new int64_t[k];

    int64_t p_done = 0;
    if (p_data_aware % 2 == 0) {
        RandBLAS::DenseDist D(k, m, RandBLAS::DenseDistName::Gaussian);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work2, state), "sampling        : ")
    } else {
        RandBLAS::DenseDist D(k, n, RandBLAS::DenseDistName::Gaussian);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work1, state), "sampling        : ")
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::Trans, k, m, n, ONE, mat_work1, k, A, 0, 0, ZERO, mat_work2, k), "right_spmm      : ")
        p_done += 1;
        TIMED_LINE(
        lu_row_stabilize(k, m, mat_work2, piv_work), "stabilization   : ")
    }

    while (p_data_aware - p_done > 0) {
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m, ONE, mat_work2, k, A, 0, 0, ZERO, mat_work1, k), "right_spmm      : ")
        TIMED_LINE(
        lu_row_stabilize(k, m, mat_work1, piv_work), "stabilization   : ")
        TIMED_LINE(
        right_spmm(Layout::ColMajor, Op::NoTrans, Op::Trans, k, m, n, ONE, mat_work1, k, A, 0, 0, ZERO, mat_work2, k), "right_spmm      : ")
        TIMED_LINE(
        lu_row_stabilize(k, m, mat_work2, piv_work), "stabilization   : ")
        p_done += 2;
    }
    TIMED_LINE(
    right_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, m, ONE, mat_work2, k, A, 0, 0, ZERO, Y, k), "right_spmm      : ")

    delete [] piv_work;
    return;
}

template <typename T, typename SpMat>
void sketch_to_tqrcp(SpMat &A, int64_t k, T* Q, int64_t ldq,  T* Y, int64_t ldy, int64_t *piv) {
    // On input, Y is a left-sketch of A.
    // On exit, Q, Y, piv are overwritten so that ...
    //      The columns of Q are an orthonormal basis for A(:, piv(:k))
    //      Y = Q' A(:, piv) is upper-triangular.
    T ONE = 1.0;
    T ZERO = 0.0;
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

    // Step 1: get the pivots
    TIMED_LINE(
    lapack::geqp3(k, n, Y, ldy, piv, tau), "\nQRCP of sketch     : ")
    for (int64_t j = 0; j < k; j++) {
        for (int64_t i = 0; i < k; ++i) {
            precond[i + k*j] = Y[i + k*j];
        }
    }
    // Step 2: copy A(:, piv(0)-1), ..., A(:, piv(k)-1) into dense Q
    for (int64_t j = 0; j < k; ++j) {
        RandBLAS::util::safe_scal(m, ZERO, Q + j*ldq, 1);
        for (int64_t ell = A.colptr[piv[j]-1]; ell < A.colptr[piv[j]]; ++ell) {
            int64_t i = A.rowidxs[ell];
            Q[i + ldq*j] = A.vals[ell];
        }
    }
    // Step 3: get explicit representation of orth(Q).
    TIMED_LINE(
    // Apply a preconditioner
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, blas::Diag::NonUnit, m, k, ONE, precond, k, Q, ldq);
    // safely cholesky-orthogonalize (we don't care about R from a hypothetical CholeskyQR)
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, ONE, Q, ldq, ZERO, precond, k);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, blas::Diag::NonUnit, m, k, ONE, precond, k, Q, ldq);,
    "orth(A(:,piv(:k))) : ")
    // Step 4: multiply Y = Q'A and pivot Y = Y(:, piv)
    TIMED_LINE(
    RandBLAS::right_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, ONE, Q, ldq, A, 0, 0, ZERO, Y, ldy);
    col_swap(k, n, k, Y, ldy, piv), "R = Q' A(:, piv)   : ")

    delete [] tau;
    delete [] precond;
    return;
}



int main(int argc, char** argv) {
    auto fn = parse_args(argc, argv);
    //auto L_coo = regularized_laplacian_from_matrix_market(fn, 0.0f);
    auto L_coo = adjacency_mat_from_matrix_market<float>(fn);
    auto m = L_coo.n_rows;
    auto n = L_coo.n_cols;
    RandBLAS::CSCMatrix<float> L(m, n);
    RandBLAS::conversions::coo_to_csc(L_coo, L);

    std::cout << "\nn_rows  : " << L.n_rows << std::endl;
    std::cout << "n_cols  : " << L.n_cols << std::endl;
    float density = ((float) L.nnz) / ((float) (L.n_rows * L.n_cols));
    std::cout << "density : " << FOUT(density) << std::endl;

    // Run the randomized algorithm!
    int64_t k = 32;
    float *Q  = new float[m*k]{};
    float *R = new float[k*n]{};
    int64_t *piv = new int64_t[n]{};
    RandBLAS::RNGState<r123::Philox4x32> state(0);

    auto start_timer = std_clock::now();
    int64_t p_data_aware = 2;
    std::cout << "p_data_aware : " << p_data_aware << std::endl << std::endl;
    TIMED_LINE(
    power_iter_col_sketch(L, k, R, p_data_aware, state, Q), "\n\tpower iter sketch  : ")
    TIMED_LINE(
    sketch_to_tqrcp(L, k, Q, m, R, k, piv), "\n\tsketch to QRCP     : ")
    auto stop_timer = std_clock::now();
    float runtime = (float) duration_cast<microseconds>(stop_timer - start_timer).count();
    std::cout << "\nTotal runtime      : " << FOUT(runtime / 1e6) << std::endl << std::endl;

    std::cout << "First k pivots : [";
    for (int i = 0; i < k-1; ++i)
        std::cout << piv[i] << ", ";
    std::cout << piv[k-1] << "]\n";

    delete [] Q;
    delete [] R;
    delete [] piv;
    return 0;
}
