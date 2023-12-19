#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include <functional>
#include <vector>
#include <tuple>


namespace test::common {

using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using RandBLAS::SparseSkOp;
using RandBLAS::DenseSkOp;
using RandBLAS::RNGState;
using RandBLAS::DenseDist;
using RandBLAS::dims_before_op;
using RandBLAS::dims64_t;


template <typename T>
std::vector<T> eye(int64_t n) {
    std::vector<T> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        A[i + n*i] = 1.0;
    return A;
}

template <typename T, typename RNG=r123::Philox4x32>
auto random_matrix(int64_t m, int64_t n, RNGState<RNG> s) {
    std::vector<T> A(m * n);
    DenseDist DA(m, n);
    auto [layout, next_state] = RandBLAS::fill_dense(DA, A.data(), s);
    std::tuple<std::vector<T>, blas::Layout, RNGState<RNG>> t{A, layout, next_state};
    return t;
}

template <typename T>
dims64_t dimensions(COOMatrix<T> &S) {return {S.n_rows, S.n_cols};}

template <typename T>
dims64_t dimensions(CSCMatrix<T> &S) {return {S.n_rows, S.n_cols};}

template <typename T>
dims64_t dimensions(CSRMatrix<T> &S) {return {S.n_rows, S.n_cols}; }

template <typename T>
dims64_t dimensions(SparseSkOp<T> &S) {return {S.dist.n_rows, S.dist.n_cols}; }

template <typename T>
dims64_t dimensions(DenseSkOp<T> &S) {return {S.dist.n_rows, S.dist.n_cols};}

template <typename T>
void to_explicit_buffer(COOMatrix<T> &a, T *mat_a, blas::Layout layout) {
    RandBLAS::sparse_data::coo::coo_to_dense(a, layout, mat_a);
    return;
}

template <typename T>
void to_explicit_buffer(SparseSkOp<T> &a, T *mat_a, blas::Layout layout) {
    auto a_coo = RandBLAS::sparse::coo_view_of_skop(a);
    to_explicit_buffer(a_coo, mat_a, layout);
    return;
}

template <typename T>
void to_explicit_buffer(DenseSkOp<T> &s, T *mat_s, blas::Layout layout) {
    DenseSkOp<T> a(s.dist, s.seed_state);
    auto n_rows = a.dist.n_rows;
    auto n_cols = a.dist.n_cols;
    int64_t stride_row = (layout == blas::Layout::ColMajor) ? 1 : n_cols;
    int64_t stride_col = (layout == blas::Layout::ColMajor) ? n_rows : 1;
    #define MAT_S(_i, _j) mat_s[(_i) * stride_row + (_j) * stride_col ]
    RandBLAS::fill_dense(a);
    int64_t buff_stride_row = (a.layout == blas::Layout::ColMajor) ? 1 : n_cols;
    int64_t buff_stride_col = (a.layout == blas::Layout::ColMajor) ? n_rows : 1;
    #define BUFF(_i, _j) a.buff[(_i) * buff_stride_row + (_j) * buff_stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            MAT_S(i, j) = BUFF(i, j);
        }
    }
    return;
}


////////////////////////////////////////////////////////////////////////
//
//
//      Multiply from the LEFT
//
//
////////////////////////////////////////////////////////////////////////


template <typename T>
void left_apply(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, SparseSkOp<T> &S, int64_t row_offset, int64_t col_offset, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse::lskges(layout, opS, opA, d, n, m, alpha, S, row_offset, col_offset, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T>
void left_apply(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, DenseSkOp<T> &S, int64_t row_offset, int64_t col_offset, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::dense::lskge3(layout, opS, opA, d, n, m, alpha, S, row_offset, col_offset, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T>
void left_apply(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m, T alpha, COOMatrix<T> &S, int64_t row_offset, int64_t col_offset, const T *A, int64_t lda, T beta, T *B, int64_t ldb, int threads = 0) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse_data::coo::lspgemm(layout, opS, opA, d, n, m, alpha, S, row_offset, col_offset, A, lda, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
    return;
}

template <typename T, typename LinOp>
void reference_left_apply(
    blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, T alpha, LinOp &S, int64_t i_os, int64_t j_os, const T *A, int64_t lda, T beta, T *B, T *E, int64_t ldb
){
    randblas_require(d > 0);
    randblas_require(m > 0);
    randblas_require(n > 0);

    // Dimensions of mat(A), rather than op(mat(A))
    auto [rows_S, cols_S] = dimensions(S);
    auto [rows_mat_A   , cols_mat_A   ] = RandBLAS::dims_before_op(m, n, transA);
    auto [rows_submat_S, cols_submat_S] = RandBLAS::dims_before_op(d, m, transS);

    // Sanity checks on dimensions and strides
    int64_t lds, s_row_stride, s_col_stride, pos, size_A, size_B;
    if (layout == blas::Layout::ColMajor) {
        lds = rows_S;
        pos = i_os + lds * j_os;
        randblas_require(lds >= rows_submat_S);
        randblas_require(lda >= rows_mat_A);
        randblas_require(ldb >= d);
        size_A = lda * (cols_mat_A - 1) + rows_mat_A;;
        size_B = ldb * (n - 1) + d;
        s_row_stride = 1;
        s_col_stride = lds;
    } else {
        lds = cols_S;
        pos = i_os * lds + j_os;
        randblas_require(lds >= cols_submat_S);
        randblas_require(lda >= cols_mat_A);
        randblas_require(ldb >= n);
        size_A = lda * (rows_mat_A - 1) + cols_mat_A;
        size_B = ldb * (d - 1) + n;
        s_row_stride = lds;
        s_col_stride = 1;
    }

    auto size_S = rows_S * cols_S;
    std::vector<T> S_dense(size_S);
    std::vector<T> S_dense_abs(size_S);
    to_explicit_buffer(S, S_dense.data(), layout);
    for (int64_t i = 0; i < rows_S; ++i) {
        for (int64_t j = 0; j < cols_S; ++j) {
            auto ell = i * s_row_stride + j * s_col_stride;
            S_dense_abs[ell] = abs(S_dense[ell]);
        }
    }

    // Compute the reference value
    T* S_ptr = S_dense.data();
    blas::gemm(layout, transS, transA, d, n, m,
        alpha, &S_ptr[pos], lds, A, lda, beta, B, ldb
    );

    // Compute the matrix needed for componentwise error bounds.
    std::vector<T> A_abs_vec(size_A);
    T* A_abs = A_abs_vec.data();
    for (int64_t i = 0; i < size_A; ++i)
        A_abs[i] = abs(A[i]);
    if (beta != 0.0) {
        for (int64_t i = 0; i < size_B; ++i)
            E[i] = abs(B[i]);
    }
    T eps = std::numeric_limits<T>::epsilon();
    T err_alpha = (abs(alpha) * m) * (2 * eps);
    T err_beta = abs(beta) * eps;
    T* S_abs_ptr = S_dense_abs.data();
    blas::gemm(layout, transS, transA, d, n, m,
        err_alpha, &S_abs_ptr[pos], lds, A_abs, lda, err_beta, E, ldb
    );
    return;
}

template <typename T, typename LinOp>
void test_left_apply_to_random(
    // B = alpha * S * A + beta*B, where A is m-by-n and random, S is m-by-d, and B is d-by-n and random
    T alpha, LinOp &S, int64_t n, T beta, blas::Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    auto A  = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(99)));
    auto B0 = std::get<0>(random_matrix<T>(d, n, RandBLAS::RNGState(42)));
    std::vector<T> B1(B0);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldb = (is_colmajor) ? d : n;

    // compute S*A. 
    left_apply<T>(
        layout, blas::Op::NoTrans, blas::Op::NoTrans,
        d, n, m,
        alpha, S, 0, 0, A.data(), lda,
        beta, B0.data(), ldb, threads 
    );

    // compute expected result (B1) and allowable error (E)
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout, blas::Op::NoTrans, blas::Op::NoTrans,
        d, n, m,
        alpha, S, 0, 0, A.data(), lda,
        beta, B1.data(), E.data(), ldb
    );

    // check the result
    RandBLAS_Testing::Util::buffs_approx_equal<T>(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    return;
}

template <typename T, typename LinOp>
void test_left_apply_submatrix_to_eye(
    // B = alpha * submat(S0) * eye + beta*B, where S = submat(S) is d1-by-m1 offset by (S_ro, S_co) in S0, and B is random.
    T alpha, LinOp &S0, int64_t d1, int64_t m1, int64_t S_ro, int64_t S_co, blas::Layout layout, T beta = 0.0, int threads = 0
) {
    auto [d0, m0] = dimensions(S0);
    randblas_require(d0 >= d1);
    randblas_require(m0 >= m1);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = m1;
    int64_t ldb = (is_colmajor) ? d1 : m1;

    // define a matrix to be sketched, and create workspace for sketch.
    auto A = eye<T>(m1);
    auto B = std::get<0>(random_matrix<T>(d1, m1, RandBLAS::RNGState(42)));
    std::vector<T> B_backup(B);

    // Perform the sketch
    left_apply(
        layout, blas::Op::NoTrans, blas::Op::NoTrans,
        d1, m1, m1,
        alpha, S0, S_ro, S_co,
        A.data(), lda,
        beta, B.data(), ldb, threads   
    );

    // Check the result
    T *expect = new T[d0 * m0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor) ? d0 : m0; 
    auto [row_stride_s, col_stride_s] = RandBLAS::layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = RandBLAS::layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < m1; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    RandBLAS_Testing::Util::matrices_approx_equal(
        layout, blas::Op::NoTrans,
        d1, m1,
        B.data(), ldb,
        &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

    delete [] expect;
}

template <typename T, typename LinOp>
void test_left_apply_transpose_to_eye(
    // B = S^T * eye, where S is m-by-d, B is d-by-m
    LinOp &S, blas::Layout layout, int threads = 0
) {
    auto [m, d] = dimensions(S);
    auto A = eye<T>(m);
    std::vector<T> B(d * m, 0.0);
    bool is_colmajor = (blas::Layout::ColMajor == layout);
    int64_t ldb = (is_colmajor) ? d : m;
    int64_t lds = (is_colmajor) ? m : d;

    left_apply<T>(
        layout,
        blas::Op::Trans,
        blas::Op::NoTrans,
        d, m, m,
        1.0, S, 0, 0, A.data(), m,
        0.0, B.data(), ldb, threads   
    );

    std::vector<T> S_dense(m * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    RandBLAS_Testing::Util::matrices_approx_equal(
        layout, blas::Op::Trans, d, m,
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}

template <typename T, typename LinOp>
void test_left_apply_to_submatrix(
    // B = S * A, where S is d-by-m, A = A0[A_ro:(A_ro + m), A_co:(A_co + n)], and A0 is random m0-by-n0.
    LinOp &S, int64_t n, int64_t m0, int64_t n0, int64_t A_ro, int64_t A_co, blas::Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    randblas_require(m0 >= m);
    randblas_require(n0 >= n);

    auto A = std::get<0>(random_matrix<T>(m0, n0, RNGState(13)));
    std::vector<T> B0(d * n, 0.0);
    bool is_colmajor = (layout == blas::Layout::ColMajor);
    int64_t lda = (is_colmajor) ? m0 : n0;
    int64_t ldb = (is_colmajor) ? d : n;

    int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
    T *A_ptr = &A.data()[a_offset]; 
    left_apply<T>(
        layout,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A_ptr, lda,
        0.0, B0.data(), ldb, threads   
    );

    std::vector<T> B1(d * n, 0.0);
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A_ptr, lda,
        0.0, B1.data(), E.data(), ldb
    );
    RandBLAS_Testing::Util::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}

template <typename T, typename LinOp>
void test_left_apply_to_transposed(
    // B = S * A^T, where S is d-by-m, A is m-by-n and random
    LinOp &S, int64_t n, blas::Layout layout, int threads = 0
) {
    auto [d, m] = dimensions(S);
    auto At = std::get<0>(random_matrix<T>(n, m, RNGState(101)));
    std::vector<T> B0(d * n, 0.0);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = (is_colmajor) ? n : m;
    int64_t ldb = (is_colmajor) ? d : n;

    left_apply<T>(
        layout,
        blas::Op::NoTrans,
        blas::Op::Trans,
        d, n, m,
        1.0, S, 0, 0,
        At.data(), lda,
        0.0, B0.data(), ldb, threads   
    );

    std::vector<T> B1(d * n, 0.0);
    std::vector<T> E(d * n, 0.0);
    reference_left_apply<T>(
        layout, blas::Op::NoTrans, blas::Op::Trans,
        d, n, m,
        1.0, S, 0, 0,
        At.data(), lda,
        0.0, B1.data(), E.data(), ldb
    );
    RandBLAS_Testing::Util::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * n,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

}


////////////////////////////////////////////////////////////////////////
//
//
//      Multiply from the RIGHT
//
//
////////////////////////////////////////////////////////////////////////

template <typename T>
void right_apply(blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, SparseSkOp<T> &S, int64_t i_os, int64_t j_os, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse::rskges(layout, transA, transS, m, d, n, alpha, A, lda, S, i_os, j_os, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T>
void right_apply(blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, DenseSkOp<T> &S, int64_t i_os, int64_t j_os, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::dense::rskge3(layout, transA, transS, m, d, n, alpha, A, lda, S, i_os, j_os, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T>
void right_apply(blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, COOMatrix<T> &S, int64_t i_os, int64_t j_os, T beta, T *B, int64_t ldb, int threads) {
    #if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        if (threads > 0)
            omp_set_num_threads(threads);
    #else
        UNUSED(threads);
    #endif
    RandBLAS::sparse_data::coo::rspgemm(layout, transA, transS, m, d, n, alpha, A, lda, S, i_os, j_os, beta, B, ldb);
    #if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
    #endif
}

template <typename T, typename LinOp>
void reference_right_apply(
    blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n, T alpha, const T *A, int64_t lda, LinOp &S0, int64_t i_os, int64_t j_os, T beta, T *B, T *E, int64_t ldb
) { 
    using blas::Layout;
    using blas::Op;
    // Check dimensions of submat(S).
    auto [submat_S_rows, submat_S_cols] = RandBLAS::dims_before_op(n, d, transS);
    auto [rows_S, cols_S] = dimensions(S0);
    randblas_require(submat_S_rows <= rows_S);
    randblas_require(submat_S_cols <= cols_S);
    // Check dimensions of mat(A).
    auto [mat_A_rows, mat_A_cols] = RandBLAS::dims_before_op(m, n, transA);
    if (layout == Layout::ColMajor) {
        randblas_require(lda >= mat_A_rows);
    } else {
        randblas_require(lda >= mat_A_cols);
    }
    //
    // Compute B = op(A) op(submat(S)) by left_apply. We start with the identity
    //
    //      B^T = op(submat(S))^T op(A)^T
    //
    // Then we interchange the operator "op" for op(A) and the operator (*)^T.
    //
    //      B^T = op(submat(S))^T op(A^T)
    //
    // We tell left_apply to process (B^T) and (A^T) in the opposite memory layout
    // compared to the layout for (A, B).
    // 
    auto trans_transS = (transS == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    reference_left_apply(
        trans_layout, trans_transS, transA,
        d, m, n, alpha, S0, i_os, j_os, A, lda, beta, B, E, ldb
    );
}

template <typename T, typename LinOp>
void test_right_apply_to_random(
    // B = alpha * A * S + beta * B, where A is m-by-n, S is n-by-d, B is m-by-d and random
    T alpha, LinOp &S, int64_t m, blas::Layout layout, T beta, int threads = 0
) {
    auto [n, d] = dimensions(S);
    auto A  = std::get<0>(random_matrix<T>(m, n, RandBLAS::RNGState(57)));
    auto B0 = std::get<0>(random_matrix<T>(m, d, RandBLAS::RNGState(10)));
    std::vector<T> B1(B0);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldb = (is_colmajor) ? m : d;

    right_apply<T>(
        layout, blas::Op::NoTrans, blas::Op::NoTrans,
        m, d, n, alpha, A.data(), lda, S, 0, 0,
        beta, B0.data(), ldb, threads
    );

    std::vector<T> E(m * d, 0.0);
    reference_right_apply(
        layout, blas::Op::NoTrans, blas::Op::NoTrans,
        m, d, n, alpha, A.data(), lda, S, 0, 0,
        beta, B1.data(), E.data(), ldb
    );

    RandBLAS_Testing::Util::buffs_approx_equal<T>(
        B0.data(), B1.data(), E.data(), m * d,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

}

template <typename T, typename LinOp>
void test_right_apply_submatrix_to_eye(
    // B = alpha * eye * submat(S) + beta*B : submat(S) is n-by-d, eye is n-by-n, B is n-by-d and random
    T alpha, LinOp &S0, int64_t n, int64_t d, int64_t S_ro, int64_t S_co, blas::Layout layout, T beta = 0.0, int threads = 0
) {
    auto [n0, d0] = dimensions(S0);
    randblas_require(n0 >= n);
    randblas_require(d0 >= d);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = n;
    int64_t ldb = (is_colmajor) ? n : d;

    auto A = eye<T>(n);
    auto B = std::get<0>(random_matrix<T>(n, d, RandBLAS::RNGState(11)));
    std::vector<T> B_backup(B);
    right_apply(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A.data(), lda, S0, S_ro, S_co, beta, B.data(), ldb, threads);

    T *expect = new T[n0 * d0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor)? n0 : d0;
    auto [row_stride_s, col_stride_s] = RandBLAS::layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = RandBLAS::layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    RandBLAS_Testing::Util::matrices_approx_equal(
        layout, blas::Op::NoTrans, n, d, B.data(), ldb, &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

    delete [] expect;
}

template <typename T, typename LinOp>
void test_right_apply_tranpose_to_eye(
    // B = eye * S^T, where S is d-by-n, so eye is order n and B is n-by-d
    LinOp &S, blas::Layout layout, int threads = 0
) {
    auto [d, n] = dimensions(S);
    auto A = eye<T>(n);
    std::vector<T> B(n * d, 0.0);
    bool is_colmajor = blas::Layout::ColMajor == layout;
    int64_t ldb = (is_colmajor) ? n : d;
    int64_t lds = (is_colmajor) ? d : n;
    
    right_apply<T>(layout, blas::Op::NoTrans, blas::Op::Trans, n, d, n, 1.0, A.data(), n, S, 0, 0, 0.0, B.data(), ldb, threads);

    std::vector<T> S_dense(n * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    RandBLAS_Testing::Util::matrices_approx_equal(
        layout, blas::Op::Trans, n, d, 
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}

template <typename T, typename LinOp>
void test_right_apply_to_submatrix(
    // B = submat(A) * S, where mat(A) is m0-by-n0, S is n-by-d, and submat(A) is m-by-n, B is m-by-d
    LinOp &S, int64_t m, int64_t m0, int64_t n0, int64_t A_ro, int64_t A_co, blas::Layout layout, int threads = 0
) {
    auto [n, d] = dimensions(S);
    randblas_require(m0 >= m);
    randblas_require(n0 >= n);
    
    auto A = std::get<0>(random_matrix<T>(m0, n0, RandBLAS::RNGState(1)));
    std::vector<T> B0(m * d, 0.0);
    bool is_colmajor = (layout == blas::Layout::ColMajor);
    int64_t lda = (is_colmajor) ? m0 : n0;
    int64_t ldb = (is_colmajor) ? m : d;
 
    int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
    T *A_ptr = &A.data()[a_offset];
    right_apply<T>(layout, blas::Op::NoTrans, blas::Op::NoTrans, m, d, n, 1.0, A_ptr, lda, S, 0, 0, 0.0, B0.data(), ldb, threads);

    std::vector<T> B1(d * m, 0.0);
    std::vector<T> E(d * m, 0.0);
    reference_right_apply<T>(
        layout,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        m, d, n,
        1.0, A_ptr, lda, S, 0, 0,
        0.0, B1.data(), E.data(), ldb
    );
    RandBLAS_Testing::Util::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), d * m,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
    
}

template <typename T, typename LinOp>
void test_right_apply_to_transposed(
    // B = A^T S, where A is n-by-m, S is n-by-d, and B is m-by-d
    LinOp &S, int64_t m, blas::Layout layout, int threads = 0
) {
    auto [n, d] = dimensions(S);
    auto At = std::get<0>(random_matrix<T>(n, m, RNGState(0)));
    std::vector<T> B0(m * d, 0.0);
    bool is_colmajor = (layout == blas::Layout::ColMajor);
    int64_t lda = (is_colmajor) ? n : m;
    int64_t ldb = (is_colmajor) ? m : d;

    right_apply<T>(layout, blas::Op::Trans, blas::Op::NoTrans, m, d, n, 1.0, At, lda, S, 0, 0, 0.0, B0.data(), ldb, threads);

    std::vector<T> B1(m * d, 0.0);
    std::vector<T> E(m * d, 0.0);
    reference_right_apply<T>(
        layout, blas::Op::Trans, blas::Op::NoTrans,
        m, d, n,
        1.0, At.data(), lda, S, 0, 0,
        0.0, B1.data(), E.data(), ldb
    );
    RandBLAS_Testing::Util::buffs_approx_equal(
        B0.data(), B1.data(), E.data(), m * d,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}


} // end namespace test::common

