#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"


namespace test::common {

// need functions that accept linear operators
// and return dense matrix representations in row-major
// or column-major format.

template <typename T>
void to_explicit_buffer(RandBLAS::sparse_data::COOMatrix<T> &a, T *mat_a, blas::Layout layout) {
    RandBLAS::sparse_data::coo::coo_to_dense(a, layout, mat_a);
    return;
}

template <typename T>
void to_explicit_buffer(RandBLAS::SparseSkOp<T> &a, T *mat_a, blas::Layout layout) {
    auto a_coo = RandBLAS::sparse::coo_view_of_skop(a);
    to_explicit_buffer(a_coo, mat_a, layout);
    return;
}

template <typename T>
void to_explicit_buffer(RandBLAS::DenseSkOp<T> &a, T *mat_a, blas::Layout layout) {
    auto n_rows = a.dist.n_rows;
    auto n_cols = a.dist.n_cols;
    int64_t stride_row = (layout == blas::Layout::ColMajor) ? 1 : n_cols;
    int64_t stride_col = (layout == blas::Layout::ColMajor) ? n_rows : 1;
    #define MAT_A(_i, _j) mat_a[(_i) * stride_row * (_j) * stride_col ]
    RandBLAS::dense::fill_dense(a);
    int64_t buff_stride_row = (a.layout == blas::Layout::ColMajor) ? 1 : n_cols;
    int64_t buff_stride_col = (a.layout == blas::Layout::ColMajor) ? n_rows : 1;
    #define BUFF(_i, _j) a.buff[(_i) * buff_stride_row + (_j) * buff_stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            MAT_A(i, j) = BUFF(i, j);
        }
    }
    return;
}


template <typename T, typename LinOp>
void reference_left_apply(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // mat(B) is d-by-n
    int64_t n, // op(mat(A)) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    LinOp &S,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,  // expected value produced by left_apply; compute via GEMM.
    T *E,  // allowable floating point error; apply theory + compute by GEMM.
    int64_t ldb
){
    randblas_require(d > 0);
    randblas_require(m > 0);
    randblas_require(n > 0);

    // Dimensions of mat(A), rather than op(mat(A))
    int64_t rows_mat_A, cols_mat_A, rows_submat_S, cols_submat_S;
    if (transA == blas::Op::NoTrans) {
        rows_mat_A = m;
        cols_mat_A = n;
    } else {
        rows_mat_A = n;
        cols_mat_A = m;
    }
    // Dimensions of submat(S), rather than op(submat(S))
    if (transS == blas::Op::NoTrans) {
        rows_submat_S = d;
        cols_submat_S = m;
    } else {
        rows_submat_S = m;
        cols_submat_S = d;
    }
    // Sanity checks on dimensions and strides
    int64_t lds, s_row_stride, s_col_stride, pos, size_A, size_B;
    if (layout == blas::Layout::ColMajor) {
        lds = S.dist.n_rows;
        pos = i_os + lds * j_os;
        randblas_require(lds >= rows_submat_S);
        randblas_require(lda >= rows_mat_A);
        randblas_require(ldb >= d);
        size_A = lda * (cols_mat_A - 1) + rows_mat_A;;
        size_B = ldb * (n - 1) + d;
        s_row_stride = 1;
        s_col_stride = lds;
    } else {
        lds = S.dist.n_cols;
        pos = i_os * lds + j_os;
        randblas_require(lds >= cols_submat_S);
        randblas_require(lda >= cols_mat_A);
        randblas_require(ldb >= n);
        size_A = lda * (rows_mat_A - 1) + cols_mat_A;
        size_B = ldb * (d - 1) + n;
        s_row_stride = lds;
        s_col_stride = 1;
    }

    auto size_S = S.dist.n_rows * S.dist.n_cols;
    std::vector<T> S_dense(size_S);
    std::vector<T> S_dense_abs(size_S);
    to_explicit_buffer(S, S_dense.data(), layout);
    for (int64_t i = 0; i < S.dist.n_rows; ++i) {
        for (int64_t j = 0; j < S.dist.n_cols; ++j) {
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
void reference_right_apply(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    LinOp &S0,
    int64_t i_os,
    int64_t j_os,
    T beta,
    T *B, // expected value produced by right_apply; compute via GEMM.
    T *E, // allowable floating point error; apply theory + compute by GEMM.
    int64_t ldb
) { 
    using blas::Layout;
    using blas::Op;
    //
    // Check dimensions of submat(S).
    //
    int64_t submat_S_rows, submat_S_cols;
    if (transS == Op::NoTrans) {
        submat_S_rows = n;
        submat_S_cols = d;
    } else {
        submat_S_rows = d;
        submat_S_cols = n;
    }
    randblas_require(submat_S_rows <= S0.dist.n_rows);
    randblas_require(submat_S_cols <= S0.dist.n_cols);
    //
    // Check dimensions of mat(A).
    //
    int64_t mat_A_rows, mat_A_cols;
    if (transA == Op::NoTrans) {
        mat_A_rows = m;
        mat_A_cols = n;
    } else {
        mat_A_rows = n;
        mat_A_cols = m;
    }
    if (layout == Layout::ColMajor) {
        randblas_require(lda >= mat_A_rows);
    } else {
        randblas_require(lda >= mat_A_cols);
    }
    //
    // Compute B = op(A) op(submat(S)) by LSKGES. We start with the identity
    //
    //      B^T = op(submat(S))^T op(A)^T
    //
    // Then we interchange the operator "op" for op(A) and the operator (*)^T.
    //
    //      B^T = op(submat(S))^T op(A^T)
    //
    // We tell LSKGES to process (B^T) and (A^T) in the opposite memory layout
    // compared to the layout for (A, B).
    // 
    auto trans_transS = (transS == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    reference_left_apply(
        trans_layout, trans_transS, transA,
        d, m, n, alpha, S0, i_os, j_os, A, lda, beta, B, E, ldb
    );
}



} // end namespace test::common

