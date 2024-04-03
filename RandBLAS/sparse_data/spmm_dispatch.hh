#ifndef randblas_sparse_data_spgemm
#define randblas_sparse_data_spgemm
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/csc_spmm_impl.hh"
#include "RandBLAS/sparse_data/csr_spmm_impl.hh"
#include "RandBLAS/sparse_data/coo_spmm_impl.hh"
#include <vector>
#include <algorithm>


namespace RandBLAS::sparse_data {

template <typename T, SparseMatrix SpMat>
void left_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t d, // C is d-by-n
    int64_t n, // \op(B) is m-by-n
    int64_t m, // \op(A) is d-by-m
    T alpha,
    SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    const T *B,
    int64_t ldb,
    T beta,
    T *C,
    int64_t ldc
) {
    using blas::Layout;
    using blas::Op;
    // handle applying a transposed sparse matrix.
    if (opA == Op::Trans) {
        using sint_t = typename SpMat::index_t;
        constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
        constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
        constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
        if constexpr (is_coo) {
            auto At = RandBLAS::sparse_data::coo::transpose(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else if constexpr (is_csc) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csr(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else if constexpr (is_csr) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csc(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else {
            randblas_require(false);
        }
        return; 
    }
    // Below this point, we can assume A is not transposed.
    randblas_require( A.index_base == IndexBase::Zero );
    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    randblas_require(is_coo || is_csr || is_csc);

    if constexpr (is_coo) {
        randblas_require(A.n_rows >= d);
        randblas_require(A.n_cols >= m);
    } else {
        randblas_require(A.n_rows == d);
        randblas_require(A.n_cols == m);
        randblas_require(ro_a == 0);
        randblas_require(co_a == 0);
    }
    
    // Dimensions of B, rather than \op(B)
    Layout layout_C = layout;
    Layout layout_opB;
    int64_t rows_B, cols_B;
    if (opB == Op::NoTrans) {
        rows_B = m;
        cols_B = n;
        layout_opB = layout;
    } else {
        rows_B = n;
        cols_B = m;
        layout_opB = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    }

    // Check dimensions and compute C = beta * C.
    //      Note: both B and C are checked based on "layout"; B is *not* checked on layout_opB.
    if (layout == Layout::ColMajor) {
        randblas_require(ldb >= rows_B);
        randblas_require(ldc >= d);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &C[i*ldc], 1);
    } else {
        randblas_require(ldc >= n);
        randblas_require(ldb >= cols_B);
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &C[i*ldc], 1);
    }

    if (alpha == (T) 0)
        return;
    
    // compute the matrix-matrix product
    if constexpr (is_coo) {
        using RandBLAS::sparse_data::coo::apply_coo_left_jki_p11;
        apply_coo_left_jki_p11(alpha, layout_opB, layout_C, d, n, m, A, ro_a, co_a, B, ldb, C, ldc);
    } else if constexpr (is_csc) {
        using RandBLAS::sparse_data::csc::apply_csc_left_jki_p11;
        apply_csc_left_jki_p11(alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
    } else {
        using RandBLAS::sparse_data::csr::apply_csr_left_jik_p11;
        apply_csr_left_jik_p11(alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
    }
    return;
}

template <typename T, SparseMatrix SpMat>
void right_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t m, // C is m-by-d
    int64_t d, // op(A) is n-by-d
    int64_t n, // op(B) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SpMat &B0,
    int64_t i_off,
    int64_t j_off,
    T beta,
    T *C,
    int64_t ldc
) { 
    //
    // Compute C = op(mat(A)) @ op(submat(B0)) by reduction to left_spmm. We start with
    //
    //      C^T = op(submat(B0))^T @ op(mat(A))^T.
    //
    // Then we interchange the operator "op(*)" in op(submat(A)) and (*)^T:
    //
    //      C^T = op(submat(B0))^T @ op(mat(A)^T).
    //
    // We tell left_spmm to process (C^T) and (B^T) in the opposite memory layout
    // compared to the layout for (B, C).
    // 
    using blas::Layout;
    using blas::Op;
    auto trans_opB = (opB == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    left_spmm(
        trans_layout, trans_opB, opA,
        d, m, n, alpha, B0, i_off, j_off, A, lda, beta, C, ldc
    );
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS {

template < typename T, SparseMatrix SpMat>
void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, int64_t i_off, int64_t j_off, const T *B, int64_t ldb, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::left_spmm(layout, opA, opB, m, n, k, alpha, A, i_off, j_off, B, ldb, beta, C, ldc);
    return;
};

template <typename T, SparseMatrix SpMat>
void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T *A, int64_t lda, SpMat &B, int64_t i_off, int64_t j_off, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::right_spmm(layout, opA, opB, m, n, k, alpha, A, lda, B, i_off, j_off, B, beta, C, ldc);
    return;
}

}

#endif
