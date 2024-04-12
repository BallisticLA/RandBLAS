#ifndef randblas_sparse_data_spmm_dispatch
#define randblas_sparse_data_spmm_dispatch
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
inline void right_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t m, // C is m-by-d
    int64_t d, // op(A) is n-by-d
    int64_t n, // op(B) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SpMat &B,
    int64_t i_off,
    int64_t j_off,
    T beta,
    T *C,
    int64_t ldc
) { 
    //
    // Compute C = op(mat(A)) @ op(submat(B)) by reduction to left_spmm. We start with
    //
    //      C^T = op(submat(B))^T @ op(mat(A))^T.
    //
    // Then we interchange the operator "op(*)" in op(submat(A)) and (*)^T:
    //
    //      C^T = op(submat(B))^T @ op(mat(A)^T).
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
        d, m, n, alpha, B, i_off, j_off, A, lda, beta, C, ldc
    );
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS {

// =============================================================================
/// \fn spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m,
///     int64_t n, int64_t k, T alpha, SpMat &A, int64_t ro_a, int64_t co_a,
///     const T *B, int64_t ldb, T beta, T *C, int64_t ldc
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Perform an SPMM-like operation, multiplying a dense matrix on the left with a (submatrix of a) sparse matrix:
///
/// .. math::
///     \mat(C) = \alpha \cdot \underbrace{\op(\submat(A))}_{m \times k} \cdot \underbrace{\op(\mat(B))}_{k \times n} + \beta \cdot \underbrace{\mat(C)}_{m \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`A` is a sparse matrix.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)` and :math:`\mat(C)`.
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\submat(A)) = \submat(A)`.
///       * If :math:`\opA` == Trans, then :math:`\op(\submat(A)) = \submat(A)^T`.
///
///      opB - [in]
///       * If :math:`\opB` = NoTrans, then :math:`\op(\mat(B)) = \mat(B)`.
///       * If :math:`\opB` = Trans, then :math:`\op(\mat(B)) = \mat(B)^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(C)`.
///       * The number of rows in :math:`\op(\submat(A))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(C)`
///       * The number of columns in :math:`\op(\mat(B))`.
///
///      k - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\submat(A))`
///       * The number of rows in :math:`\op(\mat(B))`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///       * Defines :math:`\submat(A)`.
///
///      ro_a - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(A)` are a contiguous subset of rows of :math:`A.`
///       * The rows of :math:`\submat(A)` start at :math:`A[\texttt{ro_a}, :].`
///
///      co_a - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(A)` are a contiguous subset of columns of :math:`A`.
///       * The columns :math:`\submat(A)` start at :math:`A[:,\texttt{co_a}]`. 
///
///      B - [in]
///       * Pointer to 1D array of real scalars that define :math:`\mat(B)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * The leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`C` need not be set on input.
///
///      C - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(C)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(C)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldc - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(C)` when reading from :math:`C`.
///
/// @endverbatim
template < typename T, SparseMatrix SpMat>
inline void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, int64_t ro_a, int64_t co_a, const T *B, int64_t ldb, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::left_spmm(layout, opA, opB, m, n, k, alpha, A, ro_a, co_a, B, ldb, beta, C, ldc);
    return;
};

// =============================================================================
/// \fn spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m,
///     int64_t n, int64_t k, T alpha, const T* A, int64_t lda,
///     SpMat &B, int64_t ro_b, int64_t co_b, T beta, T *C, int64_t ldc
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Perform an SPMM-like operation, multiplying a dense matrix on the right with a (submatrix of a) sparse matrix:
///
/// .. math::
///     \mat(C) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times k} \cdot \underbrace{\op(\submat(B))}_{k \times n} + \beta \cdot \underbrace{\mat(C)}_{m \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`B` is a sparse matrix.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(C)`.
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\mat(A)) = \mat(A)`.
///       * If :math:`\opA` = Trans, then :math:`\op(\mat(A)) = \mat(A)^T`.
///
///      opB - [in]
///       * If :math:`\opB` = NoTrans, then :math:`\op(\submat(B)) = \submat(B)`.
///       * If :math:`\opB` = Trans, then :math:`\op(\submat(B)) = \submat(B)^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(C)`.
///       * The number of rows in :math:`\op(\mat(A))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(C)`.
///       * The number of columns in :math:`\op(\submat(B))`.
///
///      k - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mat(A))`
///       * The number of rows in :math:`\op(\submat(B))`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A`. 
///
///      B - [in]
///       * A RandBLAS sparse matrix object.
///       * Defines :math:`\submat(B)`.
///
///      ro_b - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(B)` are a contiguous subset of rows of :math:`B`.
///       * The rows of :math:`\submat(B)` start at :math:`B[\texttt{ro_b}, :]`.
///
///      co_b - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(B)` are a contiguous subset of columns of :math:`B`.
///       * The columns :math:`\submat(B)` start at :math:`B[:,\texttt{co_a}]`.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`C` need not be set on input.
///
///      C - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(C)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(C)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldc - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(C)` when reading from :math:`C`.
///
/// @endverbatim
template <typename T, SparseMatrix SpMat>
inline void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T *A, int64_t lda, SpMat &B, int64_t ro_b, int64_t co_b, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::right_spmm(layout, opA, opB, m, n, k, alpha, A, lda, B, ro_b, co_b, B, beta, C, ldc);
    return;
}

}

#endif
