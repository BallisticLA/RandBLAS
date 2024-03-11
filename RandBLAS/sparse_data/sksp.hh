#ifndef randblas_sksp_hh
#define randblas_sksp_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/sparse_data/spgemm.hh"

#include "RandBLAS/exceptions.hh"

namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse_data;


// =============================================================================
/// \fn sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, int64_t i_off_s, int64_t j_off_s,
///     SpMatrix &A, int64_t i_off_a, int64_t j_off_a, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |opA| mathmacro:: \mathrm{opA}
///   .. |opS| mathmacro:: \mathrm{opS}
///
/// @endverbatim
/// Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\submat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, \math{A} is a sparse matrix, and \math{S} is a dense sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What's :math:`\mat(B)`?
///     It's matrix of shape :math:`d \times n`. Its contents are determined by :math:`(B, \ldb)`
///     and "layout", following the same convention as the Level 3 BLAS function "GEMM."
///
/// What are :math:`\submat(S)` and :math:`\submat(A)`?
///     Their shapes are determined implicitly by :math:`(\opS, d, m)` and :math:`(\opA, n, m)`
///     If :math:`{\submat(X)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{X}` whose upper-left corner
///     appears at index :math:`(\texttt{i_off_x}, \texttt{j_off_x})` of :math:`{X}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\submat(A)) = \submat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\submat(A)) = \submat(A)^T}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}
///     - The number of rows in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(A))}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\op(\submat(S))}
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] S
///    A DenseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] i_off_s
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_off_s}, :]}.
///
/// @param[in] j_off_s
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_off_s}]}. 
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] i_off_a
///     A nonnegative integer.
///     - The rows of \math{\submat(A)} are a contiguous subset of rows of \math{A}.
///     - The rows of \math{\submat(A)} start at \math{A[\texttt{i_off_a}, :]}.
///
/// @param[in] j_off_a
///     A nonnnegative integer.
///     - The columns of \math{\submat(A)} are a contiguous subset of columns of \math{A}.
///     - The columns \math{\submat(A)} start at \math{A[:,\texttt{j_off_a}]}. 
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(B)} when reading from \math{B}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::SignedInteger
///             .. math::
///                 \mat(B)[i, j] = B[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\ldb} must be \math{\geq} the length of a row in \math{\mat(B)}.
///
template <typename T, typename SpMatrix, typename RNG>
void sketch_sparse(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(submat(A)) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    DenseSkOp<T, RNG> &S,
    int64_t i_off_s,
    int64_t j_off_s,
    SpMatrix &A,
    int64_t i_off_a,
    int64_t j_off_a,
    T beta,
    T *B,
    int64_t ldb
) {
    // B = op(submat(S)) @ op(submat(A))
    auto [rows_submat_S, cols_submat_S] = dims_before_op(d, m, opS);
    if (!S.buff) {
        T *buff = new T[rows_submat_S * cols_submat_S];
        fill_dense(S.dist, rows_submat_S, cols_submat_S, i_off_s, j_off_s, buff, S.seed_state);
        DenseDist D{rows_submat_S, cols_submat_S, DenseDistName::BlackBox, S.dist.major_axis};
        DenseSkOp<T,RNG> S_(D, S.seed_state, buff);
        sketch_sparse(layout, opS, opA, d, n, m, alpha, S_, 0, 0, A, i_off_a, j_off_a, beta, B, ldb);
        delete [] buff;
        return;
    }

    auto [rows_submat_A, cols_submat_A] = dims_before_op(m, n, opA);
    randblas_require( A.n_rows      >= rows_submat_A + i_off_a );
    randblas_require( A.n_cols      >= cols_submat_A + j_off_a );
    randblas_require( S.dist.n_rows >= rows_submat_S + i_off_s );
    randblas_require( S.dist.n_cols >= cols_submat_S + j_off_s );
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= d);
    } else {
        randblas_require(ldb >= n);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.dist.n_rows, S.dist.n_cols, i_off_s, j_off_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    rspgemm(layout, opS, opA, d, n, m, alpha, S_ptr, lds, A, i_off_a, j_off_a, beta, B, ldb);
    return;
}

}  // end namespace RandBLAS
#endif
