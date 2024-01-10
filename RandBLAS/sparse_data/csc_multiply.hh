#ifndef randblas_sparse_data_csc_multiply
#define randblas_sparse_data_csc_multiply
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include <vector>
#include <algorithm>

namespace RandBLAS::sparse_data::csc {

template <typename T, RandBLAS::SignedInteger sint_t>
static void apply_csc_left(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    CSCMatrix<T, sint_t> &A0,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A0.index_base == IndexBase::Zero);
    T *vals = A0.vals;
    if (alpha != (T) 1.0) {
        vals = new T[A0.nnz];
        blas::axpy(A0.nnz, alpha, A0.vals, 1, vals, 1);
    }

    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);

    bool fixed_nnz_per_col = true;
    for (int64_t ell = 2; (ell < m + 1) && fixed_nnz_per_col; ++ell)
        fixed_nnz_per_col = (A_colptr[1] == A_colptr[ell]);

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    s = layout_to_strides(layout_C, ldc);
    auto C_inter_col_stride = s.inter_col_stride;
    auto C_inter_row_stride = s.inter_row_stride;

    #pragma omp parallel default(shared)
    {
        const T *B_col = nullptr;
        T *C_col = nullptr;
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < n; k++) {
            B_col = &B[B_inter_col_stride * k];
            C_col = &C[C_inter_col_stride * k];
            if (fixed_nnz_per_col) {
                apply_regular_csc_to_vector_from_left<T>(
                    vals, A.rowidxs, A.colptr[1],
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                );
            } else {
                apply_csc_to_vector_from_left<T>(
                    vals, A.rowidxs, A.colptr,
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                ); 
            }
        }
    }
    if (alpha != (T) 1.0) {
        delete [] vals;
    }
    return;
}

}
#endif
