#ifndef randblas_sparse_data_csr_multiply
#define randblas_sparse_data_csr_multiply
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include <vector>
#include <algorithm>


namespace RandBLAS::sparse_data::csr {

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static void apply_csr_to_vector_from_left_ik(
    // CSR-format data
    const T *vals,
    sint_t *rowptr,
    sint_t *colidxs,
    // input-output vector data
    const T *v,
    int64_t incv,   // stride between elements of v
    int64_t len_Av,
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    for (int64_t i = 0; i < len_Av; ++i) {
        for (int64_t ell = rowptr[i]; ell < rowptr[i+1]; ++ell) {
            int j = colidxs[ell];
            T Aij = vals[ell];
            Av[i*incAv] += Aij * v[j*incv];
            // ^ if v were a matrix, this could be an axpy with the j-th row of v, accumulated into i-th row of Av.
        }
    }
}

template <typename T, RandBLAS::SignedInteger sint_t>
static void apply_csr_left_jik_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    CSRMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);
    T *vals = A.vals;
    if (alpha != (T) 1.0) {
        vals = new T[A.nnz]{};
        blas::axpy(A.nnz, alpha, A.vals, 1, vals, 1);
    }

    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);

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
        for (int64_t j = 0; j < n; j++) {
            B_col = &B[B_inter_col_stride * j];
            C_col = &C[C_inter_col_stride * j];
            apply_csr_to_vector_from_left_ik(
                   vals, A.rowptr, A.colidxs,
                   B_col, B_inter_row_stride,
                d, C_col, C_inter_row_stride
            );
        }
    }
    if (alpha != (T) 1.0) {
        delete [] vals;
    }
    return;
}

} // end namespace RandBLAS::sparse_data::csr

#endif
