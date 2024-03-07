#ifndef randblas_sparse_data_csc_multiply
#define randblas_sparse_data_csc_multiply
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include <vector>
#include <algorithm>

namespace RandBLAS::sparse_data::csc {

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static void apply_csc_to_vector_from_left_ki(
    // CSC-format data
    const T *vals,
    sint_t *rowidxs,
    sint_t *colptr,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    int64_t i = 0;
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = v[c * incv];
        while (i < colptr[c+1]) {
            int64_t row = rowidxs[i];
            Av[row * incAv] += (vals[i] * scale);
            i += 1;
        }
    }
}

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
static void apply_regular_csc_to_vector_from_left_ki(
    // data for "regular CSC": CSC with fixed nnz per col,
    // which obviates the requirement for colptr.
    const T *vals,
    sint_t *rowidxs,
    int64_t col_nnz,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = v[c * incv];
        for (int64_t j = c * col_nnz; j < (c + 1) * col_nnz; ++j) {
            int64_t row = rowidxs[j];
            Av[row * incAv] += (vals[j] * scale);
        }
    }
}

template <typename T, RandBLAS::SignedInteger sint_t>
static void apply_csc_left_jki_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    CSCMatrix<T, sint_t> &A,
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

    bool fixed_nnz_per_col = true;
    for (int64_t ell = 2; (ell < m + 1) && fixed_nnz_per_col; ++ell)
        fixed_nnz_per_col = (A.colptr[1] == A.colptr[ell]);

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
            if (fixed_nnz_per_col) {
                apply_regular_csc_to_vector_from_left_ki<T>(
                    vals, A.rowidxs, A.colptr[1],
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                );
            } else {
                apply_csc_to_vector_from_left_ki<T>(
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
