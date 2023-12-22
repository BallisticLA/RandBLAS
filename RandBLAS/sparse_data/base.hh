#ifndef randblas_sparse_data_hh
#define randblas_sparse_data_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include <blas.hh>

namespace RandBLAS::sparse_data {

enum class IndexBase : char {
    // ---------------------------------------------------------------
    // zero-based indexing
    Zero = 'Z',
    // ---------------------------------------------------------------
    // one-based indexing
    One = 'O'
};

template <typename T>
struct SparseMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
};

template <typename T>
int64_t nnz_in_dense(
    int64_t n_rows,
    int64_t n_cols,
    int64_t stride_row,
    int64_t stride_col,
    T* mat,
    T abs_tol
) {
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    int64_t nnz = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            if (abs(MAT(i, j)) > abs_tol)
                nnz += 1;
        }
    }
    return nnz;
}

template <RandBLAS::SignedInteger sint_t = int64_t>
static inline void sorted_nonzero_locations_to_pointer_array(
    int64_t nnz,
    sint_t *sorted, // length at least max(nnz, last_ptr_index + 1)
    int64_t last_ptr_index
) {
    int64_t i;
    for (i = 1; i < nnz; ++i)
        randblas_require(sorted[i - 1] <= sorted[i]);
    
    auto temp = new sint_t[last_ptr_index + 1];
    temp[0] = 0;
    int64_t ell = 0;
    for (i = 0; i < last_ptr_index; ++i) {
        while (ell < nnz && sorted[ell] == i)
            ++ell;
        temp[i+1] = ell;
    }
    sorted[0] = 0;
    for (i = 0; i < last_ptr_index; ++i)
        sorted[i+1] = temp[i+1];
    delete [] temp;
    return;
}

} // end namespace RandBLAS::sparse_data

#endif
