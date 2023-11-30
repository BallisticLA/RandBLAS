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

static inline void filter_and_compress_sorted(
    int64_t len_sorted,
    const int64_t *sorted,
    int64_t start_val,
    int64_t stop_val,
    int64_t *compressed
) {
    int64_t ell = 0;
    for (; ell < len_sorted; ++ell) {
        if (sorted[ell] >= start_val)
            break;
    } // ell is now the first index that's in-bounds.
    compressed[0] = ell;
    for (int64_t i = start_val; i < stop_val; ++i) {
        while (ell < len_sorted && sorted[ell] == i) {
            ++ell;
        }
        compressed[(i+1) - start_val] = ell;
    }
    return;
}

} // end namespace RandBLAS::sparse_data

#endif
