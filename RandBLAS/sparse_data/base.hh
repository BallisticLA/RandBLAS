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
    int64_t k;
    for (k = 1; k < len_sorted; ++k)
        randblas_require(sorted[k-1] <= sorted[k]);
    int64_t prev, curr, j, update_limit;
    prev = start_val - 1;
    for (k = 0; k < len_sorted; ++k) {
        curr = sorted[k];
        if (curr < start_val)
            continue;
        update_limit = std::min(curr, stop_val);
        for (j = prev + 1; j <= update_limit; ++j)
            compressed[j - start_val] = k;
        prev = curr;
        if (prev >= stop_val)
            break;
    }
    return;
}

} // end namespace RandBLAS::sparse_data

#endif
