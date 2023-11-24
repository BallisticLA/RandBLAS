#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

template <typename T>
struct CSRMatrix : SparseMatrix<T> {
    const int64_t *rowptr;
    const int64_t *colidxs;
};

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csr {

template <typename T>
void csr_to_dense(
    int64_t n_rows,
    int64_t n_cols,
    RandBLAS::sparse_data::IndexBase index_base,
    const T *vals,
    const int64_t *rowptr,
    const int64_t *colidxs,
    int64_t stride_row,
    int64_t stride_col,
    T *mat
) {
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            MAT(i, j) = 0.0;
        }
    }
    // TODO: add error checks for (stride_row, stride_col, n_cols, n_rows)
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t ell = rowptr[i]; ell < rowptr[i+1]; ++ell) {
            int j = colidxs[ell];
            if (index_base == RandBLAS::sparse_data::IndexBase::One)
                j -= 1;
            MAT(i, j) = vals[ell];
        }
    }
    return;
}

template <typename T>
void csr_to_dense(
    RandBLAS::sparse_data::CSRMatrix<T> spmat,
    int64_t stride_row,
    int64_t stride_col,
    T *mat
) {
    return csr_to_dense(
        spmat.n_rows, spmat.n_cols, spmat.index_base, spmat.vals, spmat.rowptr, spmat.colidxs,
        stride_row, stride_col, mat
    );
}

} // end namespace RandBLAS::sparse_data::csr