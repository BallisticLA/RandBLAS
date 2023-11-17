#ifndef randblas_sparse_data.hh
#define randblas_sparse_data.hh

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
    const int64_t nnz;
    const IndexBase index_base;
    const T *vals;
};

template <typename T>
struct CSCMatrix : SparseMatrix<T> {
    const int64_t *colptr;
    const int64_t *rowidxs;
};

template <typename T>
struct CSRMatrix : SparseMatrix<T> {
    const int64_t *rowptr;
    const int64_t *colidxs;
};

enum class NonzeroSort : char {
    CSC = 'C',
    CSR = 'R',
    None = 'N'
};

template <typename T>
struct COOMatrix : SparseMatrix<T> {
    const int64_t *rows;
    const int64_t *cols;
    const NonzeroSort sort;
};

} // end namespace RandBLAS::sparse_data

#endif
