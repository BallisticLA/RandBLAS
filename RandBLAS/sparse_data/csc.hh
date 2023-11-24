#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

template <typename T>
struct CSCMatrix : SparseMatrix<T> {
    const int64_t *colptr;
    const int64_t *rowidxs;
};

} // end namespace RandBLAS::sparse_data

