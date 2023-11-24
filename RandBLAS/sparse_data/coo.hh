#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

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