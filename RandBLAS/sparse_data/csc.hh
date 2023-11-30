#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

template <typename T>
struct CSCMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
    int64_t *rowidxs;
    int64_t *colptr;
    bool _can_reserve = true;

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(true) {
        this->nnz = 0;
        this->vals = nullptr;
        this->rowidxs = nullptr;
        this->colptr = nullptr;
    };

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        int64_t *rowidxs,
        int64_t *colptr,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(false) {
        this->nnz = nnz;
        this->vals = vals;
        this->rowptr = rowidxs;
        this->colidxs = colptr;
    };

    ~CSCMatrix() {
        if (this->own_memory) {
            delete [] this->rowidxs;
            delete [] this->colptr;
            delete [] this->vals;
        }
    };

    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        this->rowidxs = new int64_t[nnz]{0};
        this->colptr = new int64_t[this->n_cols + 1]{0};
        this->vals = new T[nnz]{0.0};
        this->_can_reserve = false;
    };

};

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csc {



}