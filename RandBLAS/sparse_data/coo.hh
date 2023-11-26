#ifndef randblas_sparse_data_coo
#define randblas_sparse_data_coo
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

enum class NonzeroSort : char {
    CSC = 'C',
    CSR = 'R',
    None = 'N'
};

NonzeroSort coo_sort_type(int64_t nnz, int64_t *rows, int64_t *cols) {
    bool csr_okay = true;
    bool csc_okay = true;
    auto increasing_by_csr = [](int64_t i0, int64_t j0, int64_t i1, int64_t j1) {
        if (i0 > i1) {
            return false;
        } else if (i0 == i1) {
            return j0 <= j1;
        } else {
            return true;
        }
    };
    auto increasing_by_csc = [](int64_t i0, int64_t j0, int64_t i1, int64_t j1) {
        if (j0 > j1) {
            return false;
        } else if (j0 == j1) {
            return i0 <= i1;
        } else {
            return true;
        }
    };
    for (int64_t ell = 1; ell < nnz; ++ell) {
        auto i0 = rows[ell-1];
        auto j0 = cols[ell-1];
        auto i1 = rows[ell];
        auto j1 = cols[ell];
        if (csr_okay) {
            csr_okay = increasing_by_csr(i0, j0, i1, j1);
        }
        if (csc_okay) {
            csc_okay = increasing_by_csc(i0, j0, i1, j1);
        }
        if (!csr_okay && !csc_okay)
            break;
    }
    if (csr_okay) {
        return NonzeroSort::CSR;
    } else if (csc_okay) {
        return NonzeroSort::CSC;
    } else {
        return NonzeroSort::None;
    }
    
}

template <typename T>
struct COOMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
    int64_t *rows;
    int64_t *cols;
    NonzeroSort sort;
    bool _can_reserve = true;

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(true) {
        this->nnz = 0;
        this->vals = nullptr;
        this->rows = nullptr;
        this->cols = nullptr;
        this->sort = NonzeroSort::None;
    };

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        int64_t *rows,
        int64_t *cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(false) {
        this->nnz = nnz;
        this->vals = vals;
        this->rows = rows;
        this->cols = cols;
        this->sort = coo_sort_type(nnz, rows, cols);
    };

    ~COOMatrix() {
        if (this->own_memory) {
            delete [] this->vals;
            delete [] this->rows;
            delete [] this->cols;
        }
    };

    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        this->vals = new T[nnz];
        this->rows = new int64_t[nnz];
        this->cols = new int64_t[nnz];
        this->_can_reserve = false;
    }

};

template <typename T>
void sort_coo_data(
    NonzeroSort s,
    int64_t nnz,
    T *vals,
    int64_t *rows,
    int64_t *cols
) {
    // note: this implementation makes unnecessary copies
    if (s == NonzeroSort::None)
        return;
    
    // get a vector-of-triples representation of the matrix
    using tuple_type = std::tuple<int64_t, int64_t, T>;
    std::vector<tuple_type> nonzeros;
    nonzeros.reserve(nnz);
    for (int64_t ell = 0; ell < nnz; ++ell)
        nonzeros.emplace_back(rows[ell], cols[ell], vals[ell]);

    // sort the vector-of-triples representation
    auto sort_func = [s](tuple_type const& t1, tuple_type const& t2) {
        if (s == NonzeroSort::CSR) {
            if (std::get<0>(t1) < std::get<0>(t2)) {
                return true;
            } else if (std::get<0>(t1) > std::get<0>(t2)) {
                return false;
            } else if (std::get<1>(t1) < std::get<1>(t2)) {
                return true;
            } else {
                return false;
            }
        } else {
            if (std::get<1>(t1) < std::get<1>(t2)) {
                return true;
            } else if (std::get<1>(t1) > std::get<1>(t2)) {
                return false;
            } else if (std::get<0>(t1) < std::get<0>(t2)) {
                return true;
            } else {
                return false;
            }
        }
    };
    std::sort(nonzeros.begin(), nonzeros.end(), sort_func);

    // unpack the vector-of-triples rep into the triple-of-vectors rep
    for (int64_t ell = 0; ell < nnz; ++ell) {
        tuple_type tup = nonzeros[ell];
        vals[ell] = std::get<2>(tup);
        rows[ell] = std::get<0>(tup);
        cols[ell] = std::get<1>(tup);
    }
    return;
}

template <typename T>
void sort_coo_data(
    NonzeroSort s,
    COOMatrix<T> &spmat
) {
    sort_coo_data(s, spmat.vals, spmat.rows, spmat.cols);
    spmat.sort = s;
    return;
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::coo {

template <typename T>
void coo_from_diag(
    T* vals,
    int64_t nnz,
    int64_t offset,
    COOMatrix<T> &spmat
) {
    spmat.reserve(nnz);
    int64_t ell = 0;
    if (offset >= 0) {
        randblas_require(nnz <= spmat.n_rows);
        while (ell < nnz) {
            spmat.rows[ell] = ell;
            spmat.cols[ell] = ell + offset;
            spmat.vals[ell] = vals[ell];
            ++ell;
        }
    } else {
        while (ell < nnz) {
            spmat.rows[ell] = ell - offset;
            spmat.cols[ell] = ell;
            spmat.vals[ell] = vals[ell];
        }
    }
    return;
}

}
#endif
