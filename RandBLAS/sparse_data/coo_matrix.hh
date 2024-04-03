#ifndef randblas_sparse_data_coo
#define randblas_sparse_data_coo
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include <vector>
#include <tuple>
#include <algorithm>


namespace RandBLAS::sparse_data {

using RandBLAS::SignedInteger;

// =============================================================================
/// Indicates whether the (rows, cols, vals) data of a COO-format sparse matrix
/// are known to be sorted in CSC order, CSR order, or neither of those orders.
///
enum class NonzeroSort : char {
    CSC = 'C',
    CSR = 'R',
    None = 'N'
};

template <SignedInteger sint_t>
static inline bool increasing_by_csr(sint_t i0, sint_t j0, sint_t i1, sint_t j1) {
    if (i0 > i1) {
        return false;
    } else if (i0 == i1) {
        return j0 <= j1;
    } else {
        return true;
    }
}

template <SignedInteger sint_t>
static inline bool increasing_by_csc(sint_t i0, sint_t j0, sint_t i1, sint_t j1) {
    if (j0 > j1) {
        return false;
    } else if (j0 == j1) {
        return i0 <= i1;
    } else {
        return true;
    }
}

template <SignedInteger sint_t>
static inline NonzeroSort coo_sort_type(int64_t nnz, sint_t *rows, sint_t *cols) {
    bool csc_okay = true;
    bool csr_okay = true;
    for (int64_t ell = 1; ell < nnz; ++ell) {
        auto i0 = rows[ell-1];
        auto j0 = cols[ell-1];
        auto i1 = rows[ell];
        auto j1 = cols[ell];
        if (csc_okay) {
            csc_okay = increasing_by_csc(i0, j0, i1, j1);
        }
        if (csr_okay) {
            csr_okay = increasing_by_csr(i0, j0, i1, j1);
        }
        if (!csc_okay && !csr_okay)
            break;
    }
    if (csc_okay) {
        return NonzeroSort::CSC;
    } else if (csr_okay) {
        return NonzeroSort::CSR;
    } else {
        return NonzeroSort::None;
    }
}

// =============================================================================
/// A COO-format sparse matrix that complies with the SparseMatrix concept.
///
template <typename T, SignedInteger sint_t = int64_t>
struct COOMatrix {
    using scalar_t = T;
    using index_t = sint_t; 
    const int64_t n_rows;
    const int64_t n_cols;
    const bool own_memory;
    int64_t nnz = 0;
    IndexBase index_base;
    T *vals = nullptr;
    // ---------------------------------------------------------------------------
    ///  Row indicies for nonzeros.
    sint_t *rows = nullptr;
    // ---------------------------------------------------------------------------
    ///  Column indicies for nonzeros
    sint_t *cols = nullptr;
    // ---------------------------------------------------------------------------
    ///  A flag to indicate if the data in (rows, cols, vals) is sorted in a 
    ///  CSC-like order, a CSR-like order, or neither order.
    NonzeroSort sort = NonzeroSort::None;

    bool _can_reserve = true;
    // ^ A flag to indicate if we're allowed to allocate new memory for 
    //   (rows, cols, vals). Set to false after COOMatrix.reserve(...) is called.

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rows,
        sint_t *cols,
        bool compute_sort_type,
        IndexBase index_base
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), index_base(index_base) {
        this->nnz = nnz;
        this->vals = vals;
        this->rows = rows;
        this->cols = cols;
        if (compute_sort_type) {
            this->sort = coo_sort_type(nnz, rows, cols);
        } else {
            this->sort = NonzeroSort::None;
        }
    };

    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), index_base(index_base) {};

    // Constructs an empty sparse matrix of given dimensions.
    // Data can't stored in this object until a subsequent call to reserve(int64_t nnz).
    // This constructor initializes \math{\ttt{own_memory(true)},} and so
    // all data stored in this object is deleted once its destructor is invoked.
    //
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : COOMatrix(n_rows, n_cols, IndexBase::Zero) {};


    // ---------------------------------------------------------------------------
    /// @verbatim embed:rst:leading-slashes
    /// Constructs a sparse matrix based on declared dimensions and the data in three buffers
    /// (vals, rows, cols).
    /// This constructor initializes :math:`\ttt{own_memory(false)}`, and
    /// so the provided buffers are unaffected when this object's destructor
    /// is invoked.
    ///
    /// .. dropdown:: Full parameter descriptions
    ///     :animate: fade-in-slide-down
    ///
    ///      n_rows - [in]
    ///       * The number of rows in this sparse matrix.
    ///
    ///      n_cols - [in]
    ///       * The number of columns in this sparse matrix.
    ///
    ///      nnz - [in]
    ///       * The number of structural nonzeros in the matrix.
    ///
    ///      vals - [in]
    ///       * Pointer to array of real numerical type T.
    ///       * The first nnz entries store values of structural nonzeros
    ///         as part of the COO format.
    /// 
    ///      rows - [in]
    ///       * Pointer to array of sint_t.
    ///       * The first nnz entries store row indices as part of the COO format.
    ///
    ///      cols - [in]
    ///       * Pointer to array of sint_t.
    ///       * The first nnz entries store column indices as part of the COO format.
    ///
    ///      compute_sort_type - [in]
    ///       * Indicates if we should parse data in (rows, cols)
    ///         to see if it's already in CSC-like order or CSR-like order.
    ///         If you happen to know the sort order ahead of time then 
    ///         you should set this parameter to false and then manually
    ///         assign M.sort = ``<the order you already know>`` once you
    ///         have a handle on M.
    ///
    /// @endverbatim
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rows,
        sint_t *cols,
        bool compute_sort_type = true
    ) : COOMatrix(n_rows, n_cols, nnz, vals, rows, cols, compute_sort_type, IndexBase::Zero) {};

    ~COOMatrix() {
        if (this->own_memory) {
            delete [] this->vals;
            delete [] this->rows;
            delete [] this->cols;
           
        }
    };

    // @verbatim embed:rst:leading-slashes
    // Attach three buffers of length nnz for (vals, rows, cols).
    // This function can only be called if :math:`\ttt{own_memory == true},`` and
    // it can only be called once.
    //
    // @endverbatim
    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        if (this->nnz > 0) {
            this->vals = new T[nnz];
            this->rows = new sint_t[nnz];
            this->cols = new sint_t[nnz];
        }
        this->_can_reserve = false;
    }

    /////////////////////////////////////////////////////////////////////
    //
    //      Undocumented functions (don't appear in doxygen)
    //
    /////////////////////////////////////////////////////////////////////

    // move constructor
    COOMatrix(COOMatrix<T, sint_t> &&other) 
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), index_base(other.index_base) {
        this->nnz = other.nnz;
        std::swap(this->rows, other.rows);
        std::swap(this->cols, other.cols);
        std::swap(this->vals, other.vals);
        this->_can_reserve = other._can_reserve;
        other.nnz = 0;
    }    

};

template <typename T, SignedInteger sint_t>
void sort_coo_data(NonzeroSort s, int64_t nnz, T *vals, sint_t *rows, sint_t *cols) {
    if (s == NonzeroSort::None)
        return;
    auto curr_s = coo_sort_type(nnz, rows, cols);
    if (curr_s == s)
        return;
    // TODO: fix this implementation so that it's in-place.
    //  (right now we make expensive copies)

    // get a vector-of-triples representation of the matrix
    using tuple_type = std::tuple<sint_t, sint_t, T>;
    std::vector<tuple_type> nonzeros;
    nonzeros.reserve(nnz);
    for (int64_t ell = 0; ell < nnz; ++ell)
        nonzeros.emplace_back(rows[ell], cols[ell], vals[ell]);

    // sort the vector-of-triples representation
    auto sort_func = [s](tuple_type const &t1, tuple_type const &t2) {
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
void sort_coo_data(NonzeroSort s, COOMatrix<T> &spmat) {
    sort_coo_data(s, spmat.nnz, spmat.vals, spmat.rows, spmat.cols);
    spmat.sort = s;
    return;
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS::sparse_data::coo {

using namespace RandBLAS::sparse_data;
using blas::Layout;

// consider:
//      1. Adding optional share_memory flag that defaults to true.
//      2. renaming to transpose_as_coo.
template <typename T>
COOMatrix<T> transpose(COOMatrix<T> &S) {
    COOMatrix<T> St(S.n_cols, S.n_rows, S.nnz, S.vals, S.cols, S.rows, false, S.index_base);
    if (S.sort == NonzeroSort::CSC) {
        St.sort = NonzeroSort::CSR;
    } else if (S.sort == NonzeroSort::CSR) {
        St.sort = NonzeroSort::CSC;
    }
    return St;
}

template <typename T>
void dense_to_coo(int64_t stride_row, int64_t stride_col, T *mat, T abs_tol, COOMatrix<T> &spmat) {
    int64_t n_rows = spmat.n_rows;
    int64_t n_cols = spmat.n_cols;
    int64_t nnz = nnz_in_dense(n_rows, n_cols, stride_row, stride_col, mat, abs_tol);
    spmat.reserve(nnz);
    nnz = 0;
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T val = MAT(i, j);
            if (abs(val) > abs_tol) {
                spmat.vals[nnz] = val;
                spmat.rows[nnz] = i;
                spmat.cols[nnz] = j;
                nnz += 1;
            }
        }
    }
    return;
}

template <typename T>
void dense_to_coo(Layout layout, T* mat, T abs_tol, COOMatrix<T> &spmat) {
    if (layout == Layout::ColMajor) {
        dense_to_coo(1, spmat.n_rows, mat, abs_tol, spmat);
    } else {
        dense_to_coo(spmat.n_cols, 1, mat, abs_tol, spmat);
    }
}

template <typename T>
void coo_to_dense(const COOMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
    randblas_require(spmat.index_base == IndexBase::Zero);
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < spmat.n_rows; ++i) {
        for (int64_t j = 0; j < spmat.n_cols; ++j) {
            MAT(i, j) = 0.0;
        }
    }
    for (int64_t ell = 0; ell < spmat.nnz; ++ell) {
        int64_t i = spmat.rows[ell];
        int64_t j = spmat.cols[ell];
        if (spmat.index_base == IndexBase::One) {
            i -= 1;
            j -= 1;
        }
        MAT(i, j) = spmat.vals[ell];
    }
    return;
}

template <typename T>
void coo_to_dense(const COOMatrix<T> &spmat, Layout layout, T *mat) {
    if (layout == Layout::ColMajor) {
        coo_to_dense(spmat, 1, spmat.n_rows, mat);
    } else {
        coo_to_dense(spmat, spmat.n_cols, 1, mat);
    }
}

} // end namespace RandBLAS::sparse_data::coo

#endif
