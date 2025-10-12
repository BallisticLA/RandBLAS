// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>


namespace RandBLAS::sparse_data {

using RandBLAS::SignedInteger;

// =============================================================================
/// Indicates whether the (vals, rows, cols) 
/// data of a COO-format sparse matrix
/// are known to be sorted in CSC order, CSR order, or neither of those orders.
///
enum class NonzeroSort : char {
    // ---------------------------------------------------
    /// 
    CSC = 'C',
    // ---------------------------------------------------
    /// 
    CSR = 'R',
    // ---------------------------------------------------
    /// 
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

template <typename T, SignedInteger sint_t>
void alloc_coo_arrays(int64_t nnz, sint_t* &rows, sint_t* &cols, T* &vals) {
    randblas_require(nnz > 0);
    randblas_require(vals == nullptr);
    randblas_require(rows == nullptr);
    randblas_require(cols == nullptr);
    vals = new T[nnz];
    rows = new sint_t[nnz];
    cols = new sint_t[nnz];
    return;
}

template <typename T, SignedInteger sint_t>
void free_coo_arrays(T* &vals, sint_t* &rows, sint_t* &cols) {
    if (vals != nullptr) delete [] vals;
    if (rows != nullptr) delete [] rows;
    if (cols != nullptr) delete [] cols;
    vals = nullptr;
    rows = nullptr;
    cols = nullptr;
}

template <typename T, SignedInteger sint_t>
void sort_coo_arrays(NonzeroSort s, int64_t nnz, T *vals, sint_t *rows, sint_t *cols) {
    // no‐op if no sorting or already sorted
    if (s == NonzeroSort::None) return;
    auto curr_s = coo_sort_type(nnz, rows, cols);
    if (s == curr_s) return;

    // 1) computing the sorting permutation
    std::vector<int64_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    auto cmp_idx = [&](int64_t a, int64_t b) {
        if (s == NonzeroSort::CSR) {
            return increasing_by_csr(rows[a], cols[a], rows[b], cols[b]);
        } else {
            return increasing_by_csc(rows[a], cols[a], rows[b], cols[b]);
        }
    };
    std::sort(perm.begin(), perm.end(), cmp_idx);

    // 2) apply the permutation in‐place by walking each cycle
    //    we need a small visited array to mark which positions are done
    std::vector<char> visited(nnz, 0);

    for (int64_t i = 0; i < nnz; ++i) {
        // skip already‐fixed points or trivial cycles
        if (visited[i] || perm[i] == i) continue;

        // walk the cycle starting at i
        auto cur = i;
        auto saved_val = vals[i];
        auto saved_row = rows[i];
        auto saved_col = cols[i];
        while (!visited[cur]) {
            visited[cur] = 1;
            auto nxt = perm[cur];
            if (nxt == i) { // close the cycle: put saved_* into position cur
                vals[cur] = saved_val;
                rows[cur] = saved_row;
                cols[cur] = saved_col;
            } else {        // move element from nxt → cur
                vals[cur] = vals[nxt];
                rows[cur] = rows[nxt];
                cols[cur] = cols[nxt];
            }
            cur = nxt;
        }
    }
}

// =============================================================================
/// Let \math{\mtxA} denote a sparse matrix with \math{\ttt{nnz}} structural nonzeros.
/// Its COO representation consists of declared dimensions, \math{\ttt{n_rows}}
/// and \math{\ttt{n_cols}}, as well as a triplet of arrays 
/// \math{(\ttt{vals},\ttt{rows},\ttt{cols})} where
/// @verbatim embed:rst:leading-slashes
///
/// .. math::
///
///         \mtxA_{\ttt{rows}[\ell],\ttt{cols}[\ell]} = \ttt{vals}[\ell] \quad\text{for all}\quad  \ell \in \{0,\ldots,\ttt{nnz}-1\}.
///
/// @endverbatim
///  This type conforms to the SparseMatrix concept.
template <typename T, SignedInteger sint_t = int64_t>
struct COOMatrix {

    // ---------------------------------------------------------------------------
    /// Real scalar type used for structural nonzeros in this matrix.
    using scalar_t = T;

    // ---------------------------------------------------------------------------
    /// Signed integer type used in the rows and cols array members.
    using index_t = sint_t; 

    // ----------------------------------------------------------------------------
    ///  The number of rows in this sparse matrix.
    const int64_t n_rows;

    // ----------------------------------------------------------------------------
    ///  The number of columns in this sparse matrix.
    const int64_t n_cols;

    // ----------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to the reference
    ///  members of this matrix (vals, rows, and cols). If true *at destruction time*, then delete []
    ///  will be called on each non-null reference member of this matrix.
    ///
    ///  RandBLAS only writes to this member at construction time.
    ///
    bool own_memory;
    
    // ---------------------------------------------------------------------------
    ///  The number of structral nonzeros in this matrix.
    int64_t nnz;
    
    // ---------------------------------------------------------------------------
    ///  A flag to indicate whether (rows, cols) are interpreted
    ///  with zero-based or one-based indexing.
    IndexBase index_base;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds values for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.vals is non-null.
    T *vals;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds row indices for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.rows is non-null.
    ///  \endinternal
    sint_t *rows;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds column indicies for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.cols is non-null.
    ///  \endinternal
    sint_t *cols;

    // ---------------------------------------------------------------------------
    ///  A flag to indicate if the data in (vals, rows, cols) is sorted in a 
    ///  CSC-like order, a CSR-like order, or neither order.
    NonzeroSort sort;

    // ---------------------------------------------------------------------------
    ///  **Standard constructor.** Initializes n_rows and n_cols at the provided values.
    ///  The vals, rows, and cols members are set to null pointers;
    ///  nnz is set to Zero, index_base is set to
    ///  zero, and COOMatrix::own_memory is set to true.
    ///  
    ///  This constructor is intended for use with COOMatrix::reserve(int64_t nnz).
    ///
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), nnz(0), index_base(IndexBase::Zero),
        vals(nullptr), rows(nullptr), cols(nullptr), sort(NonzeroSort::None) {};

    // ---------------------------------------------------------------------------
    /// **Expert constructor.** Arguments passed to this function are used to initialize members of the same names;
    /// COOMatrix::own_memory is set to false.
    /// If compute_sort_type is true, then the sort member will be computed by inspecting
    /// the contents of (rows, cols). If compute_sort_type is false, then the sort member is set to None.
    /// 
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rows,
        sint_t *cols,
        bool compute_sort_type = true,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), nnz(nnz), index_base(index_base),
        vals(vals), rows(rows), cols(cols) {
        if (compute_sort_type) {
            sort = coo_sort_type(nnz, rows, cols);
        } else {
            sort = NonzeroSort::None;
        }
    };

    ~COOMatrix() {
        if (own_memory) free_coo_arrays(vals, rows, cols);
    };

    void reserve(int64_t arg_nnz) {
        randblas_require(arg_nnz > 0);
        randblas_require(own_memory);
        nnz = arg_nnz;
        alloc_coo_arrays(nnz, rows, cols, vals);
        return;
    };

    void sort_arrays(NonzeroSort s) {
        sort_coo_arrays(s, nnz, vals, rows, cols);
        sort = s;
    };

    /////////////////////////////////////////////////////////////////////
    //
    //      Undocumented functions (don't appear in doxygen)
    //
    /////////////////////////////////////////////////////////////////////

    // move constructor
    COOMatrix(COOMatrix<T, sint_t> &&other) 
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), nnz(other.nnz), index_base(other.index_base),
      vals(nullptr), rows(nullptr), cols(nullptr), sort(other.sort) {
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        std::swap(vals, other.vals);
        other.nnz = 0;
        other.sort = NonzeroSort::None;
    }

    // copy constructor
    COOMatrix(const COOMatrix<T, sint_t> &other)
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(true), nnz(other.nnz), index_base(other.index_base),
      vals(nullptr), rows(nullptr), cols(nullptr), sort(other.sort) {
        if (nnz > 0) {
            reserve(nnz);
            std::copy(other.rows, other.rows + nnz, rows);
            std::copy(other.cols, other.cols + nnz, cols);
            std::copy(other.vals, other.vals + nnz, vals);
        }
    }
};

#ifdef __cpp_concepts
static_assert(SparseMatrix<COOMatrix<float>>);
static_assert(SparseMatrix<COOMatrix<double>>);
#endif

template <typename COOMatrix>
void print_sparse(COOMatrix const &A) {
    std::cout << "COOMatrix information" << std::endl;
    int64_t nnz = A.nnz;
    std::cout << "\tn_rows = " << A.n_rows << std::endl;
    std::cout << "\tn_cols = " << A.n_cols << std::endl;
    if (A.rows != nullptr) {
        std::cout << "\tvector of row indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.rows[i] << ", ";
        }
    } else {
        std::cout << "\trows is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (A.cols != nullptr) {
        std::cout << "\tvector of column indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.cols[i] << ", ";
        }
    } else {
        std::cout << "\tcols is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (A.vals != nullptr) {
        std::cout << "\tvector of values\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.vals[i] << ", ";
        }
    } else {
        std::cout << "\tvals is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    return;
}

/// This function is deprecated; call M.reserve(nnz) instead.
template <typename T, SignedInteger sint_t>
inline void reserve_coo(int64_t nnz, COOMatrix<T,sint_t> &M) {
    randblas_require(M.own_memory);
    M.nnz = nnz;
    alloc_coo_arrays(M.nnz, M.rows, M.cols, M.vals);
    return;
}

/// TODO: mark as deprecated. Figure out if we need to keep it (public vs private API and all...)
template <typename T>
void sort_coo_data(NonzeroSort s, COOMatrix<T> &spmat) {
    sort_coo_arrays(s, spmat.nnz, spmat.vals, spmat.rows, spmat.cols);
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

