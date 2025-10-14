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

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include <blas.hh>

#ifdef __cpp_concepts
#include <concepts>
#endif


namespace RandBLAS::sparse_data {

// Files that include base.hh need these forward declarations 
// for functions like COOMatrix::as_csr().
template<typename T, SignedInteger sint_t> struct CSRMatrix;
template<typename T, SignedInteger sint_t> struct CSCMatrix;
template<typename T, SignedInteger sint_t> struct COOMatrix;

enum class IndexBase : int {
    // ---------------------------------------------------------------
    // zero-based indexing
    Zero = 0,
    // ---------------------------------------------------------------
    // one-based indexing
    One = 1
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
void allocate_coo_arrays(int64_t nnz, T* &vals, sint_t* &rows, sint_t* &cols) {
    randblas_require(nnz > 0);
    randblas_require(vals == nullptr);
    randblas_require(rows == nullptr);
    randblas_require(cols == nullptr);
    vals = new T[nnz]{0};
    rows = new sint_t[nnz]{0};
    cols = new sint_t[nnz]{0};
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

template <SignedInteger sint_t>
void fast_permutation_check(int64_t n, sint_t* perm) {
    sint_t min_p = n; sint_t max_p = 0;
    for (sint_t ell = 0; ell < n; ++ell) {
        auto p = perm[ell];
        randblas_require(0 <= p && p < n);
        min_p = MIN(min_p, p);
        max_p = MAX(max_p, p);
    }
    randblas_require(min_p == 0);
    randblas_require(max_p == n - 1);
    return;
}

template <SignedInteger sint_t1, SignedInteger sint_t2>
inline void apply_index_mapper(int64_t len_mapper, const sint_t1* mapper, int64_t len_indices, sint_t2* indices) {
    if (mapper == nullptr) { 
        return;
    }
    for (int64_t i = 0; i < len_indices; ++i) {
        sint_t2 j = indices[i];
        randblas_require(j < len_mapper);
        indices[i] = static_cast<sint_t2>(mapper[j]);
    }
    return;
}

template <SignedInteger sint_t>
void symperm_coo_arrays(int64_t n, const sint_t* perm, int64_t nnz, sint_t* rows, sint_t* cols) {
    std::vector<sint_t> invperm(n);
    for (sint_t k = 0; k < n; ++k) { invperm[perm[k]] = k; }
    sint_t* iperm = invperm.data();
    apply_index_mapper(n,  perm, nnz, rows);
    apply_index_mapper(n, iperm, nnz, cols);
    return;
}

template <SignedInteger sint_t1, SignedInteger sint_t2>
void compressed_ptr_from_sorted_idxs(int64_t len_idxs, const sint_t1 *idxs, int64_t num_comp, sint_t2 *ptr) {
    for (int64_t i = 1; i < len_idxs; ++i)
        randblas_require(idxs[i - 1] <= idxs[i]);
    ptr[0] = 0;
    int64_t ell = 0;
    for (int64_t i = 0; i < num_comp; ++i) {
        while (ell < len_idxs && idxs[ell] == i)
            ++ell;
        ptr[i+1] = static_cast<sint_t2>(ell);
    }
    return;
}

template <SignedInteger sint_t1, SignedInteger sint_t2>
void sorted_idxs_from_compressed_ptr(int64_t num_comp, sint_t1* ptr, int64_t len_idxs, sint_t2* idxs) {
    randblas_require(len_idxs >= ptr[num_comp]);
    for (int64_t j = 0; j < num_comp; ++j) {
        auto col_nnz = ptr[j+1] - ptr[j];
        randblas_require(col_nnz >= 0);
        std::fill(idxs, idxs + col_nnz, j);
        idxs = idxs + col_nnz;
    }
}

template <typename T, SignedInteger sint_t>
void allocate_compressed_sparse_arrays(int64_t n_comp, int64_t nnz, T* &vals, sint_t* &idxs, sint_t* &ptr) {
    randblas_require(nnz > 0);
    randblas_require(idxs == nullptr);
    randblas_require(vals == nullptr);
    idxs = new sint_t[nnz]{0};
    vals = new T[nnz]{0.0};
    // Do the ptr check last. This function is called within CSRMatrix.reserve() and
    // CSCMatrix.reserve(), and if we raise an exception for ptr then these contexts
    // will still want vals and idxs to be handled correctly.
    randblas_require(ptr == nullptr);
    ptr  = new sint_t[n_comp + 1]{0};
    return;
}

template <typename T, SignedInteger sint_t>
void free_compressed_sparse_arrays(T* &vals, sint_t* &idxs, sint_t* &ptr) {
    if (idxs != nullptr) {delete [] idxs; idxs = nullptr; }
    if (vals != nullptr) {delete [] vals; vals = nullptr; }
    if (ptr  != nullptr) {delete [] ptr;  ptr  = nullptr; }
    return;
}

template <SignedInteger sint_t>
static bool compressed_indices_are_increasing(int64_t num_vecs, sint_t *ptrs, sint_t *idxs, int64_t *failed_ind = nullptr) {
    // This function uses a pointer for the flag instead of passing an integer
    // by value because the flag is an optional argument (so it needs a default value).
    for (int64_t i = 0; i < num_vecs; ++i) {
        for (int64_t j = ptrs[i]; j < ptrs[i+1]-1; ++j) {
            if (idxs[j+1] <= idxs[j]) {
                if (failed_ind != nullptr) *failed_ind = j;
                return false;
            }
        }
    }
    return true;
}


// Idea: change all "const" attributes to for SpMatrix to return values from inlined functions. 
// Looks like there'd be no collision with function/property names for sparse matrix
// types in Eigen, SuiteSparse, OneMKL, etc.. These inlined functions could return
// nominally public members like A._n_rows and A._n_cols, which the user will only change
// at their own peril.

#ifdef __cpp_concepts
// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// An object :math:`\ttt{M}` of type :math:`\ttt{SpMat}` has the following attributes.
///
/// .. list-table::
///    :widths: 25 30 40
///    :header-rows: 1
///    
///    * - 
///      - type
///      - description
///    * - :math:`\ttt{M.n_rows}`
///      - :math:`\ttt{const int64_t}`
///      - number of rows
///    * - :math:`\ttt{M.n_cols}`
///      - :math:`\ttt{const int64_t}`
///      - number of columns
///    * - :math:`\ttt{M.nnz}`
///      - :math:`\ttt{int64_t}`
///      - number of structural nonzeros
///    * - :math:`\ttt{M.vals}`
///      - :math:`\ttt{SpMat::scalar_t *}`
///      - pointer to values of structural nonzeros
///    * - :math:`\ttt{M.own_memory}`
///      - :math:`\ttt{bool}`
///      - A flag indicating if memory attached to :math:`\ttt{M}` should be deallocated when :math:`\ttt{M}` is deleted.
///        This flag is set automatically based on the type of constructor used for :math:`\ttt{M}.` 
/// 
///
/// **Memory-owning constructors**
/// 
///     :math:`\ttt{SpMat}` must have a constructor for an empty matrix of given dimensions.
///     Conformant implementations of this constructor look like the following.
///
///     .. code:: c++
///
///        SpMat(int64_t n_rows, int64_t n_cols) 
///         : n_rows(n_rows), n_cols(n_cols), nnz(0), vals(nullptr), own_memory(true) {
///             // class-specific code ...
///         };
///
///     If we construct :math:`\ttt{SpMat M(m, n)},` then we can't store data in :math:`\ttt{M}` until a function call
///     of the form :math:`\ttt{M.reserve(nnz)}.` Here's an outline of a conformant implementation of this function.
///
///     .. code:: c++
///
///         void reserve(int64_t nnz) {
///             assert this->own_memory;
///             this->nnz = nnz;
///             this->vals = new SpMat::scalar_t[nnz];
///             // ... class-specific code ...
///         }
///
///     The destructor of :math:`\ttt{M}` is responsible for deallocating :math:`\ttt{M.vals}` and other
///     attached data. A conformant implemnentation of the destructor will look like the following.
///
///     .. code:: c++
///
///         ~SpMat() {
///             if (own_memory) {
///                 if (vals != nullptr) delete [] vals;
///                 // ... class-specific code ...
///             }
///         }        
///
/// **View constructors**
///
///     This concept doesn't place requirements on constructors for sparse matrix views of existing data. 
///     However, all of RandBLAS' sparse matrix classes offer such constructors. See individual classes'
///     documentation for details.
///
/// @endverbatim
template<typename SpMat>
concept SparseMatrix = requires(SpMat A) {
    { A.n_rows }     -> std::same_as<const int64_t&>;
    { A.n_cols }     -> std::same_as<const int64_t&>;
    { A.nnz }        -> std::same_as<int64_t&>;
    { *(A.vals) }    -> std::same_as<typename SpMat::scalar_t&>;
    { A.own_memory } -> std::same_as<bool&>;
    { SpMat(A.n_rows, A.n_cols) };
};
#else
#define SparseMatrix typename
#endif

} // end namespace RandBLAS::sparse_data

namespace RandBLAS {
    using RandBLAS::sparse_data::IndexBase;
#ifdef __cpp_concepts
    using RandBLAS::sparse_data::SparseMatrix;
#endif
}
