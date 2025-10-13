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
#include <concepts>
#include <iostream>


namespace RandBLAS::sparse_data {

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

template <SignedInteger sint_t = int64_t>
static inline void compressed_ptr_from_sorted_idxs(
    int64_t len_idxs, const sint_t *idxs, int64_t num_comp, sint_t *ptr
) {
    for (int64_t i = 1; i < len_idxs; ++i)
        randblas_require(idxs[i - 1] <= idxs[i]);
    ptr[0] = 0;
    int64_t ell = 0;
    for (int64_t i = 0; i < num_comp; ++i) {
        while (ell < len_idxs && idxs[ell] == i)
            ++ell;
        ptr[i+1] = ell;
    }
    return;
}

template <typename T, SignedInteger sint_t>
void alloc_compressed_sparse_arrays(int64_t n_comp, int64_t nnz, T* &vals, sint_t* &idxs, sint_t* &ptr) {
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
    using RandBLAS::sparse_data::SparseMatrix;
}
