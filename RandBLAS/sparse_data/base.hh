#ifndef randblas_sparse_data_hh
#define randblas_sparse_data_hh

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include <blas.hh>
#include <concepts>


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

template <RandBLAS::SignedInteger sint_t = int64_t>
static inline void sorted_nonzero_locations_to_pointer_array(
    int64_t nnz,
    sint_t *sorted, // length at least max(nnz, last_ptr_index + 1)
    int64_t last_ptr_index
) {
    int64_t i;
    for (i = 1; i < nnz; ++i)
        randblas_require(sorted[i - 1] <= sorted[i]);
    
    auto temp = new sint_t[last_ptr_index + 1];
    temp[0] = 0;
    int64_t ell = 0;
    for (i = 0; i < last_ptr_index; ++i) {
        while (ell < nnz && sorted[ell] == i)
            ++ell;
        temp[i+1] = ell;
    }
    sorted[0] = 0;
    for (i = 0; i < last_ptr_index; ++i)
        sorted[i+1] = temp[i+1];
    delete [] temp;
    return;
}

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
/// .. NOTE: \ttt expands to \texttt (its definition is given in an rst file)
///
/// Any object :math:`\ttt{M}` of type :math:`\ttt{SpMat}` has the following attributes.
///
///     .. list-table::
///        :widths: 25 30 40
///        :header-rows: 1
///        
///        * - 
///          - type
///          - description
///        * - :math:`\ttt{M.n_rows}`
///          - :math:`\ttt{const int64_t}`
///          - number of rows
///        * - :math:`\ttt{M.n_cols}`
///          - :math:`\ttt{const int64_t}`
///          - number of columns
///        * - :math:`\ttt{M.nnz}`
///          - :math:`\ttt{int64_t}`
///          - number of structural nonzeros
///        * - :math:`\ttt{M.vals}`
///          - :math:`\ttt{SpMat::scalar_t *}`
///          - pointer to values of structural nonzeros
///        * - :math:`\ttt{M.own_memory}`
///          - :math:`\ttt{const bool}`
///          - A flag indicating if memory attached to :math:`\ttt{M}` should be deallocated when :math:`\ttt{M}` is deleted.
///            This flag is set automatically based on the type of constructor used for :math:`\ttt{M}.` 
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
///             if (this-own_memory) {
///                 delete [] this->vals;
///                 // ... class-specific code ...
///             }
///         }        
///
/// **Non-owning constructors**
///
///     This concept doesn't place specific requirements constructors for non-owning sparse matrix views of existing data. 
///     However, all of RandBLAS' sparse matrix classes offer such constructors. See individual classes'
///     documentation for details.
///
/// @endverbatim
template<typename SpMat>
concept SparseMatrix = requires(SpMat A) {
    // TODO: figure out why I need to use convertible_to rather than is_same.
    { A.n_rows } -> std::convertible_to<const int64_t>;
    { A.n_cols } -> std::convertible_to<const int64_t>;
    { A.nnz } -> std::convertible_to<int64_t>;
    { *(A.vals) } -> std::convertible_to<typename SpMat::scalar_t>;
    { A.own_memory } ->  std::convertible_to<const bool>;
    { SpMat(A.n_rows, A.n_cols) }; // Is there better way to require a two-argument constructor?
    { A.reserve((int64_t) 10) };  // This will always compile, even though it might error at runtime.
};

} // end namespace RandBLAS::sparse_data

namespace RandBLAS {
    using RandBLAS::sparse_data::IndexBase;
    using RandBLAS::sparse_data::SparseMatrix;
}



#endif
