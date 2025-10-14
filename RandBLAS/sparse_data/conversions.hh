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
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"

namespace RandBLAS::sparse_data::conversions {

using namespace RandBLAS::sparse_data;
using RandBLAS::SignedInteger;

// MARK: to coo

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csc_to_coo(const CSCMatrix<T, sint_t1> &csc, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == coo.n_cols);
    randblas_require(csc.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    coo.reserve(csc.nnz);
    std::copy( csc.vals,    csc.vals    + csc.nnz, coo.vals );
    std::copy( csc.rowidxs, csc.rowidxs + csc.nnz, coo.rows );
    sorted_idxs_from_compressed_ptr(csc.n_cols, csc.colptr, csc.nnz, coo.cols);
    coo.sort = NonzeroSort::CSC;
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csr_to_coo(const CSRMatrix<T, sint_t1> &csr, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    randblas_require(csr.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    coo.reserve(csr.nnz);
    std::copy( csr.vals,    csr.vals    + csr.nnz, coo.vals );
    std::copy( csr.colidxs, csr.colidxs + csr.nnz, coo.cols );
    sorted_idxs_from_compressed_ptr(csr.n_rows, csr.rowptr, csr.nnz, coo.rows);
    coo.sort = NonzeroSort::CSR;
    return;
}

// MARK: to compressed

template <typename T, SignedInteger sint_t1, SignedInteger sint_t2>
auto coo_to_csc( const COOMatrix<T, sint_t1> &coo, CSCMatrix<T,sint_t2> &csc ) {
    randblas_require( csc.n_rows     == coo.n_rows      );
    randblas_require( csc.n_cols     == coo.n_cols      );
    randblas_require( csc.index_base == IndexBase::Zero );
    randblas_require( csc.index_base == IndexBase::Zero );
    if (coo.nnz == 0) {
        return;
    } else if (coo.sort == NonzeroSort::CSC) {
        csc.reserve(coo.nnz);
        compressed_ptr_from_sorted_idxs(coo.nnz, coo.cols, coo.n_cols, csc.colptr);
        std::copy( coo.vals, coo.vals + coo.nnz, csc.vals    );
        std::copy( coo.rows, coo.rows + coo.nnz, csc.rowidxs );
        return;
    } else {
        auto coo_copy = coo;
        coo_copy.sort_arrays(NonzeroSort::CSC);
        coo_to_csc(coo_copy, csc);
        return;
    }
}

template <typename T, SignedInteger sint_t1, SignedInteger sint_t2>
void coo_to_csr( const COOMatrix<T, sint_t1> &coo, CSRMatrix<T,sint_t2> &csr ) {
    randblas_require( csr.n_rows     == coo.n_rows      );
    randblas_require( csr.n_cols     == coo.n_cols      );
    randblas_require( csr.index_base == IndexBase::Zero );
    randblas_require( csr.index_base == IndexBase::Zero );
    if (coo.nnz == 0) {
        return;
    } else if (coo.sort == NonzeroSort::CSR) {
        csr.reserve(coo.nnz);
        compressed_ptr_from_sorted_idxs(coo.nnz, coo.rows, coo.n_rows, csr.rowptr);
        std::copy( coo.vals, coo.vals + coo.nnz, csr.vals    );
        std::copy( coo.cols, coo.cols + coo.nnz, csr.colidxs );
        return;
    } else {
        auto coo_copy = coo;
        coo_copy.sort_arrays(NonzeroSort::CSR);
        coo_to_csr(coo_copy, csr);
        return;
    }
}

// MARK: transposes

template <typename T, SignedInteger sint_t>
CSRMatrix<T, sint_t> transpose_as_csr(const CSCMatrix<T, sint_t> &A, bool share_memory = true) {
    if (share_memory) {
        CSRMatrix<T, sint_t> At(A.n_cols, A.n_rows, A.nnz, A.vals, A.colptr, A.rowidxs, A.index_base);
        return At;
    } else {
        CSRMatrix<T, sint_t> At(A.n_cols, A.n_rows);
        At.index_base = A.index_base;
        At.reserve(A.nnz);
        std::copy( A.rowidxs, A.rowidxs + A.nnz,        At.colidxs );
        std::copy( A.vals,    A.vals    + A.nnz,        At.vals    );
        std::copy( A.colptr,  A.colptr  + A.n_cols + 1, At.rowptr  );
        return At;
    }
}

template <typename T, SignedInteger sint_t>
CSCMatrix<T, sint_t> transpose_as_csc(const CSRMatrix<T, sint_t> &A, bool share_memory = true) {
    if (share_memory) {
        CSCMatrix<T, sint_t> At(A.n_cols, A.n_rows, A.nnz, A.vals, A.colidxs, A.rowptr, A.index_base);
        return At;
    } else {
        CSCMatrix<T, sint_t> At(A.n_cols, A.n_rows);
        At.index_base = A.index_base;
        At.reserve(A.nnz);
        std::copy( A.colidxs, A.colidxs + A.nnz,        At.rowidxs );
        std::copy( A.vals,    A.vals    + A.nnz,        At.vals    );
        std::copy( A.rowptr,  A.rowptr  + A.n_rows + 1, At.colptr  );
        return At;
    }
}

// MARK: reindexing

/// -----------------------------------------------------------------------
/// Given a RandBLAS SparseMatrix "A" (CSCMatrix, CSRMatrix, or COOMatrix),
/// modify its underlying datastructures as necessary so that it labels 
/// matrix elements in "desired" IndexBase.
/// 
/// Use this to convert between one-based indexing and zero-based indexing.
/// This function returns immediately if desired == A.index_base.
template <SparseMatrix SpMat>
void reindex_inplace(SpMat &A, IndexBase desired);

template <typename T, SignedInteger sint_t>
void reindex_inplace(CSCMatrix<T, sint_t> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.rowidxs[ell] -= 1;
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.rowidxs[ell] += 1;
    }
    A.index_base = desired;
    return;
}

template <typename T, SignedInteger sint_t>
void reindex_inplace(CSRMatrix<T, sint_t> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.colidxs[ell] -= 1;
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.colidxs[ell] += 1;  
    }
    A.index_base = desired;
    return;
}

template <typename T, SignedInteger sint_t>
void reindex_inplace(COOMatrix<T,sint_t> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell) {
            A.rows[ell] -= 1;
            A.cols[ell] -= 1;
        }
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell) {
            A.rows[ell] += 1;
            A.cols[ell] += 1;
        }
    }
    A.index_base = desired;
    return;
}

} // end namespace RandBLAS::sparse_data::conversions
