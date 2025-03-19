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

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void coo_to_csc(COOMatrix<T, sint_t1> &coo, CSCMatrix<T, sint_t2> &csc) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == csc.n_cols);
    randblas_require(csc.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    sort_coo_data(NonzeroSort::CSC, coo);
    reserve_csc(coo.nnz, csc);
    csc.colptr[0] = 0;
    int64_t ell = 0;
    for (int64_t j = 0; j < coo.n_cols; ++j) {
        while (ell < coo.nnz && coo.cols[ell] == j) {
            csc.rowidxs[ell] = (sint_t2) coo.rows[ell];
            csc.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csc.colptr[j+1] = (sint_t2) ell;
    }
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csc_to_coo(const CSCMatrix<T, sint_t1> &csc, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == coo.n_cols);
    randblas_require(csc.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    reserve_coo(csc.nnz, coo);
    int64_t ell = 0;
    for (int64_t j = 0; j < csc.n_cols; ++j) {
        for (int64_t i = csc.colptr[j]; i < csc.colptr[j+1]; ++i) {
            coo.vals[ell] = csc.vals[ell];
            coo.rows[ell] = (sint_t2) csc.rowidxs[i];
            coo.cols[ell] = (sint_t2) j;
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSC;
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void coo_to_csr(COOMatrix<T, sint_t1> &coo, CSRMatrix<T, sint_t2> &csr) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    randblas_require(csr.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    sort_coo_data(NonzeroSort::CSR, coo);
    reserve_csr(coo.nnz, csr);
    csr.rowptr[0] = (sint_t2) 0;
    int64_t ell = 0;
    for (int64_t i = 0; i < coo.n_rows; ++i) {
        while (ell < coo.nnz && coo.rows[ell] == i) {
            csr.colidxs[ell] = (sint_t2) coo.cols[ell];
            csr.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csr.rowptr[i+1] = (sint_t2) ell;
    }
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csr_to_coo(const CSRMatrix<T, sint_t1> &csr, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    randblas_require(csr.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    reserve_coo(csr.nnz, coo);
    int64_t ell = 0;
    for (int64_t i = 0; i < csr.n_rows; ++i) {
        for (int64_t j = csr.rowptr[i]; j < csr.rowptr[i+1]; ++j) {
            coo.vals[ell] = csr.vals[ell];
            coo.rows[ell] = (sint_t2) i;
            coo.cols[ell] = (sint_t2) csr.colidxs[j];
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSR;
    return;
}

template <typename T, SignedInteger sint_t>
CSRMatrix<T, sint_t> transpose_as_csr(const CSCMatrix<T, sint_t> &A, bool share_memory = true) {
    if (share_memory) {
        CSRMatrix<T, sint_t> At(A.n_cols, A.n_rows, A.nnz, A.vals, A.colptr, A.rowidxs, A.index_base);
        return At;
    } else {
        CSRMatrix<T, sint_t> At(A.n_cols, A.n_rows);
        At.index_base = A.index_base;
        reserve_csr(A.nnz, At);
        for (int64_t i = 0; i < A.nnz; ++i) {
            At.colidxs[i] = A.rowidxs[i];
            At.vals[i] = A.vals[i];
        }
        for (int64_t i = 0; i < A.n_cols + 1; ++i)
            At.rowptr[i] = A.colptr[i];
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
        reserve_csc(A.nnz, At);
        for (int64_t i = 0; i < A.nnz; ++i) {
            At.rowidxs[i] = A.colidxs[i];
            At.vals[i] = A.vals[i];
        }
        for (int64_t i = 0; i < A.n_rows + 1; ++i)
            At.colptr[i] = A.rowptr[i];
        return At;
    }
}

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
