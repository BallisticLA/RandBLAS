// Copyright, 2025. See LICENSE for copyright holder information.
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
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/csc_trsm_impl.hh"
#include "RandBLAS/sparse_data/csr_trsm_impl.hh"
#include <vector>
#include <algorithm>


namespace RandBLAS::sparse_data {


template <SparseMatrix SpMat>
inline void trsm_matrix_validation( const SpMat &A, blas::Uplo uplo, int expensive_checks ) {
    /***
    The current implementation requires strong sorting.
    */
    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    int64_t p, ell;
    
    const T* vals = A.vals;
    int64_t m = A.n_rows;
    int64_t flag = -1;
    if constexpr (is_csr) {
        const sint_t* ptrs = A.rowptr;
        const sint_t* idxs = A.colidxs;
        if (expensive_checks > 0) {
            bool ordered_indices = compressed_indices_are_increasing(m, ptrs, idxs, &flag);
            if (!ordered_indices) {
                std::stringstream ss;
                ss << "Ill-formed CSR matrix; indices in row " << flag << " are not sorted.";
                throw RandBLAS::Error(ss.str());
            }
        }
        for (ell = 0; ell < m; ++ell) {
            p = (uplo == blas::Uplo::Lower) ? ptrs[ell+1] - 1 : ptrs[ell];
            randblas_require(idxs[p] == ell);
            randblas_require(vals[p] != 0.0);
        }
    } else {
        const sint_t* ptrs = A.colptr;
        const sint_t* idxs = A.rowidxs;
        if (expensive_checks > 0) {
            bool ordered_indices = compressed_indices_are_increasing(m, ptrs, idxs, &flag);
            if (!ordered_indices) {
                std::stringstream ss;
                ss << "Ill-formed CSC matrix; indices in column " << flag << " are not sorted.";
                throw RandBLAS::Error(ss.str());
            }
        }
        for (ell = 0; ell < m; ++ell) {
            p = (uplo == blas::Uplo::Lower) ? ptrs[ell] : ptrs[ell+1] - 1;
            randblas_require(idxs[p] == ell);
            randblas_require(vals[p] != 0.0);
        }
    }
    return;
}

// NOTE: we call this TRSM instead of "SPTRSM" because it's in the sparse_data namespace.
template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
void left_trsm(
    blas::Layout layout, blas::Op opA, T alpha, const SpMat &A, blas::Uplo uplo, blas::Diag diag, int64_t n, T *B, int64_t ldb,
    int arg_validation_mode = 1
) {
    using blas::Op;
    using blas::Uplo;

    if (opA == Op::Trans) {
        using sint_t = typename SpMat::index_t;
        constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
        constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
        auto trans_uplo = (uplo == Uplo::Lower) ? Uplo::Upper : Uplo::Lower;
        if constexpr (is_csc) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csr(A);
            left_trsm(layout, Op::NoTrans, alpha, At, trans_uplo, diag, n, B, ldb);
        } else if constexpr (is_csr) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csc(A);
            left_trsm(layout, Op::NoTrans, alpha, At, trans_uplo, diag, n, B, ldb);
        } else {
            randblas_require(false);
        }
        return; 
    }

    randblas_require( A.n_rows == A.n_cols );
    randblas_require( A.index_base == IndexBase::Zero );

    using sint_t = typename SpMat::index_t;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    randblas_require(is_csr || is_csc);

    if (arg_validation_mode >= 0) trsm_matrix_validation( A, uplo, arg_validation_mode );

    int64_t m = A.n_rows;
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= m);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(m, alpha, &B[i*ldb], 1);
    } else {
        randblas_require(ldb >= n);
        for (int64_t i = 0; i < m; ++i)
            RandBLAS::util::safe_scal(n, alpha, &B[i*ldb], 1);
    }

    if (alpha == static_cast<T>(0))
        return;

    if constexpr (is_csr) {
        sparse_data::csr::trsm_jki_p11(layout, uplo, diag, n, A, B, ldb);
    } else {
        sparse_data::csc::trsm_jki_p11(layout, uplo, diag, n, A, B, ldb);
    }
    return;
}

}
