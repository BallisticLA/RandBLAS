// Copyright, 2026. See LICENSE for copyright holder information.
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

#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/csr_spsymm_impl.hh"
#include <blas.hh>

namespace RandBLAS::sparse_data {

// =============================================================================
// CSC fallback for symmetric sparse-times-dense (side=Left).
//
// A symmetric CSC matrix's arrays, read as CSR, describe A^T = A over the
// same buffers, with the stored triangle name flipped: a CSC Upper entry at
// (i, j) with i <= j appears in the CSR view at (j, i), which is Lower.
// So this kernel is a delegation to csr_spsymm on the lightweight
// A.transpose() view with uplo flipped. Same identity as the MKL path.
template <typename T, SignedInteger sint_t>
void csc_spsymm(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const CSCMatrix<T, sint_t>& A,
    const T* B, int64_t ldb,
    T* Y, int64_t ldy
) {
    auto At = A.transpose();
    blas::Uplo uplo_flipped = (uplo == blas::Uplo::Upper)
        ? blas::Uplo::Lower
        : blas::Uplo::Upper;
    csr_spsymm(layout, uplo_flipped, m, n, alpha, At, B, ldb, Y, ldy);
}

} // namespace RandBLAS::sparse_data
