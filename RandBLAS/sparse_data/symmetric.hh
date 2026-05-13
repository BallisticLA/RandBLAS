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

#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/exceptions.hh"


namespace RandBLAS::sparse_data {

// =============================================================================
/// Lightweight non-owning wrapper marking a SparseMatrix as symmetric with a
/// stored triangle.
///
/// @verbatim embed:rst:leading-slashes
/// Holds:
///   - A const reference to the underlying SparseMatrix :math:`A.`
///   - :math:`\ttt{blas::Uplo uplo}`: names the triangle of :math:`A` that is
///     structurally populated. The opposite triangle is implied by symmetry.
///
/// The wrapper performs **no validation** of the matrix contents --- it is the
/// caller's responsibility to guarantee that the named triangle is correctly
/// populated and that the opposite triangle is either structurally absent or
/// will be ignored by the SYMM-aware consumer (kernels in this library that
/// accept a ``Symmetric<SpMat>`` agree to consult ``uplo`` and respect this
/// contract). Construction does enforce that :math:`A` is square
/// (``A.n_rows == A.n_cols``) via :math:`\ttt{randblas\_require}`.
///
/// This wrapper is the carrier for symmetric sparse matrices into the
/// :math:`\ttt{spsymm}`-family kernels (project-plans/randblas-symm-plan.md
/// Case C and beyond). It is intentionally separate from the underlying
/// :math:`\ttt{SparseMatrix}` concept so that calling
/// :math:`\ttt{spmm}` / :math:`\ttt{spgemm}` with a ``Symmetric<SpMat>``
/// argument fails to compile rather than silently treating the matrix as
/// general --- a type-system guard against accidental loss of symmetry.
///
/// The wrapper is non-owning: it holds a reference to :math:`A`, not a copy.
/// The caller must keep :math:`A` alive for the lifetime of the wrapper.
/// @endverbatim
template <SparseMatrix SpMat>
struct Symmetric {
    const SpMat& A;
    const blas::Uplo uplo;

    using scalar_t = typename SpMat::scalar_t;
    using index_t = typename SpMat::index_t;

    Symmetric(const SpMat& A_in, blas::Uplo uplo_in) : A(A_in), uplo(uplo_in) {
        randblas_require(A_in.n_rows == A_in.n_cols);
    }
};


// =============================================================================
/// Construct a ``Symmetric`` wrapper around ``A`` with the named triangle.
/// Syntactic sugar for the constructor; lets callers write
/// ``as_symmetric(A, blas::Uplo::Upper)`` and rely on template argument
/// deduction.
template <SparseMatrix SpMat>
inline Symmetric<SpMat> as_symmetric(const SpMat& A, blas::Uplo uplo) {
    return Symmetric<SpMat>(A, uplo);
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS {
    using RandBLAS::sparse_data::Symmetric;
    using RandBLAS::sparse_data::as_symmetric;
}
