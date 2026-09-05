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

#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/symmetric.hh"
#include "RandBLAS/exceptions.hh"

#include <gtest/gtest.h>
#include <type_traits>

using namespace RandBLAS::sparse_data;


class TestSymmetricWrapper : public ::testing::Test {};


// Construct from each of the three sparse formats and confirm field access +
// trait aliases. The wrapper does not require any actual data, since it only
// stores a reference + uplo.

TEST_F(TestSymmetricWrapper, constructs_from_coo) {
    COOMatrix<double> A(4, 4);
    auto wrapped = as_symmetric(A, blas::Uplo::Upper);
    EXPECT_EQ(&wrapped.A, &A);
    EXPECT_EQ(wrapped.uplo, blas::Uplo::Upper);
    EXPECT_EQ(wrapped.A.n_rows, 4);
    EXPECT_EQ(wrapped.A.n_cols, 4);
    EXPECT_EQ(wrapped.n_rows, 4);
    EXPECT_EQ(wrapped.n_cols, 4);
    static_assert(std::is_same_v<typename decltype(wrapped)::scalar_t, double>,
                  "Symmetric<COOMatrix<double>>::scalar_t must be double");
}

TEST_F(TestSymmetricWrapper, constructs_from_csr) {
    CSRMatrix<float> A(5, 5);
    auto wrapped = as_symmetric(A, blas::Uplo::Lower);
    EXPECT_EQ(&wrapped.A, &A);
    EXPECT_EQ(wrapped.uplo, blas::Uplo::Lower);
    EXPECT_EQ(wrapped.A.n_rows, 5);
    static_assert(std::is_same_v<typename decltype(wrapped)::scalar_t, float>,
                  "Symmetric<CSRMatrix<float>>::scalar_t must be float");
}

TEST_F(TestSymmetricWrapper, constructs_from_csc) {
    CSCMatrix<double> A(3, 3);
    auto wrapped = as_symmetric(A, blas::Uplo::Upper);
    EXPECT_EQ(&wrapped.A, &A);
    EXPECT_EQ(wrapped.uplo, blas::Uplo::Upper);
    EXPECT_EQ(wrapped.A.n_rows, 3);
}

TEST_F(TestSymmetricWrapper, exports_at_randblas_scope) {
    // The wrapper and the helper must be accessible via the top-level
    // RandBLAS:: namespace (re-exported from sparse_data::).
    COOMatrix<double> A(2, 2);
    RandBLAS::Symmetric<COOMatrix<double>> wrapped_typed(A, blas::Uplo::Upper);
    EXPECT_EQ(&wrapped_typed.A, &A);
    auto wrapped_sugar = RandBLAS::as_symmetric(A, blas::Uplo::Lower);
    EXPECT_EQ(wrapped_sugar.uplo, blas::Uplo::Lower);
}

TEST_F(TestSymmetricWrapper, non_square_rejected_at_construction) {
    COOMatrix<double> A(3, 4); // not square
    EXPECT_THROW({
        auto wrapped = as_symmetric(A, blas::Uplo::Upper);
        (void) wrapped;
    }, RandBLAS::Error);
}
