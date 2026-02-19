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

#include <gtest/gtest.h>
#include "RandBLAS/sparse_data/random_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include <cmath>

using namespace RandBLAS::sparse_data;


class TestRandomSparseMatrix : public ::testing::Test {};


// ============================================================================
// CSR structural validity
// ============================================================================

TEST_F(TestRandomSparseMatrix, csr_structural_validity_double) {
    int64_t m = 100, n = 200;
    double density = 0.05;
    CSRMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csr(density, A, state);

    EXPECT_EQ(A.rowptr[0], 0);
    EXPECT_EQ(A.rowptr[m], A.nnz);
    for (int64_t i = 0; i < m; ++i) {
        EXPECT_LE(A.rowptr[i], A.rowptr[i + 1]);
        for (int64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
            EXPECT_GE(A.colidxs[k], 0);
            EXPECT_LT(A.colidxs[k], n);
            if (k > A.rowptr[i]) {
                EXPECT_LT(A.colidxs[k - 1], A.colidxs[k]);
            }
        }
    }
}

TEST_F(TestRandomSparseMatrix, csr_structural_validity_float) {
    int64_t m = 80, n = 150;
    double density = 0.1;
    CSRMatrix<float> A(m, n);
    auto state = RandBLAS::RNGState<>(99);
    random_csr(density, A, state);

    EXPECT_EQ(A.rowptr[0], 0);
    EXPECT_EQ(A.rowptr[m], A.nnz);
    for (int64_t i = 0; i < m; ++i) {
        EXPECT_LE(A.rowptr[i], A.rowptr[i + 1]);
        for (int64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
            EXPECT_GE(A.colidxs[k], 0);
            EXPECT_LT(A.colidxs[k], n);
            if (k > A.rowptr[i]) {
                EXPECT_LT(A.colidxs[k - 1], A.colidxs[k]);
            }
        }
    }
}


// ============================================================================
// CSC structural validity
// ============================================================================

TEST_F(TestRandomSparseMatrix, csc_structural_validity_double) {
    int64_t m = 100, n = 200;
    double density = 0.05;
    CSCMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csc(density, A, state);

    EXPECT_EQ(A.colptr[0], 0);
    EXPECT_EQ(A.colptr[n], A.nnz);
    for (int64_t j = 0; j < n; ++j) {
        EXPECT_LE(A.colptr[j], A.colptr[j + 1]);
        for (int64_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k) {
            EXPECT_GE(A.rowidxs[k], 0);
            EXPECT_LT(A.rowidxs[k], m);
            if (k > A.colptr[j]) {
                EXPECT_LT(A.rowidxs[k - 1], A.rowidxs[k]);
            }
        }
    }
}


// ============================================================================
// COO structural validity
// ============================================================================

TEST_F(TestRandomSparseMatrix, coo_structural_validity_double) {
    int64_t m = 100, n = 200;
    double density = 0.05;
    COOMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_coo(density, A, state);

    EXPECT_EQ(A.sort, NonzeroSort::CSR);
    for (int64_t k = 0; k < A.nnz; ++k) {
        EXPECT_GE(A.rows[k], 0);
        EXPECT_LT(A.rows[k], m);
        EXPECT_GE(A.cols[k], 0);
        EXPECT_LT(A.cols[k], n);
    }
    // Verify CSR sort order
    for (int64_t k = 1; k < A.nnz; ++k) {
        bool row_ok = (A.rows[k] > A.rows[k-1]) ||
                      (A.rows[k] == A.rows[k-1] && A.cols[k] > A.cols[k-1]);
        EXPECT_TRUE(row_ok) << "COO not in CSR order at index " << k;
    }
}


// ============================================================================
// Edge cases
// ============================================================================

TEST_F(TestRandomSparseMatrix, csr_density_zero) {
    int64_t m = 10, n = 20;
    CSRMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csr(0.0, A, state);

    EXPECT_EQ(A.nnz, 0);
    ASSERT_NE(A.rowptr, nullptr);
    for (int64_t i = 0; i <= m; ++i)
        EXPECT_EQ(A.rowptr[i], 0);
}

TEST_F(TestRandomSparseMatrix, csc_density_zero) {
    int64_t m = 10, n = 20;
    CSCMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csc(0.0, A, state);

    EXPECT_EQ(A.nnz, 0);
    ASSERT_NE(A.colptr, nullptr);
    for (int64_t j = 0; j <= n; ++j)
        EXPECT_EQ(A.colptr[j], 0);
}

TEST_F(TestRandomSparseMatrix, coo_density_zero) {
    int64_t m = 10, n = 20;
    COOMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_coo(0.0, A, state);
    EXPECT_EQ(A.nnz, 0);
}

TEST_F(TestRandomSparseMatrix, csr_density_one) {
    int64_t m = 5, n = 8;
    CSRMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csr(1.0, A, state);

    EXPECT_EQ(A.nnz, m * n);
    for (int64_t i = 0; i < m; ++i)
        EXPECT_EQ(A.rowptr[i + 1] - A.rowptr[i], n);
}

TEST_F(TestRandomSparseMatrix, csc_density_one) {
    int64_t m = 5, n = 8;
    CSCMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csc(1.0, A, state);

    EXPECT_EQ(A.nnz, m * n);
    for (int64_t j = 0; j < n; ++j)
        EXPECT_EQ(A.colptr[j + 1] - A.colptr[j], m);
}

TEST_F(TestRandomSparseMatrix, coo_density_one) {
    int64_t m = 5, n = 8;
    COOMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_coo(1.0, A, state);
    EXPECT_EQ(A.nnz, m * n);
}

TEST_F(TestRandomSparseMatrix, csr_empty_dimensions) {
    CSRMatrix<double> A1(0, 10);
    auto state = RandBLAS::RNGState<>(42);
    random_csr(0.5, A1, state);
    EXPECT_EQ(A1.nnz, 0);

    CSRMatrix<double> A2(10, 0);
    random_csr(0.5, A2, state);
    EXPECT_EQ(A2.nnz, 0);
}


// ============================================================================
// Determinism: same state → identical matrix
// ============================================================================

TEST_F(TestRandomSparseMatrix, csr_determinism) {
    int64_t m = 50, n = 50;
    double density = 0.1;

    CSRMatrix<double> A1(m, n);
    CSRMatrix<double> A2(m, n);
    auto state = RandBLAS::RNGState<>(99);
    random_csr(density, A1, state);
    random_csr(density, A2, state);

    ASSERT_EQ(A1.nnz, A2.nnz);
    for (int64_t k = 0; k < A1.nnz; ++k) {
        EXPECT_EQ(A1.vals[k], A2.vals[k]);
        EXPECT_EQ(A1.colidxs[k], A2.colidxs[k]);
    }
    for (int64_t i = 0; i <= m; ++i)
        EXPECT_EQ(A1.rowptr[i], A2.rowptr[i]);
}

TEST_F(TestRandomSparseMatrix, coo_determinism) {
    int64_t m = 50, n = 50;
    double density = 0.1;

    COOMatrix<double> A1(m, n);
    COOMatrix<double> A2(m, n);
    auto state = RandBLAS::RNGState<>(99);
    random_coo(density, A1, state);
    random_coo(density, A2, state);

    ASSERT_EQ(A1.nnz, A2.nnz);
    for (int64_t k = 0; k < A1.nnz; ++k) {
        EXPECT_EQ(A1.vals[k], A2.vals[k]);
        EXPECT_EQ(A1.rows[k], A2.rows[k]);
        EXPECT_EQ(A1.cols[k], A2.cols[k]);
    }
}


// ============================================================================
// Statistical: nnz approximately m*n*density
// ============================================================================

TEST_F(TestRandomSparseMatrix, csr_expected_nnz) {
    int64_t m = 1000, n = 1000;
    double density = 0.01;
    int64_t expected = static_cast<int64_t>(m * n * density);  // 10000

    CSRMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csr(density, A, state);

    // 4 standard deviations: std = sqrt(m*n*density*(1-density)) ≈ 99.5
    double std_dev = std::sqrt(m * n * density * (1.0 - density));
    EXPECT_GT(A.nnz, expected - 4 * std_dev);
    EXPECT_LT(A.nnz, expected + 4 * std_dev);
}

TEST_F(TestRandomSparseMatrix, csc_expected_nnz) {
    int64_t m = 1000, n = 1000;
    double density = 0.01;
    int64_t expected = static_cast<int64_t>(m * n * density);

    CSCMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_csc(density, A, state);

    double std_dev = std::sqrt(m * n * density * (1.0 - density));
    EXPECT_GT(A.nnz, expected - 4 * std_dev);
    EXPECT_LT(A.nnz, expected + 4 * std_dev);
}

TEST_F(TestRandomSparseMatrix, coo_expected_nnz) {
    int64_t m = 1000, n = 1000;
    double density = 0.01;
    int64_t expected = static_cast<int64_t>(m * n * density);

    COOMatrix<double> A(m, n);
    auto state = RandBLAS::RNGState<>(42);
    random_coo(density, A, state);

    double std_dev = std::sqrt(m * n * density * (1.0 - density));
    EXPECT_GT(A.nnz, expected - 4 * std_dev);
    EXPECT_LT(A.nnz, expected + 4 * std_dev);
}


// ============================================================================
// State advancement: next_state differs from input
// ============================================================================

TEST_F(TestRandomSparseMatrix, csr_state_advances) {
    CSRMatrix<double> A(50, 50);
    auto state = RandBLAS::RNGState<>(42);
    auto next_state = random_csr(0.1, A, state);
    EXPECT_NE(state, next_state);
}
