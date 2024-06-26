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

#include "test/test_matmul_cores/test_spmm/spmm_test_helpers.hh"
#include <algorithm>
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csc;
using blas::Layout;


template <typename T>
class TestLeftMultiply_CSC : public TestLeftMultiply_Sparse<CSCMatrix<T>> {
    CSCMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        CSCMatrix<T> A(m, n);
        std::vector<T> actual(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        dense_to_csc<T>(Layout::ColMajor, actual.data(), 0.0, A);
        return A;
    }
};

class TestLeftMultiply_CSC_double : public TestLeftMultiply_CSC<double> {};

class TestLeftMultiply_CSC_single : public TestLeftMultiply_CSC<float> {};

////////////////////////////////////////////////////////////////////////
//
//
//      Left-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_CSC_double, tall_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_double, tall_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_double, wide_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_double, wide_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSC_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSC_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSC_double, nontrivial_scales_rowmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}

////////////////////////////////////////////////////////////////////////
//
//      transpose of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_CSC_double, transpose_self_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_double, transpose_self_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::RowMajor, 0.01);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.10);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSC_single, transpose_self) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in left-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_CSC_double, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_CSC_double, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_CSC_double, submatrix_other_single) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     transpose of other
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLeftMultiply_CSC_double, sparse_times_trans_other_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSC_double, sparse_times_trans_other_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}


template <typename T>
class TestRightMultiply_CSC : public TestRightMultiply_Sparse<CSCMatrix<T>> {
    CSCMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        CSCMatrix<T> A(m, n);
        std::vector<T> actual(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        dense_to_csc<T>(Layout::ColMajor, actual.data(), 0.0, A);
        return A;
    }
};

class TestRightMultiply_CSC_double : public TestRightMultiply_CSC<double> {};

class TestRightMultiply_CSC_single : public TestRightMultiply_CSC<float> {};

////////////////////////////////////////////////////////////////////////
//
//
//      Right-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_CSC_double, wide_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSC_double, wide_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}


TEST_F(TestRightMultiply_CSC_double, tall_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSC_double, tall_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSC_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSC_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSC_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

TEST_F(TestRightMultiply_CSC_double, nontrivial_scales_rowmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

////////////////////////////////////////////////////////////////////////
//
//      transpose of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_CSC_double, transpose_self_double_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSC_double, transpose_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::RowMajor, 0.01);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.10);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSC_single, transpose_self_single) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in right-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_CSC_double, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_CSC_double, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_CSC_single, submatrix_other_single) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     transpose of other
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRightMultiply_CSC_double, trans_other_times_sparse_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSC_double, trans_other_times_sparse_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

