#include "../linop_common.hh"
#include "common.hh"
#include <gtest/gtest.h>
#include <vector>

using RandBLAS::sparse_data::coo::dense_to_coo;
using namespace test::sparse_data::common;
using namespace test::linop_common;
using blas::Layout;


template <typename SpMatrix>
class TestLeftMultiply_Sparse : public ::testing::Test
{
    using T = typename SpMatrix::scalar_t;
    // C = alpha * opA(submat(A)) @ opB(B) + beta * C
    // In what follows, "self" refers to A and "other" refers to B.
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    virtual SpMatrix make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) = 0;

    void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0);
    }

    void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        randblas_require(alpha != (T)1.0 || beta != (T)0.0);
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta);
    }

    void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_transpose_to_eye<T>(A, layout);
    }

    void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // rows in A
        int64_t m,      // columns in A, rows in B = eye(m)
        int64_t d0,     // rows in A0
        int64_t m0,     // cols in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // column offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        randblas_require(d0 > d);
        randblas_require(m0 > m);
        auto A0 = this->make_test_matrix(d0, m0, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A0, d, m, A_ro, A_co, layout, 0.0);
    }

    void submatrix_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        int64_t m0,    // rows in B0
        int64_t n0,    // cols in B0
        int64_t B_ro,  // row offset for B in B0
        int64_t B_co,  // column offset for B in B0
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        randblas_require(m0 > m);
        randblas_require(n0 > n);
        test_left_apply_to_submatrix<T>(A, n, m0, n0, B_ro, B_co, layout);
    }

    void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        test_left_apply_to_transposed<T>(A, n, layout);
    }

};


template <typename T>
class TestLeftMultiply_COO : public TestLeftMultiply_Sparse<COOMatrix<T>> {
    COOMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        COOMatrix<T> A(m, n);
        std::vector<T> actual(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        dense_to_coo<T>(Layout::ColMajor, actual.data(), 0.0, A);
        return A;
    }
};

class TestLeftMultiply_COO_double : public TestLeftMultiply_COO<double> {};

class TestLeftMultiply_COO_single : public TestLeftMultiply_COO<float> {};


////////////////////////////////////////////////////////////////////////
//
//
//      Left-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_COO_double, tall_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_double, tall_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_double, wide_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_double, wide_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_COO_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_COO_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}

TEST_F(TestLeftMultiply_COO_double, nontrivial_scales_rowmajor2) {
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

TEST_F(TestLeftMultiply_COO_double, transpose_self_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_double, transpose_self_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::RowMajor, 0.01);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.10);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_COO_single, transpose_self) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//      Submatrices of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_COO_double, submatrix_self_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_COO_double, submatrix_self_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::RowMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_COO_single, submatrix_self) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in left-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_COO_double, submatrix_other_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_COO_double, submatrix_other_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_COO_single, submatrix_other) {
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


TEST_F(TestLeftMultiply_COO_double, sparse_times_trans_other_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_COO_double, sparse_times_trans_other_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

