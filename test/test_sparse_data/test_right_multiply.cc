#include "../linop_common.hh"
#include "common.hh"
#include <gtest/gtest.h>
#include <vector>

using RandBLAS::sparse_data::coo::dense_to_coo;
using namespace test::sparse_data::common;
using namespace test::linop_common;
using blas::Layout;

template <typename T>
COOMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key=0) {
    randblas_require(nonzero_prob >= 0);
    randblas_require(nonzero_prob <= 1);
    COOMatrix<T> A(m, n);
    std::vector<T> actual(m * n);
    RandBLAS::RNGState s(key);
    iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), nonzero_prob, s);
    dense_to_coo<T>(Layout::ColMajor, actual.data(), 0.0, A);
    return A;
}

class TestRightMultiply : public ::testing::Test
{
    // C = alpha * opB(B) @ opA(submat(A)) + beta * C
    //
    //  In what follows, "self" refers to A and "other" refers to B.
    //
    protected:
    virtual void SetUp(){};
    virtual void TearDown(){};

    template <typename T>
    static void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = make_test_matrix<T>(m, n, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0, 0);
    }

    template <typename T>
    static void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        auto A = make_test_matrix<T>(m, n, p, key);
       test_right_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta, 0);
    }

    template <typename T>
    static void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = make_test_matrix<T>(m, n, p, key);
        test_right_apply_tranpose_to_eye<T>(A, layout, 0);
    }

    template <typename T>
    static void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in A, columns in B = eye(m)
        int64_t d0,     // cols in A0
        int64_t m0,     // rows in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // col offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A0 = make_test_matrix<T>(m0, d0, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A0, m, d, A_ro, A_co, layout, 0.0, 0);
    }

    template <typename T>
    static void submatrix_other(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in B
        int64_t n,      // rows in A, columns in B
        int64_t m0,     // rows in B0
        int64_t n0,     // cols in B0
        int64_t B_ro,   // row offset for B in B0
        int64_t B_co,   // col offset for B in B0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A = make_test_matrix<T>(n, d, p, key);
        test_right_apply_to_submatrix<T>(A, m, m0, n0, B_ro, B_co, layout, 0);
    }

    template <typename T>
    static void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // cols in A
        int64_t n,     // rows in A and B
        int64_t m,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = make_test_matrix<T>(n, d, p, key);
        test_right_apply_to_transposed<T>(A, m, layout, 0);
    }
};

////////////////////////////////////////////////////////////////////////
//
//
//      Right-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply, wide_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply, wide_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.80);
    }
}


TEST_F(TestRightMultiply, tall_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply, tall_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

TEST_F(TestRightMultiply, nontrivial_scales_rowmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

////////////////////////////////////////////////////////////////////////
//
//      transpose of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply, transpose_self_double_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self<double>(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self<double>(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self<double>(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply, transpose_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self<double>(key, 30, 200, Layout::RowMajor, 0.01);
        transpose_self<double>(key, 30, 200, Layout::RowMajor, 0.10);
        transpose_self<double>(key, 30, 200, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply, transpose_self_single) {
    for (uint32_t key : {0}) {
        transpose_self<float>(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self<float>(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self<float>(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//      Submatrices of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply, submatrix_self_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_self<double>(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 0.1);
        submatrix_self<double>(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply, submatrix_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_self<double>(key, 3, 10, 8, 12, 2, 1, Layout::RowMajor, 0.1);
        submatrix_self<double>(key, 3, 10, 8, 12, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply, submatrix_self_single) {
    for (uint32_t key : {0}) {
        submatrix_self<float>(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 0.1);
        submatrix_self<float>(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in right-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply, submatrix_other_single) {
    for (uint32_t key : {0}) {
        submatrix_other<float>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other<float>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     transpose of other
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRightMultiply, trans_other_times_sparse_colmajor) {
    uint32_t key = 0;
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply, trans_other_times_sparse_rowmajor) {
    uint32_t key = 0;
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

