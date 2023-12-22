#include "../common.hh"
#include "common.hh"
#include <gtest/gtest.h>
#include <vector>

using RandBLAS::sparse_data::coo::dense_to_coo;
using namespace test::sparse_data::common;
using namespace test::common;
using blas::Layout;

template <typename T>
COOMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
    randblas_require(nonzero_prob >= 0);
    randblas_require(nonzero_prob <= 1);
    COOMatrix<T> A(m, n);
    std::vector<T> actual(m * n);
    RandBLAS::RNGState s(key);
    iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), nonzero_prob, s);
    dense_to_coo<T>(Layout::ColMajor, actual.data(), 0.0, A);
    return A;
}


class TestLeftMultiply : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = make_test_matrix<T>(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0);
    }

    template <typename T>
    static void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        randblas_require(alpha != (T)1.0 || beta != (T)0.0);
        auto A = make_test_matrix<T>(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta);
    }

    template <typename T>
    static void transpose_sparse(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = make_test_matrix<T>(m, n, p, key);
        test_left_apply_transpose_to_eye<T>(A, layout);
    }

    template <typename T>
    static void submatrix_sparse(
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
        auto A0 = make_test_matrix<T>(d0, m0, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A0, d, m, A_ro, A_co, layout, 0.0);
    }

    template <typename T>
    static void submatrix_other(
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
        auto A = make_test_matrix<T>(d, m, p, key);
        randblas_require(m0 > m);
        randblas_require(n0 > n);
        test_left_apply_to_submatrix<T>(A, n, m0, n0, B_ro, B_co, layout);
    }

    template <typename T>
    static void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = make_test_matrix<T>(d, m, p, key);
        test_left_apply_to_transposed<T>(A, n, layout);
    }

};


////////////////////////////////////////////////////////////////////////
//
//
//      Left-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply, tall_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye<double>(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, tall_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye<double>(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, wide_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye<double>(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, wide_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye<double>(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}

TEST_F(TestLeftMultiply, nontrivial_scales_rowmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta<double>(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}


////////////////////////////////////////////////////////////////////////
//
//      transpose of sparse operator
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply, transpose_sparse_double_colmajor) {
    for (uint32_t key : {0}) {
        transpose_sparse<double>(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_sparse<double>(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_sparse<double>(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, transpose_sparse_double_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_sparse<double>(key, 200, 30, Layout::RowMajor, 0.01);
        transpose_sparse<double>(key, 200, 30, Layout::RowMajor, 0.10);
        transpose_sparse<double>(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply, transpose_sparse_single) {
    for (uint32_t key : {0}) {
        transpose_sparse<float>(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_sparse<float>(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_sparse<float>(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//      Submatrices of sparse operator
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply, submatrix_sparse_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_sparse<double>(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 0.1);
        submatrix_sparse<double>(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply, submatrix_sparse_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_sparse<double>(key, 3, 10, 8, 12, 3, 1, Layout::RowMajor, 0.1);
        submatrix_sparse<double>(key, 3, 10, 8, 12, 3, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply, submatrix_sparse_single) {
    for (uint32_t key : {0}) {
        submatrix_sparse<float>(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 0.1);
        submatrix_sparse<float>(key, 3, 10, 8, 12, 3, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in left-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other<double>(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply, submatrix_other_single) {
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


TEST_F(TestLeftMultiply, sparse_times_trans_other_colmajor) {
    uint32_t key = 0;
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other<double>(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply, sparse_times_trans_other_rowmajor) {
    uint32_t key = 0;
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other<double>(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

