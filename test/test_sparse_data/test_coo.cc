#include "test/test_sparse_data/test_left_multiply.hh"
#include "test/test_sparse_data/test_right_multiply.hh"
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::coo;
using namespace test::sparse_data::common;
using blas::Layout;


class TestCOO : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    void test_to_from_dense(int64_t n) {
        COOMatrix<T> A(n, n);
        std::vector<T> actual(n * n);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(n, n, Layout::ColMajor, actual.data(), 0.8, s);

        dense_to_coo(1, n, actual.data(), 0.0, A);
        std::vector<T> expect(n * n);
        coo_to_dense(A, 1, n, expect.data());

        test::comparison::buffs_approx_equal(actual.data(), expect.data(), n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        EXPECT_GT(A.nnz, 0);
        EXPECT_LT(A.nnz, n*n);
        return;
    }

    template <typename T = double>
    void test_sort_order(int64_t n) {
        COOMatrix<T> A(n, n);
        std::vector<T> mat(n * n);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(n, n, Layout::ColMajor, mat.data(), 0.25, s);
        dense_to_coo(1, n, mat.data(), 0.0, A);

        sort_coo_data(NonzeroSort::CSC, A);
        EXPECT_EQ(A.sort, NonzeroSort::CSC);
        auto sort = coo_sort_type(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSC);

        sort_coo_data(NonzeroSort::CSR, A);
        EXPECT_EQ(A.sort, NonzeroSort::CSR);
        sort = coo_sort_type(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSR);

        sort_coo_data(NonzeroSort::CSC, A);
        EXPECT_EQ(A.sort, NonzeroSort::CSC);
        sort = coo_sort_type(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSC);
        return;
    }

};

TEST_F(TestCOO, to_from_dense) {
    test_to_from_dense(3);
    test_to_from_dense(4);
    test_to_from_dense(10);
}

TEST_F(TestCOO, sort_order) {
    test_sort_order(3);
    test_sort_order(7);
    test_sort_order(10);
}


class Test_SkOp_to_COO : public ::testing::Test
{
    protected:
        std::vector<uint32_t> keys{42, 0, 1};
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};     
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    void sparse_skop_to_coo(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index, RandBLAS::MajorAxis ma) {
        RandBLAS::SparseSkOp<T> S(
            {d, m, vec_nnzs[nnz_index], ma}, keys[key_index]
        );
        auto A = RandBLAS::sparse::coo_view_of_skop(S);

        EXPECT_EQ(S.dist.n_rows, A.n_rows);
        EXPECT_EQ(S.dist.n_cols, A.n_cols);
        EXPECT_EQ(RandBLAS::sparse::nnz(S), A.nnz);

        std::vector<T> S_dense(d * m);
        sparseskop_to_dense(S, S_dense.data(), Layout::ColMajor);
        std::vector<T> A_dense(d * m);
        coo_to_dense(A, Layout::ColMajor, A_dense.data());
    
        test::comparison::buffs_approx_equal(S_dense.data(), A_dense.data(), d * m,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    } 
};

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_1) {
    sparse_skop_to_coo(7, 20, 0, 0, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 1, 0, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 2, 0, RandBLAS::MajorAxis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_2) {
    sparse_skop_to_coo(7, 20, 0, 1, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 1, 1, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 2, 1, RandBLAS::MajorAxis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_3) {
    sparse_skop_to_coo(7, 20, 0, 2, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 1, 2, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 2, 2, RandBLAS::MajorAxis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_7) {
    sparse_skop_to_coo(7, 20, 0, 3, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 1, 3, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(7, 20, 2, 3, RandBLAS::MajorAxis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_15by7) {
    sparse_skop_to_coo(15, 7, 0, 0, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(15, 7, 1, 0, RandBLAS::MajorAxis::Short);

    sparse_skop_to_coo(15, 7, 0, 1, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(15, 7, 1, 1, RandBLAS::MajorAxis::Short);

    sparse_skop_to_coo(15, 7, 0, 2, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(15, 7, 1, 2, RandBLAS::MajorAxis::Short);

    sparse_skop_to_coo(15, 7, 0, 3, RandBLAS::MajorAxis::Short);
    sparse_skop_to_coo(15, 7, 1, 3, RandBLAS::MajorAxis::Short);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_1) {
    sparse_skop_to_coo(7, 20, 0, 0, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 1, 0, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 2, 0, RandBLAS::MajorAxis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_2) {
    sparse_skop_to_coo(7, 20, 0, 1, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 1, 1, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 2, 1, RandBLAS::MajorAxis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_3) {
    sparse_skop_to_coo(7, 20, 0, 2, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 1, 2, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 2, 2, RandBLAS::MajorAxis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_7) {
    sparse_skop_to_coo(7, 20, 0, 3, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 1, 3, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(7, 20, 2, 3, RandBLAS::MajorAxis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_15by7) {
    sparse_skop_to_coo(15, 7, 0, 0, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(15, 7, 1, 0, RandBLAS::MajorAxis::Long);

    sparse_skop_to_coo(15, 7, 0, 1, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(15, 7, 1, 1, RandBLAS::MajorAxis::Long);

    sparse_skop_to_coo(15, 7, 0, 2, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(15, 7, 1, 2, RandBLAS::MajorAxis::Long);

    sparse_skop_to_coo(15, 7, 0, 3, RandBLAS::MajorAxis::Long);
    sparse_skop_to_coo(15, 7, 1, 3, RandBLAS::MajorAxis::Long);
}


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



template <typename T>
class TestRightMultiply_COO : public TestRightMultiply_Sparse<COOMatrix<T>> {
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

class TestRightMultiply_COO_double : public TestRightMultiply_COO<double> {};

class TestRightMultiply_COO_single : public TestRightMultiply_COO<float> {};

////////////////////////////////////////////////////////////////////////
//
//
//      Right-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_COO_double, wide_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_COO_double, wide_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}


TEST_F(TestRightMultiply_COO_double, tall_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_COO_double, tall_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_COO_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_COO_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_COO_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

TEST_F(TestRightMultiply_COO_double, nontrivial_scales_rowmajor2) {
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

TEST_F(TestRightMultiply_COO_double, transpose_self_double_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_COO_double, transpose_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::RowMajor, 0.01);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.10);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_COO_single, transpose_self_single) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//      Submatrices of self (sparse operator)
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_COO_double, submatrix_self_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_COO_double, submatrix_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::RowMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_COO_single, submatrix_self_single) {
    for (uint32_t key : {0}) {
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 0.1);
        submatrix_self(key, 3, 10, 8, 12, 2, 1, Layout::ColMajor, 1.0);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in right-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_COO_double, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_COO_double, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_COO_single, submatrix_other_single) {
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


TEST_F(TestRightMultiply_COO_double, trans_other_times_sparse_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_COO_double, trans_other_times_sparse_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

