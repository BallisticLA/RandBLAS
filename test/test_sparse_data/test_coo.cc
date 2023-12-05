#include "test/test_sparse_data/common.hh"
#include "RandBLAS/sparse_data/coo_multiply.hh"
#include "RandBLAS/test_util.hh"
#include <gtest/gtest.h>

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

        RandBLAS_Testing::Util::buffs_approx_equal(actual.data(), expect.data(), n * n,
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
        auto A = coo_view_of_skop(S);

        EXPECT_EQ(S.dist.n_rows, A.n_rows);
        EXPECT_EQ(S.dist.n_cols, A.n_cols);
        EXPECT_EQ(RandBLAS::sparse::nnz(S), A.nnz);

        std::vector<T> S_dense(d * m);
        RandBLAS_Testing::Util::sparseskop_to_dense(S, S_dense.data(), Layout::ColMajor);
        std::vector<T> A_dense(d * m);
        coo_to_dense(A, Layout::ColMajor, A_dense.data());
    
        RandBLAS_Testing::Util::buffs_approx_equal(S_dense.data(), A_dense.data(), d * m,
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



// class Test_Filter_COO : public ::testing::Test
// {
//     protected:
    
//     virtual void SetUp(){};

//     virtual void TearDown(){};

//     template <typename T = double>
//     void run() {
//         return;
//     }

// }