#include "test/test_sparse_data/common.hh"
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

