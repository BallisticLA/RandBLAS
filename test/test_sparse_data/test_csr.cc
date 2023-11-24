#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include <RandBLAS/sparse_data/csr.hh>

#include <gtest/gtest.h>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;



class TestCSR_Basics : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_csr_to_dense_identity(int64_t n) {
        CSRMatrix<T> A(n, n, IndexBase::Zero);
        A.reserve_nnz(n);
        for (int i = 0; i < n; ++i) {
            A.vals[i] = 1.0;
            A.rowptr[i] = i;
            A.colidxs[i] = i;
        }
        A.rowptr[n] = n;
        T *mat = new T[n*n];
        csr_to_dense(A, 1, n, mat);
        T *eye = new T[n*n]{0.0};
        for (int i = 0; i < n; ++i)
            eye[i + n*i] = 1.0;
        RandBLAS_Testing::Util::buffs_approx_equal(mat, eye, n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    }
};

TEST_F(TestCSR_Basics, trivial) {
    test_csr_to_dense_identity<double>(3);
}