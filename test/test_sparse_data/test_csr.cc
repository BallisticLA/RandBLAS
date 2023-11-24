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
        T *vals = new T[n];
        int64_t *rowptr = new int64_t[n+1];
        int64_t *colidxs = new int64_t[n];
        for (int i = 0; i < n; ++i) {
            vals[i] = 1.0;
            rowptr[i] = i;
            colidxs[i] = i;
        }
        rowptr[n] = n;
        RandBLAS::sparse_data::CSRMatrix<T> A{
            n, n, n, RandBLAS::sparse_data::IndexBase::Zero, vals, rowptr, colidxs
        };
        T *mat = new T[n*n];
        RandBLAS::sparse_data::csr::csr_to_dense(A, 1, n, mat);
        T *eye = new T[n*n]{0.0};
        for (int i = 0; i < n; ++i)
            eye[i + n*i] = 1.0;
        RandBLAS_Testing::Util::buffs_approx_equal(mat, eye, n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }
};

TEST_F(TestCSR_Basics, trivial) {
    test_csr_to_dense_identity<double>(3);
}