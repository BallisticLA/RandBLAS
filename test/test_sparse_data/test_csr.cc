#include "test/test_sparse_data/common.hh"
#include "RandBLAS/test_util.hh"
#include <gtest/gtest.h>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;
using namespace test::sparse_data::common;
using blas::Layout;



class TestCSR_Conversions : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void test_csr_to_dense_identity(int64_t n) {
        CSRMatrix<T> A(n, n, IndexBase::Zero);
        A.reserve(n);
        for (int i = 0; i < n; ++i) {
            A.vals[i] = 1.0 + (T) i;
            A.rowptr[i] = i;
            A.colidxs[i] = i;
        }
        A.rowptr[n] = n;
        T *mat = new T[n*n];
        csr_to_dense(A, 1, n, mat);
        T *eye = new T[n*n]{0.0};
        for (int i = 0; i < n; ++i)
            eye[i + n*i] = 1.0 + (T) i;
        RandBLAS_Testing::Util::buffs_approx_equal(mat, eye, n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    }

    template <typename T>
    static void test_random_sparsified(blas::Layout layout, int64_t m, int64_t n, T p) {
        // Step 1. get dense representation of random sparse matrix
        RandBLAS::RNGState s(0);
        auto dn_mat = new T[m * n];
        iid_sparsify_random_dense(m, n, layout, dn_mat, p, s);

        // Step 2. convert the dense representation into a CSR matrix
        CSRMatrix<T> spmat(m, n, IndexBase::Zero);
        dense_to_csr(layout, dn_mat, 0.0, spmat);

        // Step 3. reconstruct the dense representation of dn_mat from the CSR matrix.
        auto dn_mat_recon = new T[m * n];
        csr_to_dense(spmat, layout, dn_mat_recon);

        // check equivalence of dn_mat and dn_mat_recon
        RandBLAS_Testing::Util::buffs_approx_equal(dn_mat, dn_mat_recon, m * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        delete [] dn_mat;
        delete [] dn_mat_recon;
    }
};

TEST_F(TestCSR_Conversions, trivial) {
    test_csr_to_dense_identity<double>(3);
}

TEST_F(TestCSR_Conversions, random) {
    test_random_sparsified<double>(blas::Layout::RowMajor, 10, 5, 0.7);
    test_random_sparsified<double>(blas::Layout::ColMajor, 10, 5, 0.7);
}
