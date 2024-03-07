#include "test/test_sparse_data/common.hh"
#include "../comparison.hh"
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace test::sparse_data::common;
using namespace RandBLAS::sparse_data::csc;
using blas::Layout;

class TestSparseTranspose : public ::testing::Test 
{
    protected:
    virtual void SetUp(){};
    virtual void TearDown(){};

    template <typename T = double>
    void test_transposed_csr_as_csc(int64_t n, int64_t m, T p) {
        Layout layout = Layout::ColMajor;
        std::vector<T> A_dense(m * n);
        std::vector<T> At_dense(n * m);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(m, n, layout, A_dense.data(), p, s);
        CSRMatrix<T> A_csr(m, n);
        dense_to_csr(Layout::ColMajor, A_dense.data(), 0.0, A_csr);
        
        CSCMatrix<T> At_csc_view = transpose_as_csc(A_csr, true);
        csc_to_dense(At_csc_view, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        CSCMatrix<T> At_csc_copy = transpose_as_csc(A_csr, false);
        blas::scal(m * n, 0.0, At_dense.data(), 1);
        blas::scal(A_csr.nnz, 0.0, A_csr.vals, 1);
        csc_to_dense(At_csc_copy, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T = double>
    void test_transposed_csc_as_csr(int64_t n, int64_t m, T p) {
        Layout layout = Layout::ColMajor;
        std::vector<T> A_dense(m * n);
        std::vector<T> At_dense(n * m);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(m, n, layout, A_dense.data(), p, s);
        CSCMatrix<T> A_csc(m, n);
        dense_to_csc(Layout::ColMajor, A_dense.data(), 0.0, A_csc);
        
        CSRMatrix<T> At_csr_view = transpose_as_csr(A_csc, true);
        csr_to_dense(At_csr_view, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        CSRMatrix<T> At_csr_copy = transpose_as_csr(A_csc, false);
        blas::scal(m * n, 0.0, At_dense.data(), 1);
        blas::scal(A_csc.nnz, 0.0, A_csc.vals, 1);
        csr_to_dense(At_csr_copy, layout, At_dense.data());
        test::comparison::matrices_approx_equal(
            layout, blas::Op::Trans, m, n, A_dense.data(), m, At_dense.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }
};

TEST_F(TestSparseTranspose, CSR_TO_CSC_7x20) {
    test_transposed_csr_as_csc(7, 20, 0.05);
    test_transposed_csr_as_csc(7, 20, 0.90);
}

TEST_F(TestSparseTranspose, CSR_TO_CSC_13x5) {
    test_transposed_csr_as_csc(13, 5, 0.05);
    test_transposed_csr_as_csc(13, 5, 0.90);
}

TEST_F(TestSparseTranspose, CSC_TO_CSR_7x20) {
    test_transposed_csc_as_csr(7, 20, 0.05);
    test_transposed_csc_as_csr(7, 20, 0.90);
}

TEST_F(TestSparseTranspose, CSC_TO_CSR_13x5) {
    test_transposed_csc_as_csr(13, 5, 0.05);
    test_transposed_csc_as_csr(13, 5, 0.90);
}