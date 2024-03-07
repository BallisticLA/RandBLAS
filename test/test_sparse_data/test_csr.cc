#include "test/test_sparse_data/test_left_multiply.hh"
#include "test/test_sparse_data/test_right_multiply.hh"
#include <algorithm>
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;
using namespace test::sparse_data::common;
using blas::Layout;


class TestCSR_Conversions : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    static void test_csr_to_dense_diagonal(int64_t n) {
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
        test::comparison::buffs_approx_equal(mat, eye, n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        
        delete [] eye;
        delete [] mat;
        return;
    }

    template <typename T = double>
    static void test_csr_from_random_sparsified(Layout layout, int64_t m, int64_t n, T p) {
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
        test::comparison::buffs_approx_equal(dn_mat, dn_mat_recon, m * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        delete [] dn_mat;
        delete [] dn_mat_recon;
    }

    template <typename T = double>
    static void test_csr_from_diag_coo(int64_t m, int64_t n, int64_t offset) {
        int64_t len = (offset >= 0) ? std::min(m, n - offset) : std::min(m + offset, n);
        randblas_require(len > 0);
        T *diag = new T[len]{0.0};
        for (int i = 1; i <= len; ++i)
            diag[i-1] = (T) i * 0.5;
        T *mat_expect = new T[m * n]{0.0};
        #define MAT_EXPECT(_i, _j) mat_expect[(_i) + m*(_j)]
        if (offset >= 0) {
            for (int64_t ell = 0; ell < len; ++ell)
                MAT_EXPECT(ell, ell + offset) = diag[ell];
        } else {
            for (int64_t ell = 0; ell < len; ++ell)
                MAT_EXPECT(ell - offset, ell) = diag[ell];
        }

        CSRMatrix<T> csr(m, n);
        COOMatrix<T> coo(m, n);
        coo_from_diag(diag, len, offset, coo);
        coo_to_csr(coo, csr);
        T *mat_actual = new T[m * n]{0.0};
        csr_to_dense(csr, Layout::ColMajor, mat_actual);

        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Layout::ColMajor, blas::Op::NoTrans,
            m, n, mat_expect, m, mat_actual, m,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        delete [] mat_expect;
        delete [] diag;
        delete [] mat_actual;
        return;
    }
};

TEST_F(TestCSR_Conversions, dense_square_diagonal) {
    test_csr_to_dense_diagonal(3);
 }
 
TEST_F(TestCSR_Conversions, dense_random_rowmajor) {
    test_csr_from_random_sparsified(Layout::RowMajor, 10, 5, 0.7);
}

TEST_F(TestCSR_Conversions, dense_random_colmajor) {
    test_csr_from_random_sparsified(Layout::ColMajor, 10, 5, 0.7);
}

TEST_F(TestCSR_Conversions, coo_diagonal_square_zero_offset) {
    test_csr_from_diag_coo(5, 5, 0);
}

TEST_F(TestCSR_Conversions, coo_diagonal_square_pos_offset) {
    test_csr_from_diag_coo(5, 5, 1);
    test_csr_from_diag_coo(5, 5, 2);
    test_csr_from_diag_coo(5, 5, 3);
    test_csr_from_diag_coo(5, 5, 4);
}

TEST_F(TestCSR_Conversions, coo_diagonal_square_neg_offset) {
    test_csr_from_diag_coo(5, 5, -1);
    test_csr_from_diag_coo(5, 5, -2);
    test_csr_from_diag_coo(5, 5, -3);
    test_csr_from_diag_coo(5, 5, -4);
}

TEST_F(TestCSR_Conversions, coo_diagonal_rectangular_zero_offset) {
    test_csr_from_diag_coo(5, 10, 0);
    test_csr_from_diag_coo(10, 5, 0);
}

TEST_F(TestCSR_Conversions, coo_diagonal_rectangular_pos_offset) {
    test_csr_from_diag_coo(10, 5, 1);
    test_csr_from_diag_coo(10, 5, 2);
    test_csr_from_diag_coo(10, 5, 3);
    test_csr_from_diag_coo(10, 5, 4);
    test_csr_from_diag_coo(5, 10, 1);
    test_csr_from_diag_coo(5, 10, 2);
    test_csr_from_diag_coo(5, 10, 3);
    test_csr_from_diag_coo(5, 10, 4);
}

TEST_F(TestCSR_Conversions, coo_diagonal_rectangular_neg_offset) {
    test_csr_from_diag_coo(10, 5, -1);
    test_csr_from_diag_coo(10, 5, -2);
    test_csr_from_diag_coo(10, 5, -3);
    test_csr_from_diag_coo(10, 5, -4);
    test_csr_from_diag_coo(5, 10, -1);
    test_csr_from_diag_coo(5, 10, -2);
    test_csr_from_diag_coo(5, 10, -3);
    test_csr_from_diag_coo(5, 10, -4);
 }


// template <typename T>
// CSRMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
//     randblas_require(nonzero_prob >= 0);
//     randblas_require(nonzero_prob <= 1);
//     CSRMatrix<T> A(m, n);
//     std::vector<T> actual(m * n);
//     RandBLAS::RNGState s(key);
//     iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
//     dense_to_csr<T>(Layout::ColMajor, actual.data(), 0.0, A);
//     return A;
// }



template <typename T>
class TestLeftMultiply_CSR : public TestLeftMultiply_Sparse<CSRMatrix<T>> {
    CSRMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        CSRMatrix<T> A(m, n);
        std::vector<T> actual(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        dense_to_csr<T>(Layout::ColMajor, actual.data(), 0.0, A);
        return A;
    }
};

class TestLeftMultiply_CSR_double : public TestLeftMultiply_CSR<double> {};

class TestLeftMultiply_CSR_single : public TestLeftMultiply_CSR<float> {};

////////////////////////////////////////////////////////////////////////
//
//
//      Left-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_CSR_double, tall_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_double, tall_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_double, wide_multiply_eye_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_double, wide_multiply_eye_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSR_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSR_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 21, 4, Layout::RowMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSR_double, nontrivial_scales_rowmajor2) {
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

TEST_F(TestLeftMultiply_CSR_double, transpose_self_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_double, transpose_self_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::RowMajor, 0.01);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.10);
        transpose_self(key, 200, 30, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestLeftMultiply_CSR_single, transpose_self) {
    for (uint32_t key : {0}) {
        transpose_self(key, 200, 30, Layout::ColMajor, 0.01);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.10);
        transpose_self(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in left-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLeftMultiply_CSR_double, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_CSR_double, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestLeftMultiply_CSR_double, submatrix_other_single) {
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


TEST_F(TestLeftMultiply_CSR_double, sparse_times_trans_other_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestLeftMultiply_CSR_double, sparse_times_trans_other_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}


template <typename T>
class TestRightMultiply_CSR : public TestRightMultiply_Sparse<CSRMatrix<T>> {
    CSRMatrix<T> make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        CSRMatrix<T> A(m, n);
        std::vector<T> actual(m * n);
        RandBLAS::RNGState s(key);
        iid_sparsify_random_dense<T>(m, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        dense_to_csr<T>(Layout::ColMajor, actual.data(), 0.0, A);
        return A;
    }
};

class TestRightMultiply_CSR_double : public TestRightMultiply_CSR<double> {};

class TestRightMultiply_CSR_single : public TestRightMultiply_CSR<float> {};


////////////////////////////////////////////////////////////////////////
//
//
//      Right-muliplication
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_CSR_double, wide_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSR_double, wide_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.01);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.10);
        multiply_eye(key, 200, 30, Layout::RowMajor, 0.80);
    }
}


TEST_F(TestRightMultiply_CSR_double, tall_multiply_eye_double_colmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSR_double, tall_multiply_eye_double_rowmajor) {
    for (uint32_t key : {0}) {
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.01);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.10);
        multiply_eye(key, 51, 101, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSR_double, nontrivial_scales_colmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSR_double, nontrivial_scales_colmajor2) {
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSR_double, nontrivial_scales_rowmajor1) {
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.05);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.10);
    alpha_beta(0, alpha, beta, 4, 21, Layout::RowMajor, 0.80);
}

TEST_F(TestRightMultiply_CSR_double, nontrivial_scales_rowmajor2) {
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

TEST_F(TestRightMultiply_CSR_double, transpose_self_double_colmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSR_double, transpose_self_double_rowmajor) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::RowMajor, 0.01);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.10);
        transpose_self(key, 30, 200, Layout::RowMajor, 0.80);
    }
}

TEST_F(TestRightMultiply_CSR_single, transpose_self_single) {
    for (uint32_t key : {0}) {
        transpose_self(key, 30, 200, Layout::ColMajor, 0.01);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.10);
        transpose_self(key, 30, 200, Layout::ColMajor, 0.80);
    }
}

////////////////////////////////////////////////////////////////////////
//
//     submatrix of other operand in right-multiply
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRightMultiply_CSR_double, submatrix_other_double_colmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::ColMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_CSR_double, submatrix_other_double_rowmajor) {
    for (uint32_t key : {0}) {
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 0.1);
        submatrix_other(key, 3, 10, 5, 12, 8, 2, 1, Layout::RowMajor, 1.0);
    }
}

TEST_F(TestRightMultiply_CSR_single, submatrix_other_single) {
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


TEST_F(TestRightMultiply_CSR_double, trans_other_times_sparse_colmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::ColMajor, 0.80);
}

TEST_F(TestRightMultiply_CSR_double, trans_other_times_sparse_rowmajor) {
    uint32_t key = 0;
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.05);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.10);
    transpose_other(key, 7, 22, 5, Layout::RowMajor, 0.80);
}

