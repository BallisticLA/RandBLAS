// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include "../../comparison.hh"
#include "common.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::coo;
using namespace RandBLAS::sparse_data::csr;
using namespace test::test_datastructures::test_spmats;
using namespace RandBLAS::sparse_data::conversions;
using blas::Layout;


class TestCSR_Conversions : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    static void test_csr_to_dense_diagonal(int64_t n) {
        CSRMatrix<T> A(n, n);
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
        CSRMatrix<T> spmat(m, n);
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

    template <typename T = double>
    static void test_csr_to_coo_band_diagonal() {
        int64_t n = 8;
        int64_t nnz = 32;
        std::vector<T> vals{6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, 6, -1, -100, 99, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6};
        std::vector<int64_t> rowptr{0, 4, 8, 12, 16, 20, 24, 28, 32};
        std::vector<int64_t> colidxs{0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7, 3, 5, 6, 7};
        CSRMatrix<T> A_csr(n,n,nnz,vals.data(),rowptr.data(),colidxs.data());
        COOMatrix<T> A_coo(n,n);
        csr_to_coo(A_csr, A_coo);
        std::vector<T> A_dense_coo(n*n);
        std::vector<T> A_dense_csr(n*n);
        coo_to_dense(A_coo, Layout::ColMajor, A_dense_coo.data());
        csr_to_dense(A_csr, Layout::ColMajor, A_dense_csr.data());
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Layout::ColMajor, blas::Op::NoTrans,
            n, n, A_dense_csr.data(), n, A_dense_coo.data(), n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T = double>
    static void test_deepcopy() {
        // Use essentially the same test data as test_csr_to_coo_band_diag (just truncate one row)
        int64_t n_rows = 7;
        int64_t n_cols = 8;
        int64_t nnz = 28;
        std::vector<T> vals{6, -1, -2, -3, -1, 6, -1, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, 6, -1, -1, -1, -1, 6, -1, -1, -1, 6, -1};
        std::vector<int64_t> rowptr{0, 4, 8, 12, 16, 20, 24, 28};
        std::vector<int64_t> colidxs{0, 1, 2, 4, 0, 1, 3, 5, 0, 2, 3, 6, 1, 2, 3, 7, 0, 4, 5, 6, 1, 4, 5, 7, 2, 4, 6, 7};
        CSRMatrix<T> A(n_cols, n_rows, nnz, vals.data(), rowptr.data(), colidxs.data());
        auto A_copy = A.deepcopy();
        for (int64_t j = 0; j < n_rows; j++) {
            for (int64_t p = A.rowptr[j]; p < A.rowptr[j+1]; ++p) {
                EXPECT_EQ( A.vals[p],    A_copy.vals[p]    );
                EXPECT_EQ( A.colidxs[p], A_copy.colidxs[p] );
            }
        }
        auto vals_copy    = vals;
        auto rowptr_copy  = rowptr;
        auto colidxs_copy = colidxs;
        std::fill(A_copy.rowptr, A_copy.rowptr + n_rows + 1, 0);
        std::fill(A_copy.colidxs, A_copy.colidxs + nnz, 0);
        std::fill(A_copy.vals, A_copy.vals + nnz, 0);

        for (int64_t j = 0; j < n_rows; j++) {
            for (int64_t p = rowptr_copy[j]; p < rowptr_copy[j+1]; ++p) {
                EXPECT_EQ( vals_copy[p]   , A.vals[p]    );
                EXPECT_EQ( colidxs_copy[p], A.colidxs[p] );
            }
        }
    }
    
};

TEST_F(TestCSR_Conversions, deepcopy) {
    test_deepcopy();
}

TEST_F(TestCSR_Conversions, band) {
    test_csr_to_coo_band_diagonal();
}

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

