#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/test_util.hh"
#include <RandBLAS/sparse_data/csr.hh>

#include <gtest/gtest.h>

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;

// TODO: move me into a testing utility file.
template <typename T, typename RNG = r123::Philox4x32>
void iid_sparsify_random_dense(
    int64_t n_rows,
    int64_t n_cols,
    int64_t stride_row,
    int64_t stride_col,
    T* mat,
    T prob_of_zero,
    RandBLAS::RNGState<RNG> state
) { 
    auto spar = new T[n_rows * n_cols];
    auto dist = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
    auto [unused, next_state] = RandBLAS::fill_dense(dist, spar, state);

    auto temp = new T[n_rows * n_cols];
    auto D_mat = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
    RandBLAS::fill_dense(D_mat, temp, next_state);

    // We'll pretend both of those matrices are column-major, regardless of the layout
    // value returned by fill_dense in each case.
    #define SPAR(_i, _j) spar[(_i) + (_j) * n_rows]
    #define TEMP(_i, _j) temp[(_i) + (_j) * n_rows]
    #define MAT(_i, _j)  mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T v = (SPAR(i, j) + 1.0) / 2.0;
            if (v < prob_of_zero) {
                MAT(i, j) = 0.0;
            } else {
                MAT(i, j) = TEMP(i, j);
            }
        }
    }

    delete [] spar;
    delete [] temp;
}


class TestCSR_Conversions : public ::testing::Test
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

    template <typename T>
    static void test_random_sparsified(blas::Layout layout, int64_t m, int64_t n, T p) {
        RandBLAS::RNGState s(0);
        auto dn_mat = new T[m * n];
        int64_t stride_row, stride_col;
        // TODO: figure out what these parameters
        // should actually be to test something meaningful and
        // pass the address sanitizer ...
        if (layout == blas::Layout::ColMajor) {
            stride_row = m;
            stride_col = 1;
        } else {
            stride_row = 1;
            stride_col = n;
        }
        // get dense representation of random sparse matrix
        iid_sparsify_random_dense(m, n, stride_row, stride_col, dn_mat, p, s);
        // convert the dense representation into a CSR matrix
        CSRMatrix<T> spmat(m, n, IndexBase::Zero);
        dense_to_csr(stride_row, stride_col, dn_mat, 0.0, spmat);
        // reconstruct the dense representation of dn_mat from the CSR matrix.
        auto dn_mat_recon = new T[m * n];
        csr_to_dense(spmat, stride_row, stride_col, dn_mat_recon);

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
