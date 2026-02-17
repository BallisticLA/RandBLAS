
#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/sparse_data/trsm_dispatch.hh>
#include <RandBLAS/sparse_data/csr_trsm_impl.hh>
#include <RandBLAS/sparse_data/csc_trsm_impl.hh>
using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using RandBLAS::sparse_data::COOMatrix;
using blas::Layout;
using blas::Uplo;
using blas::Op;
using blas::Diag;
#include "../comparison.hh"
#include "../test_datastructures/test_spmats/common.hh"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <cassert>
#include <gtest/gtest.h>

using std::vector;

class TestSptrsm : public ::testing::Test
{
    protected:
    template<typename T>
    static COOMatrix<T> make_test_matrix(int64_t n, T nonzero_prob, bool upper, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        COOMatrix<T> A(n, n);
        std::vector<T> actual(n * n);
        RandBLAS::RNGState s(key);
        test::test_datastructures::test_spmats::iid_sparsify_random_dense<T>(n, n, Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        RandBLAS::sparse_data::coo::dense_to_coo<T>(Layout::ColMajor, actual.data(), 0.0, A);

        COOMatrix<T> A_triangular(n, n);
        test::test_datastructures::test_spmats::trianglize_coo<T>(A, upper, A_triangular);
        return A_triangular;
    }
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template<typename T>
    static void test_csr_solve(int64_t n, T p, bool upper, int64_t incx, uint32_t key) {
        auto A_coo = make_test_matrix(n, p, upper, key);
        CSRMatrix<T> A(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A);
        std::vector<T> rhs(incx * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i=0; i < incx * n; i++) {
            rhs_ptr[i] = i;
        }
        if (upper) {
            RandBLAS::sparse_data::csr::upper_trsv(true, A.vals, A.rowptr, A.colidxs, n, rhs_ptr, incx);
        } else {
            RandBLAS::sparse_data::csr::lower_trsv(true, A.vals, A.rowptr, A.colidxs, n, rhs_ptr, incx);
        }

        std::vector<T> reference(n);
        T* ref_ptr = reference.data();
        RandBLAS::spmm(Layout::RowMajor, Op::NoTrans, Op::NoTrans, n, 1, n, 1.0, A, rhs_ptr, incx, 0.0, ref_ptr, 1);
        
        for (int64_t i = 0; i < n; ++i) {
            randblas_require(std::abs(ref_ptr[i] - incx * i) <= RandBLAS::sqrt_epsilon<T>());
        }
        return;
    }

    template<typename T>
    static void test_csc_solve(int64_t n, T p, bool upper, int64_t incx, uint32_t key) {
        auto A_coo = make_test_matrix(n, p, upper, key);
        CSCMatrix<T> A(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A);
        std::vector<T> rhs(incx * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i=0; i < incx* n; i++) {
            rhs_ptr[i] = i;
        }
        if (upper) {
            RandBLAS::sparse_data::csc::upper_trsv(true, A.vals, A.rowidxs, A.colptr, n, rhs_ptr, incx);
        } else {
            RandBLAS::sparse_data::csc::lower_trsv(true, A.vals, A.rowidxs, A.colptr, n, rhs_ptr, incx);
        }

        std::vector<T> reference(n);
        T* ref_ptr =reference.data();
        RandBLAS::spmm(Layout::RowMajor, Op::NoTrans, Op::NoTrans, n, 1, n, 1.0, A, rhs_ptr,incx, 0.0, ref_ptr, 1);
        
        for (int64_t i = 0; i < n; ++i) {
            randblas_require(std::abs(ref_ptr[i] - incx * i) <= RandBLAS::sqrt_epsilon<T>());
        }
        return;
    }

    template<typename T>
    static void test_csc_solve_matrix(Layout layout, int64_t n, T p, Op op,  Uplo uplo, int64_t k, uint32_t key) {
        auto A_coo = make_test_matrix(n, p, uplo == Uplo::Upper, key);
        CSCMatrix<T> A(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A);
        std::vector<T> rhs(k * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i = 0; i < k * n; i++) {
            rhs_ptr[i] = i;
        }
        std::vector<T> rhs_copy(rhs);
        int64_t ldb = (layout == Layout::RowMajor) ? k : n;
        RandBLAS::sparse_data::trsm(layout, op, (T)1.0, A, uplo, Diag::NonUnit, k, rhs_ptr, ldb);

        std::vector<T> reference(k * n);
        T* ref_ptr = reference.data();
        RandBLAS::spmm(layout, op, Op::NoTrans, n, k, n, 1.0, A, rhs_ptr, ldb, 0.0, ref_ptr, ldb);
        
        T atol = RandBLAS::sqrt_epsilon<T>();
        T rtol = atol;
        test::comparison::buffs_approx_equal(k*n, rhs_copy.data(), 1, ref_ptr, 1, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol);
        return;
    }

    template<typename T>
    static void test_csr_solve_matrix(Layout layout, int64_t n, T p, Op op, Uplo uplo, int64_t k, uint32_t key) {
        auto A_coo = make_test_matrix(n, p, uplo == Uplo::Upper, key);
        CSRMatrix<T> A(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A);
        std::vector<T> rhs(k * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i= 0; i < k * n; i++) {
            rhs_ptr[i] = i;
        }
        std::vector<T> rhs_copy(rhs);
        int64_t ldb = (layout == Layout::RowMajor) ? k : n;
        RandBLAS::sparse_data::trsm(layout, op, (T)1.0, A, uplo, Diag::NonUnit, k, rhs_ptr, ldb);

        std::vector<T> reference(k * n);
        T* ref_ptr = reference.data();
        RandBLAS::spmm(layout, op, Op::NoTrans, n, k, n, 1.0, A, rhs_ptr, ldb, 0.0, ref_ptr, ldb);
        
        // for (int64_t i = 0; i < k * n; ++i) {
	    //     std::cout << ref_ptr[i] << "\t" << i << std::endl;
        //     randblas_require(std::abs(ref_ptr[i] - i) <= RandBLAS::sqrt_epsilon<T>());
        // }
        T atol = RandBLAS::sqrt_epsilon<T>();
        T rtol = atol;
        test::comparison::buffs_approx_equal(k*n, rhs_copy.data(), 1, ref_ptr, 1, __PRETTY_FUNCTION__, __FILE__, __LINE__, atol, rtol);
        return;
    }


};


// MARK: TRSV

TEST_F(TestSptrsm, upper_csr_solve) {
    test_csr_solve(1, 1.0, true, 3, 0x364A);
    test_csr_solve(2, 0.5, true, 3, 0x3643);
    test_csr_solve(5, 0.9999, true, 3, 0x219B);
      
    test_csr_solve(1, 1.0, true, 1, 0x364A);
    test_csr_solve(2, 0.5, true, 1, 0x3643);
    test_csr_solve(5, 0.9999, true, 1, 0x219B);
}

TEST_F(TestSptrsm, lower_csr_solve) {
    test_csr_solve(1, 1.0, false, 3, 0x364A);
    test_csr_solve(2, 0.5, false, 3, 0x3643);
    test_csr_solve(5, 0.9999, false, 3, 0x219B);
      
    test_csr_solve(1, 1.0, false, 1, 0x364A);
    test_csr_solve(2, 0.5, false, 1, 0x3643);
    test_csr_solve(5, 0.9999, false, 1, 0x219B);
}

TEST_F(TestSptrsm, upper_csc_solve) {
    test_csc_solve(1, 1.0, true, 3, 0x364A);
    test_csc_solve(2, 0.5, true, 3, 0x3643);
    test_csc_solve(5, 0.9999, true, 3, 0x219B);
    
    test_csc_solve(1, 1.0, true, 1, 0x364A);
    test_csc_solve(2, 0.5, true, 1, 0x3643);
    test_csc_solve(5, 0.9999, true, 1, 0x219B);
}

TEST_F(TestSptrsm, lower_csc_solve) {
    test_csc_solve(1, 1.0, false, 3, 0x364A);
    test_csc_solve(2, 0.5, false, 3, 0x3643);
    test_csc_solve(5, 0.9999, false, 3, 0x219B);

    test_csc_solve(1, 1.0, false, 1, 0x364A);
    test_csc_solve(2, 0.5, false, 1, 0x3643);
    test_csc_solve(5, 0.9999, false, 1, 0x219B);
}


// MARK: TRSM, row major

TEST_F(TestSptrsm, lower_csc_solve_matrix_rowmajor) {
    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 1, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 1, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 1, 0x219B);

    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 3, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 3, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csc_solve_matrix_rowmajor) {
    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 1, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 1, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 1, 0x219B);

    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 3, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 3, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csr_solve_matrix_rowmajor) {
    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 1, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 1, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 1, 0x219B);

    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 3, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 3, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csr_solve_matrix_rowmajor) {
    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 1, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 1, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 1, 0x219B);

    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 3, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 3, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csc_trans_solve_matrix_rowmajor) {
    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 1, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 1, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 1, 0x219B);

    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 3, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 3, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csc_trans_solve_matrix_rowmajor) {
    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 1, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 1, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 1, 0x219B);

    test_csc_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 3, 0x364A);
    test_csc_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 3, 0x3643);
    test_csc_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csr_trans_solve_matrix_rowmajor) {
    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 1, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 1, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 1, 0x219B);

    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 3, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 3, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csr_trans_solve_matrix_rowmajor) {
    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 1, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 1, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 1, 0x219B);

    test_csr_solve_matrix(Layout::RowMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 3, 0x364A);
    test_csr_solve_matrix(Layout::RowMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 3, 0x3643);
    test_csr_solve_matrix(Layout::RowMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 3, 0x219B);
}

// MARK: TRSM, column major


TEST_F(TestSptrsm, lower_csc_solve_matrix_comajor) {
    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 1, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 1, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 1, 0x219B);

    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 3, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 3, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csc_solve_matrix_comajor) {
    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 1, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 1, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 1, 0x219B);

    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 3, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 3, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csr_solve_matrix_comajor) {
    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 1, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 1, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 1, 0x219B);

    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Lower, 3, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Lower, 3, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csr_solve_matrix_comajor) {
    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 1, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 1, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 1, 0x219B);

    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::NoTrans, Uplo::Upper, 3, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::NoTrans, Uplo::Upper, 3, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::NoTrans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csc_trans_solve_matrix_comajor) {
    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 1, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 1, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 1, 0x219B);

    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 3, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 3, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csc_trans_solve_matrix_comajor) {
    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 1, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 1, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 1, 0x219B);

    test_csc_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 3, 0x364A);
    test_csc_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 3, 0x3643);
    test_csc_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 3, 0x219B);
}

TEST_F(TestSptrsm, lower_csr_trans_solve_matrix_comajor) {
    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 1, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 1, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 1, 0x219B);

    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Lower, 3, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Lower, 3, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Lower, 3, 0x219B);
}

TEST_F(TestSptrsm, upper_csr_trans_solve_matrix_comajor) {
    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 1, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 1, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 1, 0x219B);

    test_csr_solve_matrix(Layout::ColMajor, 1, 1.0,    Op::Trans, Uplo::Upper, 3, 0x364A);
    test_csr_solve_matrix(Layout::ColMajor, 2, 0.5,    Op::Trans, Uplo::Upper, 3, 0x3643);
    test_csr_solve_matrix(Layout::ColMajor, 5, 0.9999, Op::Trans, Uplo::Upper, 3, 0x219B);
}

// MARK: validation errors

TEST_F(TestSptrsm, ill_formed_csr_unsorted_indices) {
    // Build a 3x3 lower triangular CSR matrix with unsorted column indices in row 1.
    //   Row 0: (0,0)=1.0
    //   Row 1: (1,1)=1.0, (1,0)=2.0   <-- column indices [1, 0] are not sorted
    //   Row 2: (2,0)=3.0, (2,1)=4.0, (2,2)=1.0
    double  vals[]     = {1,  1,  2,  3,  4,  1};
    int64_t colidxs[]  = {0,  1,  0,  0,  1,  2};
    int64_t rowptr[]   = {0,  1,  3,  6};
    CSRMatrix<double, int64_t> A(3, 3, 6, vals, rowptr, colidxs);
    std::vector<double> B(3, 1.0);
    ASSERT_THROW(
        RandBLAS::sparse_data::trsm(Layout::ColMajor, Op::NoTrans, 1.0, A, Uplo::Lower, Diag::NonUnit, 1, B.data(), 3),
        RandBLAS::Error
    );
}

TEST_F(TestSptrsm, ill_formed_csc_unsorted_indices) {
    // Build a 3x3 upper triangular CSC matrix with unsorted row indices in column 2.
    //   Col 0: (0,0)=1.0
    //   Col 1: (0,1)=2.0, (1,1)=1.0
    //   Col 2: (2,2)=1.0, (0,2)=3.0   <-- row indices [2, 0] are not sorted
    double  vals[]     = {1,  2,  1,  1,  3};
    int64_t rowidxs[]  = {0,  0,  1,  2,  0};
    int64_t colptr[]   = {0,  1,  3,  5};
    CSCMatrix<double, int64_t> A(3, 3, 5, vals, rowidxs, colptr);
    std::vector<double> B(3, 1.0);
    ASSERT_THROW(
        RandBLAS::sparse_data::trsm(Layout::ColMajor, Op::NoTrans, 1.0, A, Uplo::Upper, Diag::NonUnit, 1, B.data(), 3),
        RandBLAS::Error
    );
}
