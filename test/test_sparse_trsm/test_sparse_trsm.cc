
#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/sparse_data/csr_trsm_impl.hh>
#include <RandBLAS/sparse_data/csc_trsm_impl.hh>
using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using RandBLAS::sparse_data::COOMatrix;
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
        test::test_datastructures::test_spmats::iid_sparsify_random_dense<T>(n, n, blas::Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        RandBLAS::sparse_data::coo::dense_to_coo<T>(blas::Layout::ColMajor, actual.data(), 0.0, A);

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
            RandBLAS::sparse_data::csr::upper_trsv(A.vals, A.rowptr, A.colidxs, n, rhs_ptr, incx);
        } else {
            RandBLAS::sparse_data::csr::lower_trsv(A.vals, A.rowptr, A.colidxs, n, rhs_ptr, incx);
        }

        std::vector<T> reference(n);
        T* ref_ptr =reference.data();
        RandBLAS::spmm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, 1, n, 1.0, A, rhs_ptr, incx, 0.0, ref_ptr, 1);
        
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
            RandBLAS::sparse_data::csc::upper_trsv(A.vals, A.rowidxs, A.colptr, n, rhs_ptr, incx);
        } else {
            RandBLAS::sparse_data::csc::lower_trsv(A.vals, A.rowidxs, A.colptr, n, rhs_ptr, incx);
        }

        std::vector<T> reference(n);
        T* ref_ptr =reference.data();
        RandBLAS::spmm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, 1, n, 1.0, A, rhs_ptr,incx, 0.0, ref_ptr, 1);
        
        for (int64_t i = 0; i < n; ++i) {
            randblas_require(std::abs(ref_ptr[i] - incx * i) <= RandBLAS::sqrt_epsilon<T>());
        }
        return;
    }

    template<typename T>
    static void test_csc_solve_matrix(int64_t n, T p, bool upper, int64_t k, uint32_t key) {
        auto A_coo = make_test_matrix(n, p, upper, key);
        CSCMatrix<T> A(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csc(A_coo, A);
        std::vector<T> rhs(k * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i=0; i < k * n; i++) {
            rhs_ptr[i] = i;
        }
        if (upper) {
            // RandBLAS::sparse_data::csc::upper_trsm_jki_p11(blas::Layout::RowMajor, n, k, A, rhs_ptr, 1); 
        } else {
            RandBLAS::sparse_data::csc::lower_trsm_jki_p11(blas::Layout::RowMajor, n, k, A, rhs_ptr, k); 
        }
        return;

        std::vector<T> reference(k * n);
        T* ref_ptr =reference.data();
        RandBLAS::spmm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, k, n, 1.0, A, rhs_ptr, k, 0.0, ref_ptr, k);
        
        for (int64_t i = 0; i < k * n; ++i) {
	    std::cout << ref_ptr[i] << "\t" << i << std::endl;
            randblas_require(std::abs(ref_ptr[i] - i) <= RandBLAS::sqrt_epsilon<T>());
        }
        return;
    }


};

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

TEST_F(TestSptrsm, lower_csc_solve_matrix) {
    test_csc_solve_matrix(1, 1.0, false, 1, 0x364A);
    test_csc_solve_matrix(2, 0.5, false, 1, 0x3643);
    test_csc_solve_matrix(5, 0.9999, false, 1, 0x219B);

    test_csc_solve_matrix(1, 1.0, false, 3, 0x364A);
    test_csc_solve_matrix(2, 0.5, false, 3, 0x3643);
    test_csc_solve_matrix(5, 0.9999, false, 3, 0x219B);
}
