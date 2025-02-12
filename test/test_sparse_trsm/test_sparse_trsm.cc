
#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/sparse_data/csr_trsm_impl.hh>
using RandBLAS::sparse_data::CSRMatrix;
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
    static CSRMatrix<T> make_test_matrix(int64_t n, T nonzero_prob, bool upper, uint32_t key = 0) {
        randblas_require(nonzero_prob >= 0);
        randblas_require(nonzero_prob <= 1);
        COOMatrix<T> A(n, n);
        std::vector<T> actual(n * n);
        RandBLAS::RNGState s(key);
        test::test_datastructures::test_spmats::iid_sparsify_random_dense<T>(n, n, blas::Layout::ColMajor, actual.data(), 1 - nonzero_prob, s);
        RandBLAS::sparse_data::coo::dense_to_coo<T>(blas::Layout::ColMajor, actual.data(), 0.0, A);

        COOMatrix<T> A_triangular(n, n);
        test::test_datastructures::test_spmats::trianglize_coo<T>(A, upper, A_triangular);
        CSRMatrix<T> A_tri_csr(n, n);
        RandBLAS::sparse_data::conversions::coo_to_csr(A_triangular, A_tri_csr);
        return A_tri_csr;
    }
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template<typename T>
    static void test_solve(int64_t n, T p, bool upper, uint32_t key) {
        auto A = make_test_matrix(n, p, upper, key);
        std::vector<T> rhs(3 * n);
        T* rhs_ptr = rhs.data();
        for (int64_t i=0; i < 3 * n; i++) {
            rhs_ptr[i] = i;
        }
        if (upper) {
            RandBLAS::sparse_data::csr::upper_trsv(A.vals, A.rowptr, A.colidxs, n, rhs_ptr, 3);
        } else {
            RandBLAS::sparse_data::csr::lower_trsv(A.vals, A.rowptr, A.colidxs, n, rhs_ptr, 3);
        }
        std::vector<T> reference(n);
        T* ref_ptr =reference.data();
        RandBLAS::spmm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, n, 1, n, 1.0, A, rhs_ptr, 3, 0.0, ref_ptr, 1);
        
        for (int64_t i = 0; i < n; ++i) {
            std::cout << ref_ptr[i] << "   " << 3 * i << std::endl;
            randblas_require(std::abs(ref_ptr[i] - 3 * i) <= RandBLAS::sqrt_epsilon<T>());
        }
        return;
    }
};

TEST_F(TestSptrsm, upper_solve) {
    test_solve(10, 0.25, true, 0x364A);
    test_solve(10, 0.05, true, 0x364A);
    test_solve(10, 0.005, true, 0x364A);
    test_solve(10, 0.0005, true, 0x364A);
    test_solve(10, 0.0005, true, 0x364A);
    test_solve(10, 0.0005, true, 0x364A);
    test_solve(10, 0.00000005, true, 0x364A);
    
}

