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

using RandBLAS::Axis;
using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::coo;
using namespace test::test_datastructures::test_spmats;
using namespace RandBLAS::sparse_data::conversions;
using blas::Layout;

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif


template <typename T, typename RNG, SignedInteger sint_t>
void sparseskop_to_dense(
    RandBLAS::SparseSkOp<T, RNG, sint_t> &S0,
    T *mat,
    Layout layout
) {
    RandBLAS::SparseDist D = S0.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;
    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };
    int64_t nnz = S0.nnz;
    for (int64_t i = 0; i < nnz; ++i) {
        sint_t row = S0.rows[i];
        sint_t col = S0.cols[i];
        T val = S0.vals[i];
        mat[idx(row, col)] = val;
    }
}


class TestCOO : public ::testing::Test {
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    void test_to_from_dense(int64_t n) {
        COOMatrix<T> A(n, n);
        std::vector<T> actual(n * n);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(n, n, Layout::ColMajor, actual.data(), 0.8, s);

        dense_to_coo(1, n, actual.data(), 0.0, A);
        std::vector<T> expect(n * n);
        coo_to_dense(A, 1, n, expect.data());

        test::comparison::buffs_approx_equal(actual.data(), expect.data(), n * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        EXPECT_GT(A.nnz, 0);
        EXPECT_LT(A.nnz, n*n);
        return;
    }

    template <typename T = double>
    void test_sort_order(int64_t n) {
        COOMatrix<T> A(n, n);
        std::vector<T> mat(n * n);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(n, n, Layout::ColMajor, mat.data(), 0.25, s);
        dense_to_coo(1, n, mat.data(), 0.0, A);

        A.sort_arrays(NonzeroSort::CSC);
        EXPECT_EQ(A.sort, NonzeroSort::CSC);
        auto sort = coo_arrays_determine_sort(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSC);

        A.sort_arrays(NonzeroSort::CSR);
        EXPECT_EQ(A.sort, NonzeroSort::CSR);
        sort = coo_arrays_determine_sort(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSR);

        A.sort_arrays(NonzeroSort::CSC);
        EXPECT_EQ(A.sort, NonzeroSort::CSC);
        sort = coo_arrays_determine_sort(A.nnz, A.rows, A.cols);
        EXPECT_EQ(sort, NonzeroSort::CSC);
        return;
    }

    template <typename T = double>
    void test_symperm_inplace(int64_t n, T prob_zero) {
        int64_t i = 0;
        
        // arrange
        RandBLAS::DenseDist D(n, n);
        std::vector<T> buff(n*n);
        fill_dense(D, buff.data(), {0});
        iid_sparsify_random_dense(n, n, Layout::ColMajor, buff.data(), prob_zero, {94});
        std::vector<int64_t> perm(n);
        for (i = 0; i < n; ++i) {
            buff[i + i*n] = 2*(i+1); // diagonal is      2,   4, ..., 2*n
            perm[i] = n - (i+1);     // permutation is n-1, n-2, ..., 0.
        }
        COOMatrix<T> A(n, n);
        dense_to_coo(Layout::ColMajor, buff.data(), (T)0.0, A);
        std::vector<T> diagA(n);
        coo_arrays_extract_diagonal(n, n, A.nnz, A.vals, A.rows, A.cols, diagA.data());
        for (i = 0; i < n; ++i) {
            EXPECT_EQ(diagA[i], 2*(i+1)); // < check that we setup the problem correctly.
        }
        A.sort_arrays(NonzeroSort::CSC);
        
        // Test 1 of 2
        auto Ap = A.deepcopy();
        Ap.symperm_inplace(perm.data());
        std::vector<T> diagAp(n);
        coo_arrays_extract_diagonal(n, n, Ap.nnz, Ap.vals, Ap.rows, Ap.cols, diagAp.data());
        for (i = 0; i < n; ++i) {
            EXPECT_EQ( diagAp[i], 2*(n-i) );
        }

        // Test 2 of 2
        Ap.symperm_inplace(perm.data()); // < perm is self-inverse, so applying again should get us A.
        EXPECT_EQ(Ap.sort, A.sort);
        for (i = 0; i < A.nnz; ++i) {
            EXPECT_EQ(Ap.rows[i], A.rows[i]);
            EXPECT_EQ(Ap.cols[i], A.cols[i]);
            EXPECT_EQ(Ap.vals[i], A.vals[i]);
        }
        return;
    }

    template <typename T = double>
    void smoketest_print() {
        int n = 10;
        int m = 11;
        COOMatrix<T> A(m, n);
        std::vector<T> mat(m * n);
        RandBLAS::RNGState s(0);
        iid_sparsify_random_dense(m, n, Layout::ColMajor, mat.data(), 0.25, s);
        dense_to_coo(1, m, mat.data(), 0.0, A);
        print_sparse(A); // no correctness check
    }
};


TEST_F(TestCOO, to_from_dense) {
    test_to_from_dense(3);
    test_to_from_dense(4);
    test_to_from_dense(10);
}

TEST_F(TestCOO, sort_order) {
    test_sort_order(3);
    test_sort_order(7);
    test_sort_order(10);
}

TEST_F(TestCOO, symperm_inplace) {
    test_symperm_inplace( 5,   (float) 0.0  );
    test_symperm_inplace( 20,  (float) 0.8  );
    test_symperm_inplace( 100, (float) 0.99 );
}


class Test_SkOp_to_COO : public ::testing::Test {
    protected:
        std::vector<uint32_t> keys{42, 0, 1};
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};     
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T = double>
    void sparse_skop_to_coo(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index, Axis major_axis) {
        RandBLAS::SparseDist D(d, m, vec_nnzs[nnz_index], major_axis);
        RandBLAS::SparseSkOp<T> S(D, keys[key_index]);
        fill_sparse(S);
        auto A = RandBLAS::sparse::coo_view_of_skop(S);

        EXPECT_EQ(S.dist.n_rows,   A.n_rows);
        EXPECT_EQ(S.dist.n_cols,   A.n_cols);
        if (major_axis == Axis::Short) {
            EXPECT_EQ(S.dist.full_nnz, A.nnz);
        } {
            EXPECT_GE(S.dist.full_nnz, A.nnz);
            EXPECT_EQ(S.nnz, A.nnz);
        }

        std::vector<T> S_dense(d * m);
        sparseskop_to_dense(S, S_dense.data(), Layout::ColMajor);
        std::vector<T> A_dense(d * m);
        coo_to_dense(A, Layout::ColMajor, A_dense.data());
    
        test::comparison::buffs_approx_equal(S_dense.data(), A_dense.data(), d * m,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        return;
    } 
};

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_1) {
    sparse_skop_to_coo(7, 20, 0, 0, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 1, 0, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 2, 0, RandBLAS::Axis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_2) {
    sparse_skop_to_coo(7, 20, 0, 1, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 1, 1, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 2, 1, RandBLAS::Axis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_3) {
    sparse_skop_to_coo(7, 20, 0, 2, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 1, 2, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 2, 2, RandBLAS::Axis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_7by20_nnz_7) {
    sparse_skop_to_coo(7, 20, 0, 3, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 1, 3, RandBLAS::Axis::Short);
    sparse_skop_to_coo(7, 20, 2, 3, RandBLAS::Axis::Short);
}

TEST_F(Test_SkOp_to_COO, SASO_Dim_15by7) {
    sparse_skop_to_coo(15, 7, 0, 0, RandBLAS::Axis::Short);
    sparse_skop_to_coo(15, 7, 1, 0, RandBLAS::Axis::Short);

    sparse_skop_to_coo(15, 7, 0, 1, RandBLAS::Axis::Short);
    sparse_skop_to_coo(15, 7, 1, 1, RandBLAS::Axis::Short);

    sparse_skop_to_coo(15, 7, 0, 2, RandBLAS::Axis::Short);
    sparse_skop_to_coo(15, 7, 1, 2, RandBLAS::Axis::Short);

    sparse_skop_to_coo(15, 7, 0, 3, RandBLAS::Axis::Short);
    sparse_skop_to_coo(15, 7, 1, 3, RandBLAS::Axis::Short);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_1) {
    sparse_skop_to_coo(7, 20, 0, 0, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 1, 0, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 2, 0, RandBLAS::Axis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_2) {
    sparse_skop_to_coo(7, 20, 0, 1, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 1, 1, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 2, 1, RandBLAS::Axis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_3) {
    sparse_skop_to_coo(7, 20, 0, 2, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 1, 2, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 2, 2, RandBLAS::Axis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_7by20_nnz_7) {
    sparse_skop_to_coo(7, 20, 0, 3, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 1, 3, RandBLAS::Axis::Long);
    sparse_skop_to_coo(7, 20, 2, 3, RandBLAS::Axis::Long);
}

TEST_F(Test_SkOp_to_COO, LASO_Dim_15by7) {
    sparse_skop_to_coo(15, 7, 0, 0, RandBLAS::Axis::Long);
    sparse_skop_to_coo(15, 7, 1, 0, RandBLAS::Axis::Long);

    sparse_skop_to_coo(15, 7, 0, 1, RandBLAS::Axis::Long);
    sparse_skop_to_coo(15, 7, 1, 1, RandBLAS::Axis::Long);

    sparse_skop_to_coo(15, 7, 0, 2, RandBLAS::Axis::Long);
    sparse_skop_to_coo(15, 7, 1, 2, RandBLAS::Axis::Long);

    sparse_skop_to_coo(15, 7, 0, 3, RandBLAS::Axis::Long);
    sparse_skop_to_coo(15, 7, 1, 3, RandBLAS::Axis::Long);
}
