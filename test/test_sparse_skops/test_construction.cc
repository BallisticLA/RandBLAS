#include <RandBLAS/dense_skops.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/util.hh>
#include "../comparison.hh"
#include <gtest/gtest.h>
#include <math.h>


class TestSparseSkOpConstruction : public ::testing::Test
{
    protected:
        std::vector<uint32_t> keys{42, 0, 1};
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename SKOP>
    void check_fixed_nnz_per_col(SKOP &S0) {
        using sint_t = typename SKOP::index_t;
        std::set<sint_t> s;
        for (int64_t i = 0; i < S0.dist.n_cols; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                sint_t row = S0.rows[offset + j];
                ASSERT_EQ(s.count(row), 0) << "row index " << row << " was duplicated in column " << i << std::endl;
                s.insert(row);
            }
        }
    }

    template <typename SKOP>
    void check_fixed_nnz_per_row(SKOP &S0) {
        using sint_t = typename SKOP::index_t;
        std::set<sint_t> s;
        for (int64_t i = 0; i < S0.dist.n_rows; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                sint_t col = S0.cols[offset + j];
                ASSERT_EQ(s.count(col), 0)  << "column index " << col << " was duplicated in row " << i << std::endl;
                s.insert(col);
            }
        }
    }

    template <RandBLAS::SignedInteger sint_t>
    void proper_saso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        using RNG = RandBLAS::SparseSkOp<float>::RNG_t;
        RandBLAS::SparseSkOp<float, RNG, sint_t> S0(
            {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Short}, keys[key_index]
        );
        RandBLAS::fill_sparse(S0);
        if (d < m) {
                check_fixed_nnz_per_col(S0);
        } else {
                check_fixed_nnz_per_row(S0);
        }
    }

    template <RandBLAS::SignedInteger sint_t>
    void proper_laso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        using RNG = RandBLAS::SparseSkOp<float>::RNG_t;
        RandBLAS::SparseSkOp<float, RNG, sint_t> S0(
            {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Long}, keys[key_index]
        );
        RandBLAS::fill_sparse(S0);
        if (d < m) {
                check_fixed_nnz_per_row(S0);
        } else {
                check_fixed_nnz_per_col(S0);
        } 
    }
};


////////////////////////////////////////////////////////////////////////
//
//
//     SASOs
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20) {
    // vec_nnz=1
    proper_saso_construction<int64_t>(7, 20, 0, 0);
    proper_saso_construction<int64_t>(7, 20, 1, 0);
    proper_saso_construction<int64_t>(7, 20, 2, 0);
    // vec_nnz=2
    proper_saso_construction<int64_t>(7, 20, 0, 1);
    proper_saso_construction<int64_t>(7, 20, 1, 1);
    proper_saso_construction<int64_t>(7, 20, 2, 1);
    // vec_nnz=3
    proper_saso_construction<int64_t>(7, 20, 0, 2);
    proper_saso_construction<int64_t>(7, 20, 1, 2);
    proper_saso_construction<int64_t>(7, 20, 2, 2);
    // vec_nnz=7
    proper_saso_construction<int64_t>(7, 20, 0, 3);
    proper_saso_construction<int64_t>(7, 20, 1, 3);
    proper_saso_construction<int64_t>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7) {
    // vec_nnz=1
    proper_saso_construction<int64_t>(15, 7, 0, 0);
    proper_saso_construction<int64_t>(15, 7, 1, 0);
    // vec_nnz=1
    proper_saso_construction<int64_t>(15, 7, 0, 1);
    proper_saso_construction<int64_t>(15, 7, 1, 1);
    // vec_nnz=3
    proper_saso_construction<int64_t>(15, 7, 0, 2);
    proper_saso_construction<int64_t>(15, 7, 1, 2);
    // vec_nnz=7
    proper_saso_construction<int64_t>(15, 7, 0, 3);
    proper_saso_construction<int64_t>(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_int32) {
    // test vec_nnz = 1, 2, 3, 7
    proper_saso_construction<int>(7, 20, 0, 0);
    proper_saso_construction<int>(7, 20, 0, 1);
    proper_saso_construction<int>(7, 20, 0, 2);
    proper_saso_construction<int>(7, 20, 0, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7_int32) {
    // test vec_nnz = 1, 2, 3, 7
    proper_saso_construction<int>(15, 7, 0, 0);
    proper_saso_construction<int>(15, 7, 0, 1);
    proper_saso_construction<int>(15, 7, 0, 2);
    proper_saso_construction<int>(15, 7, 0, 3);
}


////////////////////////////////////////////////////////////////////////
//
//
//     LASOs
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20) {
    // vec_nnz=1
    proper_laso_construction<int64_t>(7, 20, 0, 0);
    proper_laso_construction<int64_t>(7, 20, 1, 0);
    proper_laso_construction<int64_t>(7, 20, 2, 0);
    // vec_nnz=2
    proper_laso_construction<int64_t>(7, 20, 0, 1);
    proper_laso_construction<int64_t>(7, 20, 1, 1);
    proper_laso_construction<int64_t>(7, 20, 2, 1);
    // vec_nnz=3
    proper_laso_construction<int64_t>(7, 20, 0, 2);
    proper_laso_construction<int64_t>(7, 20, 1, 2);
    proper_laso_construction<int64_t>(7, 20, 2, 2);
    // vec_nnz=7
    proper_laso_construction<int64_t>(7, 20, 0, 3);
    proper_laso_construction<int64_t>(7, 20, 1, 3);
    proper_laso_construction<int64_t>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7) {
    // vec_nnz=1
    proper_laso_construction<int64_t>(15, 7, 0, 0);
    proper_laso_construction<int64_t>(15, 7, 1, 0);
    // vec_nnz=2
    proper_laso_construction<int64_t>(15, 7, 0, 1);
    proper_laso_construction<int64_t>(15, 7, 1, 1);
    // vec_nnz=3
    proper_laso_construction<int64_t>(15, 7, 0, 2);
    proper_laso_construction<int64_t>(15, 7, 1, 2);
    // vec_nnz=7
    proper_laso_construction<int64_t>(15, 7, 0, 3);
    proper_laso_construction<int64_t>(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(7, 20, 0, 0);
    proper_laso_construction<int>(7, 20, 1, 0);
    proper_laso_construction<int>(7, 20, 2, 0);
    // vec_nnz=2
    proper_laso_construction<int>(7, 20, 0, 1);
    proper_laso_construction<int>(7, 20, 1, 1);
    proper_laso_construction<int>(7, 20, 2, 1);
    // vec_nnz=3
    proper_laso_construction<int>(7, 20, 0, 2);
    proper_laso_construction<int>(7, 20, 1, 2);
    proper_laso_construction<int>(7, 20, 2, 2);
    // vec_nnz=7
    proper_laso_construction<int>(7, 20, 0, 3);
    proper_laso_construction<int>(7, 20, 1, 3);
    proper_laso_construction<int>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(15, 7, 0, 0);
    proper_laso_construction<int>(15, 7, 1, 0);
    // vec_nnz=2
    proper_laso_construction<int>(15, 7, 0, 1);
    proper_laso_construction<int>(15, 7, 1, 1);
    // vec_nnz=3
    proper_laso_construction<int>(15, 7, 0, 2);
    proper_laso_construction<int>(15, 7, 1, 2);
    // vec_nnz=7
    proper_laso_construction<int>(15, 7, 0, 3);
    proper_laso_construction<int>(15, 7, 1, 3);
}
