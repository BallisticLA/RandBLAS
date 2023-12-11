#include <RandBLAS/dense.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/util.hh>
#include <RandBLAS/test_util.hh>
#include <gtest/gtest.h>
#include <math.h>


class TestSparseSkOpConstruction : public ::testing::Test
{
    protected:
        std::vector<uint32_t> keys{42, 0, 1};
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T = double, typename RNG = RandBLAS::SparseSkOp<double>::RNG_t, RandBLAS::SignedInteger sint_t>
    void check_fixed_nnz_per_col(RandBLAS::SparseSkOp<T,RNG,sint_t> &S0) {
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

    template <typename T = double, typename RNG = RandBLAS::SparseSkOp<double>::RNG_t, RandBLAS::SignedInteger sint_t>
    void check_fixed_nnz_per_row(RandBLAS::SparseSkOp<T,RNG,sint_t> &S0) {
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

    virtual void proper_saso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index, bool use_int=false) {
        if (!use_int) {
            RandBLAS::SparseSkOp<double> S0(
                {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Short}, keys[key_index]
            );
            RandBLAS::fill_sparse(S0);
            if (d < m) {
                    check_fixed_nnz_per_col(S0);
            } else {
                    check_fixed_nnz_per_row(S0);
            }
        } else {
            using RNG = RandBLAS::SparseSkOp<double>::RNG_t;
            RandBLAS::SparseSkOp<double, RNG, int> S0(
                {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Short}, keys[key_index]
            );
            RandBLAS::fill_sparse(S0);
            if (d < m) {
                    check_fixed_nnz_per_col(S0);
            } else {
                    check_fixed_nnz_per_row(S0);
            }
        }
    } 

    virtual void proper_laso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index, bool use_int=false) {
        if (!use_int) {
            RandBLAS::SparseSkOp<double> S0(
                {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Long}, keys[key_index]
            );
            RandBLAS::fill_sparse(S0);
            if (d < m) {
                    check_fixed_nnz_per_row(S0);
            } else {
                    check_fixed_nnz_per_col(S0);
            } 
        } else {
            RandBLAS::SparseSkOp<double,r123::Philox4x32,int> S0(
                {d, m, vec_nnzs[nnz_index], RandBLAS::MajorAxis::Long}, keys[key_index]
            );
            RandBLAS::fill_sparse(S0);
            if (d < m) {
                    check_fixed_nnz_per_row(S0);
            } else {
                    check_fixed_nnz_per_col(S0);
            } 
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
    proper_saso_construction(7, 20, 0, 0);
    proper_saso_construction(7, 20, 1, 0);
    proper_saso_construction(7, 20, 2, 0);
    // vec_nnz=2
    proper_saso_construction(7, 20, 0, 1);
    proper_saso_construction(7, 20, 1, 1);
    proper_saso_construction(7, 20, 2, 1);
    // vec_nnz=3
    proper_saso_construction(7, 20, 0, 2);
    proper_saso_construction(7, 20, 1, 2);
    proper_saso_construction(7, 20, 2, 2);
    // vec_nnz=7
    proper_saso_construction(7, 20, 0, 3);
    proper_saso_construction(7, 20, 1, 3);
    proper_saso_construction(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7) {
    // vec_nnz=1
    proper_saso_construction(15, 7, 0, 0);
    proper_saso_construction(15, 7, 1, 0);
    // vec_nnz=1
    proper_saso_construction(15, 7, 0, 1);
    proper_saso_construction(15, 7, 1, 1);
    // vec_nnz=3
    proper_saso_construction(15, 7, 0, 2);
    proper_saso_construction(15, 7, 1, 2);
    // vec_nnz=7
    proper_saso_construction(15, 7, 0, 3);
    proper_saso_construction(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_int32) {
    proper_saso_construction(7, 20, 0, 0, true);
    proper_saso_construction(7, 20, 0, 1, true);
    proper_saso_construction(7, 20, 0, 2, true);
    proper_saso_construction(7, 20, 0, 3, true);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7_int32) {
    proper_saso_construction(15, 7, 0, 0, true);
    proper_saso_construction(15, 7, 0, 1, true);
    proper_saso_construction(15, 7, 0, 2, true);
    proper_saso_construction(15, 7, 0, 3, true);
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
    proper_laso_construction(7, 20, 0, 0);
    proper_laso_construction(7, 20, 1, 0);
    proper_laso_construction(7, 20, 2, 0);
    // vec_nnz=2
    proper_laso_construction(7, 20, 0, 1);
    proper_laso_construction(7, 20, 1, 1);
    proper_laso_construction(7, 20, 2, 1);
    // vec_nnz=3
    proper_laso_construction(7, 20, 0, 2);
    proper_laso_construction(7, 20, 1, 2);
    proper_laso_construction(7, 20, 2, 2);
    // vec_nnz=7
    proper_laso_construction(7, 20, 0, 3);
    proper_laso_construction(7, 20, 1, 3);
    proper_laso_construction(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7) {
    // vec_nnz=1
    proper_laso_construction(15, 7, 0, 0);
    proper_laso_construction(15, 7, 1, 0);
    // vec_nnz=2
    proper_laso_construction(15, 7, 0, 1);
    proper_laso_construction(15, 7, 1, 1);
    // vec_nnz=3
    proper_laso_construction(15, 7, 0, 2);
    proper_laso_construction(15, 7, 1, 2);
    // vec_nnz=7
    proper_laso_construction(15, 7, 0, 3);
    proper_laso_construction(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_int32) {
    // vec_nnz=1
    proper_laso_construction(7, 20, 0, 0, true);
    proper_laso_construction(7, 20, 1, 0, true);
    proper_laso_construction(7, 20, 2, 0, true);
    // vec_nnz=2
    proper_laso_construction(7, 20, 0, 1, true);
    proper_laso_construction(7, 20, 1, 1, true);
    proper_laso_construction(7, 20, 2, 1, true);
    // vec_nnz=3
    proper_laso_construction(7, 20, 0, 2, true);
    proper_laso_construction(7, 20, 1, 2, true);
    proper_laso_construction(7, 20, 2, 2, true);
    // vec_nnz=7
    proper_laso_construction(7, 20, 0, 3, true);
    proper_laso_construction(7, 20, 1, 3, true);
    proper_laso_construction(7, 20, 2, 3, true);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7_int32) {
    // vec_nnz=1
    proper_laso_construction(15, 7, 0, 0, true);
    proper_laso_construction(15, 7, 1, 0, true);
    // vec_nnz=2
    proper_laso_construction(15, 7, 0, 1, true);
    proper_laso_construction(15, 7, 1, 1, true);
    // vec_nnz=3
    proper_laso_construction(15, 7, 0, 2, true);
    proper_laso_construction(15, 7, 1, 2, true);
    // vec_nnz=7
    proper_laso_construction(15, 7, 0, 3, true);
    proper_laso_construction(15, 7, 1, 3, true);
}
