#include <RandBLAS/dense.hh>
#include <RandBLAS/sparse.hh>
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

    void check_fixed_nnz_per_col(RandBLAS::sparse::SparseSkOp<double> &S0) {
        std::set<int64_t> s;
        for (int64_t i = 0; i < S0.dist.n_cols; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                int64_t row = S0.rows[offset + j];
                ASSERT_EQ(s.count(row), 0) << "row index " << row << " was duplicated in column " << i << std::endl;
                s.insert(row);
            }
        }
    }

    void check_fixed_nnz_per_row(RandBLAS::sparse::SparseSkOp<double> &S0) {
        std::set<int64_t> s;
        for (int64_t i = 0; i < S0.dist.n_rows; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                int64_t col = S0.cols[offset + j];
                ASSERT_EQ(s.count(col), 0)  << "column index " << col << " was duplicated in row " << i << std::endl;
                s.insert(col);
            }
        }
    }

    virtual void proper_saso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        RandBLAS::sparse::SparseSkOp<double> S0(
            {d, m, RandBLAS::base::MajorAxis::Short, vec_nnzs[nnz_index]}, keys[key_index]
        );
       RandBLAS::sparse::fill_sparse(S0);
       if (d < m) {
            check_fixed_nnz_per_col(S0);
       } else {
            check_fixed_nnz_per_row(S0);
       }
    } 

    virtual void proper_laso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        RandBLAS::sparse::SparseSkOp<double> S0(
            {d, m, RandBLAS::base::MajorAxis::Long, vec_nnzs[nnz_index]}, keys[key_index]
        );
        RandBLAS::sparse::fill_sparse(S0);
       if (d < m) {
            check_fixed_nnz_per_row(S0);
       } else {
            check_fixed_nnz_per_col(S0);
       }
    } 
};

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_nnz_1) {
    proper_saso_construction(7, 20, 0, 0);
    proper_saso_construction(7, 20, 1, 0);
    proper_saso_construction(7, 20, 2, 0);
}

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_nnz_2) {
    proper_saso_construction(7, 20, 0, 1);
    proper_saso_construction(7, 20, 1, 1);
    proper_saso_construction(7, 20, 2, 1);
}

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_nnz_3) {
    proper_saso_construction(7, 20, 0, 2);
    proper_saso_construction(7, 20, 1, 2);
    proper_saso_construction(7, 20, 2, 2);
}

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_nnz_7) {
    proper_saso_construction(7, 20, 0, 3);
    proper_saso_construction(7, 20, 1, 3);
    proper_saso_construction(7, 20, 2, 3);
}

TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7) {
    proper_saso_construction(15, 7, 0, 0);
    proper_saso_construction(15, 7, 1, 0);

    proper_saso_construction(15, 7, 0, 1);
    proper_saso_construction(15, 7, 1, 1);

    proper_saso_construction(15, 7, 0, 2);
    proper_saso_construction(15, 7, 1, 2);

    proper_saso_construction(15, 7, 0, 3);
    proper_saso_construction(15, 7, 1, 3);
}

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_nnz_1) {
    proper_laso_construction(7, 20, 0, 0);
    proper_laso_construction(7, 20, 1, 0);
    proper_laso_construction(7, 20, 2, 0);
}

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_nnz_2) {
    proper_laso_construction(7, 20, 0, 1);
    proper_laso_construction(7, 20, 1, 1);
    proper_laso_construction(7, 20, 2, 1);
}

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_nnz_3) {
    proper_laso_construction(7, 20, 0, 2);
    proper_laso_construction(7, 20, 1, 2);
    proper_laso_construction(7, 20, 2, 2);
}

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_nnz_7) {
    proper_laso_construction(7, 20, 0, 3);
    proper_laso_construction(7, 20, 1, 3);
    proper_laso_construction(7, 20, 2, 3);
}

TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7) {
    proper_laso_construction(15, 7, 0, 0);
    proper_laso_construction(15, 7, 1, 0);

    proper_laso_construction(15, 7, 0, 1);
    proper_laso_construction(15, 7, 1, 1);

    proper_laso_construction(15, 7, 0, 2);
    proper_laso_construction(15, 7, 1, 2);

    proper_laso_construction(15, 7, 0, 3);
    proper_laso_construction(15, 7, 1, 3);
}
