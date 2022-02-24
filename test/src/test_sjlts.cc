#include <rblas.hh>
#include <gtest/gtest.h>


class ColWiseSJLTTest : public ::testing::Test
{
    protected:
        uint64_t n_rows = 7;
        uint64_t n_cols = 20;
        std::vector<uint64_t> keys = {42, 0, 1, 2, 3};
        std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};


    virtual void proper_construction(uint64_t key_index, uint64_t nnz_index)
    {
        struct rblas::sjlts::SJLT sjl;
        sjl.ori = rblas::sjlts::ColumnWise;
        sjl.n_rows = n_rows;
        sjl.n_cols = n_cols;
        sjl.vec_nnz = vec_nnzs[nnz_index];
        uint64_t *rows = new uint64_t[sjl.vec_nnz * sjl.n_cols];
        sjl.rows = rows;
        uint64_t *cols = new uint64_t[sjl.vec_nnz * sjl.n_cols];
        sjl.cols = cols;
        double *vals = new double[sjl.vec_nnz * sjl.n_cols];
        sjl.vals = vals;

        rblas::sjlts::fill_colwise(sjl, keys[key_index], 0);
        // check that each block of sjl.vec_nnz entries of sjl.rows is
        // sampled without replacement from 0,...,n_rows - 1.
        std::set<uint64_t> s;
        uint64_t offset = 0;
        for (uint64_t i = 0; i < sjl.n_cols; ++i)
        {
            offset = sjl.vec_nnz * i;
            s.clear();
            for (uint64_t j = 0; j < sjl.vec_nnz; ++j)
            {
                uint64_t row = sjl.rows[offset + j];
                EXPECT_EQ(s.count(row), 0);
                s.insert(row);
            }
        }
    } 
};

TEST_F(ColWiseSJLTTest, Dim7by20nnz1)
{
    proper_construction(0, 0);
    proper_construction(1, 0);
    proper_construction(2, 0);
}

TEST_F(ColWiseSJLTTest, Dim7by20nnz2)
{
    proper_construction(0, 1);
    proper_construction(1, 1);
    proper_construction(2, 1);
}

TEST_F(ColWiseSJLTTest, Dim7by20nnz3)
{
    proper_construction(0, 2);
    proper_construction(1, 2);
    proper_construction(2, 2);
}

TEST_F(ColWiseSJLTTest, Dim7by20nnz7)
{
    proper_construction(0, 3);
    proper_construction(1, 3);
    proper_construction(2, 3);
}