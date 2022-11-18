#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;


class TestSJLTConstruction : public ::testing::Test
{
    // only tests column-sparse SJLTs for now.
    protected:
        int64_t d = 7;
        int64_t m = 20;
        std::vector<uint64_t> keys = {42, 0, 1};
        std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void proper_construction(int64_t key_index, int64_t nnz_index)
    {
        struct RandBLAS::sjlts::SJLT sjl;
        sjl.ori = RandBLAS::sjlts::ColumnWise;
        sjl.n_rows = d; // > n
        sjl.n_cols = m;
        sjl.vec_nnz = vec_nnzs[nnz_index]; // <= n_rows
        sjl.rows = new int64_t[sjl.vec_nnz * m];
        sjl.cols = new int64_t[sjl.vec_nnz * m];
        sjl.vals = new double[sjl.vec_nnz * m];
        RandBLAS::sjlts::fill_colwise(sjl, keys[key_index], 0);

        // check that each block of sjl.vec_nnz entries of sjl.rows is
        // sampled without replacement from 0,...,n_rows - 1.
        std::set<int64_t> s;
        int64_t offset = 0;
        for (int64_t i = 0; i < sjl.n_cols; ++i)
        {
            offset = sjl.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < sjl.vec_nnz; ++j)
            {
                int64_t row = sjl.rows[offset + j];
                ASSERT_EQ(s.count(row), 0);
                s.insert(row);
            }
        }
    } 
};

TEST_F(TestSJLTConstruction, Dim7by20nnz1)
{
    proper_construction(0, 0);
    proper_construction(1, 0);
    proper_construction(2, 0);
}

TEST_F(TestSJLTConstruction, Dim7by20nnz2)
{
    proper_construction(0, 1);
    proper_construction(1, 1);
    proper_construction(2, 1);
}

TEST_F(TestSJLTConstruction, Dim7by20nnz3)
{
    proper_construction(0, 2);
    proper_construction(1, 2);
    proper_construction(2, 2);
}

TEST_F(TestSJLTConstruction, Dim7by20nnz7)
{
    proper_construction(0, 3);
    proper_construction(1, 3);
    proper_construction(2, 3);
}



void sjlt_to_dense_rowmajor(RandBLAS::sjlts::SJLT sjl, double *mat)
{
    int64_t nnz = sjl.n_cols * sjl.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i)
    {
        int64_t row = sjl.rows[i];
        int64_t col = sjl.cols[i];
        double val = sjl.vals[i];
        mat[row * sjl.n_cols + col] = val;
    }
}



class TestApplyCscRowMaj : public ::testing::Test
{
    // only tests column-sparse SJLTs for now.
    protected:
        int64_t d = 19;
        int64_t m = 201;
        int64_t n = 12;
        std::vector<uint64_t> keys = {42, 0, 1};
        std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7, 19};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void apply(int64_t key_index, int64_t nnz_index, int threads)
    {
        uint64_t a_seed = 99;

        // construct test data: A
        double *a = new double[m * n];
        RandBLAS::util::genmat(m, n, a, a_seed);

        // construct test data: S
        struct RandBLAS::sjlts::SJLT sjl;
        sjl.ori = RandBLAS::sjlts::ColumnWise;
        sjl.n_rows = d; // > n
        sjl.n_cols = m;
        sjl.vec_nnz = vec_nnzs[nnz_index]; // <= n_rows
        sjl.rows = new int64_t[sjl.vec_nnz * m];
        sjl.cols = new int64_t[sjl.vec_nnz * m];
        sjl.vals = new double[sjl.vec_nnz * m];
        RandBLAS::sjlts::fill_colwise(sjl, keys[key_index], 0);
        
        // compute S*A. 
        double *a_hat = new double[d * n];
        for (int64_t i = 0; i < d * n; ++i)
        {
            a_hat[i] = 0.0;
        }
        RandBLAS::sjlts::sketch_cscrow(sjl, n, a, a_hat, threads);

        // compute expected result
        double *a_hat_expect = new double[d * n];
        double *S = new double[d * m];
        sjlt_to_dense_rowmajor(sjl, S);
        int lds = (int) m;
        int lda = (int) n; 
        int ldahat = (int) n;
        blas::gemm(
            blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S, lds, a, lda,
            0.0, a_hat_expect, ldahat);

        // check the result
        double reldtol = RELDTOL;
        for (int64_t i = 0; i < d; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                int64_t ell = i*n + j;
                double expect = a_hat_expect[ell];
                double actual = a_hat[ell];
                double atol = reldtol * std::min(abs(actual), abs(expect));
                if (atol == 0.0) atol = ABSDTOL;
                ASSERT_NEAR(actual, expect, atol);
            }    
        }
    }
};


TEST_F(TestApplyCscRowMaj, OneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply(k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestApplyCscRowMaj, TwoThreads)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply(k_idx, nz_idx, 2);
        }
    }
}
