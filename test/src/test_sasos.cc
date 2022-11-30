#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>

#define RELTOL_POWER 0.7
#define ABSTOL_POWER 0.75


class TestSASOConstruction : public ::testing::Test
{
    // only tests column-sparse SASOs for now.
    protected:
        int64_t d = 7;
        int64_t m = 20;
        std::vector<uint64_t> keys = {42, 0, 1};
        std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void proper_construction(int64_t key_index, int64_t nnz_index)
    {
        struct RandBLAS::sasos::SASO<double> sas;
        sas.n_rows = d; // > n
        sas.n_cols = m;
        sas.vec_nnz = vec_nnzs[nnz_index]; // <= n_rows
        sas.rows = new int64_t[sas.vec_nnz * m];
        sas.cols = new int64_t[sas.vec_nnz * m];
        sas.vals = new double[sas.vec_nnz * m];
        sas.key = keys[key_index];
        sas.ctr = 0;
        RandBLAS::sasos::fill_colwise(sas);

        // check that each block of sas.vec_nnz entries of sas.rows is
        // sampled without replacement from 0,...,n_rows - 1.
        std::set<int64_t> s;
        int64_t offset = 0;
        for (int64_t i = 0; i < sas.n_cols; ++i)
        {
            offset = sas.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < sas.vec_nnz; ++j)
            {
                int64_t row = sas.rows[offset + j];
                ASSERT_EQ(s.count(row), 0);
                s.insert(row);
            }
        }
    } 
};

TEST_F(TestSASOConstruction, Dim7by20nnz1)
{
    proper_construction(0, 0);
    proper_construction(1, 0);
    proper_construction(2, 0);
}

TEST_F(TestSASOConstruction, Dim7by20nnz2)
{
    proper_construction(0, 1);
    proper_construction(1, 1);
    proper_construction(2, 1);
}

TEST_F(TestSASOConstruction, Dim7by20nnz3)
{
    proper_construction(0, 2);
    proper_construction(1, 2);
    proper_construction(2, 2);
}

TEST_F(TestSASOConstruction, Dim7by20nnz7)
{
    proper_construction(0, 3);
    proper_construction(1, 3);
    proper_construction(2, 3);
}


template <typename T>
void sas_to_dense_rowmajor(RandBLAS::sasos::SASO<T> &sas, T *mat)
{
    for (int64_t i = 0; i < sas.n_rows * sas.n_cols; ++i)
    {
        mat[i] = 0.0;
    }
    int64_t nnz = sas.n_cols * sas.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i)
    {
        int64_t row = sas.rows[i];
        int64_t col = sas.cols[i];
        T val = sas.vals[i];
        mat[row * sas.n_cols + col] = val;
    }
}

template <typename T>
void sas_to_dense_colmajor(RandBLAS::sasos::SASO<T> &sas, T *mat)
{
    for (int64_t i = 0; i < sas.n_rows * sas.n_cols; ++i)
    {
        mat[i] = 0.0;
    }
    int64_t nnz = sas.n_cols * sas.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i)
    {
        int64_t row = sas.rows[i];
        int64_t col = sas.cols[i];
        T val = sas.vals[i];
        mat[row + sas.n_rows * col] = val;
    }
}

class TestApplyCsc : public ::testing::Test
{
    // only tests column-sparse SASOs for now.
    protected:
        static inline int64_t d = 19;
        static inline int64_t m = 201;
        static inline int64_t n = 12;
        static inline std::vector<uint64_t> keys = {42, 0, 1};
        static inline std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7, 19};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void apply_rowmajor(int64_t key_index, int64_t nnz_index, int threads)
    {
        uint64_t a_seed = 99;
        // construct test data: A
        T *a = new T[m * n];
        RandBLAS::util::genmat(m, n, a, a_seed);

        // construct test data: S
        struct RandBLAS::sasos::SASO<T> sas;
        sas.n_rows = d; // > n
        sas.n_cols = m;
        sas.vec_nnz = vec_nnzs[nnz_index]; // <= n_rows
        sas.rows = new int64_t[sas.vec_nnz * m];
        sas.cols = new int64_t[sas.vec_nnz * m];
        sas.vals = new T[sas.vec_nnz * m];
        sas.ctr = 0;
        sas.key = keys[key_index];
        RandBLAS::sasos::fill_colwise(sas);
        
        // compute S*A. 
        T *a_hat = new T[d * n];
        for (int64_t i = 0; i < d * n; ++i)
        {
            a_hat[i] = 0.0;
        }
        RandBLAS::sasos::sketch_cscrow<T>(sas, n, a, a_hat, threads);

        // compute expected result
        T *a_hat_expect = new T[d * n];
        T *S = new T[d * m];
        sas_to_dense_rowmajor<T>(sas, S);
        int64_t lds = m;
        int64_t lda = n; 
        int64_t ldahat = n;
        blas::gemm<T>(
            blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S, lds, a, lda,
            0.0, a_hat_expect, ldahat);

        // check the result
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T abstol = std::pow(std::numeric_limits<T>::epsilon(), ABSTOL_POWER);
        for (int64_t i = 0; i < d; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                int64_t ell = i*n + j;
                T expect = a_hat_expect[ell];
                T actual = a_hat[ell];
                T atol = reltol * std::min(abs(actual), abs(expect));
                if (atol == 0.0) atol = abstol;
                ASSERT_NEAR(actual, expect, atol);
            }    
        }
    }

    template <typename T>
    static void apply_colmajor(int64_t key_index, int64_t nnz_index, int threads)
    {
        uint64_t a_seed = 99;

        // construct test data: A
        T *a = new T[m * n];
        RandBLAS::util::genmat(m, n, a, a_seed);

        // construct test data: S
        struct RandBLAS::sasos::SASO<T> sas;
        sas.n_rows = d; // > n
        sas.n_cols = m;
        sas.vec_nnz = vec_nnzs[nnz_index]; // <= n_rows
        sas.rows = new int64_t[sas.vec_nnz * m];
        sas.cols = new int64_t[sas.vec_nnz * m];
        sas.vals = new T[sas.vec_nnz * m];
        sas.ctr = 0;
        sas.key = keys[key_index];
        RandBLAS::sasos::fill_colwise<T>(sas);
        
        // compute S*A. 
        T *a_hat = new T[d * n];
        for (int64_t i = 0; i < d * n; ++i)
        {
            a_hat[i] = 0.0;
        }
        RandBLAS::sasos::sketch_csccol<T>(sas, n, a, a_hat, threads);

        // compute expected result
        T *a_hat_expect = new T[d * n]{};
        // ^ zero-initialize.
        //      This should not be necessary since it first appears
        //      in a GEMM call with "beta = 0.0". However, tests run on GitHub Actions
        //      show that NaNs can propagate if a_hat_expect is not initialized to all
        //      zeros. See
        //          https://github.com/BallisticLA/RandBLAS/actions/runs/3579737699/jobs/6021220392
        //      for a successful run with the initialization, and
        //          https://github.com/BallisticLA/RandBLAS/actions/runs/3579714658/jobs/6021178532
        //      for an unsuccessful run without this initialization.
        T *S = new T[d * m];
        sas_to_dense_colmajor<T>(sas, S);
        int64_t lds = d;
        int64_t lda = m; 
        int64_t ldahat = d;
        blas::gemm<T>(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S, lds, a, lda,
            0.0, a_hat_expect, ldahat);

        // check the result
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T abstol = std::pow(std::numeric_limits<T>::epsilon(), ABSTOL_POWER);
        for (int64_t i = 0; i < d; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                // std::pair<int64_t, int64_t> p = {i, j};
                int64_t ell = i + j*d;
                T expect = a_hat_expect[ell];
                T actual = a_hat[ell];
                T atol = reltol * std::min(abs(actual), abs(expect));
                if (atol == 0.0) atol = abstol;
                EXPECT_NEAR(actual, expect, atol) << "\t" << i << ", " << j;
            }    
        }
    }
};


TEST_F(TestApplyCsc, OneThread_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply_rowmajor<double>(k_idx, nz_idx, 1);
            apply_rowmajor<float>(k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestApplyCsc, TwoThreads_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply_rowmajor<double>(k_idx, nz_idx, 2);
            apply_rowmajor<float>(k_idx, nz_idx, 2);
        }
    }
}

TEST_F(TestApplyCsc, OneThread_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply_colmajor<double>(k_idx, nz_idx, 1);
            apply_colmajor<float>(k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestApplyCsc, TwoThreads_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply_colmajor<double>(k_idx, nz_idx, 2);
            apply_colmajor<float>(k_idx, nz_idx, 2);
        }
    }
}