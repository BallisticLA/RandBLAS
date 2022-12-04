#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <math.h>

#define RELTOL_POWER 0.7
#define ABSTOL_POWER 0.75


template <typename T>
RandBLAS::sasos::SASO<T> make_wide_saso(
    int64_t n_rows,
    int64_t n_cols,
    int64_t vec_nnz,
    uint64_t ctr_offset,
    uint64_t key
) {
    assert(d <= m);
    RandBLAS::sasos::Dist D = {
        .n_rows = n_rows,
        .n_cols = n_cols,
        .vec_nnz = vec_nnz
    };
    int64_t total_nnz = n_cols * vec_nnz;
    RandBLAS::sasos::SASO<T> sas = {
        .dist = D,
        .key = key,
        .ctr_offset = ctr_offset,
        .rows = new int64_t[total_nnz],
        .cols = new int64_t[total_nnz],
        .vals = new T[total_nnz]
    };
    RandBLAS::sasos::fill_saso(sas);
    return sas;
}


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
        RandBLAS::sasos::SASO<double> sas = make_wide_saso<double>(
            d, m, vec_nnzs[nnz_index], 0, keys[key_index]
        );

        // check that each block of sas.dist.vec_nnz entries of sas.rows is
        // sampled without replacement from 0,...,n_rows - 1.
        std::set<int64_t> s;
        int64_t offset = 0;
        for (int64_t i = 0; i < sas.dist.n_cols; ++i) {
            offset = sas.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < sas.dist.vec_nnz; ++j) {
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
void sas_to_dense_rowmajor(RandBLAS::sasos::SASO<T> &sas, T *mat) {
    RandBLAS::sasos::Dist D = sas.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;

    int64_t nnz = D.n_cols * D.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = sas.rows[i];
        int64_t col = sas.cols[i];
        T val = sas.vals[i];
        mat[row * D.n_cols + col] = val;
    }
}

template <typename T>
void sas_to_dense_colmajor(RandBLAS::sasos::SASO<T> &sas, T *mat) {
    RandBLAS::sasos::Dist D = sas.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;

    int64_t nnz = D.n_cols * D.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = sas.rows[i];
        int64_t col = sas.cols[i];
        T val = sas.vals[i];
        mat[row + sas.dist.n_rows * col] = val;
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
        RandBLAS::sasos::SASO<T> sas = make_wide_saso<T>(
            d, m, vec_nnzs[nnz_index], 0, keys[key_index]
        );
        
        // compute S*A. 
        T *a_hat = new T[d * n]{};
        RandBLAS::sasos::sketch_cscrow<T>(sas, n, a, a_hat, threads);

        // compute expected result
        T *S = new T[d * m];
        sas_to_dense_rowmajor<T>(sas, S);
        T *a_hat_expect = new T[d * n]{}; // zero-initialize.
        int64_t lds = m;
        int64_t lda = n; 
        int64_t ldahat = n;
        blas::gemm<T>(
            blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S, lds, a, lda,
            0.0, a_hat_expect, ldahat
        );

        // check the result
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T abstol = std::pow(std::numeric_limits<T>::epsilon(), ABSTOL_POWER);
        for (int64_t i = 0; i < d; ++i) {
            for (int64_t j = 0; j < n; ++j) {
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
        RandBLAS::sasos::SASO<T> sas = make_wide_saso<T>(
            d, m, vec_nnzs[nnz_index], 0, keys[key_index]
        );
        
        // compute S*A. 
        T *a_hat = new T[d * n]{}; // zero-initialize.
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
            0.0, a_hat_expect, ldahat
        );

        // check the result
        T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
        T abstol = std::pow(std::numeric_limits<T>::epsilon(), ABSTOL_POWER);
        for (int64_t i = 0; i < d; ++i) {
            for (int64_t j = 0; j < n; ++j) {
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