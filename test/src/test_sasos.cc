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

// TODO: move this to a common place where test_dense_op.cc and
//       test_sasos.cc can use it.
template <typename T>
void matrices_approx_equal(
    blas::Layout layout,
    blas::Op transB,
    int64_t m,
    int64_t n,
    const T *A,
    int64_t lda,
    const T *B,
    int64_t ldb
) {
    // check that A == op(B), where A is m-by-n.
    T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
    auto idxa = [lda, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*lda) : (j + i*lda);
    };
    auto idxb = [ldb, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*ldb) : (j + i*ldb);
    };
    if (transB == blas::Op::NoTrans) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                T actual = A[idxa(i, j)];
                T expect = B[idxb(i, j)];
                T atol = reltol * std::min(abs(actual), abs(expect));
                EXPECT_NEAR(actual, expect, atol);
            }
        }
    } else {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                T actual = A[idxa(i, j)];
                T expect = B[idxb(j, i)];
                T atol = reltol * std::min(abs(actual), abs(expect));
                EXPECT_NEAR(actual, expect, atol);
            }
        }
    }
}

template <typename T>
void sas_to_dense(
    RandBLAS::sasos::SASO<T> &sas,
    T *mat,
    blas::Layout layout
) {
    RandBLAS::sasos::Dist D = sas.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;

    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };

    int64_t nnz = D.n_cols * D.vec_nnz;
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = sas.rows[i];
        int64_t col = sas.cols[i];
        T val = sas.vals[i];
        mat[idx(row, col)] = val;
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
    static void apply(int64_t key_index, int64_t nnz_index, int threads, blas::Layout layout)
    {
        uint64_t key = keys[key_index];
        uint64_t vec_nnz = vec_nnzs[nnz_index];
        uint64_t a_seed = 99;

        // construct test data: matrix A, SASO "sas", and dense representation S
        T *a = new T[m * n];
        T *a_hat = new T[d * n]{};
        T *S = new T[d * m];
        RandBLAS::util::genmat(m, n, a, a_seed);
        auto sas = make_wide_saso<T>(d, m, vec_nnz, 0, key);
        sas_to_dense<T>(sas, S, layout);
        int64_t lda, ldahat, lds;
        if (layout == blas::Layout::RowMajor) {
            lda = n; 
            ldahat = n;
            lds = m;
        } else {
            lda = m;
            ldahat = d;
            lds = d;
        }

        // compute S*A. 
        RandBLAS::sasos::lskges<T>(
            layout, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, sas, 0, a, lda,
            0.0, a_hat, ldahat,
            threads   
        );

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
        blas::gemm<T>(
            layout, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S, lds, a, lda,
            0.0, a_hat_expect, ldahat
        );

        // check the result
        matrices_approx_equal(
            layout, blas::Op::NoTrans,
            d, n,
            a_hat, ldahat,
            a_hat_expect, ldahat
        );
    }
};


TEST_F(TestApplyCsc, OneThread_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(k_idx, nz_idx, 1, blas::Layout::RowMajor);
            apply<float>(k_idx, nz_idx, 1, blas::Layout::RowMajor);
        }
    }
}

TEST_F(TestApplyCsc, TwoThreads_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(k_idx, nz_idx, 2, blas::Layout::RowMajor);
            apply<float>(k_idx, nz_idx, 2, blas::Layout::RowMajor);
        }
    }
}

TEST_F(TestApplyCsc, OneThread_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(k_idx, nz_idx, 1, blas::Layout::ColMajor);
            apply<float>(k_idx, nz_idx, 1, blas::Layout::ColMajor);
        }
    }
}

TEST_F(TestApplyCsc, TwoThreads_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(k_idx, nz_idx, 2, blas::Layout::ColMajor);
            apply<float>(k_idx, nz_idx, 2, blas::Layout::ColMajor);
        }
    }
}