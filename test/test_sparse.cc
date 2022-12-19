#include <RandBLAS/sparse.hh>
#include <RandBLAS/util.hh>
#include <rbtutil.hh>
#include <gtest/gtest.h>
#include <math.h>


template <typename T>
RandBLAS::sparse::SparseSkOp<T> make_wide_saso(
    int64_t n_rows,
    int64_t n_cols,
    int64_t vec_nnz,
    uint32_t ctr_offset,
    uint32_t key
) {
    assert(n_rows <= n_cols);
    RandBLAS::sparse::SparseSkOp<T> sas(
        RandBLAS::sparse::SparseDistName::SASO,
        n_rows, n_cols, vec_nnz, key, ctr_offset
    );
    RandBLAS::sparse::fill_saso<T>(sas);
    return sas;
}


class TestSASOConstruction : public ::testing::Test
{
    // only tests column-sparse SASOs for now.
    protected:
        int64_t d = 7;
        int64_t m = 20;
        std::vector<uint32_t> keys = {42, 0, 1};
        std::vector<uint32_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void proper_construction(int64_t key_index, int64_t nnz_index)
    {
        RandBLAS::sparse::SparseSkOp<double> sas = make_wide_saso<double>(
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
void sas_to_dense(
    RandBLAS::sparse::SparseSkOp<T> &sas,
    T *mat,
    blas::Layout layout
) {
    RandBLAS::sparse::SparseDist D = sas.dist;
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

class TestLSKGES : public ::testing::Test
{
    // only tests column-sparse SASOs for now.
    protected:
        static inline int64_t d = 19;
        static inline int64_t m = 201;
        static inline int64_t n = 12;
        static inline std::vector<uint32_t> keys = {42, 0, 1};
        static inline std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7, 19};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void apply(blas::Layout layout, int64_t key_index, int64_t nnz_index, int threads)
    {
        uint32_t key = keys[key_index];
        uint64_t vec_nnz = vec_nnzs[nnz_index];
        uint32_t a_seed = 99;

        // construct test data: matrix A, SparseSkOp "sas", and dense representation S
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
        if (threads > 0) {
            RandBLAS::sparse::lskges<T>(
                layout, blas::Op::NoTrans, blas::Op::NoTrans,
                d, n, m,
                1.0, sas, 0, 0, a, lda,
                0.0, a_hat, ldahat,
                threads   
            );
        } else {
            RandBLAS::sparse::lskges<T>(
                layout, blas::Op::NoTrans, blas::Op::NoTrans,
                d, n, m,
                1.0, sas, 0, 0, a, lda,
                0.0, a_hat, ldahat
            );
        }

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
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::NoTrans,
            d, n,
            a_hat, ldahat,
            a_hat_expect, ldahat
        );
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d1, // rows in sketch
        int64_t m1, // size of identity matrix
        int64_t d0, // rows in S0
        int64_t m0, // cols in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        assert(d0 >= d1);
        assert(m0 >= m1);
        bool is_colmajor = layout == blas::Layout::ColMajor;
        int64_t pos = (is_colmajor) ? (S_ro + d0 * S_co) : (S_ro * m0 + S_co);
        assert(d0 * m0 >= pos + d1 * m1);

        int64_t vec_nnz = d0 / 4; // this is actually quite dense. 
        auto S0 = make_wide_saso<T>(d0, m0, vec_nnz, 0, seed);
        T *S0_dense = new T[d0 * m0];
        sas_to_dense<T>(S0, S0_dense, layout);
        int64_t lda, ldb, lds0;
        if (is_colmajor) {
            lda = m1;
            ldb = d1;
            lds0 = d0;
        } else {
            lda = m1; 
            ldb = m1;
            lds0 = m0;
        }

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m1 * m1, 0.0);
        for (int i = 0; i < m1; ++i)
            A[i + m1*i] = 1.0;
        std::vector<T> B(d1 * m1, 0.0);
        
        // Perform the sketch
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d1, m1, m1,
            1.0, S0, S_ro, S_co,
            A.data(), lda,
            0.0, B.data(), ldb   
        );
        // Check the result
        T *S_ptr = &S0_dense[pos];
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::NoTrans,
            d1, m1,
            B.data(), ldb,
            S_ptr, lds0
        );
    }

    // template <typename T>
    // static void submatrix_A(
    //     uint32_t seed_S0, // seed for S0
    //     int64_t d, // rows in S0
    //     int64_t m, // cols in S0, and rows in A.
    //     int64_t n, // cols in A
    //     int64_t m0, // rows in A0
    //     int64_t n0, // cols in A0
    //     int64_t A_ro, // row offset for A in A0
    //     int64_t A_co, // column offset for A in A0
    //     blas::Layout layout
    // ) {
    //     assert(m0 > m);
    //     assert(n0 > n);

    //     // Define the distribution for S0.
    //     RandBLAS::dense_op::Dist D = {
    //         .family = RandBLAS::dense_op::SparseDistName::Gaussian,
    //         .n_rows = d,
    //         .n_cols = m
    //     };
    //     // Define the sketching operator struct, S0.
    //     RandBLAS::dense_op::SketchingOperator<T> S0 = {
    //         .dist = D,
    //         .key = seed_S0,
    //         .layout = layout
    //     };
    //     bool is_colmajor = layout == blas::Layout::ColMajor;

    //     // define a matrix to be sketched, and create workspace for sketch.
    //     std::vector<T> A0(m0 * n0, 0.0);
    //     uint32_t ctr_A0 = 42;
    //     uint32_t seed_A0 = 42000;
    //     RandBLAS::dense_op::Dist DA0 = {.n_rows = m0, .n_cols = n0};
    //     RandBLAS::dense_op::fill_buff(A0.data(), DA0, ctr_A0, seed_A0);
    //     std::vector<T> B(d * n, 0.0);
    //     int64_t lda = (is_colmajor) ? DA0.n_rows : DA0.n_cols;
    //     int64_t ldb = (is_colmajor) ? d : n;
        
    //     // Perform the sketch
    //     int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
    //     T *A_ptr = &A0.data()[a_offset]; 
    //     RandBLAS::dense_op::lskge3<T>(
    //         S0.layout,
    //         blas::Op::NoTrans,
    //         blas::Op::NoTrans,
    //         d, n, m,
    //         1.0, S0, 0,
    //         A_ptr, lda,
    //         0.0, B.data(), ldb   
    //     );

    //     // Check the result
    //     int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
    //     std::vector<T> B_expect(d * n, 0.0);
    //     blas::gemm<T>(S0.layout, blas::Op::NoTrans, blas::Op::NoTrans,
    //         d, n, m,
    //         1.0, S0.buff, lds, A_ptr, lda,
    //         0.0, B_expect.data(), ldb
    //     );
    //     buffs_approx_equal(B.data(), B_expect.data(), d * n);
    // }
};


TEST_F(TestLSKGES, OneThread_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, TwoThreads_RowMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(blas::Layout::RowMajor, k_idx, nz_idx, 2);
            apply<float>(blas::Layout::RowMajor, k_idx, nz_idx, 2);
        }
    }
}

TEST_F(TestLSKGES, OneThread_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, TwoThreads_ColMajor)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(blas::Layout::ColMajor, k_idx, nz_idx, 2);
            apply<float>(blas::Layout::ColMajor, k_idx, nz_idx, 2);
        }
    }
}

TEST_F(TestLSKGES, DefaultThreads)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(blas::Layout::ColMajor, k_idx, nz_idx, 0);
            apply<double>(blas::Layout::RowMajor, k_idx, nz_idx, 0);
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S, column major
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGES, subset_rows_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}


TEST_F(TestLSKGES, subset_rows_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S,row major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, subset_rows_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}


TEST_F(TestLSKGES, subset_rows_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 10, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, subset_cols_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            3, 12, // (rows, cols) in S0.
            0,
            1, // The first col of S is in the second col of S0
            blas::Layout::RowMajor
        );
}


// TEST_F(TestLSKGES, submatrix_s_single) 
// {
//     for (uint32_t seed : {0})
//         submatrix_S<float>(seed,
//             3, 10, // (rows, cols) in S.
//             8, 12, // (rows, cols) in S0.
//             3, // The first row of S is in the forth row of S0
//             1, // The first col of S is in the second col of S0
//             blas::Layout::ColMajor
//         );
//}
