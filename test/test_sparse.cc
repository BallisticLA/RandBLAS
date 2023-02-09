#include <RandBLAS/dense.hh>
#include <RandBLAS/sparse.hh>
#include <RandBLAS/util.hh>
#include <RandBLAS/test_util.hh>
#include <gtest/gtest.h>
#include <math.h>


class TestSparseSkOpConstruction : public ::testing::Test
{
    // only tests column-sparse SASOs for now.
    protected:
        std::vector<uint32_t> keys = {42, 0, 1};
        std::vector<uint32_t> vec_nnzs = {1, 2, 3, 7};     
    
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
            RandBLAS::sparse::SparseDistName::SASO,
            d, m, vec_nnzs[nnz_index], keys[key_index], 0
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
            RandBLAS::sparse::SparseDistName::LASO,
            d, m, vec_nnzs[nnz_index], keys[key_index], 0
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

template <typename T>
void sparseskop_to_dense(
    RandBLAS::sparse::SparseSkOp<T> &S0,
    T *mat,
    blas::Layout layout
) {
    RandBLAS::sparse::SparseDist D = S0.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;
    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == blas::Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };
    int64_t nnz;
    if (D.family == RandBLAS::sparse::SparseDistName::SASO) {
        nnz = D.vec_nnz * MAX(D.n_rows, D.n_cols);
    } else {
        nnz = D.vec_nnz * MIN(D.n_rows, D.n_cols);
    }
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t row = S0.rows[i];
        int64_t col = S0.cols[i];
        T val = S0.vals[i];
        mat[idx(row, col)] = val;
    }
}

class TestLSKGES : public ::testing::Test
{
    protected:
        static inline std::vector<uint32_t> keys = {42, 0, 1};
        static inline std::vector<uint64_t> vec_nnzs = {1, 2, 3, 7, 19};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void apply(
        RandBLAS::sparse::SparseDistName distname,
        int64_t d,
        int64_t m,
        int64_t n,
        blas::Layout layout,
        int64_t key_index,
        int64_t nnz_index,
        int threads,
        T atol = T(10)*std::numeric_limits<T>::epsilon(),
        T rtol = std::numeric_limits<T>::epsilon()
    ) {
#if !defined (RandBLAS_HAS_OpenMP)
        UNUSED(threads);
#endif
        uint32_t a_seed = 99;

        // construct test data: matrix A, SparseSkOp "S0", and dense representation S
        T *a = new T[m * n];
        T *a_hat = new T[d * n]{};
        T *S = new T[d * m];
        RandBLAS::util::genmat(m, n, a, a_seed);
        RandBLAS::sparse::SparseSkOp<T> S0(
            distname, d, m,
            vec_nnzs[nnz_index], keys[key_index], 0
        );
        RandBLAS::sparse::fill_sparse(S0);
        sparseskop_to_dense<T>(S0, S, layout);
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
#if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        omp_set_num_threads(threads);
#endif
        RandBLAS::sparse::lskges<T>(
            layout, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S0, 0, 0, a, lda,
            0.0, a_hat, ldahat 
        );
#if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
#endif

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
            a_hat_expect, ldahat,
            __PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        );

        delete [] a;
        delete [] a_hat;
        delete [] S;
        delete [] a_hat_expect;
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

        int64_t vec_nnz = d0 / 3; // this is actually quite dense. 
        RandBLAS::sparse::SparseSkOp<T> S0(
            RandBLAS::sparse::SparseDistName::SASO,
            d0, m0, vec_nnz, seed, 0
        );
        RandBLAS::sparse::fill_sparse(S0);
        T *S0_dense = new T[d0 * m0];
        sparseskop_to_dense<T>(S0, S0_dense, layout);
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
#if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        omp_set_num_threads(1);
#endif
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d1, m1, m1,
            1.0, S0, S_ro, S_co,
            A.data(), lda,
            0.0, B.data(), ldb   
        );
#if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
#endif

        // Check the result
        T *S_ptr = &S0_dense[pos];
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::NoTrans,
            d1, m1,
            B.data(), ldb,
            S_ptr, lds0,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        delete [] S0_dense;
    }

    template <typename T>
    static void alpha_beta(
        uint32_t key,
        T alpha,
        T beta,
        int64_t m,
        int64_t d,
        blas::Layout layout
    ) {
        bool is_colmajor = (layout == blas::Layout::ColMajor);
        randblas_require(m > d);
        int64_t vec_nnz = d / 2;
        RandBLAS::sparse::SparseDist DS = {RandBLAS::sparse::SparseDistName::SASO, d, m, vec_nnz};
        RandBLAS::sparse::SparseSkOp<T> S(DS, key, 0, NULL, NULL, NULL);
        RandBLAS::sparse::fill_sparse(S);
    
        // define a matrix to be sketched
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;

        // create initialized workspace for the sketch
        std::vector<T> B0(d * m);
        RandBLAS::dense::DenseDist DB = {.n_rows = d, .n_cols = m};
        RandBLAS::base::RNGState state(42, 42000);
        RandBLAS::dense::fill_buff(B0.data(), DB, state);
        int64_t ldb = (is_colmajor) ? d : m;
        std::vector<T> B1(d * m);
        blas::copy(d * m, B0.data(), 1, B1.data(), 1);

        // perform the sketch
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m,
            alpha, S, 0, 0,
            A.data(), m,
            beta, B0.data(), ldb
        );

        // compute the reference result
        std::vector<T> S_dense(d * m);
        sparseskop_to_dense<T>(S, S_dense.data(), layout);
        int64_t lds = (is_colmajor) ? d : m;
        blas::gemm(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, m, m, 
            alpha, S_dense.data(), lds,
            A.data(), m,
            beta, B1.data(), ldb
        );

        RandBLAS_Testing::Util::buffs_approx_equal(B0.data(), B1.data(), d * m,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void transpose_S(
        RandBLAS::sparse::SparseDistName distname,
        uint32_t key,
        int64_t m,
        int64_t d,
        blas::Layout layout
    ) {
        randblas_require(m > d);
        bool is_saso = (distname == RandBLAS::sparse::SparseDistName::SASO);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        RandBLAS::sparse::SparseDist Dt = {
            .family = distname,
            .n_rows = m,
            .n_cols = d,
            .vec_nnz = vec_nnz
        };
        RandBLAS::sparse::SparseSkOp<T> S0(Dt, key, 0, NULL, NULL, NULL);
        RandBLAS::sparse::fill_sparse(S0);
        std::vector<T> S0_dense(m * d);
        sparseskop_to_dense<T>(S0, S0_dense.data(), layout);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            A[i + m*i] = 1.0;
        std::vector<T> B(d * m, 0.0);
        bool is_colmajor = (blas::Layout::ColMajor == layout);
        int64_t ldb = (is_colmajor) ? d : m;
        int64_t lds = (is_colmajor) ? m : d;

        // perform the sketch
        //  S0 is tall.
        //  We apply S0.T, which is wide.
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::Trans,
            blas::Op::NoTrans,
            d, m, m,
            1.0, S0, 0, 0, A.data(), m,
            0.0, B.data(), ldb   
        );

        // check that B == S.T
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::Trans, d, m,
            B.data(), ldb, S0_dense.data(), lds,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void submatrix_A(
        RandBLAS::sparse::SparseDistName distname,
        uint32_t seed_S0, // seed for S0
        int64_t d, // rows in S0
        int64_t m, // cols in S0, and rows in A.
        int64_t n, // cols in A
        int64_t m0, // rows in A0
        int64_t n0, // cols in A0
        int64_t A_ro, // row offset for A in A0
        int64_t A_co, // column offset for A in A0
        blas::Layout layout
    ) {
        assert(m0 > m);
        assert(n0 > n);
        bool is_colmajor = (layout == blas::Layout::ColMajor);

        // Define the distribution for S0.
        bool is_saso = (distname == RandBLAS::sparse::SparseDistName::SASO);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        RandBLAS::sparse::SparseDist D = {
            .family = distname,
            .n_rows = d,
            .n_cols = m,
            .vec_nnz = vec_nnz
        };
        RandBLAS::sparse::SparseSkOp<T> S0(D, seed_S0, 0, NULL, NULL, NULL);
        RandBLAS::sparse::fill_sparse(S0);
        std::vector<T> S0_dense(m * d);
        sparseskop_to_dense<T>(S0, S0_dense.data(), layout);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A0(m0 * n0, 0.0);
        uint32_t ctr_A0 = 42;
        uint32_t seed_A0 = 42000;
        RandBLAS::dense::DenseDist DA0 = {
            .family=RandBLAS::dense::DenseDistName::Uniform,
            .n_rows = m0,
            .n_cols = n0
        };
        RandBLAS::dense::fill_buff(A0.data(), DA0, RandBLAS::base::RNGState{ctr_A0, seed_A0});
        std::vector<T> B(d * n, 0.0);
        int64_t lda = (is_colmajor) ? DA0.n_rows : DA0.n_cols;
        int64_t ldb = (is_colmajor) ? d : n;
        
        // Perform the sketch
        int64_t a_offset = (is_colmajor) ? (A_ro + m0 * A_co) : (A_ro * n0 + A_co);
        T *A_ptr = &A0.data()[a_offset]; 
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            d, n, m,
            1.0, S0, 0, 0,
            A_ptr, lda,
            0.0, B.data(), ldb   
        );

        // Check the result
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
        std::vector<T> B_expect(d * n, 0.0);
        blas::gemm<T>(layout, blas::Op::NoTrans, blas::Op::NoTrans,
            d, n, m,
            1.0, S0_dense.data(), lds, A_ptr, lda,
            0.0, B_expect.data(), ldb
        );
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), B_expect.data(), d * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void transpose_A(
        RandBLAS::sparse::SparseDistName distname,
        uint32_t seed_S0, // seed for S0
        int64_t d, // rows in S0
        int64_t m, // cols in S0, and rows in A.
        int64_t n, // cols in A
        blas::Layout layout
    ) {
        bool is_colmajor = (layout == blas::Layout::ColMajor);

        // Define the distribution for S0.
        bool is_saso = (distname == RandBLAS::sparse::SparseDistName::SASO);
        int64_t vec_nnz = (is_saso) ?  d/2 : m/2;
        RandBLAS::sparse::SparseDist D = {
            .family = distname,
            .n_rows = d,
            .n_cols = m,
            .vec_nnz = vec_nnz
        };
        RandBLAS::sparse::SparseSkOp<T> S0(D, seed_S0, 0, NULL, NULL, NULL);
        RandBLAS::sparse::fill_sparse(S0);
        std::vector<T> S0_dense(m * d);
        sparseskop_to_dense<T>(S0, S0_dense.data(), layout);

        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> At(m * n, 0.0);
        uint32_t ctr_A = 42;
        uint32_t seed_A = 42000;
        RandBLAS::dense::DenseDist DAt = {
            .family=RandBLAS::dense::DenseDistName::Uniform,
            .n_rows = n,
            .n_cols = m
        };
        RandBLAS::dense::fill_buff(At.data(), DAt, RandBLAS::base::RNGState{ctr_A, seed_A});
        std::vector<T> B(d * n, 0.0);
        int64_t lda = (is_colmajor) ? DAt.n_rows : DAt.n_cols;
        int64_t ldb = (is_colmajor) ? d : n;
        
        // Perform the sketch
        RandBLAS::sparse::lskges<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::Trans,
            d, n, m,
            1.0, S0, 0, 0,
            At.data(), lda,
            0.0, B.data(), ldb   
        );

        // Check the result
        int64_t lds = (is_colmajor) ? S0.dist.n_rows : S0.dist.n_cols;
        std::vector<T> B_expect(d * n, 0.0);
        blas::gemm<T>(layout, blas::Op::NoTrans, blas::Op::Trans,
            d, n, m,
            1.0, S0_dense.data(), lds, At.data(), lda,
            0.0, B_expect.data(), ldb
        );
        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), B_expect.data(), d * n,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }
};


////////////////////////////////////////////////////////////////////////
//
//
//      Sketch with SASOs and LASOs.
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGES, sketch_saso_rowMajor_oneThread)
{
    // TODO: this test fails on MacOS w/ Apple Clang compiler but not with
    // brew'd Clang ?? even w a relTol of 500*epsilon it fails

    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            auto deps = std::numeric_limits<double>::epsilon();

            apply<double>(RandBLAS::sparse::SparseDistName::SASO,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1,
                10*deps, 10*deps
            );

            auto feps = std::numeric_limits<float>::epsilon();

            apply<float>(RandBLAS::sparse::SparseDistName::SASO,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1,
                10*feps, 10*feps
            );
        }
    }
}


TEST_F(TestLSKGES, sketch_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::LASO, 19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::LASO, 19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

#if defined (RandBLAS_HAS_OpenMP)
TEST_F(TestLSKGES, sketch_saso_rowMajor_FourThreads)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            auto deps = std::numeric_limits<double>::epsilon();

            apply<double>(RandBLAS::sparse::SparseDistName::SASO,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 4,
                10*deps, 10*deps
            );

            auto feps = std::numeric_limits<float>::epsilon();

            apply<float>(RandBLAS::sparse::SparseDistName::SASO,
                19, 201, 12, blas::Layout::RowMajor, k_idx, nz_idx, 4,
                10*feps, 10*feps
            );
        }
    }
}
#endif

TEST_F(TestLSKGES, sketch_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::SASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::SASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, sketch_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::LASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::LASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

#if defined (RandBLAS_HAS_OpenMP)
TEST_F(TestLSKGES, sketch_saso_colMajor_fourThreads)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::SASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 4);
            apply<float>(RandBLAS::sparse::SparseDistName::SASO, 19, 201, 12, blas::Layout::ColMajor, k_idx, nz_idx, 4);
        }
    }
}
#endif


////////////////////////////////////////////////////////////////////////
//
//
//      Lift with SASOs and LASOs.
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, lift_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            // TODO: this test fails on MacOS w/ Apple Clang compiler
            // but not with brew'd Clang ?? increase relTol by 4 resolves

            auto deps = std::numeric_limits<double>::epsilon();

            apply<double>(RandBLAS::sparse::SparseDistName::SASO,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1,
                10*deps, 4*deps
            );

            auto feps = std::numeric_limits<float>::epsilon();

            apply<float>(RandBLAS::sparse::SparseDistName::SASO,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1,
                10*feps, 4*feps
            );
        }
    }
}

TEST_F(TestLSKGES, lift_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::LASO, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::LASO, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, lift_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::SASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::SASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestLSKGES, lift_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2})
    {
        for (int64_t nz_idx: {4, 1, 2, 3, 0})
        {
            apply<double>(RandBLAS::sparse::SparseDistName::LASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparseDistName::LASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
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


////////////////////////////////////////////////////////////////////////
//
//
//      transpose of S
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, transpose_saso_double_colmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(RandBLAS::sparse::SparseDistName::SASO, seed, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, transpose_laso_double_colmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(RandBLAS::sparse::SparseDistName::LASO, seed, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, transpose_saso_double_rowmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(RandBLAS::sparse::SparseDistName::SASO, seed, 21, 4, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, transpose_laso_double_rowmajor)
{
    uint32_t seed = 0;
    transpose_S<double>(RandBLAS::sparse::SparseDistName::LASO, seed, 21, 4, blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//     submatrix of A
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, saso_submatrix_a_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            RandBLAS::sparse::SparseDistName::SASO,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, saso_submatrix_a_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            RandBLAS::sparse::SparseDistName::SASO,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKGES, laso_submatrix_a_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            RandBLAS::sparse::SparseDistName::LASO,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKGES, laso_submatrix_a_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_A<double>(
            RandBLAS::sparse::SparseDistName::LASO,
            seed,
            3, // number of rows in sketch
            10, 5, // (rows, cols) in A.
            12, 8, // (rows, cols) in A0.
            2, // The first row of A is in the third row of A0.
            1, // The first col of A is in the second col of A0.
            blas::Layout::RowMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//     transpose of A
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestLSKGES, saso_times_trans_A_colmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(RandBLAS::sparse::SparseDistName::SASO, seed, 7, 22, 5, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, laso_times_trans_A_colmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(RandBLAS::sparse::SparseDistName::LASO, seed, 7, 22, 5, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, saso_times_trans_A_rowmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(RandBLAS::sparse::SparseDistName::SASO, seed, 7, 22, 5, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, laso_times_trans_A_rowmajor)
{
    uint32_t seed = 0;
    transpose_A<double>(RandBLAS::sparse::SparseDistName::LASO, seed, 7, 22, 5, blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//     (alpha, beta), where alpha != 1.0.
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKGES, nontrivial_scales_colmajor1)
{
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_colmajor2)
{
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::ColMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_rowmajor1)
{
    double alpha = 5.5;
    double beta = 0.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::RowMajor);
}

TEST_F(TestLSKGES, nontrivial_scales_rowmajor2)
{
    double alpha = 5.5;
    double beta = -1.0;
    alpha_beta<double>(0, alpha, beta, 21, 4, blas::Layout::RowMajor);
}
