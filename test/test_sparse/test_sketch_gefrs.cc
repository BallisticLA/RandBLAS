#include <RandBLAS/dense.hh>
#include <RandBLAS/sparse.hh>
#include <RandBLAS/skge.hh>
#include <RandBLAS/util.hh>
#include <RandBLAS/test_util.hh>
#include <gtest/gtest.h>
#include <math.h>


class TestRSKGES : public ::testing::Test
{
    protected:
        static inline std::vector<uint32_t> keys = {42, 0, 1};
        static inline std::vector<int64_t> vec_nnzs = {1, 2, 3, 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void sketch_eye(
        uint32_t seed,
        int64_t m,
        int64_t d,
        RandBLAS::sparse::SparsityPattern pattern,
        int64_t vec_nnz,
        blas::Layout layout
    ) {
        RandBLAS::sparse::SparseDist D = {
            .n_rows = m,
            .n_cols = d,
            .family = pattern,
            .vec_nnz = vec_nnz
        };
        RandBLAS::sparse::SparseSkOp<T> S0(D, seed);
        RandBLAS::sparse::fill_sparse(S0);
        std::vector<T> S0_dense(m * d, 0.0);
        RandBLAS_Testing::Util::sparseskop_to_dense<T>(S0, S0_dense.data(), layout);

        std::vector<T> eye(m * m, 0.0);
        for (int i = 0; i < m; ++i)
            eye[i + i*m] = 1.0;
        std::vector<T> B(m * d, 0.0);
        int64_t ldb = (layout == blas::Layout::ColMajor) ? m : d;

        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            m, d, m,
            1.0, eye.data(), m,
            S0, 0, 0,
            0.0, B.data(), ldb
        );

        RandBLAS_Testing::Util::buffs_approx_equal(B.data(), S0_dense.data(), d*m,
             __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    template <typename T>
    static void apply(
        RandBLAS::sparse::SparsityPattern distname,
        int64_t d,
        int64_t m,
        int64_t n,
        blas::Layout layout,
        int64_t key_index,
        int64_t nnz_index,
        int threads
    ) {
#if !defined (RandBLAS_HAS_OpenMP)
        UNUSED(threads);
#endif
        uint32_t a_seed = 99;
        bool is_colmajor = layout == blas::Layout::ColMajor;

        // construct test data: matrix A, SparseSkOp "S0", and dense representation S
        T *a = new T[m * n];
        T *B0 = new T[m * d]{};
        RandBLAS::util::genmat(m, n, a, a_seed);  
        RandBLAS::sparse::SparseDist D = {
            .n_rows=n, .n_cols=d, .family=distname, .vec_nnz=vec_nnzs[nnz_index]
        };
        RandBLAS::sparse::SparseSkOp<T> S0(D, keys[key_index]);
        RandBLAS::sparse::fill_sparse(S0);
        int64_t lda = (is_colmajor) ? m : n;
        int64_t ldb = (is_colmajor) ? m : d;

        // compute S*A. 
#if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        omp_set_num_threads(threads);
#endif
        RandBLAS::sketch_general<T>(
            layout, blas::Op::NoTrans, blas::Op::NoTrans,
            m, d, n,
            1.0, a, lda,
            S0, 0, 0,
            0.0, B0, ldb 
        );
#if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
#endif

        // compute expected result (B1) and allowable error (E)
        T *B1 = new T[d * m]{};
        T *E = new T[d * m]{};
        RandBLAS_Testing::Util::reference_rskges<T>(
            layout, blas::Op::NoTrans, blas::Op::NoTrans,
            m, d, n,
            1.0, a, lda,
            S0, 0, 0,
            0.0, B1, E, ldb
        );

        // check the result
        RandBLAS_Testing::Util::buffs_approx_equal<T>(
            B0, B1, E, m * d,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        delete [] a;
        delete [] B0;
        delete [] B1;
        delete [] E;
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d1, // cols in sketch
        int64_t n1, // size of identity matrix
        int64_t d0, // cols in S0
        int64_t n0, // rows in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        blas::Layout layout
    ) {
        assert(d0 >= d1);
        assert(n0 >= n1);
        bool is_colmajor = layout == blas::Layout::ColMajor;
        int64_t pos = (is_colmajor) ? (S_ro + n0 * S_co) : (S_ro * d0 + S_co);
        assert(d0 * n0 >= pos + d1 * n1);

        int64_t vec_nnz = d0 / 3; // this is actually quite dense. 
        RandBLAS::sparse::SparseSkOp<T> S0(
            {n0, d0, RandBLAS::sparse::SparsityPattern::SASO, vec_nnz}, seed
        );
        RandBLAS::sparse::fill_sparse(S0);
        T *S0_dense = new T[n0 * d0];
        RandBLAS_Testing::Util::sparseskop_to_dense<T>(S0, S0_dense, layout);
        int64_t ldb = (is_colmajor) ? n1 : d1;
        int64_t lds0 = (is_colmajor) ? n0 : d0;


        // define a matrix to be sketched, and create workspace for sketch.
        std::vector<T> A(n1 * n1, 0.0);
        for (int i = 0; i < n1; ++i)
            A[i + n1*i] = 1.0;
        std::vector<T> B(n1 * d1, 0.0);
        
        // Perform the sketch
#if defined (RandBLAS_HAS_OpenMP)
        int orig_threads = omp_get_num_threads();
        omp_set_num_threads(1);
#endif
        RandBLAS::sketch_general<T>(
            layout,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            n1, d1, n1,
            1.0, A.data(), n1,
            S0, S_ro, S_co,
            0.0, B.data(), ldb   
        );
#if defined (RandBLAS_HAS_OpenMP)
        omp_set_num_threads(orig_threads);
#endif

        // Check the result
        RandBLAS_Testing::Util::matrices_approx_equal(
            layout, blas::Op::NoTrans,
            n1, d1,
            B.data(), ldb,
            &S0_dense[pos], lds0,
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );

        delete [] S0_dense;
    }
};

////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Basic sketching
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, right_sketch_eye_saso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::sparse::SparsityPattern::SASO, 1, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_saso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::sparse::SparsityPattern::SASO, 1,  blas::Layout::RowMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_laso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::sparse::SparsityPattern::LASO, 1,  blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_sketch_eye_laso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 3, RandBLAS::sparse::SparsityPattern::LASO, 1,  blas::Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, right_lift_eye_saso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::sparse::SparsityPattern::SASO, 5, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_lift_eye_saso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::sparse::SparsityPattern::SASO, 5, blas::Layout::RowMajor);
}

TEST_F(TestRSKGES, right_lift_eye_laso_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::sparse::SparsityPattern::LASO, 13, blas::Layout::ColMajor);
}

TEST_F(TestRSKGES, right_lift_eye_laso_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 22, 51, RandBLAS::sparse::SparsityPattern::LASO, 13, blas::Layout::RowMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: more sketching
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKGES, sketch_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::SASO,
                12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(RandBLAS::sparse::SparsityPattern::SASO,
                12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}


TEST_F(TestRSKGES, sketch_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::LASO, 12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::LASO, 12, 19, 201, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, sketch_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::SASO, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::SASO, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, sketch_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::LASO, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::LASO, 12, 19, 201, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: more lifting
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, lift_saso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::SASO,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
            apply<float>(RandBLAS::sparse::SparsityPattern::SASO,
                201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1
            );
        }
    }
}

TEST_F(TestRSKGES, lift_laso_rowMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::LASO, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::LASO, 201, 19, 12, blas::Layout::RowMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, lift_saso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::SASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::SASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}

TEST_F(TestRSKGES, lift_laso_colMajor_oneThread)
{
    for (int64_t k_idx : {0, 1, 2}) {
        for (int64_t nz_idx: {1, 2, 3, 0}) {
            apply<double>(RandBLAS::sparse::SparsityPattern::LASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
            apply<float>(RandBLAS::sparse::SparsityPattern::LASO, 201, 19, 12, blas::Layout::ColMajor, k_idx, nz_idx, 1);
        }
    }
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Submatrices of S, column major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, subset_rows_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_rows_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            3,  // The first column of S is in the forth column of S0
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_colmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::ColMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_colmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            1, // The first row of S is in the second row of S0
            0,
            blas::Layout::ColMajor
        );
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKGES: Submatrices of S,row major
//
//
////////////////////////////////////////////////////////////////////////


TEST_F(TestRSKGES, subset_rows_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_rows_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 10, // (cols, rows) in S0.
            0,
            3, // The first column of S is in the forth column of S0
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_rowmajor1) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            0,
            0,
            blas::Layout::RowMajor
        );
}

TEST_F(TestRSKGES, subset_cols_s_rowmajor2) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            3, 12, // (cols, rows) in S0.
            1, // The first row of S is in the second row of S0
            0,
            blas::Layout::RowMajor
        );
}
