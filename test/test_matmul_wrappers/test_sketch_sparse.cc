#include "test/test_matmul_cores/linop_common.hh"
// ^ That includes a ton of stuff.

using blas::Layout;
using blas::Op;

using RandBLAS::RNGState;
using RandBLAS::DenseDist;
using RandBLAS::DenseSkOp;
using RandBLAS::dims_before_op;
using RandBLAS::offset_and_ldim;
using RandBLAS::layout_to_strides;
using RandBLAS::sketch_sparse;
using namespace RandBLAS::sparse_data;

using test::linop_common::dimensions;
using test::linop_common::random_matrix;
using test::linop_common::to_explicit_buffer;
// ^ Call as to_explicit_buffer(denseskop, mat_s, layout).
//   That populates mat_s with data from the denseskop in layout
//   order with the smallest possible leading dimension.


template <SparseMatrix SpMat>
SpMat eye(int64_t m) {
    using      T = SpMat::scalar_t;
    using sint_t = SpMat::index_t;
    COOMatrix<T,sint_t> coo(m, m);
    coo.reserve(m);
    std::iota(coo.rows, coo.rows + m, (int64_t)0);
    std::iota(coo.cols, coo.cols + m, (int64_t)0);
    std::fill(coo.vals, coo.vals + m, (T) 1.0);
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    if constexpr (is_coo) {
        return coo;
    } else if constexpr (is_csc) {
        CSCMatrix<T,sint_t> csc(m, m);
        coo_to_csc(coo, csc);
        return csc;
    } else if constexpr (is_csr) {
        CSRMatrix<T,sint_t> csr(m, m);
        coo_to_csr(coo, csr);
        return csr;
    } else {
        randblas_require(false);
    }
}

// Adapted from test::linop_common::test_left_apply_transpose_to_eye.
template <typename T, typename DenseSkOp, SparseMatrix SpMat = COOMatrix<T,int64_t>>
void test_left_transposed_sketch_of_eye(
    // B = S^T * eye, where S is m-by-d, B is d-by-m
    DenseSkOp &S, Layout layout
) {
    auto [m, d] = dimensions(S);
    auto I = eye<SpMat>(m);
    std::vector<T> B(d * m, 0.0);
    bool is_colmajor = (Layout::ColMajor == layout);
    int64_t ldb = (is_colmajor) ? d : m;
    int64_t lds = (is_colmajor) ? m : d;

    lsksp3(
        layout, Op::Trans, Op::NoTrans, d, m, m,
        (T) 1.0, S, 0, 0, I, 0, 0, (T) 0.0, B.data(), ldb
    );

    std::vector<T> S_dense(m * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    test::comparison::matrices_approx_equal(
        layout, Op::Trans, d, m,
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}

// Adapted from test::linop_common::test_left_apply_submatrix_to_eye.
template <typename T, typename DenseSkOp, SparseMatrix SpMat = COOMatrix<T,int64_t>>
void test_left_submat_sketch_of_eye(
    // B = alpha * submat(S0) * eye + beta*B, where S = submat(S) is d1-by-m1 offset by (S_ro, S_co) in S0, and B is random.
    T alpha, DenseSkOp &S0, int64_t d1, int64_t m1, int64_t S_ro, int64_t S_co, Layout layout, T beta = 0.0
) {
    auto [d0, m0] = dimensions(S0);
    randblas_require(d0 >= d1);
    randblas_require(m0 >= m1);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t ldb = (is_colmajor) ? d1 : m1;

    // define a matrix to be sketched, and create workspace for sketch.
    auto I = eye<SpMat>(m1);
    auto B = std::get<0>(random_matrix<T>(d1, m1, RNGState(42)));
    std::vector<T> B_backup(B);

    // Perform the sketch
    lsksp3(
        layout, Op::NoTrans, Op::NoTrans, d1, m1, m1,
        alpha, S0, S_ro, S_co, I, 0, 0, beta, B.data(), ldb
    );

    // Check the result
    T *expect = new T[d0 * m0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor) ? d0 : m0; 
    auto [row_stride_s, col_stride_s] = layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < m1; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    test::comparison::matrices_approx_equal(
        layout, Op::NoTrans,
        d1, m1,
        B.data(), ldb,
        &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

    delete [] expect;
}

// Adapted from test::linop_common::test_right_apply_transpose_to_eye.
template <typename T, typename DenseSkOp, SparseMatrix SpMat = COOMatrix<T,int64_t>>
void test_right_transposed_sketch_of_eye(
    // B = eye * S^T, where S is d-by-n, so eye is order n and B is n-by-d
    DenseSkOp &S, Layout layout
) {
    auto [d, n] = dimensions(S);
    auto I = eye<SpMat>(n);
    std::vector<T> B(n * d, 0.0);
    bool is_colmajor = Layout::ColMajor == layout;
    int64_t ldb = (is_colmajor) ? n : d;
    int64_t lds = (is_colmajor) ? d : n;
    
    rsksp3(layout, Op::NoTrans, Op::Trans, n, d, n, (T) 1.0, I, 0, 0, S, 0, 0, (T) 0.0, B.data(), ldb);

    std::vector<T> S_dense(n * d, 0.0);
    to_explicit_buffer(S, S_dense.data(), layout);
    test::comparison::matrices_approx_equal(
        layout, Op::Trans, n, d, 
        B.data(), ldb, S_dense.data(), lds,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );
}

// Adapted from test::linop_common::test_right_apply_submatrix_to_eye.
template <typename T, typename DenseSkOp, SparseMatrix SpMat = COOMatrix<T,int64_t>>
void test_right_submat_sketch_of_eye(
    // B = alpha * eye * submat(S) + beta*B : submat(S) is n-by-d, eye is n-by-n, B is n-by-d and random
    T alpha, DenseSkOp &S0, int64_t n, int64_t d, int64_t S_ro, int64_t S_co, Layout layout, T beta = 0.0
) {
    auto [n0, d0] = dimensions(S0);
    randblas_require(n0 >= n);
    randblas_require(d0 >= d);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t ldb = (is_colmajor) ? n : d;

    auto I = eye<SpMat>(n);
    auto B = std::get<0>(random_matrix<T>(n, d, RNGState(11)));
    std::vector<T> B_backup(B);
    rsksp3(layout, Op::NoTrans, Op::NoTrans, n, d, n, alpha, I, 0, 0, S0, S_ro, S_co, beta, B.data(), ldb);

    T *expect = new T[n0 * d0];
    to_explicit_buffer(S0, expect, layout);
    int64_t ld_expect = (is_colmajor)? n0 : d0;
    auto [row_stride_s, col_stride_s] = layout_to_strides(layout, ld_expect);
    auto [row_stride_b, col_stride_b] = layout_to_strides(layout, ldb);
    int64_t offset = row_stride_s * S_ro + col_stride_s * S_co;
    #define MAT_E(_i, _j) expect[offset + (_i)*row_stride_s + (_j)*col_stride_s]
    #define MAT_B(_i, _j) B_backup[       (_i)*row_stride_b + (_j)*col_stride_b]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            MAT_E(i,j) = alpha * MAT_E(i,j) + beta * MAT_B(i, j);
        }
    }

    test::comparison::matrices_approx_equal(
        layout, Op::NoTrans, n, d, B.data(), ldb, &expect[offset], ld_expect,
        __PRETTY_FUNCTION__, __FILE__, __LINE__
    );

    delete [] expect;
}


class TestLSKSP3 : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void sketch_eye(uint32_t seed, int64_t m, int64_t d, bool preallocate, Layout layout) {
        DenseDist D(d, m);
        DenseSkOp<T> S0(D, seed);
        if (preallocate)
            RandBLAS::fill_dense(S0);
        test_left_submat_sketch_of_eye<T>(1.0, S0, d, m, 0, 0, layout, 0.0);
    }

    template <typename T>
    static void transpose_S(uint32_t seed, int64_t m, int64_t d, Layout layout) {
        DenseDist Dt(m, d);
        DenseSkOp<T> S0(Dt, seed);
        RandBLAS::fill_dense(S0);
        test_left_transposed_sketch_of_eye<T>(S0, layout);
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d,    // rows in sketch
        int64_t m,    // size of identity matrix
        int64_t d0,   // rows in S0
        int64_t m0,   // cols in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        Layout layout
    ) {
        randblas_require(d0 > d);
        randblas_require(m0 > m);
        DenseDist D(d0, m0);
        DenseSkOp<T> S0(D, seed);
        test_left_submat_sketch_of_eye<T>(1.0, S0, d, m, S_ro, S_co, layout, 0.0);
    }

};

////////////////////////////////////////////////////////////////////////
//
//
//      Basic sketching (vary preallocation, row vs col major)
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKSP3, sketch_eye_double_preallocate_colmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, sketch_eye_double_preallocate_rowmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, blas::Layout::RowMajor);
}

TEST_F(TestLSKSP3, sketch_eye_double_null_colmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, sketch_eye_double_null_rowmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, blas::Layout::RowMajor);
}

TEST_F(TestLSKSP3, sketch_eye_single_preallocate) {
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, sketch_eye_single_null) {
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, blas::Layout::ColMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKSP3, lift_eye_double_preallocate_colmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, lift_eye_double_preallocate_rowmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, blas::Layout::RowMajor);
}

TEST_F(TestLSKSP3, lift_eye_double_null_colmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, lift_eye_double_null_rowmajor) {
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, blas::Layout::RowMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      transpose of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKSP3, transpose_double_colmajor) {
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, blas::Layout::ColMajor);
}

TEST_F(TestLSKSP3, transpose_double_rowmajor) {
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, blas::Layout::RowMajor);
}

TEST_F(TestLSKSP3, transpose_single) {
    for (uint32_t seed : {0})
        transpose_S<float>(seed, 200, 30, blas::Layout::ColMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      Submatrices of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestLSKSP3, submatrix_s_double_colmajor) {
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}

TEST_F(TestLSKSP3, submatrix_s_double_rowmajor) {
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::RowMajor
        );
}

TEST_F(TestLSKSP3, submatrix_s_single) {
    for (uint32_t seed : {0})
        submatrix_S<float>(seed,
            3, 10, // (rows, cols) in S.
            8, 12, // (rows, cols) in S0.
            3, // The first row of S is in the forth row of S0
            1, // The first col of S is in the second col of S0
            blas::Layout::ColMajor
        );
}


class TestRSKSP3 : public ::testing::Test
{
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    template <typename T>
    static void sketch_eye(uint32_t seed, int64_t m, int64_t d, bool preallocate, Layout layout) {
        DenseDist D(m, d);
        DenseSkOp<T> S0(D, seed);
        if (preallocate)
            RandBLAS::fill_dense(S0);
        test_right_submat_sketch_of_eye<T>(1.0, S0, m, d, 0, 0, layout, 0.0);
    }

    template <typename T>
    static void transpose_S(uint32_t seed, int64_t m, int64_t d, Layout layout) {
        DenseDist Dt(d, m);
        DenseSkOp<T> S0(Dt, seed);
        test_right_transposed_sketch_of_eye<T>(S0, layout);
    }

    template <typename T>
    static void submatrix_S(
        uint32_t seed,
        int64_t d, // columns in sketch
        int64_t m, // size of identity matrix
        int64_t d0, // cols in S0
        int64_t m0, // rows in S0
        int64_t S_ro, // row offset for S in S0
        int64_t S_co, // column offset for S in S0
        Layout layout
    ) {
        DenseDist D(m0, d0);
        DenseSkOp<T> S0(D, seed);
        test_right_submat_sketch_of_eye<T>(1.0, S0, m, d, S_ro, S_co, layout, 0.0);
    }

};


////////////////////////////////////////////////////////////////////////
//
//
//      RSKSP3: Basic sketching (vary preallocation, row vs col major)
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKSP3, right_sketch_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, Layout::ColMajor);
}

TEST_F(TestRSKSP3, right_sketch_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, true, Layout::RowMajor);
}

TEST_F(TestRSKSP3, right_sketch_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, Layout::ColMajor);
}

TEST_F(TestRSKSP3, right_sketch_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 200, 30, false, Layout::RowMajor);
}

TEST_F(TestRSKSP3, right_sketch_eye_single_preallocate)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, true, Layout::ColMajor);
}

TEST_F(TestRSKSP3, right_sketch_eye_single_null)
{
    for (uint32_t seed : {0})
        sketch_eye<float>(seed, 200, 30, false, Layout::ColMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKSP3: Lifting
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKSP3, right_lift_eye_double_preallocate_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, Layout::ColMajor);
}

TEST_F(TestRSKSP3, right_lift_eye_double_preallocate_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, true, Layout::RowMajor);
}

TEST_F(TestRSKSP3, right_lift_eye_double_null_colmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, Layout::ColMajor);
}

TEST_F(TestRSKSP3, right_lift_eye_double_null_rowmajor)
{
    for (uint32_t seed : {0})
        sketch_eye<double>(seed, 10, 51, false, Layout::RowMajor);
}


////////////////////////////////////////////////////////////////////////
//
//
//      RSKSP3: transpose of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKSP3, transpose_double_colmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, Layout::ColMajor);
}

TEST_F(TestRSKSP3, transpose_double_rowmajor)
{
    for (uint32_t seed : {0})
        transpose_S<double>(seed, 200, 30, Layout::RowMajor);
}

TEST_F(TestRSKSP3, transpose_single)
{
    for (uint32_t seed : {0})
        transpose_S<float>(seed, 200, 30, Layout::ColMajor);
}

////////////////////////////////////////////////////////////////////////
//
//
//      RSKSP3: Submatrices of S
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestRSKSP3, submatrix_s_double_colmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::ColMajor
        );
}

TEST_F(TestRSKSP3, submatrix_s_double_rowmajor) 
{
    for (uint32_t seed : {0})
        submatrix_S<double>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::RowMajor
        );
}

TEST_F(TestRSKSP3, submatrix_s_single) 
{
    for (uint32_t seed : {0})
        submatrix_S<float>(seed,
            3, 10, // (cols, rows) in S.
            8, 12, // (cols, rows) in S0.
            2, // The first row of S is in the third row of S0
            1, // The first col of S is in the second col of S0
            Layout::ColMajor
        );
}
