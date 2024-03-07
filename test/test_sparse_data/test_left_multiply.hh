#include "../linop_common.hh"
#include "common.hh"
#include <gtest/gtest.h>

using namespace test::sparse_data::common;
using namespace test::linop_common;
using blas::Layout;


template <typename SpMatrix>
class TestLeftMultiply_Sparse : public ::testing::Test
{
    using T = typename SpMatrix::scalar_t;
    // C = alpha * opA(submat(A)) @ opB(B) + beta * C
    // In what follows, "self" refers to A and "other" refers to B.
    protected:
    
    virtual void SetUp(){};

    virtual void TearDown(){};

    virtual SpMatrix make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) = 0;

    void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0);
    }

    void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        randblas_require(alpha != (T)1.0 || beta != (T)0.0);
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta);
    }

    void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_left_apply_transpose_to_eye<T>(A, layout);
    }

    void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // rows in A
        int64_t m,      // columns in A, rows in B = eye(m)
        int64_t d0,     // rows in A0
        int64_t m0,     // cols in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // column offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        randblas_require(d0 > d);
        randblas_require(m0 > m);
        auto A0 = this->make_test_matrix(d0, m0, p, key);
        test_left_apply_submatrix_to_eye<T>(1.0, A0, d, m, A_ro, A_co, layout, 0.0);
    }

    void submatrix_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        int64_t m0,    // rows in B0
        int64_t n0,    // cols in B0
        int64_t B_ro,  // row offset for B in B0
        int64_t B_co,  // column offset for B in B0
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        randblas_require(m0 > m);
        randblas_require(n0 > n);
        test_left_apply_to_submatrix<T>(A, n, m0, n0, B_ro, B_co, layout);
    }

    void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // rows in A
        int64_t m,     // cols in A, and rows in B.
        int64_t n,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(d, m, p, key);
        test_left_apply_to_transposed<T>(A, n, layout);
    }

};


