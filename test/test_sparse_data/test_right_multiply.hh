#include "../linop_common.hh"
#include "common.hh"
#include <gtest/gtest.h>

using namespace test::sparse_data::common;
using namespace test::linop_common;
using blas::Layout;


template <SparseMatrix SpMat>
class TestRightMultiply_Sparse : public ::testing::Test
{
    // C = alpha * opB(B) @ opA(submat(A)) + beta * C
    //
    //  In what follows, "self" refers to A and "other" refers to B.
    //
    using T = typename SpMat::scalar_t;
    protected:
    virtual void SetUp(){};
    virtual void TearDown(){};

    virtual SpMat make_test_matrix(int64_t m, int64_t n, T nonzero_prob, uint32_t key = 0) = 0;

    void multiply_eye(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A, m, n, 0, 0, layout, 0.0, 0);
    }

    void alpha_beta(uint32_t key, T alpha, T beta, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
       test_right_apply_submatrix_to_eye<T>(alpha, A, m, n, 0, 0, layout, beta, 0);
    }

    void transpose_self(uint32_t key, int64_t m, int64_t n, Layout layout, T p) {
        auto A = this->make_test_matrix(m, n, p, key);
        test_right_apply_tranpose_to_eye<T>(A, layout, 0);
    }

    void submatrix_self(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in A, columns in B = eye(m)
        int64_t d0,     // cols in A0
        int64_t m0,     // rows in A0
        int64_t A_ro,   // row offset for A in A0
        int64_t A_co,   // col offset for A in A0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A0 = this->make_test_matrix(m0, d0, p, key);
        test_right_apply_submatrix_to_eye<T>(1.0, A0, m, d, A_ro, A_co, layout, 0.0, 0);
    }

    void transpose_other(
        uint32_t key,  // key for RNG that generates sparse A
        int64_t d,     // cols in A
        int64_t n,     // rows in A and B
        int64_t m,     // cols in B
        Layout layout, // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(n, d, p, key);
        test_right_apply_to_transposed<T>(A, m, layout, 0);
    }

    void submatrix_other(
        uint32_t key,   // key for RNG that generates sparse A
        int64_t d,      // cols in A
        int64_t m,      // rows in B
        int64_t n,      // rows in A, columns in B
        int64_t m0,     // rows in B0
        int64_t n0,     // cols in B0
        int64_t B_ro,   // row offset for B in B0
        int64_t B_co,   // col offset for B in B0
        Layout layout,  // layout of dense matrix input and output
        T p
    ) {
        auto A = this->make_test_matrix(n, d, p, key);
        test_right_apply_to_submatrix<T>(A, m, m0, n0, B_ro, B_co, layout, 0);
    }
};