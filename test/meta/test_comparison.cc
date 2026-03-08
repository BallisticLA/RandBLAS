#include "RandBLAS/testing/comparison.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using RandBLAS::testing::approx_equal;
using RandBLAS::testing::buffs_approx_equal;
using RandBLAS::testing::matrices_approx_equal;


// MARK: approx_equal (scalar)

class TestApproxEqualScalar : public ::testing::Test {};

// bool-returning overload

TEST_F(TestApproxEqualScalar, bool_pass_identical) {
    std::ostringstream oss;
    EXPECT_TRUE(approx_equal(1.0, 1.0, oss));
    EXPECT_TRUE(oss.str().empty());
}

TEST_F(TestApproxEqualScalar, bool_pass_within_atol) {
    double atol = 1e-6;
    double rtol = std::numeric_limits<double>::epsilon();
    double A = 0.0;
    double B = atol / 2.0;   // within atol
    std::ostringstream oss;
    EXPECT_TRUE(approx_equal(A, B, oss, atol, rtol));
}

TEST_F(TestApproxEqualScalar, bool_pass_within_rtol) {
    double rtol = 1e-6;
    double atol = std::numeric_limits<double>::epsilon();
    double A = 1.0;
    double B = A * (1.0 + rtol / 2.0);   // within rtol
    std::ostringstream oss;
    EXPECT_TRUE(approx_equal(A, B, oss, atol, rtol));
}

TEST_F(TestApproxEqualScalar, bool_fail_outside_tols) {
    double atol = 1e-10;
    double rtol = 1e-10;
    std::ostringstream oss;
    bool result = approx_equal(1.0, 2.0, oss, atol, rtol);
    EXPECT_FALSE(result);
    EXPECT_FALSE(oss.str().empty());
}

// string-returning overload

TEST_F(TestApproxEqualScalar, string_pass_returns_empty) {
    auto msg = approx_equal(1.0, 1.0, __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestApproxEqualScalar, string_fail_returns_nonempty) {
    int line = __LINE__ + 1;
    auto msg = approx_equal(1.0, 2.0, __PRETTY_FUNCTION__, __FILE__, line);
    EXPECT_FALSE(msg.empty());
    // error string should reference the file and line number
    EXPECT_NE(msg.find(__FILE__), std::string::npos);
    EXPECT_NE(msg.find(std::to_string(line)), std::string::npos);
}

TEST_F(TestApproxEqualScalar, string_respects_custom_atol) {
    // 0.0 and 1e-4 are far apart by default tolerances but within a large atol
    double A = 0.0, B = 1e-4;
    double large_atol = 1.0;
    auto msg = approx_equal(A, B, __PRETTY_FUNCTION__, __FILE__, __LINE__, large_atol);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestApproxEqualScalar, string_respects_custom_rtol) {
    double A = 1.0, B = 1.1;   // 10% relative difference
    double large_rtol = 0.5;   // 50% rtol → should pass
    auto msg = approx_equal(A, B, __PRETTY_FUNCTION__, __FILE__, __LINE__,
        std::numeric_limits<double>::epsilon(), large_rtol);
    EXPECT_TRUE(msg.empty());
}


// MARK: buffs_approx_equal (contiguous)

class TestBuffsApproxEqualContiguous : public ::testing::Test {};

TEST_F(TestBuffsApproxEqualContiguous, pass_all_equal) {
    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    auto msg = buffs_approx_equal(a.data(), a.data(), (int64_t)a.size(),
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestBuffsApproxEqualContiguous, pass_empty_buffer) {
    std::vector<double> a = {};
    auto msg = buffs_approx_equal(a.data(), a.data(), (int64_t)0,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestBuffsApproxEqualContiguous, fail_first_element) {
    std::vector<double> actual = {99.0, 2.0, 3.0, 4.0};
    std::vector<double> expect = { 1.0, 2.0, 3.0, 4.0};
    auto msg = buffs_approx_equal(actual.data(), expect.data(), (int64_t)actual.size(),
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("index 0"), std::string::npos);
}

TEST_F(TestBuffsApproxEqualContiguous, fail_middle_element) {
    std::vector<double> actual = {1.0, 2.0, 99.0, 4.0};
    std::vector<double> expect = {1.0, 2.0,  3.0, 4.0};
    auto msg = buffs_approx_equal(actual.data(), expect.data(), (int64_t)actual.size(),
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("index 2"), std::string::npos);
}

TEST_F(TestBuffsApproxEqualContiguous, pass_custom_atol) {
    // values that fail default tolerances but pass with large atol
    std::vector<double> actual = {1.0, 1.0 + 1e-4};
    std::vector<double> expect = {1.0, 1.0};
    double large_atol = 1.0;
    auto msg = buffs_approx_equal(actual.data(), expect.data(), (int64_t)actual.size(),
        __PRETTY_FUNCTION__, __FILE__, __LINE__, large_atol);
    EXPECT_TRUE(msg.empty());
}


// MARK: buffs_approx_equal (strided)

class TestBuffsApproxEqualStrided : public ::testing::Test {};

TEST_F(TestBuffsApproxEqualStrided, pass_all_equal_stride1) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    auto msg = buffs_approx_equal((int64_t)3, a.data(), (int64_t)1, a.data(), (int64_t)1,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestBuffsApproxEqualStrided, pass_stride2_ignored_odd_elements) {
    // actual[0,2,4] == expect[0,2,4]; odd positions differ but are never read
    std::vector<double> actual = {1.0, 999.0, 2.0, 999.0, 3.0, 999.0};
    std::vector<double> expect = {1.0,   0.0, 2.0,   0.0, 3.0,   0.0};
    // size=3, inc_actual=2, inc_expect=2: accesses indices 0,2,4 in each
    auto msg = buffs_approx_equal((int64_t)3,
        actual.data(), (int64_t)2, expect.data(), (int64_t)2,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestBuffsApproxEqualStrided, fail_strided_element_mentions_logical_index) {
    // corrupt logical index 2 → actual[4] (with inc=2)
    std::vector<double> actual = {1.0, 0.0, 2.0, 0.0, 99.0, 0.0};
    std::vector<double> expect = {1.0, 0.0, 2.0, 0.0,  3.0, 0.0};
    auto msg = buffs_approx_equal((int64_t)3,
        actual.data(), (int64_t)2, expect.data(), (int64_t)2,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("index 2"), std::string::npos);
}

TEST_F(TestBuffsApproxEqualStrided, fail_different_inc_actual_and_inc_expect) {
    // actual has inc=2, expect has inc=1; logical index 1 mismatches
    std::vector<double> actual = {1.0, 0.0, 99.0};   // logical: 1.0, 99.0
    std::vector<double> expect = {1.0, 2.0};           // logical: 1.0, 2.0
    auto msg = buffs_approx_equal((int64_t)2,
        actual.data(), (int64_t)2, expect.data(), (int64_t)1,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("index 1"), std::string::npos);
}


// MARK: buffs_approx_equal (bounded)

class TestBuffsApproxEqualBounded : public ::testing::Test {};

TEST_F(TestBuffsApproxEqualBounded, pass_all_within_bounds) {
    std::vector<double> actual = {1.1, 2.2, 3.3};
    std::vector<double> expect = {1.0, 2.0, 3.0};
    std::vector<double> bounds = {0.2, 0.3, 0.4};   // all errors < bounds
    auto msg = buffs_approx_equal(actual.data(), expect.data(), bounds.data(),
        (int64_t)actual.size(), __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestBuffsApproxEqualBounded, fail_at_index_1) {
    std::vector<double> actual = {1.0, 99.0, 3.0};
    std::vector<double> expect = {1.0,  2.0, 3.0};
    std::vector<double> bounds = {0.5,  0.5, 0.5};   // index 1 error = 97, far exceeds bound
    auto msg = buffs_approx_equal(actual.data(), expect.data(), bounds.data(),
        (int64_t)actual.size(), __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("index 1"), std::string::npos);
}

TEST_F(TestBuffsApproxEqualBounded, pass_heterogeneous_bounds) {
    // each element has a different, tight bound; all satisfied
    std::vector<double> actual = {1.01, 2.001, 3.0001};
    std::vector<double> expect = {1.0,  2.0,   3.0};
    std::vector<double> bounds = {0.02, 0.002, 0.0002};
    auto msg = buffs_approx_equal(actual.data(), expect.data(), bounds.data(),
        (int64_t)actual.size(), __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}


// MARK: matrices_approx_equal

class TestMatricesApproxEqual : public ::testing::Test {
protected:
    // 2-by-3 matrices stored ColMajor (lda=2) or RowMajor (lda=3)
    static constexpr int64_t M = 2, N = 3;

    // Fill a ColMajor 2x3 matrix with values 1..6
    static std::vector<double> colmajor_2x3() {
        return {1.0, 2.0,   // col 0
                3.0, 4.0,   // col 1
                5.0, 6.0};  // col 2
    }

    // Fill a RowMajor 2x3 matrix with the same logical values 1..6
    static std::vector<double> rowmajor_2x3() {
        return {1.0, 3.0, 5.0,   // row 0
                2.0, 4.0, 6.0};  // row 1
    }
};

TEST_F(TestMatricesApproxEqual, pass_colmajor_notrans) {
    auto A = colmajor_2x3();
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::NoTrans,
        M, N, A.data(), M, A.data(), M,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, pass_rowmajor_notrans) {
    auto A = rowmajor_2x3();
    auto msg = matrices_approx_equal(
        blas::Layout::RowMajor, blas::Op::NoTrans,
        M, N, A.data(), N, A.data(), N,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, fail_colmajor_notrans_mentions_index) {
    auto A = colmajor_2x3();
    auto B = colmajor_2x3();
    // corrupt logical element (1, 2): ColMajor index = 1 + 2*2 = 5
    B[5] = 999.0;
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::NoTrans,
        M, N, A.data(), M, B.data(), M,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("(1, 2)"), std::string::npos);
}

TEST_F(TestMatricesApproxEqual, pass_colmajor_trans) {
    // A is M×N, B is N×M; A[i,j] == B[j,i] → should pass with Op::Trans
    auto A = colmajor_2x3();           // 2x3 ColMajor, lda=2
    std::vector<double> B = {          // 3x2 ColMajor (transpose), lda=3
        1.0, 3.0, 5.0,   // col 0 of B = row 0 of A
        2.0, 4.0, 6.0};  // col 1 of B = row 1 of A
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::Trans,
        M, N, A.data(), M, B.data(), N,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, fail_colmajor_trans) {
    auto A = colmajor_2x3();
    std::vector<double> B = {
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0};
    // corrupt B[1,0] (logical transpose position for A[0,1]):
    // B is N×M ColMajor with lda=N=3; B[1,0] = index 1
    B[1] = 999.0;
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::Trans,
        M, N, A.data(), M, B.data(), N,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, pass_mixed_layouts) {
    // layoutA=ColMajor, layoutB=RowMajor; both represent the same logical 2x3 matrix
    auto A = colmajor_2x3();
    auto B = rowmajor_2x3();
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Layout::RowMajor,
        blas::Op::NoTrans,
        M, N, A.data(), M, B.data(), N,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, fail_mixed_layouts) {
    auto A = colmajor_2x3();
    auto B = rowmajor_2x3();
    // corrupt logical element (0,1) in B (RowMajor): index = 0*N + 1 = 1
    B[1] = 999.0;
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Layout::RowMajor,
        blas::Op::NoTrans,
        M, N, A.data(), M, B.data(), N,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, pass_single_layout_delegates) {
    auto A = colmajor_2x3();
    // single-layout overload should delegate correctly
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::NoTrans,
        M, N, A.data(), M, A.data(), M,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_TRUE(msg.empty());
}

TEST_F(TestMatricesApproxEqual, fail_single_layout_delegates) {
    auto A = colmajor_2x3();
    auto B = colmajor_2x3();
    B[0] = 999.0;
    auto msg = matrices_approx_equal(
        blas::Layout::ColMajor, blas::Op::NoTrans,
        M, N, A.data(), M, B.data(), M,
        __PRETTY_FUNCTION__, __FILE__, __LINE__);
    EXPECT_FALSE(msg.empty());
}
