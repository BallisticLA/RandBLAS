


#include "RandBLAS/base.hh"
#include "RandBLAS/config.h"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_skops.hh"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

class TestExceptions : public ::testing::Test {
    protected:
};

TEST_F(TestExceptions, randblas_require_var_arg) {
    bool successful_raise = false;
    try {
        randblas_require(successful_raise);
    } catch (RandBLAS::Error &e) {
        std::string message{e.what()};
        successful_raise = message.find("successful_raise") != std::string::npos;
    }
    ASSERT_TRUE(successful_raise);
}

TEST_F(TestExceptions, randblas_require_expr_arg) {
    int flag = 0;
    try {
        randblas_require(flag > 1);
    } catch (RandBLAS::Error &e) {
        std::string message{e.what()};
        flag = message.find("flag > 1") != std::string::npos;
    }
    ASSERT_TRUE(flag);
}

TEST_F(TestExceptions, randblas_error_if_msg_output) {
    bool error_trigger = true;
    bool expect_true = false;
    try {
        randblas_error_if_msg(error_trigger, "Custom message.");
    } catch (RandBLAS::Error &e) {
        std::string message{e.what()};
        expect_true = message.find("Custom message.") != std::string::npos;
    }
    ASSERT_TRUE(expect_true);
}

TEST_F(TestExceptions, safe_int_product_multiplies_in_output_type) {
    constexpr int32_t max_int32 = std::numeric_limits<int32_t>::max();
    constexpr int64_t expected = 2 * static_cast<int64_t>(max_int32);

    EXPECT_EQ((RandBLAS::safe_int_product<int32_t, int64_t>(max_int32, 2)), expected);
}

TEST_F(TestExceptions, safe_int_product_accepts_signed_boundary_products) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    constexpr int64_t min_int64 = std::numeric_limits<int64_t>::min();

    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(min_int64, 1), min_int64);
    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(max_int64, -1), -max_int64);
    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(-2, -3), 6);
}

TEST_F(TestExceptions, safe_int_product_rejects_output_type_overflow) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    constexpr int64_t min_int64 = std::numeric_limits<int64_t>::min();

    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(max_int64, 2), std::overflow_error);
    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(min_int64, -1), std::overflow_error);
    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(-1, min_int64), std::overflow_error);
}

TEST_F(TestExceptions, sparse_dist_rejects_full_nnz_overflow) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();

    EXPECT_THROW(
        RandBLAS::SparseDist(max_int64, 2, 2, RandBLAS::Axis::Short), std::overflow_error
    );
}

TEST_F(TestExceptions, fill_dense_rejects_allocation_size_overflow) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    RandBLAS::DenseDist dist(max_int64, 2, RandBLAS::ScalarDist::Gaussian, RandBLAS::Axis::Short);
    RandBLAS::RNGState state(0);
    RandBLAS::DenseSkOp<double> S(dist, state);

    EXPECT_THROW(RandBLAS::fill_dense(S), std::overflow_error);
}

TEST_F(TestExceptions, dense_submatrix_rejects_allocation_size_overflow) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    RandBLAS::DenseDist dist(max_int64, 2, RandBLAS::ScalarDist::Gaussian, RandBLAS::Axis::Short);
    RandBLAS::RNGState state(0);
    RandBLAS::DenseSkOp<double> S(dist, state);
    using BFO = RandBLAS::BLASFriendlyOperator<double>;

    EXPECT_THROW(
        RandBLAS::submatrix_as_blackbox<BFO>(S, max_int64, 2, 0, 0), std::overflow_error
    );
}

TEST_F(TestExceptions, fill_dense_rejects_starting_offset_overflow) {
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    RandBLAS::DenseDist dist(2, max_int64, RandBLAS::ScalarDist::Gaussian, RandBLAS::Axis::Long);
    RandBLAS::RNGState state(0);
    double buff;

    EXPECT_THROW(
        RandBLAS::fill_dense_unpacked(
            blas::Layout::RowMajor, dist, 1, 1, 1, 1, &buff, state
        ),
        std::overflow_error
    );
}
