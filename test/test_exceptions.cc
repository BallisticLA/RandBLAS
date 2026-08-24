


#include "RandBLAS/base.hh"
#include "RandBLAS/config.h"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_skops.hh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

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
    // Twice INT32_MAX cannot be represented by an int32_t, but it fits in the
    // requested int64_t output. This catches implementations that multiply in
    // the input type and cast only after the intermediate value has overflowed.
    constexpr int32_t max_int32 = std::numeric_limits<int32_t>::max();
    constexpr int64_t expected = 2 * static_cast<int64_t>(max_int32);

    EXPECT_EQ((RandBLAS::safe_int_product<int32_t, int64_t>(max_int32, 2)), expected);
}

TEST_F(TestExceptions, safe_int_product_accepts_signed_boundary_products) {
    // Overflow checks must leave every representable product alone. Exercise
    // both ends of the int64_t range, negative products, and a product of two
    // negative operands. These lie near cases where signed division requires
    // special handling.
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    constexpr int64_t min_int64 = std::numeric_limits<int64_t>::min();

    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(min_int64, 1), min_int64);
    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(max_int64, -1), -max_int64);
    EXPECT_EQ(RandBLAS::safe_int_product<int64_t>(-2, -3), 6);
}

TEST_F(TestExceptions, safe_int_product_rejects_output_type_overflow) {
    // These products all lie outside the int64_t range. The two INT64_MIN * -1
    // cases also check that overflow is detected regardless of which operand
    // contains the exceptional minimum value.
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    constexpr int64_t min_int64 = std::numeric_limits<int64_t>::min();

    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(max_int64, 2), std::overflow_error);
    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(min_int64, -1), std::overflow_error);
    EXPECT_THROW(RandBLAS::safe_int_product<int64_t>(-1, min_int64), std::overflow_error);
}

TEST_F(TestExceptions, sparse_dist_rejects_full_nnz_overflow) {
    // This is otherwise a valid short-axis distribution: each of its
    // INT64_MAX major-axis vectors would have two nonzeros. Constructing it
    // would therefore require full_nnz = 2 * INT64_MAX, so SparseDist should
    // report the overflow before storing a wrapped buffer length.
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();

    EXPECT_THROW(
        RandBLAS::SparseDist(max_int64, 2, 2, RandBLAS::Axis::Short), std::overflow_error
    );
}

TEST_F(TestExceptions, fill_dense_rejects_allocation_size_overflow) {
    // Choosing the short axis as the major axis keeps the RNG-state increment
    // representable, so constructing the operator S succeeds. Materializing S
    // would still require 2 * INT64_MAX entries. fill_dense should report that
    // allocation count as an overflow rather than passing a wrapped size to
    // new[].
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    RandBLAS::DenseDist dist(max_int64, 2, RandBLAS::ScalarDist::Gaussian, RandBLAS::Axis::Short);
    RandBLAS::RNGState state(0);
    RandBLAS::DenseSkOp<double> S(dist, state);

    EXPECT_THROW(RandBLAS::fill_dense(S), std::overflow_error);
}

TEST_F(TestExceptions, dense_submatrix_rejects_allocation_size_overflow) {
    // submatrix_as_blackbox owns the buffer it creates. Request the full
    // INT64_MAX-by-2 window from a distribution with valid individual
    // dimensions and verify that the helper rejects the overflowing element
    // count before it attempts the allocation.
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
    // In this row-major 2-by-INT64_MAX matrix, the 1-by-1 window beginning at
    // (1, 1) has linear offset INT64_MAX + 1. The output buffer itself has only
    // one entry, so this isolates overflow in the parent-matrix offset and
    // checks that fill_dense_unpacked rejects it before writing to the buffer.
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

TEST_F(TestExceptions, validate_submat_dims_accepts_boundary_windows) {
    // A submatrix may occupy its entire parent or may be empty. In particular,
    // an empty window may begin at the parent's lower-right boundary because it
    // does not address any entries beyond that boundary.
    EXPECT_NO_THROW(RandBLAS::validate_submat_dims(5, 7, 5, 7, 0, 0));
    EXPECT_NO_THROW(RandBLAS::validate_submat_dims(5, 7, 2, 3, 3, 4));
    EXPECT_NO_THROW(RandBLAS::validate_submat_dims(5, 7, 0, 0, 5, 7));
}

TEST_F(TestExceptions, validate_submat_dims_rejects_invalid_windows) {
    // The subtraction-form bounds check should reject malformed windows without
    // first adding an extent and offset, which could overflow and make an invalid
    // request appear valid. Cover each kind of bad dimension as well as INT64_MAX.
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();

    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, -1, 1, 0, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 1, -1, 0, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 1, 1, -1, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 1, 1, 0, -1), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 6, 1, 0, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 1, 8, 0, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 2, 3, 4, 0), RandBLAS::Error);
    EXPECT_THROW(RandBLAS::validate_submat_dims(5, 7, 2, 3, 0, 5), RandBLAS::Error);
    EXPECT_THROW(
        RandBLAS::validate_submat_dims(5, 7, 1, 1, max_int64, max_int64),
        RandBLAS::Error
    );
}

TEST_F(TestExceptions, thread_number_helpers_report_serial_context) {
    // Outside an OpenMP parallel region, every caller is the sole member of a
    // one-thread team. These values are also the complete fallback contract for
    // builds where RandBLAS was compiled without OpenMP.
    EXPECT_EQ(RandBLAS::randblas_get_thread_num(), 0);
    EXPECT_EQ(RandBLAS::randblas_get_num_threads(), 1);
}

#if defined(RandBLAS_HAS_OpenMP)
TEST_F(TestExceptions, thread_number_helpers_report_openmp_team) {
    // Give each physical OpenMP thread its own array slot, then compare the
    // RandBLAS wrappers with OpenMP's direct answers. This checks both the
    // thread's zero-based identity and the size of the team it belongs to.
    const int orig_dynamic = omp_get_dynamic();
    const int orig_max_threads = omp_get_max_threads();
    const int requested_threads = std::min(4, orig_max_threads);
    std::vector<int> thread_nums(requested_threads, -1);
    std::vector<int> team_sizes(requested_threads, -1);
    int actual_threads = 0;

    omp_set_dynamic(0);
    #pragma omp parallel num_threads(requested_threads)
    {
        const int thread_num = omp_get_thread_num();
        thread_nums[thread_num] = RandBLAS::randblas_get_thread_num();
        team_sizes[thread_num] = RandBLAS::randblas_get_num_threads();
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    omp_set_num_threads(orig_max_threads);
    omp_set_dynamic(orig_dynamic);

    for (int thread_num = 0; thread_num < actual_threads; ++thread_num) {
        EXPECT_EQ(thread_nums[thread_num], thread_num);
        EXPECT_EQ(team_sizes[thread_num], actual_threads);
    }
}
#endif
