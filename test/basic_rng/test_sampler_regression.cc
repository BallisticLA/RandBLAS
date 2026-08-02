// Copyright, 2026. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <RandBLAS/dense_skops.hh>
#include <RandBLAS/sparse_skops.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using State = RandBLAS::RNGState<>;

constexpr std::uint64_t seed = 0x0123456789abcdefULL;

void expect_state(State const& actual, std::uint32_t counter_word_zero) {
    EXPECT_EQ(actual.counter()[0], counter_word_zero);
    EXPECT_EQ(actual.counter()[1], 0u);
    EXPECT_EQ(actual.counter()[2], 0u);
    EXPECT_EQ(actual.counter()[3], 0u);
    EXPECT_EQ(actual.key()[0], 0x89abcdefu);
    EXPECT_EQ(actual.key()[1], 0x01234567u);
}

template <std::size_t N>
void expect_float_bits(
    std::vector<float> const& actual,
    std::array<std::uint32_t, N> const& expected
) {
    ASSERT_EQ(actual.size(), N);
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), expected[i])
            << "mismatch at index " << i;
    }
}

template <std::size_t N>
void expect_gaussian_values(
    std::vector<float> const& actual,
    std::array<std::uint32_t, N> const& expected_bits
) {
    ASSERT_EQ(actual.size(), N);
    constexpr float eps_scale = 32 * std::numeric_limits<float>::epsilon();
    for (std::size_t i = 0; i < N; ++i) {
        float expected = std::bit_cast<float>(expected_bits[i]);
        float tolerance = eps_scale * std::max(1.0f, std::abs(expected));
        EXPECT_NEAR(actual[i], expected, tolerance) << "mismatch at index " << i;
    }
}

template <std::size_t N>
void expect_sparse_case(
    RandBLAS::Axis axis,
    std::int64_t vec_nnz,
    std::array<std::int64_t, N> const& expected_rows,
    std::array<std::int64_t, N> const& expected_cols,
    std::array<std::uint32_t, N> const& expected_value_bits,
    std::uint32_t expected_counter
) {
    State initial{seed};
    RandBLAS::SparseDist dist{5, 11, vec_nnz, axis};
    std::vector<float> values(dist.full_nnz);
    std::vector<std::int64_t> rows(dist.full_nnz);
    std::vector<std::int64_t> cols(dist.full_nnz);
    std::int64_t nnz = 0;

    auto next = RandBLAS::fill_sparse_unpacked(
        dist, dist.n_rows, dist.n_cols, 0, 0, nnz,
        values.data(), rows.data(), cols.data(), initial
    );

    ASSERT_EQ(nnz, static_cast<std::int64_t>(N));
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_EQ(rows[i], expected_rows[i]) << "row mismatch at index " << i;
        EXPECT_EQ(cols[i], expected_cols[i]) << "column mismatch at index " << i;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(values[i]), expected_value_bits[i])
            << "value mismatch at index " << i;
    }
    expect_state(next, expected_counter);
}

TEST(SamplerRegression, DenseUniformDefaultStream) {
    State initial{seed};
    RandBLAS::DenseDist dist{3, 7, RandBLAS::ScalarDist::Uniform};
    std::vector<float> values(21);

    auto next = RandBLAS::fill_dense(dist, values.data(), initial);

    constexpr std::array expected{
        0xbf7854bbu, 0xbf4a7a6eu, 0x3e8f19beu, 0x3fd435c6u,
        0xbf8e649au, 0x3f8e72b0u, 0x3faf5ee4u, 0xbde318f9u,
        0xbfd9d3bcu, 0x3f8a747fu, 0x3eb80719u, 0xbf912a10u,
        0x3fdd4273u, 0x3de59866u, 0xbfa5fdceu, 0xbf1f001eu,
        0xbf877332u, 0xbf8fac99u, 0x3f9760a8u, 0xbf22c0e8u,
        0xbf47b7a7u
    };
    expect_float_bits(values, expected);
    expect_state(next, 6u);
}

TEST(SamplerRegression, DenseGaussianDefaultStream) {
    State initial{seed};
    RandBLAS::DenseDist dist{3, 7, RandBLAS::ScalarDist::Gaussian};
    std::vector<float> values(21);

    auto next = RandBLAS::fill_dense(dist, values.data(), initial);

    constexpr std::array expected{
        0xbf350b89u, 0xbe0a45ffu, 0x3f16e3a3u, 0x3f87d977u,
        0xbfadf224u, 0xbf26bf29u, 0x3f0497a1u, 0xbe6dd4e4u,
        0x3f91d018u, 0x3ffbe4e1u, 0xbf4fc3d1u, 0xbf856f0eu,
        0xbf0d0d9cu, 0x3ccb308du, 0xbee48253u, 0xbee2ab01u,
        0xbf54ee6bu, 0xbe9ac357u, 0x3f08df34u, 0xbeb11d6du,
        0xbed820b3u
    };
    expect_gaussian_values(values, expected);
    expect_state(next, 6u);
}

TEST(SamplerRegression, SparseShortAxisDefaultStream) {
    constexpr std::array<std::int64_t, 33> rows{
        0, 2, 3, 0, 2, 3, 0, 1, 4, 1, 3, 4, 2, 3, 4, 0, 2,
        4, 1, 2, 3, 0, 1, 2, 0, 1, 3, 0, 2, 4, 0, 3, 4
    };
    constexpr std::array<std::int64_t, 33> cols{
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
        5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10
    };
    constexpr std::array<std::uint32_t, 33> bits{
        0x3f800000u, 0x3f800000u, 0xbf800000u, 0x3f800000u,
        0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0xbf800000u, 0xbf800000u, 0x3f800000u, 0xbf800000u,
        0xbf800000u, 0x3f800000u, 0xbf800000u, 0xbf800000u,
        0xbf800000u, 0xbf800000u, 0x3f800000u, 0xbf800000u,
        0xbf800000u, 0xbf800000u, 0xbf800000u, 0x3f800000u,
        0xbf800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0xbf800000u, 0x3f800000u, 0xbf800000u, 0x3f800000u,
        0xbf800000u
    };
    expect_sparse_case(RandBLAS::Axis::Short, 3, rows, cols, bits, 33u);
}

TEST(SamplerRegression, SparseLongAxisDefaultStream) {
    constexpr std::array<std::int64_t, 13> rows{
        0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4
    };
    constexpr std::array<std::int64_t, 13> cols{
        2, 10, 0, 3, 8, 2, 4, 9, 0, 5, 1, 5, 9
    };
    constexpr std::array<std::uint32_t, 13> bits{
        0x3fb504f3u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
        0x3f800000u, 0xbf800000u, 0x3f800000u, 0x3f800000u,
        0xbfb504f3u, 0x3f800000u, 0xbf800000u, 0x3f800000u,
        0xbf800000u
    };
    expect_sparse_case(RandBLAS::Axis::Long, 3, rows, cols, bits, 15u);
}

TEST(SamplerRegression, SparseOneNonzeroDefaultStream) {
    constexpr std::array<std::int64_t, 11> rows{2, 2, 0, 0, 1, 2, 4, 1, 2, 4, 1};
    constexpr std::array<std::int64_t, 11> cols{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    constexpr std::array<std::uint32_t, 11> bits{
        0x3f800000u, 0xbf800000u, 0x3f800000u, 0x3f800000u,
        0x3f800000u, 0x3f800000u, 0xbf800000u, 0x3f800000u,
        0x3f800000u, 0xbf800000u, 0xbf800000u
    };
    expect_sparse_case(RandBLAS::Axis::Short, 1, rows, cols, bits, 11u);
}

} // namespace
