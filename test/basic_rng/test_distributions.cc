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

#include <RandBLAS/rng/distributions.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

template <class Word, std::size_t N>
struct FixedState {
    using res_t = std::array<Word, N>;

    res_t values{};
    std::uint64_t blocks{};

    constexpr void generate(res_t& output) const noexcept {
        output = values;
    }

    constexpr void advance(std::uint64_t amount) noexcept {
        blocks += amount;
    }
};

static_assert(RandBLAS::GeneratorState<FixedState<std::uint32_t, 4>>);

template <class State>
concept CanGenerateNormals = requires(State const& state) {
    RandBLAS::rng::boxmul::generate(state);
};

template <class Real>
void expect_near_reference(Real actual, Real expected) {
    auto scale = std::max(Real{1}, std::abs(expected));
    auto tolerance = Real{8} * std::numeric_limits<Real>::epsilon() * scale;
    EXPECT_NEAR(actual, expected, tolerance);
}

TEST(DistributionConversion, U01MatchesRetainedReferenceValues) {
    using RandBLAS::rng::u01;

    EXPECT_EQ(u01<float>(UINT32_C(0)), 0x1p-33f);
    EXPECT_EQ(u01<float>(UINT32_C(1)), 0x1.8p-32f);
    EXPECT_EQ(u01<float>(UINT32_C(0x80000000)), 0x1p-1f);
    EXPECT_EQ(u01<float>(UINT32_MAX), 0x1p+0f);
    EXPECT_EQ(u01<float>(UINT32_C(0x243f6a88)), 0x1.21fb54p-3f);

    EXPECT_EQ(u01<double>(UINT32_C(0)), 0x1p-33);
    EXPECT_EQ(u01<double>(UINT32_C(1)), 0x1.8p-32);
    EXPECT_EQ(u01<double>(UINT32_C(0x80000000)), 0x1.00000001p-1);
    EXPECT_EQ(u01<double>(UINT32_MAX), 0x1.ffffffffp-1);
    EXPECT_EQ(u01<double>(UINT32_C(0x243f6a88)), 0x1.21fb5444p-3);

    EXPECT_EQ(u01<double>(UINT64_C(0)), 0x1p-65);
    EXPECT_EQ(u01<double>(UINT64_C(1)), 0x1.8p-64);
    EXPECT_EQ(u01<double>(UINT64_C(0x8000000000000000)), 0x1p-1);
    EXPECT_EQ(u01<double>(UINT64_MAX), 0x1p+0);
    EXPECT_EQ(u01<double>(UINT64_C(0x243f6a8885a308d3)),
              0x1.21fb54442d184p-3);
}

TEST(DistributionConversion, U01HasTheRetainedEndpointConvention) {
    EXPECT_GT(RandBLAS::rng::u01<float>(UINT32_C(0)), 0.0f);
    EXPECT_LE(RandBLAS::rng::u01<float>(UINT32_MAX), 1.0f);
    EXPECT_EQ(RandBLAS::rng::u01<float>(UINT32_MAX), 1.0f);

    EXPECT_GT(RandBLAS::rng::u01<double>(UINT32_C(0)), 0.0);
    EXPECT_LT(RandBLAS::rng::u01<double>(UINT32_MAX), 1.0);
    EXPECT_GT(RandBLAS::rng::u01<double>(UINT64_C(0)), 0.0);
    EXPECT_EQ(RandBLAS::rng::u01<double>(UINT64_MAX), 1.0);
}

TEST(DistributionConversion, Uneg11MatchesRetainedReferenceValues) {
    using Policy = RandBLAS::rng::uneg11;

    EXPECT_EQ(Policy::convert<float>(UINT32_C(0)), 0x1p-32f);
    EXPECT_EQ(Policy::convert<float>(UINT32_C(1)), 0x1.8p-31f);
    EXPECT_EQ(Policy::convert<float>(UINT32_C(0x80000000)), -0x1p+0f);
    EXPECT_EQ(Policy::convert<float>(UINT32_MAX), -0x1p-32f);
    EXPECT_EQ(Policy::convert<float>(UINT32_C(0x243f6a88)),
              0x1.21fb54p-2f);

    EXPECT_EQ(Policy::convert<double>(UINT32_C(0)), 0x1p-32);
    EXPECT_EQ(Policy::convert<double>(UINT32_C(1)), 0x1.8p-31);
    EXPECT_EQ(Policy::convert<double>(UINT32_C(0x80000000)),
              -0x1.fffffffep-1);
    EXPECT_EQ(Policy::convert<double>(UINT32_MAX), -0x1p-32);
    EXPECT_EQ(Policy::convert<double>(UINT32_C(0x243f6a88)),
              0x1.21fb5444p-2);

    EXPECT_EQ(Policy::convert<double>(UINT64_C(0)), 0x1p-64);
    EXPECT_EQ(Policy::convert<double>(UINT64_C(1)), 0x1.8p-63);
    EXPECT_EQ(Policy::convert<double>(UINT64_C(0x8000000000000000)),
              -0x1p+0);
    EXPECT_EQ(Policy::convert<double>(UINT64_MAX), -0x1p-64);
    EXPECT_EQ(Policy::convert<double>(UINT64_C(0x243f6a8885a308d3)),
              0x1.21fb54442d184p-2);
}

TEST(DistributionConversion, Uneg11IsClosedAndNeverZero) {
    std::array<std::uint32_t, 5> inputs{
        0, 1, UINT32_C(0x7fffffff), UINT32_C(0x80000000), UINT32_MAX};
    for (auto input : inputs) {
        auto value = RandBLAS::rng::uneg11::convert<float>(input);
        EXPECT_GE(value, -1.0f);
        EXPECT_LE(value, 1.0f);
        EXPECT_NE(value, 0.0f);
    }
}

TEST(DistributionConversion, BoxMullerMatchesRetainedReferences) {
    auto result32 = RandBLAS::rng::boxmuller(
        UINT32_C(0x243f6a88), UINT32_C(0x85a308d3));
    expect_near_reference(result32[0], 0x1.c5857cp-1f);
    expect_near_reference(result32[1], 0x1.6f9a8ep-1f);

    auto result64 = RandBLAS::rng::boxmuller(
        UINT64_C(0x243f6a8885a308d3),
        UINT64_C(0x13198a2e03707344));
    expect_near_reference(result64[0], 0x1.c51c6804651e6p+0);
    expect_near_reference(result64[1], 0x1.6f4563170165cp+0);
}

TEST(DistributionConversion, BoxMullerUsesAngleThenRadiusAndReturnsSinThenCos) {
    constexpr std::uint32_t angle_word = UINT32_C(0x243f6a88);
    constexpr std::uint32_t radius_word = UINT32_C(0x85a308d3);
    constexpr float pi = 3.1415926535897932f;
    auto angle = pi * RandBLAS::rng::uneg11::convert<float>(angle_word);
    auto radius = std::sqrt(-2.0f *
                            std::log(RandBLAS::rng::u01<float>(radius_word)));
    auto result = RandBLAS::rng::boxmuller(angle_word, radius_word);

    expect_near_reference(result[0], std::sin(angle) * radius);
    expect_near_reference(result[1], std::cos(angle) * radius);
}

TEST(DistributionPolicy, GeneratesOneFixedBlockWithoutAdvancingState) {
    FixedState<std::uint32_t, 4> state{{
        UINT32_C(0x243f6a88), UINT32_C(0x85a308d3),
        UINT32_C(0x13198a2e), UINT32_C(0x03707344)}};
    typename decltype(state)::res_t bits{};
    state.generate(bits);
    auto state_before = state.values;
    auto uniform = RandBLAS::rng::uneg11::generate(state);
    auto normal = RandBLAS::rng::boxmul::generate(state);

    EXPECT_EQ(state.values, state_before);
    EXPECT_EQ(uniform.size(), bits.size());
    EXPECT_EQ(normal.size(), bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        EXPECT_EQ(uniform[i],
                  RandBLAS::rng::uneg11::convert<float>(bits[i]));
    }
    for (std::size_t i = 0; i < bits.size(); i += 2) {
        auto pair = RandBLAS::rng::boxmuller(bits[i], bits[i + 1]);
        EXPECT_EQ(normal[i], pair[0]);
        EXPECT_EQ(normal[i + 1], pair[1]);
    }
}

TEST(DistributionPolicy, RejectsOddNormalBlockLengths) {
    static_assert(CanGenerateNormals<FixedState<std::uint32_t, 4>>);
    static_assert(!CanGenerateNormals<FixedState<std::uint32_t, 3>>);
    SUCCEED();
}

} // namespace
