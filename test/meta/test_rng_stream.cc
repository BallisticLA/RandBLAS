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

#include <RandBLAS/testing/rng.hh>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

struct SequenceState {
    using res_t = std::array<std::uint32_t, 2>;

    res_t first{};
    std::uint64_t block{};

    void generate(res_t& output) const {
        output = {
            static_cast<std::uint32_t>(first[0] + 2 * block),
            static_cast<std::uint32_t>(first[1] + 2 * block)
        };
    }

    void advance(std::uint64_t blocks) { block += blocks; }
};

static_assert(RandBLAS::GeneratorState<SequenceState>);

using RNGStream = RandBLAS::testing::detail::RNGStream<SequenceState>;

TEST(RNGStream, NextWordBuffersOneBlockAndAdvancesOnRefill) {
    RNGStream stream(SequenceState{{10, 20}});

    EXPECT_EQ(stream.get_state().block, 0u);
    EXPECT_EQ(stream.next_word(), 10u);
    EXPECT_EQ(stream.get_state().block, 1u);
    EXPECT_EQ(stream.next_word(), 20u);
    EXPECT_EQ(stream.get_state().block, 1u);
    EXPECT_EQ(stream.next_word(), 12u);
    EXPECT_EQ(stream.get_state().block, 2u);
}

TEST(RNGStream, GaussianCachesTheSecondValue) {
    constexpr std::uint32_t angle_word = UINT32_C(0x243f6a88);
    constexpr std::uint32_t radius_word = UINT32_C(0x85a308d3);
    RNGStream stream(SequenceState{{angle_word, radius_word}});
    auto expected = RandBLAS::rng::boxmuller(angle_word, radius_word);

    EXPECT_EQ(stream.gaussian<float>(), expected[0]);
    EXPECT_EQ(stream.get_state().block, 1u);
    EXPECT_EQ(stream.gaussian<float>(), expected[1]);
    EXPECT_EQ(stream.get_state().block, 1u);
}

TEST(RNGStream, UniformAndGeometricUseScalarConversions) {
    constexpr std::uint32_t word = UINT32_C(0x243f6a88);
    RNGStream uniform_stream(SequenceState{{word, 0}});
    EXPECT_EQ(uniform_stream.uniform_01(),
              RandBLAS::rng::u01<double>(word));

    RNGStream geometric_stream(SequenceState{{word, 0}});
    double log_1_minus_p = std::log(0.75);
    double u = RandBLAS::rng::u01<double>(word);
    auto expected = static_cast<std::int64_t>(
        std::floor(std::log(1.0 - u) / log_1_minus_p));
    EXPECT_EQ(geometric_stream.geometric(log_1_minus_p), expected);
}

} // namespace
