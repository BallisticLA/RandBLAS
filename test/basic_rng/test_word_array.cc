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

#include <RandBLAS/rng/word_array.hh>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

using Array32x4 = RandBLAS::rng::WordArray<std::uint32_t, 4>;
using Array64x2 = RandBLAS::rng::WordArray<std::uint64_t, 2>;

static_assert(std::is_trivially_copyable_v<Array32x4>);
static_assert(Array32x4::static_size == 4);
static_assert(Array64x2::static_size == 2);

TEST(WordArray, ValueInitializesAndSupportsIndexedObservation) {
    Array32x4 value{};

    EXPECT_EQ(value.size(), 4u);
    for (std::size_t i = 0; i < value.size(); ++i) {
        EXPECT_EQ(value[i], 0u);
    }

    value[2] = 17u;
    Array32x4 copy = value;
    EXPECT_EQ(copy, value);
    EXPECT_EQ(copy[2], 17u);
}

TEST(WordArray, ZeroAdvanceDoesNotChangeValue) {
    Array32x4 value{{1u, 2u, 3u, 4u}};
    auto expected = value;

    value.advance(0);

    EXPECT_EQ(value, expected);
}

TEST(WordArray, AdvancesWithCarryFromLeastSignificantWord) {
    Array32x4 value{{0xffffffffu, 7u, 9u, 11u}};

    value.advance(2);

    EXPECT_EQ(value, (Array32x4{{1u, 8u, 9u, 11u}}));
}

TEST(WordArray, PropagatesCarryThroughMultipleWords) {
    Array32x4 value{{0xffffffffu, 0xffffffffu, 7u, 9u}};

    value.advance(1);

    EXPECT_EQ(value, (Array32x4{{0u, 0u, 8u, 9u}}));
}

TEST(WordArray, AddsAllBitsOfUint64To32BitWords) {
    Array32x4 value{};

    value.advance(0x0000000100000001ULL);

    EXPECT_EQ(value, (Array32x4{{1u, 1u, 0u, 0u}}));
}

TEST(WordArray, CombinesAmountWordsWithExistingCarry) {
    Array32x4 value{{0xffffffffu, 0u, 0u, 0u}};

    value.advance(0x0000000100000001ULL);

    EXPECT_EQ(value, (Array32x4{{0u, 2u, 0u, 0u}}));
}

TEST(WordArray, WrapsAtFullWidth) {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    Array32x4 value{{max, max, max, max}};

    value.advance(1);

    EXPECT_EQ(value, Array32x4{});
}

TEST(WordArray, Advances64BitWordsWithCarry) {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    Array64x2 value{{max, 5u}};

    value.advance(2);

    EXPECT_EQ(value, (Array64x2{{1u, 6u}}));
}

TEST(WordArray, Wraps64BitWordsAtFullWidth) {
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    Array64x2 value{{max, max}};

    value.advance(1);

    EXPECT_EQ(value, Array64x2{});
}

} // namespace
