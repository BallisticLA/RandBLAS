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

#include <RandBLAS/rng/philox.hh>
#include <RandBLAS/rng/repacked_output.hh>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

struct FixedEngine {
    using ctr_t = RandBLAS::rng::WordArray<std::uint32_t, 2>;
    using key_t = RandBLAS::rng::WordArray<std::uint32_t, 1>;
    using res_t = std::array<std::uint32_t, 2>;

    static constexpr key_t make_key(std::uint64_t seed) noexcept {
        key_t key{};
        key.advance(seed);
        return key;
    }

    constexpr void generate(ctr_t const&, key_t const&,
                            res_t& output) const noexcept {
        output = {0xaabbccddu, 0x01234567u};
    }
};

struct EngineWithoutMakeKey {
    using ctr_t = FixedEngine::ctr_t;
    using key_t = FixedEngine::key_t;
    using res_t = FixedEngine::res_t;

    constexpr void generate(ctr_t const&, key_t const&,
                            res_t& output) const noexcept {
        output = {0xaabbccddu, 0x01234567u};
    }
};

template <class Engine, class OutputWord>
concept CanRepack = requires {
    typename RandBLAS::rng::RepackedOutput<Engine, OutputWord>;
};

template <class Engine>
concept HasMakeKey = requires(std::uint64_t seed) {
    { Engine::make_key(seed) } -> std::same_as<typename Engine::key_t>;
};

TEST(RepackedOutput, SplitsEachSourceWordLeastSignificantChunkFirst) {
    FixedEngine::ctr_t counter{};
    FixedEngine::key_t key{};

    using Engine16 = RandBLAS::rng::RepackedOutput<FixedEngine, std::uint16_t>;
    Engine16::res_t out16{};
    Engine16{}.generate(counter, key, out16);
    EXPECT_EQ(out16, (std::array<std::uint16_t, 4>{
                         0xccddu, 0xaabbu, 0x4567u, 0x0123u}));

    using Engine8 = RandBLAS::rng::RepackedOutput<FixedEngine, std::uint8_t>;
    Engine8::res_t out8{};
    Engine8{}.generate(counter, key, out8);
    EXPECT_EQ(out8, (std::array<std::uint8_t, 8>{
                       0xddu, 0xccu, 0xbbu, 0xaau,
                       0x67u, 0x45u, 0x23u, 0x01u}));
}

TEST(RepackedOutput, RepackingPhiloxHasExactDirectAndNestedResults) {
    using Base = RandBLAS::rng::Philox<4, 32, 10>;
    using Direct16 = RandBLAS::rng::RepackedOutput<Base, std::uint16_t>;
    using Direct8 = RandBLAS::rng::RepackedOutput<Base, std::uint8_t>;
    using Nested8 = RandBLAS::rng::RepackedOutput<Direct16, std::uint8_t>;

    Base::ctr_t counter{};
    Base::key_t key{};
    Direct16::res_t out16{};
    Direct8::res_t out8{};
    Nested8::res_t nested8{};
    Direct16{}.generate(counter, key, out16);
    Direct8{}.generate(counter, key, out8);
    Nested8{}.generate(counter, key, nested8);

    EXPECT_EQ(out16, (Direct16::res_t{
                         0xe8d5u, 0x6627u, 0xc58du, 0xe169u,
                         0xac4cu, 0xbc57u, 0xdbd8u, 0x9b00u}));
    EXPECT_EQ(out8, (Direct8::res_t{
                        0xd5u, 0xe8u, 0x27u, 0x66u,
                        0x8du, 0xc5u, 0x69u, 0xe1u,
                        0x4cu, 0xacu, 0x57u, 0xbcu,
                        0xd8u, 0xdbu, 0x00u, 0x9bu}));
    EXPECT_EQ(nested8, out8);
}

TEST(RepackedOutput, PreservesBlockBitsCounterAndKeyTypes) {
    using Base = RandBLAS::rng::Philox<4, 32, 10>;
    using Repacked = RandBLAS::rng::RepackedOutput<Base, std::uint8_t>;

    static_assert(std::same_as<Repacked::ctr_t, Base::ctr_t>);
    static_assert(std::same_as<Repacked::key_t, Base::key_t>);
    static_assert(std::tuple_size_v<Repacked::res_t> *
                      std::numeric_limits<Repacked::res_t::value_type>::digits ==
                  std::tuple_size_v<Base::res_t> *
                      std::numeric_limits<Base::res_t::value_type>::digits);
    SUCCEED();
}

TEST(RepackedOutput, EqualWidthIsAnIdentityAdaptor) {
    using Identity =
        RandBLAS::rng::RepackedOutput<FixedEngine, std::uint32_t>;
    FixedEngine::ctr_t counter{};
    FixedEngine::key_t key{};
    Identity::res_t output{};

    Identity{}.generate(counter, key, output);

    EXPECT_EQ(output, (Identity::res_t{0xaabbccddu, 0x01234567u}));
}

TEST(RepackedOutput, ForwardsMakeKeyOnlyWhenTheWrappedEngineHasIt) {
    using With = RandBLAS::rng::RepackedOutput<FixedEngine, std::uint16_t>;
    using Without =
        RandBLAS::rng::RepackedOutput<EngineWithoutMakeKey, std::uint16_t>;
    static_assert(HasMakeKey<With>);
    static_assert(!HasMakeKey<Without>);
    EXPECT_EQ(With::make_key(UINT64_C(0x0123456789abcdef)),
              (With::key_t{{0x89abcdefu}}));
}

TEST(RepackedOutput, RejectsInvalidWordTypesAndWidths) {
    static_assert(CanRepack<FixedEngine, std::uint32_t>);
    static_assert(CanRepack<FixedEngine, std::uint16_t>);
    static_assert(CanRepack<FixedEngine, std::uint8_t>);
    static_assert(!CanRepack<FixedEngine, std::int16_t>);
    static_assert(!CanRepack<FixedEngine, std::uint64_t>);

    // Standard unsigned integer widths cannot express these two cases on the
    // supported hosts, so exercise the width predicate directly.
    static_assert(!RandBLAS::rng::detail::valid_repacking_widths<32, 12>);
    static_assert(!RandBLAS::rng::detail::valid_repacking_widths<24, 8>);
    SUCCEED();
}

} // namespace
