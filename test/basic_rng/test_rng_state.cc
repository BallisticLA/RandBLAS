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

#include <RandBLAS/base.hh>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>

namespace {

class OpaqueCounter {
public:
    constexpr void advance(std::uint64_t blocks) noexcept {
        value_ += blocks;
    }

    friend constexpr bool operator==(OpaqueCounter const&,
                                     OpaqueCounter const&) = default;

private:
    std::uint64_t value_ = 0;
    friend struct OpaqueEngine;
};

struct OpaqueEngine {
    using ctr_t = OpaqueCounter;
    using key_t = std::array<std::uint32_t, 1>;
    using res_t = std::array<std::uint32_t, 2>;

    static constexpr key_t make_key(std::uint64_t seed) noexcept {
        return {static_cast<std::uint32_t>(seed)};
    }

    constexpr void generate(ctr_t const& counter, key_t const& key,
                            res_t& output) const noexcept {
        output[0] = static_cast<std::uint32_t>(counter.value_);
        output[1] = static_cast<std::uint32_t>(counter.value_ >> 32) ^ key[0];
    }

    friend constexpr bool operator==(OpaqueEngine const&,
                                     OpaqueEngine const&) = default;
};

struct EngineWithoutMakeKey {
    using ctr_t = OpaqueEngine::ctr_t;
    using key_t = OpaqueEngine::key_t;
    using res_t = OpaqueEngine::res_t;

    constexpr void generate(ctr_t const& counter, key_t const& key,
                            res_t& output) const noexcept {
        OpaqueEngine{}.generate(counter, key, output);
    }
};

static_assert(RandBLAS::rng::CounterBasedEngine<OpaqueEngine>);
using OpaqueState = RandBLAS::RNGState<OpaqueEngine>;

template <class state_t>
concept HasPublicStateData = requires(state_t state) {
    state.counter;
    state.key;
    state.engine;
};

static_assert(RandBLAS::GeneratorState<OpaqueState>);
static_assert(HasPublicStateData<OpaqueState>);
static_assert(std::equality_comparable<OpaqueState>);
static_assert(std::convertible_to<std::uint64_t, OpaqueState>);
static_assert(!std::uniform_random_bit_generator<OpaqueEngine>);
static_assert(!std::uniform_random_bit_generator<
              RandBLAS::RNGState<OpaqueEngine>>);

TEST(RNGState, SupportsAllApprovedConstructionForms) {
    using State = RandBLAS::RNGState<OpaqueEngine>;
    State default_state;
    State scalar_seeded(UINT64_C(0x0123456789abcdef));
    OpaqueEngine::key_t explicit_key{UINT32_C(0x31415926)};
    State key_seeded(explicit_key);
    OpaqueCounter counter;
    counter.advance(UINT64_C(0x100000002));
    State explicit_state(counter, explicit_key);

    OpaqueEngine::res_t output{};
    default_state.generate(output);
    EXPECT_EQ(output, (OpaqueEngine::res_t{0, 0}));
    scalar_seeded.generate(output);
    EXPECT_EQ(output, (OpaqueEngine::res_t{0, UINT32_C(0x89abcdef)}));
    key_seeded.generate(output);
    EXPECT_EQ(output, (OpaqueEngine::res_t{0, UINT32_C(0x31415926)}));
    explicit_state.generate(output);
    EXPECT_EQ(output, (OpaqueEngine::res_t{2, UINT32_C(0x31415927)}));

    static_assert(!std::constructible_from<
                  RandBLAS::RNGState<EngineWithoutMakeKey>, std::uint64_t>);
}

TEST(RNGState, HasRuleOfZeroValueSemanticsAndEquality) {
    using State = RandBLAS::RNGState<OpaqueEngine>;
    static_assert(std::copyable<State>);
    static_assert(std::movable<State>);
    static_assert(std::is_copy_assignable_v<State>);
    static_assert(std::is_move_assignable_v<State>);

    State original(UINT64_C(0x12345678));
    original.advance(19);
    State copied = original;
    EXPECT_EQ(copied, original);

    State assigned;
    assigned = copied;
    EXPECT_EQ(assigned, original);

    State moved = std::move(copied);
    EXPECT_EQ(moved, original);
    State move_assigned;
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned, original);
}

TEST(RNGState, GenerateDoesNotMutateAndAdvanceDelegatesToCounter) {
    using State = RandBLAS::RNGState<OpaqueEngine>;
    State state(UINT64_C(0xa5a5a5a5));
    auto before = state;
    State::res_t output{};

    state.generate(output);

    EXPECT_EQ(state, before);
    EXPECT_EQ(output, (State::res_t{0, UINT32_C(0xa5a5a5a5)}));

    OpaqueCounter expected_counter;
    expected_counter.advance(UINT64_C(0x100000003));
    state.advance(UINT64_C(0x100000003));
    EXPECT_EQ(state.counter, expected_counter);
    state.generate(output);
    EXPECT_EQ(output, (State::res_t{3, UINT32_C(0xa5a5a5a4)}));
}

TEST(RNGState, ExposesItsValueStateAsPublicData) {
    OpaqueState state(UINT64_C(0x0123456789abcdef));
    EXPECT_EQ(state.counter, OpaqueCounter{});
    EXPECT_EQ(state.key,
              (OpaqueEngine::key_t{UINT32_C(0x89abcdef)}));
}

TEST(RNGState, RepackedStatePreservesBitsAndBlockAdvancement) {
    using BaseEngine = RandBLAS::DefaultRNG;
    using RepackedEngine =
        RandBLAS::rng::RepackedOutput<BaseEngine, std::uint16_t>;
    RandBLAS::RNGState<BaseEngine> base(UINT64_C(0x0123456789abcdef));
    RandBLAS::RNGState<RepackedEngine> repacked(
        UINT64_C(0x0123456789abcdef));
    BaseEngine::res_t base_output{};
    RepackedEngine::res_t repacked_output{};

    base.generate(base_output);
    repacked.generate(repacked_output);
    for (std::size_t i = 0; i < base_output.size(); ++i) {
        EXPECT_EQ(repacked_output[2 * i],
                  static_cast<std::uint16_t>(base_output[i]));
        EXPECT_EQ(repacked_output[2 * i + 1],
                  static_cast<std::uint16_t>(base_output[i] >> 16));
    }

    base.advance(37);
    repacked.advance(37);
    EXPECT_EQ(base.counter, repacked.counter);
    EXPECT_EQ(base.key, repacked.key);
}

} // namespace
