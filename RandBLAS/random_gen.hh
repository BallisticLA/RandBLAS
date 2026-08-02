// Copyright, 2024. See LICENSE for copyright holder information.
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
//

#pragma once

/// @file

#include "compilers.hh"
#include "rng/distributions.hh"
#include "rng/philox.hh"
#include "rng/repacked_output.hh"
#include "rng/word_array.hh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace RandBLAS::rng {

namespace detail {

template <class Block>
concept FixedUnsignedBlock = requires {
    typename Block::value_type;
    requires std::unsigned_integral<typename Block::value_type>;
    requires(std::tuple_size_v<Block> > 0);
};

} // namespace detail

/// Stateless counter-based engine producing one fixed-size result block.
template <class Engine>
concept CounterBasedEngine =
    std::semiregular<Engine> && requires {
        typename Engine::ctr_t;
        typename Engine::key_t;
        typename Engine::res_t;
        requires std::regular<typename Engine::ctr_t>;
        requires std::regular<typename Engine::key_t>;
        requires detail::FixedUnsignedBlock<typename Engine::res_t>;
    } && requires(Engine const& engine, typename Engine::ctr_t& counter,
                  typename Engine::ctr_t const& const_counter,
                  typename Engine::key_t const& key,
                  typename Engine::res_t& output, std::uint64_t blocks) {
        { counter.advance(blocks) } -> std::same_as<void>;
        { engine.generate(const_counter, key, output) } -> std::same_as<void>;
    };

template <class Engine>
concept SeedMappableEngine =
    CounterBasedEngine<Engine> && requires(std::uint64_t seed) {
        { Engine::make_key(seed) } -> std::same_as<typename Engine::key_t>;
    };

} // namespace RandBLAS::rng

namespace RandBLAS {

using DefaultRNG = rng::Philox<4, 32, 10>;

/// Copyable state that binds an engine to one counter and one key.
template <rng::CounterBasedEngine Engine = DefaultRNG>
class RNGState {
public:
    using engine_t = Engine;
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    using res_t = typename Engine::res_t;

    constexpr RNGState() = default;

    explicit constexpr RNGState(std::uint64_t seed) noexcept(
        noexcept(Engine::make_key(seed)))
        requires rng::SeedMappableEngine<Engine>
        : key_(Engine::make_key(seed)) {}

    explicit constexpr RNGState(key_t const& key) : key_(key) {}

    constexpr RNGState(ctr_t const& counter, key_t const& key)
        : counter_(counter), key_(key) {}

    constexpr void generate(res_t& output) const noexcept(
        noexcept(engine_.generate(counter_, key_, output))) {
        engine_.generate(counter_, key_, output);
    }

    constexpr void advance(std::uint64_t blocks) noexcept(
        noexcept(counter_.advance(blocks))) {
        counter_.advance(blocks);
    }

    [[nodiscard]] constexpr ctr_t const& counter() const noexcept {
        return counter_;
    }

    [[nodiscard]] constexpr key_t const& key() const noexcept {
        return key_;
    }

    friend constexpr bool operator==(RNGState const& left,
                                     RNGState const& right) {
        return left.counter_ == right.counter_ && left.key_ == right.key_;
    }

private:
    ctr_t counter_{};
    key_t key_{};
    [[no_unique_address]] Engine engine_{};
};

template <class State>
concept CounterBasedRNGState =
    std::copyable<State> && requires {
        typename State::res_t;
        requires rng::detail::FixedUnsignedBlock<typename State::res_t>;
    } && requires(State& state, State const& const_state,
                  typename State::res_t& output, std::uint64_t blocks) {
        { const_state.generate(output) } -> std::same_as<void>;
        { state.advance(blocks) } -> std::same_as<void>;
    };

using DefaultRNGState = RNGState<DefaultRNG>;

} // namespace RandBLAS
