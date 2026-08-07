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
#include "rng/concepts.hh"
#include "rng/distributions.hh"
#include "rng/philox.hh"
#include "rng/repacked_output.hh"
#include "rng/word_array.hh"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace RandBLAS {

using DefaultRNG = rng::Philox<4, 32, 10>;

/// Copyable state that binds an engine to one counter and one key.
template <rng::CounterBasedEngine Engine = DefaultRNG>
struct RNGState {
    using engine_t = Engine;
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    using res_t = typename Engine::res_t;

    ctr_t counter{};
    key_t key{};
    [[no_unique_address]] Engine engine{};

    constexpr RNGState() = default;

    constexpr RNGState(std::uint64_t seed) noexcept(
        noexcept(Engine::make_key(seed)))
        requires rng::SeedMappableEngine<Engine>
        : key(Engine::make_key(seed)) {}

    explicit constexpr RNGState(key_t const& input_key) : key(input_key) {}

    constexpr RNGState(ctr_t const& input_counter, key_t const& input_key)
        : counter(input_counter), key(input_key) {}

    constexpr void generate(res_t& output) const noexcept(
        noexcept(engine.generate(counter, key, output))) {
        engine.generate(counter, key, output);
    }

    constexpr void advance(std::uint64_t blocks) noexcept(
        noexcept(counter.advance(blocks))) {
        counter.advance(blocks);
    }

    friend constexpr bool operator==(RNGState const&, RNGState const&) = default;
};

using DefaultRNGState = RNGState<DefaultRNG>;

} // namespace RandBLAS
