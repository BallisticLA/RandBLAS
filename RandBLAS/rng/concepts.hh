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

#pragma once

#include <concepts>
#include <cstdint>
#include <tuple>

namespace RandBLAS::rng {

namespace detail {

template <class block_t>
concept FixedUnsignedBlock = requires {
    typename block_t::value_type;
    requires std::unsigned_integral<typename block_t::value_type>;
    requires(std::tuple_size_v<block_t> > 0);
};

} // namespace detail

/// Stateless counter-based engine producing one fixed-size result block.
template <class engine_t>
concept CounterBasedEngine =
    std::semiregular<engine_t> && requires {
        typename engine_t::ctr_t;
        typename engine_t::key_t;
        typename engine_t::res_t;
        requires std::regular<typename engine_t::ctr_t>;
        requires std::regular<typename engine_t::key_t>;
        requires detail::FixedUnsignedBlock<typename engine_t::res_t>;
    } && requires(engine_t const& engine, typename engine_t::ctr_t& counter,
                  typename engine_t::ctr_t const& const_counter,
                  typename engine_t::key_t const& key,
                  typename engine_t::res_t& output, std::uint64_t blocks) {
        { counter.advance(blocks) } -> std::same_as<void>;
        { engine.generate(const_counter, key, output) } -> std::same_as<void>;
    };

template <class engine_t>
concept SeedMappableEngine =
    CounterBasedEngine<engine_t> && requires(std::uint64_t seed) {
        { engine_t::make_key(seed) } ->
            std::same_as<typename engine_t::key_t>;
    };

} // namespace RandBLAS::rng

namespace RandBLAS {

/// Copyable generator state that produces and advances fixed-size blocks.
template <class state_t>
concept GeneratorState =
    std::copyable<state_t> && requires {
        typename state_t::res_t;
        requires rng::detail::FixedUnsignedBlock<typename state_t::res_t>;
    } && requires(state_t& state, state_t const& const_state,
                  typename state_t::res_t& output, std::uint64_t blocks) {
        { const_state.generate(output) } -> std::same_as<void>;
        { state.advance(blocks) } -> std::same_as<void>;
    };

} // namespace RandBLAS
