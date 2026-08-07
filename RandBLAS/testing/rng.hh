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

#include "RandBLAS/random_gen.hh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace RandBLAS::testing::detail {

/// Sequential word stream used by RandBLAS test-data generators.
template <GeneratorState state_t = DefaultRNGState>
struct RNGStream {
    using res_t = typename state_t::res_t;
    using word_t = typename res_t::value_type;
    static constexpr std::size_t block_size = std::tuple_size_v<res_t>;

    state_t state;
    res_t buffer{};
    std::size_t pos = block_size;
    double spare = 0.0;
    bool has_spare = false;

    explicit RNGStream(state_t const& initial_state) : state(initial_state) {}

    word_t next_word() {
        if (pos >= block_size) {
            state.generate(buffer);
            state.advance(1);
            pos = 0;
        }
        return buffer[pos++];
    }

    /// Return a uniform value in (0, 1].
    double uniform_01() {
        return rng::u01<double>(next_word());
    }

    /// Return one normal value, caching the second Box--Muller result.
    template <typename value_t>
    value_t gaussian() {
        if (has_spare) {
            has_spare = false;
            return static_cast<value_t>(spare);
        }
        word_t angle_word = next_word();
        word_t radius_word = next_word();
        auto [first, second] = rng::boxmuller(angle_word, radius_word);
        spare = second;
        has_spare = true;
        return static_cast<value_t>(first);
    }

    /// Return the number of failures before the first Bernoulli success.
    std::int64_t geometric(double log_1_minus_p) {
        double u = uniform_01();
        return static_cast<std::int64_t>(
            std::floor(std::log(1.0 - u) / log_1_minus_p));
    }

    /// Report state after every result block already loaded into the buffer.
    state_t get_state() const { return state; }
};

} // namespace RandBLAS::testing::detail
