// Copyright, 2026. See LICENSE for copyright holder information.
/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions, and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

// The integer conversions and Box--Muller mapping in this file are adapted
// from Random123's uniform.hpp and boxmuller.hpp.

#pragma once

#include "concepts.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>

namespace RandBLAS::rng {

namespace detail {

template <typename word_t>
using default_real_t = std::conditional_t<sizeof(word_t) == 4, float, double>;

} // namespace detail

/// Convert a random unsigned word to a floating-point value in (0, 1].
template <typename real_t, typename word_t>
[[nodiscard]] constexpr real_t u01(word_t input) noexcept {
    static_assert(std::is_unsigned_v<word_t>);
    static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
    static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>);
    constexpr real_t factor =
        real_t{1} /
        (static_cast<real_t>(std::numeric_limits<word_t>::max()) + real_t{1});
    constexpr real_t half_factor = real_t{0.5} * factor;
    return static_cast<real_t>(input) * factor + half_factor;
}

/// Symmetric-uniform conversion and dense-sampling transform policy.
struct uneg11 {
    /// Convert a random unsigned word to a floating-point value in [-1, 1].
    template <typename real_t, typename word_t>
    [[nodiscard]] static constexpr real_t convert(word_t input) noexcept {
        static_assert(std::is_unsigned_v<word_t>);
        static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
        static_assert(std::is_same_v<real_t, float> || std::is_same_v<real_t, double>);
        using signed_word_t = std::make_signed_t<word_t>;
        constexpr real_t factor =
            real_t{1} /
            (static_cast<real_t>(
                 std::numeric_limits<signed_word_t>::max()) + real_t{1});
        constexpr real_t half_factor = real_t{0.5} * factor;
        return static_cast<real_t>(static_cast<signed_word_t>(input)) * factor +
               half_factor;
    }

    template <GeneratorState state_t>
    [[nodiscard]] static auto generate(state_t const& state) {
        using bits_t = typename state_t::res_t;
        using word_t = typename bits_t::value_type;
        using real_t = detail::default_real_t<word_t>;
        constexpr std::size_t count = std::tuple_size_v<bits_t>;
        bits_t bits{};
        std::array<real_t, count> output{};
        state.generate(bits);
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = convert<real_t>(bits[i]);
        }
        return output;
    }
};

/// Transform an angle word and a radius word into sine-then-cosine normals.
template <typename word_t>
[[nodiscard]] inline auto boxmuller(word_t angle_word, word_t radius_word) {
    static_assert(std::is_unsigned_v<word_t>);
    static_assert(sizeof(word_t) == 4 || sizeof(word_t) == 8);
    using real_t = detail::default_real_t<word_t>;
    constexpr real_t pi = real_t{3.1415926535897932};
    auto angle = pi * uneg11::convert<real_t>(angle_word);
    auto radius = std::sqrt(real_t{-2} * std::log(u01<real_t>(radius_word)));
    return std::array<real_t, 2>{std::sin(angle) * radius,
                                 std::cos(angle) * radius};
}

/// Box--Muller dense-sampling transform policy.
struct boxmul {
    template <GeneratorState state_t>
        requires(std::tuple_size_v<typename state_t::res_t> % 2 == 0)
    [[nodiscard]] static auto generate(state_t const& state) {
        using bits_t = typename state_t::res_t;
        using word_t = typename bits_t::value_type;
        using real_t = detail::default_real_t<word_t>;
        constexpr std::size_t count = std::tuple_size_v<bits_t>;
        bits_t bits{};
        std::array<real_t, count> output{};
        state.generate(bits);
        for (std::size_t i = 0; i < count; i += 2) {
            auto pair = boxmuller(bits[i], bits[i + 1]);
            output[i] = pair[0];
            output[i + 1] = pair[1];
        }
        return output;
    }
};

} // namespace RandBLAS::rng
