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

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

namespace RandBLAS::rng {

namespace detail {

template <class Word>
concept SupportedDistributionWord =
    std::unsigned_integral<Word> &&
    (std::numeric_limits<Word>::digits == 32 ||
     std::numeric_limits<Word>::digits == 64);

template <class Real>
concept SupportedDistributionReal =
    std::same_as<std::remove_cv_t<Real>, float> ||
    std::same_as<std::remove_cv_t<Real>, double>;

template <SupportedDistributionWord Word>
using default_real_t =
    std::conditional_t<std::numeric_limits<Word>::digits == 32, float, double>;

template <SupportedDistributionReal Real, SupportedDistributionWord Word>
[[nodiscard]] constexpr Real uneg11_value(Word input) noexcept {
    using signed_word_t = std::make_signed_t<Word>;
    constexpr Real factor =
        Real{1} / (static_cast<Real>(std::numeric_limits<signed_word_t>::max()) +
                   Real{1});
    constexpr Real half_factor = Real{0.5} * factor;
    return static_cast<Real>(static_cast<signed_word_t>(input)) * factor +
           half_factor;
}

template <class State>
concept StateCanGenerateFixedUnsignedBlock = requires {
    typename State::res_t;
    typename State::res_t::value_type;
    requires SupportedDistributionWord<typename State::res_t::value_type>;
    requires(std::tuple_size_v<typename State::res_t> > 0);
} && requires(State const& state, typename State::res_t& output) {
    { state.generate(output) } -> std::same_as<void>;
};

} // namespace detail

/// Convert a random unsigned word to a floating-point value in (0, 1].
template <detail::SupportedDistributionReal Real,
          detail::SupportedDistributionWord Word>
[[nodiscard]] constexpr Real u01(Word input) noexcept {
    constexpr Real factor =
        Real{1} / (static_cast<Real>(std::numeric_limits<Word>::max()) +
                   Real{1});
    constexpr Real half_factor = Real{0.5} * factor;
    return static_cast<Real>(input) * factor + half_factor;
}

template <detail::SupportedDistributionReal Real,
          detail::SupportedDistributionWord Word, std::size_t N>
[[nodiscard]] constexpr auto u01_block(
    std::array<Word, N> const& input) noexcept {
    std::array<Real, N> output{};
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = u01<Real>(input[i]);
    }
    return output;
}

template <detail::SupportedDistributionReal Real,
          detail::SupportedDistributionWord Word, std::size_t N>
[[nodiscard]] constexpr auto uneg11_block(
    std::array<Word, N> const& input) noexcept {
    std::array<Real, N> output{};
    for (std::size_t i = 0; i < N; ++i) {
        output[i] = detail::uneg11_value<Real>(input[i]);
    }
    return output;
}

template <detail::SupportedDistributionWord Word, std::size_t N>
[[nodiscard]] constexpr auto uneg11_block(
    std::array<Word, N> const& input) noexcept {
    return uneg11_block<detail::default_real_t<Word>>(input);
}

/// Symmetric-uniform conversion and dense-sampling transform policy.
struct uneg11 {
    /// Convert a random unsigned word to a floating-point value in [-1, 1].
    template <detail::SupportedDistributionReal Real,
              detail::SupportedDistributionWord Word>
    [[nodiscard]] static constexpr Real convert(Word input) noexcept {
        return detail::uneg11_value<Real>(input);
    }

    template <class State>
        requires detail::StateCanGenerateFixedUnsignedBlock<State>
    [[nodiscard]] static auto generate(State const& state) {
        typename State::res_t bits{};
        state.generate(bits);
        return uneg11_block(bits);
    }
};

/// Transform an angle word and a radius word into sine-then-cosine normals.
template <detail::SupportedDistributionWord Word>
[[nodiscard]] inline auto boxmuller(Word angle_word, Word radius_word) {
    using real_t = detail::default_real_t<Word>;
    constexpr real_t pi = real_t{3.1415926535897932};
    auto angle = pi * detail::uneg11_value<real_t>(angle_word);
    auto radius = std::sqrt(real_t{-2} * std::log(u01<real_t>(radius_word)));
    return std::array<real_t, 2>{std::sin(angle) * radius,
                                 std::cos(angle) * radius};
}

template <detail::SupportedDistributionReal Real,
          detail::SupportedDistributionWord Word, std::size_t N>
    requires(N % 2 == 0)
[[nodiscard]] inline auto boxmuller_block(std::array<Word, N> const& input) {
    std::array<Real, N> output{};
    constexpr Real pi = Real{3.1415926535897932};
    for (std::size_t i = 0; i < N; i += 2) {
        auto angle = pi * detail::uneg11_value<Real>(input[i]);
        auto radius =
            std::sqrt(Real{-2} * std::log(u01<Real>(input[i + 1])));
        output[i] = std::sin(angle) * radius;
        output[i + 1] = std::cos(angle) * radius;
    }
    return output;
}

template <detail::SupportedDistributionWord Word, std::size_t N>
    requires(N % 2 == 0)
[[nodiscard]] inline auto boxmuller_block(std::array<Word, N> const& input) {
    return boxmuller_block<detail::default_real_t<Word>>(input);
}

/// Box--Muller dense-sampling transform policy.
struct boxmul {
    template <class State>
        requires detail::StateCanGenerateFixedUnsignedBlock<State> &&
                 (std::tuple_size_v<typename State::res_t> % 2 == 0)
    [[nodiscard]] static auto generate(State const& state) {
        typename State::res_t bits{};
        state.generate(bits);
        return boxmuller_block(bits);
    }
};

} // namespace RandBLAS::rng
