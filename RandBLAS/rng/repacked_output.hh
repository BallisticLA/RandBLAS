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

#include "concepts.hh"

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace RandBLAS::rng {

namespace detail {

template <std::size_t SourceBits, std::size_t OutputBits>
inline constexpr bool valid_repacking_widths =
    OutputBits > 0 && OutputBits <= SourceBits &&
    SourceBits % OutputBits == 0 &&
    std::has_single_bit(SourceBits / OutputBits);

} // namespace detail

template <class SourceWord, class OutputWord>
concept ValidRepacking =
    std::unsigned_integral<SourceWord> &&
    std::unsigned_integral<OutputWord> &&
    detail::valid_repacking_widths<std::numeric_limits<SourceWord>::digits,
                                   std::numeric_limits<OutputWord>::digits>;

/// Express each result word of an engine as fixed-width, LSB-first chunks.
///
/// The adaptor preserves the wrapped engine's counter, key, seed mapping, and
/// total number of bits per block. Equal-width adaptation is an identity.
template <class Engine, std::unsigned_integral OutputWord>
    requires CounterBasedEngine<Engine> &&
             ValidRepacking<typename Engine::res_t::value_type, OutputWord>
struct RepackedOutput {
    using source_res_t = typename Engine::res_t;
    using source_word_t = typename source_res_t::value_type;

    static constexpr std::size_t source_word_bits =
        std::numeric_limits<source_word_t>::digits;
    static constexpr std::size_t output_word_bits =
        std::numeric_limits<OutputWord>::digits;
    static constexpr std::size_t chunks_per_source_word =
        source_word_bits / output_word_bits;
    static constexpr std::size_t source_word_count =
        std::tuple_size_v<source_res_t>;

    using word_t = OutputWord;
    using ctr_t = typename Engine::ctr_t;
    using key_t = typename Engine::key_t;
    static constexpr std::size_t repacked_word_count =
        source_word_count * chunks_per_source_word;
    using res_t = std::array<OutputWord, repacked_word_count>;

    constexpr RepackedOutput() requires std::default_initializable<Engine> = default;

    constexpr explicit RepackedOutput(Engine engine) noexcept(
        std::is_nothrow_move_constructible_v<Engine>)
        : engine(std::move(engine)) {}

    [[nodiscard]] static constexpr key_t make_key(std::uint64_t seed) noexcept(
        noexcept(Engine::make_key(seed)))
        requires requires {
            { Engine::make_key(seed) } -> std::same_as<key_t>;
        }
    {
        return Engine::make_key(seed);
    }

    constexpr void generate(ctr_t const& counter, key_t const& key,
                            res_t& output) const noexcept(
        noexcept(engine.generate(counter, key,
                                 std::declval<source_res_t&>()))) {
        source_res_t source{};
        engine.generate(counter, key, source);

        constexpr source_word_t mask = [] {
            if constexpr (output_word_bits == source_word_bits) {
                return std::numeric_limits<source_word_t>::max();
            } else {
                return static_cast<source_word_t>(
                    (source_word_t{1} << output_word_bits) - 1);
            }
        }();

        std::size_t output_index = 0;
        for (source_word_t source_word : source) {
            for (std::size_t chunk = 0; chunk < chunks_per_source_word; ++chunk) {
                auto shifted = static_cast<source_word_t>(
                    source_word >> (chunk * output_word_bits));
                output[output_index++] = static_cast<OutputWord>(shifted & mask);
            }
        }
    }

    friend constexpr bool operator==(RepackedOutput const&, RepackedOutput const&) = default;

    [[no_unique_address]] Engine engine{};
};

} // namespace RandBLAS::rng
