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

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace RandBLAS::rng {

/// Fixed-size little-endian words interpreted as one modular unsigned value.
template <std::unsigned_integral Word, std::size_t WordCount>
struct WordArray {
    static_assert(WordCount > 0, "WordArray requires at least one word");

    using value_type = Word;
    static constexpr std::size_t static_size = WordCount;

    std::array<Word, WordCount> words{};

    constexpr Word& operator[](std::size_t i) noexcept {
        return words[i];
    }

    constexpr Word const& operator[](std::size_t i) const noexcept {
        return words[i];
    }

    [[nodiscard]] static constexpr std::size_t size() noexcept {
        return WordCount;
    }

    /// Add an unsigned 64-bit amount, discarding overflow past the last word.
    constexpr void advance(std::uint64_t amount) noexcept {
        constexpr auto word_bits = std::numeric_limits<Word>::digits;
        std::uint64_t remaining = amount;
        Word carry = 0;

        for (std::size_t i = 0; i < WordCount; ++i) {
            Word addend;
            if constexpr (word_bits < 64) {
                constexpr std::uint64_t mask =
                    (std::uint64_t{1} << word_bits) - 1;
                addend = static_cast<Word>(remaining & mask);
                remaining >>= word_bits;
            } else {
                addend = static_cast<Word>(remaining);
                remaining = 0;
            }

            Word after_addend = static_cast<Word>(words[i] + addend);
            bool addend_overflow = after_addend < words[i];
            Word after_carry = static_cast<Word>(after_addend + carry);
            bool carry_overflow = after_carry < after_addend;
            words[i] = after_carry;
            carry = static_cast<Word>(addend_overflow || carry_overflow);

            if (remaining == 0 && carry == 0) {
                break;
            }
        }
    }

    friend constexpr bool operator==(WordArray const&, WordArray const&) = default;
};

} // namespace RandBLAS::rng
