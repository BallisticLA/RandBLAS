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

// Adapted from Random123's Philox implementation. The public interface and
// storage types are native RandBLAS facilities; constants and round operations
// intentionally preserve Random123's bit stream.

#pragma once

#include "word_array.hh"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace RandBLAS::rng {

namespace detail {

template <std::unsigned_integral Word>
[[nodiscard]] constexpr Word mulhilo(Word left, Word right, Word* high) noexcept {
    static_assert(sizeof(Word) == 4 || sizeof(Word) == 8);

    if constexpr (sizeof(Word) == 4) {
        auto product = static_cast<std::uint64_t>(left) *
                       static_cast<std::uint64_t>(right);
        *high = static_cast<Word>(product >> 32);
        return static_cast<Word>(product);
    } else {
#if defined(_MSC_VER) && defined(_M_X64)
        unsigned __int64 native_high;
        auto low = _umul128(static_cast<unsigned __int64>(left),
                            static_cast<unsigned __int64>(right),
                            &native_high);
        *high = static_cast<Word>(native_high);
        return static_cast<Word>(low);
#elif defined(__SIZEOF_INT128__)
        using double_word_t = unsigned __int128;
        auto product = static_cast<double_word_t>(left) *
                       static_cast<double_word_t>(right);
        *high = static_cast<Word>(product >> 64);
        return static_cast<Word>(product);
#else
        static_assert(sizeof(Word) != 8,
                      "64-bit Philox requires unsigned __int128 or _umul128");
#endif
    }
}

template <std::size_t N, std::size_t W>
struct PhiloxConstants {
    using word_t = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;

    static constexpr word_t multiplier_0 = [] {
        if constexpr (W == 32 && N == 2) {
            return UINT32_C(0xd256d193);
        } else if constexpr (W == 32) {
            return UINT32_C(0xd2511f53);
        } else if constexpr (N == 2) {
            return UINT64_C(0xd2b74407b1ce6e93);
        } else {
            return UINT64_C(0xd2e7470ee14c6c93);
        }
    }();

    static constexpr word_t multiplier_1 = [] {
        if constexpr (W == 32) {
            return UINT32_C(0xcd9e8d57);
        } else {
            return UINT64_C(0xca5a826395121157);
        }
    }();

    static constexpr word_t weyl_0 = [] {
        if constexpr (W == 32) {
            return UINT32_C(0x9e3779b9);
        } else {
            return UINT64_C(0x9e3779b97f4a7c15);
        }
    }();

    static constexpr word_t weyl_1 = [] {
        if constexpr (W == 32) {
            return UINT32_C(0xbb67ae85);
        } else {
            return UINT64_C(0xbb67ae8584caa73b);
        }
    }();
};

template <std::size_t N, std::size_t W, class word_t, class key_t>
constexpr void apply_philox_round(std::array<word_t, N>& block,
                                  key_t const& key) noexcept {
    using constants_t = PhiloxConstants<N, W>;

    auto input_0 = block[0];
    auto input_1 = block[1];
    word_t high_0;
    auto low_0 = mulhilo(constants_t::multiplier_0, input_0, &high_0);

    if constexpr (N == 2) {
        block[0] = static_cast<word_t>(high_0 ^ key[0] ^ input_1);
        block[1] = low_0;
    } else {
        auto input_2 = block[2];
        auto input_3 = block[3];
        word_t high_1;
        auto low_1 = mulhilo(constants_t::multiplier_1, input_2, &high_1);
        block[0] = static_cast<word_t>(high_1 ^ input_1 ^ key[0]);
        block[1] = low_1;
        block[2] = static_cast<word_t>(high_0 ^ input_3 ^ key[1]);
        block[3] = low_0;
    }
}

template <std::size_t N, std::size_t W, class key_t>
constexpr void bump_philox_key(key_t& key) noexcept {
    using constants_t = PhiloxConstants<N, W>;
    using word_t = typename key_t::value_type;

    key[0] = static_cast<word_t>(key[0] + constants_t::weyl_0);
    if constexpr (N == 4) {
        key[1] = static_cast<word_t>(key[1] + constants_t::weyl_1);
    }
}

} // namespace detail

/// Stateless Philox counter-based random-number engine.
///
/// Word zero is the least-significant word of counters and keys. `generate`
/// maps one counter/key pair to one result block without modifying its inputs.
template <std::size_t N, std::size_t W, std::size_t R>
struct Philox {
    static_assert(N == 2 || N == 4, "Philox supports two or four words");
    static_assert(W == 32 || W == 64, "Philox supports 32- or 64-bit words");
    static_assert(R <= 16, "Philox supports at most 16 rounds");

    using word_t = std::conditional_t<W == 32, std::uint32_t, std::uint64_t>;
    using ctr_t = WordArray<word_t, N>;
    using key_t = WordArray<word_t, N / 2>;
    using res_t = std::array<word_t, N>;

    /// Map a scalar seed to a key by adding it to an all-zero key.
    [[nodiscard]] static constexpr key_t make_key(std::uint64_t seed) noexcept {
        key_t key{};
        key.advance(seed);
        return key;
    }

    /// Generate a complete result block, overwriting every output lane.
    constexpr void generate(ctr_t const& counter, key_t const& key,
                            res_t& output) const noexcept {
        res_t block{};
        for (std::size_t i = 0; i < N; ++i) {
            block[i] = counter[i];
        }

        key_t round_key = key;
        for (std::size_t round = 0; round < R; ++round) {
            detail::apply_philox_round<N, W>(block, round_key);
            if (round + 1 < R) {
                detail::bump_philox_key<N, W>(round_key);
            }
        }
        output = block;
    }

    friend constexpr bool operator==(Philox const&, Philox const&) = default;
};

} // namespace RandBLAS::rng
