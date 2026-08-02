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

#include <RandBLAS/rng/philox.hh>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct KatRecord {
    std::string family;
    std::size_t rounds{};
    std::vector<std::uint64_t> words;
    std::size_t line{};
};

std::vector<KatRecord> read_vectors() {
    std::ifstream input(PHILOX_KAT_VECTORS_PATH);
    EXPECT_TRUE(input.is_open()) << PHILOX_KAT_VECTORS_PATH;

    std::vector<KatRecord> records;
    std::string text;
    for (std::size_t line = 1; std::getline(input, text); ++line) {
        if (text.empty() || text.front() == '#') {
            continue;
        }

        KatRecord record;
        record.line = line;
        std::istringstream fields(text);
        fields >> record.family >> record.rounds;
        std::uint64_t word;
        while (fields >> std::hex >> word) {
            record.words.push_back(word);
        }
        EXPECT_TRUE(fields.eof()) << "malformed vector at line " << line;
        records.push_back(std::move(record));
    }
    return records;
}

template <class Engine>
void check_vector(KatRecord const& record) {
    using word_t = typename Engine::word_t;
    typename Engine::ctr_t counter{};
    typename Engine::key_t key{};
    typename Engine::res_t expected{};

    constexpr std::size_t expected_words =
        Engine::ctr_t::static_size + Engine::key_t::static_size +
        std::tuple_size_v<typename Engine::res_t>;
    ASSERT_EQ(record.words.size(), expected_words) << "line " << record.line;

    std::size_t offset = 0;
    for (auto& word : counter.words) {
        word = static_cast<word_t>(record.words[offset++]);
    }
    for (auto& word : key.words) {
        word = static_cast<word_t>(record.words[offset++]);
    }
    for (auto& word : expected) {
        word = static_cast<word_t>(record.words[offset++]);
    }

    typename Engine::res_t actual;
    actual.fill(std::numeric_limits<word_t>::max());
    auto counter_before = counter;
    auto key_before = key;

    Engine{}.generate(counter, key, actual);

    EXPECT_EQ(actual, expected) << "line " << record.line;
    EXPECT_EQ(counter, counter_before) << "line " << record.line;
    EXPECT_EQ(key, key_before) << "line " << record.line;
}

template <std::size_t N, std::size_t W, std::size_t... Rounds>
void check_round(KatRecord const& record, std::index_sequence<Rounds...>) {
    bool matched = false;
    ([&] {
        if (record.rounds == Rounds) {
            check_vector<RandBLAS::rng::Philox<N, W, Rounds>>(record);
            matched = true;
        }
    }(),
     ...);
    EXPECT_TRUE(matched) << "unsupported round count at line " << record.line;
}

template <class Engine, std::size_t N, std::size_t W>
consteval bool has_expected_shape() {
    using word_t = typename Engine::word_t;
    return std::unsigned_integral<word_t> &&
           std::numeric_limits<word_t>::digits == W &&
           Engine::ctr_t::static_size == N &&
           Engine::key_t::static_size == N / 2 &&
           std::tuple_size_v<typename Engine::res_t> == N &&
           std::same_as<typename Engine::res_t::value_type, word_t>;
}

TEST(Philox, AllKnownAnswerVectors) {
    auto records = read_vectors();
    ASSERT_EQ(records.size(), 4U * 17U * 3U);

    std::array<std::size_t, 4U * 17U> counts{};
    for (auto const& record : records) {
        ASSERT_LE(record.rounds, 16U) << "line " << record.line;
        std::size_t family_index;
        if (record.family == "philox2x32") {
            family_index = 0;
            check_round<2, 32>(record, std::make_index_sequence<17>{});
        } else if (record.family == "philox4x32") {
            family_index = 1;
            check_round<4, 32>(record, std::make_index_sequence<17>{});
        } else if (record.family == "philox2x64") {
            family_index = 2;
            check_round<2, 64>(record, std::make_index_sequence<17>{});
        } else if (record.family == "philox4x64") {
            family_index = 3;
            check_round<4, 64>(record, std::make_index_sequence<17>{});
        } else {
            FAIL() << "unknown family at line " << record.line;
            continue;
        }
        ++counts[family_index * 17 + record.rounds];
    }

    for (auto count : counts) {
        EXPECT_EQ(count, 3U);
    }
}

TEST(Philox, RoundZeroCopiesCounterAndOverwritesOutput) {
    using Engine = RandBLAS::rng::Philox<4, 64, 0>;
    Engine::ctr_t counter{{0, 1, UINT64_C(0x8000000000000000),
                           UINT64_C(0x0123456789abcdef)}};
    Engine::key_t key{{UINT64_MAX, UINT64_C(0x3141592653589793)}};
    Engine::res_t output;
    output.fill(UINT64_MAX);

    Engine{}.generate(counter, key, output);

    EXPECT_EQ(output, (Engine::res_t{counter[0], counter[1], counter[2],
                                     counter[3]}));
}

TEST(Philox, PublicTypesHaveExpectedShapes) {
    static_assert(has_expected_shape<RandBLAS::rng::Philox<2, 32, 0>, 2, 32>());
    static_assert(has_expected_shape<RandBLAS::rng::Philox<4, 32, 16>, 4, 32>());
    static_assert(has_expected_shape<RandBLAS::rng::Philox<2, 64, 7>, 2, 64>());
    static_assert(has_expected_shape<RandBLAS::rng::Philox<4, 64, 10>, 4, 64>());
    SUCCEED();
}

TEST(Philox, MakeKeyUsesLittleEndianSeedAddition) {
    constexpr std::uint64_t seed = UINT64_C(0x0123456789abcdef);

    EXPECT_EQ((RandBLAS::rng::Philox<2, 32, 10>::make_key(0)),
              (RandBLAS::rng::Philox<2, 32, 10>::key_t{}));
    EXPECT_EQ((RandBLAS::rng::Philox<2, 32, 10>::make_key(seed)),
              (RandBLAS::rng::Philox<2, 32, 10>::key_t{{0x89abcdef}}));
    EXPECT_EQ((RandBLAS::rng::Philox<4, 32, 10>::make_key(seed)),
              (RandBLAS::rng::Philox<4, 32, 10>::key_t{{0x89abcdef,
                                                        0x01234567}}));
    EXPECT_EQ((RandBLAS::rng::Philox<2, 64, 10>::make_key(seed)),
              (RandBLAS::rng::Philox<2, 64, 10>::key_t{{seed}}));
    EXPECT_EQ((RandBLAS::rng::Philox<4, 64, 10>::make_key(seed)),
              (RandBLAS::rng::Philox<4, 64, 10>::key_t{{seed, 0}}));
}

} // namespace
