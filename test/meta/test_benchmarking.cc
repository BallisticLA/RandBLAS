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

#include "RandBLAS/testing/benchmarking.hh"

#include <gtest/gtest.h>

#include <vector>

using RandBLAS::testing::parse_thread_counts;

TEST(ParseThreadCounts, parses_positive_counts) {
    // An explicit command-line list takes precedence over the benchmark's
    // defaults. Check the parsed order as well as the positivity decision.
    auto result = parse_thread_counts("1,3,8", {2, 4});

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.thread_counts, (std::vector<int>{1, 3, 8}));
}

TEST(ParseThreadCounts, uses_explicit_defaults_for_an_empty_list) {
    // Both benchmark programs initialize their thread configuration by
    // parsing an empty string. The caller-supplied defaults should survive
    // that path unchanged.
    auto result = parse_thread_counts("", {3, 6});

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.thread_counts, (std::vector<int>{3, 6}));
}

TEST(ParseThreadCounts, skips_empty_fields) {
    // Empty comma-separated fields did not represent thread counts in either
    // original parser. Verify that consolidating them preserves that behavior.
    auto result = parse_thread_counts("1,,4,", {2, 8});

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.thread_counts, (std::vector<int>{1, 4}));
}

TEST(ParseThreadCounts, rejects_nonpositive_counts) {
    // OpenMP thread requests must be positive. Exercise zero and a negative
    // value inside otherwise valid lists so either one invalidates the result.
    auto zero_result = parse_thread_counts("1,0,4", {2, 8});
    auto negative_result = parse_thread_counts("1,-2,4", {2, 8});

    EXPECT_FALSE(zero_result.valid);
    EXPECT_FALSE(negative_result.valid);
}

TEST(ParseThreadCounts, preserves_atoi_prefix_parsing) {
    // This extraction is deliberately behavior-preserving: the old benchmark
    // parsers used std::atoi, which accepts a numeric prefix and leading space.
    // Use both forms so a stricter parser cannot arrive as an accidental part
    // of this refactor.
    auto result = parse_thread_counts("2threads, 4", {1, 8});

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.thread_counts, (std::vector<int>{2, 4}));
}
