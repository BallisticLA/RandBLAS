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

#include "RandBLAS/testing/samplers.hh"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <vector>

// MARK: test helpers

class TestSamplers : public ::testing::Test {
protected:
    // A support sampler writes a block of vec_nnz indices for each major-axis
    // vector. This helper checks that every block represents a subset of
    // {0, ..., n - 1}: all indices are in range, and no index is repeated
    // within a block. It does not test whether those subsets are uniform.
    static void expect_valid_samples(
        int64_t n, int64_t num_vectors, int64_t vec_nnz,
        const std::vector<int64_t> &samples
    ) {
        ASSERT_EQ(samples.size(), static_cast<std::size_t>(num_vectors * vec_nnz));
        for (int64_t vector = 0; vector < num_vectors; ++vector) {
            std::vector<bool> seen(n, false);
            for (int64_t entry = 0; entry < vec_nnz; ++entry) {
                int64_t index = samples[vector * vec_nnz + entry];
                ASSERT_GE(index, 0);
                ASSERT_LT(index, n);
                ASSERT_FALSE(seen[index]);
                seen[index] = true;
            }
        }
    }
};

// MARK: support samplers

TEST_F(TestSamplers, std_sample_produces_valid_major_axis_vectors) {
    // std::sample is the most direct standard-library baseline for sampling
    // without replacement. Draw several support sets from one RNG stream and
    // pass the resulting blocks through the structural checks above.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::testing::sample_std_sample(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSamplers, partial_fisher_yates_produces_valid_major_axis_vectors) {
    // Partial Fisher-Yates stops after selecting the requested number of
    // indices rather than shuffling the full population. Check that its output
    // still has the subset structure required of every major-axis vector.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::testing::sample_partial_fisher_yates(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSamplers, full_shuffle_produces_valid_major_axis_vectors) {
    // This baseline shuffles all n indices and keeps the first vec_nnz entries.
    // The retained prefix should therefore pass the same range and uniqueness
    // checks as the samplers that avoid a full shuffle.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::testing::sample_full_shuffle(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSamplers, rejection_produces_valid_major_axis_vectors) {
    // Rejection sampling draws indices until it has vec_nnz distinct values.
    // Generate several blocks and check the property that makes a completed
    // block valid: every entry is in range and appears only once.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::testing::sample_rejection(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSamplers, floyd_produces_valid_major_axis_vectors) {
    // Floyd's algorithm constructs a subset without storing a permutation of
    // all n indices. Its internal representation is different, but its output
    // must satisfy the same per-vector subset contract.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::testing::sample_floyd(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

// MARK: end-to-end COO sampling

TEST_F(TestSamplers, saso_data_respects_major_axis_orientation) {
    // fill_saso_data turns sampled support sets into COO matrix data. Within
    // each block, the minor coordinate identifies the major-axis vector while
    // the major coordinate stores that vector's sampled support. Exercise both
    // orientations and check the full COO contract: valid sorted supports,
    // correct vector labels, and nonzero values in {-1, +1}.
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    constexpr int64_t nnz = num_vectors * vec_nnz;

    for (bool major_is_rows : {false, true}) {
        std::vector<int64_t> rows(nnz, -1);
        std::vector<int64_t> cols(nnz, -1);
        std::vector<double> values(nnz, 0.0);
        std::mt19937_64 rng(42);
        auto sampler = [](
            int64_t sampler_n, int64_t sampler_num_vectors,
            int64_t sampler_vec_nnz, int64_t *samples, auto &sampler_rng
        ) {
            RandBLAS::testing::sample_partial_fisher_yates(
                sampler_n, sampler_num_vectors, sampler_vec_nnz,
                samples, sampler_rng
            );
        };

        RandBLAS::testing::fill_saso_data(
            n, num_vectors, vec_nnz, major_is_rows,
            rows.data(), cols.data(), values.data(), rng, sampler
        );

        const std::vector<int64_t> &major = major_is_rows ? rows : cols;
        const std::vector<int64_t> &minor = major_is_rows ? cols : rows;
        expect_valid_samples(n, num_vectors, vec_nnz, major);
        for (int64_t vector = 0; vector < num_vectors; ++vector) {
            for (int64_t entry = 0; entry < vec_nnz; ++entry) {
                int64_t offset = vector * vec_nnz + entry;
                EXPECT_EQ(minor[offset], vector);
                EXPECT_TRUE(values[offset] == -1.0 || values[offset] == 1.0);
                if (entry > 0) {
                    EXPECT_LT(major[offset - 1], major[offset]);
                }
            }
        }
    }
}

// MARK: Philox URBG adapter

TEST_F(TestSamplers, philox_urbg_matches_randblas_stream) {
    // In its default mode, PhiloxURBG returns one 64-bit value per Philox
    // counter and then advances to the next counter. Generate the same counters
    // directly with DefaultRNG, combine the first two 32-bit words by hand,
    // and use those values as the reference stream.
    constexpr uint64_t seed = 42;
    RandBLAS::RNGState<> state(seed);
    RandBLAS::DefaultRNG generator;
    RandBLAS::testing::PhiloxURBG rng(seed);

    for (int64_t draw = 0; draw < 3; ++draw) {
        auto random_values = generator(state.counter, state.key);
        uint64_t expected = RandBLAS::promote_uint_pair(random_values[0], random_values[1]);
        EXPECT_EQ(rng(), expected);
        state.counter.incr();
    }
}

TEST_F(TestSamplers, philox_urbg_can_use_both_results_per_counter) {
    // When one-result-per-counter mode is disabled, the adapter exposes two
    // 64-bit values from each four-word Philox result before advancing the
    // counter. Compute both values directly and check their order.
    constexpr uint64_t seed = 42;
    RandBLAS::RNGState<> state(seed);
    RandBLAS::DefaultRNG generator;
    RandBLAS::testing::PhiloxURBG rng(seed, false);

    for (int64_t counter = 0; counter < 3; ++counter) {
        auto random_values = generator(state.counter, state.key);
        uint64_t first = RandBLAS::promote_uint_pair(random_values[0], random_values[1]);
        uint64_t second = RandBLAS::promote_uint_pair(random_values[2], random_values[3]);
        EXPECT_EQ(rng(), first);
        EXPECT_EQ(rng(), second);
        state.counter.incr();
    }
}
