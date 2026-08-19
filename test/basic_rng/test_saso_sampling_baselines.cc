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

#include "../../examples/simple-kernel-benchmarks/saso_sampling_baselines.hh"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <vector>

class TestSasoSamplingBaselines : public ::testing::Test {
protected:
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

TEST_F(TestSasoSamplingBaselines, std_sample_produces_valid_major_axis_vectors) {
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::benchmark::sample_std_sample(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSasoSamplingBaselines, partial_fisher_yates_produces_valid_major_axis_vectors) {
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::benchmark::sample_partial_fisher_yates(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSasoSamplingBaselines, full_shuffle_produces_valid_major_axis_vectors) {
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::benchmark::sample_full_shuffle(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSasoSamplingBaselines, rejection_produces_valid_major_axis_vectors) {
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::benchmark::sample_rejection(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSasoSamplingBaselines, floyd_produces_valid_major_axis_vectors) {
    constexpr int64_t n = 7;
    constexpr int64_t num_vectors = 11;
    constexpr int64_t vec_nnz = 4;
    std::vector<int64_t> samples(num_vectors * vec_nnz, -1);
    std::mt19937_64 rng(42);

    RandBLAS::benchmark::sample_floyd(
        n, num_vectors, vec_nnz, samples.data(), rng
    );

    expect_valid_samples(n, num_vectors, vec_nnz, samples);
}

TEST_F(TestSasoSamplingBaselines, saso_data_respects_major_axis_orientation) {
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
            RandBLAS::benchmark::sample_partial_fisher_yates(
                sampler_n, sampler_num_vectors, sampler_vec_nnz,
                samples, sampler_rng
            );
        };

        RandBLAS::benchmark::fill_saso_data(
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

TEST_F(TestSasoSamplingBaselines, philox_urbg_matches_randblas_stream) {
    constexpr uint64_t seed = 42;
    RandBLAS::RNGState<> state(seed);
    RandBLAS::DefaultRNG generator;
    RandBLAS::benchmark::PhiloxURBG rng(seed);

    for (int64_t draw = 0; draw < 3; ++draw) {
        auto random_values = generator(state.counter, state.key);
        uint64_t expected = RandBLAS::promote_uint_pair(random_values[0], random_values[1]);
        EXPECT_EQ(rng(), expected);
        state.counter.incr();
    }
}
