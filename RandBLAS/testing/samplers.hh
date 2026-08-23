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

#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace RandBLAS::testing {

// Adapt RandBLAS' default Philox generator to the UniformRandomBitGenerator
// interface expected by the C++ standard library.
class PhiloxURBG {
public:
    using result_type = uint64_t;

    // Initialize the adapter with a RandBLAS seed and counter zero. By default,
    // each call consumes a new counter, matching RandBLAS' sampling kernels.
    explicit PhiloxURBG(uint64_t seed, bool one_result_per_counter = true)
        : state_(seed),
          one_result_per_counter_(one_result_per_counter),
          random_values_{},
          use_second_result_(false) {}

    // Return the smallest value produced by the adapter.
    static constexpr result_type min() {
        return 0;
    }

    // Return the largest value produced by the adapter.
    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    // Draw one 64-bit value, optionally using both 64-bit pairs from each
    // Philox4x32 result before advancing to the next result.
    result_type operator()() {
        if (use_second_result_) {
            use_second_result_ = false;
            return RandBLAS::promote_uint_pair(random_values_[2], random_values_[3]);
        }
        typename RNGState<>::generator generator;
        random_values_ = generator(state_.counter, state_.key);
        state_.counter.incr();
        use_second_result_ = !one_result_per_counter_;
        return RandBLAS::promote_uint_pair(random_values_[0], random_values_[1]);
    }

private:
    RNGState<> state_;
    bool one_result_per_counter_;
    typename RNGState<>::ctr_type random_values_;
    bool use_second_result_;
};

// Use std::sample over an iota-filled vector to draw vec_nnz distinct indices
// from [0, n) for each requested vector. This scans all n candidates for every
// vector and uses O(n) workspace.
template <typename RNG>
void sample_std_sample(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, int64_t *samples, RNG &rng
) {
    std::vector<int64_t> population(n);
    std::iota(population.begin(), population.end(), int64_t{0});
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::sample(
            population.begin(), population.end(),
            samples + vector * vec_nnz, vec_nnz, rng
        );
    }
}

// Use the first vec_nnz steps of Fisher-Yates to draw distinct indices from
// [0, n). The identity permutation is rebuilt for every requested vector.
template <typename RNG>
void sample_partial_fisher_yates(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, int64_t *samples, RNG &rng
) {
    std::vector<int64_t> population(n);
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::iota(population.begin(), population.end(), int64_t{0});
        for (int64_t entry = 0; entry < vec_nnz; ++entry) {
            std::uniform_int_distribution<int64_t> pick(entry, n - 1);
            int64_t pivot = pick(rng);
            std::swap(population[entry], population[pivot]);
            samples[vector * vec_nnz + entry] = population[entry];
        }
    }
}

// Shuffle a permutation of [0, n) and copy its first vec_nnz entries for each
// requested vector. The method uses O(n) work per vector and O(n) workspace.
template <typename RNG>
void sample_full_shuffle(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, int64_t *samples, RNG &rng
) {
    std::vector<int64_t> population(n);
    std::iota(population.begin(), population.end(), int64_t{0});
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::shuffle(population.begin(), population.end(), rng);
        std::copy_n(population.begin(), vec_nnz, samples + vector * vec_nnz);
    }
}

// Draw indices uniformly from [0, n), rejecting duplicates found by a linear
// scan. The expected work is O(vec_nnz^2) per vector when vec_nnz is small.
template <typename RNG>
void sample_rejection(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, int64_t *samples, RNG &rng
) {
    std::uniform_int_distribution<int64_t> distribution(0, n - 1);
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        int64_t *vector_samples = samples + vector * vec_nnz;
        for (int64_t entry = 0; entry < vec_nnz;) {
            int64_t candidate = distribution(rng);
            auto duplicate = std::find(vector_samples, vector_samples + entry, candidate);
            if (duplicate == vector_samples + entry) {
                vector_samples[entry] = candidate;
                ++entry;
            }
        }
    }
}

// Apply Floyd's algorithm with std::unordered_set to draw vec_nnz distinct
// indices from [0, n). Expected work and workspace are O(vec_nnz).
template <typename RNG>
void sample_floyd(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, int64_t *samples, RNG &rng
) {
    std::unordered_set<int64_t> selected_values;
    selected_values.reserve(vec_nnz);
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        selected_values.clear();
        for (int64_t entry = 0; entry < vec_nnz; ++entry) {
            int64_t upper_bound = n - vec_nnz + entry;
            std::uniform_int_distribution<int64_t> pick(0, upper_bound);
            int64_t candidate = pick(rng);
            bool inserted = selected_values.insert(candidate).second;
            int64_t selected = inserted ? candidate : upper_bound;
            if (!inserted) {
                selected_values.insert(selected);
            }
            samples[vector * vec_nnz + entry] = selected;
        }
    }
}

// Fill SASO COO data from a sampler of distinct major coordinates. The minor
// coordinates identify vectors, values are random signs, and each vector's
// major coordinates are sorted into canonical order.
template <typename RNG, typename Sampler>
void fill_saso_data(
    int64_t n, int64_t num_vectors, int64_t vec_nnz, bool major_is_rows,
    int64_t *rows, int64_t *cols, double *values, RNG &rng, Sampler sampler
) {
    int64_t *major_indices = major_is_rows ? rows : cols;
    int64_t *minor_indices = major_is_rows ? cols : rows;
    sampler(n, num_vectors, vec_nnz, major_indices, rng);

    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        int64_t block_start = vector * vec_nnz;
        std::fill_n(minor_indices + block_start, vec_nnz, vector);
        for (int64_t entry = 0; entry < vec_nnz; ++entry) {
            int64_t offset = block_start + entry;
            values[offset] = (rng() & 1) == 0 ? 1.0 : -1.0;
        }

        for (int64_t entry = 1; entry < vec_nnz; ++entry) {
            int64_t key = major_indices[block_start + entry];
            double value = values[block_start + entry];
            int64_t cursor = entry - 1;
            while (cursor >= 0 && major_indices[block_start + cursor] > key) {
                major_indices[block_start + cursor + 1] =
                    major_indices[block_start + cursor];
                values[block_start + cursor + 1] = values[block_start + cursor];
                --cursor;
            }
            major_indices[block_start + cursor + 1] = key;
            values[block_start + cursor + 1] = value;
        }
    }
}

}  // namespace RandBLAS::testing
