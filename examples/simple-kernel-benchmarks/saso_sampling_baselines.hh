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
#include <ranges>
#include <vector>

namespace RandBLAS::benchmark {

class PhiloxURBG {
public:
    using result_type = uint64_t;

    explicit PhiloxURBG(uint64_t seed) : state_(seed) {}

    static constexpr result_type min() {
        return 0;
    }

    static constexpr result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() {
        typename RNGState<>::generator generator;
        auto random_values = generator(state_.counter, state_.key);
        state_.counter.incr();
        return RandBLAS::promote_uint_pair(random_values[0], random_values[1]);
    }

private:
    RNGState<> state_;
};

template <typename RNG>
void sample_std_sample(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    int64_t *samples,
    RNG &rng
) {
    auto population = std::views::iota(int64_t{0}, n);
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::sample(
            population.begin(),
            population.end(),
            samples + vector * vec_nnz,
            vec_nnz,
            rng
        );
    }
}

template <typename RNG>
void sample_partial_fisher_yates(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    int64_t *samples,
    RNG &rng
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

template <typename RNG>
void sample_full_shuffle(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    int64_t *samples,
    RNG &rng
) {
    std::vector<int64_t> population(n);
    std::iota(population.begin(), population.end(), int64_t{0});
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::shuffle(population.begin(), population.end(), rng);
        std::copy_n(
            population.begin(),
            vec_nnz,
            samples + vector * vec_nnz
        );
    }
}

template <typename RNG>
void sample_rejection(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    int64_t *samples,
    RNG &rng
) {
    std::uniform_int_distribution<int64_t> distribution(0, n - 1);
    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        int64_t *vector_samples = samples + vector * vec_nnz;
        for (int64_t entry = 0; entry < vec_nnz;) {
            int64_t candidate = distribution(rng);
            auto duplicate = std::find(
                vector_samples,
                vector_samples + entry,
                candidate
            );
            if (duplicate == vector_samples + entry) {
                vector_samples[entry] = candidate;
                ++entry;
            }
        }
    }
}

template <typename RNG>
void sample_floyd(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    int64_t *samples,
    RNG &rng
) {
    uint64_t table_size = 1;
    while (table_size < 2 * static_cast<uint64_t>(vec_nnz)) {
        table_size *= 2;
    }
    uint64_t table_mask = table_size - 1;
    std::vector<int64_t> table(table_size, -1);

    auto find_slot = [&table, table_mask](int64_t value) {
        constexpr uint64_t multiplier = 11400714819323198485ull;
        uint64_t slot = static_cast<uint64_t>(value) * multiplier & table_mask;
        while (table[slot] != -1 && table[slot] != value) {
            slot = (slot + 1) & table_mask;
        }
        return slot;
    };

    for (int64_t vector = 0; vector < num_vectors; ++vector) {
        std::fill(table.begin(), table.end(), -1);
        for (int64_t entry = 0; entry < vec_nnz; ++entry) {
            int64_t upper_bound = n - vec_nnz + entry;
            std::uniform_int_distribution<int64_t> pick(0, upper_bound);
            int64_t candidate = pick(rng);
            uint64_t candidate_slot = find_slot(candidate);
            int64_t selected = table[candidate_slot] == candidate
                ? upper_bound
                : candidate;
            uint64_t selected_slot = find_slot(selected);
            table[selected_slot] = selected;
            samples[vector * vec_nnz + entry] = selected;
        }
    }
}

template <typename RNG, typename Sampler>
void fill_saso_data(
    int64_t n,
    int64_t num_vectors,
    int64_t vec_nnz,
    bool major_is_rows,
    int64_t *rows,
    int64_t *cols,
    double *values,
    RNG &rng,
    Sampler sampler
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

}  // namespace RandBLAS::benchmark
