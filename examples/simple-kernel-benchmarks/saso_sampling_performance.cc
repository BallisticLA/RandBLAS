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

// ============================================================================
// SASO SAMPLING PERFORMANCE BENCHMARK
// ============================================================================
//
// This benchmark compares RandBLAS::repeated_fisher_yates against plausible
// C++ implementations for sampling the support of a SASO. If dim_major = n,
// num_major_axis_vectors = r, and each vector has vec_nnz = k nonzeros, every
// method produces r independent k-subsets of {0, ..., n - 1}.
//
// SUPPORT-ONLY METHODS:
//
//   * std::sample over an iota-filled vector
//   * partial Fisher-Yates with a fresh length-n iota per vector
//   * full std::shuffle followed by the first k entries
//   * uniform draws with linear duplicate rejection
//   * Floyd's algorithm with std::unordered_set
//   * RandBLAS::repeated_fisher_yates
//
// The first table is a natural-library comparison: the five alternatives use
// std::mt19937_64 and RandBLAS uses its native Philox path. The second table is
// a controlled-engine comparison: a URBG adapter exposes the same Philox stream
// to the C++ alternatives. The controlled table equalizes the generator, but
// not integer range mapping. RandBLAS currently uses a 64-bit value modulo the
// active range; standard algorithms and std::uniform_int_distribution use the
// C++ library's range mapping.
//
// The end-to-end tables also construct COO data: they write minor coordinates,
// generate Rademacher values, and sort every major-axis vector by its major
// coordinate. Both wide (major coordinates are rows) and tall (major
// coordinates are columns) SASOs are covered when the shape is nonsquare.
// Output allocation and correctness checks are outside the timed region.
// Method-owned workspace allocation remains timed, matching the natural cost
// of calling each implementation as written.
//
// METRICS:
//
//   * min and median wall time over repeated trials
//   * minimum-time nanoseconds per generated nonzero
//   * speedup relative to std::sample in the same table
//
// The comparison tables force RandBLAS to one OpenMP thread. Scaling mode times
// only RandBLAS because the competing implementations own one serial RNG
// engine. The k=1 RandBLAS row uses the library's specialized i.i.d.-uniform
// path rather than repeated Fisher-Yates.
//
// USAGE:
//
//   ./saso_sampling_performance [flags]
//   ./saso_sampling_performance [flags] n r k [trials]
//
//   flags:
//     --natural-only   skip the controlled-Philox tables
//     --support-only   skip end-to-end COO construction
//     --scaling        report RandBLAS thread scaling only
//     --threads=LIST   requested thread counts (default 1,2,4,8)
//     --help           print usage
//
// EXAMPLES:
//
//   ./saso_sampling_performance
//   ./saso_sampling_performance 256 4096 8 20
//   ./saso_sampling_performance --natural-only --support-only 1024 8192 8
//   ./saso_sampling_performance --scaling --threads=1,2,4,8 2000 100000 8 10
//
// ============================================================================

#include <RandBLAS.hh>
#include "RandBLAS/config.h"
#include "RandBLAS/testing/benchmarking.hh"
#include "RandBLAS/testing/samplers.hh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using RandBLAS::testing::current_threads;
using RandBLAS::testing::effective_threads;
using RandBLAS::testing::OpenMPSettingsGuard;
using RandBLAS::testing::set_threads;

// MARK: benchmark setup

struct Config {
    int64_t dim_major;
    int64_t num_major_axis_vectors;
    int64_t vec_nnz;
};

struct Record {
    std::string label;
    int64_t min_ns = 0;
    int64_t median_ns = 0;
    double ns_per_nonzero = -1.0;
    double speedup_vs_std_sample = -1.0;
    std::string notes;
};

struct ScalingRecord {
    int requested_threads;
    int threads;
    int64_t min_ns;
    int64_t median_ns;
    double ns_per_nonzero;
    double speedup;
    double efficiency;
};

template <typename Func>
static std::pair<int64_t, int64_t> run_trials(Func &&func, int64_t num_trials) {
    std::vector<int64_t> times;
    times.reserve(num_trials);
    for (int64_t trial = 0; trial < num_trials; ++trial) {
        auto start = std::chrono::steady_clock::now();
        func();
        auto end = std::chrono::steady_clock::now();
        times.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        );
    }
    std::sort(times.begin(), times.end());
    return {times.front(), times[num_trials / 2]};
}

static std::string format_cell(double value, int precision) {
    if (value < 0.0) {
        return "-";
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

static void print_table_header(const std::string &title) {
    std::cout << "  " << title << "\n";
    std::cout << "  " << std::left << std::setw(29) << "Implementation"
              << std::right << std::setw(13) << "Min(ns)"
              << std::setw(13) << "Median(ns)"
              << std::setw(13) << "ns/nonzero"
              << std::setw(12) << "vs sample"
              << "  notes\n";
    std::cout << "  " << std::string(105, '-') << "\n";
}

static void print_table_record(const Record &record) {
    std::cout << "  " << std::left << std::setw(29) << record.label
              << std::right << std::setw(13) << record.min_ns
              << std::setw(13) << record.median_ns
              << std::setw(13) << format_cell(record.ns_per_nonzero, 2)
              << std::setw(12) << format_cell(record.speedup_vs_std_sample, 2)
              << "  " << record.notes << "\n";
}

static void fill_speedups(std::vector<Record> &records) {
    if (records.empty() || records.front().min_ns <= 0) {
        return;
    }
    double baseline = static_cast<double>(records.front().min_ns);
    for (Record &record : records) {
        if (record.min_ns > 0) {
            record.speedup_vs_std_sample = baseline
                / static_cast<double>(record.min_ns);
        }
    }
}

// MARK: correctness checks

static bool support_is_valid(const Config &config, const std::vector<int64_t> &samples) {
    if (samples.size() != static_cast<std::size_t>(
            config.num_major_axis_vectors * config.vec_nnz
        )) {
        return false;
    }
    std::vector<bool> seen(config.dim_major, false);
    for (int64_t vector = 0; vector < config.num_major_axis_vectors; ++vector) {
        std::fill(seen.begin(), seen.end(), false);
        for (int64_t entry = 0; entry < config.vec_nnz; ++entry) {
            int64_t index = samples[vector * config.vec_nnz + entry];
            if (index < 0 || index >= config.dim_major || seen[index]) {
                return false;
            }
            seen[index] = true;
        }
    }
    return true;
}

static bool saso_data_is_valid(
    const Config &config, bool major_is_rows,
    const std::vector<int64_t> &rows, const std::vector<int64_t> &cols,
    const std::vector<double> &values
) {
    const std::vector<int64_t> &major = major_is_rows ? rows : cols;
    const std::vector<int64_t> &minor = major_is_rows ? cols : rows;
    if (!support_is_valid(config, major)) {
        return false;
    }
    for (int64_t vector = 0; vector < config.num_major_axis_vectors; ++vector) {
        for (int64_t entry = 0; entry < config.vec_nnz; ++entry) {
            int64_t offset = vector * config.vec_nnz + entry;
            if (minor[offset] != vector) {
                return false;
            }
            if (values[offset] != -1.0 && values[offset] != 1.0) {
                return false;
            }
            if (entry > 0 && major[offset - 1] >= major[offset]) {
                return false;
            }
        }
    }
    return true;
}

// MARK: support-only benchmarks

template <typename RNG, typename Sampler>
static Record benchmark_support_method(
    const std::string &label, const std::string &notes, const Config &config,
    int64_t num_trials, uint64_t seed, Sampler sampler
) {
    int64_t nnz = config.num_major_axis_vectors * config.vec_nnz;
    std::vector<int64_t> samples(nnz, -1);
    RNG rng(seed);
    auto sample = [&]() {
        sampler(
            config.dim_major, config.num_major_axis_vectors,
            config.vec_nnz, samples.data(), rng
        );
    };

    sample();
    bool valid = support_is_valid(config, samples);
    auto [min_ns, median_ns] = run_trials(sample, num_trials);

    Record record;
    record.label = label;
    record.min_ns = min_ns;
    record.median_ns = median_ns;
    record.ns_per_nonzero = static_cast<double>(min_ns) / static_cast<double>(nnz);
    record.notes = valid ? notes : "FAIL: invalid support; " + notes;
    return record;
}

static Record benchmark_randblas_support(
    const Config &config, int64_t num_trials, uint64_t seed
) {
    int64_t nnz = config.num_major_axis_vectors * config.vec_nnz;
    std::vector<int64_t> samples(nnz, -1);
    RandBLAS::RNGState<> state(seed);
    auto sample = [&]() {
        state = RandBLAS::repeated_fisher_yates(
            config.vec_nnz, config.dim_major, config.num_major_axis_vectors,
            samples.data(), state
        );
    };

    sample();
    bool valid = support_is_valid(config, samples);
    auto [min_ns, median_ns] = run_trials(sample, num_trials);

    Record record;
    record.label = "RandBLAS repeated FY";
    record.min_ns = min_ns;
    record.median_ns = median_ns;
    record.ns_per_nonzero = static_cast<double>(min_ns) / static_cast<double>(nnz);
    if (config.vec_nnz == 1) {
        record.notes = "O(r), specialized i.i.d. path";
    } else {
        record.notes = "O(n + r*k), restore k swaps";
    }
    if (!valid) {
        record.notes = "FAIL: invalid support; " + record.notes;
    }
    return record;
}

template <typename RNG>
static std::vector<Record> support_records(
    const Config &config, int64_t num_trials, uint64_t seed
) {
    using namespace RandBLAS::testing;
    std::vector<Record> records;
    records.push_back(benchmark_support_method<RNG>(
        "std::sample(iota vector)", "O(r*n), O(n) workspace",
        config, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_std_sample(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_support_method<RNG>(
        "partial FY + iota reset", "O(r*n), reset dominates for k << n",
        config, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_partial_fisher_yates(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_support_method<RNG>(
        "full std::shuffle", "O(r*n), take first k", config, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_full_shuffle(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_support_method<RNG>(
        "draw/reject + linear find", "O(r*k^2) expected when k << n",
        config, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_rejection(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_support_method<RNG>(
        "Floyd + unordered_set", "O(r*k) expected",
        config, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_floyd(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_randblas_support(config, num_trials, seed));
    fill_speedups(records);
    return records;
}

// MARK: end-to-end COO benchmarks

template <typename RNG, typename Sampler>
static Record benchmark_saso_data_method(
    const std::string &label, const std::string &notes, const Config &config,
    bool major_is_rows, int64_t num_trials, uint64_t seed, Sampler sampler
) {
    int64_t nnz = config.num_major_axis_vectors * config.vec_nnz;
    std::vector<int64_t> rows(nnz, -1);
    std::vector<int64_t> cols(nnz, -1);
    std::vector<double> values(nnz, 0.0);
    RNG rng(seed);
    auto sample = [&]() {
        RandBLAS::testing::fill_saso_data(
            config.dim_major, config.num_major_axis_vectors, config.vec_nnz,
            major_is_rows, rows.data(), cols.data(), values.data(), rng, sampler
        );
    };

    sample();
    bool valid = saso_data_is_valid(config, major_is_rows, rows, cols, values);
    auto [min_ns, median_ns] = run_trials(sample, num_trials);

    Record record;
    record.label = label;
    record.min_ns = min_ns;
    record.median_ns = median_ns;
    record.ns_per_nonzero = static_cast<double>(min_ns) / static_cast<double>(nnz);
    record.notes = valid ? notes : "FAIL: invalid COO data; " + notes;
    return record;
}

static Record benchmark_randblas_saso_data(
    const Config &config, bool major_is_rows, int64_t num_trials, uint64_t seed
) {
    int64_t nnz = config.num_major_axis_vectors * config.vec_nnz;
    int64_t n_rows = major_is_rows
        ? config.dim_major
        : config.num_major_axis_vectors;
    int64_t n_cols = major_is_rows
        ? config.num_major_axis_vectors
        : config.dim_major;
    std::vector<int64_t> rows(nnz, -1);
    std::vector<int64_t> cols(nnz, -1);
    std::vector<double> values(nnz, 0.0);
    RandBLAS::SparseDist distribution(
        n_rows, n_cols, config.vec_nnz, RandBLAS::Axis::Short
    );
    RandBLAS::RNGState<> state(seed);
    int64_t sampled_nnz = nnz;
    auto sample = [&]() {
        sampled_nnz = nnz;
        state = RandBLAS::fill_sparse_unpacked(
            distribution, n_rows, n_cols, 0, 0, sampled_nnz,
            values.data(), rows.data(), cols.data(), state
        );
    };

    sample();
    bool valid = sampled_nnz == nnz
        && saso_data_is_valid(config, major_is_rows, rows, cols, values);
    auto [min_ns, median_ns] = run_trials(sample, num_trials);

    Record record;
    record.label = "RandBLAS fill unpacked";
    record.min_ns = min_ns;
    record.median_ns = median_ns;
    record.ns_per_nonzero = static_cast<double>(min_ns) / static_cast<double>(nnz);
    record.notes = config.vec_nnz == 1
        ? "fused support/sign; i.i.d. path; sorted"
        : "fused support/sign; restore swaps; sorted";
    if (!valid) {
        record.notes = "FAIL: invalid COO data; " + record.notes;
    }
    return record;
}

template <typename RNG>
static std::vector<Record> saso_data_records(
    const Config &config, bool major_is_rows, int64_t num_trials, uint64_t seed
) {
    using namespace RandBLAS::testing;
    std::vector<Record> records;
    records.push_back(benchmark_saso_data_method<RNG>(
        "std::sample(iota vector)", "support + minor + sign + sort",
        config, major_is_rows, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_std_sample(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_saso_data_method<RNG>(
        "partial FY + iota reset", "support + minor + sign + sort",
        config, major_is_rows, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_partial_fisher_yates(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_saso_data_method<RNG>(
        "full std::shuffle", "support + minor + sign + sort",
        config, major_is_rows, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_full_shuffle(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_saso_data_method<RNG>(
        "draw/reject + linear find", "support + minor + sign + sort",
        config, major_is_rows, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_rejection(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_saso_data_method<RNG>(
        "Floyd + unordered_set", "support + minor + sign + sort",
        config, major_is_rows, num_trials, seed,
        [](int64_t n, int64_t r, int64_t k, int64_t *output, auto &rng) {
            sample_floyd(n, r, k, output, rng);
        }
    ));
    records.push_back(benchmark_randblas_saso_data(
        config, major_is_rows, num_trials, seed
    ));
    fill_speedups(records);
    return records;
}

static void print_records(
    const std::string &title, const std::vector<Record> &records
) {
    print_table_header(title);
    for (const Record &record : records) {
        print_table_record(record);
    }
    std::cout << "\n";
}

// MARK: thread scaling

static bool run_scaling(
    const Config &config, int64_t num_trials, const std::vector<int> &thread_counts
) {
    constexpr uint64_t seed = 12345;
    const int64_t nnz = config.num_major_axis_vectors * config.vec_nnz;
    std::vector<int64_t> samples(nnz, -1);
    std::vector<int64_t> expected_samples;
    RandBLAS::RNGState<> expected_state(seed);
    std::vector<ScalingRecord> records;
    records.reserve(thread_counts.size());
    bool exact = true;
    int64_t baseline_ns = 0;
    int baseline_threads = 1;

    for (int thread_count : thread_counts) {
        set_threads(thread_count);
        const int policy_threads = RandBLAS::sparse::sparse_sampling_thread_count(
            config.dim_major, config.num_major_axis_vectors,
            config.vec_nnz, config.vec_nnz > 1
        );
        const int actual_threads = effective_threads(policy_threads);
        RandBLAS::RNGState<> state(seed);
        auto end_state = RandBLAS::repeated_fisher_yates(
            config.vec_nnz, config.dim_major, config.num_major_axis_vectors,
            samples.data(), state
        );
        exact = exact && support_is_valid(config, samples);
        if (records.empty()) {
            expected_samples = samples;
            expected_state = end_state;
        } else {
            exact = exact
                && samples == expected_samples
                && end_state == expected_state;
        }

        auto [min_ns, median_ns] = run_trials([&]() {
            RandBLAS::RNGState<> trial_state(seed);
            end_state = RandBLAS::repeated_fisher_yates(
                config.vec_nnz, config.dim_major, config.num_major_axis_vectors,
                samples.data(), trial_state
            );
        }, num_trials);
        if (records.empty()) {
            baseline_ns = min_ns;
            baseline_threads = actual_threads;
        }
        const double speedup = min_ns > 0
            ? static_cast<double>(baseline_ns) / static_cast<double>(min_ns)
            : -1.0;
        const double relative_threads = static_cast<double>(actual_threads)
            / static_cast<double>(baseline_threads);
        records.push_back({
            thread_count, actual_threads, min_ns, median_ns,
            static_cast<double>(min_ns) / static_cast<double>(nnz),
            speedup, speedup / relative_threads
        });
    }

    std::cout << "=== RANDBLAS THREAD SCALING: n=" << config.dim_major
              << " r=" << config.num_major_axis_vectors
              << " k=" << config.vec_nnz
              << ", trials=" << num_trials << " ===\n";
#if !defined(RandBLAS_HAS_OpenMP)
    std::cout << "  (built without OpenMP -- thread sweep is a no-op)\n";
#endif
    std::cout << "\n"
              << "  " << std::right << std::setw(7) << "Request"
              << std::setw(8) << "Threads"
              << std::setw(13) << "Min(ns)"
              << std::setw(13) << "Median(ns)"
              << std::setw(13) << "ns/nonzero"
              << std::setw(11) << "Spd(min)"
              << std::setw(11) << "Eff(min)" << "\n"
              << "  " << std::string(74, '-') << "\n";
    for (const ScalingRecord &record : records) {
        std::cout << "  " << std::right << std::setw(7) << record.requested_threads
                  << std::setw(8) << record.threads
                  << std::setw(13) << record.min_ns
                  << std::setw(13) << record.median_ns
                  << std::setw(13) << format_cell(record.ns_per_nonzero, 2)
                  << std::setw(11) << format_cell(record.speedup, 2)
                  << std::setw(11) << format_cell(record.efficiency, 2) << "\n";
    }
    std::cout << "\n  Exact output/state check: "
              << (exact ? "PASS" : "FAIL") << "\n\n";
    return exact;
}

static void run_support_tables(const Config &config, int64_t num_trials, bool include_controlled) {
    constexpr uint64_t seed = 12345;
    std::cout << "=== SUPPORT ONLY: n=" << config.dim_major
              << " r=" << config.num_major_axis_vectors
              << " k=" << config.vec_nnz
              << ", trials=" << num_trials << " ===\n\n";
    print_records(
        "natural C++ RNGs: std::mt19937_64 versus native RandBLAS Philox",
        support_records<std::mt19937_64>(config, num_trials, seed)
    );
    if (include_controlled) {
        print_records(
            "controlled engine: Philox for every implementation",
            support_records<RandBLAS::testing::PhiloxURBG>(config, num_trials, seed)
        );
    }
}

static void run_saso_data_tables(
    const Config &config, bool major_is_rows, int64_t num_trials, bool include_controlled
) {
    constexpr uint64_t seed = 67890;
    const char *shape = major_is_rows
        ? "wide SASO (major coordinates are rows)"
        : "tall SASO (major coordinates are columns)";
    std::cout << "=== END-TO-END COO: " << shape
              << ", n=" << config.dim_major
              << " r=" << config.num_major_axis_vectors
              << " k=" << config.vec_nnz
              << ", trials=" << num_trials << " ===\n\n";
    print_records(
        "natural C++ RNGs: support, minor coordinates, signs, and sorting",
        saso_data_records<std::mt19937_64>(
            config, major_is_rows, num_trials, seed
        )
    );
    if (include_controlled) {
        print_records(
            "controlled engine: support, minor coordinates, signs, and sorting",
            saso_data_records<RandBLAS::testing::PhiloxURBG>(
                config, major_is_rows, num_trials, seed
            )
        );
    }
}

// MARK: command-line interface

static bool config_is_valid(const Config &config, int64_t num_trials) {
    return config.dim_major > 0
        && config.num_major_axis_vectors >= config.dim_major
        && config.vec_nnz > 0
        && config.vec_nnz <= config.dim_major
        && num_trials > 0;
}

static std::vector<int> parse_threads(const std::string &csv) {
    std::vector<int> thread_counts;
    std::stringstream stream(csv);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            thread_counts.push_back(std::atoi(token.c_str()));
        }
    }
    if (thread_counts.empty()) {
        thread_counts = {1, 2, 4, 8};
    }
    return thread_counts;
}

static bool thread_counts_are_valid(const std::vector<int> &thread_counts) {
    return std::all_of(
        thread_counts.begin(), thread_counts.end(),
        [](int thread_count) { return thread_count > 0; }
    );
}

static void print_usage(const char *program) {
    std::cout << "Usage:\n"
              << "  " << program << " [flags]\n"
              << "  " << program << " [flags] n r k [trials]\n\n"
              << "Constraints: 0 < k <= n <= r and trials > 0.\n"
              << "Flags: --natural-only, --support-only, --scaling, "
              << "--threads=1,2,4,8, --help\n";
}

int main(int argc, char **argv) {
    OpenMPSettingsGuard openmp_settings;
    bool include_controlled = true;
    bool support_only = false;
    bool scaling = false;
    std::vector<int> thread_counts{1, 2, 4, 8};
    std::vector<std::string> positional;
    for (int arg = 1; arg < argc; ++arg) {
        std::string value = argv[arg];
        if (value == "--natural-only") {
            include_controlled = false;
        } else if (value == "--support-only") {
            support_only = true;
        } else if (value == "--scaling") {
            scaling = true;
        } else if (value.rfind("--threads=", 0) == 0) {
            thread_counts = parse_threads(value.substr(10));
        } else if (value == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (value.rfind("--", 0) == 0) {
            std::cerr << "Unknown flag: " << value << "\n";
            print_usage(argv[0]);
            return 1;
        } else {
            positional.push_back(value);
        }
    }

    if (!positional.empty() && positional.size() != 3 && positional.size() != 4) {
        print_usage(argv[0]);
        return 1;
    }
    if (!thread_counts_are_valid(thread_counts)) {
        std::cerr << "Invalid thread list. Expected positive integers.\n";
        return 1;
    }
    if (!scaling) {
        set_threads(1);
    }

    std::cout << "\n============================================================\n"
              << "SASO SAMPLING PERFORMANCE BENCHMARK\n"
              << "============================================================\n";
    if (scaling) {
        std::cout << "RandBLAS-only thread scaling; output allocation is not timed.\n"
                  << "Configured OpenMP maximum: " << current_threads() << "\n\n";
    } else {
        std::cout << "Comparison mode; allocations for output arrays are not timed.\n"
                  << "RandBLAS is forced to one OpenMP thread.\n"
                  << "Speedup is relative to std::sample in the same table.\n\n";
    }

    if (!positional.empty()) {
        Config config{
            std::atoll(positional[0].c_str()), std::atoll(positional[1].c_str()),
            std::atoll(positional[2].c_str())
        };
        int64_t num_trials = positional.size() == 4
            ? std::atoll(positional[3].c_str())
            : 10;
        if (!config_is_valid(config, num_trials)) {
            std::cerr << "Invalid configuration. Expected 0 < k <= n <= r "
                      << "and trials > 0.\n";
            return 1;
        }
        if (scaling) {
            return run_scaling(config, num_trials, thread_counts) ? 0 : 2;
        }
        run_support_tables(config, num_trials, include_controlled);
        if (!support_only) {
            run_saso_data_tables(
                config, true, num_trials, include_controlled
            );
            if (config.num_major_axis_vectors > config.dim_major) {
                run_saso_data_tables(
                    config, false, num_trials, include_controlled
                );
            }
        }
        return 0;
    }

    constexpr int64_t num_trials = 5;
    std::vector<Config> support_configs{
        {64, 4096, 8},
        {256, 4096, 1},
        {256, 4096, 4},
        {256, 4096, 16},
        {1024, 4096, 8},
        {1024, 4096, 64},
        {4096, 4096, 8},
    };
    if (scaling) {
        bool exact = true;
        for (const Config &config : support_configs) {
            exact = run_scaling(config, num_trials, thread_counts) && exact;
        }
        return exact ? 0 : 2;
    }
    for (const Config &config : support_configs) {
        run_support_tables(config, num_trials, include_controlled);
    }

    if (!support_only) {
        Config data_config{256, 4096, 8};
        run_saso_data_tables(
            data_config, true, num_trials, include_controlled
        );
        run_saso_data_tables(
            data_config, false, num_trials, include_controlled
        );
    }
    return 0;
}
