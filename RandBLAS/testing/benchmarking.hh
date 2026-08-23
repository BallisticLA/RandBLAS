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

#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace RandBLAS::testing {

// A parsed list and the result of applying the benchmark's positivity rule.
struct ThreadCountsParseResult {
    std::vector<int> thread_counts;
    bool valid;
};

// Parse and validate a comma-separated thread-count list. Empty fields are
// skipped, and an empty result is replaced with default_thread_counts.
inline ThreadCountsParseResult parse_thread_counts(const std::string &csv, const std::vector<int> &default_thread_counts) {
    std::vector<int> thread_counts;
    std::stringstream stream(csv);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            thread_counts.push_back(std::atoi(token.c_str()));
        }
    }
    if (thread_counts.empty()) {
        thread_counts = default_thread_counts;
    }
    const bool valid = std::all_of(
        thread_counts.begin(), thread_counts.end(),
        [](int thread_count) { return thread_count > 0; }
    );
    return {std::move(thread_counts), valid};
}

// Return the maximum OpenMP thread count, or one in a serial build.
inline int current_threads() {
#if defined(RandBLAS_HAS_OpenMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Disable dynamic teams and set the maximum OpenMP thread count. This is a
// no-op in a serial build.
inline void set_threads(int thread_count) {
#if defined(RandBLAS_HAS_OpenMP)
    omp_set_dynamic(0);
    omp_set_num_threads(thread_count);
#else
    (void) thread_count;
#endif
}

// Return the team size OpenMP actually provides for the requested count, or
// one in a serial build.
inline int effective_threads(int requested_threads) {
#if defined(RandBLAS_HAS_OpenMP)
    int actual_threads = 1;
    #pragma omp parallel num_threads(requested_threads)
    {
        #pragma omp single
        {
            actual_threads = omp_get_num_threads();
        }
    }
    return actual_threads;
#else
    (void) requested_threads;
    return 1;
#endif
}

// Restore the OpenMP dynamic-team setting and maximum thread count when a
// benchmark scope exits.
class OpenMPSettingsGuard {
public:
    OpenMPSettingsGuard() {
#if defined(RandBLAS_HAS_OpenMP)
        dynamic_ = omp_get_dynamic();
        threads_ = omp_get_max_threads();
#endif
    }

    ~OpenMPSettingsGuard() {
#if defined(RandBLAS_HAS_OpenMP)
        omp_set_num_threads(threads_);
        omp_set_dynamic(dynamic_);
#endif
    }

private:
#if defined(RandBLAS_HAS_OpenMP)
    int dynamic_;
    int threads_;
#endif
};

}  // namespace RandBLAS::testing
