// Copyright, 2024. See LICENSE for copyright holder information.
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
//

#pragma once

#include <numeric>
#include <vector>
#include <cmath>
#include <cstdint>

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"


namespace RandBLAS::sparse_data {

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif


namespace detail {

// Sequential wrapper around a Random123 CBRNG. Philox4x32 produces 4 uint32_t
// values per counter increment; this helper dispenses them one at a time and
// provides uniform, Gaussian, and geometric draws.
template <typename RNG = RandBLAS::DefaultRNG>
struct PhiloxStream {
    using ctr_t = typename RNG::ctr_type;
    using key_t = typename RNG::key_type;
    static constexpr int ctr_size = ctr_t::static_size;

    RNG rng;
    ctr_t counter;
    key_t key;
    ctr_t buffer;
    int pos;

    PhiloxStream(const RandBLAS::RNGState<RNG> &state)
        : counter(state.counter), key(state.key), pos(ctr_size) {}

    uint32_t next_u32() {
        if (pos >= ctr_size) {
            buffer = rng(counter, key);
            counter.incr();
            pos = 0;
        }
        return buffer.v[pos++];
    }

    // Uniform in (0, 1], never 0.0 (safe for log).
    double uniform_01() {
        return r123::u01<double>(next_u32());
    }

    // Box-Muller Gaussian. Consumes 2 uint32_t, returns one value (discards the other).
    template <typename T>
    T gaussian() {
        uint32_t u1 = next_u32();
        uint32_t u2 = next_u32();
        auto [g1, g2] = r123::boxmuller(u1, u2);
        (void)g2;
        return static_cast<T>(g1);
    }

    // Geometric distribution: number of failures before first success in Bernoulli(p).
    // Uses inverse CDF: floor(log(1 - u) / log(1 - p)) with u ~ Uniform(0, 1].
    // Since u01 returns (0, 1], we have 1 - u in [0, 1). The only problematic value
    // is 1 - u = 0, i.e., u = 1.0 exactly. With u01<double>(uint32_t), the maximum
    // is 1.0 - 2^-33, so this never happens.
    int64_t geometric(double log_1_minus_p) {
        double u = uniform_01();
        return static_cast<int64_t>(std::floor(std::log(1.0 - u) / log_1_minus_p));
    }

    RandBLAS::RNGState<RNG> get_state() const {
        return RandBLAS::RNGState<RNG>{counter, key};
    }
};

} // end namespace detail


// ============================================================================
// Generate a random m-by-n CSR matrix with approximately m*n*density nonzeros.
// Each potential entry is independently included with probability "density",
// using geometric skips for O(nnz + m) expected time.
// Nonzero values are iid standard Gaussian.
//
// The matrix A must be constructed with the desired (n_rows, n_cols) before
// calling this function. It must not have been reserved or populated yet.
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
RandBLAS::RNGState<RNG> random_csr(
    double density,
    CSRMatrix<T, sint_t> &A,
    const RandBLAS::RNGState<RNG> &state
) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    randblas_require(density >= 0.0 && density <= 1.0);

    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        if (m > 0) {
            A.rowptr = new sint_t[m + 1]{};
        }
        return stream.get_state();
    }

    if (density >= 1.0) {
        int64_t total = m * n;
        A.reserve(total);
        int64_t idx = 0;
        for (int64_t i = 0; i < m; ++i) {
            A.rowptr[i] = static_cast<sint_t>(idx);
            for (int64_t j = 0; j < n; ++j) {
                A.vals[idx]    = stream.template gaussian<T>();
                A.colidxs[idx] = static_cast<sint_t>(j);
                ++idx;
            }
        }
        A.rowptr[m] = static_cast<sint_t>(idx);
        return stream.get_state();
    }

    // General case: geometric skips for O(nnz + m) expected time.
    double log_1_minus_p = std::log(1.0 - density);

    std::vector<T>      vals_vec;
    std::vector<sint_t> colidxs_vec;
    std::vector<sint_t> rowptr_vec(m + 1);

    int64_t expected_nnz = static_cast<int64_t>(m * n * density * 1.2) + 16;
    vals_vec.reserve(expected_nnz);
    colidxs_vec.reserve(expected_nnz);

    for (int64_t i = 0; i < m; ++i) {
        rowptr_vec[i] = static_cast<sint_t>(vals_vec.size());
        int64_t j = stream.geometric(log_1_minus_p);
        while (j < n) {
            vals_vec.push_back(stream.template gaussian<T>());
            colidxs_vec.push_back(static_cast<sint_t>(j));
            j += 1 + stream.geometric(log_1_minus_p);
        }
    }
    rowptr_vec[m] = static_cast<sint_t>(vals_vec.size());

    int64_t nnz = static_cast<int64_t>(vals_vec.size());
    if (nnz == 0) {
        A.rowptr = new sint_t[m + 1]{};
        return stream.get_state();
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(),    vals_vec.end(),    A.vals);
    std::copy(colidxs_vec.begin(), colidxs_vec.end(), A.colidxs);
    std::copy(rowptr_vec.begin(),  rowptr_vec.end(),  A.rowptr);

    return stream.get_state();
}


// ============================================================================
// Generate a random m-by-n CSC matrix with approximately m*n*density nonzeros.
// Uses geometric skips for O(nnz + n) expected time.
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
RandBLAS::RNGState<RNG> random_csc(
    double density,
    CSCMatrix<T, sint_t> &A,
    const RandBLAS::RNGState<RNG> &state
) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    randblas_require(density >= 0.0 && density <= 1.0);

    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        if (n > 0) {
            A.colptr = new sint_t[n + 1]{};
        }
        return stream.get_state();
    }

    if (density >= 1.0) {
        int64_t total = m * n;
        A.reserve(total);
        int64_t idx = 0;
        for (int64_t j = 0; j < n; ++j) {
            A.colptr[j] = static_cast<sint_t>(idx);
            for (int64_t i = 0; i < m; ++i) {
                A.vals[idx]    = stream.template gaussian<T>();
                A.rowidxs[idx] = static_cast<sint_t>(i);
                ++idx;
            }
        }
        A.colptr[n] = static_cast<sint_t>(idx);
        return stream.get_state();
    }

    double log_1_minus_p = std::log(1.0 - density);

    std::vector<T>      vals_vec;
    std::vector<sint_t> rowidxs_vec;
    std::vector<sint_t> colptr_vec(n + 1);

    int64_t expected_nnz = static_cast<int64_t>(m * n * density * 1.2) + 16;
    vals_vec.reserve(expected_nnz);
    rowidxs_vec.reserve(expected_nnz);

    for (int64_t j = 0; j < n; ++j) {
        colptr_vec[j] = static_cast<sint_t>(vals_vec.size());
        int64_t i = stream.geometric(log_1_minus_p);
        while (i < m) {
            vals_vec.push_back(stream.template gaussian<T>());
            rowidxs_vec.push_back(static_cast<sint_t>(i));
            i += 1 + stream.geometric(log_1_minus_p);
        }
    }
    colptr_vec[n] = static_cast<sint_t>(vals_vec.size());

    int64_t nnz = static_cast<int64_t>(vals_vec.size());
    if (nnz == 0) {
        A.colptr = new sint_t[n + 1]{};
        return stream.get_state();
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(),    vals_vec.end(),    A.vals);
    std::copy(rowidxs_vec.begin(), rowidxs_vec.end(), A.rowidxs);
    std::copy(colptr_vec.begin(),  colptr_vec.end(),  A.colptr);

    return stream.get_state();
}


// ============================================================================
// Generate a random m-by-n COO matrix with approximately m*n*density nonzeros.
// Uses geometric skips on a linearized row-major index for O(nnz) expected time.
// The resulting entries are in CSR sort order.
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
RandBLAS::RNGState<RNG> random_coo(
    double density,
    COOMatrix<T, sint_t> &A,
    const RandBLAS::RNGState<RNG> &state
) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    randblas_require(density >= 0.0 && density <= 1.0);

    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        return stream.get_state();
    }

    int64_t total = m * n;

    if (density >= 1.0) {
        A.reserve(total);
        for (int64_t k = 0; k < total; ++k) {
            A.rows[k] = static_cast<sint_t>(k / n);
            A.cols[k] = static_cast<sint_t>(k % n);
            A.vals[k] = stream.template gaussian<T>();
        }
        A.sort = NonzeroSort::CSR;
        return stream.get_state();
    }

    double log_1_minus_p = std::log(1.0 - density);

    int64_t expected_nnz = static_cast<int64_t>(total * density * 1.2) + 16;
    std::vector<T>      vals_vec;
    std::vector<sint_t> rows_vec;
    std::vector<sint_t> cols_vec;
    vals_vec.reserve(expected_nnz);
    rows_vec.reserve(expected_nnz);
    cols_vec.reserve(expected_nnz);

    int64_t pos = stream.geometric(log_1_minus_p);
    while (pos < total) {
        rows_vec.push_back(static_cast<sint_t>(pos / n));
        cols_vec.push_back(static_cast<sint_t>(pos % n));
        vals_vec.push_back(stream.template gaussian<T>());
        pos += 1 + stream.geometric(log_1_minus_p);
    }

    int64_t nnz = static_cast<int64_t>(vals_vec.size());
    if (nnz == 0) {
        return stream.get_state();
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(), vals_vec.end(), A.vals);
    std::copy(rows_vec.begin(), rows_vec.end(), A.rows);
    std::copy(cols_vec.begin(), cols_vec.end(), A.cols);
    A.sort = NonzeroSort::CSR;

    return stream.get_state();
}


} // end namespace RandBLAS::sparse_data
