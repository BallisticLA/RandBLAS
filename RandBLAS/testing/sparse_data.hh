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
#include <utility>
#include <vector>
#include <cmath>
#include <cstdint>

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/conversions.hh"


namespace RandBLAS::testing {

using blas::Layout;
using RandBLAS::sparse_data::CSRMatrix;
using RandBLAS::sparse_data::CSCMatrix;
using RandBLAS::sparse_data::COOMatrix;
using RandBLAS::sparse_data::NonzeroSort;

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif


namespace detail {

// Sequential wrapper around a Random123 CBRNG. Philox4x32 produces 4 uint32_t
// values per counter increment; this helper dispenses them one at a time and
// provides uniform, Gaussian, and geometric draws.
//
// Subclasses RNGState so that counter and key are inherited directly.
template <typename RNG = RandBLAS::DefaultRNG>
struct PhiloxStream : public RandBLAS::RNGState<RNG> {
    using state_t = RandBLAS::RNGState<RNG>;
    using ctr_t   = typename state_t::ctr_type;
    static constexpr int ctr_size = state_t::len_c;

    RNG rng;
    ctr_t buffer;
    int pos;
    double spare;
    bool has_spare;

    PhiloxStream(const state_t &state)
        : state_t(state), pos(ctr_size), spare(0.0), has_spare(false) {}

    uint32_t next_u32() {
        if (pos >= ctr_size) {
            buffer = rng(this->counter, this->key);
            this->counter.incr();
            pos = 0;
        }
        return buffer.v[pos++];
    }

    // Uniform in (0, 1], never 0.0 (safe for log).
    double uniform_01() {
        return r123::u01<double>(next_u32());
    }

    // Box-Muller Gaussian. Each call to boxmuller produces two independent values;
    // we cache the second and return it on the next invocation.
    template <typename T>
    T gaussian() {
        if (has_spare) {
            has_spare = false;
            return static_cast<T>(spare);
        }
        uint32_t u1 = next_u32();
        uint32_t u2 = next_u32();
        auto [g1, g2] = r123::boxmuller(u1, u2);
        spare = g2;
        has_spare = true;
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

    state_t get_state() const {
        return state_t{this->counter, this->key};
    }
};

} // end namespace detail


template <typename T, typename RNG = r123::Philox4x32>
void iid_sparsify_random_dense(
    int64_t n_rows,
    int64_t n_cols,
    int64_t stride_row,
    int64_t stride_col,
    T* mat,
    T prob_of_zero,
    RandBLAS::RNGState<RNG> state
) {
    auto spar = new T[n_rows * n_cols];
    auto dist = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::ScalarDist::Uniform);
    auto next_state = RandBLAS::fill_dense(dist, spar, state);

    auto temp = new T[n_rows * n_cols];
    auto D_mat = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::ScalarDist::Uniform);
    RandBLAS::fill_dense(D_mat, temp, next_state);

    // We'll pretend both of those matrices are column-major, regardless of the layout
    // value returned by fill_dense in each case.
    #define SPAR(_i, _j) spar[(_i) + (_j) * n_rows]
    #define TEMP(_i, _j) temp[(_i) + (_j) * n_rows]
    #define MAT(_i, _j)  mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T v = (SPAR(i, j) + 1.0) / 2.0;
            if (v < prob_of_zero) {
                MAT(i, j) = 0.0;
            } else {
                MAT(i, j) = TEMP(i, j);
            }
        }
    }

    delete [] spar;
    delete [] temp;
}


template <typename T, typename RNG = r123::Philox4x32>
void iid_sparsify_random_dense(
    int64_t n_rows,
    int64_t n_cols,
    Layout layout,
    T* mat,
    T prob_of_zero,
    RandBLAS::RNGState<RNG> state
) {
    if (layout == Layout::ColMajor) {
        iid_sparsify_random_dense(n_rows, n_cols, 1, n_rows, mat, prob_of_zero, state);
    } else {
        iid_sparsify_random_dense(n_rows, n_cols, n_cols, 1, mat, prob_of_zero, state);
    }
    return;
}

template <typename T>
void coo_from_diag(
    T* vals,
    int64_t nnz,
    int64_t offset,
    RandBLAS::sparse_data::COOMatrix<T> &spmat
) {
    spmat.reserve(nnz);
    int64_t ell = 0;
    if (offset >= 0) {
        randblas_require(nnz <= spmat.n_rows);
        while (ell < nnz) {
            spmat.rows[ell] = ell;
            spmat.cols[ell] = ell + offset;
            spmat.vals[ell] = vals[ell];
            ++ell;
        }
    } else {
        while (ell < nnz) {
            spmat.rows[ell] = ell - offset;
            spmat.cols[ell] = ell;
            spmat.vals[ell] = vals[ell];
            ++ell;
        }
    }
    return;
}

template <typename T>
int64_t trianglize_coo(
    RandBLAS::sparse_data::COOMatrix<T> &spmat,
    bool upper,
    RandBLAS::sparse_data::COOMatrix<T> &spmat_out
) {
    int64_t ell = 0;
    int64_t new_nnz = 0;
    while (ell < spmat.nnz) {
        if (upper && spmat.rows[ell] <= spmat.cols[ell] && spmat.vals[ell] != 0.0) {
            new_nnz += 1;
	} else if (!upper && spmat.rows[ell] >= spmat.cols[ell] && spmat.vals[ell] != 0.0) {
            new_nnz += 1;
  } else {
            spmat.vals[ell] = 0.0;
	}
	++ell;
    }
    spmat_out.reserve(new_nnz);
    ell = 0;
    int64_t ell_new = 0;
    while (ell < spmat.nnz) {
	if (spmat.vals[ell] != 0.0) {
	    spmat_out.rows[ell_new] = spmat.rows[ell];
	    spmat_out.cols[ell_new] = spmat.cols[ell];
	    spmat_out.vals[ell_new] = spmat.vals[ell];
	    ++ell_new;
	}
        ++ell;
    }
    return new_nnz;
}


// ============================================================================
// Generate a random m-by-n CSR matrix with approximately m*n*density nonzeros.
// Each potential entry is independently included with probability "density",
// using geometric skips for O(nnz + m) expected time.
// Nonzero values are iid standard Gaussian.
//
// Returns {CSRMatrix, next_state}. Use with structured bindings:
//     auto [A, next_state] = random_csr<double>(m, n, density, state);
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
std::pair<CSRMatrix<T, sint_t>, RandBLAS::RNGState<RNG>> random_csr(
    int64_t m,
    int64_t n,
    double density,
    const RandBLAS::RNGState<RNG> &state
) {
    randblas_require(density >= 0.0 && density <= 1.0);

    CSRMatrix<T, sint_t> A(m, n);
    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        if (m > 0) {
            A.rowptr = new sint_t[m + 1]{};
        }
        return {std::move(A), stream.get_state()};
    }

    if (density >= 1.0) {
        int64_t total = m * n;
        A.reserve(total);
        int64_t idx = 0;
        for (int64_t i = 0; i < m; ++i) {
            A.rowptr[i] = static_cast<sint_t>(idx);
            for (int64_t j = 0; j < n; ++j) {
                // ".template" disambiguates gaussian<T> as a template member
                // function call; without it the compiler may parse "<" as a
                // comparison operator since stream's type depends on RNG.
                A.vals[idx]    = stream.template gaussian<T>();
                A.colidxs[idx] = static_cast<sint_t>(j);
                ++idx;
            }
        }
        A.rowptr[m] = static_cast<sint_t>(idx);
        return {std::move(A), stream.get_state()};
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
        return {std::move(A), stream.get_state()};
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(),    vals_vec.end(),    A.vals);
    std::copy(colidxs_vec.begin(), colidxs_vec.end(), A.colidxs);
    std::copy(rowptr_vec.begin(),  rowptr_vec.end(),  A.rowptr);

    return {std::move(A), stream.get_state()};
}


// ============================================================================
// Generate a random m-by-n CSC matrix with approximately m*n*density nonzeros.
// Uses geometric skips for O(nnz + n) expected time.
//
// Returns {CSCMatrix, next_state}. Use with structured bindings:
//     auto [A, next_state] = random_csc<double>(m, n, density, state);
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
std::pair<CSCMatrix<T, sint_t>, RandBLAS::RNGState<RNG>> random_csc(
    int64_t m,
    int64_t n,
    double density,
    const RandBLAS::RNGState<RNG> &state
) {
    randblas_require(density >= 0.0 && density <= 1.0);

    CSCMatrix<T, sint_t> A(m, n);
    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        if (n > 0) {
            A.colptr = new sint_t[n + 1]{};
        }
        return {std::move(A), stream.get_state()};
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
        return {std::move(A), stream.get_state()};
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
        return {std::move(A), stream.get_state()};
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(),    vals_vec.end(),    A.vals);
    std::copy(rowidxs_vec.begin(), rowidxs_vec.end(), A.rowidxs);
    std::copy(colptr_vec.begin(),  colptr_vec.end(),  A.colptr);

    return {std::move(A), stream.get_state()};
}


// ============================================================================
// Generate a random m-by-n COO matrix with approximately m*n*density nonzeros.
// Uses geometric skips on a linearized row-major index for O(nnz) expected time.
// The resulting entries are in CSR sort order.
//
// Returns {COOMatrix, next_state}. Use with structured bindings:
//     auto [A, next_state] = random_coo<double>(m, n, density, state);
// ============================================================================
template <typename T, SignedInteger sint_t = int64_t, typename RNG = RandBLAS::DefaultRNG>
std::pair<COOMatrix<T, sint_t>, RandBLAS::RNGState<RNG>> random_coo(
    int64_t m,
    int64_t n,
    double density,
    const RandBLAS::RNGState<RNG> &state
) {
    randblas_require(density >= 0.0 && density <= 1.0);

    COOMatrix<T, sint_t> A(m, n);
    detail::PhiloxStream<RNG> stream(state);

    if (density == 0.0 || m == 0 || n == 0) {
        return {std::move(A), stream.get_state()};
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
        return {std::move(A), stream.get_state()};
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
        return {std::move(A), stream.get_state()};
    }

    A.reserve(nnz);
    std::copy(vals_vec.begin(), vals_vec.end(), A.vals);
    std::copy(rows_vec.begin(), rows_vec.end(), A.rows);
    std::copy(cols_vec.begin(), cols_vec.end(), A.cols);
    A.sort = NonzeroSort::CSR;

    return {std::move(A), stream.get_state()};
}


} // end namespace RandBLAS::testing
