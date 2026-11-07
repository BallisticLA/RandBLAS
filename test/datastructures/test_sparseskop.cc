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

#include <RandBLAS/dense_skops.hh>
#include <RandBLAS/sparse_skops.hh>
#include <RandBLAS/util.hh>
#include "RandBLAS/testing/comparison.hh"

#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

using std::vector;
using RandBLAS::RNGState;

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif

using RandBLAS::SparseDist;
using RandBLAS::SparseSkOp;
using RandBLAS::Axis;
using RandBLAS::SparseBlocking;
using RandBLAS::fill_sparse;
using RandBLAS::fill_sparse_unpacked_nosub;

// A sparse sampling call has five observable outputs: the number of stored
// entries, three COO arrays, and the RNG state that follows the sample. Keep
// them together so the parallel tests can compare a complete result rather
// than checking only the sampled coordinates.
template <typename T, typename sint_t = int64_t>
struct SparseSnapshot {
    int64_t nnz;
    std::vector<T> vals;
    std::vector<sint_t> rows;
    std::vector<sint_t> cols;
    RandBLAS::RNGState<> end_state;
};

// Allocate the largest COO buffers permitted by the requested window, sample
// the window, and trim those buffers to the number of entries actually written.
// Trimming matters for LASOs, where restricting a sampled vector to a window
// can remove entries and leave the output shorter than its upper bound.
template <typename T, typename sint_t = int64_t>
static SparseSnapshot<T, sint_t> sample_sparse_snapshot(
    const RandBLAS::SparseDist &dist, int64_t n_rows_sub, int64_t n_cols_sub,
    int64_t row_offset, int64_t col_offset,
    const RandBLAS::RNGState<> &seed
) {
    const bool short_is_rows = dist.n_rows <= dist.n_cols;
    const int64_t short_sub = short_is_rows ? n_rows_sub : n_cols_sub;
    const int64_t long_sub = short_is_rows ? n_cols_sub : n_rows_sub;
    const int64_t num_vectors = dist.major_axis == RandBLAS::Axis::Short
        ? long_sub
        : short_sub;
    const int64_t capacity = dist.vec_nnz * num_vectors;
    SparseSnapshot<T, sint_t> result{
        -1, std::vector<T>(capacity), std::vector<sint_t>(capacity),
        std::vector<sint_t>(capacity), seed
    };
    result.end_state = RandBLAS::fill_sparse_unpacked(
        dist, n_rows_sub, n_cols_sub, row_offset, col_offset, result.nnz,
        result.vals.data(), result.rows.data(), result.cols.data(), seed
    );
    result.vals.resize(result.nnz);
    result.rows.resize(result.nnz);
    result.cols.resize(result.nnz);
    return result;
}


class TestSparseSkOpConstruction : public ::testing::Test {
    protected:
        std::vector<uint32_t> keys{42, 0, 1};
        std::vector<int64_t> vec_nnzs{(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 7};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename SKOP>
    void check_fixed_nnz_per_col(SKOP &S0) {
        using sint_t = typename SKOP::index_t;
        std::set<sint_t> s;
        for (int64_t i = 0; i < S0.dist.n_cols; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                sint_t row = S0.rows[offset + j];
                ASSERT_EQ(s.count(row), 0) << "row index " << row << " was duplicated in column " << i << std::endl;
                s.insert(row);
            }
        }
    }

    template <typename SKOP>
    void check_fixed_nnz_per_row(SKOP &S0) {
        using sint_t = typename SKOP::index_t;
        std::set<sint_t> s;
        for (int64_t i = 0; i < S0.dist.n_rows; ++i) {
            int64_t offset = S0.dist.vec_nnz * i;
            s.clear();
            for (int64_t j = 0; j < S0.dist.vec_nnz; ++j) {
                sint_t col = S0.cols[offset + j];
                ASSERT_EQ(s.count(col), 0)  << "column index " << col << " was duplicated in row " << i << std::endl;
                s.insert(col);
            }
        }
    }

    template <SignedInteger sint_t>
    void proper_saso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        using RNG = SparseSkOp<float>::state_t::generator;
        SparseDist D0(d, m, vec_nnzs[nnz_index], Axis::Short);
        SparseSkOp<float, RNG, sint_t> S0(D0, keys[key_index]);
        fill_sparse(S0);
        if (d < m) {
            check_fixed_nnz_per_col(S0);
        } else {
            check_fixed_nnz_per_row(S0);
        }
    }

    template <SignedInteger sint_t>
    void proper_laso_construction(int64_t d, int64_t m, int64_t key_index) {
        using RNG = SparseSkOp<float>::state_t::generator;
        int64_t vec_nnz = 1;
        SparseDist D0(d, m, vec_nnz, Axis::Long);
        SparseSkOp<float, RNG, sint_t> S0(D0, keys[key_index]);
        fill_sparse(S0);
        if (d < m) {
            check_fixed_nnz_per_row(S0);
        } else {
            check_fixed_nnz_per_col(S0);
        } 
    }

    template <SignedInteger sint_t, typename T = float>
    void respect_ownership(int64_t d, int64_t m) {
        RNGState state(0);
        SparseDist sd(d, m, 2, Axis::Short);

        std::vector<sint_t> rows(sd.full_nnz, -1);
        std::vector<sint_t> cols(sd.full_nnz, -1);
        std::vector<T> vals(sd.full_nnz, -0.5);
        auto rows_copy = rows;
        auto cols_copy = cols;
        auto vals_copy = vals;

        auto next_state = state; // it's safe to pass in a nonsense value, since we aren't going reference this again.
        auto S = new SparseSkOp(sd, state, next_state, -1, vals.data(), rows.data(), cols.data());
        // check that nothing has changed
        std::string msg;
        msg = RandBLAS::testing::buffs_approx_equal(rows.data(), rows_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(cols.data(), cols_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(vals.data(), vals_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (T) 0, (T) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        fill_sparse(*S);
        rows_copy = rows;
        cols_copy = cols;
        vals_copy = vals;
        // check that everything has been overwritten
        for (int i = 0; i < sd.full_nnz; ++i) {
            EXPECT_GE(rows[i], 0);
            EXPECT_GE(cols[i], 0);
            EXPECT_NE(vals[i], -0.5);
        }
        // delete S, and make sure the rows,cols,vals are unchanged from before the deletion.
        delete S;
        msg = RandBLAS::testing::buffs_approx_equal(rows.data(), rows_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(cols.data(), cols_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(vals.data(), vals_copy.data(), sd.full_nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__, (T) 0, (T) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    void unpacked_nosub(const SparseDist &D) {
        RNGState<RandBLAS::DefaultRNG> s(1);
        SparseSkOp<float> S(D, s);
        auto expect_next = S.next_state;
        fill_sparse(S);
        vector<int64_t> rows(D.full_nnz);
        vector<int64_t> cols(D.full_nnz);
        vector<float>   vals(D.full_nnz);
        int64_t nnz = 0;
        auto actual_next = fill_sparse_unpacked_nosub(
            D, nnz, vals.data(), rows.data(), cols.data(), s
        );
        EXPECT_EQ(S.nnz, nnz);
        EXPECT_TRUE(actual_next == expect_next);
        std::string msg;
        msg = RandBLAS::testing::buffs_approx_equal(
            vals.data(), S.vals, nnz, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(
            rows.data(), S.rows, nnz,  __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(
            cols.data(), S.cols, nnz,  __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    // Equivalence oracle for submatrix sampling: for a given (ro_s, co_s, sub dims),
    // fill_sparse_unpacked(...) must equal -- entry-for-entry as a sparse matrix -- the
    // corresponding submatrix of the FULL operator from fill_sparse_unpacked_nosub(...).
    // We compare by densifying both into dense n_rows_sub-by-n_cols_sub buffers, which is
    // order-independent (important since SASO/LASO entry order may differ).
    template <typename T, SignedInteger sint_t = int64_t>
    void fill_unpacked_sub_one(
        Axis major_axis, SparseBlocking blocking, int64_t vec_nnz, int64_t d, int64_t m,
        int64_t ro_s, int64_t co_s, int64_t n_rows_sub, int64_t n_cols_sub, uint32_t key
    ) {
        SparseDist D {d, m, vec_nnz, major_axis, blocking};
        RNGState<RandBLAS::DefaultRNG> seed(key);

        // Full operator via the no-submatrix path.
        int64_t full_nnz_out = -1;
        vector<T> full_vals(D.full_nnz);
        vector<sint_t> full_rows(D.full_nnz);
        vector<sint_t> full_cols(D.full_nnz);
        fill_sparse_unpacked_nosub(
            D, full_nnz_out, full_vals.data(), full_rows.data(), full_cols.data(), seed
        );

        // Submatrix via the new path.
        // Worst-case nnz = vec_nnz * (#major-axis vectors intersecting the submatrix).
        bool short_is_rows = (d <= m);
        int64_t long_sub  = short_is_rows ? n_cols_sub : n_rows_sub;
        int64_t short_sub = short_is_rows ? n_rows_sub : n_cols_sub;
        int64_t minor_sub = (major_axis == Axis::Short) ? long_sub : short_sub;
        int64_t cap = vec_nnz * minor_sub;
        int64_t sub_nnz_out = -1;
        vector<T> sub_vals(cap > 0 ? cap : 1);
        vector<sint_t> sub_rows(cap > 0 ? cap : 1);
        vector<sint_t> sub_cols(cap > 0 ? cap : 1);
        RandBLAS::fill_sparse_unpacked(
            D, n_rows_sub, n_cols_sub, ro_s, co_s,
            sub_nnz_out, sub_vals.data(), sub_rows.data(), sub_cols.data(), seed
        );
        ASSERT_GE(sub_nnz_out, 0);
        ASSERT_LE(sub_nnz_out, cap);

        // Densify the oracle (restrict-after-sampling).
        vector<T> dense_oracle(n_rows_sub * n_cols_sub, (T) 0);
        for (int64_t i = 0; i < full_nnz_out; ++i) {
            int64_t r = full_rows[i];
            int64_t c = full_cols[i];
            if (ro_s <= r && r < ro_s + n_rows_sub && co_s <= c && c < co_s + n_cols_sub) {
                dense_oracle[(r - ro_s) * n_cols_sub + (c - co_s)] += full_vals[i];
            }
        }
        // Densify the submatrix (restrict-while-sampling).
        vector<T> dense_sub(n_rows_sub * n_cols_sub, (T) 0);
        for (int64_t i = 0; i < sub_nnz_out; ++i) {
            int64_t r = sub_rows[i];
            int64_t c = sub_cols[i];
            ASSERT_GE(r, 0); ASSERT_LT(r, n_rows_sub);
            ASSERT_GE(c, 0); ASSERT_LT(c, n_cols_sub);
            dense_sub[r * n_cols_sub + c] += sub_vals[i];
        }
        std::string msg = RandBLAS::testing::buffs_approx_equal(
            dense_sub.data(), dense_oracle.data(), n_rows_sub * n_cols_sub,
            __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    template <typename T, SignedInteger sint_t = int64_t>
    void fill_unpacked_sub(Axis major_axis, SparseBlocking blocking, int64_t vec_nnz, int64_t d, int64_t m) {
        for (auto key : keys) {
            // full-size (regression guard: must reproduce nosub exactly)
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, 0, d, m, key);
            // corner offsets / partial windows
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 1, 0, d - 1, m,     key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, 1, d,     m - 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 1, 1, d - 1, m - 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 2, 3, d - 3, m - 4, key);
            // single row / single column
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, 0, 1, m, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, 0, d, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, d - 1, 0, 1, m, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, m - 1, d, 1, key);
            // 1x1 windows at a few positions (some likely empty for SASO)
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, 0, 0, 1, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, d - 1, m - 1, 1, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, blocking, vec_nnz, d, m, d / 2, m / 2, 1, 1, key);
        }
        return;
    }

};

TEST_F(TestSparseSkOpConstruction, submatrix_sampling_rejects_invalid_windows) {
    // Submatrix validation must reject each negative argument and every window
    // that extends past the parent operator. Exercise workspace-query mode so
    // the check has to run before size arithmetic, allocation, RNG counter
    // updates, or output writes. The near-INT64_MAX offsets specifically catch
    // addition-form bounds checks that can overflow before making a decision.
    struct Window {
        int64_t n_rows;
        int64_t n_cols;
        int64_t row_offset;
        int64_t col_offset;
    };
    constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
    constexpr std::array<Window, 10> invalid_windows{{
        {-1, 5, 0, 0}, {5, -1, 0, 0}, {5, 5, -1, 0}, {5, 5, 0, -1},
        {8, 5, 0, 0}, {5, 14, 0, 0}, {7, 5, 1, 0}, {5, 13, 0, 1},
        {1, 1, max_int64, 0}, {1, 1, 0, max_int64}
    }};
    RandBLAS::SparseDist dist(7, 13, 2, RandBLAS::Axis::Short);
    RandBLAS::RNGState<> seed(314);
    RandBLAS::SparseSkOp<double> S(dist, seed);

    for (const Window &window : invalid_windows) {
        int64_t capacity = -1;
        EXPECT_THROW(
            RandBLAS::fill_sparse_unpacked(
                dist, window.n_rows, window.n_cols,
                window.row_offset, window.col_offset, capacity,
                static_cast<double *>(nullptr),
                static_cast<int64_t *>(nullptr),
                static_cast<int64_t *>(nullptr), seed
            ),
            RandBLAS::Error
        );
        EXPECT_THROW(
            RandBLAS::sparse::submatrix_as_coo(
                S, window.n_rows, window.n_cols,
                window.row_offset, window.col_offset
            ),
            RandBLAS::Error
        );
    }
}

TEST_F(TestSparseSkOpConstruction, submatrix_sampling_accepts_empty_boundary_window) {
    // A zero-by-zero window at the lower-right corner is inside the parent
    // operator. Verify that both the workspace query and the owning-COO helper
    // accept it, report no entries, and leave caller-owned sentinel storage
    // untouched when the sampling overload receives nonnull buffers.
    RandBLAS::SparseDist dist(7, 13, 2, RandBLAS::Axis::Short);
    RandBLAS::RNGState<> seed(2718);
    int64_t capacity = -1;
    RandBLAS::fill_sparse_unpacked(
        dist, 0, 0, 7, 13, capacity,
        static_cast<double *>(nullptr),
        static_cast<int64_t *>(nullptr),
        static_cast<int64_t *>(nullptr), seed
    );
    EXPECT_EQ(capacity, 0);

    double val = 1.25;
    int64_t row = 23;
    int64_t col = 29;
    int64_t nnz = -1;
    RandBLAS::fill_sparse_unpacked(
        dist, 0, 0, 7, 13, nnz, &val, &row, &col, seed
    );
    EXPECT_EQ(nnz, 0);
    EXPECT_EQ(val, 1.25);
    EXPECT_EQ(row, 23);
    EXPECT_EQ(col, 29);

    RandBLAS::SparseSkOp<double> S(dist, seed);
    auto empty = RandBLAS::sparse::submatrix_as_coo(S, 0, 0, 7, 13);
    EXPECT_EQ(empty.n_rows, 0);
    EXPECT_EQ(empty.n_cols, 0);
    EXPECT_EQ(empty.nnz, 0);
}

#if defined(RandBLAS_HAS_OpenMP)
TEST_F(TestSparseSkOpConstruction, sampling_is_thread_count_independent) {
    // Changing the OpenMP thread count must not change a sampled sparse
    // operator. For SASOs and LASOs of several shapes and sparsity levels, use
    // a serial full-operator sample and a serial submatrix sample as reference
    // results. Repeat each call with one, two, and four threads, then compare
    // every COO array, the number of entries written, and the returned RNG
    // state. A nonzero initial counter also checks that work is partitioned
    // relative to the supplied state rather than an implicit zero state.
    struct Case {
        int64_t n_rows;
        int64_t n_cols;
        int64_t vec_nnz;
        RandBLAS::Axis major_axis;
        SparseBlocking blocking;
    };
    constexpr auto B = SparseBlocking::Blocked;
    constexpr auto U = SparseBlocking::Unblocked;
    constexpr std::array<Case, 16> cases{{
        {7, 31, 1, RandBLAS::Axis::Short, B},
        {7, 31, 5, RandBLAS::Axis::Short, B},
        {31, 7, 5, RandBLAS::Axis::Short, B},
        {257, 2048, 1, RandBLAS::Axis::Short, B},
        {257, 2048, 12, RandBLAS::Axis::Short, B},
        {2048, 257, 12, RandBLAS::Axis::Short, B},
        {7, 31, 1, RandBLAS::Axis::Short, U},
        {7, 31, 5, RandBLAS::Axis::Short, U},
        {31, 7, 5, RandBLAS::Axis::Short, U},
        {257, 2048, 1, RandBLAS::Axis::Short, U},
        {257, 2048, 12, RandBLAS::Axis::Short, U},
        {7, 31, 1, RandBLAS::Axis::Long, U},
        {7, 31, 12, RandBLAS::Axis::Long, U},
        {31, 7, 12, RandBLAS::Axis::Long, U},
        {257, 257, 128, RandBLAS::Axis::Long, U},
        {2048, 4096, 1, RandBLAS::Axis::Long, U}
    }};
    constexpr std::array<int, 3> thread_counts{1, 2, 4};
    const int saved_dynamic = omp_get_dynamic();
    const int saved_max_threads = omp_get_max_threads();
    omp_set_dynamic(0);

    RandBLAS::RNGState<> seed(20260817);
    seed.counter.incr(41);
    for (const Case &test_case : cases) {
        RandBLAS::SparseDist dist(
            test_case.n_rows, test_case.n_cols,
            test_case.vec_nnz, test_case.major_axis, test_case.blocking
        );
        omp_set_num_threads(1);
        auto expected_full = sample_sparse_snapshot<double>(
            dist, dist.n_rows, dist.n_cols, 0, 0, seed
        );
        auto expected_sub = sample_sparse_snapshot<double>(
            dist, dist.n_rows - 2, dist.n_cols - 3, 1, 2, seed
        );

        for (int thread_count : thread_counts) {
            omp_set_num_threads(thread_count);
            auto actual_full = sample_sparse_snapshot<double>(
                dist, dist.n_rows, dist.n_cols, 0, 0, seed
            );
            auto actual_sub = sample_sparse_snapshot<double>(
                dist, dist.n_rows - 2, dist.n_cols - 3, 1, 2, seed
            );
            EXPECT_EQ(actual_full.nnz, expected_full.nnz);
            EXPECT_EQ(actual_full.vals, expected_full.vals);
            EXPECT_EQ(actual_full.rows, expected_full.rows);
            EXPECT_EQ(actual_full.cols, expected_full.cols);
            EXPECT_EQ(actual_full.end_state, expected_full.end_state);
            EXPECT_EQ(actual_sub.nnz, expected_sub.nnz);
            EXPECT_EQ(actual_sub.vals, expected_sub.vals);
            EXPECT_EQ(actual_sub.rows, expected_sub.rows);
            EXPECT_EQ(actual_sub.cols, expected_sub.cols);
            EXPECT_EQ(actual_sub.end_state, expected_sub.end_state);
        }
    }

    omp_set_num_threads(saved_max_threads);
    omp_set_dynamic(saved_dynamic);
}

TEST_F(TestSparseSkOpConstruction, parallel_saso_k1_writes_full_coo_data) {
    // SASOs with one nonzero per major-axis vector use a specialized sampling
    // path. Compare serial and four-thread samples entry-for-entry using float
    // values and 32-bit indices, so this test checks that the fast path writes
    // all three COO arrays and advances the RNG state for nondefault types.
    const int saved_dynamic = omp_get_dynamic();
    const int saved_max_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    RandBLAS::SparseDist dist(257, 2048, 1, RandBLAS::Axis::Short);
    RandBLAS::RNGState<> seed(991);
    seed.counter.incr(73);

    omp_set_num_threads(1);
    auto expected = sample_sparse_snapshot<float, int32_t>(
        dist, dist.n_rows, dist.n_cols, 0, 0, seed
    );
    omp_set_num_threads(4);
    auto actual = sample_sparse_snapshot<float, int32_t>(
        dist, dist.n_rows, dist.n_cols, 0, 0, seed
    );

    EXPECT_EQ(actual.nnz, expected.nnz);
    EXPECT_EQ(actual.vals, expected.vals);
    EXPECT_EQ(actual.rows, expected.rows);
    EXPECT_EQ(actual.cols, expected.cols);
    EXPECT_EQ(actual.end_state, expected.end_state);

    omp_set_num_threads(saved_max_threads);
    omp_set_dynamic(saved_dynamic);
}

TEST_F(TestSparseSkOpConstruction, parallel_sampling_rejects_two_word_rng) {
    // Signed sparse sampling currently requires a counter-based generator
    // result with at least four words so one counter can supply the sampled
    // index and its sign. Force the parallel path with Philox2x32, which has
    // only two words, and verify that both SASO and LASO sampling reject it
    // with a RandBLAS error instead of reading nonexistent random words.
    using TwoWordRNG = r123::Philox2x32;
    const int saved_dynamic = omp_get_dynamic();
    const int saved_max_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    for (RandBLAS::Axis axis : {RandBLAS::Axis::Short, RandBLAS::Axis::Long}) {
        RandBLAS::SparseDist dist(257, 2048, 4, axis);
        RandBLAS::RNGState<TwoWordRNG> seed(17);
        std::vector<double> vals(dist.full_nnz);
        std::vector<int64_t> rows(dist.full_nnz);
        std::vector<int64_t> cols(dist.full_nnz);
        int64_t nnz = -1;
        EXPECT_THROW(
            RandBLAS::fill_sparse_unpacked(
                dist, dist.n_rows, dist.n_cols, 0, 0, nnz,
                vals.data(), rows.data(), cols.data(), seed
            ),
            RandBLAS::Error
        );

        // The owning-COO convenience path allocates its output buffers before
        // sampling. It must propagate the same validation error while allowing
        // those partially constructed buffers to be reclaimed during unwinding.
        RandBLAS::SparseSkOp<double, TwoWordRNG> S(dist, seed);
        EXPECT_THROW(
            RandBLAS::sparse::submatrix_as_coo(
                S, dist.n_rows, dist.n_cols, 0, 0
            ),
            RandBLAS::Error
        );
    }

    omp_set_num_threads(saved_max_threads);
    omp_set_dynamic(saved_dynamic);
}
#endif

TEST_F(TestSparseSkOpConstruction, respect_ownership) {
    respect_ownership<int64_t>(7, 20);
    respect_ownership<int64_t>(7, 20);
    respect_ownership<int64_t>(7, 20);

    respect_ownership<int>(7, 20);
    respect_ownership<int>(7, 20);
    respect_ownership<int>(7, 20);
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_nosub_saso) {
    for (SparseBlocking blocking : {SparseBlocking::Blocked, SparseBlocking::Unblocked}) {
        unpacked_nosub({10, 20, 7, Axis::Short, blocking});
        unpacked_nosub({20, 10, 7, Axis::Short, blocking});
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_nosub_laso) {
    unpacked_nosub({10,20,7,Axis::Long});
    unpacked_nosub({20,10,7,Axis::Long});
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_saso) {
    for (SparseBlocking blocking : {SparseBlocking::Blocked, SparseBlocking::Unblocked}) {
        for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
            fill_unpacked_sub<float>(Axis::Short, blocking, vnz, 7, 13);   // wide
            fill_unpacked_sub<float>(Axis::Short, blocking, vnz, 13, 7);   // tall
            fill_unpacked_sub<float>(Axis::Short, blocking, vnz, 10, 10);  // square
        }
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_laso) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float>(Axis::Long, SparseBlocking::Unblocked, vnz, 7, 13);    // wide
        fill_unpacked_sub<float>(Axis::Long, SparseBlocking::Unblocked, vnz, 13, 7);    // tall
        fill_unpacked_sub<float>(Axis::Long, SparseBlocking::Unblocked, vnz, 10, 10);   // square
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_saso_int32) {
    for (SparseBlocking blocking : {SparseBlocking::Blocked, SparseBlocking::Unblocked}) {
        for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
            fill_unpacked_sub<float, int>(Axis::Short, blocking, vnz, 7, 13);
            fill_unpacked_sub<float, int>(Axis::Short, blocking, vnz, 13, 7);
        }
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_laso_int32) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float, int>(Axis::Long, SparseBlocking::Unblocked, vnz, 7, 13);
        fill_unpacked_sub<float, int>(Axis::Long, SparseBlocking::Unblocked, vnz, 13, 7);
    }
}

// MARK: SASO blocking

// A Philox4x32 that counts its evaluations, so a test can see how many blocks a
// sampling call draws.
struct CountingPhilox : r123::Philox4x32 {
    static inline std::atomic<int64_t> calls{0};
    ctr_type operator()(ctr_type c, key_type k) const {
        ++calls;
        return r123::Philox4x32::operator()(c, k);
    }
};

// The block of short-axis coordinates that blocked sampling assigns to entry j
// of a short-axis vector of length dim_major with vec_nnz nonzeros.
static std::pair<int64_t, int64_t> saso_block_bounds(int64_t dim_major, int64_t vec_nnz, int64_t j) {
    const int64_t len = dim_major / vec_nnz;
    const int64_t num_long = dim_major % vec_nnz;
    const int64_t start = j * len + std::min(j, num_long);
    return {start, start + len + ((j < num_long) ? 1 : 0)};
}

// An independent implementation of blocked sampling, written from the documented
// counter layout: vector i, block j uses counter (seed counter + i * vec_nnz + j);
// the first two words of the generator's output there give the location and the
// third word gives the sign.
template <typename T, typename sint_t>
static SparseSnapshot<T, sint_t> reference_blocked_saso(
    const SparseDist &dist, const RNGState<> &seed
) {
    const bool major_is_rows = dist.n_rows <= dist.n_cols;
    SparseSnapshot<T, sint_t> out{
        dist.full_nnz, std::vector<T>(dist.full_nnz), std::vector<sint_t>(dist.full_nnz),
        std::vector<sint_t>(dist.full_nnz), seed
    };
    std::vector<sint_t> &major = major_is_rows ? out.rows : out.cols;
    std::vector<sint_t> &minor = major_is_rows ? out.cols : out.rows;
    RandBLAS::DefaultRNG gen;
    for (int64_t i = 0; i < dist.dim_minor; ++i) {
        for (int64_t j = 0; j < dist.vec_nnz; ++j) {
            auto ctr = seed.counter;
            ctr.incr(i * dist.vec_nnz + j);
            auto rv = gen(ctr, seed.key);
            auto [lo, hi] = saso_block_bounds(dist.dim_major, dist.vec_nnz, j);
            const std::uint64_t r = RandBLAS::promote_uint_pair(rv[0], rv[1]);
            const int64_t e = i * dist.vec_nnz + j;
            major[e] = static_cast<sint_t>(lo + static_cast<int64_t>(r % static_cast<std::uint64_t>(hi - lo)));
            minor[e] = static_cast<sint_t>(i);
            out.vals[e] = (rv[2] % 2 == 0) ? (T) 1 : (T) -1;
        }
    }
    out.end_state = seed;
    out.end_state.counter.incr(dist.full_nnz);
    return out;
}

template <typename T, typename sint_t>
static void check_blocked_saso(int64_t n_rows, int64_t n_cols, int64_t vec_nnz, uint32_t key) {
    SparseDist dist(n_rows, n_cols, vec_nnz, Axis::Short, SparseBlocking::Blocked);
    RNGState<> seed(key);
    seed.counter.incr(17);
    auto actual = sample_sparse_snapshot<T, sint_t>(dist, n_rows, n_cols, 0, 0, seed);
    auto expected = reference_blocked_saso<T, sint_t>(dist, seed);
    ASSERT_EQ(actual.nnz, dist.full_nnz);
    EXPECT_EQ(actual.rows, expected.rows);
    EXPECT_EQ(actual.cols, expected.cols);
    EXPECT_EQ(actual.vals, expected.vals);
    EXPECT_EQ(actual.end_state, expected.end_state);
    EXPECT_EQ(actual.end_state, RandBLAS::compute_next_state(dist, seed));

    // Structure: entry j of every vector lies in block j, and values are +/- 1.
    const bool major_is_rows = n_rows <= n_cols;
    const auto &major = major_is_rows ? actual.rows : actual.cols;
    const auto &minor = major_is_rows ? actual.cols : actual.rows;
    for (int64_t i = 0; i < dist.dim_minor; ++i) {
        for (int64_t j = 0; j < vec_nnz; ++j) {
            const int64_t e = i * vec_nnz + j;
            auto [lo, hi] = saso_block_bounds(dist.dim_major, vec_nnz, j);
            ASSERT_EQ(minor[e], i);
            ASSERT_GE(major[e], lo);
            ASSERT_LT(major[e], hi);
            ASSERT_TRUE(actual.vals[e] == (T) 1 || actual.vals[e] == (T) -1);
        }
    }
}

TEST_F(TestSparseSkOpConstruction, saso_blocking_defaults_follow_major_axis) {
    // A short-axis-major distribution is blocked unless the caller asks
    // otherwise; long-axis vectors are never blocked.
    EXPECT_EQ(SparseDist(7, 20, 3).blocking, SparseBlocking::Blocked);
    EXPECT_EQ(SparseDist(20, 7, 3, Axis::Short).blocking, SparseBlocking::Blocked);
    EXPECT_EQ(SparseDist(7, 20, 3, Axis::Long).blocking, SparseBlocking::Unblocked);
    EXPECT_EQ(SparseDist(7, 20, 3, Axis::Short, SparseBlocking::Unblocked).blocking, SparseBlocking::Unblocked);
    // The convenience constructors for non-int64 dimensions forward or resolve the blocking.
    EXPECT_EQ(SparseDist(7.0, 20.0, 3, Axis::Short, SparseBlocking::Unblocked).blocking, SparseBlocking::Unblocked);
    EXPECT_EQ(SparseDist(7.0, 20.0, 3).blocking, SparseBlocking::Blocked);
    // Blocking does not change the isometry scale.
    EXPECT_EQ(
        SparseDist(7, 20, 3, Axis::Short, SparseBlocking::Blocked).isometry_scale,
        SparseDist(7, 20, 3, Axis::Short, SparseBlocking::Unblocked).isometry_scale
    );
}

TEST_F(TestSparseSkOpConstruction, long_axis_rejects_blocking) {
    EXPECT_THROW(SparseDist(7, 20, 3, Axis::Long, SparseBlocking::Blocked), RandBLAS::Error);
    EXPECT_THROW(SparseDist(20, 7, 3, Axis::Long, SparseBlocking::Blocked), RandBLAS::Error);
    EXPECT_THROW(SparseDist(7, 20, 1, Axis::Long, SparseBlocking::Blocked), RandBLAS::Error);
}

TEST_F(TestSparseSkOpConstruction, blocked_saso_matches_reference_sampler) {
    // Shapes cover wide, tall, and square operators; sparsity levels cover
    // vec_nnz = 1, divisors and non-divisors of dim_major, and the extremes
    // vec_nnz = dim_major - 1 and vec_nnz = dim_major.
    struct Shape { int64_t n_rows; int64_t n_cols; };
    for (Shape shape : {Shape{7, 20}, Shape{20, 7}, Shape{12, 12}, Shape{64, 1000}, Shape{1000, 61}}) {
        const int64_t dim_major = std::min(shape.n_rows, shape.n_cols);
        for (int64_t vec_nnz : {(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 4, dim_major - 1, dim_major}) {
            for (uint32_t key : {0u, 42u}) {
                check_blocked_saso<double, int64_t>(shape.n_rows, shape.n_cols, vec_nnz, key);
                check_blocked_saso<float, int32_t>(shape.n_rows, shape.n_cols, vec_nnz, key);
            }
        }
    }
}

TEST_F(TestSparseSkOpConstruction, blocked_saso_windows_match_full_operator) {
    // Blocked submatrix sampling draws only the blocks that meet the short-axis
    // window. Check windows that start or end inside a block, lie on block
    // boundaries, cover a single coordinate, sit in the last block, or are
    // empty, for a wide and a tall operator with unequal blocks
    // (100 = 7 * 14 + 2).
    struct Window { int64_t off; int64_t len; };
    const std::array<Window, 10> windows{{
        {0, 0}, {0, 1}, {0, 15}, {15, 15}, {14, 2}, {30, 1}, {40, 25}, {86, 14}, {99, 1}, {100, 0}
    }};
    for (uint32_t key : {3u, 11u}) {
        for (const Window &w : windows) {
            // wide 100 x 300: the short axis is the rows
            fill_unpacked_sub_one<double>(Axis::Short, SparseBlocking::Blocked, 7, 100, 300, w.off, 5, w.len, 290, key);
            // tall 300 x 100: the short axis is the columns
            fill_unpacked_sub_one<double>(Axis::Short, SparseBlocking::Blocked, 7, 300, 100, 5, w.off, 290, w.len, key);
        }
    }
    // Each submatrix below ends at the last column, so sampling it returns the
    // state that follows the full operator.
    SparseDist dist(100, 4000, 7, Axis::Short, SparseBlocking::Blocked);
    RNGState<> seed(5);
    seed.counter.incr(9);
    auto full_state = seed;
    full_state.counter.incr(4000 * dist.vec_nnz);
    for (const Window &w : windows) {
        auto snapshot = sample_sparse_snapshot<double>(dist, w.len, 3990, w.off, 10, seed);
        EXPECT_EQ(snapshot.end_state, full_state);
    }
#if defined(RandBLAS_HAS_OpenMP)
    // The same windows give identical results and states with one and four threads.
    const int saved_dynamic = omp_get_dynamic();
    const int saved_max_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    for (const Window &w : windows) {
        omp_set_num_threads(1);
        auto expected = sample_sparse_snapshot<double>(dist, w.len, 3990, w.off, 10, seed);
        omp_set_num_threads(4);
        auto actual = sample_sparse_snapshot<double>(dist, w.len, 3990, w.off, 10, seed);
        EXPECT_EQ(actual.nnz, expected.nnz);
        EXPECT_EQ(actual.rows, expected.rows);
        EXPECT_EQ(actual.cols, expected.cols);
        EXPECT_EQ(actual.vals, expected.vals);
        EXPECT_EQ(actual.end_state, expected.end_state);
    }
    omp_set_num_threads(saved_max_threads);
    omp_set_dynamic(saved_dynamic);
#endif
}

TEST_F(TestSparseSkOpConstruction, blocked_saso_window_draws_only_meeting_blocks) {
    // dim_major = 100 and vec_nnz = 7 give the blocks [0,15) [15,30) [30,44)
    // [44,58) [58,72) [72,86) [86,100). Each range is (first block, number of blocks).
    using RandBLAS::sparse::blocks_meeting_window;
    using Range = std::pair<int64_t, int64_t>;
    EXPECT_EQ(blocks_meeting_window(100, 7, 0, 0), Range(0, 0));
    EXPECT_EQ(blocks_meeting_window(100, 7, 0, 15), Range(0, 1));
    EXPECT_EQ(blocks_meeting_window(100, 7, 14, 2), Range(0, 2));
    EXPECT_EQ(blocks_meeting_window(100, 7, 30, 1), Range(2, 1));
    EXPECT_EQ(blocks_meeting_window(100, 7, 40, 25), Range(2, 3));
    EXPECT_EQ(blocks_meeting_window(100, 7, 44, 1), Range(3, 1));
    EXPECT_EQ(blocks_meeting_window(100, 7, 86, 14), Range(6, 1));
    EXPECT_EQ(blocks_meeting_window(100, 7, 99, 1), Range(6, 1));
    EXPECT_EQ(blocks_meeting_window(100, 7, 0, 100), Range(0, 7));

    // fill_sparse_unpacked draws one location, and so evaluates the generator once,
    // per vector and meeting block.
    SparseDist dist(100, 300, 7, Axis::Short, SparseBlocking::Blocked);
    RNGState<CountingPhilox> seed(5);
    std::vector<double> vals(dist.full_nnz);
    std::vector<int64_t> rows(dist.full_nnz);
    std::vector<int64_t> cols(dist.full_nnz);
    struct Case { int64_t off; int64_t len; int64_t num_blocks; };
    for (Case c : {Case{30, 1, 1}, Case{14, 2, 2}, Case{86, 14, 1}, Case{0, 100, 7}, Case{50, 0, 0}}) {
        CountingPhilox::calls = 0;
        int64_t nnz = -1;
        RandBLAS::fill_sparse_unpacked(
            dist, c.len, 300, c.off, 0, nnz, vals.data(), rows.data(), cols.data(), seed
        );
        EXPECT_EQ(CountingPhilox::calls.load(), 300 * c.num_blocks);
    }
    // With one nonzero per vector, an empty window draws nothing either.
    SparseDist dist1(100, 300, 1, Axis::Short, SparseBlocking::Blocked);
    CountingPhilox::calls = 0;
    int64_t nnz1 = -1;
    RandBLAS::fill_sparse_unpacked(
        dist1, 0, 300, 50, 0, nnz1, vals.data(), rows.data(), cols.data(), seed
    );
    EXPECT_EQ(CountingPhilox::calls.load(), 0);
}

TEST_F(TestSparseSkOpConstruction, saso_blockings_agree_when_vec_nnz_is_one) {
    // With one nonzero per vector there is a single block, so both blockings
    // describe the same distribution and draw it from the same counters.
    for (auto [n_rows, n_cols] : {std::pair<int64_t, int64_t>{7, 31}, {31, 7}, {257, 2048}}) {
        SparseDist blocked(n_rows, n_cols, 1, Axis::Short, SparseBlocking::Blocked);
        SparseDist unblocked(n_rows, n_cols, 1, Axis::Short, SparseBlocking::Unblocked);
        RNGState<> seed(99);
        auto a = sample_sparse_snapshot<double>(blocked, n_rows, n_cols, 0, 0, seed);
        auto b = sample_sparse_snapshot<double>(unblocked, n_rows, n_cols, 0, 0, seed);
        EXPECT_EQ(a.rows, b.rows);
        EXPECT_EQ(a.cols, b.cols);
        EXPECT_EQ(a.vals, b.vals);
        EXPECT_EQ(a.end_state, b.end_state);
        auto c = sample_sparse_snapshot<double>(blocked, n_rows - 2, n_cols - 3, 1, 2, seed);
        auto d = sample_sparse_snapshot<double>(unblocked, n_rows - 2, n_cols - 3, 1, 2, seed);
        EXPECT_EQ(c.rows, d.rows);
        EXPECT_EQ(c.cols, d.cols);
        EXPECT_EQ(c.vals, d.vals);
        EXPECT_EQ(c.end_state, d.end_state);
    }
}

TEST_F(TestSparseSkOpConstruction, unblocked_saso_reproduces_previous_default) {
    // Golden values from the last revision before SparseBlocking existed, when
    // every short-axis-major SparseDist sampled without replacement. Requesting
    // SparseBlocking::Unblocked must reproduce those operators exactly.
    {
        SparseDist dist(5, 9, 2, Axis::Short, SparseBlocking::Unblocked);
        auto s = sample_sparse_snapshot<double>(dist, 5, 9, 0, 0, RNGState<>(1));
        EXPECT_EQ(s.rows, (std::vector<int64_t>{3, 4, 2, 3, 3, 4, 2, 3, 1, 4, 0, 1, 1, 4, 3, 4, 3, 4}));
        EXPECT_EQ(s.cols, (std::vector<int64_t>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8}));
        EXPECT_EQ(s.vals, (std::vector<double>{1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1}));
        EXPECT_EQ(s.end_state.counter.v[0], 18u);
    }
    {
        SparseDist dist(9, 4, 3, Axis::Short, SparseBlocking::Unblocked);
        RNGState<> seed(7);
        seed.counter.incr(5);
        auto s = sample_sparse_snapshot<double>(dist, 9, 4, 0, 0, seed);
        EXPECT_EQ(s.rows, (std::vector<int64_t>{0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8}));
        EXPECT_EQ(s.cols, (std::vector<int64_t>{0, 1, 2, 0, 1, 2, 0, 1, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 1, 2, 0, 1, 3}));
        EXPECT_EQ(s.vals, (std::vector<double>{-1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1}));
        EXPECT_EQ(s.end_state.counter.v[0], 32u);
    }
}

////////////////////////////////////////////////////////////////////////
//
//
//     SASOs
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20) {
    // vec_nnz=1
    proper_saso_construction<int64_t>(7, 20, 0, 0);
    proper_saso_construction<int64_t>(7, 20, 1, 0);
    proper_saso_construction<int64_t>(7, 20, 2, 0);
    // vec_nnz=2
    proper_saso_construction<int64_t>(7, 20, 0, 1);
    proper_saso_construction<int64_t>(7, 20, 1, 1);
    proper_saso_construction<int64_t>(7, 20, 2, 1);
    // vec_nnz=3
    proper_saso_construction<int64_t>(7, 20, 0, 2);
    proper_saso_construction<int64_t>(7, 20, 1, 2);
    proper_saso_construction<int64_t>(7, 20, 2, 2);
    // vec_nnz=7
    proper_saso_construction<int64_t>(7, 20, 0, 3);
    proper_saso_construction<int64_t>(7, 20, 1, 3);
    proper_saso_construction<int64_t>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7) {
    // vec_nnz=1
    proper_saso_construction<int64_t>(15, 7, 0, 0);
    proper_saso_construction<int64_t>(15, 7, 1, 0);
    // vec_nnz=1
    proper_saso_construction<int64_t>(15, 7, 0, 1);
    proper_saso_construction<int64_t>(15, 7, 1, 1);
    // vec_nnz=3
    proper_saso_construction<int64_t>(15, 7, 0, 2);
    proper_saso_construction<int64_t>(15, 7, 1, 2);
    // vec_nnz=7
    proper_saso_construction<int64_t>(15, 7, 0, 3);
    proper_saso_construction<int64_t>(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_7by20_int32) {
    // test vec_nnz = 1, 2, 3, 7
    proper_saso_construction<int>(7, 20, 0, 0);
    proper_saso_construction<int>(7, 20, 0, 1);
    proper_saso_construction<int>(7, 20, 0, 2);
    proper_saso_construction<int>(7, 20, 0, 3);
}


TEST_F(TestSparseSkOpConstruction, SASO_Dim_15by7_int32) {
    // test vec_nnz = 1, 2, 3, 7
    proper_saso_construction<int>(15, 7, 0, 0);
    proper_saso_construction<int>(15, 7, 0, 1);
    proper_saso_construction<int>(15, 7, 0, 2);
    proper_saso_construction<int>(15, 7, 0, 3);
}


////////////////////////////////////////////////////////////////////////
//
//
//     LASOs
//
//
////////////////////////////////////////////////////////////////////////

TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20) {
    // vec_nnz=1
    proper_laso_construction<int64_t>(7, 20, 0);
    proper_laso_construction<int64_t>(7, 20, 1);
    proper_laso_construction<int64_t>(7, 20, 2);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7) {
    // vec_nnz=1
    proper_laso_construction<int64_t>(15, 7, 0);
    proper_laso_construction<int64_t>(15, 7, 1);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(7, 20, 0);
    proper_laso_construction<int>(7, 20, 1);
    proper_laso_construction<int>(7, 20, 2);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(15, 7, 0);
    proper_laso_construction<int>(15, 7, 1);
}
