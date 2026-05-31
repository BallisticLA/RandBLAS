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
#include <gtest/gtest.h>
#include <cmath>

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
using RandBLAS::fill_sparse;
using RandBLAS::fill_sparse_unpacked_nosub;


class TestSparseSkOpConstruction : public ::testing::Test
{
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
    void proper_laso_construction(int64_t d, int64_t m, int64_t key_index, int64_t nnz_index) {
        using RNG = SparseSkOp<float>::state_t::generator;
        SparseDist D0(d, m, vec_nnzs[nnz_index], Axis::Long);
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
        msg = RandBLAS::testing::buffs_approx_equal(rows.data(), rows_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(cols.data(), cols_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(vals.data(), vals_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (T) 0, (T) 0);
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
        msg = RandBLAS::testing::buffs_approx_equal(rows.data(), rows_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(cols.data(), cols_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (sint_t) 0, (sint_t) 0);
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(vals.data(), vals_copy.data(), sd.full_nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__, (T) 0, (T) 0);
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
            vals.data(), S.vals, nnz, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(
            rows.data(), S.rows, nnz,  __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        msg = RandBLAS::testing::buffs_approx_equal(
            cols.data(), S.cols, nnz,  __PRETTY_FUNCTION__, __FILE__, __LINE__
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
        Axis major_axis, int64_t vec_nnz, int64_t d, int64_t m,
        int64_t ro_s, int64_t co_s, int64_t n_rows_sub, int64_t n_cols_sub, uint32_t key
    ) {
        SparseDist D {d, m, vec_nnz, major_axis};
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
            __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
        if (msg.size() > 0) {
            FAIL() << msg;
        }
        return;
    }

    template <typename T, SignedInteger sint_t = int64_t>
    void fill_unpacked_sub(Axis major_axis, int64_t vec_nnz, int64_t d, int64_t m) {
        for (auto key : keys) {
            // full-size (regression guard: must reproduce nosub exactly)
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, 0, d, m, key);
            // corner offsets / partial windows
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 1, 0, d - 1, m,     key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, 1, d,     m - 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 1, 1, d - 1, m - 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 2, 3, d - 3, m - 4, key);
            // single row / single column
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, 0, 1, m, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, 0, d, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, d - 1, 0, 1, m, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, m - 1, d, 1, key);
            // 1x1 windows at a few positions (some likely empty for SASO)
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, 0, 0, 1, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, d - 1, m - 1, 1, 1, key);
            fill_unpacked_sub_one<T, sint_t>(major_axis, vec_nnz, d, m, d / 2, m / 2, 1, 1, key);
        }
        return;
    }
};

TEST_F(TestSparseSkOpConstruction, respect_ownership) {
    respect_ownership<int64_t>(7, 20);
    respect_ownership<int64_t>(7, 20);
    respect_ownership<int64_t>(7, 20);

    respect_ownership<int>(7, 20);
    respect_ownership<int>(7, 20);
    respect_ownership<int>(7, 20);
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_nosub_saso) {
    unpacked_nosub({10,20,7,Axis::Short});
    unpacked_nosub({20,10,7,Axis::Short});
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_nosub_laso) {
    unpacked_nosub({10,20,7,Axis::Long});
    unpacked_nosub({20,10,7,Axis::Long});
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_saso) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float>(Axis::Short, vnz, 7, 13);   // wide
        fill_unpacked_sub<float>(Axis::Short, vnz, 13, 7);   // tall
        fill_unpacked_sub<float>(Axis::Short, vnz, 10, 10);  // square
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_laso) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float>(Axis::Long, vnz, 7, 13);    // wide
        fill_unpacked_sub<float>(Axis::Long, vnz, 13, 7);    // tall
        fill_unpacked_sub<float>(Axis::Long, vnz, 10, 10);   // square
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_saso_int32) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float, int>(Axis::Short, vnz, 7, 13);
        fill_unpacked_sub<float, int>(Axis::Short, vnz, 13, 7);
    }
}

TEST_F(TestSparseSkOpConstruction, fill_unpacked_sub_laso_int32) {
    for (int64_t vnz : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 7}) {
        fill_unpacked_sub<float, int>(Axis::Long, vnz, 7, 13);
        fill_unpacked_sub<float, int>(Axis::Long, vnz, 13, 7);
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
    proper_laso_construction<int64_t>(7, 20, 0, 0);
    proper_laso_construction<int64_t>(7, 20, 1, 0);
    proper_laso_construction<int64_t>(7, 20, 2, 0);
    // // vec_nnz=2
    // proper_laso_construction<int64_t>(7, 20, 0, 1);
    // proper_laso_construction<int64_t>(7, 20, 1, 1);
    // proper_laso_construction<int64_t>(7, 20, 2, 1);
    // // vec_nnz=3
    // proper_laso_construction<int64_t>(7, 20, 0, 2);
    // proper_laso_construction<int64_t>(7, 20, 1, 2);
    // proper_laso_construction<int64_t>(7, 20, 2, 2);
    // // vec_nnz=7
    // proper_laso_construction<int64_t>(7, 20, 0, 3);
    // proper_laso_construction<int64_t>(7, 20, 1, 3);
    // proper_laso_construction<int64_t>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7) {
    // vec_nnz=1
    proper_laso_construction<int64_t>(15, 7, 0, 0);
    proper_laso_construction<int64_t>(15, 7, 1, 0);
    // // vec_nnz=2
    // proper_laso_construction<int64_t>(15, 7, 0, 1);
    // proper_laso_construction<int64_t>(15, 7, 1, 1);
    // // vec_nnz=3
    // proper_laso_construction<int64_t>(15, 7, 0, 2);
    // proper_laso_construction<int64_t>(15, 7, 1, 2);
    // // vec_nnz=7
    // proper_laso_construction<int64_t>(15, 7, 0, 3);
    // proper_laso_construction<int64_t>(15, 7, 1, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_7by20_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(7, 20, 0, 0);
    proper_laso_construction<int>(7, 20, 1, 0);
    proper_laso_construction<int>(7, 20, 2, 0);
    // // vec_nnz=2
    // proper_laso_construction<int>(7, 20, 0, 1);
    // proper_laso_construction<int>(7, 20, 1, 1);
    // proper_laso_construction<int>(7, 20, 2, 1);
    // // vec_nnz=3
    // proper_laso_construction<int>(7, 20, 0, 2);
    // proper_laso_construction<int>(7, 20, 1, 2);
    // proper_laso_construction<int>(7, 20, 2, 2);
    // // vec_nnz=7
    // proper_laso_construction<int>(7, 20, 0, 3);
    // proper_laso_construction<int>(7, 20, 1, 3);
    // proper_laso_construction<int>(7, 20, 2, 3);
}


TEST_F(TestSparseSkOpConstruction, LASO_Dim_15by7_int32) {
    // vec_nnz=1
    proper_laso_construction<int>(15, 7, 0, 0);
    proper_laso_construction<int>(15, 7, 1, 0);
    // // vec_nnz=2
    // proper_laso_construction<int>(15, 7, 0, 1);
    // proper_laso_construction<int>(15, 7, 1, 1);
    // // vec_nnz=3
    // proper_laso_construction<int>(15, 7, 0, 2);
    // proper_laso_construction<int>(15, 7, 1, 2);
    // // vec_nnz=7
    // proper_laso_construction<int>(15, 7, 0, 3);
    // proper_laso_construction<int>(15, 7, 1, 3);
}
