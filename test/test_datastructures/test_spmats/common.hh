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

#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/conversions.hh"


namespace test::test_datastructures::test_spmats {

using blas::Layout;

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
    reserve_coo(nnz, spmat);
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
    reserve_coo(new_nnz, spmat_out);
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

}
