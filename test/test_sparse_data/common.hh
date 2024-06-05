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

#ifndef randblas_test_sparse_data_common_hh
#define randblas_test_sparse_data_common_hh
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


namespace test::sparse_data::common {

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;
using namespace RandBLAS::sparse_data::conversions;
using blas::Layout;

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
void sparseskop_to_dense(
    RandBLAS::SparseSkOp<T, RNG, sint_t> &S0,
    T *mat,
    Layout layout
) {
    RandBLAS::SparseDist D = S0.dist;
    for (int64_t i = 0; i < D.n_rows * D.n_cols; ++i)
        mat[i] = 0.0;
    auto idx = [D, layout](int64_t i, int64_t j) {
        return  (layout == Layout::ColMajor) ? (i + j*D.n_rows) : (j + i*D.n_cols);
    };
    int64_t nnz = RandBLAS::sparse::nnz(S0);
    for (int64_t i = 0; i < nnz; ++i) {
        sint_t row = S0.rows[i];
        sint_t col = S0.cols[i];
        T val = S0.vals[i];
        mat[idx(row, col)] = val;
    }
}

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
    auto dist = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
    auto [unused, next_state] = RandBLAS::fill_dense(dist, spar, state);

    auto temp = new T[n_rows * n_cols];
    auto D_mat = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
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
    COOMatrix<T> &spmat
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


}

#endif
