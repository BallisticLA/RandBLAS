#include "RandBLAS/config.h"
#include "RandBLAS/base.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo.hh"
#include "RandBLAS/sparse_data/csr.hh"

namespace test::sparse_data::common {

using namespace RandBLAS::sparse_data;
using namespace RandBLAS::sparse_data::csr;
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




}

