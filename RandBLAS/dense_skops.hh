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

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/util.hh"

#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>

#include <math.h>
#include <typeinfo>


namespace RandBLAS::dense {

template <typename T_IN, typename T_OUT>
inline void copy_promote(int n, const T_IN &a, T_OUT* b) {
    for (int i = 0; i < n; ++i)
        b[i] = static_cast<T_OUT>(a[i]);
    return;
}

/** 
 * Fill buff with random values so it gives a row-major representation of an n_srows \math{\times} n_scols
 * submatrix of some implicitly defined parent matrix.
 * 
 * The implicit parent matrix is **imagined** as a buffer in row-major order with "n_cols" columns.
 * "ptr" is the pointer offset for the desired submatrix in the imagined buffer of the parent matrix.
 *
 * @tparam T the data type of the matrix
 * @tparam RNG a random123 CBRNG type
 * @tparam OP an operator that transforms raw random values into matrix
 *           elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_cols
 *      The number of columns in the implicitly defined parent matrix.
 * @param[in] smat
 *      A pointer to a region of memory with space for n_rows \math{\times} lda elements of type T.
 *      This memory will be filled with random values by wrting rows of length "n_scols"
 *      with an inter-row stride of length "lda".
 * @param[in] n_srows
 *      The number of rows in the submatrix.
 * @param[in] n_scols
 *      The number of columns in the submatrix.
 * @param[in] ptr
 *      The starting locaiton within the random matrix, for which 
 *      the submatrix is to be generated
 * @param[in] seed
 *      A CBRNG state
 * @param[in] lda
 *      If positive then must be >= n_scols.
 *      Otherwise, we automatically set it to n_scols.
 *
 * @returns the updated CBRNG state
 * 
 * Notes
 * -----
 * If RandBLAS is compiled with OpenMP threading support enabled, the operation is parallelized
 * using OMP_NUM_THREADS. The sequence of values generated does not depend on the number of threads.
 * 
 */
template<typename T, typename RNG, typename OP>
static RNGState<RNG> fill_dense_submat_impl(int64_t n_cols, T* smat, int64_t n_srows, int64_t n_scols, int64_t ptr, const RNGState<RNG> &seed, int64_t lda = 0) {
    if (lda <= 0) {
        lda = n_scols;
    } else {
        randblas_require(lda >= n_scols);
    }
    randblas_require(n_cols >= n_scols);
    RNG rng;
    using CTR_t = typename RNG::ctr_type;
    using KEY_t = typename RNG::key_type;
    const int64_t ctr_size = CTR_t::static_size;
    
    int64_t pad = 0;
    // ^ computed such that n_cols+pad is divisible by ctr_size
    if (n_cols % ctr_size != 0) {
        pad = ctr_size - n_cols % ctr_size;
    }
    
    const int64_t ptr_padded = ptr + ptr / n_cols * pad;
    // ^ ptr corresponding to the padded matrix
    const int64_t ctr_mat_start     = ptr_padded / ctr_size;
    const int64_t first_block_start = ptr_padded % ctr_size;
    // ^ counter and [position within the counter's array] for index "ptr_padded".
    const int64_t ctr_mat_row_end =  (ptr_padded + n_scols - 1) / ctr_size;
    const int64_t last_block_stop = ((ptr_padded + n_scols - 1) % ctr_size) + 1;
    // ^ counter and [1 + position within the counter's array] for index "(ptr_padded + n_scols - 1)".
    const int64_t ctr_inter_row_stride = (n_cols + pad) / ctr_size;
    // ^ number of counters between the first counter of a given row to the first counter of the next row;
    const bool  one_block_per_row = ctr_mat_start == ctr_mat_row_end;
    const int64_t first_block_len = ((one_block_per_row) ? last_block_stop : ctr_size) - first_block_start;

    CTR_t temp_c = seed.counter;
    temp_c.incr(ctr_mat_start);
    const CTR_t c = temp_c;
    const KEY_t k = seed.key;

    #pragma omp parallel
    {
    #pragma omp for schedule(static)
    for (int64_t row = 0; row < n_srows; row++) {

        int64_t incr_from_c = safe_int_product(ctr_inter_row_stride, row);
    
        auto c_row = c;
        c_row.incr(incr_from_c);
        auto rv = OP::generate(rng, c_row, k);

        T* smat_row = smat + row*lda;
        for (int i = 0; i < first_block_len; i++) {
            smat_row[i] = rv[i+first_block_start];
        }
        if ( one_block_per_row ) {
            continue;
        }
        // middle blocks
        int64_t ind = first_block_len;
        for (int i = 0; i < (ctr_mat_row_end - ctr_mat_start - 1); ++i) {
            c_row.incr();
            rv = OP::generate(rng, c_row, k);
            copy_promote(ctr_size, rv, smat_row + ind);
            ind = ind + ctr_size;
        }
        // last block
        c_row.incr();
        rv = OP::generate(rng, c_row, k);
        copy_promote(last_block_stop, rv, smat_row + ind);
    }
    }
    
    // find the largest counter in the counter array
    CTR_t max_c = c;
    max_c.incr(n_srows * ctr_inter_row_stride);
    return RNGState<RNG> {max_c, k};
}

template <typename RNG, typename DD>
RNGState<RNG> compute_next_state(DD dist, RNGState<RNG> state) {
    if (dist.major_axis == MajorAxis::Undefined) {
        // implies dist.family = DenseDistName::BlackBox
        return state;
    }
    // ^ This is the only place where MajorAxis is actually used to some 
    //   productive end.
    int64_t major_len = major_axis_length(dist);
    int64_t minor_len = dist.n_rows + (dist.n_cols - major_len);
    int64_t ctr_size = RNG::ctr_type::static_size;
    int64_t pad = 0;
    if (major_len % ctr_size != 0) {
        pad = ctr_size - major_len % ctr_size;
    }
    int64_t ctr_major_axis_stride = (major_len + pad) / ctr_size;
    int64_t full_incr = safe_int_product(ctr_major_axis_stride, minor_len);
    state.counter.incr(full_incr);
    return state;
}

} // end namespace RandBLAS::dense


namespace RandBLAS {

// =============================================================================
/// We support two distributions for dense sketching operators: those whose
/// entries are iid Gaussians or iid uniform over a symmetric interval.
/// For implementation reasons, we also expose an option to indicate that an
/// operator's distribution is unknown but it is still represented by a buffer
/// that can be used in GEMM.
enum class DenseDistName : char {
    // ---------------------------------------------------------------------------
    ///  Indicates the Gaussian distribution with mean 0 and variance 1.
    Gaussian = 'G',

    // ---------------------------------------------------------------------------
    ///  Indicates the uniform distribution over [-r, r] where r := sqrt(3)
    ///  is the radius that provides for a variance of 1.
    Uniform = 'U',

    // ---------------------------------------------------------------------------
    /// Indicates that the sketching operator's entries will only be specified by
    /// a user-provided buffer.
    BlackBox = 'B'
};

// =============================================================================
/// A distribution over dense sketching operators.
struct DenseDist {
    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many columns.
    const int64_t n_cols;

    // ---------------------------------------------------------------------------
    ///  The distribution used for the entries of the sketching operator.
    const DenseDistName family;

    // ---------------------------------------------------------------------------
    ///  This member affects whether samples from this distribution have their
    ///  entries filled row-wise or column-wise. While there is no statistical 
    ///  difference between these two filling orders, there are situations
    ///  where one order or the other might be preferred.
    ///  
    /// @verbatim embed:rst:leading-slashes
    /// .. dropdown:: *Notes for experts*
    ///    :animate: fade-in-slide-down
    ///
    ///     Deciding the value of this member is only needed
    ///     in algorithms where (1) there's a need to iteratively generate panels of
    ///     a larger sketching operator and (2) one of larger operator's dimensions
    ///     cannot be known before the  iterative process starts.
    ///
    ///     Essentially, a column-wise fill order lets us stack operators horizontally
    ///     in a consistent way, while row-wise fill order lets us stack vertically
    ///     in a consistent way. The mapping from major_axis to fill order is given below.
    /// 
    ///        .. list-table::
    ///           :widths: 34 33 33
    ///           :header-rows: 1
    ///
    ///           * -  
    ///             - :math:`\texttt{major_axis} = \texttt{Long}`
    ///             - :math:`\texttt{major_axis} = \texttt{Short}`
    ///           * - :math:`\texttt{n_rows} > \texttt{n_cols}`
    ///             - column-wise
    ///             - row-wise
    ///           * - :math:`\texttt{n_rows} \leq \texttt{n_cols}`
    ///             - row-wise
    ///             - column-wise
    /// @endverbatim
    const MajorAxis major_axis;

    // ---------------------------------------------------------------------------
    ///  A distribution over matrices of shape (n_rows, n_cols) with entries drawn
    ///  iid from either the default choice of standard normal distribution, or from
    ///  the uniform distribution over [-r, r], where r := sqrt(3) provides for
    ///  unit variance.
    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        DenseDistName dn = DenseDistName::Gaussian
    ) : n_rows(n_rows), n_cols(n_cols), family(dn), major_axis( (dn == DenseDistName::BlackBox) ? MajorAxis::Undefined : MajorAxis::Long) { }

    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        DenseDistName dn,
        MajorAxis ma
    ) : n_rows(n_rows), n_cols(n_cols), family(dn), major_axis(ma) {
        if (dn == DenseDistName::BlackBox) {
            randblas_require(ma == MajorAxis::Undefined);
        } else {
            randblas_require(ma != MajorAxis::Undefined);
        }  
    }

};


inline blas::Layout dist_to_layout(const DenseDist &D) {
    randblas_require(D.major_axis != MajorAxis::Undefined);
    bool is_wide = D.n_rows < D.n_cols;
    bool fa_long = D.major_axis == MajorAxis::Long;
    if (is_wide && fa_long) {
        return blas::Layout::RowMajor;
    } else if (is_wide) {
        return blas::Layout::ColMajor;
    } else if (fa_long) {
        return blas::Layout::ColMajor;
    } else {
        return blas::Layout::RowMajor;
    }
}

inline int64_t major_axis_length(const DenseDist &D) {
    randblas_require(D.major_axis != MajorAxis::Undefined);
    return (D.major_axis == MajorAxis::Long) ? 
        std::max(D.n_rows, D.n_cols) : std::min(D.n_rows, D.n_cols);
}

template <typename T>
inline T isometry_scale_factor(DenseDist D) {
    if (D.family == DenseDistName::BlackBox) {
        throw std::runtime_error("Unrecognized distribution.");
    }
    // When we sample from the scalar distributions we always
    // scale things so they're variance-1. 
    return std::pow((T) std::min(D.n_rows, D.n_cols), -0.5);
}


// =============================================================================
/// A sample from a distribution over dense sketching operators.
///
template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    using state_t  = RNGState<RNG>;
    using scalar_t = T;

    const int64_t n_rows;
    const int64_t n_cols;

    // ---------------------------------------------------------------------------
    ///  The distribution from which this sketching operator is sampled.
    ///  This member specifies the number of rows and columns of the sketching
    ///  operator.
    const DenseDist dist;

    // ---------------------------------------------------------------------------
    ///  The state that should be passed to the RNG when the full sketching 
    ///  operator needs to be sampled from scratch. 
    const RNGState<RNG> seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used by the next call to an RNG *after* the
    ///  full sketching operator has been sampled.
    const RNGState<RNG> next_state;


    T *buff = nullptr;                      // memory
    blas::Layout layout;                    // matrix storage order
    bool del_buff_on_destruct = false;      // only applies if fill_dense(S) has been called.


    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    DenseSkOp(
        int64_t n_rows,
        int64_t n_cols,
        DenseDist dist,
        RNGState<RNG> const &seed_state,
        RNGState<RNG> const &next_state,
        T *buff,
        blas::Layout layout,
        bool del_buff_on_destruct
    ) : 
        n_rows(n_rows), n_cols(n_cols), dist(dist),
        seed_state(seed_state), next_state(next_state),
        buff(buff), layout(layout), del_buff_on_destruct(del_buff_on_destruct) { }

    ///---------------------------------------------------------------------------
    /// Construct a DenseSkOp object, \math{S}.
    ///
    /// @param[in] dist
    ///     DenseDist.
    ///     - Specifies the dimensions of \math{S}.
    ///     - Specifies the (scalar) distribution of \math{S}'s entries.
    ///
    /// @param[in] state
    ///     RNGState.
    ///     - The RNG will use this as the starting point to generate all 
    ///       random numbers needed for \math{S}.
    ///
    DenseSkOp(
        DenseDist dist,
        RNGState<RNG> const &state
    ) : // variable definitions
        n_rows(dist.n_rows),
        n_cols(dist.n_cols),
        dist(dist),
        seed_state(state),
        next_state(dense::compute_next_state(dist, state)),
        buff(nullptr),
        layout(dist_to_layout(dist))
    {   // sanity checks
        randblas_require(this->dist.n_rows > 0);
        randblas_require(this->dist.n_cols > 0);
        if (dist.family == DenseDistName::BlackBox)
            randblas_require(this->buff != nullptr);
    };

    // Destructor
    ~DenseSkOp() {
        if (this->del_buff_on_destruct)
            delete [] this->buff;
    }
};


// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |mat|   mathmacro:: \operatorname{mat}
///   .. |buff|  mathmacro:: \mathtt{buff}
///   .. |D|     mathmacro:: \mathcal{D}
///   .. |nrows| mathmacro:: \mathtt{n\_rows}
///   .. |ncols| mathmacro:: \mathtt{n\_cols}
///   .. |ioff| mathmacro:: \mathtt{ro\_s}
///   .. |joff| mathmacro:: \mathtt{co\_s}
///   .. |layout| mathmacro:: \mathtt{layout}
///   .. |matS| mathmacro:: \mathbf{S}
///
/// @endverbatim
/// Fill \math{\buff} so that (1) \math{\mat(\buff)} is a submatrix of
/// an _implicit_ random sample from \math{\D}, and (2) \math{\mat(\buff)}
/// is determined by reading from \math{\buff} in \math{\layout} order.
/// 
/// If we denote the implicit sample from \math{\D} by \math{\matS}, then on exit
/// we have
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(\buff)[i,j] = \matS[\ioff + i, \joff + j]
/// @endverbatim
/// assuming the standard convention of zero-indexing.
///
/// This function is for generating low-level representations of matrices
/// that are equivalent to a submatrix of a RandBLAS DenseSkOp, but 
/// without using the DenseSkOp abstraction. This can be useful if you want
/// to sketch a structured matrix that RandBLAS doesn't support (like a symmetric
/// matrix whose values are only stored in the upper or lower triangle).
///
/// Note that since the entries of \math{\buff} are sampled iid from a common
/// distribution, the value of \math{\layout} is unlikely to have mathematical significance.
/// However, the value of \math{\layout} can affect this function's efficiency.
/// For best efficiency we recommend \math{\layout = \mathtt{dist\_to\_layout}(\D).}
/// If a different value of \math{\layout} is used, then this function will internally
/// allocate extra memory for an out-of-place layout change.
///
/// @param[in] layout
///     blas::Layout::RowMajor or blas::Layout::ColMajor
///      - The storage order for \math{\mat(\buff)} on exit.
/// @param[in] D
///      A DenseDist object.
///      - A distribution over random matrices of shape (D.n_rows, D.n_cols).
/// @param[in] n_rows
///      A positive integer.
///      - The number of rows in \math{\mat(\buff)}.
/// @param[in] n_cols
///      A positive integer.
///      - The number of columns in \math{\mat(\buff)}.
/// @param[in] ro_s
///      A nonnegative integer.
///      - The row offset for \math{\mat(\buff)} as a submatrix of \math{\matS}. 
///      - We require that \math{\ioff + \nrows} is at most D.n_rows.
/// @param[in] co_s
///      A nonnegative integer.
///      - The column offset for \math{\mat(\buff)} as a submatrix of \math{\matS}. 
///      - We require that \math{\joff + \ncols} is at most D.n_cols.
/// @param[in] buff
///     Buffer of type T.
///     - Length must be at least \math{\nrows \cdot \ncols}.
/// @param[in] seed
///      A CBRNG state
///      - Used to define \math{\matS} as a sample from \math{\D}.
/// 
template<typename T, typename RNG = r123::Philox4x32>
RNGState<RNG> fill_dense(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s, T* buff, const RNGState<RNG> &seed) {
    using RandBLAS::dense::fill_dense_submat_impl;
    randblas_require(D.n_rows >= n_rows + ro_s);
    randblas_require(D.n_cols >= n_cols + co_s);
    blas::Layout natural_layout = dist_to_layout(D);
    int64_t ma_len = major_axis_length(D);
    int64_t n_rows_, n_cols_, ptr;
    if (natural_layout == blas::Layout::ColMajor) {
        // operate on the transpose in row-major
        n_rows_ = n_cols;
        n_cols_ = n_rows;
        ptr = ro_s + safe_int_product(co_s, ma_len);
    } else {
        n_rows_ = n_rows;
        n_cols_ = n_cols;
        ptr = safe_int_product(ro_s, ma_len) + co_s;
    }
    RNGState<RNG> next_state{};
    switch (D.family) {
        case DenseDistName::Gaussian: {
            next_state = fill_dense_submat_impl<T,RNG,r123ext::boxmul>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            break;
        }
        case DenseDistName::Uniform: {
            next_state = fill_dense_submat_impl<T,RNG,r123ext::uneg11>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            blas::scal(n_rows_ * n_cols_, (T)std::sqrt(3), buff, 1);
            break;
        }
        case DenseDistName::BlackBox: {
            throw std::invalid_argument(std::string("fill_dense cannot be called with the BlackBox distribution."));
        }
        default: {
            throw std::runtime_error(std::string("Unrecognized distribution."));
        }
    }
    int64_t size_mat = n_rows * n_cols;
    if (layout != natural_layout) {
        T* flip_work = new T[size_mat];
        blas::copy(size_mat, buff, 1, flip_work, 1);
        auto [irs_nat, ics_nat] = layout_to_strides(natural_layout, n_rows, n_cols);
        auto [irs_req, ics_req] = layout_to_strides(layout, n_rows, n_cols);
        util::omatcopy(n_rows, n_cols, flip_work, irs_nat, ics_nat, buff, irs_req, ics_req);
        delete [] flip_work;
    }
    return next_state;
}
 
// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |mat|  mathmacro:: \operatorname{mat}
///   .. |buff| mathmacro:: \mathtt{buff}
///   .. |D|    mathmacro:: \mathcal{D} 
///
/// @endverbatim
/// Fill \math{\buff} so that \math{\mat(\buff)} is a sample from \math{\D} using
/// seed \math{\mathtt{seed}}.
///
/// @param[in] D
///      A DenseDist object.
/// @param[in] buff
///     Buffer of type T.
///     - Length must be at least D.n_rows * D.n_cols.
///     - The leading dimension of \math{\mat(\buff)} when reading from \math{\buff}
///       is either D.n_rows or D.n_cols, depending on the return value of this function
///       that indicates row-major or column-major layout.
/// @param[in] seed
///      A CBRNG state
///      - Used to define \math{\mat(\buff)} as a sample from \math{\D}.
///
/// @returns
///     A std::pair consisting of "layout" and "next_state".
///     - \math{\buff} must be read in "layout" order 
///       to recover \math{\mat(\buff)}. This layout is determined
///       from \math{\D} and cannot be controlled directly.
///     - If this function returns a layout that is undesirable then it is
///       the caller's responsibility to perform a transpose as needed.
/// 
template <typename T, typename RNG = r123::Philox4x32>
RNGState<RNG> fill_dense(const DenseDist &D, T *buff, const RNGState<RNG> &seed) {
    return fill_dense(dist_to_layout(D), D, D.n_rows, D.n_cols, 0, 0, buff, seed);
}

// ============================================================================= 
/// Performs the work in sampling S from its underlying distribution. This entails
/// allocating a buffer of size S.dist.n_rows * S.dist.n_cols, attaching that
/// buffer to S as S.buff, and finally sampling iid random variables to populate
/// S.buff. A flag is set on S so its destructor will deallocate S.buff.
/// 
/// By default, RandBLAS allocates and populates buffers for dense sketching operators
/// just before they are needed in some operation, and then it deletes these buffers
/// once the operation is complete. Calling this function bypasses that policy.
///
/// @param[in] S
///     A DenseSkOp object.
///    
template <typename DenseSkOp>
void fill_dense(DenseSkOp &S) {
    using T = typename DenseSkOp::scalar_t;
    randblas_require(S.buff == nullptr);
    randblas_require(S.dist.family != DenseDistName::BlackBox);
    S.buff = new T[S.dist.n_rows * S.dist.n_cols];
    fill_dense(S.dist, S.buff, S.seed_state);
    S.del_buff_on_destruct = true;
    return;
}

template <typename T, typename RNG>
DenseSkOp<T,RNG> submatrix_as_blackbox(const DenseSkOp<T,RNG> &S, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s) {
    T *buff = new T[n_rows * n_cols];
    auto dl = dist_to_layout(S.dist);
    fill_dense(dl, S.dist, n_rows, n_cols, ro_s, co_s, buff, S.seed_state);
    DenseDist submatrix_dist{n_rows, n_cols, DenseDistName::BlackBox, MajorAxis::Undefined};
    DenseSkOp<T,RNG> submatrix{n_rows, n_cols, submatrix_dist, S.seed_state, S.next_state, buff, dl, true};
    return submatrix;
}

}  // end namespace RandBLAS
