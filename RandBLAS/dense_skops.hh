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
    if (dist.major_axis == Axis::Undefined) {
        // implies dist.family = ScalarDist::BlackBox
        throw std::invalid_argument("Cannot compute next_state when dist.family is BlackBox");
    }
    // ^ This is the only place where Axis is actually used to some 
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

// We only template this function because ScalarDistribution has defined later.
template <typename ScalarDistribution>
inline double isometry_scale(ScalarDistribution sd, int64_t n_rows, int64_t n_cols) {
    return (sd == ScalarDistribution::BlackBox) ? 1.0 : std::pow(std::min(n_rows, n_cols), -0.5);
}

inline blas::Layout natural_layout(Axis major_axis, int64_t n_rows, int64_t n_cols) {
    if (major_axis == Axis::Undefined || n_rows == n_cols)
        return blas::Layout::ColMajor;
    bool is_wide = n_rows < n_cols;
    bool fa_long = major_axis == Axis::Long;
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

} // end namespace RandBLAS::dense


namespace RandBLAS {

// =============================================================================
/// We support two distributions for dense sketching operators: those whose
/// entries are iid Gaussians or iid uniform over a symmetric interval.
/// For implementation reasons, we also expose an option to indicate that an
/// operator's distribution is unknown but it is still represented by a buffer
/// that can be used in GEMM.
enum class ScalarDist : char {
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
///  A distribution over matrices whose entries are iid mean-zero variance-one
///  random variables.
///  This type conforms to the SketchingDistribution concept.
struct DenseDist {
    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many columns.
    const int64_t n_cols;

    // ---------------------------------------------------------------------------
    ///  This member affects whether samples from this distribution have their
    ///  entries filled row-wise or column-wise. While there is no statistical 
    ///  difference between these two filling orders, there are situations
    ///  where one order or the other might be preferred.
    ///
    ///  For more information, see the DenseDist::natural_layout and the section of the
    ///  RandBLAS tutorial on
    ///  @verbatim embed:rst:inline :ref:`updating sketches <sketch_updates>`. @endverbatim 
    const Axis major_axis;

    // ---------------------------------------------------------------------------
    ///  A sketching operator sampled from this distribution should be multiplied
    ///  by this constant in order for sketching to preserve norms in expectation.
    const double isometry_scale;

    // ---------------------------------------------------------------------------
    ///  The distribution used for the entries of the sketching operator.
    const ScalarDist family;

    // ---------------------------------------------------------------------------
    ///  @verbatim embed:rst:leading-slashes
    ///  The fill order (row major or column major) implied by major_axis,
    ///  n_rows, and n_cols, according to the following table.
    ///        .. list-table::
    ///           :widths: 34 33 33
    ///           :header-rows: 1
    ///
    ///           * -  
    ///             - :math:`\texttt{major_axis} = \texttt{Long}`
    ///             - :math:`\texttt{major_axis} = \texttt{Short}`
    ///           * - :math:`\texttt{n_rows} > \texttt{n_cols}`
    ///             - column major
    ///             - row major
    ///           * - :math:`\texttt{n_rows} \leq \texttt{n_cols}`
    ///             - row major
    ///             - column major
    ///
    ///  If you want to sample a dense sketching operator represented as 
    ///  buffer in a layout different than the one given here, then a 
    ///  change-of-layout has to be performed explicitly. 
    ///  @endverbatim
    const blas::Layout natural_layout;

    // ---------------------------------------------------------------------------
    ///  This constructor is the preferred way to instantiate DenseDist objects.
    ///  It correctly sets isometry_scale and natural_layout as a function of the
    ///  other members. Optional trailing arguments can be used to specify the
    ///  family or major_axis members.
    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        ScalarDist family = ScalarDist::Gaussian,
        Axis ma = Axis::Long
    ) :  // variable definitions
        n_rows(n_rows), n_cols(n_cols), major_axis(ma),
        isometry_scale(dense::isometry_scale(family, n_rows, n_cols)),
        family(family),
        natural_layout(dense::natural_layout(ma, n_rows, n_cols))
    {   // argument validation
        randblas_require(n_rows > 0);
        randblas_require(n_cols > 0);
        if (family == ScalarDist::BlackBox) {
            randblas_require(ma == Axis::Undefined);
        } else {
            randblas_require(ma != Axis::Undefined);
        }  
    }

};

#ifdef __cpp_concepts
static_assert(SketchingDistribution<DenseDist>);
#endif

inline int64_t major_axis_length(const DenseDist &D) {
    randblas_require(D.major_axis != Axis::Undefined);
    return (D.major_axis == Axis::Long) ? 
        std::max(D.n_rows, D.n_cols) : std::min(D.n_rows, D.n_cols);
}


// =============================================================================
///  A sample from a distribution over matrices whose entries are iid
///  mean-zero variance-one random variables.
///  This type conforms to the SketchingOperator concept.
template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    // ---------------------------------------------------------------------------
    /// Type alias.
    using state_t = RNGState<RNG>;

    // ---------------------------------------------------------------------------
    /// Real scalar type used in matrix representations of this operator.
    using scalar_t = T;

    // ---------------------------------------------------------------------------
    ///  The distribution from which this operator is sampled;
    ///  this member specifies the number of rows and columns of this operator.
    const DenseDist dist;

    // ---------------------------------------------------------------------------
    ///  The state that should be passed to the RNG when the full sketching 
    ///  operator needs to be sampled from scratch. 
    const state_t seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used by the next call to an RNG *after* the
    ///  full sketching operator has been sampled.
    /// 
    ///  The memory-owning constructor sets next_state automatically as a function
    ///  of seed_state and dist. This automatic determination will raise an error
    ///  if dist.family == ScalarDist::BlackBox.
    const state_t next_state;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_cols.
    const int64_t n_cols;

    // ---------------------------------------------------------------------------
    ///  If own_memory = true then RandBLAS *can* store an explicit
    ///  representation of this sketching operator in \math{\ttt{buff},} and it
    ///  *must* free any memory that \math{\ttt{buff}} points to
    ///  when the operator's destructor is invoked.
    /// 
    ///  This member is set automatically based on whether the memory-owning
    ///  or view constructor is called.
    bool own_memory;

    // ---------------------------------------------------------------------------
    ///  The memory-owning DenseSkOp constructor initializes \math{\ttt{buff}} as the null pointer.
    ///  Whenever \math{\ttt{buff}} is the null pointer, this sketching operator can
    ///  be used in any of RandBLAS' sketching functions. (Those functions will take
    ///  responsibility for allocating workspace and performing random sampling as
    ///  needed, and they will deallocate that workspace before returning.)
    ///
    ///  If \math{\ttt{buff}} is non-null then we assume its length is at least 
    ///  \math{\ttt{dist.n_cols * dist.n_rows}} and that it contains the
    ///  random samples from \math{\ttt{dist}} implied by \math{\ttt{seed_state}.}
    ///  The contents of \math{\ttt{buff}} will be read in \math{\ttt{layout}} order
    ///  using the smallest value for the leading dimension that's consistent with
    ///  the \math{(\ttt{n_rows},\ttt{n_cols})}.
    T *buff = nullptr; 

    // ---------------------------------------------------------------------------
    ///  The storage order that should be used for any read or write operations
    ///  with \math{\ttt{buff}.} 
    ///
    ///  The memory-owning DenseSkOp constructor automatically initializes this 
    ///  to \math{\ttt{dist.natural_layout}.}
    blas::Layout layout;


    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    ///---------------------------------------------------------------------------
    ///  **Memory-owning constructor**. This constructor initializes the 
    ///  operator's \math{\ttt{buff}} member to the null pointer. Any array pointed to by 
    ///  \math{\ttt{buff}} will be deleted when this operator's destructor is invoked.
    ///
    ///  Using this operator to some productive end will inevitably require memory allocation
    ///  and random sampling. RandBLAS will handle these steps automatically when needed
    ///  as long as \math{\ttt{buff}} is the null pointer. 
    ///
    /// @endverbatim
    DenseSkOp(
        DenseDist dist,
        const state_t &seed_state
    ) : // variable definitions
        dist(dist),
        seed_state(seed_state),
        next_state(dense::compute_next_state(dist, seed_state)),
        n_rows(dist.n_rows),
        n_cols(dist.n_cols),
        own_memory(true),
        // We won't take advantage of own_memory unless this is passed to fill_dense(S).
        // Still, I think it's reasonable to default to ownership in this case, because if
        // ther user happened to have memory they wanted us to use then they could have just
        // called the other constructor. Except -- that isn't true. The user might not
        // want to deal with next_state.
        buff(nullptr),
        layout(dist.natural_layout) { }

    ///---------------------------------------------------------------------------
    /// **View constructor**. The arguments provided to this
    /// function are used to initialize members of the same names, with no error checking.
    /// The \math{\ttt{own_memory}} member is set to false.
    /// The user takes all responsibility for ensuring that the semantics of
    /// \math{\ttt{buff}} and \math{\ttt{layout}} are respected.
    ///
    DenseSkOp(
        DenseDist dist,
        const state_t &seed_state,
        const state_t &next_state,
        // ^ It would be nice to set next_state in an initializer list based on seed_state like we do with SparseSkOp.
        //   We can't do that since the possibility of dist.family == BlackBox means we might be allowed to handle
        //   random number generation. When this constructor is used it's the user's responsibility to set next_state
        //   correctly based on the value of buff (or the value of buff that they intend to use eventually). If a user
        //   is confident that they won't need next_state then they can just set it to state_t(0).
        T *buff,
        blas::Layout layout
    ) : 
        dist(dist), seed_state(seed_state), next_state(next_state),
        n_rows(dist.n_rows), n_cols(dist.n_cols), 
        own_memory(false), buff(buff), layout(layout) { }

    //  Move constructor
    DenseSkOp(
        DenseSkOp<T,RNG> &&S
    ) : // Initializations
        dist(S.dist),
        seed_state(S.seed_state),
        next_state(S.next_state),
        n_rows(dist.n_rows), n_cols(dist.n_cols),
        own_memory(S.own_memory), buff(S.buff), layout(S.layout)
    {   // Body
        S.buff = nullptr;
        // ^ Since own_memory is const, the only way we can protect
        //   this the original contents of S.buff from deletion is
        //   is to reassign S.buff to the null pointer.
    }

    //  Destructor
    ~DenseSkOp() {
        if (this->own_memory && !(this->buff == nullptr)) {
            delete [] this->buff;
        }
    }
};

#ifdef __cpp_concepts
static_assert(SketchingOperator<DenseSkOp<float>>);
static_assert(SketchingOperator<DenseSkOp<double>>);
#endif

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |buff|  mathmacro:: \mathtt{buff}
///   .. |nrows| mathmacro:: \mathtt{n\_rows}
///   .. |ncols| mathmacro:: \mathtt{n\_cols}
///   .. |ioff| mathmacro:: \mathtt{ro\_s}
///   .. |joff| mathmacro:: \mathtt{co\_s}
///   .. |layout| mathmacro:: \mathtt{layout}
///
/// @endverbatim
/// Fill \math{\buff} so that (1) \math{\mat(\buff)} is a submatrix of
/// an _implicit_ random sample from \math{\D}, and (2) \math{\mat(\buff)}
/// is determined by reading from \math{\buff} in \math{\layout} order.
/// 
/// If we denote the implicit sample from \math{\D} by \math{\mtxS}, then on exit
/// we have
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(\buff)_{ij} = \mtxS_{(i+\ioff)(\joff + j)}
/// @endverbatim
/// assuming the standard convention of zero-indexing.
///
/// This function is for generating low-level representations of matrices
/// that are equivalent to a submatrix of a RandBLAS DenseSkOp, but 
/// without using the DenseSkOp abstraction. This can be useful if you want
/// to sketch a structured matrix that RandBLAS doesn't support (like a symmetric
/// matrix whose values are only stored in the upper or lower triangle).
///
/// @verbatim embed:rst:leading-slashes
/// .. dropdown:: Full parmaeter descriptions
///   :animate: fade-in-slide-down
///
///     layout      
///      - blas::Layout::RowMajor or blas::Layout::ColMajor
///      - The storage order for :math:`\mat(\buff)` on exit. The leading dimension
///        is the smallest value permitted for a matrix of dimensions (n_rows, n_cols)
///        in the given layout. I.e., it's n_rows if layout == ColMajor and 
///        n_cols if layout == RowMajor.
///      - Note that since the entries of :math:`\buff` are sampled iid from a common
///        distribution, the value of :math:`\layout` is unlikely to have mathematical significance.
///        However, the value of :math:`\layout` can affect this function's efficiency.
///        For best efficiency we recommend :math:`\ttt{layout=}\D{}\ttt{.natural_layout}.`
///        If a different value of :math:`\layout` is used, then this function will internally
///        allocate extra memory for an out-of-place layout change.
///
///     D
///      - A DenseDist object.
///      - A distribution over random matrices of shape (D.n_rows, D.n_cols).
///
///     n_rows
///      - A positive integer.
///      - The number of rows in :math:`\mat(\buff).`
///
///     n_cols
///      - A positive integer.
///      - The number of columns in :math:`\mat(\buff).`
///
///     ro_s
///      - A nonnegative integer.
///      - The row offset for :math:`\mat(\buff)` as a submatrix of :math:`\mtxS.` 
///      - We require that :math:`\ioff + \nrows` is at most D.n_rows.
///
///     co_s
///      - A nonnegative integer.
///      - The column offset for :math:`\mat(\buff)` as a submatrix of :math:`\mtxS.` 
///      - We require that :math:`\joff + \ncols` is at most D.n_cols.
///
///     buff
///      - Buffer of type T.
///      - Length must be at least :math:`\nrows \cdot \ncols.`
///
///     seed
///      - A CBRNG state
///      - Used to define :math:`\mtxS` as a sample from :math:`\D.`
///
/// @endverbatim
template<typename T, typename RNG = r123::Philox4x32>
RNGState<RNG> fill_dense(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s, T* buff, const RNGState<RNG> &seed) {
    using RandBLAS::dense::fill_dense_submat_impl;
    randblas_require(D.n_rows >= n_rows + ro_s);
    randblas_require(D.n_cols >= n_cols + co_s);
    blas::Layout natural_layout = D.natural_layout;
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
        case ScalarDist::Gaussian: {
            next_state = fill_dense_submat_impl<T,RNG,r123ext::boxmul>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            break;
        }
        case ScalarDist::Uniform: {
            next_state = fill_dense_submat_impl<T,RNG,r123ext::uneg11>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            blas::scal(n_rows_ * n_cols_, (T)std::sqrt(3), buff, 1);
            break;
        }
        case ScalarDist::BlackBox: {
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
/// Fill \math{\buff} so that \math{\mat(\buff)} is a sample from \math{\D} using
/// seed \math{\mathtt{seed}.} The buffer's layout is \math{\D\ttt{.natural_layout}.}
///
/// @param[in] D
///      A DenseDist object.
/// @param[in] buff
///     Buffer of type T.
///     - Length must be at least D.n_rows * D.n_cols.
///     - The leading dimension of \math{\mat(\buff)} when reading from \math{\buff}
///       is either D.n_rows or D.n_cols, depending on \math{\D\ttt{.natural_layout}.}
/// @param[in] seed
///      A CBRNG state
///      - Used to define \math{\mat(\buff)} as a sample from \math{\D}.
///
template <typename T, typename RNG = r123::Philox4x32>
RNGState<RNG> fill_dense(const DenseDist &D, T *buff, const RNGState<RNG> &seed) {
    return fill_dense(D.natural_layout, D, D.n_rows, D.n_cols, 0, 0, buff, seed);
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
template <typename DenseSkOp>
void fill_dense(DenseSkOp &S) {
    if (S.own_memory && S.buff == nullptr) {
        using T = typename DenseSkOp::scalar_t;
        S.buff = new T[S.n_rows * S.n_cols];
    }
    randblas_require(S.buff != nullptr);
    fill_dense(S.dist, S.buff, S.seed_state);
    return;
}

template <typename DenseSkOp>
DenseSkOp submatrix_as_blackbox(const DenseSkOp &S, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s) {
    randblas_require(ro_s + n_rows <= S.n_rows);
    randblas_require(co_s + n_cols <= S.n_cols);
    using T = typename DenseSkOp::scalar_t;
    T *buff = new T[n_rows * n_cols];
    auto layout = S.dist.natural_layout;
    fill_dense(layout, S.dist, n_rows, n_cols, ro_s, co_s, buff, S.seed_state);
    DenseDist submatrix_dist(n_rows, n_cols, ScalarDist::BlackBox, Axis::Undefined);
    DenseSkOp submatrix(submatrix_dist, S.seed_state, S.next_state, buff, layout);
    submatrix.own_memory = true;
    return submatrix;
}

}  // end namespace RandBLAS
