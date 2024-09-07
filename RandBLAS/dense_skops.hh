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
    int64_t major_len = dist.dim_major;
    int64_t minor_len = dist.dim_minor;
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

inline blas::Layout natural_layout(Axis major_axis, int64_t n_rows, int64_t n_cols) {
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
/// 
enum class ScalarDist : char {
    // ---------------------------------------------------------------------------
    ///  Indicates the Gaussian distribution with mean 0 and variance 1.
    Gaussian = 'G',

    // ---------------------------------------------------------------------------
    ///  Indicates the uniform distribution over [-r, r] where r := sqrt(3)
    ///  is the radius that provides for a variance of 1.
    Uniform = 'U'
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
    ///
    const Axis major_axis;

    // ---------------------------------------------------------------------------
    ///  Defined as
    ///  @verbatim embed:rst:leading-slashes
    ///
    ///  .. math::
    ///
    ///      \ttt{dim_major} = \begin{cases} \,\min\{ \ttt{n_rows},\, \ttt{n_cols} \} &\text{ if }~~ \ttt{major_axis} = \ttt{Short} \\ \max\{ \ttt{n_rows},\,\ttt{n_cols} \} & \text{ if } ~~\ttt{major_axis} = \ttt{Long} \end{cases}.
    ///
    ///  @endverbatim
    ///
    const int64_t dim_major;

    // ---------------------------------------------------------------------------
    ///  Defined as \math{\ttt{n_rows} + \ttt{n_cols} - \ttt{dim_major}.} This is
    ///  just whichever of \math{(\ttt{n_rows}, \ttt{n_cols})} wasn't identified
    ///  as \math{\ttt{dim_major}.}
    ///
    const int64_t dim_minor;

    // ---------------------------------------------------------------------------
    ///  A sketching operator sampled from this distribution should be multiplied
    ///  by this constant in order for sketching to preserve norms in expectation.
    const double isometry_scale;

    // ---------------------------------------------------------------------------
    ///  The distribution on \math{\mathbb{R}} for entries of operators sampled from this distribution.
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
    ///
    const blas::Layout natural_layout;

    // ---------------------------------------------------------------------------
    ///  Arguments passed to this function are used to initialize members of the same names.
    ///  The members \math{\ttt{dim_major},} \math{\ttt{dim_minor},} \math{\ttt{isometry_scale},}
    ///  and \math{\ttt{natural_layout}} are automatically initialized to be consistent with these arguments.
    ///  
    ///  This constructor will raise an error if \math{\min\\{\ttt{n_rows}, \ttt{n_cols}\\} \leq 0.}
    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        ScalarDist family = ScalarDist::Gaussian,
        Axis major_axis = Axis::Long
    ) :  // variable definitions
        n_rows(n_rows), n_cols(n_cols),
        major_axis(major_axis),
        dim_major((major_axis == Axis::Long) ? std::max(n_rows, n_cols) : std::min(n_rows, n_cols)),
        dim_minor((major_axis == Axis::Long) ? std::min(n_rows, n_cols) : std::max(n_rows, n_cols)),
        isometry_scale(std::pow(dim_minor, -0.5)),
        family(family),
        natural_layout(dense::natural_layout(major_axis, n_rows, n_cols))
    {   // argument validation
        randblas_require(n_rows > 0);
        randblas_require(n_cols > 0);
    }

};

#ifdef __cpp_concepts
static_assert(SketchingDistribution<DenseDist>);
#endif


// =============================================================================
///  A sample from a distribution over matrices whose entries are iid
///  mean-zero variance-one random variables.
///  This type conforms to the SketchingOperator concept.
template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    // ---------------------------------------------------------------------------
    /// Type alias.
    using distribution_t = DenseDist;

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
    ///  The state that should be passed to the RNG when the full 
    ///  operator needs to be sampled from scratch. 
    const state_t seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used in the next call to a random sampling function
    ///  whose output should be statistically independent from properties of this
    ///  operator.
    const state_t next_state;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Alias for dist.n_cols.
    const int64_t n_cols;

    // ----------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to this operator's
    ///  \math{\ttt{buff}} member. If true *at destruction time*, then delete []
    ///  will be called on \math{\ttt{buff}} if it is non-null.
    ///
    ///  RandBLAS only writes to this member at construction time.
    ///
    bool own_memory;

    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds the explicit representation of this operator
    ///  as a dense matrix.
    ///
    ///  If non-null this must point to an array of length at least 
    ///  \math{\ttt{dist.n_cols * dist.n_rows},} and this array must contain the 
    ///  random samples from \math{\ttt{dist}} implied by \math{\ttt{seed_state}.} See DenseSkOp::layout for more information.
    T *buff = nullptr; 

    // ---------------------------------------------------------------------------
    ///  The storage order for \math{\ttt{buff}.} The leading dimension of
    ///  \math{\mat(\ttt{buff})} when reading from \math{\ttt{buff}} is
    ///  \math{\ttt{dist.dim_major}.}
    const blas::Layout layout;


    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    // ---------------------------------------------------------------------------
    ///  Arguments passed to this function are 
    ///  used to initialize members of the same names. \math{\ttt{own_memory}} is initialized to true,
    ///  \math{\ttt{buff}} is initialized to nullptr, and \math{\ttt{layout}} is initialized
    ///  to \math{\ttt{dist.natural_layout}.} The \math{\ttt{next_state}} member is 
    ///  computed automatically from \math{\ttt{dist}} and \math{\ttt{next_state}.}
    ///
    ///  Although \math{\ttt{own_memory}} is initialized to true, RandBLAS will not attach
    ///  memory to \math{\ttt{buff}} unless fill_dense(DenseSkOp &S) is called. 
    ///
    ///  If RandBLAS function needs an explicit representation of this operator and
    ///  yet \math{\ttt{buff}} is null, then RandBLAS will construct a temporary
    ///  explicit representation of this operator and delete that representation before returning.
    ///  
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
        // ^ We won't take advantage of own_memory unless this is passed to fill_dense(S).
        buff(nullptr),
        layout(dist.natural_layout) { }

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
        // ^ Our memory management policy prohibits us from changing
        //   S.own_memory after S was constructed. But overwriting
        //   S.buff with the null pointer is allowed since we 
        //   can gaurantee that won't cause a memory leak.
    }

    //  Destructor
    ~DenseSkOp() {
        if (own_memory && buff != nullptr) {
            delete [] buff;
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
/// This function provides the underlying implementation of fill_dense(DenseSkOp &S).
/// Unlike fill_dense(DenseSkOp &S), this function can be used to generate explicit representations
/// of *submatrices* of dense sketching operators.
///
/// Formally, if \math{\mtxS} were sampled from \math{\D} with \math{\ttt{seed_state=seed}},
/// then on exit we'd have
///
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(\buff)_{ij} = \mtxS_{(i+\ioff)(\joff + j)}
///
/// where :math:`\mat(\cdot)` reads from :math:`\buff` in :math:`\layout` order.
/// @endverbatim
/// If \math{\ttt{layout != dist.natural_layout}}
/// then this function internally allocates \math{\ttt{n_rows * n_cols}} words of workspace,
/// and deletes this workspace before returning.
///
/// @verbatim embed:rst:leading-slashes
/// .. dropdown:: Full parameter descriptions
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
RNGState<RNG> fill_dense_unpacked(blas::Layout layout, const DenseDist &D, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s, T* buff, const RNGState<RNG> &seed) {
    using RandBLAS::dense::fill_dense_submat_impl;
    randblas_require(D.n_rows >= n_rows + ro_s);
    randblas_require(D.n_cols >= n_cols + co_s);
    blas::Layout natural_layout = D.natural_layout;
    int64_t ma_len = D.dim_major;
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
    return fill_dense_unpacked(D.natural_layout, D, D.n_rows, D.n_cols, 0, 0, buff, seed);
}

// =============================================================================
/// If \math{\ttt{S.own_memory}} is true then we enter an allocation stage. If
/// \math{\ttt{S.buff}} is equal to \math{\ttt{nullptr}} then it is redirected to the
/// start of an array allocated with ``new []``. The length of any allocated
/// array is \math{\ttt{S.n_rows * S.n_cols}.} 
///
/// After the allocation stage, we check \math{\ttt{S.buff}} and we raise
/// an error if it's null.
///
/// If \math{\ttt{S.buff}} is are non-null, then we'll assume it has length at least
///  \math{\ttt{S.n_rows * S.n_cols}.} We'll proceed to populate \math{\ttt{S.buff}} 
/// with the data for the explicit representation of \math{\ttt{S}.}
/// On exit, one can encode a BLAS-style representation of \math{\ttt{S}} with the tuple
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     
///     (\ttt{S.layout},~\ttt{S.n_rows},~\ttt{S.n_cols},~\ttt{S.buff},~\ttt{S.dist.dim_major})
///
/// In BLAS parlance, the last element of this tuple would be called the "leading dimension"
/// of :math:`\ttt{S}.`
/// @endverbatim
template <typename DenseSkOp>
void fill_dense(DenseSkOp &S) {
    if (S.own_memory && S.buff == nullptr) {
        using T = typename DenseSkOp::scalar_t;
        S.buff = new T[S.n_rows * S.n_cols];
    }
    randblas_require(S.buff != nullptr);
    fill_dense_unpacked(S.layout, S.dist, S.n_rows, S.n_cols, 0, 0, S.buff, S.seed_state);
    return;
}

template <typename T>
struct BLASFriendlyOperator {
    using scalar_t = T;
    const blas::Layout layout;
    const int64_t n_rows;
    const int64_t n_cols;
    T* buff;
    const int64_t ldim;
    const bool own_memory;

    ~BLASFriendlyOperator() {
        if (own_memory && buff != nullptr) {
            delete [] buff;
        }
    }
};

template <typename BFO, typename DenseSkOp>
BFO submatrix_as_blackbox(const DenseSkOp &S, int64_t n_rows, int64_t n_cols, int64_t ro_s, int64_t co_s) {
    randblas_require(ro_s + n_rows <= S.n_rows);
    randblas_require(co_s + n_cols <= S.n_cols);
    using T = typename DenseSkOp::scalar_t;
    T *buff = new T[n_rows * n_cols];
    auto layout = S.layout;
    fill_dense_unpacked(layout, S.dist, n_rows, n_cols, ro_s, co_s, buff, S.seed_state);
    int64_t dim_major = S.dist.dim_major;
    BFO submatrix{layout, n_rows, n_cols, buff, dim_major, true};
    return submatrix;
}

}  // end namespace RandBLAS
