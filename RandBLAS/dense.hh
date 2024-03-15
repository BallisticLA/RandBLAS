#ifndef randblas_dense_hh
#define randblas_dense_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"

#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>

#include <math.h>
#include <typeinfo>


namespace RandBLAS {
// =============================================================================
/// We call a sketching operator "dense" if (1) it is naturally represented with a
/// buffer and (2) the natural way to apply that operator to a matrix is
/// to use the operator's buffer in GEMM.
///
/// We support two distributions for dense sketching operators: those whose
/// entries are iid Gaussians or iid uniform over a symmetric interval.
/// For implementation reasons, we also expose an option to indicate that an
/// operator's distribution is unknown but it is still represented by a buffer
/// that can be used in GEMM.
enum class DenseDistName : char {
    // ---------------------------------------------------------------------------
    ///  Indicates the Gaussian distribution with mean 0 and standard deviation 1.
    Gaussian = 'G',

    // ---------------------------------------------------------------------------
    ///  Indicates the uniform distribution over [-1, 1].
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
    ///  This member indirectly sets the storage order of buffers of
    ///  sketching operators that are sampled from this distribution.
    ///
    ///  We note that the storage order of a DenseSkOp's underlying buffer does not
    ///  affect whether the operator can be applied to row-major or column-major data.
    ///  Mismatched data layouts are resolved automatically and 
    ///  with zero copies inside RandBLAS::sketch_general. Therefore most users need
    ///  not spend any brain power thinking about how this value should be set.
    ///  
    /// @verbatim embed:rst:leading-slashes
    ///  *Notes for experts*. Deciding the value of this member is only needed
    ///  in algorithms where (1) there's a need to iteratively generate panels of
    ///  a larger sketching operator and (2) one of larger operator's dimensions
    ///  cannot be known before the  iterative process starts.
    ///
    ///  Essentially, column-major storage order lets us
    ///  stack operators horizontally in a consistent way, while row-major storage order
    ///  lets us stack operators vertically in a consistent way. The mapping from
    ///  major_axis to storage order is given in the table below.
    /// 
    ///     .. list-table::
    ///        :widths: 34 33 33
    ///        :header-rows: 1
    ///
    ///        * -  
    ///          - :math:`\texttt{major_axis} = \texttt{Long}`
    ///          - :math:`\texttt{major_axis} = \texttt{Short}`
    ///        * - :math:`\texttt{n_rows} > \texttt{n_cols}`
    ///          - column major
    ///          - row major
    ///        * - :math:`\texttt{n_rows} \leq \texttt{n_cols}`
    ///          - row major
    ///          - column major
    /// @endverbatim
    const MajorAxis major_axis;

    // ---------------------------------------------------------------------------
    ///  A distribution over matrices of shape (n_rows, n_cols) with entries drawn
    ///  iid from either the standard normal distribution or the uniform distribution
    ///  over [-1, 1]. 
    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        DenseDistName dn = DenseDistName::Gaussian
    ) : n_rows(n_rows), n_cols(n_cols), family(dn), major_axis(MajorAxis::Long) {
        randblas_require(dn != DenseDistName::BlackBox);
    };

    // Only use with struct initializer.
    DenseDist(
        int64_t n_rows,
        int64_t n_cols,
        DenseDistName dn,
        MajorAxis ma
    ) : n_rows(n_rows), n_cols(n_cols), family(dn), major_axis(ma) { };

};


inline blas::Layout dist_to_layout(
    DenseDist D
) {
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

inline int64_t major_axis_length(
    DenseDist D
) {
    return (D.major_axis == MajorAxis::Long) ? 
        std::max(D.n_rows, D.n_cols) : std::min(D.n_rows, D.n_cols);
}

// =============================================================================
/// A sample from a distribution over dense sketching operators.
///
template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    using generator = RNG;
    using state_type = RNGState<RNG>;
    using buffer_type = T;

    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to dense sketching operators
    //
    /////////////////////////////////////////////////////////////////////

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
    RNGState<RNG> next_state;

    T *buff = nullptr;                         // memory
    const blas::Layout layout;                 // matrix storage order
    bool del_buff_on_destruct = false;         // only applies if fill_dense(S) has been called.

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    DenseSkOp(
        DenseDist dist,
        RNGState<RNG> const& state,
        T *buff = nullptr
    );

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDist dist,
        uint32_t key,
        T *buff = nullptr
    ) : DenseSkOp(dist, RNGState<RNG>(key), buff) {};

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        uint32_t key,
        T *buff = nullptr,
        MajorAxis ma = MajorAxis::Long
    ) : DenseSkOp(DenseDist{n_rows, n_cols, family, ma}, RNGState<RNG>(key), buff) {};

    // Destructor
    ~DenseSkOp();
};

template <typename T, typename RNG>
DenseSkOp<T,RNG>::DenseSkOp(
    DenseDist dist,
    RNGState<RNG> const& state,
    T *buff
) : // variable definitions
    dist(dist),
    seed_state(state),
    next_state{},
    buff(buff),
    layout(dist_to_layout(dist))
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    if (dist.family == DenseDistName::BlackBox)
        randblas_require(this->buff != nullptr);
}

template <typename T, typename RNG>
DenseSkOp<T,RNG>::~DenseSkOp() {
    if (this->del_buff_on_destruct) {
        delete [] this->buff;
    }
}

template<typename RNG>
bool compare_ctr(typename RNG::ctr_type c1, typename RNG::ctr_type c2) {
    int len = c1.size();
    
    for (int ind = len - 1; ind >= 0; ind--) {
        if (c1[ind] > c2[ind]) {
            return true;
        } else if (c1[ind] < c2[ind]) {
            return false;
        }
    }
    return false;
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
static RNGState<RNG> fill_dense_submat_impl(
    int64_t n_cols,
    T* smat,
    int64_t n_srows,
    int64_t n_scols,
    int64_t ptr,
    const RNGState<RNG> & seed,
    int64_t lda = 0
) {
    if (lda <= 0) {
        lda = n_scols;
    } else {
        randblas_require(lda >= n_scols);
    }
    randblas_require(n_cols >= n_scols);
    RNG rng;
    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type k = seed.key;
    
    int64_t pad = 0;
    // ^ computed such that  n_cols+pad is divisible by RNG::static_size
    if (n_cols % RNG::ctr_type::static_size != 0) {
        pad = RNG::ctr_type::static_size - n_cols % RNG::ctr_type::static_size;
    }

    int64_t n_cols_padded = n_cols + pad;
    // ^ smallest number of columns, greater than or equal to n_cols, that would be divisible by RNG::ctr_type::static_size 
    int64_t ptr_padded = ptr + ptr / n_cols * pad;
    // ^ ptr corresponding to the padded matrix
    int64_t r0_padded = ptr_padded / RNG::ctr_type::static_size;
    // ^ starting counter corresponding to ptr_padded 
    int64_t r1_padded = (ptr_padded + n_scols - 1) / RNG::ctr_type::static_size;
    // ^ ending counter corresponding to ptr of the last element of the row
    int64_t ctr_gap = n_cols_padded / RNG::ctr_type::static_size; 
    // ^ number of counters between the first counter of the row to the first counter of the next row;
    int64_t s0 = ptr_padded % RNG::ctr_type::static_size; 
    int64_t e1 = (ptr_padded + n_scols - 1) % RNG::ctr_type::static_size;

    int64_t num_thrds = 1;
#if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel 
    {
        num_thrds = omp_get_num_threads();
    }
#endif

    //Instead of using thrd_arr just initialize ctr_arr to be zero counters;
    typename RNG::ctr_type ctr_arr[num_thrds];
    for (int i = 0; i < num_thrds; i++) {
        ctr_arr[i] = c;
    }

    #pragma omp parallel firstprivate(c, k)
    {

    auto cc = c;
    int64_t prev = 0;
    int64_t i;
    int64_t r0, r1;
    int64_t ind;
    int64_t thrd = 0;

    #pragma omp for
    for (int row = 0; row < n_srows; row++) {
        
    #if defined(RandBLAS_HAS_OpenMP)
        thrd = omp_get_thread_num();
    #endif

        ind = 0;
        r0 = r0_padded + ctr_gap*row;
        r1 = r1_padded + ctr_gap*row; 

        cc.incr(r0 - prev);
        prev = r0;
        auto rv =  OP::generate(rng, cc, k);
        int64_t range = (r1 > r0)? RNG::ctr_type::static_size-1 : e1;
        for (i = s0; i <= range; i++) {
            smat[ind + row * lda] = rv[i];
            ind++;
        }
        // middle 
        int64_t tmp = r0;
        while( tmp < r1 - 1) {
            cc.incr();
            prev++;
            rv = OP::generate(rng, cc, k);
            for (i = 0; i < RNG::ctr_type::static_size; i++) {
                smat[ind + row * lda] = rv[i];
                ind++;
            }
            tmp++;
        }

        // end
        if ( r1 > r0 ) {
            cc.incr();
            prev++;
            rv = OP::generate(rng, cc, k);
            for (i = 0; i <= e1; i++) {
                smat[ind + row * lda] = rv[i];
                ind++;
            }
        }
        ctr_arr[thrd] = cc;
    }

    }
    
    //finds the largest counter in the counter array
    typename RNG::ctr_type max_c = ctr_arr[0];
    for (int i = 1; i < num_thrds; i++) {  
        if (compare_ctr<RNG>(ctr_arr[i], max_c)) {
            max_c = ctr_arr[i];
        }
    }

    max_c.incr();
    return RNGState<RNG> {max_c, k};
}

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |mat|   mathmacro:: \operatorname{mat}
///   .. |buff|  mathmacro:: \mathtt{buff}
///   .. |D|     mathmacro:: \mathcal{D}
///   .. |nrows| mathmacro:: \mathtt{n\_rows}
///   .. |ncols| mathmacro:: \mathtt{n\_cols}
///   .. |ioff| mathmacro:: \mathtt{i\_off}
///   .. |joff| mathmacro:: \mathtt{j\_off}
///
/// @endverbatim
/// Fill \math{\buff} so that \math{\mat(\buff)} is a submatrix of
/// an _implicit_ random sample from \math{\D}.
/// 
/// If we denote the implicit sample from \math{\D} by \math{S}, then we have
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(\buff) = S[\ioff:(\ioff + \nrows),\, \joff:(\joff + \ncols)]
/// @endverbatim
/// on exit.
///
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
///      - The row offset for \math{\mat(\buff)} as a submatrix of \math{S}. 
///      - We require that \math{\ioff + \nrows} is at most D.n_rows.
/// @param[in] co_s
///      A nonnegative integer.
///      - The column offset for \math{\mat(\buff)} as a submatrix of \math{S}. 
///      - We require that \math{\joff + \ncols} is at most D.n_cols.
/// @param[in] buff
///     Buffer of type T.
///     - Length must be at least \math{\nrows \cdot \ncols}.
///     - The leading dimension of \math{\mat(\buff)} when reading from \math{\buff}
///       is either \math{\nrows} or \math{\ncols}, depending on the return value of this function
///       that indicates row-major or column-major layout.
/// @param[in] seed
///      A CBRNG state
///      - Used to define \math{S} as a sample from \math{\D}.
///
/// @returns
///     A std::pair consisting of "layout" and "next_state".
///     - \math{\buff} must be read in "layout" order 
///       to recover \math{\mat(\buff)}. This layout is determined
///       from \math{\D} and cannot be controlled directly.
///     - If this function returns a layout that is undesirable then it is
///       the caller's responsibility to perform a transpose as needed.
/// 
template<typename T, typename RNG>
std::pair<blas::Layout, RandBLAS::RNGState<RNG>> fill_dense(
    const DenseDist &D,
    int64_t n_rows,
    int64_t n_cols,
    int64_t ro_s,
    int64_t co_s,
    T* buff,
    const RNGState<RNG> &seed
) {
    randblas_require(D.n_rows >= n_rows + ro_s);
    randblas_require(D.n_cols >= n_cols + co_s);
    blas::Layout layout = dist_to_layout(D);
    int64_t ma_len = major_axis_length(D);
    int64_t n_rows_, n_cols_, ptr;
    if (layout == blas::Layout::ColMajor) {
        // operate on the transpose in row-major
        n_rows_ = n_cols;
        n_cols_ = n_rows;
        ptr = ro_s + co_s * ma_len;
    } else {
        n_rows_ = n_rows;
        n_cols_ = n_cols;
        ptr = ro_s * ma_len + co_s;
    }
    switch (D.family) {
        case DenseDistName::Gaussian: {
            auto next_state_g = fill_dense_submat_impl<T,RNG,r123ext::boxmul>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            return std::make_pair(layout, next_state_g);
        }
        case DenseDistName::Uniform: {
            auto next_state_u = fill_dense_submat_impl<T,RNG,r123ext::uneg11>(ma_len, buff, n_rows_, n_cols_, ptr, seed);
            return std::make_pair(layout, next_state_u);
        }
        case DenseDistName::BlackBox: {
            throw std::invalid_argument(std::string("fill_buff cannot be called with the BlackBox distribution."));
        }
        default: {
            throw std::runtime_error(std::string("Unrecognized distribution."));
        }
    }
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
template <typename T, typename RNG>
std::pair<blas::Layout, RandBLAS::RNGState<RNG>> fill_dense(
    const DenseDist &D,
    T *buff,
    const RNGState<RNG> &seed
) {
    return fill_dense(D, D.n_rows, D.n_cols, 0, 0,  buff, seed);
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
/// @return
///     An RNGState object. This is the state that should be used the next 
///     time the program needs to generate random numbers for use in a randomized
///     algorithm.
///    
template <typename T, typename RNG>
RNGState<RNG> fill_dense(
    DenseSkOp<T,RNG> &S
) {
    randblas_require(!S.buff);
    randblas_require(S.dist.family != DenseDistName::BlackBox);
    S.buff = new T[S.dist.n_rows * S.dist.n_cols];
    auto [layout, next_state] = fill_dense<T, RNG>(S.dist, S.buff, S.seed_state);
    S.next_state = next_state;
    S.del_buff_on_destruct = true;
    return next_state;
}
}  // end namespace RandBLAS

namespace RandBLAS::dense {

using namespace RandBLAS;

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |opA| mathmacro:: \mathrm{opA}
///   .. |opS| mathmacro:: \mathrm{opS}
///
/// @endverbatim
/// LSKGE3: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator that takes Level 3 BLAS effort to apply.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}
///     - The number of rows in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(A))}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\op(\submat(S))}
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] S
///    A DenseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] ro_s
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{ro_s}, :]}.
///
/// @param[in] co_s
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{co_s}]}. 
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename RNG>
void lskge3(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    DenseSkOp<T,RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
){
    auto [rows_submat_S, cols_submat_S] = dims_before_op(d, m, opS);
    if (!S.buff) {
        // We'll make a shallow copy of the sketching operator, take responsibility for filling the memory
        // of that sketching operator, and then call LSKGE3 with that new object.
        T *buff = new T[rows_submat_S * cols_submat_S];
        fill_dense(S.dist, rows_submat_S, cols_submat_S, ro_s, co_s, buff, S.seed_state);
        DenseDist D{rows_submat_S, cols_submat_S, DenseDistName::BlackBox, S.dist.major_axis};
        DenseSkOp S_(D, S.seed_state, buff);
        lskge3(layout, opS, opA, d, n, m, alpha, S_, 0, 0, A, lda, beta, B, ldb);
        delete [] buff;
        return;
    }
    randblas_require( S.dist.n_rows >= rows_submat_S + ro_s );
    randblas_require( S.dist.n_cols >= cols_submat_S + co_s );
    auto [rows_A, cols_A] = dims_before_op(m, n, opA);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.dist.n_rows, S.dist.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    blas::gemm(layout, opS, opA, d, n, m, alpha, S_ptr, lds, A, lda, beta, B, ldb);
    return;
}

// =============================================================================
/// RSKGE3: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator that takes Level 3 BLAS effort to apply.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, n, d)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}.
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\op(\mat(A))}
///     - The number of rows in \math{\op(\submat(S))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] S
///    A DenseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] ro_s
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{ro_s}, :]}.
///
/// @param[in] co_s
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{co_s}]}. 
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename RNG>
void rskge3(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    DenseSkOp<T,RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
){
    auto [rows_submat_S, cols_submat_S] = dims_before_op(n, d, opS);
    if (!S.buff) {
        // We'll make a shallow copy of the sketching operator, take responsibility for filling the memory
        // of that sketching operator, and then call RSKGE3 with that new object.
        T *buff = new T[rows_submat_S * cols_submat_S];
        fill_dense(S.dist, rows_submat_S, cols_submat_S, ro_s, co_s, buff, S.seed_state);
        DenseDist D{rows_submat_S, cols_submat_S, DenseDistName::BlackBox, S.dist.major_axis};
        DenseSkOp S_(D, S.seed_state, buff);
        rskge3(layout, opA, opS, m, d, n, alpha, A, lda, S_, 0, 0, beta, B, ldb);
        delete [] buff;
        return;
    }
    randblas_require( S.dist.n_rows >= rows_submat_S + ro_s );
    randblas_require( S.dist.n_cols >= cols_submat_S + co_s );
    auto [rows_A, cols_A] = dims_before_op(m, n, opA);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= m);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= d);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.dist.n_rows, S.dist.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    blas::gemm(layout, opA, opS, m, d, n, alpha, A, lda, S_ptr, lds, beta, B, ldb);
    return;
}

} // end namespace RandBLAS::dense

#endif
