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

#include <math.h>
#include <typeinfo>


/*
Paradigm for APIs involving structs:
    Free-functions when there's no memory to manage
    Member functions when there IS memory to manage, or in initializing.
        We want to make this library hard to misuse in C++.
    We provide APIs that we require people use to ensure that structs are
        in a valid state. If you want to initialize the struct yourself
        we won't stop you, but we also take no responsibility for the 
        inevitable segfaults.

TODO: have a discussion around using smart pointers for memory safety.
    Burlen thinks we should seriously consider using smart pointers.
*/

/*
Currently have non-deterministic behavior (tests pass sometimes, fail sometimes).
I suspect there's some memory management mistake leading to undefined behavior.

      Start 18: TestDenseMoments.Gaussian
    18/34 Test #18: TestDenseMoments.Gaussian ....................***Failed    0.11 sec
    Running main() from /tmp/googletest-20220910-45435-1kz3pjx/googletest-release-1.12.1/googletest/src/gtest_main.cc
    Note: Google Test filter = TestDenseMoments.Gaussian
    [==========] Running 1 test from 1 test suite.
    [----------] Global test environment set-up.
    [----------] 1 test from TestDenseMoments
    [ RUN      ] TestDenseMoments.Gaussian
    /Users/riley/BALLISTIC_RNLA/randla/RandBLAS/test/src/test_dense.cc:48: Failure
    The difference between mean and 0.0 is 0.01195285380042985, which exceeds 1e-2, where
    mean evaluates to -0.01195285380042985,
    0.0 evaluates to 0, and
    1e-2 evaluates to 0.01.
    [  FAILED  ] TestDenseMoments.Gaussian (112 ms)
    [----------] 1 test from TestDenseMoments (112 ms total)
*/

namespace RandBLAS::dense {

using namespace RandBLAS::base;

// =============================================================================
/// We call a sketching operator "dense" if it takes Level 3 BLAS work to 
/// apply to a dense matrix. All such sketching operators supported by
/// RandBLAS currently have i.i.d. entries. This enumeration specifies
/// the distribution of the entries of such a sketching operator.
enum class DenseDistName : char {

    // ---------------------------------------------------------------------------
    ///  Gaussian distribution with mean 0 and standard deviation 1
    Gaussian = 'G',
    // ---------------------------------------------------------------------------
    ///  uniform distribution over [-1, 1].
    Uniform = 'U'
    
    //// ---------------------------------------------------------------------------
    ///  uniform distribution over \math{\{-1, 1\}}.
    // Rademacher = 'R',
    //// ---------------------------------------------------------------------------
    ///  A flag that a sketching operator's distribution should be 
    ///  uniform over row-orthonormal or column-orthonormal matrices.
    // Haar = 'H',
    
    //DisjointIntervals = 'I' // might require additional metadata.
};

// =============================================================================
/// A distribution over dense sketching operators.
///
struct DenseDist {
    // ---------------------------------------------------------------------------
    ///  The distribution used for the entries of the sketching operator.
    const DenseDistName family = DenseDistName::Gaussian;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many rows.
    const int64_t n_rows;

    // ---------------------------------------------------------------------------
    ///  Matrices drawn from this distribution have this many columns.
    const int64_t n_cols;
};

// =============================================================================
/// A sample from a prescribed distribution over dense sketching operators.
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
    const base::RNGState<RNG> seed_state;

    // ---------------------------------------------------------------------------
    ///  The state that should be used by the next call to an RNG *after* the
    ///  full sketching operator has been sampled.
    base::RNGState<RNG> next_state;

    // ---------------------------------------------------------------------------
    /// We need workspace to store a representation of the sampled sketching
    /// operator. This member indicates who is responsible for allocating and 
    /// deallocating this workspace. If own_memory is true, then 
    /// RandBLAS is responsible.
    const bool own_memory = true;

    T *buff = nullptr;       // memory
    bool filled = false;     // a flag that indicates if the memory was initialized
    bool persistent = true;  // explanation ...

    const blas::Layout layout = blas::Layout::ColMajor; ///< matrix storage order

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    DenseSkOp(
        DenseDist dist_,
        RNGState<RNG> const& state_,
        T *buff_,
        bool filled_,
        bool persistent_,
        blas::Layout layout_
    );

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDist dist,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(dist, RNGState<RNG>(key), buff, filled, persistent, layout) {};

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(DenseDist{family, n_rows, n_cols}, RNGState<RNG>(key),
                  buff, filled, persistent, layout) {};

    // Destructor
    ~DenseSkOp();
};

template <typename T, typename RNG>
DenseSkOp<T,RNG>::DenseSkOp(
    DenseDist dist_,
    RNGState<RNG> const& state_,
    T *buff_,           
    bool filled_,       
    bool persistent_,   
    blas::Layout layout_ 
) : // variable definitions
    dist(dist_),
    seed_state(state_),
    next_state{},
    own_memory(!buff_),
    buff(buff_),
    filled(filled_),
    persistent(persistent_),
    layout(layout_)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    // Initialization logic
    //
    //      own_memory is a bool that's true iff buff_ is nullptr.
    //
    if (this->own_memory) {
        randblas_require(!this->filled);
        // We own the rights to the memory, and the memory
        // hasn't been allocated, so there's no way that the memory exists yet.
    } else {
        randblas_require(this->persistent);
        // If the user gives us any memory to work with, then we cannot take
        // responsibility for deallocating on exit from LSKGE3 / RSKGE3.
    }
}

template <typename T, typename RNG>
DenseSkOp<T,RNG>::~DenseSkOp() {
    if (this->own_memory) {
        delete [] this->buff;
    }
}






/** Fill a n_rows \times n_cols matrix with random values. If RandBLAS is
 * compiled with OpenMP threading support enabled, the operation is
 * parallelized using OMP_NUM_THREADS. The sequence of values genrated is not
 * dependent on the number of OpenMP threads.
 *
 * @tparam T the data type of the matrix
 * @tparam RNG a random123 CBRNG type
 * @tparm OP an operator that transforms raw random values into matrix
 *           elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] mat a pointer to a contiguous region of memory with space for
 *                n_rows \times n_cols elements of type T. This memory will be
 *                filled with random values.
 * @param[in] seed A CBRNG state
 *
 * @returns the updated CBRNG state
 */
template <typename T, typename RNG, typename OP>
auto fill_rmat(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState<RNG> & seed
) {
    RNG rng;
    // clang chokes on this w/ omp due to internal use of lambdas, fixed in C++20
    //auto [c, k] = seed;
    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type k = seed.key;

    int64_t dim = n_rows * n_cols;
    int64_t nit = dim / RNG::ctr_type::static_size;
    int64_t nlast = dim % RNG::ctr_type::static_size;
}

#if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel firstprivate(c, k)
    {
        // decompose the work into a set of approximately equal size chunks.
        // if the number of iterations is not evenly divisible by the number
        // of threads, the left over itertions are distributed one each among
        // the first threads.
        int ti = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int64_t chs = nit / nt; // chunk size
        int64_t nlg = nit % nt; // number of large chunks
        int64_t i0 = chs * ti + (ti < nlg ? ti : nlg); // this threads start
        int64_t i1 = i0 + chs + (ti < nlg ? 1 : 0);    // this threads end

        // add the start index to the counter in order to make the sequence
        // deterministic independent of the number of threads.
        auto cc = c;
        cc.incr(i0);
#else
        int64_t i0 = 0;
        int64_t i1 = nit;
        auto &cc = c;
#endif
        for (int64_t i = i0; i < i1; ++i) {

            auto rv = OP::generate(rng, cc, k);

            for (int j = 0; j < RNG::ctr_type::static_size; ++j) {
               mat[RNG::ctr_type::static_size * i + j] = rv[j];
            }

            cc.incr();
        }
#if defined(RandBLAS_HAS_OpenMP)
    }
    // puts the counter in the correct state when threads are used.
    c.incr(nit);
#endif

    if (nlast) {
        auto rv = OP::generate(rng, c, k);

        for (int64_t j = 0; j < nlast; ++j) {
            mat[RNG::ctr_type::static_size * nit + j] = rv[j];
        }

        c.incr();
    }

    return RNGState<RNG> {c, k};
}


template <typename T, typename RNG>
auto fill_buff(
    T *buff,
    const DenseDist &D,
    RNGState<RNG> const& state
) {
    switch (D.family) {
        case DenseDistName::Gaussian:
            return fill_rmat<T,RNG,r123ext::boxmul>(D.n_rows, D.n_cols, buff, state);
        case DenseDistName::Uniform:
            return fill_rmat<T,RNG,r123ext::uneg11>(D.n_rows, D.n_cols, buff, state);
        //case DenseDistName::Rademacher:
        //    throw std::runtime_error(std::string("Not implemented."));
        //case DenseDistName::Haar:
            // This won't be filled IID, but a Householder representation
            // of a column-orthonormal matrix Q can be stored in the lower
            // triangle of Q (with "tau" on the diagonal). So the size of
            // buff will still be D.n_rows*D.n_cols.
        //    throw std::runtime_error(std::string("Not implemented."));
        default:
            throw std::runtime_error(std::string("Unrecognized distribution."));
    }

    return state;
}

template <typename SKOP>
auto fill_skop_buff(
    SKOP &S0
) {
    auto S0_ptr = S0.buff;
    if (S0_ptr == nullptr) {
        S0_ptr = new typename SKOP::buffer_type [S0.dist.n_rows * S0.dist.n_cols];
        S0.next_state = fill_buff(S0_ptr, S0.dist, S0.seed_state);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
        return S0_ptr;
    } else if (!S0.filled) {
        S0.next_state = fill_buff(S0_ptr, S0.dist, S0.seed_state);
        S0.filled = true;
        return S0_ptr;
    } else {
        return S0_ptr;
    }
}

// =============================================================================
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |transA| mathmacro:: \mathrm{transA}
///   .. |transS| mathmacro:: \mathrm{transS}
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
///     Their shapes are defined implicitly by :math:`(d, m, n, \transA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\transS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{i_os}, \texttt{j_os})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] transS
///      - If \math{\transS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\transS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
/// @param[in] transA
///      - If \math{\transA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\transA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
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
/// @param[in] i_os
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_os}, :]}.
///
/// @param[in] j_os
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_os}]}. 
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
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    DenseSkOp<T,RNG> &S0,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
){
    randblas_require(S0.layout == layout);

    auto S0_ptr = fill_skop_buff(S0);

    // Dimensions of A, rather than op(A)
    int64_t rows_A, cols_A, rows_S, cols_S;
    if (transA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
    } else {
        rows_A = n;
        cols_A = m;
    }
    // Dimensions of S, rather than op(S)
    if (transS == blas::Op::NoTrans) {
        rows_S = d;
        cols_S = m;
    } else {
        rows_S = m;
        cols_S = d;
    }

    // Sanity checks on dimensions and strides
    int64_t lds, pos;
    if (layout == blas::Layout::ColMajor) {
        lds = S0.dist.n_rows;
        pos = i_os + lds * j_os;
        randblas_require(lds >= rows_S);
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        lds = S0.dist.n_cols;
        pos = i_os * lds + j_os;
        randblas_require(lds >= cols_S);
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }
    // Perform the sketch.
    blas::gemm<T>(
        layout, transS, transA,
        d, n, m,
        alpha,
        &S0_ptr[pos], lds,
        A, lda,
        beta,
        B, ldb
    );
    return;
}

} // end namespace RandBLAS::dense

#endif
