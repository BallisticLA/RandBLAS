#ifndef randblas_dense_hh
#define randblas_dense_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"

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

enum class DenseDistName : char {
    Gaussian = 'G',         ///< a Gaussian with mean 0 and standard deviation 1
    Uniform = 'U',          ///< uniform over the interval [-1, 1].
    Rademacher = 'R',       ///< uniform over {+1, -1}.
    Haar = 'H',             ///< uniform over row-orthonormal or column-orthonormal matrices.
    DisjointIntervals = 'I' ///< might require additional metadata.
};

struct DenseDist {
    const DenseDistName family = DenseDistName::Gaussian;
    const int64_t n_rows;
    const int64_t n_cols;
};

template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    using generator = RNG;
    using state_type = RNGState<RNG>;

    const DenseDist dist;            ///< the name of the distribution and matrix size
    const RNGState<RNG> seed_state;  ///< the initial CBRNG state
    RNGState<RNG> next_state;        ///< the current CBRNG state
    const bool own_memory = true;    ///< a flag that indicates who owns the memory

    T *buff = nullptr;               ///< memory
    bool filled = false;             ///< a flag that indicates if the memory was initialized
    bool persistent = true;          ///< ???

    const blas::Layout layout = blas::Layout::ColMajor; ///< matrix storage order


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
        uint32_t ctr_offset,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(dist, {{{ctr_offset}}, {{key}}}, buff, filled, persistent, layout) {};

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        uint32_t ctr_offset,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(DenseDist{family, n_rows, n_cols}, ctr_offset,
                  key, buff, filled, persistent, layout) {};

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






/** Fill a n by m matrix with random values. If RandBLAS is compiled with
 * OpenMP threading support enabled, the operation is parallelized using
 * OMP_NUM_THREADS. The sequence of values genrated is not dependent on the
 * number of OpenMP threads.
 *
 * @tparam T the data type of the matrix
 * @tparam RNG a random123 CBRNG type
 * @tparm OP an operator that transforms raw random values into matrix
 *           elements. See RandBLAS::base::uneg11 and RandBLAS::base::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] mat a pointer to a contiguous region of memory with space for
 *                n_rows*n_cols elements of type T. This memory will be filled
 *                with random values.
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
    auto [c, k] = seed;

    int64_t dim = n_rows * n_cols;
    int64_t nit = dim / RNG::ctr_type::static_size;
    int64_t nlast = dim % RNG::ctr_type::static_size;

#if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel firstprivate(c, k)
    {
        // add the start index to the counter in order to make the sequence
        // deterministic independent of the number of threads.
        int ti = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int64_t chs = nit / nt;
        int64_t nlg = nit % nt;
        int64_t i0 = chs * ti + (ti < nlg ? ti : nlg);
        int64_t i1 = i0 + chs + (ti < nlg ? 1 : 0);

        auto cc = c; // because of pointers used internal to RNG::ctr_type

        cc.incr(i0);
#else
        int64_t i0 = 0;
        int64_t i1 = nit;
#endif
        for (int64_t i = i0; i < i1; ++i)
        {
            auto rv = OP::generate(rng, cc, k);

            for (int j = 0; j < RNG::ctr_type::static_size; ++j)
            {
               mat[RNG::ctr_type::static_size * i + j] = rv[j];
            }

            cc.incr();
        }
#if defined(RandBLAS_HAS_OpenMP)
    }
    // puts the counter in the correct state when threads are used.
    c.incr(nit);
#endif

    if (nlast)
    {
        auto rv = OP::generate(rng, c, k);

        for (int64_t j = 0; j < nlast; ++j)
        {
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
            return fill_rmat<T,RNG,boxmul>(D.n_rows, D.n_cols, buff, state);

        case DenseDistName::Uniform:
            return fill_rmat<T,RNG,uneg11>(D.n_rows, D.n_cols, buff, state);

        case DenseDistName::Rademacher:
            throw std::runtime_error(std::string("Not implemented."));

        case DenseDistName::Haar:
            // This won't be filled IID, but a Householder representation
            // of a column-orthonormal matrix Q can be stored in the lower
            // triangle of Q (with "tau" on the diagonal). So the size of
            // buff will still be D.n_rows*D.n_cols.
            throw std::runtime_error(std::string("Not implemented."));

        default:
            throw std::runtime_error(std::string("Unrecognized distribution."));
    }

    return state;
}

template <typename T, typename RNG>
T* fill_skop_buff(
    DenseSkOp<T,RNG> &S0
) {
    T *S0_ptr = S0.buff;
    if (S0_ptr == nullptr) {
        S0_ptr = new T[S0.dist.n_rows * S0.dist.n_cols];
        S0.next_state = fill_buff<T,RNG>(S0_ptr, S0.dist, S0.seed_state);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
        return S0_ptr;
    } else if (!S0.filled) {
        S0.next_state = fill_buff<T>(S0_ptr, S0.dist, S0.seed_state);
        S0.filled = true;
        return S0_ptr;
    } else {
        return S0_ptr;
    }
}

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
    randblas_require(d <= m);
    randblas_require(S0.layout == layout);

    T *S0_ptr = fill_skop_buff<T>(S0);

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
