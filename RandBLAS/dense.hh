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
    Gaussian = 'G',         
    Uniform = 'U',          // uniform over the interval [-1, 1].
    Rademacher = 'R',       // uniform over {+1, -1}.
    Haar = 'H',             // uniform over row-orthonormal or column-orthonormal matrices.
    DisjointIntervals = 'I' // might require additional metadata.
};

struct DenseDist {
    const DenseDistName family = DenseDistName::Gaussian;
    const int64_t n_rows;
    const int64_t n_cols;
};

template <typename T>
struct DenseSkOp {
    const DenseDist dist;
    const RNGState seed_state;
    RNGState next_state;
    const bool own_memory = true;
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to dense sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    T *buff = nullptr;
    bool filled = false;
    bool persistent = true;
    const blas::Layout layout = blas::Layout::ColMajor;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    DenseSkOp(
        DenseDist dist_,
        const RNGState &state_,
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
    ) : DenseSkOp(dist, RNGState{ctr_offset, key}, buff, filled, persistent, layout) {};

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        uint32_t key,
        uint32_t ctr_offset,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(DenseDist{family, n_rows, n_cols}, RNGState{key, ctr_offset},
        buff, filled, persistent, layout) {};

    // Destructor
    ~DenseSkOp();
};

template <typename T>
DenseSkOp<T>::DenseSkOp(
    DenseDist dist_,
    const RNGState &state_,
    T *buff_,           
    bool filled_,       
    bool persistent_,   
    blas::Layout layout_ 
) : // variable definitions
    dist(dist_),
    seed_state(state_),
    next_state(),
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

template <typename T>
DenseSkOp<T>::~DenseSkOp() {
    if (this->own_memory) {
        delete [] this->buff;
    }
}










template <typename T, typename T_gen>
static RNGState gen_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState &state
) {
    int64_t dim = n_rows * n_cols;
    int64_t i;
    Random123_RNGState<T_gen> impl_state(state);
    T_gen gen;
    typedef typename T_gen::ctr_type ctr_type;
    ctr_type rout;
    for (i = 0; i + 3 < dim; i += 4) {
        rout = gen(impl_state.ctr, impl_state.key);
        mat[i] = r123::uneg11<T>(rout.v[0]);
        mat[i + 1] = r123::uneg11<T>(rout.v[1]);
        mat[i + 2] = r123::uneg11<T>(rout.v[2]);
        mat[i + 3] = r123::uneg11<T>(rout.v[3]);
        impl_state.ctr.incr(4);
    }
    rout = gen(impl_state.ctr, impl_state.key);
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  r123::uneg11<T>(rout.v[j]);
        ++i;
        ++j;
    }
    return impl_state;
}



template <typename T>
RNGState gen_rmat_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState &state
) {
    switch (state.rng_name) {
        case RNGName::Philox:
            return gen_unif<T, Philox>(n_rows, n_cols, mat, state);
        case RNGName::Threefry:
            return gen_unif<T, Threefry>(n_rows, n_cols, mat, state);
        default:
            throw std::runtime_error(std::string("Unrecognized generator."));
    }
}

template <typename T, typename T_gen>
static RNGState gen_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState &state
) {
    T_gen gen;
    Random123_RNGState<T_gen> impl_state(state);
    int64_t dim = n_rows * n_cols;
    int64_t i;
    typedef typename T_gen::ctr_type ctr_type;
    ctr_type rout;
    r123::float2 pair_1, pair_2;
    for (i = 0; i + 3 < dim; i += 4) {
        rout = gen(impl_state.ctr, impl_state.key);
        pair_1 = r123::boxmuller(rout.v[0], rout.v[1]);
        pair_2 = r123::boxmuller(rout.v[2], rout.v[3]);
        mat[i] = pair_1.x;
        mat[i + 1] = pair_1.y;
        mat[i + 2] = pair_2.x;
        mat[i + 3] = pair_2.y;
        impl_state.ctr.incr(4);
    }
    rout = gen(impl_state.ctr, impl_state.key);
    pair_1 = r123::boxmuller(rout.v[0], rout.v[1]);
    pair_2 = r123::boxmuller(rout.v[2], rout.v[3]);
    T v[4] = {pair_1.x, pair_1.y, pair_2.x, pair_2.y};
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  v[j];
        ++i;
        ++j;
    }
    RNGState out_state(state);
    return out_state;
}

template <typename T>
static RNGState gen_rmat_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState &state
) {
    switch (state.rng_name) {
        case RNGName::Philox:
            return gen_norm<T, Philox>(n_rows, n_cols, mat, state);
        case RNGName::Threefry:
            return gen_norm<T, Threefry>(n_rows, n_cols, mat, state);
        default:
            throw std::runtime_error(std::string("Unrecognized generator."));
    }
}

template <typename T>
RNGState fill_buff(
    T *buff,
    DenseDist D,
    const RNGState &state
) {
    switch (D.family) { // no break statements needed as-written
        case DenseDistName::Gaussian:
            return gen_rmat_norm<T>(D.n_rows, D.n_cols, buff, state);
        case DenseDistName::Uniform:
            return gen_rmat_unif<T>(D.n_rows, D.n_cols, buff, state);
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
}

template <typename T>
T* fill_skop_buff(
    DenseSkOp<T> &S0
) {
    T *S0_ptr = S0.buff;
    if (S0_ptr == nullptr) {
        S0_ptr = new T[S0.dist.n_rows * S0.dist.n_cols];
        S0.next_state = fill_buff<T>(S0_ptr, S0.dist, S0.seed_state);
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

template <typename T>
void lskge3(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    DenseSkOp<T> &S0,
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
