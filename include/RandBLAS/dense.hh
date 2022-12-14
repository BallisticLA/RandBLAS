#pragma once

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_STATE_HH
#include <RandBLAS/state.hh>
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

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

namespace RandBLAS::dense {

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
    const RNGState state;
    const bool own_memory = true;
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to dense sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    T *buff = NULL;
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
        RNGState state_,
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
    RNGState state_,
    T *buff_,           
    bool filled_,       
    bool persistent_,   
    blas::Layout layout_ 
) : // variable definitions
    dist(dist_),
    state(state_),
    buff(buff_),
    filled(filled_),
    persistent(persistent_),
    layout(layout_),
    own_memory(!buff_)
{   // Initialization logic
    //
    //      own_memory is a bool that's true iff buff_ is NULL.
    //
    if (this->own_memory) {
        assert(!this->filled);
        // We own the rights to the memory, and the memory
        // hasn't been allocated, so there's no way that the memory exists yet.
    } else {
        assert(this->persistent);
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

template <typename T>
RNGState fill_buff(
    T *buff,
    DenseDist D,
    RNGState state
);

template <typename T>
RNGState fill_buff(
    T *buff,
    DenseDist D,
    uint32_t key,
    uint32_t ctr_offset
);

// Compute B = alpha * op(S) * op(A) + beta * B
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
);

} // end namespace RandBLAS::dense

#endif  // define RandBLAS_DO_HH
