#pragma once

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_STATE_HH
#include <RandBLAS/state.hh>
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sparse {

enum class SparseDistName : char {
    SASO = 'S',      // short-axis-sparse operator
    LASO = 'L'       // long-axis-sparse operator
};

struct SparseDist {
    const SparseDistName family = SparseDistName::SASO;
    //const RandBLAS::dense::Dist dist4nz = RandBLAS::dense::DistName::Rademacher;
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
};

template <typename T>
struct SparseSkOp {
    const SparseDist dist;
    const RNGState state;
    const bool own_memory = true;
    
    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    int64_t *rows = NULL;
    int64_t *cols = NULL;
    T *vals = NULL;

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    SparseSkOp(
        SparseDist dist_,
        RNGState state_,
        int64_t *rows_ = NULL,
        int64_t *cols_ = NULL,
        T *vals_ = NULL 
    );

    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows = NULL,
        int64_t *cols = NULL,
        T *vals = NULL 
    );
    
    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows = NULL,
        int64_t *cols = NULL,
        T *vals = NULL 
    );

    //  Destructor
    ~SparseSkOp();
};

template <typename T>
RNGState fill_saso(
    SparseSkOp<T> &sas
);

// Compute B = alpha * op(S) * op(A) + beta * B
template <typename T>
void lskges(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SparseSkOp<T> &S0,
    int64_t i_os,
    int64_t j_os,
    T *A, // TODO: make const
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb,
    int threads = 4
);

template <typename T>
void print_saso(SparseSkOp<T> &sas);

} // end namespace RandBLAS::sparse_ops

#endif // define RandBLAS_SASOS_HH
