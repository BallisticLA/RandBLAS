#pragma once

#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_EXCEPTIONS_HH
#include <RandBLAS/exceptions.hh>
#endif

#ifndef RandBLAS_STATE_HH
#include <RandBLAS/base.hh>
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
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
};

template <typename T>
struct SparseSkOp {
    const SparseDist dist;
    const base::RNGState seed_state; // maybe "self_seed"
    base::RNGState next_state; // maybe "next_seed"
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
        base::RNGState state_,
        int64_t *rows_ = NULL,
        int64_t *cols_ = NULL,
        T *vals_ = NULL 
    );

    //  Convenience constructor (a wrapper)
    SparseSkOp(
        SparseDist dist,
        uint32_t key,
        uint32_t ctr_offset,
        int64_t *rows,
        int64_t *cols,
        T *vals 
    ) : SparseSkOp(dist, base::RNGState{ctr_offset, key}, rows, cols, vals) {};
    
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
    ) : SparseSkOp(SparseDist{family, n_rows, n_cols, vec_nnz},
        key, ctr_offset, rows, cols, vals) {};

    //  Destructor
    ~SparseSkOp();
};

// Implementation of elementary constructor
template <typename T>
SparseSkOp<T>::SparseSkOp(
    SparseDist dist_,
    base::RNGState state_,
    int64_t *rows_,
    int64_t *cols_,
    T *vals_
) : // variable definitions
    dist(dist_),
    seed_state(state_),
    own_memory(!rows_ && !cols_ && !vals_)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    // Initialization logic
    //
    //      own_memory is a bool that's true iff the
    //      rows_, cols_, and vals_ pointers were all NULL.
    //
    int64_t long_ax_len = MAX(this->dist.n_rows, this->dist.n_cols);
    if (this->own_memory) {
        int64_t nnz = this->dist.vec_nnz * long_ax_len;
        this->rows = new int64_t[nnz];
        this->cols = new int64_t[nnz];
        this->vals = new T[nnz];
    } else {
        randblas_require(rows_ && cols_ && vals_);
        //  If any of rows_, cols_, and vals_ are not NULL,
        //  then none of them are NULL.
        this->rows = rows_;
        this->cols = cols_;
        this->vals = vals_;
    }
};

template <typename T>
SparseSkOp<T>::~SparseSkOp() {
    if (this->own_memory) {
        delete [] this->rows;
        delete [] this->cols;
        delete [] this->vals;
    }
};


template <typename T>
base::RNGState fill_sparse(
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
    const T *A,
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
