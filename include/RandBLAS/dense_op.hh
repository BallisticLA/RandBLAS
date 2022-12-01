#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

// Below comes from .cc

#include <iostream>
#include <stdio.h>
#include <omp.h>

#include <math.h>
#include <typeinfo>

namespace RandBLAS::dense_op {

enum class DistName : char {Gaussian = 'G', Normal = 'G', Uniform = 'U', Rademacher = 'R', Haar = 'H'};

struct Dist {
    DistName family = DistName::Gaussian;
    int64_t n_rows;
    int64_t n_cols;
    bool scale = false;
    // Guarantee for iid-dense distributions:
    //      (*) Swapping n_rows and n_cols can only affect
    //          random number generation up to scaling.
    //      (*) When a buffer is needed, it will be
    //          populated in a way that is agnoistic
    //          to row-major or column-major interpretation.  
};

template <typename T>
struct SketchingOperator {
    // Unlike a plain buffer that we might use in BLAS,
    // SketchingOperators in the RandBLAS::dense_op namespace
    // carry metadata to unambiguously define their dimensions
    // and the values of their entries.
    // 
    // Dimensions are specified with the distribution, in "dist".
    //
    Dist dist{};
    int64_t ctr_offset = 0;
    int64_t key = 0;
    T *op_data = NULL;
    bool populated = false;
    bool persistent = false;
    blas::Layout layout = blas::Layout::ColMajor;
    //      ^ Technically, users are allowed to change layout
    //      at will. For example, they might want to get
    //      rid of a transpose in a sketching operation by
    //      asserting a different layout than was
    //      originally used by the SketchingBuffer. However,
    //      users do this at their own peril, since changing
    //      layout usually requires swapping the numbers of
    //      rows and columns, and *that* requires changing
    //      the "dist" field of this struct.
};

template <typename T>
void gen_rmat_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
);

template <typename T>
void gen_rmat_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
);

template <typename T>
void populate_dense_buff(
    Dist D,
    uint32_t key,
    uint32_t ctr_offset,
    T *buff
);

template <typename T>
void lskge3(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SketchingOperator<T> &S0,
    int64_t pos, // pointer offset for S in S0
    T *A_ptr,
    int64_t lda,
    T beta,
    T *B_ptr,
    int64_t ldb
);

} // end namespace RandBLAS::dense_op

#endif  // define RandBLAS_UTIL_HH
