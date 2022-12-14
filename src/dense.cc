#include "dense.hh"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <omp.h>

#include <math.h>
#include <typeinfo>

#if !defined(R123_NO_SINCOS) && defined(__APPLE__)
/* MacOS X 10.10.5 (2015) doesn't have sincosf */
// use "-D __APPLE__" as a compiler flag to make sure this is hit.
#define R123_NO_SINCOS 1
#endif

#if R123_NO_SINCOS /* enable this if sincos and sincosf are not in the math library */
static inline void sincosf(float x, float *s, float *c) {
    *s = sinf(x);
    *c = cosf(x);
}

static inline void sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}
#endif /* sincos is not in the math library */


// The following two functions are part of NVIDIA device side math library.
// Random123 relies on them in both host and device sources.  We can work
// around this by defining them when not compiling device code.
#if !defined(__CUDACC__)

static inline void sincospif(float x, float *s, float *c) {
    const float PIf = 3.1415926535897932f;
    sincosf(PIf*x, s, c);
}

static inline void sincospi(double x, double *s, double *c) {
    const double PI = 3.1415926535897932;
    sincos(PI*x, s, c);
}
#endif
#include <Random123/philox.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>


namespace RandBLAS::dense {

// Actual work - uniform dirtibution
template <typename T, typename T_gen>
static RNGState gen_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
) {
    int64_t dim = n_rows * n_cols;
    int64_t i;
    r123::ReinterpretCtr<RNGState::r123_ctr, T_gen> gen;
    RNGState::r123_ctr rout;
    for (i = 0; i + 3 < dim; i += 4) {
        // mathematically, rin = (int128) ctr_offset + (int128) i.
        rout = gen(state._c, state._k);
        mat[i] = r123::uneg11<T>(rout.v[0]);
        mat[i + 1] = r123::uneg11<T>(rout.v[1]);
        mat[i + 2] = r123::uneg11<T>(rout.v[2]);
        mat[i + 3] = r123::uneg11<T>(rout.v[3]);
        state._c.incr(4);
    }
    rout = gen(state._c, state._k);
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  r123::uneg11<T>(rout.v[j]);
        ++i;
        ++j;
    }
    return state;
}

template <typename T>
static RNGState gen_rmat_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
) {
    typedef r123::Philox4x32 CBRNG;
    if (typeid(T) == typeid(float)) {
        RNGState s = gen_unif<float, CBRNG>(n_rows, n_cols, (float *) mat, state);
        return s;
    } else if (typeid(T) == typeid(double)) {
        RNGState s = gen_unif<double, CBRNG>(n_rows, n_cols, (double *) mat, state);
        return s;
    } else {
        throw std::runtime_error("\nType error. Only float and double are currently supported.\n");
    }
}

// Actual work - normal distribution
template <typename T, typename T_gen>
static RNGState gen_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
) {
    r123::ReinterpretCtr<RNGState::r123_ctr, T_gen> gen;
    int64_t dim = n_rows * n_cols;
    int64_t i;
    RNGState::r123_ctr rout;
    r123::float2 pair_1, pair_2;
    for (i = 0; i + 3 < dim; i += 4) {
        // mathematically: rin = (int128) ctr_offset + (int128) i
        rout = gen(state._c, state._k);
        pair_1 = r123::boxmuller(rout.v[0], rout.v[1]);
        pair_2 = r123::boxmuller(rout.v[2], rout.v[3]);
        mat[i] = pair_1.x;
        mat[i + 1] = pair_1.y;
        mat[i + 2] = pair_2.x;
        mat[i + 3] = pair_2.y;
        state._c.incr(4);
    }
    rout = gen(state._c, state._k);
    pair_1 = r123::boxmuller(rout.v[0], rout.v[1]);
    pair_2 = r123::boxmuller(rout.v[2], rout.v[3]);
    T *v = new T[4] {pair_1.x, pair_1.y, pair_2.x, pair_2.y};
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  v[j];
        ++i;
        ++j;
    }
    delete[] v;
    return state;
}

template <typename T>
static RNGState gen_rmat_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
) {
    typedef r123::Philox4x32 CBRNG;
    // ^ the CBRNG generates 4 random numbers at a time, represents state with 4 32-bit words.
    if (typeid(T) == typeid(float)) {
        RNGState s = gen_norm<float, CBRNG>(n_rows, n_cols, (float*) mat, state);
        return s;
    } else if (typeid(T) == typeid(double)) {
        RNGState s = gen_norm<double, CBRNG>(n_rows, n_cols, (double *) mat, state);
        return s;
    } else {
        throw std::runtime_error("\nType error. Only float and double are currently supported.\n");
    }
}

template <typename T>
RNGState fill_buff(
    T *buff,
    DenseDist D,
    RNGState state
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
    assert(d <= m); // Left-sketching can't increase the size of the output.
    if (S0.layout != layout) {
        throw std::runtime_error(std::string("Inconsistent layouts."));
    }
    // Would be nice to factor out the code for defining S0_ptr, but then
    // it's not clear what we'd return from that function. Can't just return
    // a pointer for dynamically allocated memory that isn't assigned to
    // something. 
    T *S0_ptr = S0.buff;
    if (S0_ptr == NULL) {
        S0_ptr = new T[S0.dist.n_rows * S0.dist.n_cols];
        fill_buff<T>(S0_ptr, S0.dist, S0.state);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
    } else if (!S0.filled) {
        fill_buff<T>(S0_ptr, S0.dist, S0.state);
        S0.filled = true;
    }
    // TODO: add a state check.

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
        assert(lds >= rows_S);
        assert(lda >= rows_A);
        assert(ldb >= d);
    } else {
        lds = S0.dist.n_cols;
        pos = i_os * lds + j_os;
        assert(lds >= cols_S);
        assert(lda >= cols_A);
        assert(ldb >= n);
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

// Explicit instantiation of template functions
template void lskge3(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, double alpha,
    DenseSkOp<double> &S0, int64_t i_os, int64_t j_os, const double *A, int64_t lda, double beta, double *B, int64_t ldb);
template void lskge3(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    DenseSkOp<float> &S0, int64_t i_os, int64_t j_os, const float *A, int64_t lda, float beta, float *B, int64_t ldb);

template RNGState gen_rmat_unif<float>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_rmat_unif<double>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

template RNGState gen_rmat_norm<float>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_rmat_norm<double>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

} // end namespace RandBLAS::dense_op
