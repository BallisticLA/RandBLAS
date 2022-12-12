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
static uint32_t gen_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
) {
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;
    key_type typed_key = {{key}};
    T_gen gen;
    int64_t dim = n_rows * n_cols;
    int64_t i;
    ctr_type rin {ctr_offset,0,0,0};
    ctr_type rout;
    for (i = 0; i + 3 < dim; i += 4) {
        // mathematically, rin = (int128) ctr_offset + (int128) i.
        rout = gen(rin, typed_key);
        mat[i] = r123::uneg11<T>(rout.v[0]);
        mat[i + 1] = r123::uneg11<T>(rout.v[1]);
        mat[i + 2] = r123::uneg11<T>(rout.v[2]);
        mat[i + 3] = r123::uneg11<T>(rout.v[3]);
        rin = rin.incr(4);
    }
    rout = gen(rin, typed_key);
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  r123::uneg11<T>(rout.v[j]);
        ++i;
        ++j;
    }
    return rin.v[0];
}

template <typename T>
static uint32_t gen_rmat_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
) {
    typedef r123::Philox4x32 CBRNG;
    // ^ the CBRNG generates 4 random numbers at a time, represents state with 4 32-bit words.
    if (typeid(T) == typeid(float)) {
        uint32_t next_ctr_offset = gen_unif<float, CBRNG>(
            n_rows, n_cols, (float *) mat, key, ctr_offset
        );
        // ^ cast on "mat" is needed to avoid compiler error
        return next_ctr_offset;
    } else if (typeid(T) == typeid(double)) {
        uint32_t next_ctr_offset = gen_unif<double, CBRNG>(
            n_rows, n_cols, (double *) mat, key, ctr_offset
        );
        // ^ cast on "mat" is needed to avoid compiler error
        return next_ctr_offset;
    } else {
        printf("\nType error. Only float and double are currently supported.\n");
        return 0;
    }
}

// Actual work - normal distribution
template <typename T, typename T_gen, typename T_fun>
static uint32_t gen_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
) {
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;
    key_type typed_key = {{key}};
    T_gen gen;
    int64_t dim = n_rows * n_cols;
    int64_t i;
    ctr_type rin {ctr_offset,0,0,0};
    ctr_type rout;
    T_fun pair_1, pair_2;
    for (i = 0; i + 3 < dim; i += 4) {
        // mathematically: rin = (int128) ctr_offset + (int128) i
        rout = gen(rin, typed_key);
        pair_1 = r123::boxmuller(rout.v[0], rout.v[1]);
        pair_2 = r123::boxmuller(rout.v[2], rout.v[3]);
        mat[i] = pair_1.x;
        mat[i + 1] = pair_1.y;
        mat[i + 2] = pair_2.x;
        mat[i + 3] = pair_2.y;
        rin.incr(4);
    }
    rout = gen(rin, typed_key);
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
    return rin.v[0];
}

template <typename T>
static uint32_t gen_rmat_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    uint32_t key,
    uint32_t ctr_offset
) {
    typedef r123::Philox4x32 CBRNG;
    // ^ the CBRNG generates 4 random numbers at a time, represents state with 4 32-bit words.
    if (typeid(T) == typeid(float)) {
        uint32_t next_ctr_offset = gen_norm<float, CBRNG, r123::float2>(
            n_rows, n_cols, (float *) mat, key, ctr_offset
        );
        // ^ cast on "mat" is needed to avoid compiler error
        return next_ctr_offset;
    } else if (typeid(T) == typeid(double)) {
        uint32_t next_ctr_offset = gen_norm<double, CBRNG, r123::float2>(
            n_rows, n_cols, (double *) mat, key, ctr_offset
        );
        // ^ cast on "mat" is needed to avoid compiler error
        return next_ctr_offset;
    } else {
        printf("\nType error. Only float and double are currently supported.\n");
        return 0;
    }
}

template <typename T>
uint32_t fill_buff(
    T *buff,
    DenseDist D,
    uint32_t key,
    uint32_t ctr_offset
) {
    uint32_t next_ctr_offset;
    switch (D.family) {
    case DenseDistName::Gaussian:
        next_ctr_offset = gen_rmat_norm<T>(D.n_rows, D.n_cols, buff, key, ctr_offset);
        break;
    case DenseDistName::Uniform:
        next_ctr_offset = gen_rmat_unif<T>(D.n_rows, D.n_cols, buff, key, ctr_offset);
        break;
    case DenseDistName::Rademacher:
        throw std::runtime_error(std::string("Not implemented."));
        break;
    case DenseDistName::Haar:
        // This won't be filled IID, but a Householder representation
        // of a column-orthonormal matrix Q can be stored in the lower
        // triangle of Q (with "tau" on the diagonal). So the size of
        // buff will still be D.n_rows*D.n_cols.
        throw std::runtime_error(std::string("Not implemented."));
        break;
    default:
        throw std::runtime_error(std::string("Unrecognized distribution."));
        break;
    }
    return next_ctr_offset;
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
        S0.next_ctr_offset = fill_buff<T>(S0_ptr, S0.dist, S0.key, S0.ctr_offset);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
    } else if (!S0.filled) {
        S0.next_ctr_offset = fill_buff<T>(S0_ptr, S0.dist, S0.key, S0.ctr_offset);
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

template uint32_t gen_rmat_unif<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template uint32_t gen_rmat_unif<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

template uint32_t gen_rmat_norm<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template uint32_t gen_rmat_norm<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

} // end namespace RandBLAS::dense_op
