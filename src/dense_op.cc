#include "dense_op.hh"

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


namespace RandBLAS::dense_op {

/*
Note from Random123: Simply incrementing the counter (or key) is effectively indistinguishable
from a sequence of samples of a uniformly distributed random variable.

Notes:
1. Currently, Not sure how to make generalized implementations to support both 2 and 4 random number generations.
2. Note on eventually needing to add 8 and 16-bit generations.
*/



// Actual work - uniform dirtibution
template <typename T, typename T_gen>
static void gen_unif(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
{
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;
    key_type typed_key = {{key}};
    T_gen gen;
    int64_t dim = n_rows * n_cols;
    uint32_t i;
    ctr_type r;
    for (i = 0; i + 3 < dim; i += 4) {
        r = gen({{ctr_offset + i,0,0,0}}, typed_key);
        mat[i] = r123::uneg11<T>(r.v[0]);
        mat[i + 1] = r123::uneg11<T>(r.v[1]);
        mat[i + 2] = r123::uneg11<T>(r.v[2]);
        mat[i + 3] = r123::uneg11<T>(r.v[3]);
    }
    r = gen({{ctr_offset + i,0,0,0}}, typed_key);
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  r123::uneg11<T>(r.v[j]);
        ++i;
        ++j;
    }
}

template <typename T>
static void gen_rmat_unif(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
{
    if (typeid(T) == typeid(float))
    {
        // 4 32-bit generated values
        typedef r123::Philox4x32 CBRNG;
        // Casting is done only so that the compiler does not throw an error
        gen_unif<float, CBRNG>(n_rows, n_cols, (float*) mat, key, ctr_offset);
    }
    else if (typeid(T) == typeid(double))
    {
        // 4 64-bit generated values
        typedef r123::Philox4x64 CBRNG;
        // Casting is done only so that the compiler does not throw an error
        gen_unif<double, CBRNG>(n_rows, n_cols, (double*) mat, key, ctr_offset);
    }
    else
    {
        printf("\nType error. Only float and double are currently supported.\n");
    }
}

// Actual work - normal distribution
template <typename T, typename T_gen, typename T_fun>
static void gen_norm(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
{
    typedef typename T_gen::key_type key_type;
    typedef typename T_gen::ctr_type ctr_type;
    key_type typed_key = {{key}};
    T_gen gen;
    int64_t dim = n_rows * n_cols;
    uint32_t i;
    ctr_type r;
    T_fun pair_1, pair_2;
    for (i = 0; i + 3 < dim; i += 4) {
        r = gen({{ctr_offset + i,0,0,0}}, typed_key);
        pair_1 = r123::boxmuller(r.v[0], r.v[1]);
        pair_2 = r123::boxmuller(r.v[2], r.v[3]);
        mat[i] = pair_1.x;
        mat[i + 1] = pair_1.y;
        mat[i + 2] = pair_2.x;
        mat[i + 3] = pair_2.y;
    }
    r = gen({{ctr_offset + i,0,0,0}}, typed_key);
    pair_1 = r123::boxmuller(r.v[0], r.v[1]);
    pair_2 = r123::boxmuller(r.v[2], r.v[3]);
    T *v = new T[4] {pair_1.x, pair_1.y, pair_2.x, pair_2.y};
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  v[j];
        ++i;
        ++j;
    }
    delete[] v;
}

template <typename T>
static void gen_rmat_norm(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
{
    if (typeid(T) == typeid(float))
    {
        // 4 32-bit generated values
        typedef r123::Philox4x32 CBRNG;
        // Casting is done only so that the compiler does not throw an error
        gen_norm<float, CBRNG, r123::float2>(n_rows, n_cols, (float *) mat, key, ctr_offset);

    }
    else if (typeid(T) == typeid(double))
    {
        // 4 32-bit generated values
        typedef r123::Philox4x64 CBRNG;
        // Casting is done only so that the compiler does not throw an error
        gen_norm<double, CBRNG, r123::double2>(n_rows, n_cols, (double *) mat, key, ctr_offset);
    }    
    else
    {
        printf("\nType error. Only float and double are currently supported.\n");
    }
}

template <typename T>
void fill_buff(T *buff, Dist D, uint32_t key, uint32_t ctr_offset) {
    switch (D.family) {
    case DistName::Gaussian:
        gen_rmat_norm<T>(D.n_rows, D.n_cols, buff, key, ctr_offset);
        break;
    case DistName::Uniform:
        gen_rmat_unif<T>(D.n_rows, D.n_cols, buff, key, ctr_offset);
        break;
    case DistName::Rademacher:
        throw std::runtime_error(std::string("Not implemented."));
        break;
    case DistName::Haar:
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
    return;
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
    SketchingOperator<T> &S0,
    int64_t pos, // pointer offset for S in S0
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
        fill_buff<T>(S0_ptr, S0.dist, S0.key, S0.ctr_offset);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
    } else if (!S0.filled) {
        fill_buff<T>(S0_ptr, S0.dist, S0.key, S0.ctr_offset);
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
    int64_t lds;
    if (layout == blas::Layout::ColMajor) {
        lds = S0.dist.n_rows;
        assert(lds >= rows_S);
        assert(lda >= rows_A);
        assert(ldb >= d);
    } else {
        lds = S0.dist.n_cols;
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
    SketchingOperator<double> &S0, int64_t pos, const double *A, int64_t lda, double beta, double *B, int64_t ldb);
template void lskge3(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    SketchingOperator<float> &S0, int64_t pos, const float *A, int64_t lda, float beta, float *B, int64_t ldb);

template void gen_rmat_unif<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template void gen_rmat_unif<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

template void gen_rmat_norm<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template void gen_rmat_norm<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

} // end namespace RandBLAS::dense_op
