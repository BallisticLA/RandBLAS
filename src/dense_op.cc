#include "dense_op.hh"

#include <iostream>
#include <stdio.h>
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
    // Definde the generator
    T_gen gen;
    int64_t dim = n_rows * n_cols;
    // Need every thread to have its own version of key for the outer loop to be parallelizable
    // Need to figure out when fork/join overhead becomes less than time saved by parallelization
    // Effectively, below structure is similar to unrolling by a factor of 4
    uint32_t i;
    ctr_type r;
    for (i = 0; i + 3 < dim; i += 4) {
        // Adding critical section around the increment should make outer loop parallelizable?
        // Below array represents counter updating
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
void gen_rmat_unif(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
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
    /*typename T_gen::key_type key = {{seed}};
    // Definde the generator
    T_gen gen;

    uint64_t dim = n_rows * n_cols;

    // Effectively, below structure is similar to unrolling by a factor of 4
    uint32_t i = 0;
    // Compensation code - we would effectively not use at least 1 and up to 3 generated random numbers 
    int comp = dim % 4;
    if (comp){

        // Below array represents counter updating
        typename T_gen::ctr_type r = gen({{1, 0, 0, 0}}, key);

        // Take 2 32 or 64-bit unsigned random vals, return 2 random floats/doubles
        // Since generated vals are indistinguishable form uniform, feed them into box-muller right away
        // Uses uneg11 & u01 under the hood
        T_fun pair_1 = r123::boxmuller(r.v[0], r.v[1]);
        T_fun pair_2 = r123::boxmuller(r.v[2], r.v[3]);

        T_fun pair_1 = r123::boxmuller(r.v[0], r.v[1]);
        T_fun pair_2 = r123::boxmuller(r.v[2], r.v[3]);

        // Only 3 cases here, so using nested ifs
        mat[i] = pair_1.x;
        ++i;
        if (i < comp)
        {
            mat[i] = pair_1.y;
            ++i;
            if (i < comp)
            {
                mat[i] = pair_2.x;
                ++i;
            }
        }
    }
    // Unrolling
    for (; i < dim; i += 4)
    {
        // Below array represents counter updating
        typename T_gen::ctr_type r = gen({{i, 0, 0, 0}}, key);
        // Paralleleize

        // Take 2 32 or 64-bit unsigned random vals, return 2 random floats/doubles
        // Since generated vals are indistinguishable form uniform, feed them into box-muller right away
        // Uses uneg11 & u01 under the hood
        T_fun pair_1 = r123::boxmuller(r.v[0], r.v[1]);
        T_fun pair_2 = r123::boxmuller(r.v[2], r.v[3]);

        mat[i] = pair_1.x;
        mat[i + 1] = pair_1.y;
        mat[i + 2] = pair_2.x;
        mat[i + 3] = pair_2.y;*/
}

template <typename T>
void gen_rmat_norm(int64_t n_rows, int64_t n_cols, T* mat, uint32_t key, uint32_t ctr_offset)
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

// template <typename T>
// populated_dense_buff(SketchingBuffer &S, T *buff) {
//     int64_t size = S.n_rows * S.n_cols;
//     switch (S.dist) {
//     case DenseDist::Gaussian:
//         gen_rmat_norm<T>(S.n_rows, S.n_cols, )
//         break;
//     case DenseDist::Uniform:
//         break;
//     case DenseDist::Rademacher:
//         throw std::exception("Not implemented.");
//         break;
//     default:
//         break;
//     }
// }

// Explicit instantiation of template functions - workaround to avoid header implementations
template void gen_rmat_unif<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template void gen_rmat_unif<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

template void gen_rmat_norm<float>(int64_t n_rows, int64_t n_cols, float* mat, uint32_t key, uint32_t ctr_offset);
template void gen_rmat_norm<double>(int64_t n_rows, int64_t n_cols, double* mat, uint32_t key, uint32_t ctr_offset);

} // end namespace RandBLAS::dense_op
