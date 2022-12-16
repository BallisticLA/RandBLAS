#include <RandBLAS/exceptions.hh>
#include <RandBLAS/dense.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <omp.h>

#include <math.h>
#include <typeinfo>

#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>


namespace RandBLAS::dense {

template <typename T, typename T_gen>
static RNGState gen_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
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
    RNGState out_state(impl_state);
    return out_state;
}

template RNGState gen_unif<float, Philox>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_unif<double, Philox>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);
template RNGState gen_unif<float, Threefry>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_unif<double, Threefry>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

template <typename T>
RNGState gen_rmat_unif(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
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
    RNGState state
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
    T *v = new T[4] {pair_1.x, pair_1.y, pair_2.x, pair_2.y};
    int32_t j = 0;
    while (i < dim) {
        mat[i] =  v[j];
        ++i;
        ++j;
    }
    delete[] v;
    RNGState out_state(state);
    return out_state;
}

template  RNGState gen_norm<float, Philox>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template  RNGState gen_norm<double, Philox>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);
template  RNGState gen_norm<float, Threefry>(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template  RNGState gen_norm<double, Threefry>(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

template <typename T>
static RNGState gen_rmat_norm(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    RNGState state
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
T* fill_skop_buff(
    DenseSkOp<T> &S0
) {
    T *S0_ptr = S0.buff;
    if (S0_ptr == NULL) {
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

// Explicit instantiation of template functions
template void lskge3(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, double alpha,
    DenseSkOp<double> &S0, int64_t i_os, int64_t j_os, const double *A, int64_t lda, double beta, double *B, int64_t ldb);
template void lskge3(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d, int64_t n, int64_t m, float alpha,
    DenseSkOp<float> &S0, int64_t i_os, int64_t j_os, const float *A, int64_t lda, float beta, float *B, int64_t ldb);

template float* fill_skop_buff(DenseSkOp<float> &S0);
template double* fill_skop_buff(DenseSkOp<double> &S0);

template RNGState fill_buff(float *buff, DenseDist D, RNGState state);
template RNGState fill_buff(double *buff, DenseDist D, RNGState state);

template RNGState gen_rmat_unif(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_rmat_unif(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

template RNGState gen_rmat_norm(int64_t n_rows, int64_t n_cols, float* mat, RNGState state);
template RNGState gen_rmat_norm(int64_t n_rows, int64_t n_cols, double* mat, RNGState state);

} // end namespace RandBLAS::dense_op
