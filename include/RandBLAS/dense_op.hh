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

#include <Random123/philox.h>
#include <Random123/boxmuller.hpp>
#include <Random123/uniform.hpp>

namespace RandBLAS::dense_op {

template <typename T>
void gen_rmat_unif(
        int64_t n_rows,
        int64_t n_cols,
        T* mat,
        uint32_t seed
        );

template <typename T>
void gen_rmat_norm(
        int64_t n_rows,
        int64_t n_cols,
        T* mat,
        uint32_t seed
        );

} // end namespace RandBLAS::dense_op

#endif  // define RandBLAS_UTIL_HH
