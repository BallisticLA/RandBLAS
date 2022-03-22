#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

namespace RandBLAS::dense_op {


void gen_rmat_unif(
        int64_t n_rows,
        int64_t n_cols,
        float* mat,
        uint64_t seed);

void gen_rmat_norm(
        int64_t n_rows,
        int64_t n_cols,
        float* mat,
        uint64_t seed);

} // end namespace RandBLAS::dense_op

#endif  // define RandBLAS_UTIL_HH
