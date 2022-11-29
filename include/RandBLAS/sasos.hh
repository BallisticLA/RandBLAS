#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sasos {

struct SASO {
    int64_t n_rows;
    int64_t n_cols;
    int64_t vec_nnz;
    uint64_t key = 0;
    uint64_t ctr = 0;
    int64_t *rows = NULL;
    int64_t *cols = NULL;
    double *vals = NULL;
};

void fill_colwise(SASO sas);

void sketch_cscrow(SASO sas, int64_t n, double *a, double *a_hat, int threads);

void sketch_csccol(SASO sas, int64_t m, double *a, double *a_hat, int threads);

void print_saso(SASO sas);

} // end namespace RandBLAS::sasos

#endif // define RandBLAS_SASOS_HH
