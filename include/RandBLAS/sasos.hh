#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sasos {

template <typename T>
struct SASO {
    int64_t n_rows;
    int64_t n_cols;
    int64_t vec_nnz;
    uint64_t key = 0;
    uint64_t ctr = 0;
    int64_t *rows = NULL;
    int64_t *cols = NULL;
    T *vals = NULL;
};

template <typename T>
void fill_colwise(SASO<T> &sas);

template <typename T>
void sketch_cscrow(SASO<T> &sas, int64_t n, T *a, T *a_hat, int threads);

template <typename T>
void sketch_csccol(SASO<T> &sas, int64_t m, T *a, T *a_hat, int threads);

template <typename T>
void print_saso(SASO<T> &sas);

} // end namespace RandBLAS::sasos

#endif // define RandBLAS_SASOS_HH
