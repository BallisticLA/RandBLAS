#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sasos {

struct Dist {
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
    const bool scale = false;
};


template <typename T>
struct SASO {
    const Dist dist{};
    uint64_t key = 0;
    uint64_t ctr_offset = 0;
    int64_t *rows = NULL;
    int64_t *cols = NULL;
    T *vals = NULL;
};

template <typename T>
void fill_saso(SASO<T> &sas);

template <typename T>
void sketch_cscrow(SASO<T> &sas, int64_t n, T *a, T *a_hat, int threads);

template <typename T>
void sketch_csccol(SASO<T> &sas, int64_t m, T *a, T *a_hat, int threads);

template <typename T>
void print_saso(SASO<T> &sas);

} // end namespace RandBLAS::sasos

#endif // define RandBLAS_SASOS_HH
