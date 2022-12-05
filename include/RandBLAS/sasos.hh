#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sasos {

// Next step: modify sketch_cscrow and sketch_csccol so 
// they can be applied to submatrices. (This will require
// changing their function signatures a fair amount.)
//
// step after that: write LSKGES. Raise an error if 
// transA or transS == Trans.
//
// Step after that: prepare a big PR for Burlen to review.
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
void sketch_cscrow(
    SASO<T>& sas,
    int64_t n,
    T *a, // todo: make this const
    int64_t lda,
    T *a_hat,
    int64_t lda_hat,
    int threads
);

template <typename T>
void sketch_csccol(
    SASO<T>& sas,
    int64_t n,
    T *a, // todo: make this const
    int64_t lda,
    T *a_hat,
    int64_t lda_hat,
    int threads
);

template <typename T>
void print_saso(SASO<T> &sas);

} // end namespace RandBLAS::sasos

#endif // define RandBLAS_SASOS_HH
