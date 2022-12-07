#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sparse_op {

enum class DistName : char {SASO = 'S'};

struct Dist {
    const DistName family = DistName::SASO;
    //const RandBLAS::dense_op::Dist dist4nz = RandBLAS::dense_op::DistName::Rademacher;
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
    const bool scale = false;
};

template <typename T>
struct SketchingOperator {
    const Dist dist{};
    const uint64_t key = 0;
    const uint64_t ctr_offset = 0;
    int64_t *rows = NULL;
    int64_t *cols = NULL;
    T *vals = NULL;
    // ANY_STRUCT metadata = NULL;
};

template <typename T>
void fill_saso(
    SketchingOperator<T> &sas
);

// Compute B = alpha * op(S) * op(A) + beta * B
template <typename T>
void lskges(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SketchingOperator<T> &S0,
    int64_t i_os,
    int64_t j_os,
    T *A, // TODO: make const
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb,
    int threads = 4
);

template <typename T>
void print_saso(SketchingOperator<T> &sas);

} // end namespace RandBLAS::sparse_ops

#endif // define RandBLAS_SASOS_HH
