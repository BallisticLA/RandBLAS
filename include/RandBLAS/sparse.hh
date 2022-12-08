#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_SASOS_HH
#define RandBLAS_SASOS_HH

namespace RandBLAS::sparse {

enum class SparseDistName : char {SASO = 'S'};

struct SparseDist {
    const SparseDistName family = SparseDistName::SASO;
    //const RandBLAS::dense::Dist dist4nz = RandBLAS::dense::DistName::Rademacher;
    const int64_t n_rows;
    const int64_t n_cols;
    const int64_t vec_nnz;
    const bool scale = false;
};

template <typename T>
struct SparseSkOp {
    const SparseDist dist{};
    const uint64_t key = 0;
    const uint64_t ctr_offset = 0;
    
    /////////////////////////////////////////////////////////////////////
    //
    // Properties specific to sparse sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    int64_t *rows = NULL;
    int64_t *cols = NULL;
    T *vals = NULL;

    /////////////////////////////////////////////////////////////////////
    //
    // Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: requires a struct as input
    SparseSkOp(
        SparseDist dist_,
        uint64_t key_,
        uint64_t ctr_offset_
    ) : dist(dist_), key(key_), ctr_offset(ctr_offset_),
        rows(NULL), cols(NULL), vals(NULL) {};
    
    //  Unpacked constructor: uses elementary types as inputs
    SparseSkOp(
        SparseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz,
        bool scale,
        uint64_t key,
        uint64_t ctr_offset
    );
    
    //  Unpacked constructor with default values
    SparseSkOp(
        int64_t n_rows,
        int64_t n_cols,
        int64_t vec_nnz,
        uint64_t key,
        uint64_t ctr_offset
    ) : SparseSkOp(SparseDistName::SASO, n_rows, n_cols,
        vec_nnz, false, key, ctr_offset) {};
    
    //  Destructor
    ~SparseSkOp();
};

template <typename T>
SparseSkOp<T>::SparseSkOp(
    SparseDistName family,
    int64_t n_rows,
    int64_t n_cols,
    int64_t vec_nnz,
    bool scale,
    uint64_t key_,
    uint64_t ctr_offset_
) : dist{family, n_rows, n_cols, vec_nnz, scale}, key(key_), ctr_offset(ctr_offset_) {
    assert(n_rows <= n_cols);
    this->rows = new int64_t[vec_nnz * n_cols];
    this->cols = new int64_t[vec_nnz * n_cols];
    this->vals = new T[vec_nnz * n_cols];
}

template <typename T>
SparseSkOp<T>::~SparseSkOp() {
    delete [] this->rows;
    delete [] this->cols;
    delete [] this->vals;
};

template <typename T>
void fill_saso(
    SparseSkOp<T> &sas
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
    SparseSkOp<T> &S0,
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
void print_saso(SparseSkOp<T> &sas);

} // end namespace RandBLAS::sparse_ops

#endif // define RandBLAS_SASOS_HH
