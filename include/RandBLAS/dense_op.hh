#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

namespace RandBLAS::dense_op {

enum class DistName : char {
    Gaussian = 'G',
    Normal = 'G', // alias, for user convenience
    Uniform = 'U', // uniform over the interval [-1, 1].
    Rademacher = 'R', // uniform over {+1, -1}.
    Haar = 'H', // uniform over row-orthonormal or column-orthonormal matrices.
    DisjointIntervals = 'I' // might require additional metadata.
};

struct Dist {
    const DistName family = DistName::Gaussian;
    const int64_t n_rows;
    const int64_t n_cols;
    const bool scale = false;
    // Guarantee for iid-dense distributions:
    //      (*) Swapping n_rows and n_cols can only affect
    //          random number generation up to scaling.
    //      (*) When a buffer is needed, it will be
    //          filled in a way that is agnoistic
    //          to row-major or column-major interpretation.  
};

template <typename T>
struct SketchingOperator {
    // Unlike a plain buffer that we might use in BLAS,
    // SketchingOperators in the RandBLAS::dense_op namespace
    // carry metadata to unambiguously define their dimensions
    // and the values of their entries.
    // 
    // Dimensions are specified with the distribution, in "dist".
    //
    const Dist dist{};
    const int64_t ctr_offset = 0;
    const int64_t key = 0;
    T *buff = NULL;
    bool filled = false;
    bool persistent = true;
    const blas::Layout layout = blas::Layout::ColMajor;
};

template <typename T>
void fill_buff(
    T *buff,
    Dist D,
    uint32_t key,
    uint32_t ctr_offset
);

// Compute B = alpha * op(S) * op(A) + beta * B
template <typename T>
void lskge3(
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
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
);

} // end namespace RandBLAS::dense_op

#endif  // define RandBLAS_UTIL_HH
