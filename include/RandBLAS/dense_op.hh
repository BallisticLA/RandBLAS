#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

namespace RandBLAS::dense_op {

enum class DistName : char {Gaussian = 'G', Normal = 'G', Uniform = 'U', Rademacher = 'R', Haar = 'H'};

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
    int64_t pos, // pointer offset for S in S0
    const T *A_ptr,
    int64_t lda,
    T beta,
    T *B_ptr,
    int64_t ldb
);

} // end namespace RandBLAS::dense_op

#endif  // define RandBLAS_UTIL_HH
