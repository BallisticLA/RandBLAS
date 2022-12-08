#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#ifndef RandBLAS_DO_HH
#define RandBLAS_DO_HH

/*
Paradigm for APIs involving structs:
    Free-functions when there's no memory to manage
    Member functions when there IS memory to manage, or in initializing.
        We want to make this library hard to misuse in C++.
    We provide APIs that we require people use to ensure that structs are
        in a valid state. If you want to initialize the struct yourself
        we won't stop you, but we also take no responsibility for the 
        inevitable segfaults.

TODO: have a discussion around using smart pointers for memory safety.
    Burlen thinks we should seriously consider using smart pointers.
*/

namespace RandBLAS::dense {

enum class DenseDistName : char {
    Gaussian = 'G',
    Normal = 'G', // alias, for user convenience
    Uniform = 'U', // uniform over the interval [-1, 1].
    Rademacher = 'R', // uniform over {+1, -1}.
    Haar = 'H', // uniform over row-orthonormal or column-orthonormal matrices.
    DisjointIntervals = 'I' // might require additional metadata.
};

struct DenseDist {
    const DenseDistName family = DenseDistName::Gaussian;
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
struct DenseSkOp {
    // Unlike a plain buffer that we might use in BLAS,
    // SketchingOperators in the RandBLAS::dense namespace
    // carry metadata to unambiguously define their dimensions
    // and the values of their entries.
    // 
    // Dimensions are specified with the distribution, in "dist".
    //
    const DenseDist dist{};
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
    DenseDist D,
    uint32_t key,
    uint32_t ctr_offset
);
// ^ A "free function."

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
    DenseSkOp<T> &S0,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
);

} // end namespace RandBLAS::dense

#endif  // define RandBLAS_UTIL_HH
