#ifndef randblas_dense_hh
#define randblas_dense_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/error.hh"

#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>

#include <math.h>
#include <typeinfo>

#if defined(RandBLAS_HAS_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace RandBLAS::dense {

using namespace RandBLAS::base;

enum class DenseDistName : char {
    Gaussian = 'G',         ///< a Gaussian with mean 0 and standard deviation 1
    Uniform = 'U',          ///< uniform over the interval [-1, 1].
    Rademacher = 'R',       ///< uniform over {+1, -1}.
    Haar = 'H',             ///< uniform over row-orthonormal or column-orthonormal matrices.
    DisjointIntervals = 'I' ///< might require additional metadata.
};

struct DenseDist {
    const DenseDistName family = DenseDistName::Gaussian;
    const int64_t n_rows;
    const int64_t n_cols;
};

template <typename T, typename RNG = r123::Philox4x32>
struct DenseSkOp {

    using generator = RNG;
    using state_type = RNGState<RNG>;
    using buffer_type = T;

    /////////////////////////////////////////////////////////////////////
    //
    //      Properties specific to dense sketching operators
    //
    /////////////////////////////////////////////////////////////////////

    const DenseDist dist;            ///< the name of the distribution and matrix size
    const RNGState<RNG> seed_state;  ///< the initial CBRNG state
    RNGState<RNG> next_state;        ///< the current CBRNG state
    const bool own_memory = true;    ///< a flag that indicates who owns the memory

    T *buff = nullptr;               ///< memory
    bool filled = false;             ///< a flag that indicates if the memory was initialized
    bool persistent = true;          ///< ???

    const blas::Layout layout = blas::Layout::ColMajor; ///< matrix storage order

    /////////////////////////////////////////////////////////////////////
    //
    //      Member functions must directly relate to memory management.
    //
    /////////////////////////////////////////////////////////////////////

    //  Elementary constructor: needs an implementation
    DenseSkOp(
        DenseDist dist_,
        RNGState<RNG> const& state_,
        T *buff_,
        bool filled_,
        bool persistent_,
        blas::Layout layout_
    );

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDist dist,
        uint32_t ctr_offset,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(dist, {ctr_offset, key}, buff, filled, persistent, layout) {};

    //  Convenience constructor (a wrapper)
    DenseSkOp(
        DenseDistName family,
        int64_t n_rows,
        int64_t n_cols,
        uint32_t ctr_offset,
        uint32_t key,
        T *buff,
        bool filled,
        bool persistent,
        blas::Layout layout
    ) : DenseSkOp(DenseDist{family, n_rows, n_cols}, ctr_offset,
                  key, buff, filled, persistent, layout) {};

    // Destructor
    ~DenseSkOp();
};

template <typename T, typename RNG>
DenseSkOp<T,RNG>::DenseSkOp(
    DenseDist dist_,
    RNGState<RNG> const& state_,
    T *buff_,           
    bool filled_,       
    bool persistent_,   
    blas::Layout layout_ 
) : // variable definitions
    dist(dist_),
    seed_state(state_),
    next_state{},
    own_memory(!buff_),
    buff(buff_),
    filled(filled_),
    persistent(persistent_),
    layout(layout_)
{   // sanity checks
    randblas_require(this->dist.n_rows > 0);
    randblas_require(this->dist.n_cols > 0);
    // Initialization logic
    //
    //      own_memory is a bool that's true iff buff_ is nullptr.
    //
    if (this->own_memory) {
        randblas_require(!this->filled);
        // We own the rights to the memory, and the memory
        // hasn't been allocated, so there's no way that the memory exists yet.
    } else {
        randblas_require(this->persistent);
        // If the user gives us any memory to work with, then we cannot take
        // responsibility for deallocating on exit from LSKGE3 / RSKGE3.
    }
}

template <typename T, typename RNG>
DenseSkOp<T,RNG>::~DenseSkOp() {
    if (this->own_memory) {
        delete [] this->buff;
    }
}





/** Fill a n_rows \times n_cols matrix with random values. If RandBLAS is
 * compiled with OpenMP threading support enabled, the operation is
 * parallelized using OMP_NUM_THREADS. The sequence of values genrated is not
 * dependent on the number of OpenMP threads.
 *
 * @tparam T the data type of the matrix
 * @tparam RNG a random123 CBRNG type
 * @tparm OP an operator that transforms raw random values into matrix
 *           elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] mat a pointer to a contiguous region of memory with space for
 *                n_rows \times n_cols elements of type T. This memory will be
 *                filled with random values.
 * @param[in] seed A CBRNG state
 *
 * @returns the updated CBRNG state
 */
template <typename T, typename RNG, typename OP>
auto fill_rmat(
    int64_t n_rows,
    int64_t n_cols,
    T* mat,
    const RNGState<RNG> & seed
) {
    RNG rng;
    // clang chokes on this w/ omp due to internal use of lambdas, fixed in C++20
    //auto [c, k] = seed;
    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type k = seed.key;

    int64_t dim = n_rows * n_cols;
    int64_t nit = dim / RNG::ctr_type::static_size;
    int64_t nlast = dim % RNG::ctr_type::static_size;

#if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel firstprivate(c, k)
    {
        // decompose the work into a set of approximately equal size chunks.
        // if the number of iterations is not evenly divisible by the number
        // of threads, the left over itertions are distributed one each among
        // the first threads.
        int ti = omp_get_thread_num();
        int nt = omp_get_num_threads();

        int64_t chs = nit / nt; // chunk size
        int64_t nlg = nit % nt; // number of large chunks
        int64_t i0 = chs * ti + (ti < nlg ? ti : nlg); // this threads start
        int64_t i1 = i0 + chs + (ti < nlg ? 1 : 0);    // this threads end

        // add the start index to the counter in order to make the sequence
        // deterministic independent of the number of threads.
        auto cc = c;
        cc.incr(i0);
#else
        int64_t i0 = 0;
        int64_t i1 = nit;
        auto &cc = c;
#endif
        for (int64_t i = i0; i < i1; ++i) {

            auto rv = OP::generate(rng, cc, k);

            for (int j = 0; j < RNG::ctr_type::static_size; ++j) {
               mat[RNG::ctr_type::static_size * i + j] = rv[j];
            }

            cc.incr();
        }
#if defined(RandBLAS_HAS_OpenMP)
    }
    // puts the counter in the correct state when threads are used.
    c.incr(nit);
#endif

    if (nlast) {
        auto rv = OP::generate(rng, c, k);

        for (int64_t j = 0; j < nlast; ++j) {
            mat[RNG::ctr_type::static_size * nit + j] = rv[j];
        }

        c.incr();
    }

    return RNGState<RNG> {c, k};
}


template <typename T, typename RNG>
auto fill_buff(
    T *buff,
    const DenseDist &D,
    RNGState<RNG> const& state
) {
    switch (D.family) {
        case DenseDistName::Gaussian:
            return fill_rmat<T,RNG,r123ext::boxmul>(D.n_rows, D.n_cols, buff, state);
        case DenseDistName::Uniform:
            return fill_rmat<T,RNG,r123ext::uneg11>(D.n_rows, D.n_cols, buff, state);
        case DenseDistName::Rademacher:
            throw std::runtime_error(std::string("Not implemented."));
        case DenseDistName::Haar:
            // This won't be filled IID, but a Householder representation
            // of a column-orthonormal matrix Q can be stored in the lower
            // triangle of Q (with "tau" on the diagonal). So the size of
            // buff will still be D.n_rows*D.n_cols.
            throw std::runtime_error(std::string("Not implemented."));
        default:
            throw std::runtime_error(std::string("Unrecognized distribution."));
    }

    return state;
}

template <typename SKOP>
auto fill_skop_buff(
    SKOP &S0
) {
    auto S0_ptr = S0.buff;
    if (S0_ptr == nullptr) {
        S0_ptr = new typename SKOP::buffer_type [S0.dist.n_rows * S0.dist.n_cols];
        S0.next_state = fill_buff(S0_ptr, S0.dist, S0.seed_state);
        if (S0.persistent) {
            S0.buff = S0_ptr;
            S0.filled = true;
        }
        return S0_ptr;
    } else if (!S0.filled) {
        S0.next_state = fill_buff(S0_ptr, S0.dist, S0.seed_state);
        S0.filled = true;
        return S0_ptr;
    } else {
        return S0_ptr;
    }
}

template <typename T, typename RNG>
void lskge3(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    DenseSkOp<T,RNG> &S0,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
){
    randblas_require(S0.layout == layout);

    auto S0_ptr = fill_skop_buff(S0);

    // Dimensions of A, rather than op(A)
    int64_t rows_A, cols_A, rows_S, cols_S;
    if (transA == blas::Op::NoTrans) {
        rows_A = m;
        cols_A = n;
    } else {
        rows_A = n;
        cols_A = m;
    }
    // Dimensions of S, rather than op(S)
    if (transS == blas::Op::NoTrans) {
        rows_S = d;
        cols_S = m;
    } else {
        rows_S = m;
        cols_S = d;
    }

    // Sanity checks on dimensions and strides
    int64_t lds, pos;
    if (layout == blas::Layout::ColMajor) {
        lds = S0.dist.n_rows;
        pos = i_os + lds * j_os;
        randblas_require(lds >= rows_S);
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        lds = S0.dist.n_cols;
        pos = i_os * lds + j_os;
        randblas_require(lds >= cols_S);
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }
    // Perform the sketch.
    blas::gemm(
        layout, transS, transA,
        d, n, m,
        alpha,
        &S0_ptr[pos], lds,
        A, lda,
        beta,
        B, ldb
    );
    return;
}


#if defined(RandBLAS_HAS_CUDA)
/** fills the column major storage order matrix with random values
 *
 * @tparam T the data type
 * @tparam RNG a counter based random number generator
 * @tparam OP an operator that transforms raw random values into matrix
 *            elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] ldim the number of conseecutive elements between columns
 * @param[in] mat a contiguous section of memory with ldim \times n_cols elements
 * @param[in] seed the counter state
 */
template <typename T, typename RNG, typename OP>
__global__
void fill_rmat_row_maj(
    int64_t n_rows,
    int64_t n_cols,
    int64_t ldim,
    T *mat,
    RNGState<RNG> seed
)  {
    RNG rng;

    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type const & k = seed.key;

    // work in static_size blocks of elements
    int64_t n_elem = n_rows * n_cols;

    int64_t n_elem_ss = n_elem / RNG::ctr_type::static_size +
        (n_elem % RNG::ctr_type::static_size ? 1 : 0);

    int64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // add the start index to the counter in order to make the sequence
    // deterministic independent of the number of threads.
    c.incr(thread_id);
    auto rv = OP::generate(rng, c, k);

    for (int q = 0; q < RNG::ctr_type::static_size; ++q) {

        // get the flat index
        int64_t idx = thread_id + n_elem_ss * q;

        // convert to 2D i,j indices
        int64_t i = idx % n_cols;
        int64_t j = idx / n_cols;

        // bounds check
        if ((idx > n_elem) || (j > n_cols))
            continue;

        // store the random value
        mat[i*ldim + j] = rv[q];
    }
}

/** fills the column major storage order matrix with random values
 *
 * @tparam T the data type
 * @tparam RNG a counter based random number generator
 * @tparam OP an operator that transforms raw random values into matrix
 *            elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] ldim the number of conseecutive elements between columns
 * @param[in] mat a contiguous section of memory with ldim \times n_cols elements
 * @param[in] seed the counter state
 */
template <typename T, typename RNG, typename OP>
__global__
void fill_rmat_col_maj(
    int64_t n_rows,
    int64_t n_cols,
    int64_t ldim,
    T *mat,
    RNGState<RNG> seed
)  {
    RNG rng;

    typename RNG::ctr_type c = seed.counter;
    typename RNG::key_type const & k = seed.key;

    // work in static_size blocks of elements
    int64_t n_elem = n_rows * n_cols;

    int64_t n_elem_ss = n_elem / RNG::ctr_type::static_size +
        (n_elem % RNG::ctr_type::static_size ? 1 : 0);

    int64_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // add the start index to the counter in order to make the sequence
    // deterministic independent of the number of threads.
    c.incr(thread_id);
    auto rv = OP::generate(rng, c, k);

    for (int q = 0; q < RNG::ctr_type::static_size; ++q) {

        // get the flat index
        int64_t idx = thread_id + n_elem_ss * q;

        // convert to 2D i,j indices
        int64_t i = idx / n_rows;
        int64_t j = idx % n_rows;

        // bounds check
        if ((idx > n_elem) || (i > n_rows))
            continue;

        // store the random value
        mat[j*ldim + i] = rv[q];
    }
}

/** partition CBRNG work for the GPU
 *
 * @param[in] n_elem the number of rows times the number of columns
 * @param[in] ctr_size the number of values returned by a single call to the
 *                     the CBRNG
 * @param[in] block_size the number of CUDA threads per block
 * @returns the number of blocks
 */
static
auto partition_cbrng_threads(
    int64_t n_elem,
    int ctr_size,
    int block_size = 256
) {
    // work in chunks of static size elements. random123 return this number of
    // values per call.
    int64_t n_elem_ss = n_elem / ctr_size + (n_elem % ctr_size ? 1 : 0);

    // this is the number of thread blocks we will use
    int64_t n_blocks = n_elem_ss / block_size + (n_elem_ss % block_size ? 1 : 0);

    // check that we haven't exceeded device capabilities. we could
    // go to a 2 or 3D decomp to recover, left for a future enhancement
    if (n_blocks >= ((1l << 31) - 1)) {
        RB_RUNTIME_ERROR("Exceeded max number of CUDA blocks")
    }

    return std::make_tuple(dim3(n_blocks), dim3(block_size));
}

/** Fill a n_rows \times n_cols matrix with random values.
 *
 * @tparam T the data type of the matrix
 * @tparam RNG a random123 CBRNG type
 * @tparm OP an operator that transforms raw random values into matrix
 *           elements. See r123ext::uneg11 and r123ext::boxmul.
 *
 * @param[in] n_rows the number of rows in the matrix
 * @param[in] n_cols the number of columns in the matrix
 * @param[in] mat a pointer to a contiguous region of memory with space for
 *                n_rows \times n_cols elements of type T. This memory will be
 *                filled with random values.
 * @param[in] seed A CBRNG state
 *
 * @returns the updated CBRNG state
 */
template <typename T, typename RNG, typename OP>
auto fill_rmat(
    blas::Layout layout,
    int64_t n_rows,
    int64_t n_cols,
    int64_t ldim,
    T* mat,
    const RNGState<RNG> & seed,
    cudaStream_t strm
) {
    using ctr_type = typename RNG::ctr_type;
    using key_type = typename RNG::key_type;

    ctr_type c = seed.counter;
    const key_type &k = seed.key;

    int64_t n_elem = n_rows*n_cols;

    // determine kernel launch parameters
    auto [blocks, threads] = partition_cbrng_threads(n_elem, ctr_type::static_size);

    if (layout == blas::Layout::ColMajor) {
        // generate the matrix column major layout
        fill_rmat_col_maj<<<blocks, threads, 0, strm>>>(n_rows, n_cols, ldim, mat, seed);
    }
    else if (layout == blas::Layout::RowMajor) {
        // generate the matrix row major layout
        fill_rmat_row_maj<<<blocks, threads, 0, strm>>>(n_rows, n_cols, ldim, mat, seed);
    }
    else {
        RB_RUNTIME_ERROR("Invalid layout " << (int)layout)
        return seed;
    }

    // check for error in kernel launch
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RB_RUNTIME_ERROR("Kernel launch failed. " << cudaGetErrorString(ierr))
        return seed;
    }

    // update the counter state
    c.incr(n_elem);

    return RNGState<RNG> {c, k};
}
#endif


} // end namespace RandBLAS::dense

#endif
