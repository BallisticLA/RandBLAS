#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include <RandBLAS/sparse_skops.hh>

#include <Random123/philox.h>
#include <blas.hh>
#include <lapack.hh>
#include <omp.h>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <random>

#include <math.h>
#include <typeinfo>

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS {

    // =============================================================================
    /// WARNING: None of the following functions or overloads thereof are part of the
    /// public API
    ///
    template<SignedInteger sint_t>
    // void generateRademacherVector_r123(std::vector<int64_t> &buff, uint32_t key_seed, uint32_t ctr_seed, int64_t n) {
    void generateRademacherVector_r123(sint_t* buff, uint32_t key_seed, uint32_t ctr_seed, int64_t n) {
        typedef r123::Philox2x32 RNG;
        // std::vector<int64_t> rademacherVector(n);

        // Use OpenMP to parallelize the Rademacher vector generation
        #pragma omp parallel
        {
            // Each thread has its own RNG instance
            RNG rng;

            RNG::key_type k = {{key_seed + omp_get_thread_num()}}; // Unique key for each thread

            // Thread-local counter
            RNG::ctr_type c;

            // Parallel for loop
            #pragma omp for
            for (int i = 0; i < n; ++i) {
                // Set the counter for each random number (unique per thread)
                c[0] = ctr_seed + i; // Ensure the counter is unique across threads

                // Generate a 2x32-bit random number using the Philox generator
                RNG::ctr_type r = rng(c, k);

                // Convert the random number into a float in [0, 1) using u01fixedpt
                float randValue = r123::u01fixedpt<float>(r.v[0]);

                // Convert the float into a Rademacher entry (-1 or 1)
                buff[i] = randValue < 0.5 ? -1 : 1;
            }
        }
    }

    std::vector<int64_t> generateRademacherVector_parallel(int64_t n) {
        std::vector<int64_t> rademacherVec(n);

        #pragma omp parallel
        {
            // Each thread gets its own random number generator
            std::random_device rd;
            std::mt19937 gen(rd());
            std::bernoulli_distribution bernoulli(0.5);

            #pragma omp for
            for (int64_t i = 0; i < n; ++i) {
                rademacherVec[i] = bernoulli(gen) ? 1 : -1;
            }
        }

        return rademacherVec;
    }

    template<typename T>
    void applyDiagonalRademacher(int64_t rows, int64_t cols, T* A) {
        std::vector<int64_t> diag = generateRademacherVector_parallel(cols);

        for(int col=0; col < cols; col++) {
        if(diag[col] > 0)
            continue;
        blas::scal(rows, diag[col], &A[col * rows], 1);
    }
    }

    template<typename T, SignedInteger sint_t>
    // void applyDiagonalRademacher(int64_t rows, int64_t cols, T* A, std::vector<int64_t> &diag) {
    void applyDiagonalRademacher(
                                blas::Layout layout,
                                int64_t rows,
                                int64_t cols,
                                T* A,
                                sint_t* diag
                                )
    {
        //TODO: Only considers sketching from the left + ColMajor format as of now
        if(layout == blas::Layout::ColMajor) {
            // In a `ColMajor` setting we parallelize over columns
            for(int col=0; col < cols; col++) {
                if(diag[col] > 0)
                    continue;
                blas::scal(rows, diag[col], &A[col * rows], 1);
            }
        }
        else {
            // In a `RowMajor` setting we vectorize over rows
        }
    }

    // `applyDiagonalRademacher`should have a similar API as `sketch_general`
    // Will be called inside of `lskget`
    // B <- alpha * diag(Rademacher) * op(A) + beta * B
    // `alpha`, `beta` aren't needed and shape(A) == shape(B)
    // lda == ldb
    template<typename T, SignedInteger sint_t>
    void applyDiagonalRademacher(
        blas::Layout layout,
        blas::Op opA, // taken from `sketch_general/lskget`
        int64_t n, // everything is the same size as `op(A)`: m-by-n
        int64_t m,
        // std::vector<int64_t> &diag,
        sint_t* diag,
        const T* A, // The data matrix that won't be modified
        // int64_t lda,
        T* B // The destination matrix
        // int64_t ldb
    ) {
        // B <- alpha * diag(Rademacher) * op(A) + beta * B
        // `alpha`, `beta` aren't needed
        // && shape(A) == shape(B) && lda == ldb && shape({A | B}) = m-by-n
        //NOTE: Use `layout` and `opA` for working with RowMajor data
        int64_t lda = m;
        int64_t ldb = m;

        //NOTE: Should the copy be made inside of `applyDiagonalRademacher` or
        // should it be inside of `lskget`?
        lapack::lacpy(lapack::MatrixType::General, m, n, A, lda, B, ldb);

        applyDiagonalRademacher(m, n, B, diag);
    }


    // `permuteRowsToTop` should just take in `B` as an input
    // `B` will be passed in by the user to `sketch_general/lskget` and `permuteRowsToTop` will take in
    // the `B` that has been modified by `applyRademacherDiagonal`
    // Will also be called inside of `lskget`
    template<typename T, SignedInteger sint_t>
    void permuteRowsToTop(
                          blas::Layout layout,
                          int64_t rows,
                          int64_t cols,
                          sint_t* selectedRows,
                          int64_t d, // size of `selectedRows`
                          T* A
                          ) {
        //NOTE: There should be a similar `permuteColsToLeft`
        int top = 0;  // Keeps track of the topmost unselected row

        int64_t lda = rows;
        if(layout == blas::Layout::RowMajor)
            lda = cols;

        for (int i=0; i < d; i++) {
            if (selectedRows[i] != top) {
                // Use BLAS swap to swap the entire rows
                // Swapping row 'selected' with row 'top'
                blas::swap(cols, &A[selectedRows[i]], lda, &A[top], lda);
            }
            top++;
        }
    }

    void fht_left_col_major(double *buf, int log_n, int num_rows, int num_cols) {
        // No Padding of the columns in this implementation,
        // the #rows must exactly be a power of 2
        // Padding would be straight-forward to address
        int n = 1 << log_n;
        std::cout << n << std::endl;

        // Apply FHT to each column independently
        for (int col = 0; col < num_cols; ++col) {
            // Pointer to the beginning of the current column in the Column-Major order
            double* col_buf = buf + col * num_rows;

            // Apply the original FHT on this column
            for (int i = 0; i < log_n; ++i) {
                int s1 = 1 << i;
                int s2 = s1 << 1;
                for (int j = 0; j < n; j += s2) {
                    for (int k = 0; k < s1; ++k) {
                        // For implicitly padding the input we just have to make sure
                        // we replace all out-of-bounds accesses with zeros
                        bool b1 = j + k < num_rows;
                        bool b2 = j + k + s1 < num_rows;
                        double u = b1 ? col_buf[j + k] : 0;
                        double v = b2 ? col_buf[j + k + s1] : 0;
                        if(b1 && b2) {
                            col_buf[j + k] = u + v;
                            col_buf[j + k + s1] = u - v;
                        }
                        else if(!b2 && b1) {
                            col_buf[j + k] = u + v;
                        }
                        else if(!b2 && !b1)
                            continue;
                    }
                }
            }
        }
    }

    enum class TrigDistName: char {
        Fourier = 'F',

        // ---------------------------------------------------------------------------

        Hadamard = 'H'
    };

    struct TrigDist {
        const int64_t n_rows;
        const int64_t n_cols;

        int64_t dim_short;
        int64_t dim_long;

        const TrigDistName family;

        TrigDist(
            int64_t n_rows,
            int64_t n_cols,
            TrigDistName tn = TrigDistName::Hadamard
        ) : n_rows(n_rows), n_cols(n_cols), family(tn) {
            dim_short = MIN(n_rows, n_cols);
            dim_long = MAX(n_rows, n_cols);
}
};

    template<typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
    struct TrigSkOp {
        using generator = RNG;
        using state_type = RNGState<RNG>;
        using buffer_type = T;

        //TODO: Where should the logic for deciding the size of `H` to use go?
        // Since that will be accompanied by padding of (DA) maybe it should
        // go inside of `lskget`?
        const int64_t n_rows;
        const int64_t n_cols;
        int64_t dim_short;
        int64_t dim_long;
        // int64_t n_sampled;

        const TrigDist dist;


        const RNGState<RNG> seed_state;
        RNGState<RNG> next_state;

        const blas::Layout layout;
        const bool sketchFromLeft = true;
        bool known_filled = false;

        sint_t* DiagonalRademacher = nullptr;
        sint_t* SampledRows = nullptr;

        TrigSkOp(
            TrigDist dist,
            RNGState<RNG> const &state,
            blas::Layout layout,
            bool known_filled = false
        ) : n_rows(dist.n_rows), n_cols(dist.n_cols), dist(dist), seed_state(state), known_filled(known_filled), dim_short(dist.dim_short), dim_long(dist.dim_long), layout(layout){
            // Memory for Rademacher diagonal gets allocated here
            // Number of rows/cols to be sampled gets decided here
            // i.e. `n_sampled` gets set
            if(sketchFromLeft)
                DiagonalRademacher = new sint_t[n_rows];
            else
                DiagonalRademacher = new sint_t[n_cols];

            SampledRows = new sint_t[dim_short];

            //TODO: Logic to compute the number of samples that we require `r`
            // this can be shown to depend on the maximal row norm of the data matrix
            //NOTE: Do not have access to this in the data-oblivious regime --- how
            // do we get access?
};

        TrigSkOp(
            TrigDistName family,
            int64_t n_rows,
            int64_t n_cols,
            uint32_t key
        ) : n_rows(n_rows), n_cols(n_cols), dist(TrigDist{n_rows, n_cols, family}), seed_state() {};

        //TODO: Write a proper deconstructor
        //Free up DiagonalRademacher && SampledRows
        ~TrigSkOp();
};

template <typename T, typename RNG, SignedInteger sint_t>
TrigSkOp<T, RNG, sint_t>::~TrigSkOp() {
    delete [] this->DiagonalRademacher;
    delete [] this->SampledRows;
};

template<typename T, typename RNG, SignedInteger sint_t>
RandBLAS::RNGState<RNG> fill_trig(
    TrigSkOp<T, RNG, sint_t> &Tr
) {
    /**
     * Will do the work of filling in the diagonal Rademacher entries
     * and selecting the rows/cols to be sampled
     */
    auto [ctr, key] = Tr.seed_state;

    // Fill in the Rademacher diagonal
    if(Tr.sketchFromLeft)
        generateRademacherVector_r123(Tr.DiagonalRademacher, key[0], ctr[0], Tr.n_rows);
    else
        generateRademacherVector_r123(Tr.DiagonalRademacher, key[0], ctr[0], Tr.n_cols);

    //NOTE: Select the rows/cols to be sampled --- use the `repeated_fisher_yates` function
    int64_t r = Tr.dim_short;
    int64_t d = Tr.dim_long;

    std::vector<sint_t> idxs_minor(r); // Placeholder
    std::vector<T> vals(r); // Placeholder

    Tr.next_state = RandBLAS::repeated_fisher_yates<T, RNG, sint_t>(
        Tr.seed_state,
        r,         // Number of samples (vec_nnz)
        d,         // Total number of elements (dim_major)
        1,         // Single sample round (dim_minor)
        Tr.SampledRows,  // Holds the required output
        idxs_minor.data(),  // Placeholder
        vals.data()         // Placeholder
    );
    Tr.known_filled = true;
    return Tr.next_state;
}
}


namespace RandBLAS::trig {

// Performs the actual application of the fast trigonometric transform
// Only called after calling `fill_trig`
template <typename T, typename RNG>
inline void lskget(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    TrigSkOp<T, RNG> &Tr,
    int64_t ro_s,
    int64_t co_s,
    const T* A, // data-matrix
    int64_t lda,
    T beta,
    T* B, // output matrix
    int64_t ldb
) {
    if (!Tr.known_filled)
        fill_trig(Tr);

    // Applying the diagonal transformation
    applyDiagonalRademacher(layout, opA, n, m, Tr.DiagonalRademacher, A, B);
    // applyDiagonalRademacher(m, n, A, Tr.DiagonalRademacher);

    //TODO: Apply the Hadamard transform

    //... and finally permute the rows
    // permuteRowsToTop(m, n, Tr.SampledRows, B, ldb);
}


/*
 * These functions apply an in-place, SRHT-like transform to the input matrix
 * i.e. A <- (\Pi H D)A OR A <- A(D H \Pi) (which is equivalent to A <- A(\Pi H D)^{-1})
 * layout: Layout of the input matrix (`ColMajor/RowMajor`)
 * A: (m x n), input dimensions of `A`
 * d: The number of rows/columns that will be permuted by the action of $\Pi$
 */
template <typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
inline void lmiget(
    blas::Layout layout,
    RandBLAS::RNGState<RNG> random_state,
    int64_t m, // `A` is `(m x n)`
    int64_t n,
    int64_t d, // `d` is the number of rows that have to be permuted by `\Pi`
    T* A // data-matrix
)
{
    // Size of the Rademacher entries = |A_cols|
    //TODO: Change `diag` to float/doubles (same data type as the matrix)
    sint_t* diag = new sint_t[n];
    sint_t* selected_rows = new sint_t[d];

    auto [ctr, key] = random_state;

    //Step 1: Scale with `D`
        //Populating `diag`
    generateRademacherVector_r123(diag, key[0], ctr[0], n);
    // applyDiagonalRademacher(layout, m, n, A, diag);

    //Step 2: Apply the Hadamard transform
    fht_left_col_major(A, std::log2(MAX(m, n)), m, n);

    //Step 3: Permute the rows
    std::vector<sint_t> idxs_minor(d); // Placeholder
    std::vector<T> vals(d); // Placeholder

        // Populating `selected_rows`
        //TODO: Do I return this at some point?
    RandBLAS::RNGState<RNG> next_state = RandBLAS::repeated_fisher_yates<T, RNG, sint_t>(
        random_state,
        d,         // Number of samples (vec_nnz)
        m,         // Total number of elements (dim_major)
        1,         // Single sample round (dim_minor)
        selected_rows,  // Holds the required output
        idxs_minor.data(),  // Placeholder
        vals.data()         // Placeholder
    );

    // permuteRowsToTop(layout, m, n, selected_rows, d, A);

    free(diag);
    free(selected_rows);
}
}
