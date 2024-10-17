#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include <RandBLAS/sparse_skops.hh>

#include <Random123/philox.h>
#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>

#include <math.h>
#include <typeinfo>

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS {

    // =============================================================================
    /// WARNING: None of the following functions or overloads thereof are part of the
    /// public API
    ///

    // Generates a vector of Rademacher entries using the Random123 library
    template<SignedInteger sint_t, typename RNG = r123::Philox4x32>
    void generate_rademacher_vector_r123(sint_t* buff, uint32_t key_seed, uint32_t ctr_seed, int64_t n) {
        RNG rng;

        typename RNG::ctr_type c;
        typename RNG::key_type key = {{key_seed}};

        // Sequential loop for generating Rademacher entries
        for (int64_t i = 0; i < n; ++i) {
            // Set the counter for each random number
            c[0] = ctr_seed + i;  // Ensure each counter is unique

            // Generate a 2x32-bit random number using the Philox generator
            typename RNG::ctr_type r = rng(c, key);

            // Convert the random number into a float in [0, 1) using u01fixedpt
            float rand_value = r123::u01fixedpt<float>(r.v[0]);

            // Convert the float into a Rademacher entry (-1 or 1)
            buff[i] = rand_value < 0.5 ? -1 : 1;
        }
    }

    template<SignedInteger sint_t, typename RNG = r123::Philox4x32>
    RandBLAS::RNGState<RNG> generate_rademacher_vector_r123(sint_t* buff, int64_t n, RandBLAS::RNGState<RNG> seed_state) {
        RNG rng;
        auto [ctr, key] = seed_state;

        for (int64_t i = 0; i < n; ++i) {
            typename RNG::ctr_type r = rng(ctr, key);

            float rand_value = r123::u01fixedpt<float>(r.v[0]);

            buff[i] = rand_value < 0.5 ? -1 : 1;

            ctr.incr();
        }

        // Return the updated RNGState (with the incremented counter)
        return RandBLAS::RNGState<RNG> {ctr, key};
    }

    // Catch-all method for applying the diagonal Rademacher
    // entries in-place to an input matrix, `A`
    template<typename T, SignedInteger sint_t>
    void apply_diagonal_rademacher(
                                bool left,
                                blas::Layout layout,
                                int64_t rows,
                                int64_t cols,
                                T* A,
                                sint_t* diag
                                ) {
        //TODO: Investigate better schemes for performing the scaling
        //TODO: Move to `RandBLAS/util.hh`
        if(left && layout == blas::Layout::ColMajor) {
            for(int64_t col=0; col < cols; col++) {
                if(diag[col] > 0)
                    continue;
                blas::scal(rows, diag[col], &A[col * rows], 1);
            }
        }
        else if(left && layout == blas::Layout::RowMajor) {
            for(int64_t col=0; col < cols; col++) {
                if(diag[col] > 0)
                    continue;
                blas::scal(rows, diag[col], &A[col], cols);
            }
        }
        else if(!left && layout == blas::Layout::ColMajor) {
            for(int64_t row = 0; row < rows; row++) {
                if(diag[row] > 0)
                    continue;
                blas::scal(cols, diag[row], &A[row], rows);
            }
        }
        else {
            for(int64_t row = 0; row < rows; row++) {
                if(diag[row] > 0)
                    continue;
                blas::scal(cols, diag[row], &A[row * cols], 1);
            }
        }
    }

    template<typename T, SignedInteger sint_t>
    void permuteRowsToTop(
                          blas::Layout layout,
                          int64_t rows,
                          int64_t cols,
                          sint_t* selected_rows,
                          int64_t d, // size of `selected_rows`
                          T* A
                          ) {
        int64_t top = 0;  // Keeps track of the topmost unselected row

        //TODO: discuss precise semantics of `selected_rows` in this function
        if(layout == blas::Layout::ColMajor) {
            for (int64_t i=0; i < d; i++) {
                randblas_error_if_msg(selected_rows[i] == top,
                                      "The list of provided indices should be unique");
                if (selected_rows[i] != top) {
                    // Use BLAS swap to swap the entire rows
                    // Swapping row 'selected' with row 'top'
                    blas::swap(cols, &A[top], rows, &A[selected_rows[i]], rows);
                    // top = selected_rows[i];
                }
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                randblas_error_if_msg(selected_rows[i] == top,
                                      "The list of provided indices should be unique");
                std::cout << "see here" << selected_rows[i] << std::endl;
                if (selected_rows[i] != top) {
                    blas::swap(cols, &A[cols * selected_rows[i]], 1, &A[cols * top], 1);
                    // top = selected_rows[i];
                }
            }
        }
    }

    template<typename T, SignedInteger sint_t>
    void permuteColsToLeft(
                          blas::Layout layout,
                          int64_t rows,
                          int64_t cols,
                          sint_t* selected_cols,
                          int64_t d, // size of `selectedRows`
                          T* A
                          ) {
        int64_t left = 0;  // Keeps track of the topmost unselected column

        if(layout == blas::Layout::ColMajor) {
            for (int64_t i=0; i < d; i++) {
                if (selected_cols[i] != left) {
                    // Use BLAS::swap to swap entire columns at once
                    // Swapping col 'selected' with col 'top'
                    blas::swap(rows, &A[rows * selected_cols[i]], 1, &A[rows * left], 1);
                }
                // left++;
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                if (selected_cols[i] != left) {
                    blas::swap(rows, &A[selected_cols[i]], cols, &A[left], cols);
                }
                // left++;
            }
        }
    }

    template <typename T>
    void fht_left_col_major(T *buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each column independently
        for (int64_t col = 0; col < num_cols; ++col) {
            // Pointer to the beginning of the current column in the Column-Major order
            T* col_buf = buf + col * num_rows;

            // Apply the original FHT on this column
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        // For implicitly padding the input we just have to make sure
                        // we replace all out-of-bounds accesses with zeros
                        bool b1 = j + k < num_rows;
                        bool b2 = j + k + s1 < num_rows;
                        T u = b1 ? col_buf[j + k] : 0;
                        T v = b2 ? col_buf[j + k + s1] : 0;
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

    template <typename T>
    void fht_left_row_major(T *buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each column independently
        for (int64_t col = 0; col < num_cols; ++col) {
            // Apply the original FHT on this column
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        // For implicitly padding the input we just have to make sure
                        // we replace all out-of-bounds accesses with zeros
                        bool b1 = j + k < num_rows;
                        bool b2 = j + k + s1 < num_rows;
                        T u = b1 ? buf[(j + k) * num_cols + col] : 0;
                        T v = b2 ? buf[(j + k + s1) * num_cols + col] : 0;
                        if(b1 && b2) {
                            buf[(j + k) * num_cols + col] = u + v;
                            buf[(j + k + s1) * num_cols + col] = u - v;
                        }
                        else if(!b2 && b1) {
                            buf[(j + k) * num_cols + col] = u + v;
                        }
                        else if(!b2 && !b1)
                            continue;
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_right_row_major(T *buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each row independently
        for (int64_t row = 0; row < num_rows; ++row) {
            // Pointer to the beginning of the current row in RowMajor order
            T * row_buf = buf + row * num_cols;

            // Apply the original FHT on this row
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        // For implicitly padding the input we just have to make sure
                        // we replace all out-of-bounds accesses with zeros
                        bool b1 = j + k < num_cols;
                        bool b2 = j + k + s1 < num_cols;
                        T u = b1 ? row_buf[j + k] : 0;
                        T v = b2 ? row_buf[j + k + s1] : 0;
                        if(b1 && b2) {
                            row_buf[j + k] = u + v;
                            row_buf[j + k + s1] = u - v;
                        }
                        else if(!b2 && b1) {
                            row_buf[j + k] = u + v;
                        }
                        else if(!b2 && !b1)
                            continue;
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_right_col_major(T *buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each row independently
        for (int64_t row= 0; row < num_rows; ++row) {
            // Apply the original FHT on this column
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        // For implicitly padding the input we just have to make sure
                        // we replace all out-of-bounds accesses with zeros
                        bool b1 = j + k < num_cols;
                        bool b2 = j + k + s1 < num_cols;
                        T u = b1 ? buf[(j + k) * num_rows + row] : 0;
                        T v = b2 ? buf[(j + k + s1) * num_rows + row] : 0;
                        if(b1 && b2) {
                            buf[(j + k) * num_rows + row] = u + v;
                            buf[(j + k + s1) * num_rows + row] = u - v;
                        }
                        else if(!b2 && b1) {
                            buf[(j + k) * num_rows + row] = u + v;
                        }
                        else if(!b2 && !b1)
                            continue;
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_dispatch(
        bool left,
        blas::Layout layout,
        T* buff,
        int64_t log_n,
        int64_t num_rows,
        int64_t num_cols
        )
    {
        if(left && layout == blas::Layout::ColMajor)
            fht_left_col_major(buff, log_n, num_rows, num_cols);
        else if(left && layout == blas::Layout::RowMajor)
            fht_left_row_major(buff, log_n, num_rows, num_cols);
        else if(!left && layout == blas::Layout::ColMajor)
            fht_right_col_major(buff, log_n, num_rows, num_cols);
        else
            fht_right_row_major(buff, log_n, num_rows, num_cols);
    }
}


namespace RandBLAS::trig {
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
) {
    // Size of the Rademacher entries = |A_cols|
    //TODO: Change `diag` to float/doubles (same data type as the matrix)
    sint_t* diag = new sint_t[n];
    sint_t* selected_rows = new sint_t[d];

    auto [ctr, key] = random_state;

    //Step 1: Scale with `D`
        //Populating `diag`
    generate_rademacher_vector_r123(diag, key[0], ctr[0], n);
    apply_diagonal_rademacher(true, layout, m, n, A, diag);

    //Step 2: Apply the Hadamard transform
    fht_dispatch(true, layout, A, std::log2(MAX(m, n)), m, n);

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

    permuteRowsToTop(layout, m, n, selected_rows, d, A);

    free(diag);
    free(selected_rows);
}


template <typename T, typename RNG = r123::Philox4x32, SignedInteger sint_t = int64_t>
inline void rmiget(
    blas::Layout layout,
    RandBLAS::RNGState<RNG> random_state,
    int64_t m, // `A` is `(m x n)`
    int64_t n,
    int64_t d, // `d` is the number of cols that have to be permuted by `\Pi`
    T* A // data-matrix
)
{
    // Size of the Rademacher entries = |A_cols|
    //TODO: Change `diag` to float/doubles (same data type as the matrix)
    sint_t* diag = new sint_t[m];
    sint_t* selected_cols = new sint_t[d];

    auto [ctr, key] = random_state;

    //Step 1: Scale with `D`
        //Populating `diag`
    generate_rademacher_vector_r123(diag, key[0], ctr[0], n);
    apply_diagonal_rademacher(false, layout, m, n, A, diag);

    //Step 2: Apply the Hadamard transform
    fht_dispatch(false, layout, A, std::log2(MAX(m, n)), m, n);

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
        selected_cols,  // Holds the required output
        idxs_minor.data(),  // Placeholder
        vals.data()         // Placeholder
    );

    permuteColsToLeft(layout, m, n, selected_cols, d, A);

    free(diag);
    free(selected_cols);
}
}
