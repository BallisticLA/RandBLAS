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
#include <sys/_types/_int64_t.h>
#include <tuple>

#include <math.h>
#include <typeinfo>
#include <vector>

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

namespace RandBLAS {
    // =============================================================================
    /// WARNING: None of the following functions or overloads thereof are part of the
    /// public API
    ///

    // Generates a vector of Rademacher entries using the Random123 library
    template<SignedInteger sint_t = int64_t, typename RNG = DefaultRNG>
    RNGState<RNG> generate_rademacher_vector_r123(sint_t* buff, int64_t n, RNGState<RNG> seed_state) {
        RNG rng;
        auto [ctr, key] = seed_state;

        for (int64_t i = 0; i < n; ++i) {
            typename RNG::ctr_type r = rng(ctr, key);

            float rand_value = r123::u01fixedpt<float>(r.v[0]);

            buff[i] = rand_value < 0.5 ? -1 : 1;

            ctr.incr();
        }

        // Return the updated RNGState (with the incremented counter)
        return RNGState<RNG> {ctr, key};
    }

    // Catch-all method for applying the diagonal Rademacher
    // entries in-place to an input matrix, `A`
    template<typename T, SignedInteger sint_t = int64_t>
    void apply_diagonal_rademacher(
                                bool left, // Pre-multiplying?
                                blas::Layout layout,
                                int64_t rows,
                                int64_t cols,
                                T* A,
                                sint_t* diag
                                ) {
        //TODO: Investigate better schemes for performing the scaling
        //TODO: Move to `RandBLAS/util.hh`
        if(left && layout == blas::Layout::ColMajor) {
            for(int64_t row = 0; row < rows; row++) {
                if(diag[row] > 0)
                    continue;
                blas::scal(cols, diag[row], &A[row], rows);
            }
        }
        else if(left && layout == blas::Layout::RowMajor) {
            for(int64_t row = 0; row < rows; row++) {
                if(diag[row] > 0)
                    continue;
                blas::scal(cols, diag[row], &A[row * cols], 1);
            }
        }
        else if(!left && layout == blas::Layout::ColMajor) {
            for(int64_t col=0; col < cols; col++) {
                if(diag[col] > 0)
                    continue;
                blas::scal(rows, diag[col], &A[col * rows], 1);
            }
        }
        else {
            for(int64_t col=0; col < cols; col++) {
                if(diag[col] > 0)
                    continue;
                blas::scal(rows, diag[col], &A[col], cols);
            }
        }
    }

    template<typename T, SignedInteger sint_t = int64_t>
    void permute_rows_to_top(
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
                if (selected_rows[i] != top) {
                    // Use BLAS swap to swap the entire rows
                    // Swapping row 'selected' with row 'top'
                    blas::swap(cols, &A[top], rows, &A[selected_rows[i]], rows);
                }
                else
                    continue;
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                if (selected_rows[i] != top) {
                    blas::swap(cols, &A[cols * selected_rows[i]], 1, &A[cols * top], 1);
                }
                else
                    continue;
            }
        }
    }

    template<typename T, SignedInteger sint_t = int64_t>
    void permute_cols_to_left(
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
                else
                    continue;
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                if (selected_cols[i] != left) {
                    blas::swap(rows, &A[selected_cols[i]], cols, &A[left], cols);
                }
                else
                    continue;
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
        bool left, // Pre-multiplying?
        blas::Layout layout,
        int64_t num_rows,
        int64_t num_cols,
        int64_t log_n,
        T* A
        )
    {
        if(left && layout == blas::Layout::ColMajor)
            fht_left_col_major(A, log_n, num_rows, num_cols);
        else if(left && layout == blas::Layout::RowMajor)
            fht_left_row_major(A, log_n, num_rows, num_cols);
        else if(!left && layout == blas::Layout::ColMajor)
            fht_right_col_major(A, log_n, num_rows, num_cols);
        else
            fht_right_row_major(A, log_n, num_rows, num_cols);
    }
}


namespace RandBLAS::trig {
template <SignedInteger sint_t = int64_t>
struct HadamardMixingOp{
    sint_t* diag_scale;
    sint_t* selected_idxs;
    blas::Layout layout;
    int64_t m;
    int64_t n;
    int64_t d;
    bool left;
    bool filled = false; // will be updated by `miget`

    // Constructor
    HadamardMixingOp(bool left,
                     blas::Layout layout,
                     int64_t m,
                     int64_t n,
                     int64_t d
                    ) : left(left), layout(layout), m(m), n(n), d(d) {
        if(left)
            diag_scale = new sint_t[m];
        else
            diag_scale = new sint_t[n];
        selected_idxs = new sint_t[d];
    }

    // Destructor
    ~HadamardMixingOp() {
        free(this->diag_scale);
        free(this->selected_idxs);
    }

    private:
};

/*
 * A free-function that performs an inversion of a matrix transformed by `lmiget | rmiget`
 * the inversion is also performed in-place
*/
template <typename T, SignedInteger sint_t = int64_t>
void invert(
    HadamardMixingOp<sint_t> &hmo, // details about the transform
    T* SA // sketched matrix
) {
    // We have to make sure we apply the operation in the inverted order
    // with the operations appropriately inverted
    randblas_error_if_msg(!hmo.filled, "You are trying to call `invert` on an untransformed matrix,\
                                       please call `miget` on your matrix before calling `invert`");

    // Creating a vector out of `selected_idxs` to be able to conveniently reverse
    std::vector<sint_t> selected_idxs(hmo.selected_idxs, hmo.selected_idxs + hmo.d);
    // Reversing the indices for performing the inverse
    std::reverse(selected_idxs.begin(), selected_idxs.end());

    //Step 1: Permute the rows/cols
        // Perform the permutation (for the way we define permutations
        // invSelected_idxs = reverse(Selected_idxs))
    if(hmo.left)
        permute_rows_to_top(hmo.layout, hmo.m, hmo.n, selected_idxs.data(), hmo.d, SA);
    else
        permute_cols_to_left(hmo.layout, hmo.m, hmo.n, selected_idxs.data(), hmo.d, SA);

    //Step 2: Apply the Hadamard transform (invH = H.T = H)
    int ld = (hmo.left) ? hmo.m : hmo.n;

    T log_sz = std::log2(ld);
    T log_int_sz, log_final_sz;
    T log_frac_sz = std::modf(log_sz, &log_int_sz);

    if(log_frac_sz < 1e-3)
        log_final_sz = log_int_sz;
    else
        log_final_sz = log_int_sz + 1;

    fht_dispatch(hmo.left, hmo.layout, hmo.m, hmo.n, log_final_sz, SA);
    blas::scal(hmo.m * hmo.n, 1/std::pow(2, int(std::log2(ld))), SA, 1);

    //Step 3: Scale with `D` (invD = D for rademacher entries)
    apply_diagonal_rademacher(hmo.left, hmo.layout, hmo.m, hmo.n, SA, hmo.diag_scale);
}

/*
 * These functions apply an in-place, SRHT-like transform to the input matrix
 * i.e. A <- (\Pi H D)A OR A <- A(D H \Pi) (which is equivalent to A <- A(\Pi H D)^{-1})
 * layout: Layout of the input matrix (`ColMajor/RowMajor`)
 * A: (m x n), input dimensions of `A`
 * d: The number of rows/columns that will be permuted by the action of $\Pi$
 */
template <typename T, typename RNG = DefaultRNG, SignedInteger sint_t = int64_t>
inline RNGState<RNG> miget(
    HadamardMixingOp<sint_t> &hmo, // All information about `A` && the $\mathbb{\Pi\text{RHT}}$
    const RNGState<RNG> &random_state,
    T* A // The data-matrix
) {
    auto [ctr, key] = random_state;

    //Step 1: Scale with `D`
        //Populating `diag`
    RNGState<RNG> state_idxs = generate_rademacher_vector_r123(hmo.diag_scale, hmo.n, random_state);
    apply_diagonal_rademacher(hmo.left, hmo.layout, hmo.m, hmo.n, A, hmo.diag_scale);

    //Step 2: Apply the Hadamard transform
    int ld = (hmo.left) ? hmo.m : hmo.n;
    T log_sz = std::log2(ld);
    T log_int_sz, log_final_sz;
    T log_frac_sz = std::modf(log_sz, &log_int_sz);

    if(log_frac_sz < 1e-3)
        log_final_sz = log_int_sz;
    else
        log_final_sz = log_int_sz + 1;
    fht_dispatch(hmo.left, hmo.layout, hmo.m, hmo.n, log_final_sz, A);

    //Step 3: Permute the rows
        // Uniformly samples `d` entries from the index set [0, ..., m - 1]
    RNGState<RNG> next_state = repeated_fisher_yates<sint_t>(
        hmo.d,
        hmo.m,
        1,
        hmo.selected_idxs,
        state_idxs
    );

    if(hmo.left)
        permute_rows_to_top(hmo.layout, hmo.m, hmo.n, hmo.selected_idxs, hmo.d, A);
    else
        permute_cols_to_left(hmo.layout, hmo.m, hmo.n, hmo.selected_idxs, hmo.d, A);

    // `invert` can now be called with this instance of `HadamardMixingOp`
    hmo.filled = true;

    return next_state;
}
}
