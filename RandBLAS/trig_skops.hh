#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/sparse_skops.hh"
#include "util.hh"

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
    template<SignedInteger sint_t = int64_t, typename state_t = RNGState<DefaultRNG>>
    state_t generate_rademacher_vector_r123(sint_t* buff, int64_t n, state_t &seed_state) {
        DefaultRNG rng;
        auto [ctr, key] = seed_state;

        for (int64_t i = 0; i < n; ++i) {
            typename DefaultRNG::ctr_type r = rng(ctr, key);

            float rand_value = r123::u01fixedpt<float>(r.v[0]);

            buff[i] = rand_value < 0.5 ? -1 : 1;

            ctr.incr();
        }

        // Return the updated RNGState (with the incremented counter)
        return state_t {ctr, key};
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
                } // else, continue;
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                if (selected_rows[i] != top) {
                    blas::swap(cols, &A[cols * selected_rows[i]], 1, &A[cols * top], 1);
                } // else, continue;
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
            }
        }
        else {
            // For `RowMajor` ordering
            for (int64_t i=0; i < d; i++) {
                if (selected_cols[i] != left) {
                    blas::swap(rows, &A[selected_cols[i]], cols, &A[left], cols);
                }
            }
        }
    }

    template <typename T>
    void fht_left_col_major(T *buf, T* workspace_buf, int64_t workspace_ld, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each column independently
        for (int64_t col = 0; col < num_cols; ++col) {
            // Pointer to the beginning of the current column in the Column-Major order
            T* col_buf = buf + col * num_rows;
            T* col_buf_workspace = workspace_buf + col * workspace_ld;

            // Apply the original FHT on this column
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        bool b1 = j + k < num_rows;
                        bool b2 = j + k + s1 < num_rows;
                        T u = b1 ? col_buf[j + k] : col_buf_workspace[j + k - num_rows];
                        T v = b2 ? col_buf[j + k + s1] : col_buf_workspace[j + k + s1 - num_rows];
                        if(b1 && b2) {
                            col_buf[j + k] = u + v;
                            col_buf[j + k + s1] = u - v;
                        }
                        else if(!b2 && b1) {
                            col_buf[j + k] = u + v;
                            col_buf_workspace[j + k + s1 - num_rows] = u - v;
                        }
                        else if(!b2 && !b1) {
                            col_buf_workspace[j + k - num_rows] = u + v;
                            col_buf_workspace[j + k + s1 - num_rows] = u - v;
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_left_row_major(T *buf, T* workspace_buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each column independently
        for (int64_t col = 0; col < num_cols; ++col) {
            // Apply the original FHT on this column
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        bool b1 = (j + k) * num_cols + col < num_rows * num_cols;
                        bool b2 = (j + k + s1) * num_cols + col < num_rows * num_cols;
                        T u = b1 ? buf[(j + k) * num_cols + col] : workspace_buf[(j + k) * num_cols + col - num_rows * num_cols];
                        T v = b2 ? buf[(j + k + s1) * num_cols + col] : workspace_buf[(j + k + s1) * num_cols + col - num_rows * num_cols];
                        if(b1 && b2) {
                            buf[(j + k) * num_cols + col] = u + v;
                            buf[(j + k + s1) * num_cols + col] = u - v;
                        }
                        else if(!b2 && b1) {
                            buf[(j + k) * num_cols + col] = u + v;
                            workspace_buf[(j + k + s1) * num_cols + col - num_rows * num_cols] = u - v;
                        }
                        else if(!b2 && !b1) {
                            workspace_buf[(j + k) * num_cols + col - num_rows * num_cols] = u + v;
                            workspace_buf[(j + k + s1) * num_cols + col - num_rows * num_cols] = u - v;
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_right_row_major(T *buf, T* workspace_buf, int64_t workspace_ld, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each row independently
        for (int64_t row = 0; row < num_rows; ++row) {
            // Pointer to the beginning of the current row in RowMajor order
            // for both the main buffer and the workspace
            T * row_buf = buf + row * num_cols;
            T* row_buf_workspace = workspace_buf + row * workspace_ld;

            // Apply the original FHT on this row
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        bool b1 = j + k < num_cols;
                        bool b2 = j + k + s1 < num_cols;
                        T u = b1 ? row_buf[j + k] : row_buf_workspace[j + k - num_cols];
                        T v = b2 ? row_buf[j + k + s1] : row_buf_workspace[j + k + s1 - num_cols];
                        if(b1 && b2) {
                            row_buf[j + k] = u + v;
                            row_buf[j + k + s1] = u - v;
                        }
                        else if(!b2 && b1) {
                            row_buf[j + k] = u + v;
                            row_buf_workspace[j + k + s1 - num_cols] = u - v;
                        }
                        else if(!b2 && !b1) {
                            row_buf_workspace[j + k - num_cols] = u + v;
                            row_buf_workspace[j + k + s1 - num_cols] = u - v;
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void fht_right_col_major(T *buf, T* workspace_buf, int64_t log_n, int64_t num_rows, int64_t num_cols) {
        int64_t n = 1 << log_n;

        // Apply FHT to each row independently
        for (int64_t row= 0; row < num_rows; ++row) {
            // Apply the original FHT on this row
            for (int64_t i = 0; i < log_n; ++i) {
                int64_t s1 = 1 << i;
                int64_t s2 = s1 << 1;
                for (int64_t j = 0; j < n; j += s2) {
                    for (int64_t k = 0; k < s1; ++k) {
                        bool b1 = (j + k) * num_rows + row < num_cols * num_rows;
                        bool b2 = (j + k + s1) * num_rows + row < num_cols * num_rows;
                        T u = b1 ? buf[(j + k) * num_rows + row] : workspace_buf[(j + k) * num_rows + row - num_rows * num_cols];
                        T v = b2 ? buf[(j + k + s1) * num_rows + row] : workspace_buf[(j + k + s1) * num_rows + row - num_rows * num_cols];
                        if(b1 && b2) {
                            buf[(j + k) * num_rows + row] = u + v;
                            buf[(j + k + s1) * num_rows + row] = u - v;
                        }
                        else if(!b2 && b1) {
                            buf[(j + k) * num_rows + row] = u + v;
                            workspace_buf[(j + k + s1) * num_rows + row - num_rows * num_cols] = u - v;
                        }
                        else if(!b2 && !b1) {
                            workspace_buf[(j + k) * num_rows + row - num_rows * num_cols] = u + v;
                            workspace_buf[(j + k + s1) * num_rows + row - num_rows * num_cols] = u - v;
                        }
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
        T* A,
        int64_t workspace_ld=0, // leading dimension of the workspace buffer
        T* workspace_buf=nullptr
        )
    {
        if(left && layout == blas::Layout::ColMajor)
            fht_left_col_major(A, workspace_buf, workspace_ld, log_n, num_rows, num_cols);
        else if(left && layout == blas::Layout::RowMajor)
            fht_left_row_major(A, workspace_buf, log_n, num_rows, num_cols);
        else if(!left && layout == blas::Layout::ColMajor)
            fht_right_col_major(A, workspace_buf, log_n, num_rows, num_cols);
        else
            fht_right_row_major(A, workspace_buf, workspace_ld, log_n, num_rows, num_cols);
    }
}


namespace RandBLAS::trig {

/*
 * A class for organizing all data required by the \PiRHT
 * while applying the transform
 */
template <typename T>
struct HadamardMatrixCouple {
    int64_t n_rows;
    int64_t n_cols;
    blas::Layout layout;
    T* A; // Data-matrix
    int64_t lda;
    T* workspace; // Extra space of sz: (2 ** n_closest - lda) * cols/rows
                  // Memory **has** to be managed by the user
                  //NOTE: I am also going to assume that the user is kind enough to
                  // populate this fully with zeros
    //NOTE: We could (should?) explore the possibility of managing the buffer by ourselves
    int64_t workspace_ld;
    bool left;

    // Trivial constructor
    HadamardMatrixCouple(
        bool left,
        blas::Layout layout,
        int64_t rows,
        int64_t cols,
        T* A,
        int64_t lda,
        T* workspace
    ) : left(left), layout(layout), n_rows(rows), n_cols(cols), A(A), lda(lda), workspace(workspace) {};
};

/*
 * A `pure` description of the \PiRHT as a «data-oblvious» transform
 */
template <typename T, SignedInteger sint_t = int64_t, typename state_t = RNGState<DefaultRNG>>
struct HadamardMixingOp{
    sint_t* diag_scale = nullptr;
    sint_t* selected_idxs = nullptr;
    const state_t seed_state;
    int64_t dim; // The dimension to be sketched
    int64_t d; // The number of cols/rows to be permuted at the end of the operation
    bool own_memory = false; // Do we manage our own memory?

    // Constructor, we manage the memory
    HadamardMixingOp(int64_t dim,
                     int64_t d,
                     state_t seed_state
                    ) : dim(dim), d(d), seed_state(seed_state), own_memory(true) {};


    // Constructor, user-managed memory
    HadamardMixingOp(int64_t dim,
                     int64_t d,
                     sint_t* diag_scale,
                     sint_t* selected_idxs
                    ) : dim(dim), d(d), diag_scale(diag_scale), selected_idxs(selected_idxs), seed_state(seed_state), own_memory(false) {};

    bool filled() {
        if(this -> diag_scale == nullptr \
        && this -> selected_idxs == nullptr)
            return false;
        else
            return true;
    }

    // Destructor
    ~HadamardMixingOp() {
        if (own_memory) {
            if (diag_scale != nullptr) delete [] diag_scale;
            if (selected_idxs != nullptr) delete [] selected_idxs;
        }
    }

    private:
};

/*
* Populates a `HadamardMixingOp`: `selected_idxs` and `diag_scale`
*/
template <typename T, typename state_t = RNGState<DefaultRNG>, SignedInteger sint_t = int64_t>
state_t fill_hadamard(
                    HadamardMixingOp<T, sint_t> &hmo
                  ) {
    if(!hmo.filled()) {
        hmo.diag_scale = new sint_t[hmo.dim];

        hmo.selected_idxs = new sint_t[hmo.d];

        auto [ctr, key] = hmo.seed_state;

        // Populating `diag`
        auto next_state = generate_rademacher_vector_r123(hmo.diag_scale, hmo.dim, hmo.seed_state);

        // Populating `selected_idxs`
        next_state = repeated_fisher_yates<sint_t>(
            hmo.d,
            hmo.dim,
            1,
            hmo.selected_idxs,
            next_state
        );

        return next_state;
    }
}

/*
 * A free-function that performs (\PiRHT)^{-1} matrix transformed by `miget`
 * the inversion is also performed in-place
*/
template <typename T, SignedInteger sint_t = int64_t>
void invert_hadamard(
    HadamardMixingOp<T, sint_t> &hmo, // details about the transform
    HadamardMatrixCouple<T> &hmc
) {
    // We have to make sure we apply the operation in the inverted order
    // with the operations appropriately inverted
    randblas_error_if_msg(!hmo.filled(), "You are trying to call `invert` on an uninitialized transform,\
                                        please call `miget` (or `fill_hadamard`) on your matrix before calling `invert_hadamard`");

    // Creating a vector out of `selected_idxs` to be able to conveniently reverse
    std::vector<sint_t> selected_idxs(hmo.selected_idxs, hmo.selected_idxs + hmo.d);
    // Reversing the indices for performing the inverse
    std::reverse(selected_idxs.begin(), selected_idxs.end());

    //Step 1: Permute the rows/cols
    if(hmc.left)
        permute_rows_to_top(hmc.layout, hmc.n_rows, hmc.n_cols, selected_idxs.data(), hmo.d, hmc.A);
    else
        permute_cols_to_left(hmc.layout, hmc.n_rows, hmc.n_cols, selected_idxs.data(), hmo.d, hmc.A);

    //Step 2: Apply the Hadamard transform (invH = H.T = H)
    // This has to be a `double` because `blas::scal` really doesn't like being passed integer types
    double padded_ld = std::pow(2, int(std::log2(hmo.dim)) + 1);
    int64_t workspace_ld = (padded_ld - hmo.dim);

    T log_sz = std::log2(hmo.dim);
    T log_int_sz, log_final_sz;
    T log_frac_sz = std::modf(log_sz, &log_int_sz);

    if(log_frac_sz < 1e-3)
        log_final_sz = log_int_sz;
    else
        log_final_sz = log_int_sz + 1;

    fht_dispatch(hmc.left, hmc.layout, hmc.n_rows, hmc.n_cols, log_final_sz, hmc.A, workspace_ld, hmc.workspace);

    // Scaling appropriately
    blas::scal(hmc.n_rows * hmc.n_cols, 1 / padded_ld, hmc.A, 1);

    //Step 3: Scale with `D` (invD = D for rademacher entries)
    apply_diagonal_rademacher(hmc.left, hmc.layout, hmc.n_rows, hmc.n_cols, hmc.A, hmo.diag_scale);
}

/*
 * Helper function to compute the size of the scratch workspace required
 * Note, that this follows the same convention as `cuSolver` where the library provides
 * helper functions for computing the size of scratch workspace required for a driver, often with
 * a very similar signature to the driver itself
*/
template <typename T, typename state_t = RNGState<DefaultRNG>, SignedInteger sint_t = int64_t>
inline int64_t miget_workspace_sz(
    HadamardMixingOp<T, sint_t, state_t> &hmo,
    HadamardMatrixCouple<T> &hmc) {
    int64_t stride = hmc.left ? hmc.n_cols : hmc.n_rows;

    // Grabs the power of 2 that is just bigger than the leading dimension
    int64_t padded_ld = std::pow(2, int(std::log2(hmo.dim)) + 1);

    return (padded_ld - hmo.dim) * stride;
}

/*
 * Applies an in-place, SRHT-like transform to the input matrix
 * i.e. A <- (\Pi H D)A OR A <- A(D H \Pi) (which is equivalent to A <- A(\Pi H D)^{-1})
 * `HadamardMixingOp hmo`:  Data-oblivious description of the \PiRHT at hand
 * `HadamardMatrixCouple hmc`: Description of the data to be sketched
 */
template <typename T, typename state_t = RNGState<DefaultRNG>, SignedInteger sint_t = int64_t>
inline void miget(
    HadamardMixingOp<T, sint_t> &hmo, // All information about `A` && the $\mathbb{\Pi\text{RHT}}$
    HadamardMatrixCouple<T> &hmc
) {

    if(!hmo.filled()) {
        fill_hadamard<T, state_t>(hmo);
    }

    //Step 1: Scale with `D`
    apply_diagonal_rademacher(hmc.left, hmc.layout, hmc.n_rows, hmc.n_cols, hmc.A, hmo.diag_scale);

    //Step 2: Apply the Hadamard transform
    int ld = (hmc.left) ? hmc.n_rows : hmc.n_cols;
    T log_sz = std::log2(ld);
    T log_int_sz, log_final_sz;
    T log_frac_sz = std::modf(log_sz, &log_int_sz);

    int64_t padded_ld = std::pow(2, int(std::log2(hmo.dim)) + 1);
    int64_t workspace_ld = (padded_ld - hmo.dim);

    if(log_frac_sz < 1e-3)
        log_final_sz = log_int_sz;
    else
        log_final_sz = log_int_sz + 1;

    fht_dispatch(hmc.left, hmc.layout, hmc.n_rows, hmc.n_cols, log_final_sz, hmc.A, workspace_ld, hmc.workspace);

    // Step 3: Permute the rows/cols
    if(hmc.left)
        permute_rows_to_top(hmc.layout, hmc.n_rows, hmc.n_cols, hmo.selected_idxs, hmo.d, hmc.A);
    else
        permute_cols_to_left(hmc.layout, hmc.n_rows, hmc.n_cols, hmo.selected_idxs, hmo.d, hmc.A);
    }
}
