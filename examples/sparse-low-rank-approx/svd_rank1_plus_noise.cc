#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <unordered_map>

using RandBLAS::sparse_data::COOMatrix;

auto parse_dimension_args(int argc, char** argv) {
    int64_t m;
    int64_t n;
    int64_t vec_nnz;
                         
    if (argc == 1) {
        m = 10000;
        n = 500;
        vec_nnz = 4;
    } else if (argc == 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        vec_nnz = atoi(argv[3]);
    } else {
        std::cout << "Invalid parameters; must be called with no parameters, or with three positive integers." << '\n';
        exit(1);
    }
    return std::make_tuple(m, n, vec_nnz);
}

template <typename T, typename RNG = r123::Philox4x32>
void iid_sparsify_random_dense(
    int64_t n_rows, int64_t n_cols, int64_t stride_row, int64_t stride_col, T* mat, T prob_of_zero, RandBLAS::RNGState<RNG> state
) { 
    auto spar = new T[n_rows * n_cols];
    auto dist = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
    auto [unused, next_state] = RandBLAS::fill_dense(dist, spar, state);

    auto temp = new T[n_rows * n_cols];
    auto D_mat = RandBLAS::DenseDist(n_rows, n_cols, RandBLAS::DenseDistName::Uniform);
    RandBLAS::fill_dense(D_mat, temp, next_state);

    #define SPAR(_i, _j) spar[(_i) + (_j) * n_rows]
    #define TEMP(_i, _j) temp[(_i) + (_j) * n_rows]
    #define MAT(_i, _j)  mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T v = (SPAR(i, j) + 1.0) / 2.0;
            if (v < prob_of_zero) {
                MAT(i, j) = 0.0;
            } else {
                MAT(i, j) = TEMP(i, j);
            }
        }
    }

    delete [] spar;
    delete [] temp;
}

template <typename SpMat>
SpMat sum_of_coo_matrices(SpMat &A, SpMat &B) {
    randblas_require(A.n_rows == B.n_rows);
    randblas_require(A.n_cols == B.n_cols);

    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;
    constexpr bool valid_type = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    randblas_require(valid_type);

    using Tuple = std::pair<int64_t, int64_t>;
    struct TupleHasher {
        size_t operator()(const Tuple &coordinate_pair) const {
            // an implementation suggested by my robot friend.
            size_t hash1 = std::hash<int64_t>{}(coordinate_pair.first);
            size_t hash2 = std::hash<int64_t>{}(coordinate_pair.second);
            size_t hash3 = hash1;
            hash3 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);
            return hash3;
        }
    };
    std::unordered_map<Tuple, T, TupleHasher> c_dict{};

    for (int ell = 0; ell < A.nnz; ++ell) {
        Tuple curr_idx(A.rows[ell], A.cols[ell]);
        c_dict[curr_idx] = A.vals[ell];
    }
    for (int ell = 0; ell < B.nnz; ++ell) {
        Tuple curr_idx(B.rows[ell], B.cols[ell]);
        c_dict[curr_idx] = B.vals[ell] + c_dict[curr_idx];
    }

    SpMat C(A.n_rows, A.n_cols);
    C.reserve(c_dict.size());
    int64_t ell = 0;
    for (auto iter : c_dict) {
        Tuple t = iter.first;
        auto [i, j] = t;
        C.rows[ell] = i;
        C.cols[ell] = j;
        C.vals[ell] = iter.second;
        ++ell;
    }
    return C;
}


template <typename SpMat>
void make_signal_matrix(double signal_scale, int64_t m, int64_t n, int64_t vec_nnz, double* signal_dense, SpMat &signal_sparse) {
    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;
    constexpr bool valid_type = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    randblas_require(valid_type);
    signal_sparse.reserve(vec_nnz * vec_nnz);

    // populate signal_dense and signal_sparse.
    RandBLAS::RNGState u_state(0);
    double *work_vals  = new double[2*vec_nnz]{};
    int64_t *work_idxs = new int64_t[2*vec_nnz]{};
    int64_t *trash     = new int64_t[vec_nnz]{};

    auto v_state    = RandBLAS::repeated_fisher_yates(u_state, vec_nnz, m, 1, work_idxs, trash, work_vals);
    auto next_state = RandBLAS::repeated_fisher_yates(v_state, vec_nnz, n, 1, work_idxs+vec_nnz, trash, work_vals+vec_nnz);
    double *u = new double[m]{};
    double *v = new double[n]{};
    for (int j = 0; j < vec_nnz; ++j) {
        for (int i = 0; i < vec_nnz; ++i) {
            int temp = i + j*vec_nnz;
            signal_sparse.rows[temp] = work_idxs[i];
            signal_sparse.cols[temp] = work_idxs[j+vec_nnz];
            signal_sparse.vals[temp] = work_vals[i] * work_vals[j+vec_nnz];
        }
        u[work_idxs[j]] = work_vals[j];
        v[work_idxs[j + vec_nnz]] = work_vals[j + vec_nnz];
    }
    blas::ger(blas::Layout::ColMajor, m, n, signal_scale, u, 1, v, 1, signal_dense, m);

    delete [] work_vals;
    delete [] work_idxs;
    delete [] trash;
    delete [] u;
    delete [] v;
    return;
}


template <typename SpMat>
void make_noise_matrix(double noise_scale, int64_t m, int64_t n, double prob_of_nonzero, double* noise_dense, SpMat &noise_sparse) {
    // populate noise_dense and noise_sparse.
    //
    //  NOTE: it would be more efficient to sample vec_nnz*vec_nnz elements without replacement from the index set
    //  from 0 to m*n-1, then de-vectorize those indices (in either row-major or col-major interpretation) and
    //  only sample the values of the nonzeros for these pre-determined structural nonzeros. The current implementation
    //  has to generate to dense m-by-n matrices whose entries are iid uniform [-1, 1].
    //
    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;
    constexpr bool valid_type = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    randblas_require(valid_type);

    RandBLAS::RNGState noise_state(1);
    double prob_of_zero = 1 - prob_of_nonzero;
    iid_sparsify_random_dense(m, n, 1, m, noise_dense, prob_of_zero, noise_state);
    blas::scal(m * n, noise_scale, noise_dense, 1);
    RandBLAS::sparse_data::coo::dense_to_coo(blas::Layout::ColMajor, noise_dense, 0.0, noise_sparse);
    return;
}

/*Utilities
3. Basic QB-based SVD with power iteration and HouseholderQR stabilization.
*/

int main(int argc, char** argv) {
    auto [m, n, vec_nnz] = parse_dimension_args(argc, argv);
    // First we set up problem data: a sparse matrix of low numerical rank
    // given by a sum of a sparse "signal" matrix of rank 1 and a sparse 
    // "noise" matrix that has very small norm.
    double signal_scale = 1e+2;
    double noise_scale  = 1e-6;
    double prob_nonzero = 1e-4;
    RandBLAS::sparse_data::COOMatrix<double> signal_sparse(m, n);
    RandBLAS::sparse_data::COOMatrix<double> noise_sparse(m, n);
    auto mn = m * n;
    double *signal_dense = new double[mn]{};
    double *noise_dense  = new double[mn];
    
    make_signal_matrix(signal_scale, m, n, vec_nnz, signal_dense, signal_sparse);
    make_noise_matrix(noise_scale, m, n, prob_nonzero, noise_dense, noise_sparse);

    // Add the two matrices together. 
    auto mat_sparse = sum_of_coo_matrices(noise_sparse, signal_sparse);
    std::cout << signal_sparse.nnz << std::endl;
    std::cout << noise_sparse.nnz << std::endl;
    std::cout << mat_sparse.nnz << std::endl;
    double *mat_dense = new double[mn]{};
    blas::copy(mn, noise_dense, 1, mat_dense, 1);
    blas::axpy(mn, 1.0, signal_dense, 1, mat_dense, 1);

    delete [] signal_dense;
    delete [] noise_dense;
    delete [] mat_dense;
    return 0;
}