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
#include <iomanip> 
#include <limits> 
#include <numbers>
#include <chrono>

using RandBLAS::sparse_data::COOMatrix;

#define DOUT(_d) std::setprecision(std::numeric_limits<double>::max_digits10) << _d

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
void make_signal_matrix(double signal_scale, double* u, int64_t m, double* v, int64_t n, int64_t vec_nnz, double* signal_dense, SpMat &signal_sparse) {
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

    double uv_scale = 1.0 / std::sqrt((double) vec_nnz);


    auto v_state    = RandBLAS::repeated_fisher_yates(u_state, vec_nnz, m, 1, work_idxs, trash, work_vals);
    auto next_state = RandBLAS::repeated_fisher_yates(v_state, vec_nnz, n, 1, work_idxs+vec_nnz, trash, work_vals+vec_nnz);
    for (int j = 0; j < vec_nnz; ++j) {
        for (int i = 0; i < vec_nnz; ++i) {
            int temp = i + j*vec_nnz;
            signal_sparse.rows[temp] = work_idxs[i];
            signal_sparse.cols[temp] = work_idxs[j+vec_nnz];
            signal_sparse.vals[temp] = work_vals[i] * work_vals[j+vec_nnz];
        }
        u[work_idxs[j]] = uv_scale * work_vals[j];
        v[work_idxs[j + vec_nnz]] = uv_scale * work_vals[j + vec_nnz];
    }
    blas::ger(blas::Layout::ColMajor, m, n, signal_scale, u, 1, v, 1, signal_dense, m);

    delete [] work_vals;
    delete [] work_idxs;
    delete [] trash;
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

template <typename T>
int householder_orth(int64_t m, int64_t n, T* mat, T* work) {
    if(lapack::geqrf(m, n, mat, m, work))
        return 1;
    lapack::ungqr(m, n, n, mat, m, work);
    return 0;
}

template <typename SpMat, typename T, typename STATE>
void qb_decompose_sparse_matrix(SpMat &A, int64_t k, T* Q, T* B, int64_t p, STATE state, T* work, int64_t lwork) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    using RandBLAS::sparse_data::left_spmm;
    using RandBLAS::sparse_data::right_spmm;
    using blas::Op;
    using blas::Layout;

    // We use Q and B as workspace and to store the final result.
    // To distinguish the semantic use of workspace from the final result,
    // we define some alias pointers to Q's and B's memory.
    randblas_require(lwork >= std::max(m, n));
    T* mat_work1 = Q;
    T* mat_work2 = B;
    int64_t p_done = 0;

    // Step 1: fill S := mat_work2 with the data needed to feed it into power iteration.
    if (p % 2 == 0) {
        RandBLAS::DenseDist D(n, k);
        RandBLAS::fill_dense(D, mat_work2, state);
    } else {
        RandBLAS::DenseDist D(m, k);
        RandBLAS::fill_dense(D, mat_work1, state);
        left_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, 0, 0, mat_work1, m, 0.0, mat_work2, n);
        p_done += 1;
        householder_orth(n, k, mat_work2, work);
    }

    // Step 2: fill S := mat_work2 with data needed to feed it into the rangefinder.
    while (p - p_done > 0) {
        // Update S = orth(A' * orth(A * S))
        left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, 0, 0, mat_work2, n, 0.0, mat_work1, m);
        householder_orth(m, k, mat_work1, work);
        left_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, 0, 0, mat_work1, m, 0.0, mat_work2, n);
        householder_orth(n, k, mat_work2, work);
        p_done += 2;
    }

    // Step 3: compute Q = orth(A * S) and B = Q'A.
    left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, 0, 0, mat_work2, n, 0.0, Q, m);
    householder_orth(m, k, Q, work);
    right_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, 1.0, Q, m, A, 0, 0, 0.0, B, k);
    return;
}

template <typename T>
void qb_to_svd(int64_t m, int64_t n, int64_t k, T* Q, T* svals, int64_t ldq, T* B, int64_t ldb, T* work, int64_t lwork) {
    // Input: (Q, B) defining a matrix A = Q*B, where
    //      Q is m-by-k and column orthonormal
    // and
    //      B is k-by-n and otherwise unstructured.
    //
    // Output:
    //      Q holds the top-k left singular vectors of A.
    //      B holds a matrix that can be described in two equivalent ways:
    //          1. a column-major representation of the top-k transposed right singular vectors of A.
    //          2. a row-major representation of the top-k right singular vectors of A.
    //      svals holds the top-k singular values of A.
    //
    using blas::Op;
    using blas::Layout;
    using lapack::Job;
    using lapack::MatrixType;

    // Compute the SVD of B: B = U diag(svals) VT, where B is overwritten by VT.
    int64_t extra_work_size = lwork - k*k;
    randblas_require(extra_work_size >= 0);
    T* U = work; // <-- just a semantic alias for the start of work.
    lapack::gesdd(Job::OverwriteVec, k, n, B, ldb, svals, U, k, nullptr, k);

    // update Q = Q U.
    T* more_work = work + k*(k+1);
    bool allocate_more_work = extra_work_size < m*k;
    if (allocate_more_work)
        more_work = new T[m*k];
    lapack::lacpy(MatrixType::General, m, k, Q, ldq, more_work, m);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, more_work, m, U, k, 0.0, Q, ldq);

    if (allocate_more_work)
        delete [] more_work;

    return;
}

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
    double *u_top = new double[m]{};
    double *v_top = new double[n]{};
    
    make_signal_matrix(signal_scale, u_top, m, v_top, n, vec_nnz, signal_dense, signal_sparse);
    make_noise_matrix(noise_scale, m, n, prob_nonzero, noise_dense, noise_sparse);

    // Add the two matrices together. 
    auto mat_sparse = sum_of_coo_matrices(noise_sparse, signal_sparse);
    std::cout << signal_sparse.nnz << std::endl;
    std::cout << noise_sparse.nnz << std::endl;
    std::cout << mat_sparse.nnz << std::endl;
    double *mat_dense = new double[mn]{};
    blas::copy(mn, noise_dense, 1, mat_dense, 1);
    blas::axpy(mn, 1.0, signal_dense, 1, mat_dense, 1);

    // Run the randomized algorithm!
    int64_t k = std::max((int64_t) 3, vec_nnz); // the matrix is really rank-1 plus noise
    auto start_timer = std::chrono::high_resolution_clock::now();
    double *U  = new double[m*k]{};
    double *VT = new double[k*n]{}; 
    double *qb_work = new double[std::max(m, n)];
    RandBLAS::RNGState state(0);
    qb_decompose_sparse_matrix(mat_sparse, k, U, VT, 2, state, qb_work, std::max(m,n));
    double *svals = new double[std::min(m,n)];
    double *conversion_work = new double[m*k + k*k];
    qb_to_svd(m, n, k, U, svals, m, VT, k, conversion_work, m*k + k*k);
    auto stop_timer = std::chrono::high_resolution_clock::now();
    double runtime = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop_timer - start_timer).count();
    runtime = runtime / 1e6;

    // compute angles between (u_top, v_top) and the top singular vectors 
    double cos_utopu = blas::dot(m, u_top, 1, U, 1);
    double cos_vtopv = blas::dot(n, v_top, 1, VT, k);
    double theta_utopu = std::acos(cos_utopu) / (std::numbers::pi);
    double theta_vtopv = std::acos(cos_vtopv) / (std::numbers::pi);

    std::cout << "runtime of low-rank approximation : " << DOUT(runtime) << std::endl;
    std::cout << "Relative angle between top left singular vectors  : " << DOUT(theta_utopu)  << std::endl;
    std::cout << "Relative angle between top right singular vectors : " << DOUT(theta_vtopv) << std::endl; 

    delete [] u_top;
    delete [] v_top;
    delete [] qb_work;
    delete [] conversion_work;
    delete [] svals;
    delete [] signal_dense;
    delete [] noise_dense;
    delete [] mat_dense;
    return 0;
}
