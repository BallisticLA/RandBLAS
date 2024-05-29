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
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <unordered_map>
#include <iomanip> 
#include <limits> 
#include <numbers>
#include <chrono>
#include <fstream>

using RandBLAS::sparse_data::COOMatrix;
using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;


#define DOUT(_d) std::setprecision(std::numeric_limits<double>::max_digits10) << _d

std::string parse_args(int argc, char** argv) {
    if (argc > 1) {
        return std::string{argv[1]};
    } else {
        return "../sparse-data-matrices/bcsstk17/bcsstk17.mtx";
    }
}

template <typename T>
COOMatrix<T> from_matrix_market(std::string fn) {

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows,  cols, vals
    );

    COOMatrix<T> out(n_rows, n_cols);
    out.reserve(vals.size());
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

template <typename T>
int householder_orth(int64_t m, int64_t n, T* mat, T* work) {
    if(lapack::geqrf(m, n, mat, m, work))
        return 1;
    lapack::ungqr(m, n, n, mat, m, work);
    return 0;
}


#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << DOUT(dtime / 1e6) << std::endl; \
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

    std::string sample_log  = "sample                : ";
    std::string lspmmN_log  = "left_spmm (NoTrans)   : ";
    std::string orth_log    = "orth                  : ";
    std::string lspmmT_log  = "left_spmm (Trans)     : ";

    // Convert to CSC.
    // CSR would also be okay, but it seems that CSC is faster in this case.
    RandBLAS::sparse_data::CSCMatrix<T> A_compressed(A.n_rows, A.n_cols);
    TIMED_LINE(
    RandBLAS::sparse_data::conversions::coo_to_csc(A, A_compressed), "COO to CSC            : ")

    // Step 1: fill S := mat_work2 with the data needed to feed it into power iteration.
    if (p % 2 == 0) {
        RandBLAS::DenseDist D(n, k);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work2, state), sample_log)
    } else {
        RandBLAS::DenseDist D(m, k);
        TIMED_LINE(
        RandBLAS::fill_dense(D, mat_work1, state), sample_log)
        TIMED_LINE(
        left_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_compressed, 0, 0, mat_work1, m, 0.0, mat_work2, n), lspmmT_log)
        TIMED_LINE(
        householder_orth(n, k, mat_work2, work), orth_log)
        p_done += 1;
    }

    // Step 2: fill S := mat_work2 with data needed to feed it into the rangefinder.
    while (p - p_done > 0) {
        // Update S = orth(A' * orth(A * S))
        TIMED_LINE(
        left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_compressed, 0, 0, mat_work2, n, 0.0, mat_work1, m), lspmmN_log)
        TIMED_LINE(
        householder_orth(m, k, mat_work1, work), orth_log)
        TIMED_LINE(
        left_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A_compressed, 0, 0, mat_work1, m, 0.0, mat_work2, n), lspmmT_log)
        TIMED_LINE(
        householder_orth(n, k, mat_work2, work), orth_log)
        p_done += 2;
    }

    // Step 3: compute Q = orth(A * S) and B = Q'A.
    TIMED_LINE(
    left_spmm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A_compressed, 0, 0, mat_work2, n, 0.0, Q, m), lspmmN_log)
    TIMED_LINE(
    householder_orth(m, k, Q, work), orth_log)
    TIMED_LINE(
    right_spmm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, 1.0, Q, m, A_compressed, 0, 0, 0.0, B, k), "right_spmm            : ")
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
    bool allocate_more_work = extra_work_size < m*k;
    T* more_work = (allocate_more_work) ? new T[m*k] : (work + k*(k+1)); 
    lapack::lacpy(MatrixType::General, m, k, Q, ldq, more_work, m);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, more_work, m, U, k, 0.0, Q, ldq);

    if (allocate_more_work)
        delete [] more_work;

    return;
}

int main(int argc, char** argv) {

    auto fn = parse_args(argc, argv);
    auto mat_sparse = from_matrix_market<double>(fn);
    auto m = mat_sparse.n_rows;
    auto n = mat_sparse.n_cols;

    // Run the randomized algorithm!
    int64_t k = 64;
    double *U  = new double[m*k]{};
    double *VT = new double[k*n]{}; 
    double *qb_work = new double[std::max(m, n)];
    RandBLAS::RNGState<r123::Philox4x32> state(0);
    /*
    Effect of various parameters on performance:
        It's EXTREMELY important to use -O3 if you want reasonably
        fast sparse matrix conversion inside RandBLAS. We're talking
        a more-than-10x speedup.
        
    */
    auto start_timer = std_clock::now();
    qb_decompose_sparse_matrix(mat_sparse, k, U, VT, 2, state, qb_work, std::max(m,n));
    double *svals = new double[std::min(m,n)];
    double *conversion_work = new double[m*k + k*k];
    qb_to_svd(m, n, k, U, svals, m, VT, k, conversion_work, m*k + k*k);
    auto stop_timer = std_clock::now();
    double runtime = (double) std::chrono::duration_cast<std::chrono::microseconds>(stop_timer - start_timer).count();
    runtime = runtime / 1e6;

    std::cout << "n_rows  : " << mat_sparse.n_rows << std::endl;
    std::cout << "n_cols  : " << mat_sparse.n_cols << std::endl;
    double density = ((double) mat_sparse.nnz) / ((double) (mat_sparse.n_rows * mat_sparse.n_cols));
    std::cout << "density : " << DOUT(density) << std::endl;
    std::cout << "runtime of low-rank approximation : " << DOUT(runtime) << std::endl;

    delete [] qb_work;
    delete [] conversion_work;
    delete [] svals;
    return 0;
}