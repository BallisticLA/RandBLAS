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
#include <fast_matrix_market/app/Eigen.hpp>
#include <unordered_map>
#include <iomanip> 
#include <limits> 
#include <numbers>
#include <chrono>
#include <fstream>
#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>


using std_clock = std::chrono::high_resolution_clock;
using timepoint_t = std::chrono::time_point<std_clock>;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using Eigen::SparseMatrix;

#define DOUT(_d) std::setprecision(8) << _d

std::string parse_args(int argc, char** argv) {
    if (argc > 1) {
        return std::string{argv[1]};
    } else {
        return "../sparse-low-rank-approx/data-matrices/bcsstk17/bcsstk17.mtx";
    }
}


#define TIMED_LINE(_op, _name) { \
        auto _tp0 = std_clock::now(); \
        _op; \
        auto _tp1 = std_clock::now(); \
        double dtime = (double) duration_cast<microseconds>(_tp1 - _tp0).count(); \
        std::cout << _name << DOUT(dtime / 1e6) << std::endl; \
        }



int main(int argc, char** argv) {
    auto fn = parse_args(argc, argv);

    std::ifstream f(fn);

    using SpMat = typename Eigen::SparseMatrix<double>;
    using SparseQR = typename Eigen::SparseQR<SpMat, Eigen::AMDOrdering<SpMat::StorageIndex>>;

    SpMat mat_eigsp;
    fast_matrix_market::read_matrix_market_eigen(f, mat_eigsp);

    auto m = mat_eigsp.rows();
    auto n = mat_eigsp.cols();
    std::cout << "\nn_rows  : " << m << std::endl;
    std::cout << "n_cols  : " << n << std::endl;
    double density = ((double) mat_eigsp.nonZeros()) / ((double) (m * n));
    std::cout << "density : " << DOUT(density) << std::endl << std::endl;

    TIMED_LINE(
    mat_eigsp.makeCompressed(), "\nCompress mat_eigsp   : ")

    auto start_timer = std_clock::now();
    SparseQR sqrcp(mat_eigsp);
    auto stop_timer = std_clock::now();
    double runtime = (double) duration_cast<microseconds>(stop_timer - start_timer).count();
    std::cout << "\nMake SparseQR object : " << DOUT(runtime / 1e6) << std::endl;

    TIMED_LINE(
        sqrcp.compute(mat_eigsp), "\nsqrcp.compute() time : "
    )

    return 0;
}