// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>

#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


//TODO: Have the user choose between dense and sketch sketching operator (4 nnz per col)

void init_noisy_data(int64_t m, int64_t n, int64_t d, double* AB){
    double target_x[n*d];
    double eps[m*d];
    for (int i = 0; i < n; i++) {  
        target_x[i] = 1;   // Target X is the vector of 1's
    }

    RandBLAS::DenseDist Dist_A(m,n); 
    RandBLAS::DenseDist Dist_eps(m,d); 
    RandBLAS::RNGState state(0);
    RandBLAS::RNGState state1(1);

    RandBLAS::fill_dense(Dist_A, AB, state);  //Fill A to be a random gaussian
    RandBLAS::fill_dense(Dist_eps, eps, state1);  //Fill A to be a random gaussian

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, d, n, 1, AB, m, target_x, n, 0, &AB[m*n], m);

    for (int i = 0; i < m*d; i++){
        AB[m*n + i] += eps[i];   // Add Gaussian Noise to right hand side
    }
}

template <typename T>
void total_least_squares(int64_t m, int64_t n, T* AB, int64_t ldab, T* x,  T* work_s, T* work_vt) {
    // AB is m-by-(n+1) and stored in column-major format with leading dimension "ldab".
    // Its first n columns contain a matrix "A", and its last column contains a vector "B".
    //
    // This function overwrites x with the solution to
    //      (A+E)x = B+R
    // where (E, R) solve
    //      solve min{ ||[E, R]||_F : B+R in range(A+E) }.
    //
    // On exit, AB will have been overwritten by its matrix of left singular vectors, 
    // its singular values will be stored in work_s, and its (transposed) right singular
    // vectors will be stored in work_vt.
    lapack::gesdd(lapack::Job::OverwriteVec, m, n+1, AB, ldab, work_s, nullptr, 1, work_vt, n+1);
    T scale = work_vt[(n+1)*(n+1)-1];
    for (int i = 0; i < n; i++) {
        x[i] = -work_vt[n + i*(n+1)] / scale;
    }
    return;
}

/* Let A be a tall data matrix of dimensions m by n where m > n and b be a vector of dimension m.
 * In ordinary least squares it assumes that the error lies only in the right hand side vector b,
 * and it aims to find a vector x that minimizes ||A*x - b||_2.
 * On the other hand, total least squares assumes that the input data matrix A could also incur errors.
 * Total least squares aims to find a solution where the error is orthogonal to the regression model.
 */

// To call the executable run ./TLS_DenseSkOp <m> <n> where <m> <n> are the number of rows and columns
// of A respectively. We expect m > 2*n.
int main(int argc, char* argv[]){

    // Initialize dimensions
    int64_t m;           // Number of rows of A, B
    int64_t n;           // Number of columns of A
                         
    if (argc == 1) {
        m = 10000;
        n = 500;
    } else if (argc == 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        if (n > m) {
            std::cout << "Make sure number of rows are greater than number of cols" << '\n';
            exit(0);
        }
    } else {
        std::cout << "Invalid arguments" << '\n';
        exit(1);
    }

    // Define number or rows of the sketching operator
    int64_t sk_dim = 2*(n+1);

    // Initialize workspace
    double *AB = new double[m*(n+1)];
    double *SAB = new double[sk_dim*(n+1)];
    double *sketch_x = new double[n];
    double *svals = new double[n+1];
    double *VT = new double[(n+1)*(n+1)];

    // Initialize noisy gaussian data
    init_noisy_data(m, n, 1, AB);

    std::cout << "\nDimensions of the augmented matrix [A|B]   :  " << m << " by " << n+1 << '\n'; 
    std::cout << "Embedding dimension                        :  " << sk_dim << '\n'; 

    // Sample the sketching operator 
    auto time_constructsketch1 = high_resolution_clock::now();
    RandBLAS::SparseDist Dist(
        sk_dim,                  // Number of rows of the sketching operator 
        m,                       // Number of columns of the sketching operator
        8,                       // Number of non-zero entires per column,
        RandBLAS::Axis::Short    // A "SASO" (aka SJLT, aka OSNAP, aka generalized CountSketch)
    );
    uint32_t seed = 1997;
    RandBLAS::SparseSkOp<double> S(Dist, seed);  
    RandBLAS::fill_sparse(S);
    auto time_constructsketch2 = high_resolution_clock::now();
    double sampling_time = (double) duration_cast<milliseconds>(time_constructsketch2 - time_constructsketch1).count()/1000;
    std::cout << "\nTime to sample S                           :  " << sampling_time << " seconds" << '\n';

    // Sketch AB
    // SAB = 1.0 * S * AB +  0.0 * SAB
    auto time_sketch1 = high_resolution_clock::now();
    RandBLAS::sketch_general(
            blas::Layout::ColMajor,    // Matrix storage layout of AB and SAB
            blas::Op::NoTrans,         // NoTrans => \op(S) = S, Trans => \op(S) = S^T
            blas::Op::NoTrans,         // NoTrans => \op(AB) = AB, Trans => \op(AB) = AB^T
            sk_dim,                    // Number of rows of S and SAB
            n + 1,                     // Number of columns of AB and SAB
            m,                         // Number of rows of AB and columns of S
            1.0,                       // Scalar alpha - if alpha is zero AB is not accessed
            S,                         // A DenseSkOp or SparseSkOp
            AB,                        // Matrix to be sketched
            m,                         // Leading dimension of AB
            0.0,                       // Scalar beta - if beta is zero the initial value of SAB is not accessed
            SAB,                       // Sketched matrix SAB
            sk_dim                     // Leading dimension of SAB
    );
    auto time_sketch2 = high_resolution_clock::now();
    double sketching_time = (double) duration_cast<milliseconds>(time_sketch2 - time_sketch1).count()/1000;
    std::cout << "Time to compute SAB = S * AB               :  " << sketching_time << " seconds\n";

    auto time_sketched_TLS1 = high_resolution_clock::now();
    total_least_squares(sk_dim, n, SAB, sk_dim, sketch_x, svals, VT);
    auto time_sketched_TLS2 = high_resolution_clock::now();
    double sketched_solve_time = (double) duration_cast<milliseconds>(time_sketched_TLS2 - time_sketched_TLS1).count()/1000;
    std::cout << "Time to perform TLS on sketched data       :  " << sketched_solve_time << " seconds\n\n";

    double total_randomized_time = sampling_time + sketching_time + sketched_solve_time;
    std::cout << "Total time for the randomized TLS method   :  " << total_randomized_time << " seconds\n";

    double* true_x = new double[n];
    auto time_true_TLS1 = high_resolution_clock::now();
    total_least_squares(m, n, AB, m, true_x, svals, VT);
    auto time_true_TLS2 = high_resolution_clock::now();
    double true_solve_time = (double) duration_cast<milliseconds>(time_true_TLS2 - time_true_TLS1).count()/1000;
    std::cout << "Time for the classical TLS method          :  " << true_solve_time << " seconds" << "\n";
    
    std::cout << "Speedup of sketched vs classical method    :  " << true_solve_time / total_randomized_time << "\n\n";

    double* delta = new double[n];
    blas::copy(n, sketch_x, 1, delta, 1);
    blas::axpy(n, -1, true_x, 1, delta, 1);
    double distance = blas::nrm2(n, delta, 1);
    double scale = blas::nrm2(n, true_x, 1);
    std::cout << "||sketch_x - true_x|| / ||true_x||         :  " << distance/scale << "\n\n";

    delete[] delta;
    delete[] true_x;
    delete[] AB;
    delete[] SAB;
    delete[] sketch_x;
    delete[] svals;
    delete[] VT;
    return 0;
}
