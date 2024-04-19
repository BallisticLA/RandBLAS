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

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


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

    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, d, n, 1.0, AB, m, target_x, n, 0.0, &AB[m*n], m);

    for (int i = 0; i < m*d; i++){
        AB[m*n + i] += eps[i];   // Add Gaussian Noise to right hand side
    }
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

    int64_t sk_dim = 2*(n+1);

    // Initialize workspace
    double *AB = new double[m*(n + 1)]; // Store [A B] in column major format
    double *SAB = new double[sk_dim*(n+1)];
    double *X = new double[n];
    double *res = new double[n];

    // Initialize workspace for the sketched svd 
    double *U = new double[sk_dim*sk_dim];
    double *svals = new double[n+1];
    double *VT = new double[(n+1)*(n+1)];

    // Initialize noisy gaussian data
    init_noisy_data(m, n, 1, AB);

    // Define properties of the sketching operator

    // Initialize seed for random number generation
    uint32_t seed = 0;

    // Define the dense distribution that the sketching operator will sample from
    /* Additional dense distributions: RandBLAS::DenseDistName::Gaussian - entries are iid standard normal
     *                                 RandBLAS::DenseDistName::BlackBox - entires are user provided through a buffer
    */
    auto time_constructsketch1 = high_resolution_clock::now();
    RandBLAS::DenseDistName dn = RandBLAS::DenseDistName::Uniform;
    
    // Initialize dense distribution struct for the sketching operator
    RandBLAS::DenseDist Dist(sk_dim,   // Number of rows of the sketching operator 
                             m,        // Number of columns of the sketching operator
                             dn);     // Distribution of the entires

    //Construct the dense sketching operator
    RandBLAS::DenseSkOp<double> S(Dist, seed);
    auto t = omp_get_num_threads();
    omp_set_num_threads(8);
    RandBLAS::fill_dense(S);
    omp_set_num_threads(t);
    auto time_constructsketch2 = high_resolution_clock::now();

    // Sketch AB
    // SAB = alpha * \op(S) * \op(AB) + beta * SAB
    auto time_sketch1 = high_resolution_clock::now();
    RandBLAS::sketch_general(
            blas::Layout::ColMajor,    // Matrix storage layout of AB and SAB
            blas::Op::NoTrans,         // NoTrans => \op(S) = S, Trans => \op(S) = S^T
            blas::Op::NoTrans,         // NoTrans => \op(AB) = AB, Trans => \op(AB) = AB^T
            sk_dim,                    // Number of rows of S and SAB
            n + 1,                     // Number of columns of AB and SAB
            m,                         // Number of rows of AB and columns of S
            1.0,                       // Scalar alpha - if alpha is zero AB is not accessed
            S,                         // A DenseSkOp or SparseSkOp sketching operator
            AB,                        // Matrix to be sketched
            m,                         // Leading dimension of AB
            0.0,                       // Scalar beta - if beta is zero SAB is not accessed
            SAB,                       // Sketched matrix SAB
            sk_dim                     // Leading dimension of SAB
    );
    auto time_sketch2 = high_resolution_clock::now();

    // Perform SVD operation on SAB
    auto time_TLS1 = high_resolution_clock::now();
    lapack::gesdd(lapack::Job::AllVec, sk_dim, (n+1), SAB, sk_dim, svals, U, sk_dim, VT, n+1);
            
    for (int i = 0; i < n; i++) {
        X[i] = VT[n + i*(n+1)]; // Take the right n by 1 block of V
    }

    // Scale X by the inverse of the 1 by 1 bottom right block of V
    blas::scal(n, -1/VT[(n+1)*(n+1)-1], X, 1); 
    auto time_TLS2 = high_resolution_clock::now();

    //Check TLS solution. Expected to be close to a vector of 1's
    double res_infnorm = 0;
    double res_twonorm = 0;

    for (int i = 0; i < n; i++) {
        res[i] = abs(X[i] - 1);
        res_twonorm += res[i]*res[i];
        if (res_infnorm < res[i]) {
            res_infnorm = res[i];
        }
    }

    std::cout << "Matrix dimensions:                                                " << m << " by " << n+1 << '\n'; 
    std::cout << "Sketch dimension:                                                 " << sk_dim << '\n'; 
    std::cout << "Time to create dense sketching operator:                          " << (double) duration_cast<milliseconds>(time_constructsketch2 - time_constructsketch1).count()/1000 << " seconds" << '\n';
    std::cout << "Time to sketch AB:                                                " << (double) duration_cast<milliseconds>(time_sketch2 - time_sketch1).count()/1000 << " seconds" <<'\n';
    std::cout << "Time to perform TLS on sketched matrix:                           " << (double) duration_cast<milliseconds>(time_TLS2 - time_TLS1).count()/1000 << " seconds" << '\n';
    std::cout << "Inf-norm distance from TLS solution to vector of all ones:        " << res_infnorm << '\n';
    std::cout << "Two-norm distance from TLS solution to vector of all ones:        " << sqrt(res_twonorm) << '\n';

    delete[] AB;
    delete[] SAB;
    delete[] X;
    delete[] res;
    delete[] U;
    delete[] svals;
    delete[] VT;
    return 0;
}


    

