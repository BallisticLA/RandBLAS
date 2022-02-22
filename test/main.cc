#include <sparseops.hh>
#include <blas.hh>
#include "gtest/gtest.h"


int main(int argc, char *argv[]) {
    //int64_t n = 20;
    //int64_t m = 2*n;
    //run_pcgls_ex(n, m);

    double a[25];
    int64_t k = 5;
	genmat(k, k, a, 0); 
     
    ::testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
