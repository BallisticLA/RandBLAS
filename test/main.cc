#include <sparseops.hh>
#include <blas.hh>


int main(int argc, const char *argv[]) {
    int64_t n = 20;
    int64_t m = 2*n;
    run_pcgls_ex(n, m);

    double a[25];
    int64_t k = 5;
	genmat(k, k, a, 0); 
    return 0;
}
