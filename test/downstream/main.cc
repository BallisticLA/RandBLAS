// Downstream consumer smoke test for find_package(RandBLAS).
//
// The pattern below is taken directly from
// examples/total-least-squares/tls_dense_skop.cc (lines 56-62, the
// init_noisy_data routine) so it tracks real working RandBLAS usage rather
// than a hand-rolled snippet. Goal: catch breakage in the install/export
// path, not exercise numerical behavior — keep it tiny.
#include <RandBLAS.hh>

int main() {
    constexpr int64_t m = 16;
    constexpr int64_t n = 4;

    double A[m * n];

    RandBLAS::DenseDist Dist_A(m, n);
    RandBLAS::RNGState state(0);
    RandBLAS::fill_dense(Dist_A, A, state);

    return 0;
}
