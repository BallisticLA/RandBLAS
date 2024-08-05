#include <iostream>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <utility>
#include <RandBLAS.hh>
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include <utils.hh>
#include "rng_common.hh"

int main()
{
    const int64_t d = 9; // Range
    const int64_t k = 7; // Number of selected elements
    const int64_t n = 32; // Number of samples
    double sig = 0.01; // Significance value for KS test

    // Perform a single Fisher-Yates test with k=33, getting the KS statistic and significance level
    std::vector<int64_t> idxs_major(k * n);
    std::vector<int64_t> idxs_minor(k * n);
    std::vector<double> vals(k * n);

    RandBLAS::RNGState state(0); // Fisher-Yates needs an RNG-State passed

     // Generate repeated Fisher-Yates in idxs_major
    state = RandBLAS::repeated_fisher_yates(
        state, k, d, n,  // k=vec_nnz, d=dim_major, n=dim_minor
        idxs_major.data(), idxs_minor.data(), vals.data());

    // Make the empirical cdf and print it
    std::vector<double> empirical_pmf = fisher_yates_pmf(idxs_major, k, n);
    std::vector<double> empirical_cdf(empirical_pmf.size());
    std::partial_sum(empirical_pmf.begin(), empirical_pmf.end(), empirical_cdf.begin());
    std::cout << "Empirical CDF: ";
    for (const auto &val : empirical_cdf) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Calculate and print a specific value of the hypergeometric PMF
    double pmf_test = RandBLAS_StatTests::hypergeometric_pmf(d, d, k, 5);
    std::cout << "Hypergeometric PMF for d=" << d << ", k=" << k << ", observed_k=5: " << pmf_test << std::endl;

    // Generate the theoretical hypergeometric CDF and print it
    // std::vector<double> theoretical_pmf = hypergeometric_pmf_arr(d, d, k);
    // std::cout << "Theoretical PMF: ";
    // for (const auto &val : theoretical_pmf) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;
    // std::vector<double> theoretical_cdf(theoretical_pmf.size());
    // std::partial_sum(theoretical_pmf.begin(), theoretical_pmf.end(), theoretical_cdf.begin());
    // std::cout << "Theoretical CDF: ";
    // for (const auto &val : theoretical_cdf) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    // // Compute the critval and check against it
    // double crit_value;
    // int override_sample;
    // double override_sig;
    // std::tie(crit_value, override_sample, override_sig) = RandBLAS_StatTests::KolmogorovSmirnovConstants::critical_value_rep(n, sig);
    // std::cout << "Critical value: " << crit_value << std::endl;
    // std::pair<int, double> result = ks_check_critval(theoretical_cdf, empirical_cdf, crit_value);
    // if (result.first != -1) {
    // std::cout << "KS test failed at index " << result.first << " with difference " << result.second << std::endl;
    // } else {
    //     std::cout << "KS test passed." << std::endl;
    // }

    return 0;
}
