#include <string>
#include <limits>
#include <vector>
#include <cstdint>
#include <numeric>
#include <cassert>
#include <iostream>
#include <RandBLAS.hh> // This includes sparse_skops.hh which has fisher_yates
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include "rng_common.hh"



//
// Mark: Things I'm testing with that I have here so I don't have to recompile RandBLAS repeatedly
//
double hypergeometric_pmf2(int64_t N, int64_t K, int64_t D, int64_t observed_k) {
    randblas_require(0 <= K && K <= N);
    randblas_require(0 <= D && D <= N);
    randblas_require(0 <= observed_k && observed_k <= K);

    // Special case where the probability is zero
    if (observed_k > K || D > N || observed_k > D) {
        return 0.0;
    }

    double lognum = RandBLAS_StatTests::log_binomial_coefficient(N - K, D - observed_k) + RandBLAS_StatTests::log_binomial_coefficient(K, observed_k);
    double logden = RandBLAS_StatTests::log_binomial_coefficient(N, D);
    double exparg = lognum - logden;
    double out = std::exp(exparg);
    if (std::isnan(out)) {
            return 0.0;  // Small probabilies will be nan, so we return 0.0
        }
    return out;
}



//
// Mark: Function that I'm not sure where to put in RandBLAS
//

// Count how many values are less than or equal to vec_nnz in vec_nnz size blocks of idxs_major, then scale to make a pdf
// We use vec_nnz as a stand-in for k, and can simply check less than or equal instead of shared elements by a symmetry argument!
std::vector<double> fisher_yates_pmf(const std::vector<int64_t> &idxs_major, int64_t vec_nnz, int64_t dim_minor)
{
    // Count how many values in vec_work are less than or equal to k
    std::vector<int64_t> counter(vec_nnz + 1, 0);
    for (int64_t i = 0; i < vec_nnz * dim_minor; i += vec_nnz)
    {
        int count = 0;
        for (int64_t j = 0; j < vec_nnz; ++j)
        {
            if (idxs_major[i + j] < vec_nnz)
            {
                count += 1;
            }
        }
        counter[count] += 1;
    }

    // Normalize the counter to get the empirical pdf
    std::vector<double> empirical_pmf(counter.size());
    for (size_t i = 0; i < counter.size(); ++i)
    {
        empirical_pmf[i] = static_cast<double>(counter[i]) / dim_minor;
    }

    return empirical_pmf;
}



//
// Mark: OLD things I might want later
//

// // Perform the Kolmogorov-Smirnov test for fisher_yates with a given 'k, d, and n'
// std::pair<double, double> fisher_yates_ks_test(int64_t k, int64_t d, int64_t n)
// {
//     std::vector<int64_t> idxs_major(k * n);
//     std::vector<int64_t> idxs_minor(k * n);
//     std::vector<double> vals(k * n);

//     RandBLAS::RNGState state(0); // fisher-yates needs an RNG-State Passed

//     // Generate repeated Fisher-Yates in idxs_major
//     state = RandBLAS::repeated_fisher_yates(
//         state, k, d, n,
//         idxs_major.data(), idxs_minor.data(), vals.data());

//     // Make the empirical cdf
//     std::vector<double> empirical_pmf = fisher_yates_pmf(idxs_major, k, n);
//     std::vector<double> empirical_cdf(empirical_pmf.size());
//     std::partial_sum(empirical_pmf.begin(), empirical_pmf.end(), empirical_cdf.begin());

//     // Generate the theoretical hypergeometric CDF
//     std::vector<double> theoretical_pmf = hypergeometric_pmf_arr2(d, k);
//     std::vector<double> theoretical_cdf(theoretical_pmf.size());
//     std::partial_sum(theoretical_pmf.begin(), theoretical_pmf.end(), theoretical_cdf.begin());

//     // Compute the K-S statistic and find the significance level
//     double statistic = ks_stat(empirical_cdf, theoretical_cdf);
//     double significance = ks_stat_to_signif(statistic, n);

//     // Print the empirical and theoretical PMFs
//     std::cout << "Empirical PMF: ";
//     for (const auto &val : empirical_pmf) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Theoretical PMF: ";
//     for (const auto &val : theoretical_pmf) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;

//     return {statistic, significance};
// }

// void fisher_yates_ks_tests(int64_t d, int64_t n)
// {
//     std::vector<std::tuple<int64_t, double, double>> results;

//     for (int64_t k = 1; k <= d; ++k)
//     {
//         auto result = fisher_yates_ks_test(k, d, n);
//         results.push_back({k, result.first, result.second});
//     }

//     for (const auto &result : results)
//     {
//         std::cout << "k: " << std::get<0>(result)
//                   << ", KS Statistic: " << std::get<1>(result)
//                   << ", Significance: " << std::get<2>(result)
//                   << std::endl;
//     }
// }
