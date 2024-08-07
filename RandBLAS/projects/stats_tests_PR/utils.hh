#include <string>
#include <limits>
#include <vector>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <RandBLAS.hh> // This includes sparse_skops.hh which has fisher_yates
#include <RandBLAS/random_gen.hh>
#include <RandBLAS/exceptions.hh>
#include "rng_common.hh"


//
// Mark: Utility functions that I've been using for debugging
//
void print_vector_debug(const std::vector<double>& vec, const std::string& label, bool debug) {
    if (debug) {
        std::cout << label << ": ";
        if (vec.size() > 100) {
            for (size_t i = 0; i < 10; ++i) {
                std::cout << vec[i] << " ";
            }
            std::cout << "... ";
            for (size_t i = vec.size() - 10; i < vec.size(); ++i) {
                std::cout << vec[i] << " ";
            }
        } else {
            for (const auto &val : vec) {
                std::cout << val << " ";
            }
        }
        std::cout << std::endl;
    }
}
// Example use cases:
//// print_vector_debug(empirical_cdf, "Empirical CDF", debug);
//// print_vector_debug(theoretical_cdf, "Theoretical CDF", debug);




//
// Mark: Function that I'm not sure where to put in RandBLAS
//
std::vector<double> fisher_yates_pmf(const std::vector<int64_t> &idxs_major, int64_t d, int64_t k, int64_t n) {
    std::vector<double> empirical_pmf;

    int64_t min_observed_k = std::max<int64_t>(0, k-(d-k));  // If you are picking more elements than non-distinguished elements, the difference becomes the minimum
    int64_t max_observed_k = std::min<int64_t>(d, k);  // Can't pick more distinguished elements than there are in the set

    // If k is 0, then there's nothing to count over and we should just return 1
    if (k == 0) {
        empirical_pmf.push_back(1.0);
    } else {
        // Count how many values in idxs_major are less than k
        std::vector<int64_t> counter(max_observed_k - min_observed_k + 1, 0);
        for (int64_t i = 0; i < k * n; i += k) {
            int count = 0;
            for (int64_t j = 0; j < k ; ++j) {
                if (idxs_major[i + j] < k) {
                    count += 1;
                }
            }
            std::cout << "Count: " << count << std::endl;
            counter[count] += 1;
        }

         // Normalize the counts to get the empirical pmf
        empirical_pmf.resize(counter.size());
        for (int64_t i = 0; i < counter.size(); ++i) {
            empirical_pmf[i] = static_cast<double>(counter[i]) / n;
        }
    }

    return empirical_pmf;
}


bool fisher_yates_test(int64_t d, int64_t k, int64_t n, double sig = 0.01, bool debug = false) {
    // Initialize arguments for repeated_fisher_yates
    std::vector<int64_t> idxs_major(k * n);
    std::vector<int64_t> idxs_minor(k * n);
    std::vector<double> vals(k * n);
    RandBLAS::RNGState state(0);

    // Generate repeated Fisher-Yates in idxs_major
    state = RandBLAS::repeated_fisher_yates(
        state, k, d, n,  // k=vec_nnz, d=dim_major, n=dim_minor
        idxs_major.data(), idxs_minor.data(), vals.data());

    // Calculate the empirical cdf
    std::vector<double> empirical_pmf = fisher_yates_pmf(idxs_major, d, k, n);
    std::vector<double> empirical_cdf(empirical_pmf.size());
    std::partial_sum(empirical_pmf.begin(), empirical_pmf.end(), empirical_cdf.begin());
    print_vector_debug(empirical_cdf, "Empirical CDF", debug);

    // Generate the theoretical hypergeometric cdf
    std::vector<double> theoretical_pmf = RandBLAS_StatTests::hypergeometric_pmf_arr(d, k, k);
    std::vector<double> theoretical_cdf(theoretical_pmf.size());
    std::partial_sum(theoretical_pmf.begin(), theoretical_pmf.end(), theoretical_cdf.begin());
    // Take the max value of the theoretical cdf and divide all items by it
    double max_val = *std::max_element(theoretical_cdf.begin(), theoretical_cdf.end());
    print_vector_debug(theoretical_cdf, "Theoretical CDF", debug);
    // for (auto &val : theoretical_cdf) {
    //     val /= max_val;
    // }
    // print_vector_debug(theoretical_cdf, "Theoretical CDF", debug);

    // Compute the critval and check against it
    double crit_value = RandBLAS_StatTests::KolmogorovSmirnovConstants::critical_value_rep_mutator(n, sig);
    std::pair<int, double> result = ks_check_critval(theoretical_cdf, empirical_cdf, crit_value);
    std::cout << std::endl;
    if (result.first != -1) {
        std::cout << std::endl;
        std::cout << "KS test failed at index " << result.first << " with difference " << result.second << " and critical value " << crit_value << std::endl;
        std::cout << "Test parameters: " << "d=" << d << " " "k=" << k << " " "n=" << n << std::endl;
        return true;
    }
    else {
        std::cout << "LINE BEFORE ERROR" << std::endl;
        return false;
    }
}

void exhaustive_fisher_yates_tests(int64_t d, int64_t n, double sig = 0.01, bool debug = false) {
    for (int64_t k = 0; k <= d; ++k) {
        std::cout << "Test1" << std::endl;
        if (debug) {
            std::cout << "Testing d=" << d << " k=" << k << " n=" << n << " sig=" << sig << std::endl;
        }
        std::cout << "Test2" << std::endl;
        if(fisher_yates_test(d, k, n, sig, debug)) {
            std::cout << "Test3" << std::endl;
            break;
        }
        std::cout << "Test4" << std::endl;
    }
}
