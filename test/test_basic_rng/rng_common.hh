#pragma once

#include "RandBLAS.hh"
#include <vector>
#include <array>

namespace RandBLAS_StatTests {

//
// MARK: constants 
// ^ and functions to perform lookups for the constants we store.

namespace KolmogorovSmirnovConstants {


/*** From scipy.stats:  critical_value = kstwo.ppf(1-significance, sample_size) */

const int SMALLEST_SAMPLE = 8;
const int LARGEST_SAMPLE = 16777216;

const std::vector<int> sample_sizes {
          8,       16,       32,       64,      128,      256,
        512,     1024,     2048,     4096,     8192,    16384,
      32768,    65536,   131072,   262144,   524288,  1048576,
    2097152,  4194304,  8388608, 16777216
};

const double WEAKEST_SIGNIFICANCE = 0.05;
const double STRONGEST_SIGNIFICANCE = 1e-6;

const std::vector<double> significance_levels {
    0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6
};

const std::array<const std::array<double, 22>, 6> critical_values {{{
    // significance of 0.05
    4.54266591e-01, 3.27333470e-01, 2.34240860e-01, 1.66933746e-01,
    1.18658276e-01, 8.42018587e-02, 5.96844982e-02, 4.22742678e-02,
    2.99273847e-02, 2.11791555e-02, 1.49845091e-02, 1.05999173e-02,
    7.49739992e-03, 5.30252270e-03, 3.74997893e-03, 2.65189975e-03,
    1.87530828e-03, 1.32610915e-03, 9.37733728e-04, 6.63094351e-04,
    4.68886747e-04, 3.31557115e-04
}, { 
    // significance of 1e-2
    5.41792524e-01, 3.92007307e-01, 2.80935776e-01, 2.00288899e-01,
    1.42362543e-01, 1.01005285e-01, 7.15810977e-02, 5.06916722e-02,
    3.58812433e-02, 2.53898259e-02, 1.79621350e-02, 1.27054989e-02,
    8.98630003e-03, 6.35534434e-03, 4.49443988e-03, 3.17831443e-03,
    2.24754012e-03, 1.58931697e-03, 1.12384982e-03, 7.94698321e-04,
    5.61944814e-04, 3.97359107e-04
}, {
    // significance of 1e-3
    6.40978605e-01, 4.67504918e-01, 3.36105510e-01, 2.39914323e-01,
    1.70596759e-01, 1.21045341e-01, 8.57783224e-02, 6.07400729e-02,
    4.29898739e-02, 3.04175670e-02, 2.15177021e-02, 1.52198122e-02,
    1.07642402e-02, 7.61255632e-03, 5.38342952e-03, 3.80692734e-03,
    2.69203739e-03, 1.90362429e-03, 1.34609876e-03, 9.51852090e-04,
    6.73069322e-04, 4.75936005e-04
}, {
    // significance of 1e-4
    7.20107998e-01, 5.30358433e-01, 3.82763541e-01, 2.73655631e-01,
    1.94715231e-01, 1.38189709e-01, 9.79338402e-02, 6.93467436e-02,
    4.90797287e-02, 3.47251628e-02, 2.45641333e-02, 1.73741413e-02,
    1.22876435e-02, 8.68978729e-03, 6.14515467e-03, 4.34555112e-03,
    3.07290290e-03, 2.17293722e-03, 1.53653188e-03, 1.08650868e-03,
    7.68285928e-04, 5.43264318e-04
}, {
    // significance of 1e-5
    7.84314235e-01, 5.84534035e-01, 4.23688590e-01, 3.03463697e-01,
    2.16091906e-01, 1.53407046e-01, 1.08732306e-01, 7.69956230e-02,
    5.44929246e-02, 3.85544959e-02, 2.72724540e-02, 1.92894157e-02,
    1.36420186e-02, 9.64750036e-03, 6.82236902e-03, 4.82441714e-03,
    3.41151342e-03, 2.41237141e-03, 1.70583756e-03, 1.20622593e-03,
    8.52938821e-04, 6.03122960e-0
}, {
    // significance of 1e-6
    8.36962528e-01, 6.32173765e-01, 4.60387149e-01, 3.30395198e-01,
    2.35470356e-01, 1.67220735e-01, 1.18543880e-01, 8.39483725e-02,
    5.94144379e-02, 4.20363448e-02, 2.97351313e-02, 2.10310168e-02,
    1.48735960e-02, 1.05183852e-02, 7.43818754e-03, 5.25987011e-03,
    3.71942641e-03, 2.63009921e-03, 1.85979452e-03, 1.31508999e-03,
    9.29917360e-04, 6.57555013e-04
}}};

/***
 * Returns the index in significance_levels for the "least significant" value
 * that is "more significant" than "sig".
 * 
 * The correctness of this function depends on significance_levels being sorted
 * in decreasing order (which corresponds to weakest to strongest significances).
 */
int significance_rep(double sig) {
    randblas_require(STRONGEST_SIGNIFICANCE <= sig && sig <= WEAKEST_SIGNIFICANCE);
    int num_siglevels = (int) significance_levels.size();
    for (int i = 0; i < num_siglevels; ++i) {
        if (significance_levels[i] <= sig)
            return i;
    }
    // This code shouldn't be reachable!
    randblas_require(false);
    return -1;
}

/***
 * Returns the index in sample_sizes for the smallest sample size that's >= n.
 * 
 * The correctness of this function depends on sample_sizes being sorted in
 * increasing order.
 */
int sample_size_rep(int n) {
    randblas_require(SMALLEST_SAMPLE <= n && n <= LARGEST_SAMPLE);
    int num_sample_sizes = (int) sample_sizes.size();
    for (int i = 0; i < num_sample_sizes; ++i) {
        if (sample_sizes[i] >= n)
            return i;
    }
    // This code shouldn't be reachable!
    randblas_require(false);
    return -1;
}

std::tuple<double,int,double> critical_value_rep(int n, double sig) {
    int i = significance_rep(sig);
    auto override_sig = significance_levels[i];
    int j = sample_size_rep(n);
    auto override_sample = sample_sizes[j];
    auto cv = critical_values[i][j];
    return {cv, override_sample, override_sig};
}

template <typename TI>
double critical_value_rep_mutator(TI &n, double &sig) {
    int i = significance_rep(sig);
    sig = significance_levels[i];
    int j = sample_size_rep(n);
    n = (TI) sample_sizes[j];
    double cv = critical_values[i][j];
    return cv;
}

}

//
// MARK: combinatorics
//

double log_binomial_coefficient(int64_t n, int64_t k) {
    double result = 0.0;
    for (int64_t i = 1; i <= k; ++i) {
        result += std::log(static_cast<double>(n - i + 1)) - std::log(static_cast<double>(i));
    }
    return result;
}

//
// MARK: hypergeometric 
//

/***
 * Compute the probability mass function of the hypergeometric distribution with parameters N, K, D.
 * Concretely ... 
 * 
 *      Suppose we draw D items without replacement from a set of size N that has K distinguished elements.
 *      This function returns the probability that the sample of D items will contain observed_k elements
 *      from the distinguished set.
 */
double hypergeometric_pmf(int64_t N, int64_t K, int64_t D, int64_t observed_k) {
    randblas_require(0 <= K && K <= N);
    randblas_require(0 <= D && D <= N);
    randblas_require(0 <= observed_k && observed_k <= K);
    double lognum = log_binomial_coefficient(N - K, D - observed_k) + log_binomial_coefficient(K, observed_k);
    double logden = log_binomial_coefficient(N, D);
    double exparg = lognum - logden;
    double out = std::exp(exparg);
    return out;
}

// Call hypergeometric_pmf for a range to make hypergeometric_pmf_arr
std::vector<double> hypergeometric_pmf_arr(int64_t N, int64_t K, int64_t D) {
    randblas_require(0 <= K && K <= N);
    randblas_require(0 <= D && D <= N);
    std::vector<double> pmf(D + 1);
    for (int64_t observed_k = 0; observed_k <= D; ++observed_k)
    {
        pmf[k] = hypergeometric_pmf(N, K, D, observed_k);
    }
    return pmf;
}

double hypergeometric_mean(int64_t N, int64_t K, int64_t D) {
    double dN = (double) N;
    double dK = (double) K;
    double dD = (double) D;
    return dD * dK / dN;
}

double hypergeometric_variance(int64_t N, int64_t K, int64_t D) {
    double dN = (double) N;
    double dK = (double) K;
    double dD = (double) D;

    auto t1 = dK / dN;
    auto t2 = (dN - dK) / dN;
    auto t3 = (dN - dD) / (dN - 1.0);
    return dD * t1 * t2 * t3;
}

} // end namespace RandBLAS_StatTests



//
// MARK Kolmogorov-Smirnov Calculations
//

// Function to check the KS-Stat against crit values
std::pair<int,double> ks_check_critval(const std::vector<double> &cdf1, const std::vector<double> &cdf2, double critical_value)
{
    assert(cdf1.size() == cdf2.size()); // Vectors must be of same size to perform test

    for (size_t i = 0; i < cdf1.size(); ++i) {
        double diff = std::abs(cdf1[i] - cdf2[i]);
        if (diff > critical_value) {
            return {i, diff}; // the test failed.
        }
    }
    return {-1, 0.0};  // interpret a negative return value as the test passing.
}