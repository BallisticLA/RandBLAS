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
    exhaustive_fisher_yates_tests(14, 10, 0.05);
    return 0;
}
