#include <rblas/util.hh>
#include <determiter.hh>
#include <rblas/sjlts.hh>
#include <blas.hh>
#include "gtest/gtest.h"


int main(int argc, char *argv[])
{    
    ::testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();
    return res;
}
