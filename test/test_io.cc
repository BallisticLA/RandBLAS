

#include <cmath>
#include <blas.hh>

#include "RandBLAS/base.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/config.h"

using blas::Layout;
using RandBLAS::ArrayStyle;
using RandBLAS::print_buff_to_stream;

#include <iostream>
#include <vector>
#include <gtest/gtest.h>


class TestIO: public ::testing::Test {
    protected:

    std::vector<double> get_pi_mat(int64_t n_rows, int64_t n_cols) {
        std::vector<double> out(n_rows * n_cols, std::acos(-1.0));
        return out;
    }
};

TEST_F(TestIO, test_matlab_tiny) {
    auto vec = get_pi_mat( 2, 1);
    print_buff_to_stream(std::cout, Layout::ColMajor, 1, 1, vec.data(), 1, "mat1x1", 9, ArrayStyle::MATLAB);
    print_buff_to_stream(std::cout, Layout::ColMajor, 2, 1, vec.data(), 2, "mat2x1", 9, ArrayStyle::MATLAB);
    print_buff_to_stream(std::cout, Layout::ColMajor, 1, 2, vec.data(), 1, "mat1x2", 9, ArrayStyle::MATLAB);
}

TEST_F(TestIO, test_matlab_medium) {
    auto vec = get_pi_mat( 7, 3);
    vec[1] *= -1;
    print_buff_to_stream(std::cout, Layout::ColMajor, 7, 3, vec.data(), 7, "Neg in first column\nmat7x3_a", 3, ArrayStyle::MATLAB);
    print_buff_to_stream(std::cout, Layout::RowMajor, 7, 3, vec.data(), 3, "Neg in first row\nmat7x3_b", 3, ArrayStyle::MATLAB);
    print_buff_to_stream(std::cout, Layout::RowMajor, 3, 7, vec.data(), 7, "Neg in first row\ntrans_mat7x3_a", 3, ArrayStyle::MATLAB);
}


TEST_F(TestIO, test_python_tiny) {
    auto vec = get_pi_mat( 2, 1);
    print_buff_to_stream(std::cout, Layout::ColMajor, 1, 1, vec.data(), 1, "arr1x1", 9, ArrayStyle::Python);
    print_buff_to_stream(std::cout, Layout::ColMajor, 2, 1, vec.data(), 2, "arr2x1", 9, ArrayStyle::Python);
    print_buff_to_stream(std::cout, Layout::ColMajor, 1, 2, vec.data(), 1, "arr1x2", 9, ArrayStyle::Python);
}


TEST_F(TestIO, test_python_medium) {
    auto vec = get_pi_mat( 7, 3);
    vec[1] *= -1;
    print_buff_to_stream(std::cout, Layout::ColMajor, 7, 3, vec.data(), 7, "Neg in first column\nmat7x3_a", 3, ArrayStyle::Python);
    print_buff_to_stream(std::cout, Layout::RowMajor, 7, 3, vec.data(), 3, "Neg in first row\nmat7x3_b", 3, ArrayStyle::Python);
    print_buff_to_stream(std::cout, Layout::RowMajor, 3, 7, vec.data(), 7, "Neg in first row\ntrans_mat7x3_a", 3, ArrayStyle::Python);
}
