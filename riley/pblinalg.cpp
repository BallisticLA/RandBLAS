#include <pybind11/pybind11.h>
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

// EIG=/home/riley/Documents/Research/protorandlapack/rlapy/extern/eigen-3.4.0
//  c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) pblinalg.cpp -o pblinalg$(python3-config --extension-suffix) -I $EIG

int add(int i, int j) {
    return i + j;
}

void scale_by_2(Eigen::Ref<Eigen::VectorXd> v) {
    v *= 2;
}

void scale_by_2_stl(std::vector<double>& vec) {
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] *= 2.0;
}

PYBIND11_MODULE(pblinalg, m) {
    m.doc() = "pybind11 linear algebra plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers.");
    m.def("scale_by_2", &scale_by_2, "A function that scales an input vector in-place by two.");
    m.def("scale_by_2_stl", &scale_by_2_stl, "Like scale_by_2, but with the C++ standard library.");
    // ^ Doesn't work. Maybe some copy operation is happening.
}

