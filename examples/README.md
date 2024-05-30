# RandBLAS examples

Files in this directory show how RandBLAS can be used to implement high-level RandNLA algorithms.
Right now we have two types of examples.
 1. A sketch-and-solve approach to the total least squares problem. (Two executables.)
 2. Basic methods for low-rank approximation of sparse matrices. (Three executables.)

There is a _lot_ of code duplication within the ``.cc`` files for a given type of example.
This is a necessary evil to ensure that each example file can stand on its own.

## Building the examples
The examples are built with CMake. Before building the examples you have to build _and install_
RandBLAS and lapackpp.

We've given an example CMake configuration line below. The values we use in the configuration line
assume that ``cmake`` is invoked in a ``build`` folder that's one level beneath this file's directory.
The values also reflect specific install locations for RandBLAS and lapackpp relative to where the 
command is invoked; your situation might require specifying different paths.

```shell
cmake -DCMAKE_BINARY_DIR=`pwd` -DRandBLAS_DIR=`pwd`/../../../RandBLAS-install/lib/cmake -Dlapackpp_DIR=`pwd`/../../../lapackpp-install/lib/cmake/lapackpp ..
```

The curious are welcome to look at the examples' CMakeLists.txt file. That file also contains 
a lot of code duplication that could be avoided with some CMake-foo, but we chose to keep things
verbose so others can easily copy-paste portions of the CMakeLists.txt into their own codebase.
