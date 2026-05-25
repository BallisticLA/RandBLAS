# Minimal downstream-consumer smoke test for find_package(RandBLAS).
#
# This is NOT part of the in-tree test build. It is configured separately by
# .github/workflows/downstream-consumer.yml against an *installed* RandBLAS, so
# that the install/export rules in RandBLAS/CMakeLists.txt are exercised the
# way a real downstream project would exercise them.
cmake_minimum_required(VERSION 3.11)
project(randblas_downstream_smoke CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RandBLAS REQUIRED)

add_executable(smoke main.cc)
target_link_libraries(smoke PRIVATE RandBLAS)
