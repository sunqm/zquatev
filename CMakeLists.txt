cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0048 OLD)
project(libzquatev)

find_package(BLAS)

add_subdirectory(pybind11)
pybind11_add_module(libzquatev src/zquatev.cc src/blocked.cc src/unblocked.cc src/transpose.cc src/wrapper.cc)
target_link_libraries(libzquatev PUBLIC ${BLAS_LIBRARIES})
