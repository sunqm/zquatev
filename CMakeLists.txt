cmake_minimum_required (VERSION 3.10)
project(libzquatev)

set(CMAKE_BUILD_TYPE RELWITHDEBINFO)

# A given BLAS_LIBRARIES (e.g. -DBLAS_LIBRARIES=mkl_rt, as pyscf/lib passes
# it) is used as is and also provides LAPACK; otherwise FindBLAS/FindLAPACK.
if(NOT BLAS_LIBRARIES)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

add_library(zquatev SHARED 
  src/zquatev.cc src/blocked.cc src/unblocked.cc src/transpose.cc src/wrapper.cc)

target_link_libraries(zquatev PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

install(TARGETS zquatev
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib)
