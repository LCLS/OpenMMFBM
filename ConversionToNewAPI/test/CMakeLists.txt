project( OpenMMFBM_Tests)
cmake_minimum_required( VERSION 2.6 )

include_directories("include")
include_directories("../include")
include_directories("../src")

file( GLOB OPENMMFBM_TEST_HEADERS include/OpenMMFBM/*.h )
source_group( "include\\OpenMMFBM" FILES ${OPENMMFBM_TEST_HEADERS} )

file( GLOB OPENMMFBM_TEST_SOURCES src/OpenMMFBM/*.cpp )
source_group( "src\\OpenMMFBM" FILES ${OPENMMFBM_TEST_SOURCES} )

file( GLOB OPENMMFBM_TEST_CUDA_SOURCES src/OpenMMFBM/*.cu )
source_group( "src\\OpenMMFBM" FILES ${OPENMMFBM_TEST_CUDA_SOURCES} )

find_package(CUDA REQUIRED)

#file(GLOB OPENMMFBM_CUDA_KERNELS ../src/OpenMMFBM/kernels/*.cu)
#source_group("Kernels\\OpenMMFBM" FILES ${OPENMMFBM_CUDA_KERNELS})

# Intel compiler patch
if( ${CMAKE_CXX_COMPILER} MATCHES "icpc.*$" OR ${CMAKE_CXX_COMPILER} MATCHES "icc.*$" )
    list( APPEND CUDA_NVCC_FLAGS "-ccbin=icc" )
endif()

CUDA_ADD_EXECUTABLE(OpenMMFBMCudaTests ${OPENMMFBM_TEST_HEADERS} ${OPENMMFBM_TEST_SOURCES} ${OPENMMFBM_TEST_CUDA_SOURCES} ${OPENMMFBM_CUDA_KERNELS})
