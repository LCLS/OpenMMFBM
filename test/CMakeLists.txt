project( OpenMMFBM_Tests )
cmake_minimum_required( VERSION 2.6 )

find_package( CUDA REQUIRED )

include_directories("include")

# CPPUnit
set( CPPUNIT_DIR "" CACHE PATH "CPPUnit Install Directory" )
if( CPPUNIT_DIR )
	link_directories( "${CPPUNIT_DIR}/lib" )
	include_directories( "${CPPUNIT_DIR}/include" )
endif( CPPUNIT_DIR )

# QR Code Test
option( BUILD_QR "Build QR code tests" Off )
if( BUILD_QR )
    include_directories("../src/")

    set( TEST_HEADERS "include/QRTest.h" )
    set( TEST_SOURCES "src/QRTest.cpp" "../src/OpenMMFBM/kernels/qr.cu" )
endif( BUILD_QR )

# OpenMM FBM Tests
option( BUILD_FBM "Build FBM tests" Off )
if( BUILD_FBM )
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
endif( BUILD_FBM )

find_library( CPPUNIT_LIB "cppunit" hint "${CPPUNIT_DIR}/lib" )
list( APPEND LIBS ${CPPUNIT_LIB} )

# Build
CUDA_ADD_EXECUTABLE( FBMTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} )
target_link_libraries( FBMTest ${LIBS} )
