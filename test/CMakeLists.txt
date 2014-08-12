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
    include_directories("$ENV{CUDA_HOME}/samples/common/inc/" )

    set( TEST_HEADERS "include/QRTest.h" )
    #set( TEST_SOURCES "src/QRTest.cpp" "../src/OpenMMFBM/kernels/qr.cu" )
    set( TEST_SOURCES "src/QRTest.cpp" "../src/FBM/CUDA/kernels/qr.cu" )
endif( BUILD_QR )

# OpenMM FBM Tests
option( BUILD_FBM "Build FBM tests" Off )
if( BUILD_FBM )
    include_directories("include")
    include_directories("../include")
	set( TEST_HEADERS ${TEST_HEADERS} "include/CUDATest.h" )
	set( TEST_SOURCES ${TEST_SOURCES} "src/CUDATest.cpp")

	FILE(GLOB CUDA_KERNELS ../src/FBM/CUDA/kernels/*.cu)
	FOREACH( file ${CUDA_KERNELS} )
		set( TEST_SOURCES ${TEST_SOURCES} ${file} )
	ENDFOREACH( file )
endif( BUILD_FBM )

find_library( CPPUNIT_LIB "cppunit" hint "${CPPUNIT_DIR}/lib" )
list( APPEND LIBS ${CPPUNIT_LIB} )

# Build
CUDA_ADD_EXECUTABLE( FBMTest main.cpp ${TEST_SOURCES} ${TEST_HEADERS} )
target_link_libraries( FBMTest ${LIBS} )
