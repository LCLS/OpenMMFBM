project( OpenMMFBMCuda )
cmake_minimum_required( VERSION 2.6 )

include_directories( "include" )

# MKL
option( BUILD_MKL "Build with Intel MKL code" Off )
if( BUILD_MKL )
	add_definitions( "-DINTEL_MKL" )

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl=parallel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -mkl=parallel")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mkl=parallel")
endif( BUILD_MKL )

# OpenMP
option( BUILD_OPENMP "Build with openmp support" Off )
if( BUILD_OPENMP )
    # Intel
    if (${CMAKE_CXX_COMPILER} MATCHES "icc.*$")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -openmp")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -openmp")
    else()
      # Other
      include( FindOpenMP )
      if( OPENMP_FOUND )
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      endif( OPENMP_FOUND )
		endif()
endif( BUILD_OPENMP )

# OpenMM
set( OPENMM_DIR "$ENV{OPENMM_HOME}" CACHE PATH "OpenMM Install Directory")
if( OPENMM_DIR )
	link_directories( "${OPENMM_DIR}/lib/" )
	include_directories( "${OPENMM_DIR}/include/" )
endif()

set( OPENMM_SOURCE_DIR "$ENV{OPENMM_SOURCE}" CACHE PATH "OpenMM Source Directory")
if( OPENMM_SOURCE_DIR )
	# JAMA
	include_directories( "${OPENMM_SOURCE_DIR}/libraries/jama/include/" )
endif()

find_library( OPENMM_LIB "OpenMM" HINT ${OPENMM_DIR}/lib )
find_library( OPENMM_CUDA_LIB "OpenMMCuda" HINT "${OPENMM_DIR}/lib/plugins/")

# First Order
option( BUILD_FIRST_ORDER "Build with first order perterb" Off )
if( BUILD_FIRST_ORDER )
	add_definitions( "-DFIRST_ORDER" )
endif( BUILD_FIRST_ORDER )

# OpenMMFBM Library
file( GLOB OPENMMFBM_HEADERS include/FBM/*.h )
source_group( "include\\FBM" FILES ${OPENMMFBM_HEADERS} )

file( GLOB OPENMMFBM_SOURCES src/OpenMMFBM/*.cpp )
source_group( "src\\FBM" FILES ${OPENMMFBM_SOURCES} )

message( STATUS "${OPENMMFBM_HEADERS} ${OPENMMFBM_SOURCES} ${OPENMMFBM_KERNEL_SOURCES}")

# GPU bits
# eventually we'll need to move the GPU code into a separate directory
# to prevent adding it with the glob below or add a define around it
# to prevent it from being compiled
option( BUILD_GPU "Build GPU code" Off )
if( BUILD_GPU )
    find_package( CUDA REQUIRED )

    if(OPENMM_SOURCE_DIR)
		include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/src/kernels")
		include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/include")
		include_directories("${OPENMM_SOURCE_DIR}/platforms/cuda/src")
    endif()

    set( MAGMA_DIR "$ENV{MAGMA_HOME}" CACHE PATH "MAGMA Install Directory" )
    if( MAGMA_DIR )
		add_definitions( "-DHAVE_CUBLAS" )
    	link_directories("${MAGMA_DIR}/lib/" )
		include_directories("${MAGMA_DIR}/include")
        find_library( MAGMA_LIB "magma" HINT "${MAGMA_DIR}/lib/")
    endif()

    set( CULA_DIR "$ENV{CULA_HOME}" CACHE PATH "CULA Install Directory" )
    if( CULA_DIR )
    	#link_directories("${CULA_DIR}/lib64/" )
		include_directories("${CULA_DIR}/include")
    endif()

    file( GLOB OPENMMFBM_CUDA_HEADERS include/FBM/*.h )
    source_group( "include\\FBM" FILES ${OPENMMFBM_CUDA_HEADERS})

    file( GLOB OPENMMFBM_CUDA_SOURCES src/FBM/*.cpp )
    source_group( "src\\FBM" FILES ${OPENMMFBM_CUDA_SOURCES} )

    # Intel compiler patch
    if( ${CMAKE_CXX_COMPILER} MATCHES "icpc.*$" OR ${CMAKE_CXX_COMPILER} MATCHES "icc.*$" )
    	list( APPEND CUDA_NVCC_FLAGS "-ccbin=icc" )
    endif()

	# OS X compiler patch
	if( ${CMAKE_CXX_COMPILER} MATCHES "clang.*$" )
		list( APPEND CUDA_NVCC_FLAGS "-ccbin=clang" )
	endif()

    CUDA_ADD_LIBRARY( OpenMMFBMCuda SHARED ${OPENMMFBM_CUDA_HEADERS} ${OPENMMFBM_CUDA_SOURCES} ${OPENMMFBM_HEADERS} ${OPENMMFBM_SOURCES})
    set_target_properties( OpenMMFBMCuda PROPERTIES COMPILE_FLAGS "-DOPENMMFBM_CUDA")
endif()

target_link_libraries( OpenMMFBMCuda ${OPENMM_LIB} ${OPENMM_CUDA_LIB} ${MAGMA_LIB} )

# Testing
option( BUILD_TESTING "Build test code" Off )
if( BUILD_TESTING )
	add_subdirectory( "test" )
endif( BUILD_TESTING )

# Installation
install(
	DIRECTORY "include/"
	DESTINATION "include"
	PATTERN "*.h"
	PATTERN ".*" EXCLUDE
)

install( TARGETS OpenMMFBMCuda LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib" )
