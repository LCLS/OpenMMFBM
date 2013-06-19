project( OpenMMFBM )
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
		if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$") 
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

# First Order
option( BUILD_FIRST_ORDER "Build with first order perterb" Off )
if( BUILD_FIRST_ORDER )
	add_definitions( "-DFIRST_ORDER" )
endif( BUILD_FIRST_ORDER )

# OpenMMFBM Library
file( GLOB OPENMMFBM_HEADERS include/OpenMMFBM/*.h )
source_group( "include\\OpenMMFBM" FILES ${OPENMMFBM_HEADERS} )

file( GLOB OpenMMFBM_SOURCES src/OpenMMFBM/*.cpp )
source_group( "src\\OpenMMFBM" FILES ${OpenMMFBM_SOURCES} )

add_library( OpenMMFBM SHARED ${OpenMMFBM_HEADERS} ${OpenMMFBM_SOURCES})
target_link_libraries( OpenMMFBM ${OPENMM_LIB} )

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

install( TARGETS OpenMMFBM LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib" )
