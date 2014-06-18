#include "CUDATest.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION( FBM::CUDA );

extern "C" void TestMakeProjection( int n, int m, float *eigvec, int *indices, float *E, float *Et );
extern "C" void TestOddEvenSort( const int n, float *eigenvalues, float *eigenvectors );
extern "C" void TestCopyFrom( const int n, float4 *input, float *output );
extern "C" void TestCopyTo( const int n, float* input, float4* output );
extern "C" void TestSymmetrize1D( const size_t blocks, const size_t totalSize, float* output, float* blockHessian, float* blockSizes, int* startDof );

namespace FBM {
	#define MDIM 5
	#define NDIM 8

	void CUDA::MakeProjection() {
		// Eigenvectors
		int value = 0;
		float eigvec[NDIM * MDIM];
		for( int i = 0; i < NDIM; i++ ) {
			for( int j = 0; j < MDIM; j++ ) {
				eigvec[i * MDIM + j] = value;
				value++;
			}
		}

		// Initial indices
		int indices[MDIM];
		for( int i = 0; i < MDIM; i++ ) {
			indices[i] = i;
		}

		// Swap random indices
		for( int i = 0; i < 100; i++ ) {
			int index1 = rand() % MDIM;
			int index2 = rand() % MDIM;
			int tmp = indices[index1];
			indices[index1] = indices[index2];
			indices[index2] = tmp;
		}

		// Run Function
		float Et[MDIM * NDIM], E[NDIM * MDIM];
		TestMakeProjection( NDIM, MDIM, eigvec, indices, E, Et );

		// Test Normal Matrix
		for( int i = 0; i < NDIM; i++ ) {
			for( int j = 0; j < MDIM; j++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( E[i * MDIM + j], eigvec[i * MDIM + indices[j]], 1e-5 );
			}
		}

		// Test Transpose
		for( int i = 0; i < NDIM; i++ ) {
			for( int j = 0; j < MDIM; j++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( Et[j * NDIM + i], eigvec[i * MDIM + indices[j]], 1e-5 );
			}
		}
	}

	void CUDA::OddEvenSort() {
		// Multiples of 20
		float eigenvalues[NDIM];
		for( int i = 0; i < NDIM; i++ ) {
			eigenvalues[i] = ( NDIM - 1 ) * 20 - ( 20 * i );
		}

		// Scramble
		for( int i = 0; i < 100; i++ ) {
			int index1 = rand() % NDIM;
			int index2 = rand() % NDIM;
			float tmp = eigenvalues[index1];
			eigenvalues[index1] = eigenvalues[index2];
			eigenvalues[index2] = tmp;
		}


		// Eigenvectors
		float eigenvectors[NDIM * NDIM];
		for( int i = 0; i < NDIM; i++ ) {
			for( int j = 0; j < NDIM; j++ ) {
				eigenvectors[j * NDIM + i] = eigenvalues[i];
			}
		}

		// Run Kernel
		TestOddEvenSort( NDIM, eigenvalues, eigenvectors );

		// Test Order
		for( int i = 0; i < NDIM - 1; i++ ) {
			CPPUNIT_ASSERT( eigenvalues[i] < eigenvalues[i + 1] );
		}

		// Test Values
		for( int i = 0; i < NDIM; i++ ) {
			for( int j = 0; j < NDIM; j++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( eigenvectors[j * NDIM + i], eigenvalues[i], 1e-5 );
			}
		}
	}

	void CUDA::CopyFrom() {
		const int numBlocks = 300;
		const int numThreads = 1;

		float4 in_positions[100];
		for( unsigned int i = 0; i < 100; i++ ) {
			float4 temp;
			temp.x = ( float ) i;
			temp.y = ( float ) i;
			temp.z = ( float ) i;
			temp.w = 0.0f;
			in_positions[i] = temp;
		}

		float out_positions[300];
		for( unsigned int i = 0; i < 300; i++ ) {
			out_positions[i] = ( float ) i;
		}

		TestCopyFrom( 100, in_positions, out_positions );

		for( unsigned int i = 0; i < 100; i++ ) {
			CPPUNIT_ASSERT_DOUBLES_EQUAL( out_positions[3 * i + 0], in_positions[i].x, 1e-5 );
			CPPUNIT_ASSERT_DOUBLES_EQUAL( out_positions[3 * i + 1], in_positions[i].y, 1e-5 );
			CPPUNIT_ASSERT_DOUBLES_EQUAL( out_positions[3 * i + 2], in_positions[i].z, 1e-5 );
		}
	}

	void CUDA::BlockCopyFrom() {

	}

	void CUDA::CopyTo() {
		float4 out_positions[100];
		for( unsigned int i = 0; i < 100; i++ ) {
            float4 temp;
			temp.x = 0.0f;
			temp.y = 0.0f;
			temp.z = 0.0f;
			temp.w = 0.0f;
			out_positions[i] = temp;
		}

		float in_positions[300];
		for( unsigned int i = 0; i < 300; i++ ) {
			in_positions[i] = ( float )( i % 3 );
		}

        TestCopyTo( 100, in_positions, out_positions );

		for( unsigned int i = 0; i < 100; i++ ) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL( in_positions[3 * i + 0], out_positions[i].x, 1e-5 );
            CPPUNIT_ASSERT_DOUBLES_EQUAL( in_positions[3 * i + 1], out_positions[i].y, 1e-5 );
            CPPUNIT_ASSERT_DOUBLES_EQUAL( in_positions[3 * i + 2], out_positions[i].z, 1e-5 );
		}
	}

	void CUDA::BlockCopyTo() {

	}

	void CUDA::MatrixMultiplication() {

	}

	void CUDA::Symmetrize1D() {
		unsigned int totalSize = 0;

		int blockSizes[100];
		int startDof[100];
		for( unsigned int i = 0; i < 100; i++ ) {
			// vary the sizes by +/- 9
			const int size = 3 * ( 100 - ( ( i - 50 ) % 3 ) );
			blockSizes[i] = size / 3;
			startDof[i] = totalSize;
			totalSize += size * size;
		}

		float blockHessian[totalSize];
		for( unsigned int blockNum = 0; blockNum < 100; blockNum++ ) {
			const unsigned int blockSize = 3 * blockSizes[blockNum];

			float *block = &( blockHessian[startDof[blockNum]] );
			for( unsigned int row = 0; row < blockSize; row++ ) {
				for( unsigned int col = 0; col < blockSize; col++ ) {
					if( row < col ) {
						block[row * blockSize + col] = 1.0;
					} else {
						block[col * blockSize + row] = 0.0;
					}
				}
			}
		}


		float outBlockHessian[totalSize];
        TestSymmetrize1D( 100, totalSize, outBlockHessian, blockHessian, blockSizes, startDof);

		for( unsigned int blockNum = 0; blockNum < 100; blockNum++ ) {
			const unsigned int blockSize = 3 * blockSizes[blockNum];

			float *block = &( outBlockHessian[startDof[blockNum]] );
			for( unsigned int row = 0; row < blockSize; row++ ) {
				for( unsigned int col = row; col < blockSize; col++ ) {
                    CPPUNIT_ASSERT_DOUBLES_EQUAL( block[row * blockSize + col], block[col * blockSize + row], 1e-5 );
				}
			}
		}
	}

	void CUDA::Symmetrize2D() {
		/*const unsigned int size = 300 * 300;
		const unsigned int numThreads = 512;
		const unsigned int numBlocks = size / 512 + 1;

		float in_array[size], out_array[size];
		for( unsigned int i = 0; i < 300; i++ ) {
			for( unsigned int j = 0; j < 300; j++ ) {
				if( i < j ) {
					in_array[i * 300 + j] = 0.0f;
				} else {
					in_array[i * 300 + j] = 1.0f;
				}
				out_array[i * 300 + j] = 0.0f;
			}
		}

		float *gpu_array = NULL;
		cudaMalloc( ( void ** ) &gpu_array, size * sizeof( float ) );
		cudaMemcpy( gpu_array, in_array, size * sizeof( float ), cudaMemcpyHostToDevice );

		symmetrize2D <<< numBlocks, numThreads>>>( gpu_array, 100 );

		cudaMemcpy( out_array, gpu_array, size * sizeof( float ), cudaMemcpyDeviceToHost );

		cudaFree( gpu_array );

		for( unsigned int i = 0; i < 300; i++ ) {
			for( unsigned int j = i; j < 300; j++ ) {
                CPPUNIT_ASSERT_DOUBLES_EQUAL( out_array[i * 300 + j], out_array[j * 300 + i], 1e-5 );
			}
		}*/
	}

	void CUDA::MakeEigenvalues() {
		/*// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
		// The special 'eigenvalue' will be -1.
		// All other positive values will be equal to the matrix size.
		int size[5];
		size[0] = rand() % 10 + 1;
		size[1] = rand() % 10 + 1;
		size[2] = rand() % 10 + 1;
		size[3] = rand() % 10 + 1;
		size[4] = rand() % 10 + 1;

		int eigsize = 0;
		int hesssize = 0;

		for( int i = 0; i < 5; i++ ) {
			eigsize += 3 * size[i];
			hesssize += 3 * size[i] * 3 * size[i];
		}

		float eigenvalues[eigsize], hessian[hesssize];
		int blocknums[5], hessiannums[5];

		// Populate hessian
		int pos = 0;
		blocknums[0] = 0;
		hessiannums[0] = 0;
		for( int i = 0; i < 5; i++ ) {
			if( i > 0 ) {
				blocknums[i] = blocknums[i - 1] + size[i - 1];
				hessiannums[i] = pos;
			}
			for( int j = 0; j < 3 * size[i]; j++ )
				for( int k = 0; k < 3 * size[i]; k++ ) {
					if( j != k ) {
						hessian[pos] = size[i];
					} else {
						hessian[pos] = -12;
					}
					pos++;
				}
		}


		int numBlocks = eigsize / 512 + 1;
		int numThreads = eigsize % 512;

		float *gpuEigs;
		float *gpuHess;
		int *gpuBlockNums;
		int *gpuHessNums;
		int *gpuBlockSizes;

		cudaError_t result;
		result = cudaMalloc( ( void ** ) &gpuEigs, eigsize * sizeof( float ) );
		result = cudaMalloc( ( void ** ) &gpuHess, hesssize * sizeof( float ) );
		cudaMemcpy( gpuHess, hessian, hesssize * sizeof( float ), cudaMemcpyHostToDevice );
		result = cudaMalloc( ( void ** ) &gpuBlockNums, 5 * sizeof( int ) );
		cudaMemcpy( gpuBlockNums, blocknums, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		result = cudaMalloc( ( void ** ) &gpuBlockSizes, 5 * sizeof( int ) );
		cudaMemcpy( gpuBlockSizes, size, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		result = cudaMalloc( ( void ** ) &gpuHessNums, 5 * sizeof( int ) );
		cudaMemcpy( gpuHessNums, hessiannums, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		makeEigenvalues <<< numBlocks, numThreads>>>( gpuEigs, gpuHess, gpuBlockNums, gpuBlockSizes, gpuHessNums, eigsize, 5 );

		cudaMemcpy( eigenvalues, gpuEigs, eigsize * sizeof( float ), cudaMemcpyDeviceToHost );

        cudaFree( gpuEigs );
        cudaFree( gpuHess );
        cudaFree( gpuBlockNums );
        cudaFree( gpuBlockSizes );
        cudaFree( gpuHessNums );

		bool res = true;
		for( int i = 0; i < eigsize; i++ ){
			if( eigenvalues[i] != -12 ) {
				printf( "Incorrect eigenvalue: %f\n", eigenvalues[i] );
				res = false;
			}
        }*/
	}

	void CUDA::MakeBlockHessian() {
		/*// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
		// The special 'eigenvalue' will be -1.
		// All other positive values will be equal to the matrix size.
		int size[5];
		size[0] = rand() % 10 + 1;
		size[1] = rand() % 10 + 1;
		size[2] = rand() % 10 + 1;
		size[3] = rand() % 10 + 1;
		size[4] = rand() % 10 + 1;

		int hessiansizes[5];
		int hesssize = 0;
		int numatoms = 0;
		int biggestblock = 0;
		for( int i = 0; i < 5; i++ ) {
			numatoms += size[i];
			hesssize += 3 * size[i] * 3 * size[i];
			hessiansizes[i] = 3 * size[i] * 3 * size[i];
			if( size[i] > biggestblock ) {
				biggestblock = size[i];
			}
		}

		float hessian[hesssize], testhessian[hesssize];
		int blocknums[5], hessiannums[5];

		// Populate hessian
		int pos = 0;
		blocknums[0] = 0;
		hessiannums[0] = 0;
		for( int i = 0; i < 5; i++ ) {
			if( i > 0 ) {
				blocknums[i] = blocknums[i - 1] + size[i - 1];
				hessiannums[i] = hessiannums[i - 1] + 3 * size[i - 1] * 3 * size[i - 1];
			}
		}

		int numBlocks = 5 / 512 + 1;
		int numThreads = 5 % 512;

		float blockDelta = 1e-3;
		float forces1[3 * numatoms], forces2[3 * numatoms], masses[numatoms];

		// Assign forces1 random values between 0 and 1
		for( int i = 0; i < 3 * numatoms; i++ ) {
			forces1[i] = ( float ) rand() / RAND_MAX + 1;
			forces2[i] = ( float ) rand() / RAND_MAX + 1;
		}

		for( int i = 0; i < numatoms; i++ ) {
			masses[i] = ( rand() % 10 ) + ( ( float ) rand() / RAND_MAX + 1 );
		}

		// Compute the full blocks
		// For our test, we'll just assume we are 'perturbing' dof 0
		pos = 0;
		for( int j = 0; j < 5; j++ ) {
			int startdof = 3 * blocknums[j];
			int dof = startdof + 0;
			int enddof;
			if( j == 4 ) {
				enddof = 3 * numatoms - 1;
			} else {
				enddof = 3 * blocknums[j + 1] - 1;
			}
			// The location should be:
			// The starting spot for this hessian, plus:
			// The difference between k and the starting degree of freedom times
			// the y-dimension of this block hessian (sqrt(the total number of elements), plus:
			// The difference between the perturbed degree of freedom and the starting degree of freedom.
			for( int k = startdof; k <= enddof; k++ ) {
				testhessian[hessiannums[j] + ( k - startdof ) * 3 * size[j] + ( dof - startdof )] = ( forces1[k] - forces2[k] ) * ( 1.0 / ( blockDelta * sqrt( masses[k / 3] * masses[dof / 3] ) ) );
			}
		}



		float *gpu_hess;
		float *gpu_force1;
		float *gpu_force2;
		float *gpu_mass;
		int *gpu_bnums;
		int *gpu_bsizes;
		int *gpu_hnums;
		int *gpu_hsizes;

		cudaMalloc( &gpu_hess, hesssize * sizeof( float ) );
		cudaMalloc( &gpu_force1, 3 * numatoms * sizeof( float ) );
		cudaMalloc( &gpu_force2, 3 * numatoms * sizeof( float ) );
		cudaMalloc( &gpu_mass, numatoms * sizeof( float ) );
		cudaMalloc( &gpu_bnums, 5 * sizeof( int ) );
		cudaMalloc( &gpu_bsizes, 5 * sizeof( int ) );
		cudaMalloc( &gpu_hnums, 5 * sizeof( int ) );
		cudaMalloc( &gpu_hsizes, 5 * sizeof( int ) );

		//cudaMemcpy(gpu_hess, testhessian, hesssize*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy( gpu_force1, forces1, 3 * numatoms * sizeof( float ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_force2, forces2, 3 * numatoms * sizeof( float ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_mass, masses, numatoms * sizeof( float ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_bnums, blocknums, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_bsizes, size, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_hnums, hessiannums, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( gpu_hsizes, hessiansizes, 5 * sizeof( int ), cudaMemcpyHostToDevice );

		int i = 0; // Assume degree of freedom 0 for now
		makeBlockHessian <<< numBlocks, numThreads>>>( gpu_hess, gpu_force1, gpu_force2, gpu_mass, blockDelta, gpu_bnums, gpu_bsizes, 5, gpu_hnums, gpu_hsizes, i, numatoms );

		cudaMemcpy( hessian, gpu_hess, hesssize * sizeof( float ), cudaMemcpyDeviceToHost );

		cudaFree( gpu_hess );
		cudaFree( gpu_force1 );
		cudaFree( gpu_force2 );
		cudaFree( gpu_mass );
		cudaFree( gpu_bnums );
		cudaFree( gpu_bsizes );
		cudaFree( gpu_hnums );
		cudaFree( gpu_hsizes );

		bool res = true;
		for( int j = 0; j < 5; j++ ) {
			int startdof = 3 * blocknums[j];
			int dof = startdof + 0;
			int enddof;
			if( j == 4 ) {
				enddof = 3 * numatoms - 1;
			} else {
				enddof = 3 * blocknums[j + 1] - 1;
			}
			// The location should be:
			// The starting spot for this hessian, plus:
			// The difference between k and the starting degree of freedom times
			// the y-dimension of this block hessian (sqrt(the total number of elements), plus:
			// The difference between the perturbed degree of freedom and the starting degree of freedom.
			for( int k = startdof; k <= enddof; k++ ) {
				int d = hessiannums[j] + ( k - startdof ) * 3 * size[j] + ( dof - startdof );
				if( fabs( hessian[d] - testhessian[d] ) > 0.0001 ) {
					printf( "Element %d got %f expected %f\n", d, hessian[d], testhessian[d] );
					res = false;
				}
			}
		}*/
	}

	void CUDA::PerturbPositions() {
		/*// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
		// The special 'eigenvalue' will be -1.
		// All other positive values will be equal to the matrix size.
		int size[5];
		size[0] = rand() % 10 + 1;
		size[1] = rand() % 10 + 1;
		size[2] = rand() % 10 + 1;
		size[3] = rand() % 10 + 1;
		size[4] = rand() % 10 + 1;

		int numatoms = 0;
		int biggestblock = 0;
		for( int i = 0; i < 5; i++ ) {
			numatoms += size[i];
			if( size[i] > biggestblock ) {
				biggestblock = size[i];
			}
		}



		// Perturb the test
		int *blocnums = ( int * ) malloc( 5 * sizeof( int ) );
		// Populate hessian
		blocnums[0] = 0;
		for( int i = 0; i < 5; i++ ) {
			if( i > 0 ) {
				blocnums[i] = blocnums[i - 1] + size[i - 1];
			}
		}

		float *testpos = ( float * )malloc( 3 * numatoms * sizeof( float ) );
		float4 *pos = ( float4 * )malloc( numatoms * sizeof( float4 ) );
		for( int i = 0; i < numatoms; i++ ) {
			testpos[3 * i] = pos[i].x = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			testpos[3 * i + 1] = pos[i].y = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			testpos[3 * i + 2] = pos[i].z = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
		}

		float *gpu_posorig;
		cudaMalloc( &gpu_posorig, 3 * numatoms * sizeof( float ) );
		cudaMemcpy( gpu_posorig, testpos, 3 * numatoms * sizeof( float ), cudaMemcpyHostToDevice );

		float delt = ( float ) rand() / RAND_MAX + 1;
		for( int j = 0; j < 5; j++ ) {
			int startspot = 3 * blocnums[j];
			testpos[startspot + 0] += delt;
		}
		float4 *gpu_pos;
		cudaMalloc( &gpu_pos, numatoms * sizeof( float4 ) );
		cudaMemcpy( gpu_pos, pos, numatoms * sizeof( float4 ), cudaMemcpyHostToDevice );
		int *gpu_blocnums;
		cudaMalloc( &gpu_blocnums, 5 * sizeof( int ) );
		cudaMemcpy( gpu_blocnums, blocnums, 5 * sizeof( int ), cudaMemcpyHostToDevice );
		perturbPositions <<< 1, 5>>>( gpu_posorig, gpu_pos, delt, gpu_blocnums, 5, 0, numatoms );
		float4 *check3 = ( float4 * )malloc( numatoms * sizeof( float4 ) );
		cudaMemcpy( check3, gpu_pos, numatoms * sizeof( float4 ), cudaMemcpyDeviceToHost );
		float *check4 = ( float * )malloc( 3 * numatoms * sizeof( float ) );
		cudaMemcpy( check4, gpu_posorig, 3 * numatoms * sizeof( float ), cudaMemcpyDeviceToHost );
		bool res = true;
		for( int i = 0; i < numatoms; i++ ) {
			if( testpos[3 * i] != check3[i].x ) {
				printf( "3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i], check3[i].x );
				res = false;
			}
			if( testpos[3 * i + 1] != check3[i].y ) {
				printf( "3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i + 1], check3[i].y );
				res = false;
			}
			if( testpos[3 * i + 2] != check3[i].z ) {
				printf( "3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i + 2], check3[i].z );
				res = false;
			}
			if( pos[i].x != check4[3 * i] ) {
				printf( "4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].x, check4[3 * i] );
				res = false;
			}
			if( pos[i].y != check4[3 * i + 1] ) {
				printf( "4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].y, check4[3 * i + 1] );
				res = false;
			}
			if( pos[i].z != check4[3 * i + 2] ) {
				printf( "4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].z, check4[3 * i + 2] );
				res = false;
			}
		}*/
	}

	void CUDA::PerturbByE() {
		/*int numatoms = 1000;
		int m = 500;
		// Create initial positions
		float *testpos = ( float * )malloc( 3 * numatoms * sizeof( float ) );
		float4 *pos = ( float4 * )malloc( numatoms * sizeof( float4 ) );
		for( int i = 0; i < numatoms; i++ ) {
			testpos[3 * i] = pos[i].x = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			testpos[3 * i + 1] = pos[i].y = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			testpos[3 * i + 2] = pos[i].z = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
		}

		// Create delta value
		float delt = ( float ) rand() / RAND_MAX + 1;

		// Create E
		float *E = ( float * ) malloc( m * 3 * numatoms * sizeof( float ) );
		for( int i = 0; i < 3 * numatoms; i++ ) {
			for( int j = 0; j < m; j++ ) {
				E[i * m + j] = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			}
		}

		// Create masses
		float *masses = ( float * ) malloc( numatoms * sizeof( float ) );
		for( int i = 0; i < numatoms; i++ ) {
			masses[i] = ( rand() % 10 ) + ( ( float ) rand() / RAND_MAX + 1 );
		}

		float *gpu_tmppos;
		float4 *gpu_pos;
		float *gpu_E;
		float *gpu_masses;

		cudaMalloc( ( void ** )&gpu_tmppos, 3 * numatoms * sizeof( float ) );
		cudaMalloc( ( void ** )&gpu_pos, numatoms * sizeof( float4 ) );
		cudaMemcpy( gpu_pos, pos, numatoms * sizeof( float4 ), cudaMemcpyHostToDevice );
		cudaMalloc( ( void ** )&gpu_E, m * 3 * numatoms * sizeof( float ) );
		cudaMemcpy( gpu_E, E, m * 3 * numatoms * sizeof( float ), cudaMemcpyHostToDevice );
		cudaMalloc( ( void ** )&gpu_masses, numatoms * sizeof( float ) );
		cudaMemcpy( gpu_masses, masses, numatoms * sizeof( float ), cudaMemcpyHostToDevice );
		bool res = true;
		for( int k = 0; k < m; k++ ) {
			int dof = numatoms * 3;
			int numblocks, numthreads;
			if( dof > 512 ) {
				numblocks = dof / 512 + 1;
				numthreads = 512;
			} else {
				numblocks = 1;
				numthreads = dof;
			}

			perturbByE <<< numblocks, numthreads>>>( gpu_tmppos, gpu_pos, delt, gpu_E, gpu_masses, k, m, 3 * numatoms );

			for( int i = 0; i < numatoms; i++ ) {
				pos[i].x = testpos[3 * i];
				pos[i].y = testpos[3 * i + 1];
				pos[i].z = testpos[3 * i + 2];
				testpos[3 * i] += delt * E[3 * i * m + k] / sqrt( masses[i] );
				testpos[3 * i + 1] += delt * E[( 3 * i + 1 ) * m + k] / sqrt( masses[i] );
				testpos[3 * i + 2] += delt * E[( 3 * i + 2 ) * m + k] / sqrt( masses[i] );
			}

			float4 *check3 = ( float4 * )malloc( numatoms * sizeof( float4 ) );
			cudaMemcpy( check3, gpu_pos, numatoms * sizeof( float4 ), cudaMemcpyDeviceToHost );
			float *check4 = ( float * )malloc( 3 * numatoms * sizeof( float ) );
			cudaMemcpy( check4, gpu_tmppos, 3 * numatoms * sizeof( float ), cudaMemcpyDeviceToHost );
			for( int i = 0; i < numatoms; i++ ) {
				if( fabs( testpos[3 * i] - check3[i].x ) > .01 ) {
					printf( "3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i], check3[i].x );
					res = false;
				}
				if( fabs( testpos[3 * i + 1] - check3[i].y ) > .01 ) {
					printf( "3 Error on y-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i + 1], check3[i].y );
					res = false;
				}
				if( fabs( testpos[3 * i + 2] - check3[i].z ) > .01 ) {
					printf( "3 Error on z-coordinate of atom  %i expected %f got %f\n", i, testpos[3 * i + 2], check3[i].z );
					res = false;
				}
				if( fabs( pos[i].x - check4[3 * i] ) > .01 ) {
					printf( "4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].x, check4[3 * i] );
					res = false;
				}
				if( fabs( pos[i].y - check4[3 * i + 1] ) > .01 ) {
					printf( "4 Error on y-coordinate of atom  %i expected %f got %f\n", i, pos[i].y, check4[3 * i + 1] );
					res = false;
				}
				if( fabs( pos[i].z - check4[3 * i + 2] ) > .01 ) {
					printf( "4 Error on z-coordinate of atom  %i expected %f got %f\n", i, pos[i].z, check4[3 * i + 2] );
					res = false;
				}
			}

			free( check3 );
			free( check4 );
		}*/
	}

	void CUDA::ComputeNormsAndCenter() {
		/*// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
		// The special 'eigenvalue' will be -1.
		// All other positive values will be equal to the matrix size.
		int size[5];
		size[0] = rand() % 10 + 1;
		size[1] = rand() % 10 + 1;
		size[2] = rand() % 10 + 1;
		size[3] = rand() % 10 + 1;
		size[4] = rand() % 10 + 1;

		int numatoms = 0;
		int biggestblock = 0;
		for( int i = 0; i < 5; i++ ) {
			numatoms += size[i];
			if( size[i] > biggestblock ) {
				biggestblock = size[i];
			}
		}

		// Perturb the test
		int blocnums[5];

		// Populate hessian
		blocnums[0] = 0;
		for( int i = 0; i < 5; i++ ) {
			if( i > 0 ) {
				blocnums[i] = blocnums[i - 1] + size[i - 1];
			}
		}

		float4 pos[numatoms];
		for( int i = 0; i < numatoms; i++ ) {
			pos[i].x = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			pos[i].y = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
			pos[i].z = rand() % 10 + ( float ) rand() / RAND_MAX + 1;
		}

		// Create masses
		float masses[numatoms];
		for( int i = 0; i < numatoms; i++ ) {
			masses[i] = ( rand() % 10 ) + ( ( float ) rand() / RAND_MAX + 1 );
		}

		// Centers
        float4 *gpu_pos;
		cudaMalloc( ( void ** ) &gpu_pos, numatoms * sizeof( float4 ) );
		cudaMemcpy( gpu_pos, pos, numatoms * sizeof( float4 ), cudaMemcpyHostToDevice );

        float *gpu_masses;
		cudaMalloc( ( void ** ) &gpu_masses, numatoms * sizeof( float ) );
		cudaMemcpy( gpu_masses, masses, numatoms * sizeof( float ), cudaMemcpyHostToDevice );

        int *gpu_blocknums;
		cudaMalloc( ( void ** ) &gpu_blocknums, 5 * sizeof( int ) );
		cudaMemcpy( gpu_blocknums, blocnums, 5 * sizeof( int ), cudaMemcpyHostToDevice );

        int *gpu_blocksizes;
		cudaMalloc( ( void ** ) &gpu_blocksizes, 5 * sizeof( int ) );
		cudaMemcpy( gpu_blocksizes, size, 5 * sizeof( int ), cudaMemcpyHostToDevice );

		float *gpu_centers;
        cudaMalloc( ( void ** ) &gpu_centers, 5 * 3 * sizeof( float ) );

		float *gpu_norms;
		cudaMalloc( ( void ** ) &gpu_norms, 5 * sizeof( float ) );

		computeNormsAndCenter <<< 1, 5>>>( gpu_norms, gpu_centers, gpu_masses, gpu_pos, gpu_blocknums, gpu_blocksizes );

		float centers[15];
		cudaMemcpy( centers, gpu_centers, 15 * sizeof( float ), cudaMemcpyDeviceToHost );

        float norms[5];
		cudaMemcpy( norms, gpu_norms, 5 * sizeof( float ), cudaMemcpyDeviceToHost );

		float c[15], n[5];
		for( int i = 0; i < 5; i++ ) {
			float totalmass = 0.0;
			for( int j = blocnums[i]; j <= blocnums[i] + size[i] - 1; j += 3 ) {
				float mass = masses[ j / 3 ];
				c[i * 3 + 0] = pos[j / 3].x * mass;
				c[i * 3 + 1] = pos[j / 3].y * mass;
				c[i * 3 + 2] = pos[j / 3].z * mass;
				totalmass += mass;
			}

			n[i] = sqrt( totalmass );
			c[i * 3 + 0] /= totalmass;
			c[i * 3 + 1] /= totalmass;
			c[i * 3 + 2] /= totalmass;
		}

		float *Qi_gdof = ( float * ) malloc( 5 * biggestblock * 6 * sizeof( float ) );
		float *Qi_gdof2 = ( float * ) malloc( 5 * biggestblock * 6 * sizeof( float ) );

		float *gpu_qdof;
		cudaMalloc( ( void ** ) &gpu_qdof, 5 * biggestblock * 6 * sizeof( float ) );

		geometricDOF <<< 1, 5>>>( gpu_qdof, gpu_pos, gpu_masses, gpu_blocknums, gpu_blocksizes, biggestblock, gpu_norms, gpu_centers );
		cudaMemcpy( Qi_gdof, gpu_qdof, 5 * biggestblock * 6 * sizeof( float ), cudaMemcpyDeviceToHost );
		printf( "Testing Geometric DOF....\n" );
		printf( "1\n" );
		cudaFree( gpu_pos );
		printf( "2\n" );
		cudaFree( gpu_masses );
		printf( "3\n" );
		cudaFree( gpu_blocknums );
		printf( "5\n" );
		cudaFree( gpu_centers );
		printf( "6\n" );
		cudaFree( gpu_norms );
		bool result = testGeometricDOF( Qi_gdof, pos, masses, blocnums, size, biggestblock, norms, centers );
		if( result ) {
			printf( "Test GeometricDOF Passed\n" );
		} else {
			printf( "Test GeometricDOF Failed\n" );
		}


		orthogonalize23 <<< 1, 5>>>( gpu_qdof, gpu_blocksizes, 5, biggestblock );
		cudaMemcpy( Qi_gdof2, gpu_qdof, 5 * biggestblock * 6 * sizeof( float ), cudaMemcpyDeviceToHost );
		printf( "Testing Orthogonalize23... \n" );
		result = testOrthogonalize23( Qi_gdof2, Qi_gdof, size, 5, biggestblock );
		if( result ) {
			printf( "Test Orthogonalize23 Passed\n" );
		} else {
			printf( "Test Orthogonalize23 Failed\n" );
		}

		printf( "4\n" );
		cudaFree( gpu_blocksizes );
		printf( "7\n" );
		cudaFree( gpu_qdof );
		printf( "8\n" );
		free( Qi_gdof );
		printf( "9\n" );
		free( blocnums );
		printf( "10\n" );
		free( pos );
		printf( "11\n" );
		free( masses );
		printf( "12\n" );
		bool res = true;
		for( int i = 0; i < 5; i++ ) {
			for( int j = 0; j < 3; j++ ) {
				if( fabs( c[i * 3 + j] - centers[i * 3 + j] ) > 0.01 ) {
					printf( "Error in centers (%d, %d): expected %f got %f", i, j, c[i * 3 + j], centers[i * 3 + j] );
					res = false;
				}
			}
			if( fabs( n[i] - norms[i] ) > 0.01 ) {
				printf( "Error in norms (%d): expected %f got %f", i, n[i], norms[i] );
				res = false;
			}
		}*/
	}

	void CUDA::MakeHE() {
		/*int numatoms = 100;
		int m = 50;

		// Create masses
		float masse[numatoms];
		for( int i = 0; i < numatoms; i++ ) {
			masses[i] = ( rand() % 10 ) + ( ( float ) rand() / RAND_MAX + 1 );
		}


		float *force1[3 * numatoms];
		float4 *force2[numatoms];

		// Assign forces1 random values between 0 and 1
		for( int i = 0; i < numatoms; i++ ) {
			force1[3 * i] = ( float ) rand() / RAND_MAX + 1;
			force1[3 * i + 1] = ( float ) rand() / RAND_MAX + 1;
			force1[3 * i + 2] = ( float ) rand() / RAND_MAX + 1;
			force2[i].x = ( float ) rand() / RAND_MAX + 1;
			while( force2[i].x == force1[3 * i] ) {
				force2[i].x = ( float ) rand() / RAND_MAX + 1;
			}
			force2[i].y = ( float ) rand() / RAND_MAX + 1;
			while( force2[i].y == force1[3 * i + 1] ) {
				force2[i].y = ( float ) rand() / RAND_MAX + 1;
			}
			force2[i].z = ( float ) rand() / RAND_MAX + 1;
			while( force2[i].z == force1[3 * i + 2] ) {
				force2[i].z = ( float ) rand() / RAND_MAX + 1;
			}
		}

		float delt = ( float ) rand() / RAND_MAX + 1;

        float *gpu_HE;
		cudaMalloc( ( void ** ) &gpu_HE, 3 * numatoms * m * sizeof( float ) );

        float *gpu_force1;
		cudaMalloc( ( void ** ) &gpu_force1, 3 * numatoms * sizeof( float ) );
		cudaMemcpy( gpu_force1, force1, 3 * numatoms * sizeof( float ), cudaMemcpyHostToDevice );

        float4 *gpu_force2;
		cudaMalloc( ( void ** ) &gpu_force2, numatoms * sizeof( float4 ) );
		cudaMemcpy( gpu_force2, force2, numatoms * sizeof( float4 ), cudaMemcpyHostToDevice );

        float *gpu_masses;
		cudaMalloc( ( void ** ) &gpu_masses, numatoms * sizeof( float ) );
		cudaMemcpy( gpu_masses, masses, numatoms * sizeof( float ), cudaMemcpyHostToDevice );


		float HE[3 * numatoms * m];
		for( int k = 0; k < m; k++ ) {
			int dof = numatoms * 3;
			int numblocks, numthreads;
			if( dof > 512 ) {
				numblocks = dof / 512 + 1;
				numthreads = 512;
			} else {
				numblocks = 1;
				numthreads = dof;
			}
			makeHE <<< numblocks, numthreads>>>( gpu_HE, gpu_force1, gpu_force2, gpu_masses, delt, k, m, 3 * numatoms );
			cudaMemcpy( HE, gpu_HE, 3 * numatoms * m * sizeof( float ), cudaMemcpyDeviceToHost );
			for( int i = 0; i < numatoms; i++ ) {
				float hex = ( force1[3 * i] - force2[i].x ) / ( sqrt( masses[i] ) * 1.0 * delt );
				float hey = ( force1[3 * i + 1] - force2[i].y ) / ( sqrt( masses[i] ) * 1.0 * delt );
				float hez = ( force1[3 * i + 2] - force2[i].z ) / ( sqrt( masses[i] ) * 1.0 * delt );

				if( fabs( HE[3 * i * m + k] - hex ) > 0.01 ) {
					printf( "%f %f", force1[3 * i], force2[i].x );
					printf( "Error in HE X (%d, %d): got %f expected %f\n", i, k, HE[3 * i * m + k], hex );
					//return false;
				}
				if( fabs( HE[( 3 * i + 1 )*m + k] - hey ) > 0.01 ) {
					printf( "%f %f", force1[3 * i + 1], force2[i].y );
					printf( "Error in HE Y (%d, %d): got %f expected %f\n", i, k, HE[( 3 * i + 1 )*m + k], hey );
					//return false;
				}
				if( fabs( HE[( 3 * i + 2 )*m + k] - hez ) > 0.01 ) {
					printf( "%f %f", force1[3 * i + 2], force2[i].z );
					printf( "Error in HE Z (%d, %d): got %f expected %f\n", i, k, HE[( 3 * i + 2 )*m + k], hez );
					//return false;
				}
			}
		}*/
	}
}
