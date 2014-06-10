#include "OpenMMFBM/FBMCuda.h"
#include "kernels/Kernels.cu"
#include "kernels/qr.cu"
//#include <cuda.h>
#include "OpenMMFBM/CudaFBMKernelSources.h"
#include "magma_d.h"
//#include "cula_lapack_device.h"
//#include "cula.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
//#include "gputypes.h"
#include "OpenMM.h"
using namespace OpenMM;

#include <iostream>
using namespace std;

#define MAXTHREADSPERBLOCK 512
#define CONSERVEDDEGREESOFFREEDOM 6

void FBMCuda::getBlockHessian( std::vector<std::vector<double > > &blockHessianVectors ) const {
	blockHessianVectors.resize( _3N, std::vector<double>( _3N, 0.0 ) );

	int size = 0;
	int enddof;
	for( int i = 0; i < numHessBlocks; i++ ) {
		const int startdof = blocknums[i];
		if( i == numHessBlocks - 1 ) {
			enddof = _3N;
		} else {
			enddof = blocknums[i + 1];
		}
		size += ( enddof - startdof ) * ( enddof - startdof );
	}

	vector<float> buffer( size, 0.0f );

	cudaMemcpy( &buffer[0], blockHessian, size, cudaMemcpyDeviceToHost );

	int pos = 0;
	for( int i = 0; i < numHessBlocks; i++ ) {
		const int startdof = blocknums[i];
		if( i == numHessBlocks - 1 ) {
			enddof = _3N;
		} else {
			enddof = blocknums[i + 1];
		}
		for( int j = startdof; j < enddof; j++ ) {
			for( int k = startdof; k < enddof; k++ ) {
				blockHessianVectors[j][k] = buffer[pos];
				pos++;
			}
		}
	}
}

void FBMCuda::getBlockEigenvectors( std::vector<std::vector<double > > &blockVectors ) const {
	blockVectors.resize( _3N, std::vector<double>( _3N, 0.0 ) );
	vector<float> buffer( _3N, 0.0f );

	for( unsigned int i = 0; i < _3N; i++ ) {
		cudaMemcpy( &buffer[0], &blockEigenvectors[_3N + i], _3N, cudaMemcpyDeviceToHost );
		for( unsigned int j = 0; j < _3N; j++ ) {
			blockVectors[j][i] = buffer[j];
		}
	}
}

void FBMCuda::getProjectionMatrix( std::vector<std::vector<double > > &projMatrix ) const {
	projMatrix.resize( _3N, std::vector<double>( m, 0.0 ) );
	vector<float> buffer( _3N, 0.0f );

	for( unsigned int i = 0; i < m; i++ ) {
		cudaMemcpy( &buffer[0], &E[_3N + i], _3N, cudaMemcpyDeviceToHost );
		for( unsigned int j = 0; j < _3N; j++ ) {
			projMatrix[j][i] = buffer[j];
		}
	}
}

void FBMCuda::getHE( std::vector<std::vector<double > > &HEout ) const {
	HEout.resize( _3N, std::vector<double>( m, 0.0 ) );
	vector<float> buffer( _3N, 0.0f );

	for( unsigned int i = 0; i < m; i++ ) {
		cudaMemcpy( &buffer[0], &HE[_3N + i], _3N, cudaMemcpyDeviceToHost );
		for( unsigned int j = 0; j < _3N; j++ ) {
			HEout[j][i] = buffer[j];
		}
	}
}

void FBMCuda::getCoarseGrainedHessian( std::vector<std::vector<double > > &coarseGrainedHessian ) const {
	coarseGrainedHessian.resize( m, std::vector<double>( m, 0.0 ) );
	vector<float> buffer( m, 0.0f );

	for( unsigned int i = 0; i < m; i++ ) {
		cudaMemcpy( &buffer[0], &S[m + i], m, cudaMemcpyDeviceToHost );
		for( unsigned int j = 0; j < m; j++ ) {
			coarseGrainedHessian[j][i] = buffer[j];
		}
	}
}


FBMCuda::FBMCuda( Context &c, Context &bC, OpenMMFBM::FBMParameters &p ) : FBMAbstract( c, bC, p ) {
	// These will give us access to GPU pointers
	data = reinterpret_cast<CudaPlatform::PlatformData *>( getContextImpl( c ).getPlatformData() );
	blockData = reinterpret_cast<CudaPlatform::PlatformData *>( getContextImpl( bC ).getPlatformData() );

	// Set up the masses...
	System &system = c.getSystem();
	int _N = context.getState( State::Positions ).getPositions().size() / 3;
	float *tmpmass = new float[_N];
	for( int i = 0; i < _N; i++ ) {
		tmpmass[i] = system.getParticleMass( i );
	}
	cudaMalloc( ( void ** ) &masses, _N * sizeof( float ) );
	cudaMemcpy( masses, tmpmass, _N * sizeof( float ), cudaMemcpyHostToDevice );
	delete tmpmass;
}

void FBMCuda::makeBlocksAndThreads( int count ) {
	if( count <= MAXTHREADSPERBLOCK ) {
		numBlocks = 1;
		numThreads = count;
	} else {
		numBlocks = count / MAXTHREADSPERBLOCK + 1;
		numThreads = MAXTHREADSPERBLOCK;
	}
}

void FBMCuda::formBlocks() {
	// Make the blocks and put on GPU
	int block_start = 0;
	largestBlockSize = 0;
	vector<int> blocks;
	vector<int> blocksize;
	vector<int> hessiannum;
	vector<int> hessiansize;
	for( int i = 0; i < params.residue_sizes.size(); i++ ) {
		if( i % params.res_per_block == 0 ) {
			blocks.push_back( block_start );
			hessiannum.push_back( block_start * block_start );
			if( hessiannum.size() != 1 ) {
				hessiansize.push_back( hessiannum[hessiannum.size() - 1] );
			}
		}
		block_start += params.residue_sizes[i];
	}
	hessiansize.push_back( _3N - hessiannum[hessiannum.size() - 1] );

	cout << "block sizes " << blocks[0] << " ";
	for( int i = 1; i < blocks.size(); i++ ) {
		cout << blocks[i] << " ";
		int block_size = blocks[i] - blocks[i - 1];
		blocksize.push_back( block_size );
		if( block_size > largestBlockSize ) {
			largestBlockSize = block_size;
		}
	}
	cout << endl;
	numHessBlocks = blocks.size();
	largestBlockSize *= 3;
	cout << "Allocating memory" << endl;
	cudaMalloc( ( void ** ) &blocknums, blocks.size()*sizeof( int ) );
	cudaMalloc( ( void ** ) &blocksizes, blocks.size()*sizeof( int ) );
	cudaMalloc( ( void ** ) &hessiannums, blocks.size()*sizeof( int ) );
	cudaMalloc( ( void ** ) &hessiansizes, blocks.size()*sizeof( int ) );
	cudaMemcpy( blocknums, &blocks[0], blocks.size()*sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( blocksizes, &blocksize[0], blocks.size()*sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( hessiannums, &hessiannum[0], blocks.size()*sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( hessiansizes, &hessiansize[0], blocks.size()*sizeof( int ), cudaMemcpyHostToDevice );


	// Allocate a block Hessian on the GPU
	State state = context.getState( State::Positions | State::Forces );
	_3N = state.getPositions().size();
	int _N = _3N / 3;
	//cudaMalloc( (void**) &blockHessian, _3N*_3N*sizeof(float));
	// The total number of elements in a linear block Hessian will be the sum of the squares of the block sizes
	int numelements = 0;
	for( int i = 0; i < blocksize.size(); i++ ) {
		numelements += blocksize[i] * blocksize[i];
	}
	cudaMalloc( ( void ** ) &blockHessian, numelements * sizeof( float ) );

	// Temporary buffers to hold positions and forces
	CudaArray *pos1 = new CudaArray( *( blockData->contexts[0] ), _N, sizeof( float4 ), "PerturbedP" );
	//float* pos1;
	CudaArray *force1 = new CudaArray( *( blockData->contexts[0] ), _N, sizeof( float4 ), "PerturbedF" );
//float* force1;
	//cudaMalloc( (void**) &pos1, _3N*sizeof(float));
	cudaMalloc( ( void ** ) &force1, _3N * sizeof( float ) );

	// GPU Thread organization...
	makeBlocksAndThreads( _N );

	cout << "Computing hessian" << endl;

	float4 *positionArray;
	float4 myfloats;
	myfloats.x = 1.0;
	myfloats.y = 0.0;
	myfloats.z = 0.0;
	cudaMalloc( ( void ** ) &positionArray, _N * sizeof( float4 ) );
	vector<float4> buffer( _N, myfloats );
	cudaMemcpy( positionArray, &buffer[0], _N * sizeof( float4 ), cudaMemcpyHostToDevice );

	blockContext.getState( State::Positions | State::Forces ).getForces();

	ContextImpl impl = getContextImpl( blockContext );

	cout << "got Impl" << endl;

	impl.getTime();

	cout << "called impl positions" << endl;

	CudaPlatform::PlatformData *blockData = ( CudaPlatform::PlatformData * ) impl.getPlatformData();

	if( !blockData ) {
		cout << "Block data is NULL" << endl;
	} else {
		cout << "Block data is not NULL!" << endl;
	}

	vector<Vec3> positionsCheck( _N );
	impl.getPositions( positionsCheck );

	cout << "Got positions" << endl;

	cout << positionsCheck[0][0] << " " << positionsCheck[0][1] << endl;

	vector<Vec3> try4 = blockContext.getState( State::Positions ).getPositions();

	cout << try4[0][0] << " " << try4[0][1] << endl;

	/*if(blockData->gpu)
	cout << "GPU data not NULL!";
	else
	cout << "GPU data is NULL!";

	if(blockData->gpu->psPosq4)
	cout << "CUDAStream is not NULL!";
	else
	cout << "CUDAStream is NULL!";

	blockData->gpu->psPosq4->Upload();

	float4* positions = blockData->gpu->psPosq4->_pDevData;  */

	// Populate the Hessian
	numelements = 0;
	CudaContext *cu = blockData->contexts[0];
	CUmodule perturbModule = cu->createModule( CudaFBMKernelSources::perturb );
	CUfunction perturbKernel = cu->getKernel( perturbModule, "perturbPositions" );
	CUmodule copyModule = cu->createModule( CudaFBMKernelSources::copy );
	CUfunction copyFromKernel = cu->getKernel( copyModule, "copyFromOpenMM" );
	CUfunction blockCopyFromKernel = cu->getKernel( copyModule, "blockCopyFromOpenMM" );
	CUfunction copyToKernel = cu->getKernel( copyModule, "copyToOpenMM" );
	CUfunction blockCopyToKernel = cu->getKernel( copyModule, "blockCopyToOpenMM" );
	CUmodule blockHessModule = cu->createModule( CudaFBMKernelSources::blockHess );
	CUfunction blockHessKernel = cu->getKernel( blockHessModule, "makeBlockHessian" );
	CUmodule symmetrizeModule = cu->createModule( CudaFBMKernelSources::symmetrize );
	CUfunction symmetrize1DKernel = cu->getKernel( symmetrizeModule, "symmetrize1D" );

	for( unsigned int i = 0; i < largestBlockSize; i++ ) {
		cout << "current dof " << i << endl;
		// Perturb in the (+) direction
		// After this, pos1 will hold the old positions and the positions pointer
		// has been directly perturbed
		int numblocks = blocks.size();
		void *perturbArgs[] = {&pos1->getDevicePointer(),
							   &blockData->contexts[0]->getPosq().getDevicePointer(),
							   &params.blockDelta,
							   &blocknums,
							   &numblocks,
							   &i,
							   &_N
							  };
		cu->executeKernel( perturbKernel, perturbArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//perturbPositions<<<numBlocks, numThreads>>>(pos1, (void*) &blockData->contexts[0]->getPosq().getDevicePointer(), params.blockDelta, blocknums, blocks.size(), i, _N);

		cout << " Calling forces" << endl;
		// Compute forces
		blockContext.getState( State::Forces ).getForces();


		// Save first force vector
		void *copyFromArgs[] = {&force1->getDevicePointer(),
								&data->contexts[0]->getForce().getDevicePointer(),
								&_3N
							   };
		void *blockCopyFromArgs[] = {&force1->getDevicePointer(),
									 &blockData->contexts[0]->getForce().getDevicePointer(),
									 &blocknums,
									 &numblocks,
									 &i,
									 &_N
									};
		cu->executeKernel( copyFromKernel, copyFromArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		cu->executeKernel( blockCopyFromKernel, blockCopyFromArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//copyFromOpenMM<<<numBlocks, numThreads>>>(force1, &((*(data->gpu->psForce4))[0].w), _3N);
		//blockcopyFromOpenMM<<<numBlocks, numThreads>>>(force1, &((*(blockData->gpu->psPosq4))[0].w), blocknums, blocks.size(), i, _N);

		// Copy back positions
		void *blockCopyToArgs[] = {&blockData->contexts[0]->getPosq().getDevicePointer(),
								   &pos1->getDevicePointer(),
								   &blocknums,
								   &numblocks,
								   &i,
								   &_N
								  };
		cu->executeKernel( blockCopyToKernel, blockCopyToArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//blockcopyToOpenMM<<<numBlocks, numThreads>>>(&((*(blockData->gpu->psPosq4))[0].w), pos1, blocknums, blocks.size(), i, _N);

		// Perturb in the the (-) direction
		int negDelta = -params.blockDelta;
		perturbArgs[2] = &negDelta;
		cu->executeKernel( perturbKernel, perturbArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//perturbPositions<<<numBlocks, numThreads>>>(pos1, &((*(blockData->gpu->psPosq4))[0].w), -(params.blockDelta), blocknums, blocks.size(), i, _N);

		// Compute forces
		blockContext.getState( State::Forces ).getForces();

		// Allocate GPU memory for the Hessian
		// TMC THIS HAS A BUG
		// You have to determine that last parameter (starting spot) inside the function by passing block sizes.
		// It is not determined by degree of freedom
		// So I believe the second call is right, not the first

		void *blockHessArgs[] = {&blockHessian,
								 &force1->getDevicePointer(),
								 &blockData->contexts[0]->getForce().getDevicePointer(),
								 &masses,
								 &params.blockDelta,
								 &blocknums,
								 &blocksizes,
								 &numblocks,
								 &i,
								 &_N
								};

		cu->executeKernel( blockHessKernel, blockHessArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//makeBlockHessian<<<numBlocks, numThreads>>>(blockHessian, force1, &((*(blockData->gpu->psForce4))[0].w), masses, params.blockDelta, blocknums, blocks.size(), i, _N, numelements);
		//makeBlockHessian<<<numBlocks, numThreads>>>(blockHessian, force1, &((*(blockData->gpu->psForce4))[0].w), masses, params.blockDelta, blocknums, blocksizes, blocks.size(), i, _N);

		numelements += blocksize[i] * blocksize[i];
	}

	cout << "computed hessian" << endl;

	// Symmetrize the Hessian
	makeBlocksAndThreads( _3N * _3N );
	void *symmArgs[] = {&blockHessian, &blockData->contexts[0]->getPosq().getDevicePointer(), &blocksize, &_3N};
	cu->executeKernel( symmetrize1DKernel, symmArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
	//symmetrize1D<<<numBlocks, numThreads>>>(blockHessian, positions, blocksize, _3N);
}


void FBMCuda::diagonalizeBlocks() {
	CudaContext *cu = blockData->contexts[0];
// The QR code expects the blocks to be placed in a linear array
	// Note: It also assumes that the number of matricies to diagonalize will be
	// smaller than MAXBLOCKSPERTHREAD
	// I think this will mostly be the case anyway...
	float *tmp;
	cudaMalloc( ( void ** ) &blockEigenvalues, _3N * sizeof( float ) );
	cudaMalloc( ( void ** ) &tmp, _3N * _3N * sizeof( float ) );
	cudaMalloc( ( void ** ) &blockEigenvectors, _3N * sizeof( float ) );

	int one = 1;
	float eps = 1e-8;
	CUmodule qrModule = cu->createModule( CudaFBMKernelSources::qr );
	CUfunction blockQRKernel = cu->getKernel( qrModule, "block_QR" );
	CUmodule eigenModule = cu->createModule( CudaFBMKernelSources::eigen );
	CUfunction eigenKernel = cu->getKernel( eigenModule, "makeEigenvalues" );
	CUmodule sortModule = cu->createModule( CudaFBMKernelSources::sort );
	CUfunction sortKernel = cu->getKernel( sortModule, "blockEigSort" );
	CUmodule dofModule = cu->createModule( CudaFBMKernelSources::dof );
	CUfunction normKernel = cu->getKernel( dofModule, "computeNormsAndCenter" );
	CUfunction dofKernel = cu->getKernel( dofModule, "geometricDOF" );
	CUmodule orthoModule = cu->createModule( CudaFBMKernelSources::orthogonalize );
	CUfunction orthoKernel = cu->getKernel( orthoModule, "orthogonalize" );
	CUfunction ortho23Kernel = cu->getKernel( orthoModule, "orthogonalize23" );


	void *qrArgs[] = {&numHessBlocks,
					  &blockHessian,
					  &hessiannums,
					  &hessiansizes,
					  &tmp,
					  &hessiannums,
					  &one,
					  &eps
					 };
	cu->executeKernel( blockQRKernel, qrArgs, 1, numHessBlocks );
	//block_QR<<<1, numHessBlocks>>>(numHessBlocks, blockHessian, hessiannums, hessiansizes, tmp, hessiannums, 1, 1e-8);

	// Transfer elements from the diagonal of the block Hessian to the block Eigenvalues
	// blockEigenvectors should be good
	makeBlocksAndThreads( _3N );
	int _N = _3N / 3;
	void *eigenArgs[] = {&blockEigenvalues,
						 &blockHessian,
						 &blocknums,
						 &blocksizes,
						 &hessiannums,
						 &_N,
						 &numHessBlocks
						};
	cu->executeKernel( eigenKernel, eigenArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
	//makeEigenvalues<<<numBlocks,numThreads>>>(blockEigenvalues, blockHessian, blocknums, blocksizes, hessiannums, _3N/3, numHessBlocks);

	// Sort eigenvectors within each block
	makeBlocksAndThreads( numHessBlocks );
	void *sortArgs[] = {&blockEigenvalues, &blockEigenvectors, &blocknums, &hessiansizes, &_N};
	cu->executeKernel( sortKernel, sortArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
	//blockEigSort<<<numBlocks, numThreads>>>(blockEigenvalues, blockEigenvectors, blocknums, hessiansizes, _3N/3);

	// LEFT OFF HERE
	// Compute geometric degrees of freedom
	float *norms;
	float *poscenter;
	cudaMalloc( ( void ** ) &norms, numHessBlocks * sizeof( float ) );
	cudaMalloc( ( void ** ) &poscenter, numHessBlocks * 3 * sizeof( float ) );

	void *normsArgs[] = {&norms,
						 &poscenter,
						 &masses,
						 &blockData->contexts[0]->getPosq().getDevicePointer(),
						 &blocknums,
						 &blocksizes
						};
	cu->executeKernel( normKernel, normsArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
	//computeNormsAndCenter<<<numBlocks, numThreads>>>(norms, poscenter, masses, &((*(blockData->gpu->psPosq4))[0].w), blocknums, blocksizes);

	float *Qi_gdof;
	cudaMalloc( ( void ** ) &Qi_gdof, numHessBlocks * largestBlockSize * largestBlockSize );

	void *dofArgs[] = {&Qi_gdof,
					   &blockData->contexts[0]->getPosq().getDevicePointer(),
					   &masses, &blocknums, &blocksizes, &norms, &poscenter
					  };
	cu->executeKernel( dofKernel, dofArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	//geometricDOF<<<numBlocks, numThreads>>>(Qi_gdof, &((*(blockData->gpu->psPosq4))[0].w), masses, blocknums, blocksizes, norms, poscenter);

	void *ortho23Args[] = {&Qi_gdof, &blocksizes};
	int cdof = CONSERVEDDEGREESOFFREEDOM;
	cu->executeKernel( ortho23Kernel, ortho23Args, numHessBlocks, CONSERVEDDEGREESOFFREEDOM - 4 );
	void *orthoArgs[] = {&blockEigenvectors, &Qi_gdof, &cdof, &blocksizes, &blocknums};
	cu->executeKernel( orthoKernel, orthoArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	//orthogonalize23<<<numHessBlocks, CONSERVEDDEGREESOFFREEDOM-4>>>(Qi_gdof, blocksizes);
	//orthogonalize<<<numBlocks, numThreads>>>(blockEigenvectors, Qi_gdof, CONSERVEDDEGREESOFFREEDOM, blocksizes, blocknums);
}


void FBMCuda::formProjectionMatrix() {
	CudaContext *cu = blockData->contexts[0];
// Sort eigenvalues
	int oddcount = _3N / 2;
	int evencount;
	if( _3N % 2 == 0 ) {
		evencount = oddcount + 1;
	} else {
		evencount = oddcount;
	}

	CUmodule sortModule = cu->createModule( CudaFBMKernelSources::sort );
	CUfunction oddevenKernel = cu->getKernel( sortModule, "oddEvenEigSort" );

	int zero = 0;
	int one = 1;
	void *oddevenArgs[] = {&blockEigenvalues, &blockEigenvectors, &zero};
	for( int i = 0; i < ceil( _3N / 2 ); i++ ) {
		makeBlocksAndThreads( evencount );
		cu->executeKernel( oddevenKernel, oddevenArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//oddEvenEigSort<<<numBlocks, numThreads>>>(blockEigenvalues, blockEigenvectors);
		makeBlocksAndThreads( oddcount );
		oddevenArgs[2] = &one;
		cu->executeKernel( oddevenKernel, oddevenArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//oddEvenEigSort<<<numBlocks, numThreads>>>(blockEigenvalues, blockEigenvectors, 1);
		oddevenArgs[2] = &zero;
	}

	// Copy the eigenvalues back and compute m (no way to parallelize that I can see)
	float *tmpeig = new float[_3N];
	cudaMemcpy( tmpeig, blockEigenvalues, _3N * sizeof( float ), cudaMemcpyDeviceToHost );
	int max_eigs = params.bdof * numHessBlocks;
	float cutEigen = blockEigenvalues[max_eigs];
	vector<int> indices;
	int m = 0;
	for( int i = 0; i < _3N; i++ ) {
		if( blockEigenvalues[i] < cutEigen ) {
			indices.push_back( i );
		}
		m++;
	}

	// Now we form the m X n matrix
	int *index;
	makeBlocksAndThreads( m * _3N );
	cudaMalloc( ( void ** ) &Et, m * _3N * sizeof( float ) );
	cudaMalloc( ( void ** ) &E, _3N * m * sizeof( float ) );
	cudaMalloc( ( void ** ) &index, indices.size()*sizeof( int ) );
	cudaMemcpy( index, &indices[0], indices.size()*sizeof( int ), cudaMemcpyHostToDevice );

	CUmodule projectModule = cu->createModule( CudaFBMKernelSources::project );
	CUfunction projectKernel = cu->getKernel( projectModule, "makeProjection" );
	void *projectArgs[] = {&Et, &E, &blockEigenvectors, &index, &_3N};
	cu->executeKernel( projectKernel, projectArgs, blockData->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
	//makeProjection<<<numBlocks, numThreads>>>(Et, E, blockEigenvectors, index, _3N);
}

void FBMCuda::computeHE() {
	CudaContext *cu = data->contexts[0];
	// Temporary buffers to hold positions and forces
	float *pos1;
	float *force1;
	cudaMalloc( ( void ** ) &pos1, _3N * sizeof( float ) );
	cudaMalloc( ( void ** ) &force1, _3N * sizeof( float ) );

	CUmodule perturbModule = cu->createModule( CudaFBMKernelSources::perturb );
	CUfunction perturbKernel = cu->getKernel( perturbModule, "perturbPositions" );
	CUmodule copyModule = cu->createModule( CudaFBMKernelSources::copy );
	CUfunction copyFromKernel = cu->getKernel( copyModule, "copyFromOpenMM" );
	CUfunction copyToKernel = cu->getKernel( copyModule, "copyToOpenMM" );
	CUmodule projectModule = cu->createModule( CudaFBMKernelSources::project );
	CUfunction projectKernel = cu->getKernel( projectModule, "makeHE" );

	void *copyFromArgs[] = {&force1, &data->contexts[0]->getPosq().getDevicePointer(), &_3N};
	void *copyToArgs[] = {&data->contexts[0]->getPosq().getDevicePointer(), &pos1, &_3N};

	for( int k = 0; k < m; k++ ) {
		// Peturb positions, +
		makeBlocksAndThreads( _3N );

		// Perturb in the positive direction
		void *perturbArgs[] = {&pos1, &data->contexts[0]->getPosq().getDevicePointer(), &params.sDelta, &E, &masses, &k, &_3N};
		cu->executeKernel( perturbKernel, perturbArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//perturbByE<<<numBlocks, numThreads>>>(pos1, &((*(data->gpu->psPosq4))[0].w), params.sDelta, E, masses, k, _3N);

		// Calculate Forces
		context.getState( State::Forces ).getForces();

		// Save first force vector
		cu->executeKernel( copyFromKernel, copyFromArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//copyFromOpenMM<<<numBlocks, numThreads>>>(force1, &((*(data->gpu->psForce4))[0].w), _3N);

		// Copy back positions
		cu->executeKernel( copyToKernel, copyToArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//copyToOpenMM<<<numBlocks, numThreads>>>(&((*(data->gpu->psPosq4))[0].w), pos1, _3N);

		// Perturb in the the (-) direction
		int negdelta = -params.sDelta;
		perturbArgs[2] = &negdelta;
		cu->executeKernel( perturbKernel, perturbArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//perturbByE<<<numBlocks, numThreads>>>(pos1, &((*(data->gpu->psPosq4))[0].w), -params.sDelta, E, masses, k, _3N);
		perturbArgs[2] = &params.sDelta;

		// Calculate Forces
		context.getState( State::Forces ).getForces();

		// Make HE
		cudaMalloc( ( void ** ) &HE, _3N * m * sizeof( float ) );
		void *projectArgs[] = {&HE, &force1, &data->contexts[0]->getForce().getDevicePointer(), &masses, &params.sDelta, &k, &_3N};
		cu->executeKernel( projectKernel, projectArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//makeHE<<<numBlocks, numThreads>>>(HE, force1, &((*(data->gpu->psForce4))[0].w), masses, params.sDelta, k, _3N);

		// Put back positions
		cu->executeKernel( copyToKernel, copyToArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
		//copyToOpenMM<<<numBlocks, numThreads>>>(&((*(data->gpu->psPosq4))[0].w), pos1, _3N);
	}
}

void FBMCuda::computeS() {
	CudaContext *cu = data->contexts[0];
	makeBlocksAndThreads( m * m );

	cudaMalloc( ( void ** ) &S, m * m * sizeof( float ) );

	CUmodule matModule = cu->createModule( CudaFBMKernelSources::matmul );
	CUfunction matFunction = cu->getKernel( matModule, "MatMul" );
	void *matArgs[] = {&S, &Et, &HE, &m, &_3N};
	cu->executeKernel( matFunction, matArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	CUmodule symmModule = cu->createModule( CudaFBMKernelSources::symmetrize );
	CUfunction symmFunction = cu->getKernel( symmModule, "symmetrize2D" );
	void *symmArgs[] = {&S, &m};
	cu->executeKernel( symmFunction, symmArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	//MatMulKernel<<<numBlocks, numThreads>>>(S, Et, HE, m, _3N);
	//symmetrize2D<<<numBlocks, numThreads>>>(S, m);
}

void FBMCuda::diagonalizeS() {
	// Initialize Cula and check for errors
	//culaStatus status = culaInitialize();
	//if(status != culaNoError)
	//{
	//  cout << culaGetStatusString(status) << endl;
	//}

	// Temporary, for eigenvectors and eigenvalues of S
	// Cula populates the same array
	//float* tmpEigval = (float*) malloc(m*sizeof(float));
	//float** tmpEigvec = (float**) malloc(m*m*sizeof(float));

	// In the future we may want to find a way to directly access
	// Cula GPU arrays, rather than copy twice

	// Copy the eigenvalues into dS and eigenvectors into Q
	cudaMalloc( ( void ** ) &dS, m * sizeof( float ) );
	//cudaMemcpy(dS, tmpEigval, m*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc( ( float ** ) &Q, m * m * sizeof( float ) );
	cudaMemcpy( Q, S, m * m * sizeof( float ), cudaMemcpyDeviceToHost );
	//cudaMemcpy(Q, tmpEigvec, m*m*sizeof(float), cudaMemcpyHostToDevice);
	//status = culaDeviceDsyev('V', 'U', m, (double*) &(Q[0]), m, (double*)&dS);

	int lwork = 1 + 6 * m + 2 * m * m;
	double *work = ( double * ) malloc( lwork * sizeof( double ) );
	int liwork = 3 + 5 * m;
	int *iwork = ( int * )malloc( liwork * sizeof( int ) );
	int info;
	magma_dsyevd( MagmaVec, MagmaUpper, m, ( double * ) & ( Q[0] ), m, ( double * )&dS, work,  lwork, iwork, liwork, &info );
}

void FBMCuda::computeModes( vector<double> &eigenvalues, vector<vector<Vec3> > &modes ) {
	CudaContext *cu = data->contexts[0];
	// A matrix multiply, but we have to copy back into formats the user expects
	makeBlocksAndThreads( _3N * m );
	float **U;
	cudaMalloc( ( float ** ) &U, _3N * m * sizeof( float ) );

	CUmodule matModule = cu->createModule( CudaFBMKernelSources::matmul );
	CUfunction matFunction = cu->getKernel( matModule, "MatMul" );
	void *matArgs[] = {&U, &E, &Q, &_3N, &m};
	cu->executeKernel( matFunction, matArgs, data->contexts[0]->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	//MatMulKernel<<<numBlocks, numThreads>>>(U, E, Q, _3N, m);

	eigenvalues.resize( m );
	cudaMemcpy( &( eigenvalues[0] ), dS, m * sizeof( float ), cudaMemcpyDeviceToHost );

	float **myU = ( float ** ) malloc( _3N * m * sizeof( float ) );
	cudaMemcpy( myU, U, _3N * m * sizeof( float ), cudaMemcpyDeviceToHost );

	modes.resize( m );

	for( int i = 0; i < m; i++ ) {
		modes[i].resize( _3N / 3 );
	}

	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < _3N; j += 3 ) {
			modes[i][j][0] = myU[i][j];
			modes[i][j][1] = myU[i][j + 1];
			modes[i][j][2] = myU[i][j + 2];
		}
	}
}
