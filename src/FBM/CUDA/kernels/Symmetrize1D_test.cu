#include "Symmetrize1D_kernel.cu"

#define THREADSPERBLOCK 512

extern "C" void TestSymmetrize1D( const size_t blocks, const size_t totalSize, float* output, float* blockHessian, float* blockSizes, int* startDof ) {
    float *gpuBlockHessian;
    cudaMalloc( ( void ** ) &gpuBlockHessian, totalSize * sizeof( float ) );
    cudaMemcpy( gpuBlockHessian, blockHessian, totalSize * sizeof( float ), cudaMemcpyHostToDevice );

    int *gpuBlockSizes;
    cudaMalloc( ( void ** ) &gpuBlockSizes, blocks * sizeof( int ) );
    cudaMemcpy( gpuBlockSizes, blockSizes, blocks * sizeof( int ), cudaMemcpyHostToDevice );

    int *gpuBlockPositions;
    cudaMalloc( ( void ** ) &gpuBlockPositions, blocks * sizeof( int ) );
    cudaMemcpy( gpuBlockPositions, startDof, blocks * sizeof( int ),	cudaMemcpyHostToDevice );

    symmetrize1D <<< blocks / (THREADSPERBLOCK + 1), THREADSPERBLOCK>>>( gpuBlockHessian, gpuBlockPositions, gpuBlockSizes, blocks );

    float outBlockHessian[totalSize];
    cudaMemcpy( outBlockHessian, gpuBlockHessian, totalSize * sizeof( float ), cudaMemcpyDeviceToHost );

    cudaFree( gpuBlockHessian );
    cudaFree( gpuBlockSizes );
    cudaFree( gpuBlockPositions );
}
