#include "MakeProjection_kernel.cu"

#define THREADSPERBLOCK 512

extern "C" void TestMakeProjection( int n, int m, float* eigvec, int* indices, float* E, float* Et ){
    float *gpuEt;
    cudaMalloc( ( void ** ) &gpuEt, m * n * sizeof( float ) );

    float *gpuE;
    cudaMalloc( ( void ** ) &gpuE, n * m * sizeof( float ) );

    float *gpuEigvec;
    cudaMalloc( ( void ** ) &gpuEigvec, n * m * sizeof( float ) );
    cudaMemcpy( gpuEigvec, eigvec, n * m * sizeof( float ), cudaMemcpyHostToDevice );

    int *gpuIndices;
    cudaMalloc( ( void ** ) &gpuIndices, m * sizeof( int ) );
    cudaMemcpy( gpuIndices, indices, m * sizeof( int ), cudaMemcpyHostToDevice );

    if( m * n <= THREADSPERBLOCK ) {
        makeProjection <<< 1, m*n >>>( gpuEt, gpuE, gpuEigvec, gpuIndices, m, n );
    }else{
        makeProjection <<< ( m * n ) / THREADSPERBLOCK + 1, THREADSPERBLOCK >>>( gpuEt, gpuE, gpuEigvec, gpuIndices, m, n );
    }

    cudaMemcpy( Et, gpuEt, m * n * sizeof( float ), cudaMemcpyDeviceToHost );
    cudaMemcpy( E, gpuE, m * n * sizeof( float ), cudaMemcpyDeviceToHost );

    cudaFree( gpuEt );
    cudaFree( gpuE );
    cudaFree( gpuEigvec );
    cudaFree( gpuIndices );
}
