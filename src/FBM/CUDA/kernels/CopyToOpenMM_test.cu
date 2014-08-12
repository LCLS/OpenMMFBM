#include "CopyToOpenMM_kernel.cu"

extern "C" void TestCopyTo( const int n, float* input, float4* output ) {
    float *out_gpu_positions;
    cudaMalloc( ( void ** ) &out_gpu_positions, n * sizeof( float4 ) );
    cudaMemcpy( out_gpu_positions, output, n * sizeof( float4 ), cudaMemcpyHostToDevice );

    float *in_gpu_positions;
    cudaMalloc( ( void ** ) &in_gpu_positions, 3 * n * sizeof( float ) );
    cudaMemcpy( in_gpu_positions, input, 3 * n * sizeof( float ), cudaMemcpyHostToDevice );

    copyToOpenMM<<<3 * n, 1>>>( out_gpu_positions, in_gpu_positions, 3 * n );

    cudaMemcpy( output, out_gpu_positions, n * sizeof( float4 ), cudaMemcpyDeviceToHost );

    cudaFree( in_gpu_positions );
    cudaFree( out_gpu_positions );
}
