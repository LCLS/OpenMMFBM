#include "CopyFromOpenMM_kernel.cu"

extern "C" void TestCopyFrom( const int n, float4* input, float* output) {
    float* in_gpu_positions;
    cudaMalloc( (void**) &in_gpu_positions, n * sizeof(float4) );
    cudaMemcpy(in_gpu_positions, input, n * sizeof(float4), cudaMemcpyHostToDevice );

    float* out_gpu_positions;
    cudaMalloc( (void**) &out_gpu_positions, n * 3 * sizeof(float) );
    cudaMemcpy(out_gpu_positions, output, n * 3 * sizeof(float), cudaMemcpyHostToDevice );

    copyFromOpenMM<<<3 * n, 1>>>(out_gpu_positions, in_gpu_positions, n * 3);

    cudaMemcpy(output, out_gpu_positions, n * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_gpu_positions);
    cudaFree(out_gpu_positions);
}
