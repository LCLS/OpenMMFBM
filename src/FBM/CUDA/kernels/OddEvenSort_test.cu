#include "OddEvenSort_kernel.cu"

#define THREADSPERBLOCK 512

extern "C" void TestOddEvenSort( const int n, float* eigenvalues, float* eigenvectors ) {
    float* gpu_eigenvalues;
    cudaMalloc( (void**) &gpu_eigenvalues, n*sizeof(float));
    cudaMemcpy( gpu_eigenvalues, eigenvalues, n*sizeof(float), cudaMemcpyHostToDevice);

    float* gpu_eigenvectors;
    cudaMalloc( (void**) &gpu_eigenvectors, n*n*sizeof(float));
    cudaMemcpy( gpu_eigenvectors, eigenvectors, n*n*sizeof(float), cudaMemcpyHostToDevice);

    int oddcount = n/2;
    int evencount = (n % 2 == 0) ? oddcount : oddcount+1;

    for (int i = 0; i < ceil(n/2); i++) {
        if (oddcount <= THREADSPERBLOCK) {
            oddEvenEigSort<<<1, oddcount>>>(gpu_eigenvalues, gpu_eigenvectors, n);
        } else {
            oddEvenEigSort<<<oddcount/THREADSPERBLOCK + 1, THREADSPERBLOCK>>>(gpu_eigenvalues, gpu_eigenvectors, n);
        }

        if (evencount <= THREADSPERBLOCK) {
            oddEvenEigSort<<<1, evencount>>>(gpu_eigenvalues, gpu_eigenvectors, n, 1);
        } else {
            oddEvenEigSort<<<evencount/THREADSPERBLOCK + 1, THREADSPERBLOCK>>>(gpu_eigenvalues, gpu_eigenvectors, n);
        }
    }

    cudaMemcpy(eigenvalues, gpu_eigenvalues, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenvectors, gpu_eigenvectors, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_eigenvalues);
    cudaFree(gpu_eigenvectors);
}
