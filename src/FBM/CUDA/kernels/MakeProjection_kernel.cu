/*
 * makeProjection()
 * Forms E and E^T matrices from eigenvectors
 * float** eT: E^T, populated by function
 * float** e: E, populated by function
 * float** eigenvec: matrix of eigenvectors, unsorted
 * int* indices: indices to accept from eigenvectors
 * int N: degrees of freedom
*/
__global__ void makeProjection( float *eT, float *e, float *eigenvec, int *indices, int M, int N ) {
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    if( elementNum >= M * N ) {
        return;
    }
    int m = elementNum / N;
    int n = elementNum % N;
    e[n * M + m] = eigenvec[n * M + indices[m]];
    eT[m * N + n] = e[n * M + m];
}
