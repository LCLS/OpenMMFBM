/* Odd-even sort
 * This will need to be called within a loop that runs from 0 to
 * the ceiling of N/2 - 1, where N is the number of eigenvalues
 * We assume a linear array of threads and it will be the caller's
 * responsibility to ensure the thread indices are in bounds
 * Note to self: There is a GPU Quicksort available, but I have to modify
 * it to also move around eigenvectors... challenging, striving for accuracy
 */
__global__ void oddEvenEigSort( float *eigenvalues, float *eigenvectors, int N, int odd = 0 ) {
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    elementNum *= 2;
    if( odd ) {
        elementNum++;
    }
    if( elementNum >= N - 1 ) {
        return;
    }

    if( eigenvalues[elementNum] > eigenvalues[elementNum + 1] ) {
        float tmp = eigenvalues[elementNum];
        eigenvalues[elementNum] = eigenvalues[elementNum + 1];
        eigenvalues[elementNum + 1] = tmp;

        for( int i = 0; i < N; i++ ) {
            tmp = eigenvectors[i * N + elementNum];
            eigenvectors[i * N + elementNum] = eigenvectors[i * N + elementNum + 1];
            eigenvectors[i * N + elementNum + 1] = tmp;
        }
    }
}
