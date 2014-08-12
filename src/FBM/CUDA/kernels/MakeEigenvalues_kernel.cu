/*
 * makeEigenvalues()
 * float* eigenvalues: Will be populated by the function
 * float** eigenvectors: Will be populated by the function
 * float* blockHessian: A linear array containing the block Hessian
 *                      matrices in sorted order.  Note that these
 *                      have different sizes.  Thus block 1 (aXa) will
 *                      occupy slots 0 to a*a-1, block 2 (bXb) will
 *                      occupy slots a*a to b*b-1, and so on.
 * float* array1D: Eigenvectors as a linear array.
 * int* evecstarts: Array of starting positions of every eigenvector
 *                    in array1D.
 * int* blocknums: Array of starting atoms of each block.
 * int* blocksize: Array of sizes of each block.
 * int N: Total degrees of freedom.
 */
__global__ void makeEigenvalues( float *eigenvalues, float *blockHessian, int *blocknums, int *blocksizes, int *hessiannums, int N, int numblocks ) {
    // elementnum is the degree of freedom (0 to 3n-1)
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    if( elementNum >= N ) {
        return;
    }

    // b is the block number in which DOF elementnum resides
    // blocknums contains atom numbers, so we must divide by 3
    // We find the first index with an atom number larger than
    // ours, and take one less (or numblocks-1 if we are at the end)
    int b = 0;
    while( b < numblocks ) {
        if( blocknums[b] > elementNum / 3 ) {
            break;
        }
        b++;
    }
    b--;

    // 3*blocknums[b] is the starting degree of freedom for our block
    // We must compute an offset from that, call it x.
    int x = elementNum - 3 * blocknums[b];

    // We initialize our spot to hessiannums[b], which is the starting
    // Hessian location for our block.
    // We then want to take the diagonal entry from that offset
    // So element (x,x)
    int spot = hessiannums[b] + x * ( 3 * blocksizes[b] ) + x;

    eigenvalues[elementNum] = blockHessian[spot];
}
