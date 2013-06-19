#ifndef KERNELS_H
#define KERNELS_H

/* This function perturbs each element of the vector initpos by delta
 * and places the result in blockpos.
 * Note you can do a negative perturbation by passing a delta < 0
 * The function assumes threads and blocks to be organized linearly
 */
__global__ void perturbOneSet(float* blockpos, float* initpos, int delta, int* blocks, int numblocks, int setnum, int N) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   int dof = 3*blocks[blockNum]+setNum;
   int atom = dof/3;
   
   if (atom >= N || (blockNum != numblocks && atom >= blocks[blockNum+1])) return;  // Out of bounds

   blockPos[dof] = initpos[dof] + delta;
}


/* This function will populate the matrix with block Hessians on the
 * diagonal (h) which is assumed to be two dimensional.
 * The function assumes that thread blocks are one-dimensional
 */
__global__ void makeBlockHessian(float** h, float* forces1, float* forces2, float* mass, float blockDelta, int* blocks, int numblocks, int setnum, int N) {
   int elementNum = blockIdx.x*blockDim.x + threadIdx.x
   int dof = 3*blocks[blockNum]+setNum;
   int atom = dof / 3;
   
   if (atom >= N || (blockNum != numblocks && atom >= blocks[blockNum+1])) return;  // Out of bounds

   int start_dof = 3 * blocks[blockNum];
   int end_dof;
   if( blockNum == numblocks - 1 ) 
	end_dof = 3 * N;
   else
        end_dof = 3 * blocks[blockNum+1];
 
   /* I also would like to parallelize this at some point as well */
   for( int k = start_dof; k < end_dof; k++ ) {
      float blockScale = 1.0 / (blockDelta * sqrt(mass[atom] * mass[k/3]));
      h[k][atom] = (forces1[k] - forces2[k]) * blockScale;
   }
}


/* Take a matrix, make it symmetric
 * Assumes a 2D block of threads
 */
__global__ void symmetrize(float** h, int xdim) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   int x = elementNum / xdim;
   int y = elementNum % xdim;

   if (y > x) return;
   else h[x][y] = 0.5 * (h[x][y]+h[y][x]);
}

/* Matrix multiplication kernel
 * From the CUDA programmer's guide, with modifications
 * Performs C = A * B
 * width is the number of columns in A and the number of rows in B
 */
__global__ void MatMulKernel(float** C, float** A, float** B, int Aheight, int Awidth)
{
    float result = 0;
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;

    int row = elementNum / Aheight;
    int col = elementNum % Aheight;
    for (int e = 0; e < Awidth; e++)
       result += A[row][e] * B[e][col];
    C[row][col] = result;
}


/* Odd-even sort
 * This will need to be called within a loop that runs from 0 to
 * the ceiling of N/2 - 1, where N is the number of eigenvalues
 * We assume a linear array of threads and it will be the caller's
 * responsibility to ensure the thread indices are in bounds
 * Note to self: There is a GPU Quicksort available, but I have to modify
 * it to also move around eigenvectors... challenging, striving for accuracy
 */
__global__ void oddEvenEigSort(float* eigenvalues, float** eigenvectors, int even=0) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   if (even) elementNum++; // Even step will be one index higher

   if (eigenvalues[elementNum] > eigenvalues[elementNum+1]) {
      float tmp = eigenvalues[elementNum];
      eigenvalues[elementNum] = eigenvalues[elementNum+1];
      eigenvalues[elementNum+1] = tmp;

      // Swapping addresses
      float* tmpaddr = eigenvectors[elementNum];
      eigenvectors[elementNum] = eigenvectors[elementNum+1];;
      eigenvectors[elementNum+1] = tmpaddr;
   }
}

/* I want this to go away eventually if possible
 * Right now it is only used for the computation of E
 */
__global__ void makeTranspose(float** eT, float** e, int width)
{
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    int row = elementNum / width;
    int col = elementNum % width;
    eT[col][row] = e[row][col];
}

/* Create degrees of freedom vectors */
__global__ void geometricDOF(float** Qi_gdof, float* positions, int* mass, int startatom, float norm, float* pos_center) {
   int j = blockIdx.x * blockDim.x + threadIdx.x;

   float atom_index = ( startatom + j ) / 3;
   float mass = mass[atom_index];
   float factor = sqrt( mass ) / norm;

   float diff0 = positions[3*atom_index] - pos_center[0];
   float diff1 = positions[3*atom_index+1] - pos_center[1];
   float diff2 = positions[3*atom_index+2] - pos_center[2];
			   
   Qi_gdof[j + 1][3] =  diff[2] * factor;
   Qi_gdof[j + 2][3] = -diff[1] * factor;

   Qi_gdof[j][4]   = -diff[2] * factor;
   Qi_gdof[j + 2][4] =  diff[0] * factor;

   Qi_gdof[j][5]   =  diff[1] * factor;
   Qi_gdof[j + 1][5] = -diff[0] * factor;
}


/* Assumes eigvec is sorted */
__global__ void orthogonalize(float** eigvec, float** Qi_gdof) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    
}






#endif
