#ifndef KERNELS_H
#define KERNELS_H

/* This function perturbs each element of the vector initpos by delta
 * and places the result in blockpos.
 * Note you can do a negative perturbation by passing a delta < 0
 * The function assumes threads and blocks to be organized linearly
 */
__global__ void perturbPositions(float* blockpos, float4* initpos, float delta, int* blocks, int numblocks, int setnum, int N) {
//__global__ void perturbPositions(float* blockpos, float* initpos, float delta, int* blocks, int numblocks, int setnum, int N) {
   //return;
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;

   if(blockNum >= numblocks) {
      return;
   }

   int dof = 3*blocks[blockNum]+setnum;
   int atom = dof/3;
   
   if (atom >= N || (blockNum != (numblocks-1) && atom >= blocks[blockNum+1])) return;  // Out of bounds
  
   // Assumes struct is linear
   // Accessing struct as if it were a linear array
   // Note we have to account for ws.., so we need to add one extra per atom
   //int index = 4*atom + dof%3 + 1;
   //blockpos[dof] = initpos[index];
   
   //initpos[index] += delta;
   
   int axis = dof % 3;
   
   
   if(axis == 0) {
      blockpos[dof] = initpos[atom].x;
      initpos[atom].x += delta;
   } else if(axis == 1) {
      blockpos[dof] = initpos[atom].y;
     initpos[atom].y += delta;
   } else {
      blockpos[dof] = initpos[atom].z;
     initpos[atom].z += delta;
   }
     
   
}

__global__ void perturbByE(float* tmppos, float4* mypos, float eps, float* E, float* masses, int k, int m, int N) {
   int dof = blockIdx.x * blockDim.x + threadIdx.x;
   if (dof >= N) return;
   int atom = dof/3;
 
   int axis = dof % 3;
   if (axis == 0) {
      tmppos[dof] = mypos[atom].x;
      mypos[atom].x += eps*E[dof*m+k] / sqrt(masses[atom]);
   }
   else if (axis == 1) {
      tmppos[dof] = mypos[atom].y;
      mypos[atom].y += eps*E[dof*m+k] / sqrt(masses[atom]);
   }
   else {
      tmppos[dof] = mypos[atom].z;
      mypos[atom].z += eps*E[dof*m+k] / sqrt(masses[atom]);
   }
}


__global__ void blockcopyFromOpenMM(float* target, float* source, int* blocks, int numblocks, int setnum, int N) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   int dof = 3*blocks[blockNum]+setnum;
   int atom = dof/3;
   
   if (atom >= N || (blockNum != numblocks && atom >= blocks[blockNum+1])) return;  // Out of bounds

   target[dof]= *( source + (dof+atom+1)*sizeof(float) ); // Save the old
}

__global__ void blockcopyToOpenMM(float* target, float* source, int* blocks, int numblocks, int setnum, int N) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   int dof = 3*blocks[blockNum]+setnum;
   int atom = dof/3;
   
   if (atom >= N || (blockNum != numblocks && atom >= blocks[blockNum+1])) return;  // Out of bounds

   *( target + (dof+atom+1)*sizeof(float) ) = source[dof]; // Save the old
}


__global__ void copyFromOpenMM(float* target, float* source, int N) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   int atom = elementNum/3;
   if (elementNum > N) return;
   else target[elementNum] = source[4*atom + elementNum%3];   
}

__global__ void copyToOpenMM(float* target, float* source, int N) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   int atom = elementNum / 3;
   if (elementNum > N) return;
   //else target[elementNum] = source[elementNum];
   else target[4*atom + elementNum%3] = source[elementNum];
}

/* This function will populate the matrix with block Hessians on the
 * diagonal (h) which is assumed to be two dimensional.
 * The function assumes that thread blocks are one-dimensional
 */
__global__ void makeBlockHessian(float* h, float* forces1, float* forces2, float* mass, float blockDelta, int* blocks, int* blocksizes, int numblocks, int* hessiannums, int* hessiansizes, int setnum, int N) {
   int blockNum = blockIdx.x*blockDim.x + threadIdx.x;
   int dof = 3*blocks[blockNum]+setnum;
   int atom = dof / 3;
   if (atom >= N || (blockNum != numblocks-1 && atom >= blocks[blockNum+1])) return;  // Out of bounds

   int start_dof = 3 * blocks[blockNum];
   int end_dof;
   if( blockNum == numblocks - 1 ) 
	end_dof = 3 * N;
   else
        end_dof = 3 * blocks[blockNum+1];
 
   /* I also would like to parallelize this at some point as well */
   for( int k = start_dof; k < end_dof; k++ ) {
      float blockScale = 1.0 / (blockDelta * sqrt(mass[atom] * mass[k/3]));
      //h[startspot+i] = (forces1[k] - forces2[k]) * blockScale;
      h[hessiannums[blockNum] + (k-start_dof)*(3*blocksizes[blockNum]) + (dof - start_dof)] = (forces1[k] - forces2[k]) * blockScale;
   }
}


/* Take a matrix, make it symmetric
 * Assumes a 2D block of threads
 */
__global__ void symmetrize1D(float* h, int* blockPositions, int* blockSizes, int numBlocks) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   if (blockNum >= numBlocks) return;

   // blockSizes are given in terms of atoms, convert to dof
   const unsigned int blockSize = 3 * blockSizes[blockNum];

   float* block = &(h[blockPositions[blockNum]]);
   for(unsigned int r = 0; r < blockSize - 1; r++) {
       for(unsigned int c = r + 1; c < blockSize; c++) {
          const float avg = 0.5f * (block[r * blockSize + c] + block[c * blockSize + r]);
          block[r * blockSize + c] = avg;
          block[c * blockSize +	r] = avg;
       }
   }

}

__global__ void symmetrize2D(float* h, int natoms) {
   const int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   const int dof = 3 * natoms;
   if (elementNum >= dof * dof) return;
   int r = elementNum / dof;
   int c = elementNum % dof;

   if (r > c) return;
   else {
        const float avg = 0.5 * (h[r * dof + c] + h[c * dof + r]); 
	h[r * dof + c] = avg;
        h[c * dof + r] = avg;
   }
}
/* Matrix multiplication kernel
 * From the CUDA programmer's guide, with modifications
 * Performs C = A * B
 * width is the number of columns in A and the number of rows in B
 */
__global__ void MatMulKernel(float* C, float* A, float* B, int Aheight, int Awidth, int Bwidth)
{
    float result = 0;
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (elementNum > Aheight*Bwidth) return;
    int row = elementNum / Bwidth;
    int col = elementNum % Bwidth;
    for (int e = 0; e < Awidth; e++)
       result += A[row*Awidth+e] * B[e*Bwidth+col];
    C[row*Bwidth+col] = result;
}


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
__global__ void makeEigenvalues(float* eigenvalues, float* blockHessian, int* blocknums, int* blocksizes, int* hessiannums, int N, int numblocks) {
   // elementnum is the degree of freedom (0 to 3n-1)
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   if (elementNum >= N) return;

   // b is the block number in which DOF elementnum resides
   // blocknums contains atom numbers, so we must divide by 3
   // We find the first index with an atom number larger than 
   // ours, and take one less (or numblocks-1 if we are at the end)
   int b = 0;
   while (b < numblocks) {
      if (blocknums[b] > elementNum/3) break; 
      b++;
   }
   b--;

   // 3*blocknums[b] is the starting degree of freedom for our block
   // We must compute an offset from that, call it x.
   int x = elementNum - 3*blocknums[b];

   // We initialize our spot to hessiannums[b], which is the starting
   // Hessian location for our block.
   // We then want to take the diagonal entry from that offset 
   // So element (x,x)
   int spot = hessiannums[b] + x*(3*blocksizes[b])+x;

   eigenvalues[elementNum] = blockHessian[spot];
   //eigenvectors[elementNum] = &(array1D[evecstarts[elementNum]]);
}


/* Odd-even sort
 * This will need to be called within a loop that runs from 0 to
 * the ceiling of N/2 - 1, where N is the number of eigenvalues
 * We assume a linear array of threads and it will be the caller's
 * responsibility to ensure the thread indices are in bounds
 * Note to self: There is a GPU Quicksort available, but I have to modify
 * it to also move around eigenvectors... challenging, striving for accuracy
 */
__global__ void oddEvenEigSort(float* eigenvalues, float* eigenvectors, int N, int odd=0) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   elementNum *= 2;
   if (odd) elementNum++;
   if (elementNum >= N-1) return;

   if (eigenvalues[elementNum] > eigenvalues[elementNum+1]) {
      float tmp = eigenvalues[elementNum];
      eigenvalues[elementNum] = eigenvalues[elementNum+1];
      eigenvalues[elementNum+1] = tmp;

      for (int i = 0; i < N; i++) {
         tmp = eigenvectors[i*N+elementNum];
	 eigenvectors[i*N+elementNum] = eigenvectors[i*N+elementNum+1];
	 eigenvectors[i*N+elementNum+1] = tmp;
      }
   }
}


/* Sorting within the blocks */
__global__ void blockEigSort(float* eigenvalues, float* eigenvectors, int* blocknums, int* blocksizes, int N) {
   int blockNumber = blockIdx.x * blockDim.x + threadIdx.x;
   int startspot = blocknums[blockNumber];
   int endspot = startspot+blocksizes[blockNumber]-1;

   // Bubble sort for now, thinking blocks are relatively small
   // We may fix it later
   for (int i = startspot; i < endspot; i++) {
      for (int j = startspot; j < i; j++) {
         if (eigenvalues[j] > eigenvalues[j+1])
	    {
               float tmp = eigenvalues[j];
               eigenvalues[j] = eigenvalues[j+1];
               eigenvalues[j+1] = tmp;

               // Swapping addresses
      		for (int i = 0; i < N; i++) {
         		tmp = eigenvectors[i*N+j];
			eigenvectors[i*N+j] = eigenvectors[i*N+j+1];
	 		eigenvectors[i*N+j+1] = tmp;
      		}
               /*float* tmpaddr = eigenvectors[j];
               eigenvectors[j] = eigenvectors[j+1];;
               eigenvectors[j+1] = tmpaddr;*/
	    }
	 }
   }
}


/* 
 * makeProjection()
 * Forms E and E^T matrices from eigenvectors
 * float** eT: E^T, populated by function
 * float** e: E, populated by function
 * float** eigenvec: matrix of eigenvectors, unsorted
 * int* indices: indices to accept from eigenvectors
 * int N: degrees of freedom
*/
__global__ void makeProjection(float* eT, float* e, float* eigenvec, int* indices, int M, int N)
{
    int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
    if (elementNum >= M*N) return;
    int m = elementNum / N;
    int n = elementNum % N;
    e[n*M+m] = eigenvec[n*M+indices[m]];
    eT[m*N+n] = e[n*M+m];
}


__global__ void computeNormsAndCenter(float* norms, float* center, float* masses, float4* positions, int* blocknums, int* blocksizes) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   float totalmass = 0.0;
   for( int j = blocknums[blockNum]; j <= blocknums[blockNum]+blocksizes[blockNum]-1; j += 3 ) {
      float mass = masses[ j / 3 ];
      center[blockNum*3+0] = positions[j/3].x*mass;
      center[blockNum*3+1] = positions[j/3].y*mass;    
      center[blockNum*3+2] = positions[j/3].z*mass;    
      totalmass += mass;
   }

   norms[blockNum] = sqrt(totalmass);
   center[blockNum*3+0] /= totalmass;
   center[blockNum*3+1] /= totalmass;
   center[blockNum*3+2] /= totalmass;
}

/* Create degrees of freedom vectors */
//__global__ void geometricDOF(float*** Qi_gdof, float* positions, float* masses, int* blocknums, int* blocksizes, float* norm, float** pos_center) {
__global__ void geometricDOF(float* Qi_gdof, float4* positions, float* masses, int* blocknums, int* blocksizes, int largestsize, float* norm, float* pos_center) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;
   for (int j = 0; j < blocksizes[blockNum]-3; j += 3) {

      int atom = ( blocknums[blockNum] + j ) / 3;
      float mass = masses[atom];
      float factor = sqrt( mass ) / norm[atom];

            Qi_gdof[blockNum*largestsize*6 + j*6 + 0]   = factor;
	    Qi_gdof[blockNum*largestsize*6 + (j+1)*6 + 1] = factor;
	    Qi_gdof[blockNum*largestsize*6 + (j+2)*6 + 2]= factor;

      float diff0 = positions[atom].x - pos_center[atom*3+0];
      float diff1 = positions[atom].y - pos_center[atom*3+1];
      float diff2 = positions[atom].z - pos_center[atom*3+2];
			   
      Qi_gdof[blockNum*largestsize*6 + (j+1)*6 + 3] = diff2 * factor;
      Qi_gdof[blockNum*largestsize*6 + (j+2)*6 + 3] = -diff1 * factor;

      Qi_gdof[blockNum*largestsize*6 + (j)*6 + 4] = -diff2 * factor;
      Qi_gdof[blockNum*largestsize*6 + (j+2)*6 + 4] = -diff0 * factor;
      
      Qi_gdof[blockNum*largestsize*6 + (j)*6 + 5] = diff1 * factor;
      Qi_gdof[blockNum*largestsize*6 + (j+1)*6 + 5] = -diff0 * factor;
  }
   // Normalize first vector
   float rotnorm = 0.0;
   for (int j = 0; j < blocksizes[blockNum]; j++) 
      rotnorm += Qi_gdof[blockNum*largestsize*6 +j*6+3] * Qi_gdof[blockNum*largestsize*6+j*6+3];

   rotnorm = 1.0 / sqrt (rotnorm);

   for (int j = 0; j < blocksizes[blockNum]; j++)
      Qi_gdof[blockNum*largestsize*6+j*6+3] *= rotnorm;
}


/* Assumes eigvec is sorted */
__global__ void orthogonalize23(float* Qi_gdof, int* blocksizes, int numblocks, int largestblock) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   for (int j = 4; j < 6; j++) {
   for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
      float dot_prod = 0.0;
      for (int l = 0; l < blocksizes[i]; l++)
	 dot_prod += Qi_gdof[i*6*largestblock + l*6 + k] * Qi_gdof[i*6*largestblock + l*6 + j];
         //dot_prod += Qi_gdof[i][l][k] * Qi_gdof[i][l][j];
      for (int l = 0; l < blocksizes[i]; l++)
         Qi_gdof[i*6*largestblock + l*6 + j] -= Qi_gdof[i*6*largestblock + l*6 + k] * dot_prod;
         //Qi_gdof[i][l][j] -= Qi_gdof[i][l][k] * dot_prod;
   }

   float rotnorm = 0.0;
   for (int l = 0; l < blocksizes[i]; l++)
      rotnorm += Qi_gdof[i*6*largestblock + l*6 + j] * Qi_gdof[i*6*largestblock + l*6 + j];
      //rotnorm += Qi_gdof[i][l][j] * Qi_gdof[i][l][j];

   rotnorm = 1.0 / sqrt(rotnorm);

   for (int l = 0; l < blocksizes[i]; l++)
      Qi_gdof[i*6*largestblock + l*6 + j] *= rotnorm;
      //Qi_gdof[i][l][j] *= rotnorm;
      }
}


//__global__ void orthogonalize(float** eigvec, float*** Qi_gdof, int cdof, int* blocksizes, int* blocknums) {
__global__ void orthogonalize(float* eigvec, float* Qi_gdof, int cdof, int* blocksizes, int* blocknums, int largestblock) {
   int blockNum = blockIdx.x * blockDim.x + threadIdx.x;

        // orthogonalize original eigenvectors against gdof
        // number of evec that survive orthogonalization
        int curr_evec = 6;
	int size = blocksizes[blockNum];
        int startatom = blocknums[blockNum] / 3;
        for( int j = 0; j < size; j++ ) { // <-- vector we're orthogonalizing
            // to match ProtoMol we only include size instead of size + cdof vectors
            // Note: for every vector that is skipped due to a low norm,
            // we add an additional vector to replace it, so we could actually
            // use all size original eigenvectors
            if( curr_evec == size ) {
                break;
            }

            // orthogonalize original eigenvectors in order from smallest magnitude
            // eigenvalue to biggest
	    // TMC The eigenvectors are sorted now
            //int col = sortedPairs.at( j ).second;

            // copy original vector to Qi_gdof -- updated in place
            for( int l = 0; l < size; l++ ) {
	        //Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] = eigvec[blocknums[blockNum]+l][j];
	        Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] = eigvec[(blocknums[blockNum]+l)*largestblock+j];
            }

            // get dot products with previous vectors
            for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
                // dot product between original vector and previously
                // orthogonalized vectors
                double dot_prod = 0.0;
                for( int l = 0; l < size; l++ ) {
                    //dot_prod += Qi_gdof[blockNum*6*largestblock+l*6+k] * eigvec[blocknums[blockNum]+l][j];
                    dot_prod += Qi_gdof[blockNum*6*largestblock+l*6+k] * eigvec[(blocknums[blockNum]+l)*largestblock+j];
                }

                // subtract from current vector -- update in place
                for( int l = 0; l < size; l++ ) {
                    Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] = Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] - Qi_gdof[blockNum*6*largestblock+l*6+k] * dot_prod;
                }
            }

            //normalize residual vector
            double norm = 0.0;
            for( int l = 0; l < size; l++ ) {
                norm += Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] * Qi_gdof[blockNum*6*largestblock+l*6+curr_evec];
            }

            // if norm less than 1/20th of original
            // continue on to next vector
            // we don't update curr_evec so this vector
            // will be overwritten
            if( norm < 0.05 ) {
                continue;
            }

            // scale vector
            norm = sqrt( norm );
            for( int l = 0; l < size; l++ ) {
                Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] = Qi_gdof[blockNum*6*largestblock+l*6+curr_evec] / norm;
            }

            curr_evec++;
        }
        
        // 4. Copy eigenpairs to big array
        //    This is necessary because we have to sort them, and determine
        //    the cutoff eigenvalue for everybody.
        // we assume curr_evec <= size
        for( int j = 0; j < curr_evec; j++ ) {
            //eval[startatom + j] = di[col]; No longer necessary

            // orthogonalized eigenvectors already sorted by eigenvalue
            for( int k = 0; k < size; k++ ) {
                //eigvec[startatom + k][startatom + j] = Qi_gdof[blockNum*6*largestblock+k*6+j];
                eigvec[(startatom + k)*largestblock+(startatom + j)] = Qi_gdof[blockNum*6*largestblock+k*6+j];
            }
        }
}


__global__ void makeHE(float* HE, float* force1, float4* force2, float* masses, float eps, int k, int m, int N) {
   int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
   int atom = elementNum / 3;
   if (elementNum >= N) return;

   int axis = elementNum % 3;
   if (axis == 0)
      HE[elementNum*m+k] = (force1[elementNum] - force2[atom].x) / (sqrt(masses[atom]) * 1.0 * eps);
   else if (axis == 1)
      HE[elementNum*m+k] = (force1[elementNum] - force2[atom].y) / (sqrt(masses[atom]) * 1.0 * eps);
   else
      HE[elementNum*m+k] = (force1[elementNum] - force2[atom].z) / (sqrt(masses[atom]) * 1.0 * eps);
}



#endif
