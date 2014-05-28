#define MDIM 5
#define NDIM 8
#define THREADSPERBLOCK 512
#include <stdio.h>
#include <iostream>
#include "OpenMMFBM/kernels/Kernels.cu"

using namespace std;

bool testMakeProjection() {
   float* eigvec = (float*) malloc(NDIM*MDIM*sizeof(float));
   int* indices = (int*) malloc(MDIM*sizeof(int));

   // Eigenvectors
   int value = 0;
   for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < MDIM; j++) {
          eigvec[i*MDIM+j] = value;
	  value++;
      }
   }

   // Initial indices
   for (int i = 0; i < MDIM; i++)
      indices[i] = i;
   
   // Swap random indices
   for (int i = 0; i < 100; i++) {
      int index1 = rand() % MDIM;
      int index2 = rand() % MDIM;
      int tmp = indices[index1];
      indices[index1] = indices[index2];
      indices[index2] = tmp;
   }


   float* Et = (float*) malloc(MDIM*NDIM*sizeof(float));
   float* E = (float*) malloc(NDIM*MDIM*sizeof(float));


   int numBlocks;
   int numThreads;
   if (MDIM*NDIM <= THREADSPERBLOCK) {
      numBlocks = 1;
      numThreads = MDIM*NDIM;
   }
   else {
      numBlocks = (MDIM*NDIM)/THREADSPERBLOCK + 1;
      numThreads = THREADSPERBLOCK;
   }

   float* gpuEt;
   cudaMalloc( (void**) &gpuEt, MDIM*NDIM*sizeof(float));
   float* gpuE;
   cudaMalloc( (void**) &gpuE, NDIM*MDIM*sizeof(float));
   float* gpuEigvec;
   cudaMalloc( (void**) &gpuEigvec, NDIM*MDIM*sizeof(float));
   cudaMemcpy(gpuEigvec, eigvec, NDIM*MDIM*sizeof(float), cudaMemcpyHostToDevice);  
   int* gpuIndices;
   cudaMalloc( (void**) &gpuIndices, MDIM*sizeof(int));
   cudaMemcpy(gpuIndices, indices, MDIM*sizeof(int), cudaMemcpyHostToDevice);  



   makeProjection<<<numBlocks, numThreads>>>(gpuEt, gpuE, gpuEigvec, gpuIndices, MDIM, NDIM);


   cudaMemcpy(Et, gpuEt, MDIM*NDIM*sizeof(float), cudaMemcpyDeviceToHost);  
   cudaMemcpy(E, gpuE, NDIM*MDIM*sizeof(float), cudaMemcpyDeviceToHost);  

   cudaFree(gpuEt);
   cudaFree(gpuE);
   cudaFree(gpuEigvec);
   cudaFree(gpuIndices);
   for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < MDIM; j++) {
         if (E[i*MDIM+j] != eigvec[i*MDIM+indices[j]])
	    {
               printf("Error for E[%d][%d]: Got %f expected %f\n", i, j, E[i*MDIM+j], eigvec[i*MDIM+indices[j]]);
   free(eigvec);
   free(indices);
	       free(E);
	       free(Et);
	       return false;
	    }
	 if (Et[j*NDIM+i] != eigvec[i*MDIM+indices[j]])
	 {
               printf("Error for Et[%d][%d]: Got %f expected %f\n", j, i, Et[j*NDIM+i], eigvec[i*MDIM+indices[j]]);
   free(eigvec);
   free(indices);
	       free(E);
	       free(Et);
	       return false;
	 }
      }
   }
   free(eigvec);
   free(indices);
	       free(E);
	       free(Et);
   return true;
}

bool testOddEvenEigSort() {
   /*
    * Odd-even sort
   */

   float* eigenvalues = (float*) malloc (NDIM*sizeof(float));
   float* eigenvectors = (float*) malloc (NDIM*NDIM*sizeof(float));

   // Multiples of 20
   for (int i = 0; i < NDIM; i++) {
      eigenvalues[i] = (NDIM-1)*20 - (20*i);
   }

   // Scramble
   for (int i = 0; i < 100; i++) {
      int index1 = rand() % NDIM;
      int index2 = rand() % NDIM;
      float tmp = eigenvalues[index1];
      eigenvalues[index1] = eigenvalues[index2];
      eigenvalues[index2] = tmp;
   }

   
   // Eigenvectors
   for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < NDIM; j++) {
         eigenvectors[j*NDIM+i] = eigenvalues[i];
      }
   }

   float* gpu_eigenvalues;
   float* gpu_eigenvectors;
   cudaMalloc( (void**) &gpu_eigenvalues, NDIM*sizeof(float));
   cudaMemcpy( gpu_eigenvalues, eigenvalues, NDIM*sizeof(float), cudaMemcpyHostToDevice);
   cudaMalloc( (void**) &gpu_eigenvectors, NDIM*NDIM*sizeof(float));
   cudaMemcpy( gpu_eigenvectors, eigenvectors, NDIM*NDIM*sizeof(float), cudaMemcpyHostToDevice);

   int oddcount = NDIM/2;
   int evencount;
   if (NDIM % 2 == 0) evencount = oddcount;
   else evencount = oddcount+1;
   int numBlocks, numThreads;
   for (int i = 0; i < ceil(NDIM/2); i++) {
      if (oddcount <= THREADSPERBLOCK) {
        numBlocks = 1;
        numThreads = oddcount;
      }
      else {
         numBlocks = oddcount/THREADSPERBLOCK + 1;
         numThreads = THREADSPERBLOCK;
      }
      oddEvenEigSort<<<numBlocks, numThreads>>>(gpu_eigenvalues, gpu_eigenvectors, NDIM);
 
     if (evencount <= THREADSPERBLOCK) {
        numBlocks = 1;
        numThreads = evencount;
      }
      else {
         numBlocks = evencount/THREADSPERBLOCK + 1;
         numThreads = THREADSPERBLOCK;
      }
      oddEvenEigSort<<<numBlocks, numThreads>>>(gpu_eigenvalues, gpu_eigenvectors, NDIM, 1);
   }
   
   cudaMemcpy(eigenvalues, gpu_eigenvalues, NDIM*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(eigenvectors, gpu_eigenvectors, NDIM*NDIM*sizeof(float), cudaMemcpyDeviceToHost);


   cudaFree(gpu_eigenvalues);
   cudaFree(gpu_eigenvectors);
   for (int i = 0; i < NDIM-1; i++) {
      if (eigenvalues[i] > eigenvalues[i+1]){
            printf("ERROR.  Unsorted eigenvalues.\n");
	    free(eigenvalues);
	    free(eigenvectors);
	    return false;
	    }
   }


   for (int i = 0; i < NDIM; i++) {
      for (int j = 0; j < NDIM; j++) {
         if (eigenvectors[j*NDIM+i] != eigenvalues[i]) {
	    printf("ERROR.  Eigenvectors did not move properly.  For eigenvalue %d expected %f got %f\n", i, eigenvalues[i], eigenvectors[j*NDIM+i]);
	    free(eigenvalues);
	    free(eigenvectors);
	    return false;
	    }
      }
   }

	    free(eigenvalues);
	    free(eigenvectors);
   return true;
}

#define NUM_PARTICLES 100
#define NUM_DOF 300
#define NUM_BLOCKS 100


bool testBlockCopyFrom() {
// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
   // The special 'eigenvalue' will be -1.
   // All other positive values will be equal to the matrix size.
   int size[5];
   size[0] = rand() % 10 + 1;
   size[1] = rand() % 10 + 1;
   size[2] = rand() % 10 + 1;
   size[3] = rand() % 10 + 1;
   size[4] = rand() % 10 + 1;

   int numatoms = 0;
   int biggestblock = 0;
   for (int i = 0; i < 5; i++) {
      numatoms += size[i];
      if (size[i] > biggestblock)
         biggestblock = size[i];
   }
   


   // Perturb the test
   int* blocnums = (int*) malloc(5*sizeof(int));
   // Populate hessian
   blocnums[0] = 0;
   for (int i = 0; i < 5; i++) {
      if (i > 0) {
         blocnums[i] = blocnums[i-1]+size[i-1];
      }
   }

   float *testpos = (float*)malloc(3*numatoms*sizeof(float));
   float4 *pos = (float4*)malloc(numatoms*sizeof(float4));
   for (int i = 0; i < numatoms; i++) {
	testpos[3*i] = pos[i].x = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+1] = pos[i].y = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+2] = pos[i].z = rand() % 10 + (float) rand() / RAND_MAX + 1;
   }
   
   float* gpu_posorig;
   cudaMalloc(&gpu_posorig, 3*numatoms*sizeof(float));
   cudaMemcpy(gpu_posorig, testpos, 3*numatoms*sizeof(float), cudaMemcpyHostToDevice);

   float delt = (float) rand() / RAND_MAX + 1;
   for (int j = 0; j < 5; j++) {
      int startspot = 3*blocnums[j];
      testpos[startspot+0] += delt;
   }
   float* gpu_pos;
   cudaMalloc(&gpu_pos, numatoms*sizeof(float)*4);
   cudaMemcpy(gpu_pos, pos, numatoms*sizeof(float)*4, cudaMemcpyHostToDevice);
   int* gpu_blocnums;
   cudaMalloc(&gpu_blocnums, 5*sizeof(int));
   cudaMemcpy(gpu_blocnums, blocnums, 5*sizeof(int), cudaMemcpyHostToDevice);
   blockcopyFromOpenMM<<<1, 5>>>(gpu_posorig, gpu_pos, gpu_blocnums, 5, 0, numatoms);
   float4* check3 = (float4*)malloc(numatoms*sizeof(float4));
   cudaMemcpy(check3, gpu_pos, numatoms*sizeof(float4), cudaMemcpyDeviceToHost);
   float* check4 = (float*)malloc(3*numatoms*sizeof(float));
   cudaMemcpy(check4, gpu_posorig, 3*numatoms*sizeof(float), cudaMemcpyDeviceToHost);
   bool res = true;
   for (int i = 0; i < numatoms; i++){
      if (testpos[3*i] != check3[i].x) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i], check3[i].x);res=false;}
      if (testpos[3*i+1] != check3[i].y) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+1], check3[i].y);res=false;}
      if (testpos[3*i+2] != check3[i].z) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+2], check3[i].z);res=false;}
      if (pos[i].x != check4[3*i]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].x, check4[3*i]);res=false;}
      if (pos[i].y != check4[3*i+1]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].y, check4[3*i+1]); res=false;}
      if (pos[i].z != check4[3*i+2]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].z, check4[3*i+2]);res=false;}
   }

   cudaFree(gpu_posorig);
   cudaFree(gpu_pos);
   cudaFree(gpu_blocnums);
   free(blocnums);
   free(testpos);
   free(pos);
   free(check3);
   free(check4);
   return res;


}



bool testCopyFrom() {
    const int numBlocks = NUM_DOF;
    const int numThreads = 1;

    float4 in_positions[NUM_PARTICLES];
    float4 temp;
    for(unsigned int i = 0; i < NUM_PARTICLES; i++) {
        temp.x = (float) i;
    	temp.y = (float) i;
    	temp.z = (float) i;
    	temp.w = 0.0f;
    	in_positions[i] = temp;
    }

    float out_positions[300];
    for(unsigned int i = 0; i < NUM_DOF; i++) {
        out_positions[i] = (float) i;
    }

    float* in_gpu_positions;
    float* out_gpu_positions;

    cudaMalloc( (void**) &in_gpu_positions, NUM_PARTICLES * sizeof(float4) );
    cudaMalloc( (void**) &out_gpu_positions, NUM_DOF * sizeof(float) );

    cudaMemcpy(in_gpu_positions, in_positions, NUM_PARTICLES * sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy(out_gpu_positions, out_positions, NUM_DOF * sizeof(float), cudaMemcpyHostToDevice );

    copyFromOpenMM<<<numBlocks, numThreads>>>(out_gpu_positions, in_gpu_positions, NUM_DOF);

    cudaMemcpy(out_positions, out_gpu_positions, NUM_DOF * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_gpu_positions);
    cudaFree(out_gpu_positions);
    bool result = true;
    for(unsigned int i = 0; i < NUM_PARTICLES; i++) {
       if(out_positions[3 * i] != in_positions[i].x) {
           printf("Atom %d x not equal ( %f %f )", i, out_positions[3*i], in_positions[i].x);
	   result = false;
       }
       if(out_positions[3 * i + 1] != in_positions[i].y)
           result = false;
       if(out_positions[3 * i + 2] != in_positions[i].z)
           result = false;
    }
    
    return result;
}

bool testCopyTo() {
    const int numBlocks = NUM_DOF;
    const int numThreads = 1;

    float4 out_positions[NUM_PARTICLES];
    float4 temp;
    for(unsigned int i = 0; i < NUM_PARTICLES; i++) {
        temp.x = 0.0f;
    	temp.y = 0.0f;
    	temp.z = 0.0f;
    	temp.w = 0.0f;
    	out_positions[i] = temp;
    }

    float in_positions[300];
    for(unsigned int i = 0; i < NUM_DOF; i++) {
        in_positions[i] = (float) (i % 3);
    }

    float* in_gpu_positions;
    float* out_gpu_positions;

    cudaMalloc( (void**) &out_gpu_positions, NUM_PARTICLES * sizeof(float4) );
    cudaMalloc( (void**) &in_gpu_positions, NUM_DOF * sizeof(float) );

    cudaMemcpy(out_gpu_positions, out_positions, NUM_PARTICLES * sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy(in_gpu_positions, in_positions, NUM_DOF * sizeof(float), cudaMemcpyHostToDevice );

    copyToOpenMM<<<numBlocks, numThreads>>>(out_gpu_positions, in_gpu_positions, NUM_DOF);

    cudaMemcpy(out_positions, out_gpu_positions, NUM_PARTICLES * sizeof(float4), cudaMemcpyDeviceToHost);

    cudaFree(in_gpu_positions);
    cudaFree(out_gpu_positions);

    bool result = true;
    for(unsigned int i = 0; i < NUM_PARTICLES; i++) {
       if(in_positions[3 * i] != out_positions[i].x) {
           printf("Atom %d x not equal ( %f %f )", i, in_positions[3*i], out_positions[i].x);
	   result = false;

       }
       if(in_positions[3 * i + 1] != out_positions[i].y)
           result = false;
       if(in_positions[3 * i + 2] != out_positions[i].z)
           result = false;
       }

    return result;
}

bool testSymmetrize2D() {
    const unsigned int size = NUM_DOF * NUM_DOF;
    const unsigned int numThreads = 512;
    const unsigned int numBlocks = size / 512 + 1;
     
    float* in_array = new float[size];
    float* out_array = new float[size];
     
    for(unsigned int i = 0; i < NUM_DOF; i++) {
        for(unsigned int j = 0; j < NUM_DOF; j++) {
            if(i < j)
                in_array[i * NUM_DOF + j] = 0.0f;
            else
                in_array[i * NUM_DOF + j] = 1.0f;
	    out_array[i * NUM_DOF + j] = 0.0f;
        }
    }

    float* gpu_array = NULL;
    cudaMalloc( (void **) &gpu_array, size * sizeof(float) );

    cudaMemcpy(gpu_array, in_array, size * sizeof(float), cudaMemcpyHostToDevice);

    symmetrize2D<<<numBlocks, numThreads>>>(gpu_array, NUM_PARTICLES);

    cudaMemcpy(out_array, gpu_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_array);
    bool result = true;
    for(unsigned int i = 0; i < NUM_DOF; i++) {
        for(unsigned int j = i; j < NUM_DOF; j++) {
            if(out_array[i * NUM_DOF + j] != out_array[j * NUM_DOF + i]) {
                result = false;
		printf("Not equal! %d %d %f %f", i, j, out_array[i*NUM_DOF+j], out_array[j*NUM_DOF+i]);
            }
        }
    }

    return result;
              

}

bool testSymmetrize1D() {
     unsigned int totalSize = 0;
     int blockSizes[NUM_BLOCKS];
     int startDof[NUM_BLOCKS];
     for(unsigned int i = 0; i < NUM_BLOCKS; i++) {
         // vary the sizes by +/- 9
         const int size = 3 * (100 - ((i - 50) % 3));
         blockSizes[i] = size / 3;
         startDof[i] = totalSize;
         totalSize += size * size;
     }

     const unsigned int numThreads = 512;
     const unsigned int numBlocks = NUM_BLOCKS / 512 + 1;    

     float* blockHessian = new float[totalSize];
     for(unsigned int blockNum = 0; blockNum < NUM_BLOCKS; blockNum++) {
         const unsigned int blockSize = 3 * blockSizes[blockNum];
         float *block = &(blockHessian[startDof[blockNum]]);
         for(unsigned int row = 0; row < blockSize; row++) {
             for(unsigned int col = 0; col < blockSize; col++) {
                 if(row < col)
                     block[row * blockSize + col] = 1.0;
                 else
                     block[col * blockSize + row] = 0.0;
             }
         }
     }

     float* gpuBlockHessian = NULL;
     int* gpuBlockSizes = NULL;
     int* gpuBlockPositions = NULL;

     cudaError_t gg = cudaMalloc( (void**) &gpuBlockHessian, totalSize * sizeof(float));
     cudaMalloc( (void**) &gpuBlockSizes, NUM_BLOCKS * sizeof(int));
     cudaMalloc( (void**) &gpuBlockPositions, NUM_BLOCKS * sizeof(int));

     cudaMemcpy(gpuBlockHessian, blockHessian, totalSize * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(gpuBlockSizes, blockSizes, NUM_BLOCKS * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(gpuBlockPositions, startDof, NUM_BLOCKS * sizeof(int),	cudaMemcpyHostToDevice);

     symmetrize1D<<<numBlocks, numThreads>>>(gpuBlockHessian, gpuBlockPositions, gpuBlockSizes, NUM_BLOCKS);

     float* outBlockHessian = new float[totalSize];
    
     cudaMemcpy(outBlockHessian, gpuBlockHessian, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

     cudaFree(gpuBlockHessian);
     cudaFree(gpuBlockSizes);
     cudaFree(gpuBlockPositions);
     bool result = true;
     for(unsigned int blockNum = 0; blockNum < NUM_BLOCKS; blockNum++) {
         const unsigned int blockSize = 3 * blockSizes[blockNum];
         float *block = &(outBlockHessian[startDof[blockNum]]);
         for(unsigned int row = 0; row < blockSize; row++) {
             for(unsigned int col = row; col < blockSize; col++) {
                if(block[row * blockSize + col] != block[col * blockSize + row]) {
		    printf("Not equal! %d %d %d %f %f", blockNum, row, col, block[row*blockSize+col], block[col*blockSize+row]);
                     result = false;
                }
             }
          }
      }
     delete outBlockHessian;
     return result;
}

bool testMatMul() {
    float matrixA[NUM_DOF * NUM_PARTICLES];
    float matrixB[NUM_PARTICLES * NUM_DOF];
    float matrixC[NUM_DOF * NUM_DOF];
    float testMatrixC[NUM_DOF * NUM_DOF];

    const unsigned int numThreads = 512;
    const unsigned int numBlocks = NUM_DOF * NUM_DOF / 512 + 1;

    for(unsigned int i = 0; i < NUM_DOF; i++) {
        for(unsigned int j = 0; j < NUM_PARTICLES; j++) {
            matrixA[i * NUM_PARTICLES + j] = 1.0;
            matrixB[j * NUM_DOF + i] = 3.0;
        }
    }

    
    for (int row = 0; row < NUM_DOF; row++) {
       for (int col = 0; col < NUM_DOF; col++) {
          float result = 0.0;
          for (int elem = 0; elem < NUM_PARTICLES; elem++) {
              result += matrixA[row*NUM_PARTICLES+elem]*matrixB[elem*NUM_DOF+col];
	  }
	  testMatrixC[row*NUM_DOF+col] = result;
       }
    }



   // TODO: multiply

   float* gpuMatrixA = NULL;
   float* gpuMatrixB = NULL;
   float* gpuMatrixC = NULL;
   cudaMalloc( (void**) &gpuMatrixA, NUM_DOF * NUM_PARTICLES * sizeof(float));
   cudaMalloc( (void**) &gpuMatrixB, NUM_DOF * NUM_PARTICLES * sizeof(float));
   cudaMalloc( (void**) &gpuMatrixC, NUM_DOF * NUM_DOF * sizeof(float));

   cudaMemcpy(gpuMatrixA, matrixA, NUM_DOF * NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpuMatrixB, matrixB, NUM_DOF * NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
   //cudaMemcpy(gpuMatrixC, matrixC, NUM_DOF * NUM_DOF * sizeof(float), cudaMemcpyHostToDevice);

   //float D[NUM_DOF*NUM_DOF];
   /*float* D = (float*) malloc(NUM_DOF*NUM_DOF*sizeof(float));
   for (int i = 0; i < NUM_DOF; i++)
      for (int j = 0; j < NUM_DOF; j++)
         D[i*NUM_DOF+j] = 9;
   float* gpuD;// = NULL;
   cudaMalloc( (void**) &gpuD, NUM_DOF*NUM_DOF*sizeof(float));
   printf( "NumBlocks: %d  NumThreads: %d", numBlocks, numThreads);
*/
   MatMulKernel<<<numBlocks, numThreads>>>(gpuMatrixC, gpuMatrixA, gpuMatrixB, NUM_DOF, NUM_PARTICLES, NUM_DOF);
   //MatMulKernel<<<1, 1>>>(gpuE, gpuD, gpuMatrixC, gpuMatrixA, gpuMatrixB, NUM_DOF, NUM_PARTICLES, NUM_DOF);


   cudaMemcpy(matrixC, gpuMatrixC, NUM_DOF * NUM_DOF * sizeof(float), cudaMemcpyDeviceToHost);
   //cudaMemcpy(D, gpuD, NUM_DOF * NUM_DOF * sizeof(float), cudaMemcpyDeviceToHost);



/*
int* E = (int*) malloc(sizeof(int));
   *E = 2;
   int* gpuE;
   cudaError_t returnCode;
   returnCode = cudaMalloc( (void**) &gpuE, sizeof(int) );
   cout << "Cuda Malloc return code success? " << cudaGetErrorString(returnCode) << endl;
   returnCode = cudaMemcpy(gpuE, E, sizeof(int), cudaMemcpyHostToDevice);
   
   cout << "Cuda Malloc return code success? " << (returnCode == cudaSuccess) << endl;
   changeInt<<<1, 1>>>(gpuE);
   cudaMemcpy(E, gpuE, sizeof(int), cudaMemcpyDeviceToHost);
   printf("\nE: %d\n", *E);
*/

   cudaFree(gpuMatrixA);
   cudaFree(gpuMatrixB);
   cudaFree(gpuMatrixC);

   for (int i = 0; i < NUM_DOF; i++)
      for (int j = 0; j < NUM_DOF; j++)
         if (testMatrixC[i*NUM_DOF+j] != matrixC[i*NUM_DOF+j]) {
	    //if (D[i*NUM_DOF+j] != 0) printf("D[%d]: %f\n", i*NUM_DOF+j, D[i*NUM_DOF+j]);
	    printf("Error on element (%d, %d): Expected %f got %f\n", i, j, testMatrixC[i*NUM_DOF+j], matrixC[i*NUM_DOF+j]);
	    return false;
         }
   return true;
}

bool testMakeEigenvalues() {
   // Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
   // The special 'eigenvalue' will be -1.
   // All other positive values will be equal to the matrix size.
   int size[5];
   size[0] = rand() % 10 + 1;
   size[1] = rand() % 10 + 1;
   size[2] = rand() % 10 + 1;
   size[3] = rand() % 10 + 1;
   size[4] = rand() % 10 + 1;

   int eigsize = 0;
   int hesssize = 0;
   
   for (int i = 0; i < 5; i++) {
      eigsize += 3*size[i];
      hesssize += 3*size[i]*3*size[i];
   }

   float* eigenvalues = (float*) malloc(eigsize*sizeof(float));
   float* hessian = (float*) malloc(hesssize*sizeof(float));
   int* blocknums = (int*) malloc(5*sizeof(int));
   int* hessiannums = (int*) malloc(5*sizeof(int));
 
   // Populate hessian
   int pos = 0;
   blocknums[0] = 0;
   hessiannums[0] = 0;
   for (int i = 0; i < 5; i++) {
      if (i > 0) {
         blocknums[i] = blocknums[i-1]+size[i-1];
	 hessiannums[i] = pos;
      }
      for (int j = 0; j < 3*size[i]; j++)
         for (int k = 0; k < 3*size[i]; k++) {
	    if (j != k)
               hessian[pos] = size[i];
	    else
	       hessian[pos] = -12;
	    pos++;
	 }
   }
 

   int numBlocks = eigsize / 512 + 1;
   int numThreads = eigsize % 512;
 
   float* gpuEigs;
   float* gpuHess;
   int* gpuBlockNums;
   int* gpuHessNums;
   int* gpuBlockSizes;

   cudaError_t result;
   result = cudaMalloc( (void**) &gpuEigs, eigsize*sizeof(float) );
   result = cudaMalloc( (void**) &gpuHess, hesssize*sizeof(float) );
   cudaMemcpy(gpuHess, hessian, hesssize*sizeof(float), cudaMemcpyHostToDevice);
   result = cudaMalloc( (void**) &gpuBlockNums, 5*sizeof(int) );
   cudaMemcpy(gpuBlockNums, blocknums, 5*sizeof(int), cudaMemcpyHostToDevice);
   result = cudaMalloc( (void**) &gpuBlockSizes, 5*sizeof(int) );
   cudaMemcpy(gpuBlockSizes, size, 5*sizeof(int), cudaMemcpyHostToDevice);
   result = cudaMalloc( (void**) &gpuHessNums, 5*sizeof(int) );
   cudaMemcpy(gpuHessNums, hessiannums, 5*sizeof(int), cudaMemcpyHostToDevice);
   makeEigenvalues<<<numBlocks, numThreads>>>(gpuEigs, gpuHess, gpuBlockNums, gpuBlockSizes, gpuHessNums, eigsize, 5);

   cudaMemcpy(eigenvalues, gpuEigs, eigsize*sizeof(float), cudaMemcpyDeviceToHost );

   bool res = true;
   for (int i = 0; i < eigsize; i++)
      if (eigenvalues[i] != -12)
         {
            printf("Incorrect eigenvalue: %f\n", eigenvalues[i]);
            res = false;
	 }

   cudaFree(gpuEigs);
   cudaFree(gpuHess);
   cudaFree(gpuBlockNums);
   cudaFree(gpuBlockSizes);
   cudaFree(gpuHessNums);
   free(eigenvalues);
   free(hessian);
   free(blocknums);
   free(hessiannums);
   
   return res;

   

}

bool testMakeBlockHessian() {
   // Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
   // The special 'eigenvalue' will be -1.
   // All other positive values will be equal to the matrix size.
   int size[5];
   size[0] = rand() % 10 + 1;
   size[1] = rand() % 10 + 1;
   size[2] = rand() % 10 + 1;
   size[3] = rand() % 10 + 1;
   size[4] = rand() % 10 + 1;

   int* hessiansizes = (int*) malloc(5*sizeof(int));
   int hesssize = 0;
   int numatoms = 0;
   int biggestblock = 0;
   for (int i = 0; i < 5; i++) {
      numatoms += size[i];
      hesssize += 3*size[i]*3*size[i];
      hessiansizes[i] = 3*size[i]*3*size[i];
      if (size[i] > biggestblock)
         biggestblock = size[i];
   }

   float* hessian = (float*) malloc(hesssize*sizeof(float));
   float* testhessian = (float*) malloc(hesssize*sizeof(float));
   int* blocknums = (int*) malloc(5*sizeof(int));
   int* hessiannums = (int*) malloc(5*sizeof(int));
   // Populate hessian
   int pos = 0;
   blocknums[0] = 0;
   hessiannums[0] = 0;
   for (int i = 0; i < 5; i++) {
      if (i > 0) {
         blocknums[i] = blocknums[i-1]+size[i-1];
	 hessiannums[i] = hessiannums[i-1]+3*size[i-1]*3*size[i-1];
      }
   }
   
   int numBlocks = 5 / 512 + 1;
   int numThreads = 5 % 512;
 
   float blockDelta = 1e-3;
   float* forces1 = (float*) malloc(3*numatoms*sizeof(float));
   float* forces2 = (float*) malloc(3*numatoms*sizeof(float));
   float* masses = (float*) malloc(numatoms*sizeof(float));

   // Assign forces1 random values between 0 and 1
   for (int i = 0; i < 3*numatoms; i++) {
      forces1[i] = (float) rand() / RAND_MAX + 1;
      forces2[i] = (float) rand() / RAND_MAX + 1;
   }

   for (int i = 0; i < numatoms; i++) {
      masses[i] = (rand() % 10) + ((float) rand() / RAND_MAX + 1);
   }
 
   // Compute the full blocks
   // For our test, we'll just assume we are 'perturbing' dof 0
   pos=0;
   for (int j = 0; j < 5; j++) {
      int startdof = 3*blocknums[j];
      int dof = startdof + 0;
      int enddof;
      if (j == 4) enddof = 3*numatoms-1;
      else enddof = 3*blocknums[j+1]-1;
      // The location should be:
      // The starting spot for this hessian, plus:
      // The difference between k and the starting degree of freedom times 
      // the y-dimension of this block hessian (sqrt(the total number of elements), plus:
      // The difference between the perturbed degree of freedom and the starting degree of freedom.
      for (int k = startdof; k <= enddof; k++) {
            testhessian[hessiannums[j]+(k-startdof)*3*size[j]+(dof-startdof)] = (forces1[k] - forces2[k]) * (1.0 / (blockDelta*sqrt(masses[k/3]*masses[dof/3])));
      }
   }



   float* gpu_hess;
   float* gpu_force1;
   float* gpu_force2;
   float* gpu_mass;
   int* gpu_bnums;
   int* gpu_bsizes;
   int* gpu_hnums;
   int* gpu_hsizes;

   cudaMalloc(&gpu_hess, hesssize*sizeof(float));
   cudaMalloc(&gpu_force1, 3*numatoms*sizeof(float));
   cudaMalloc(&gpu_force2, 3*numatoms*sizeof(float));
   cudaMalloc(&gpu_mass, numatoms*sizeof(float));
   cudaMalloc(&gpu_bnums, 5*sizeof(int));
   cudaMalloc(&gpu_bsizes, 5*sizeof(int));
   cudaMalloc(&gpu_hnums, 5*sizeof(int));
   cudaMalloc(&gpu_hsizes, 5*sizeof(int));

   //cudaMemcpy(gpu_hess, testhessian, hesssize*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_force1, forces1, 3*numatoms*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_force2, forces2, 3*numatoms*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_mass, masses, numatoms*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_bnums, blocknums, 5*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_bsizes, size, 5*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_hnums, hessiannums, 5*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_hsizes, hessiansizes, 5*sizeof(int), cudaMemcpyHostToDevice);

   int i = 0; // Assume degree of freedom 0 for now
   makeBlockHessian<<<numBlocks, numThreads>>>(gpu_hess, gpu_force1, gpu_force2, gpu_mass, blockDelta, gpu_bnums, gpu_bsizes, 5, gpu_hnums, gpu_hsizes, i, numatoms); 

   cudaMemcpy(hessian, gpu_hess, hesssize*sizeof(float), cudaMemcpyDeviceToHost);

   cudaFree(gpu_hess);
   cudaFree(gpu_force1);
   cudaFree(gpu_force2);
   cudaFree(gpu_mass);
   cudaFree(gpu_bnums);
   cudaFree(gpu_bsizes);
   cudaFree(gpu_hnums);
   cudaFree(gpu_hsizes);

   bool res = true;
   for (int j = 0; j < 5; j++) {
      int startdof = 3*blocknums[j];
      int dof = startdof + 0;
      int enddof;
      if (j == 4) enddof = 3*numatoms-1;
      else enddof = 3*blocknums[j+1]-1;
      // The location should be:
      // The starting spot for this hessian, plus:
      // The difference between k and the starting degree of freedom times 
      // the y-dimension of this block hessian (sqrt(the total number of elements), plus:
      // The difference between the perturbed degree of freedom and the starting degree of freedom.
      for (int k = startdof; k <= enddof; k++) {
	    int d = hessiannums[j]+(k-startdof)*3*size[j]+(dof-startdof);
	    if (fabs(hessian[d]-testhessian[d]) > 0.0001) {
	       printf("Element %d got %f expected %f\n", d, hessian[d], testhessian[d]);
	       res = false;
	    }
      }
   }
   free(hessiansizes);
   free(hessian);
   free(testhessian);
   free(blocknums);
   free(hessiannums);
   free(forces1);
   free(forces2);
   free(masses);
   return res;
}

bool testPerturbPositions() {

// Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
   // The special 'eigenvalue' will be -1.
   // All other positive values will be equal to the matrix size.
   int size[5];
   size[0] = rand() % 10 + 1;
   size[1] = rand() % 10 + 1;
   size[2] = rand() % 10 + 1;
   size[3] = rand() % 10 + 1;
   size[4] = rand() % 10 + 1;

   int numatoms = 0;
   int biggestblock = 0;
   for (int i = 0; i < 5; i++) {
      numatoms += size[i];
      if (size[i] > biggestblock)
         biggestblock = size[i];
   }
   


   // Perturb the test
   int* blocnums = (int*) malloc(5*sizeof(int));
   // Populate hessian
   blocnums[0] = 0;
   for (int i = 0; i < 5; i++) {
      if (i > 0) {
         blocnums[i] = blocnums[i-1]+size[i-1];
      }
   }

   float *testpos = (float*)malloc(3*numatoms*sizeof(float));
   float4 *pos = (float4*)malloc(numatoms*sizeof(float4));
   for (int i = 0; i < numatoms; i++) {
	testpos[3*i] = pos[i].x = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+1] = pos[i].y = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+2] = pos[i].z = rand() % 10 + (float) rand() / RAND_MAX + 1;
   }
   
   float* gpu_posorig;
   cudaMalloc(&gpu_posorig, 3*numatoms*sizeof(float));
   cudaMemcpy(gpu_posorig, testpos, 3*numatoms*sizeof(float), cudaMemcpyHostToDevice);

   float delt = (float) rand() / RAND_MAX + 1;
   for (int j = 0; j < 5; j++) {
      int startspot = 3*blocnums[j];
      testpos[startspot+0] += delt;
   }
   float4* gpu_pos;
   cudaMalloc(&gpu_pos, numatoms*sizeof(float4));
   cudaMemcpy(gpu_pos, pos, numatoms*sizeof(float4), cudaMemcpyHostToDevice);
   int* gpu_blocnums;
   cudaMalloc(&gpu_blocnums, 5*sizeof(int));
   cudaMemcpy(gpu_blocnums, blocnums, 5*sizeof(int), cudaMemcpyHostToDevice);
   perturbPositions<<<1, 5>>>(gpu_posorig, gpu_pos, delt, gpu_blocnums, 5, 0, numatoms);
   float4* check3 = (float4*)malloc(numatoms*sizeof(float4));
   cudaMemcpy(check3, gpu_pos, numatoms*sizeof(float4), cudaMemcpyDeviceToHost);
   float* check4 = (float*)malloc(3*numatoms*sizeof(float));
   cudaMemcpy(check4, gpu_posorig, 3*numatoms*sizeof(float), cudaMemcpyDeviceToHost);
   bool res = true;
   for (int i = 0; i < numatoms; i++){
      if (testpos[3*i] != check3[i].x) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i], check3[i].x);res=false;}
      if (testpos[3*i+1] != check3[i].y) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+1], check3[i].y);res=false;}
      if (testpos[3*i+2] != check3[i].z) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+2], check3[i].z);res=false;}
      if (pos[i].x != check4[3*i]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].x, check4[3*i]);res=false;}
      if (pos[i].y != check4[3*i+1]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].y, check4[3*i+1]); res=false;}
      if (pos[i].z != check4[3*i+2]) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].z, check4[3*i+2]);res=false;}
   }

   cudaFree(gpu_posorig);
   cudaFree(gpu_pos);
   cudaFree(gpu_blocnums);
   free(blocnums);
   free(testpos);
   free(pos);
   free(check3);
   free(check4);
   return res;
}


bool testPerturbByE() {
   int numatoms = 1000;
   int m = 500;
   // Create initial positions
   float *testpos = (float*)malloc(3*numatoms*sizeof(float));
   float4 *pos = (float4*)malloc(numatoms*sizeof(float4));
   for (int i = 0; i < numatoms; i++) {
	testpos[3*i] = pos[i].x = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+1] = pos[i].y = rand() % 10 + (float) rand() / RAND_MAX + 1;
	testpos[3*i+2] = pos[i].z = rand() % 10 + (float) rand() / RAND_MAX + 1;
   }
   
   // Create delta value
   float delt = (float) rand() / RAND_MAX + 1;

   // Create E
   float *E = (float*) malloc(m*3*numatoms*sizeof(float));
   for (int i = 0; i < 3*numatoms; i++) {
      for (int j = 0; j < m; j++) {
	E[i*m+j] = rand() % 10 + (float) rand() / RAND_MAX + 1;
      }
   }

   // Create masses
   float* masses = (float*) malloc(numatoms*sizeof(float));
   for (int i = 0; i < numatoms; i++) {
      masses[i] = (rand() % 10) + ((float) rand() / RAND_MAX + 1);
   }

   float* gpu_tmppos;
   float4* gpu_pos;
   float* gpu_E;
   float* gpu_masses;

   cudaMalloc((void**)&gpu_tmppos, 3*numatoms*sizeof(float));
   cudaMalloc((void**)&gpu_pos, numatoms*sizeof(float4));
   cudaMemcpy(gpu_pos, pos, numatoms*sizeof(float4), cudaMemcpyHostToDevice);
   cudaMalloc((void**)&gpu_E, m*3*numatoms*sizeof(float)); 
   cudaMemcpy(gpu_E, E, m*3*numatoms*sizeof(float), cudaMemcpyHostToDevice);
   cudaMalloc((void**)&gpu_masses, numatoms*sizeof(float));
   cudaMemcpy(gpu_masses, masses, numatoms*sizeof(float), cudaMemcpyHostToDevice);
   bool res=true;
   for (int k = 0; k < m; k++) {
      int dof = numatoms * 3;
      int numblocks, numthreads;
      if (dof > 512) {
         numblocks = dof / 512 + 1;
	 numthreads = 512;
      }
      else {
         numblocks = 1;
	 numthreads = dof;
      }

      perturbByE<<<numblocks, numthreads>>>(gpu_tmppos, gpu_pos, delt, gpu_E, gpu_masses, k, m, 3*numatoms);
  
      for (int i = 0; i < numatoms; i++) {
         pos[i].x = testpos[3*i];
	 pos[i].y = testpos[3*i+1];
	 pos[i].z = testpos[3*i+2];
         testpos[3*i] += delt*E[3*i*m+k]/sqrt(masses[i]);
	 testpos[3*i+1] += delt*E[(3*i+1)*m+k]/sqrt(masses[i]);
	 testpos[3*i+2] += delt*E[(3*i+2)*m+k]/sqrt(masses[i]);
      }

      float4* check3 = (float4*)malloc(numatoms*sizeof(float4));
      cudaMemcpy(check3, gpu_pos, numatoms*sizeof(float4), cudaMemcpyDeviceToHost);
      float* check4 = (float*)malloc(3*numatoms*sizeof(float));
      cudaMemcpy(check4, gpu_tmppos, 3*numatoms*sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < numatoms; i++){
         if (fabs(testpos[3*i] - check3[i].x) > .01) {printf("3 Error on x-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i], check3[i].x);res=false;}
         if (fabs(testpos[3*i+1] - check3[i].y) > .01) {printf("3 Error on y-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+1], check3[i].y);res=false;}
         if (fabs(testpos[3*i+2] - check3[i].z) > .01) {printf("3 Error on z-coordinate of atom  %i expected %f got %f\n", i, testpos[3*i+2], check3[i].z);res=false;}
         if (fabs(pos[i].x - check4[3*i]) > .01) {printf("4 Error on x-coordinate of atom  %i expected %f got %f\n", i, pos[i].x, check4[3*i]);res=false;}
         if (fabs(pos[i].y - check4[3*i+1]) > .01) {printf("4 Error on y-coordinate of atom  %i expected %f got %f\n", i, pos[i].y, check4[3*i+1]); res=false;}
         if (fabs(pos[i].z - check4[3*i+2]) > .01) {printf("4 Error on z-coordinate of atom  %i expected %f got %f\n", i, pos[i].z, check4[3*i+2]);res=false;}
      }
 
      free(check3);
      free(check4);
   }
   cudaFree(gpu_tmppos);
   cudaFree(gpu_pos);
   cudaFree(gpu_E);
   cudaFree(gpu_masses);
   free(testpos);
   free(pos);
   free(E);
   free(masses);
   return res;
}

bool testGeometricDOF(float* Qi_gdof, float4* positions, float* masses, int* blocknums, int* blocksizes, int bigblock, float* norms, float* centers);
bool testOrthogonalize23(float* Q, float* Qi_gdof, int* blocksizes, int numblocks, int largestsize);



bool testComputeNormsAndCenter() {
   // Pick 5 random sizes, these will be matrices with eigenvalues on the diagonal.
   // The special 'eigenvalue' will be -1.
   // All other positive values will be equal to the matrix size.
   int size[5];
   size[0] = rand() % 10 + 1;
   size[1] = rand() % 10 + 1;
   size[2] = rand() % 10 + 1;
   size[3] = rand() % 10 + 1;
   size[4] = rand() % 10 + 1;

   int numatoms = 0;
   int biggestblock = 0;
   for (int i = 0; i < 5; i++) {
      numatoms += size[i];
      if (size[i] > biggestblock)
         biggestblock = size[i];
   }
   


   // Perturb the test
   int* blocnums = (int*) malloc(5*sizeof(int));
   // Populate hessian
   blocnums[0] = 0;
   for (int i = 0; i < 5; i++) {
      if (i > 0) {
         blocnums[i] = blocnums[i-1]+size[i-1];
      }
   }

   float4 *pos = (float4*)malloc(numatoms*sizeof(float4));
   for (int i = 0; i < numatoms; i++) {
	pos[i].x = rand() % 10 + (float) rand() / RAND_MAX + 1;
	pos[i].y = rand() % 10 + (float) rand() / RAND_MAX + 1;
        pos[i].z = rand() % 10 + (float) rand() / RAND_MAX + 1;
   }
  
   // Create masses
   float* masses = (float*) malloc(numatoms*sizeof(float));
   for (int i = 0; i < numatoms; i++) {
      masses[i] = (rand() % 10) + ((float) rand() / RAND_MAX + 1);
   }

   // Centers
   float4* gpu_pos;
   float* gpu_masses;
   int* gpu_blocknums;
   int* gpu_blocksizes;
   cudaMalloc((void**) &gpu_pos, numatoms*sizeof(float4));
   cudaMemcpy(gpu_pos, pos, numatoms*sizeof(float4), cudaMemcpyHostToDevice);
   cudaMalloc((void**) &gpu_masses, numatoms*sizeof(float));
   cudaMemcpy(gpu_masses, masses, numatoms*sizeof(float), cudaMemcpyHostToDevice);
   cudaMalloc((void**) &gpu_blocknums, 5*sizeof(int));
   cudaMemcpy(gpu_blocknums, blocnums, 5*sizeof(int), cudaMemcpyHostToDevice);
   cudaMalloc((void**) &gpu_blocksizes, 5*sizeof(int));
   cudaMemcpy(gpu_blocksizes, size, 5*sizeof(int), cudaMemcpyHostToDevice);

   float* gpu_centers;
   float* gpu_norms;
   cudaMalloc((void**) &gpu_centers, 5*3*sizeof(float));
   cudaMalloc((void**) &gpu_norms, 5*sizeof(float));
   
   computeNormsAndCenter<<<1, 5>>>(gpu_norms, gpu_centers, gpu_masses, gpu_pos, gpu_blocknums, gpu_blocksizes);

   float* centers = (float*) malloc(15*sizeof(float));
   float* norms = (float*) malloc(5*sizeof(float));

   cudaMemcpy(centers, gpu_centers, 15*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(norms, gpu_norms, 5*sizeof(float), cudaMemcpyDeviceToHost);

   float* c = (float*) malloc(15*sizeof(float));
   float* n = (float*) malloc(5*sizeof(float));

   for (int i = 0; i < 5; i++) {
      float totalmass = 0.0;
      for (int j = blocnums[i]; j <= blocnums[i]+size[i]-1; j += 3 ) {
           float mass = masses[ j / 3 ];
           c[i*3+0] = pos[j/3].x*mass;
	   c[i*3+1] = pos[j/3].y*mass;
	   c[i*3+2] = pos[j/3].z*mass;
	   totalmass += mass;
      }

      n[i] = sqrt(totalmass);
      c[i*3+0] /= totalmass;
      c[i*3+1] /= totalmass;
      c[i*3+2] /= totalmass;
   }

   float* Qi_gdof = (float*) malloc(5*biggestblock*6*sizeof(float));
   float* Qi_gdof2 = (float*) malloc(5*biggestblock*6*sizeof(float));
   float* gpu_qdof;
   cudaMalloc((void**) &gpu_qdof, 5*biggestblock*6*sizeof(float));

   geometricDOF<<<1, 5>>>(gpu_qdof, gpu_pos, gpu_masses, gpu_blocknums, gpu_blocksizes, biggestblock, gpu_norms, gpu_centers);
   cudaMemcpy(Qi_gdof, gpu_qdof, 5*biggestblock*6*sizeof(float), cudaMemcpyDeviceToHost);
   printf("Testing Geometric DOF....\n");
   printf("1\n");
   cudaFree(gpu_pos);
   printf("2\n");
   cudaFree(gpu_masses);
   printf("3\n");
   cudaFree(gpu_blocknums);
   printf("5\n");
   cudaFree(gpu_centers);
   printf("6\n");
   cudaFree(gpu_norms);
   bool result = testGeometricDOF(Qi_gdof, pos, masses, blocnums, size, biggestblock, norms, centers);  
   if (result) printf("Test GeometricDOF Passed\n");
   else printf("Test GeometricDOF Failed\n");
 
   
   orthogonalize23<<<1, 5>>>(gpu_qdof, gpu_blocksizes, 5, biggestblock);
   cudaMemcpy(Qi_gdof2, gpu_qdof, 5*biggestblock*6*sizeof(float), cudaMemcpyDeviceToHost);
   printf("Testing Orthogonalize23... \n");
   result = testOrthogonalize23(Qi_gdof2, Qi_gdof, size, 5, biggestblock);
   if (result) printf("Test Orthogonalize23 Passed\n");
   else printf("Test Orthogonalize23 Failed\n");

   printf("4\n");
   cudaFree(gpu_blocksizes);
   printf("7\n");
   cudaFree(gpu_qdof);
   printf("8\n");
   free(Qi_gdof);
   printf("9\n");
   free(blocnums);
   printf("10\n");
   free(pos);
   printf("11\n");
   free(masses);
   printf("12\n");
   bool res=true;
   for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 3; j++) {
         if( fabs(c[i*3+j] - centers[i*3+j]) > 0.01 )
	     {
                printf("Error in centers (%d, %d): expected %f got %f", i, j, c[i*3+j], centers[i*3+j]);
		res=false;
	     }
      }
      if( fabs(n[i] - norms[i]) > 0.01 )
      {
                printf("Error in norms (%d): expected %f got %f", i, n[i], norms[i]);
		res=false;
      }
   }
   free(c);
   free(n);
   free(centers);
   free(norms);
   return res;
}

bool testMakeHE() {
   int numatoms = 100;
   int m = 50;

   // Create masses
   float* masses = (float*) malloc(numatoms*sizeof(float));
   for (int i = 0; i < numatoms; i++) {
      masses[i] = (rand() % 10) + ((float) rand() / RAND_MAX + 1);
   }

   
   float* force1 = (float*) malloc(3*numatoms*sizeof(float));
   float4* force2 = (float4*) malloc(numatoms*sizeof(float4));

   // Assign forces1 random values between 0 and 1
   for (int i = 0; i < numatoms; i++) {
      force1[3*i] = (float) rand() / RAND_MAX + 1;
      force1[3*i+1] = (float) rand() / RAND_MAX + 1;
      force1[3*i+2] = (float) rand() / RAND_MAX + 1;
      force2[i].x = (float) rand() / RAND_MAX + 1;
      while (force2[i].x == force1[3*i])
         force2[i].x = (float) rand() /RAND_MAX + 1;
      force2[i].y = (float) rand() / RAND_MAX + 1;
      while (force2[i].y == force1[3*i+1])
         force2[i].y = (float) rand() /RAND_MAX + 1;
      force2[i].z = (float) rand() / RAND_MAX + 1;
      while (force2[i].z == force1[3*i+2])
         force2[i].z = (float) rand() /RAND_MAX + 1;
   }

   float delt = (float) rand() / RAND_MAX + 1;

   float* gpu_HE;
   float* gpu_force1;
   float4* gpu_force2;
   float* gpu_masses;

   cudaMalloc((void**) &gpu_HE, 3*numatoms*m*sizeof(float));
   
   cudaMalloc((void**) &gpu_force1, 3*numatoms*sizeof(float));
   cudaMemcpy(gpu_force1, force1, 3*numatoms*sizeof(float), cudaMemcpyHostToDevice);
   
   cudaMalloc((void**) &gpu_force2, numatoms*sizeof(float4));
   cudaMemcpy(gpu_force2, force2, numatoms*sizeof(float4), cudaMemcpyHostToDevice);
   
   cudaMalloc((void**) &gpu_masses, numatoms*sizeof(float));
   cudaMemcpy(gpu_masses, masses, numatoms*sizeof(float), cudaMemcpyHostToDevice);


   float* HE = (float*) malloc(3*numatoms*m*sizeof(float));
   for (int k = 0; k < m; k++) {
      int dof = numatoms*3;
      int numblocks, numthreads;
      if (dof > 512) {
         numblocks = dof / 512 + 1;
	 numthreads = 512;
      }
      else {
         numblocks = 1;
	 numthreads = dof;
      }
      makeHE<<<numblocks, numthreads>>>(gpu_HE, gpu_force1, gpu_force2, gpu_masses, delt, k, m, 3*numatoms);
      cudaMemcpy(HE, gpu_HE, 3*numatoms*m*sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < numatoms; i++) {
          float hex = (force1[3*i] - force2[i].x) / (sqrt(masses[i]) * 1.0 * delt);
          float hey = (force1[3*i+1] - force2[i].y) / (sqrt(masses[i]) * 1.0 * delt);
          float hez = (force1[3*i+2] - force2[i].z) / (sqrt(masses[i]) * 1.0 * delt);

          if (fabs(HE[3*i*m+k] - hex) > 0.01) {
	     printf("%f %f", force1[3*i], force2[i].x);
             printf("Error in HE X (%d, %d): got %f expected %f\n", i, k, HE[3*i*m+k], hex);
	     //return false;
	  }
          if (fabs(HE[(3*i+1)*m+k] - hey) > 0.01) {
	     printf("%f %f", force1[3*i+1], force2[i].y);
             printf("Error in HE Y (%d, %d): got %f expected %f\n", i, k, HE[(3*i+1)*m+k], hey);
	     //return false;
	  }
          if (fabs(HE[(3*i+2)*m+k] - hez) > 0.01) {
	     printf("%f %f", force1[3*i+2], force2[i].z);
             printf("Error in HE Z (%d, %d): got %f expected %f\n", i, k, HE[(3*i+2)*m+k], hez);
	     //return false;
	  }
      }
      

   }
   cudaFree(gpu_HE);
   cudaFree(gpu_force1);
   cudaFree(gpu_force2);
   cudaFree(gpu_masses);
   free(HE);
   free(force1);
   free(force2);
   free(masses);
   return true;
}

bool testOrthogonalize23(float* Q, float* Qi_gdof, int* blocksizes, int numblocks, int largestsize) {
   bool res=true;
   for (int blockNum = 0; blockNum < 5; blockNum++) {
   int size = blocksizes[blockNum];
  // orthogonalize rotational vectors 2 and 3
  for( int j = 4; j < 6; j++ ) { // <-- vector we're orthogonalizing
     for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
        double dot_prod = 0.0;
	for( int l = 0; l < size; l++ ) {
	   dot_prod += Qi_gdof[blockNum*largestsize*6 + l*6 + k] * Qi_gdof[blockNum*largestsize*6 + l*6 + j];
									                  }
	for( int l = 0; l < size; l++ ) {
	   Qi_gdof[blockNum*largestsize*6 + l*6 + j] = Qi_gdof[blockNum*largestsize*6 + l*6 + j] - Qi_gdof[blockNum*largestsize*6 + l*6 + k] * dot_prod;
        }
																			              // normalize residual vector
																				      double rotnorm = 0.0;
	for( int l = 0; l < size; l++ ) {
	   rotnorm += Qi_gdof[blockNum*largestsize*6 + l*6 + j] * Qi_gdof[blockNum*largestsize*6 + l*6 + j];
	}
																				      rotnorm = 1.0 / sqrt( rotnorm );

	for( int l = 0; l < size; l++ ) {
	   Qi_gdof[blockNum*largestsize*6 + l*6 + j] = Qi_gdof[blockNum*largestsize*6 + l*6 + j] * rotnorm;
	}
   }
 }
     for (int j = 0; j < blocksizes[blockNum]; j++) {
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+3] - Q[blockNum*largestsize*6+j*6+3]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 3 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+3], Q[blockNum*largestsize*6+j*6+3]);
	   res=false;
	}
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+4] - Q[blockNum*largestsize*6+j*6+4]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 4 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+4], Q[blockNum*largestsize*6+j*6+4]);
	   res=false;
	}
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+5] - Q[blockNum*largestsize*6+j*6+5]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 5 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+5], Q[blockNum*largestsize*6+j*6+5]);
	   res=false;
	}
     }
}  


	//   free(Qi_gdof);

  return res;
}

bool testGeometricDOF(float* Q, float4* positions, float* masses, int* blocknums, int* blocksizes, int largestsize, float* norm, float* pos_center) {
   float* Qi_gdof = (float*)malloc(5*largestsize*6*sizeof(float));
   bool res=true;
   for (int blockNum = 0; blockNum < 5; blockNum++) {
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


     for (int j = 0; j < blocksizes[blockNum]; j++) {
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+3] - Q[blockNum*largestsize*6+j*6+3]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 3 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+3], Q[blockNum*largestsize*6+j*6+3]);
	   res=false;
	}
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+4] - Q[blockNum*largestsize*6+j*6+4]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 4 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+4], Q[blockNum*largestsize*6+j*6+4]);
	   res=false;
	}
        if (fabs(Qi_gdof[blockNum*largestsize*6+j*6+5] - Q[blockNum*largestsize*6+j*6+5]) > 0.001)
	{
           printf("Error in QDOF block %d, DOF %d VEC 5 expected %f got %f\n", blockNum, j, Qi_gdof[blockNum*largestsize*6+j*6+5], Q[blockNum*largestsize*6+j*6+5]);
	   res=false;
	}
     }
}


	//   free(Qi_gdof);

  return res;
}


int main() 
{
   bool result;
   printf("Testing Make Projection....\n");
   result = testMakeProjection();
   if (result) printf("Make Projection Test Passed\n");
   else printf("Make Projection Test Failed\n");

   printf("Testing Odd-Even Eigenvalue Sort....\n");
   result = testOddEvenEigSort();
   if (result) printf("Odd-Even Eigenvalue Sort Test Passed\n");
   else printf("Odd-Even Eigenvalue Sort Test Failed\n");


   printf("Testing CopyFrom OpenMM....\n");
   result = testCopyFrom();
   if (result) printf("CopyFrom OpenMM Test Passed\n");
   else printf("CopyToFrom OpenMM Test Failed\n");


   printf("Testing CopyTo OpenMM....\n");
   result = testCopyTo();
   if (result) printf("CopyTo OpenMM Test Passed\n");
   else printf("CopyTo OpenMM Test Failed\n");

   printf("Testing Matrix Multiplication....\n");
   result = testMatMul();
   if (result) printf("Matrix Multiplication Test Passed\n");
   else printf("Matrix Multiplication Test Failed\n");

   printf("Testing Symmetrize 2D....\n");
   result = testSymmetrize2D();
   if (result) printf("Symmetrize 2D Test Passed\n");
   else printf("Symmetrize 2D Test Failed\n");


   printf("Testing Symmetrize 1D....\n");
   result = testSymmetrize1D();
   if (result) printf("Symmetrize 1D Test Passed\n");
   else printf("Symmetrize 1D Test Failed\n");
   

   printf("Testing Make Eigenvalues....\n");
   result = testMakeEigenvalues();
   if (result) printf("Make Eigenvalues Test Passed\n");
   else printf("Make Eigenvalues Test Failed\n");

   printf("Testing Make BlockHessian....\n");
   result = testMakeBlockHessian();
   if (result) printf("Make Block Hessian Passed\n");
   else printf("Make Block Hessian Failed\n");
  
   
   printf("Testing Perturb Positions....\n");
   result = testPerturbPositions();
   if (result) printf("Test PeturbPositions Passed\n");
   else printf("Test PerturbPositions Failed\n");
   
   printf("Testing Perturb By E....\n");
   result = testPerturbByE();
   if (result) printf("Test PerturbByE Passed\n");
   else printf("Test PerturbByE Failed\n");

   printf("Testing Compute Norms and Center....\n");
   result = testComputeNormsAndCenter();
   if (result) printf("Test ComputeNormsandCenter Passed\n");
   else printf("Test ComputeNormsAndCenter Failed\n");

   printf("Testing Make HE....\n");
   result = testMakeHE();
   if (result) printf("Test MakeHE Passed\n");
   else printf("Test MakeHE Failed\n");

}


