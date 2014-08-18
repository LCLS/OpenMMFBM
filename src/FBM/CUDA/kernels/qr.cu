#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//Input Matrix: z
//(vector workspaces) size n: vector, vector1
//(matrix workspaces) size n*n: prevm, Newm, Q, NewQ, R, z, z1
//Output for eigenvectors: eigenvector
//Output for eigenvalues: vector
__global__ void block_QR(float* z, float* z1, float* vector, float* vector1, float* Q, float* NewQ, float* R, float* PrevM, float* NewM, int* converged, float* eigenvector, const int *WidthOfMatrix, const int *ind, const int *vind)
{
int n = WidthOfMatrix[blockIdx.x];
int index = ind[blockIdx.x];
int vectindex = vind[blockIdx.x];
*converged = 0;
int iteration = 0;
if(threadIdx.x<n*n){
	int i;
	for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
		//set eigenvector to the identity matrix.
		if(i/n==i%n)eigenvector[i+index]=1;
		else eigenvector[i+index]=0;
	}
	for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
		z1[i+index]=z[i+index];
		Q[i+index]=z[i+index];
		PrevM[i+index]=z[i+index];
	}
do{
	iteration++;
	int k, j, PowOf2;
	for(k=0;k<n-1;k++){
		//Householder Code
		//STEP 0: Get value of z[k*n+k] for use in step 4
			float NormCheck = z[k*n+k+index];
		//STEP 1: Find minor matrix of the input matrix z and sets it to z
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i%n==i/n&&i/n<k)z[i+index]=1;
				else if(i/n>=k&&i%n>=k)z[i+index]=z[i+index];
				else z[i+index]=0;
			}
			__syncthreads();
		//STEP 2: Find kTH column of z and set to vector
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					vector[i+vectindex] = z[i*n+k+index];
				}
			}
		//STEP 3: Find the norm of the kTh column and set to NormOfKcol
			float NormOfKcol;
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					//Need a temporary for vector that we can change the values of since we need mcol later
					vector1[i+vectindex] = vector[i+vectindex];
					vector1[i+vectindex] = vector1[i+vectindex] * vector1[i+vectindex];
				}
			}
			PowOf2 = 1;
			__syncthreads();
			//add all x's together, 2 at a time. O((log(n)) function
			for(i = 0;i < ((float)n)/2.0;i++){
			      for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
					if(j<n&&j%(PowOf2*2)==0&&j+PowOf2<n){
						vector1[j+vectindex] = vector1[j+vectindex] + vector1[j+PowOf2+vectindex];
					}
			      }
				__syncthreads();
				//PowOf2 = pow(2,i)
			      for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
					if(j<n){
						PowOf2 = PowOf2 * 2;
					}
			      }
			}
			NormOfKcol = sqrt(vector1[0+vectindex]);
		//STEP 4: Make Norm Negative if NormCheck is > 0
			if(NormCheck > 0) NormOfKcol = -NormOfKcol;
		//STEPS 5+6 Combined: add NormOfKcol to tmp[k]
			if(k==threadIdx.x)vector[k+vectindex]=vector[k+vectindex]+NormOfKcol;
                        __syncthreads();
		//STEP 7: Finds the addition of the new kcol and stores it in tmp[0]
		//used in ||tmp|| 
		       for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					vector1[i+vectindex] = vector[i+vectindex] * vector[i+vectindex];
				}
				if(i<n){
					PowOf2 = 1;
				}
			}
			__syncthreads();
			//add all tmp's together, 2 at a time. O(n(log(n)) function
			for(i = 0;i < ((float)n)/2.0;i++){
			      for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
					if(j<n&&j%(PowOf2*2)==0&&j+PowOf2<n){
						vector1[j+vectindex] = vector1[j+vectindex] + vector1[j+PowOf2+vectindex];
					}
			      }
				__syncthreads();
				//PowOf2 = pow(2,i)
			      for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
					if(j<n){
						PowOf2 = PowOf2 * 2;
					}
			      }
			}
			__syncthreads();
		//STEP 8: Divide vector Vmadd by the Norm[0] and set it to Vdiv
		// Vdiv = Vmadd / norm 
		      for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					vector[i+vectindex] = vector[i+vectindex]/(sqrt(vector1[0+vectindex]));
					//vector[i+vectindex] = 20;
				}
			}
			__syncthreads();
		//STEP 9: Multiply the Vdiv vector by its transverse and subtract that from I, store the resulting matrix in Vmul
		// Vmul = I - 2 * Vdiv * Vdiv^T
			//threadIdx.x%n = column
			//threadIdx.x/n = row (integer division)
		      for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				R[i/n*n+i%n+index] = -2 * vector[i/n+vectindex] * vector[i%n+vectindex];
		       }
			//if on the diagonal(row==column)
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i/n==i%n){
					R[i/n*n+i%n+index] += 1;
				}
			}
			__syncthreads();
		//STEP 10: Multiply Vmul by input matrix z1 and store in VmulZ
		       for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z[i+index]=0;
				for(j=0;j<n;j++){
					z[i+index]+= R[i/n*n+j+index] * z1[j*n+i%n+index];
				}
			}	
		//STEP 11: if k!=0 Multiply Vmul by input matrix Q and set to NewQ
			if(k!=0){
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					NewQ[i+index]=0;
					for(j=0;j<n;j++)
					{
						NewQ[i+index]+= R[i/n*n+j+index] * Q[j*n+i%n+index];
					}
				}
			}
			__syncthreads();
		//STEP 12.1: If first iteration of k, set Q to vmul for use in next iteration of k
			if(k==0){
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					Q[i+index] = R[i+index];
				}
			}
		//STEP 12.2: If after first iteration of k, set Q to NewQ, which was found by multiplying the old Q by Vmul.
			else {
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					Q[i+index] = NewQ[i+index];
				}
			}
		//STEP 12.3: Set z and z1 to VmulZ for use in the next iteration of k.
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z1[i+index] = z[i+index];
			}
			__syncthreads();
	}
	//Once for loop is completed:
	//STEP 13: Multiply matrices Q and m to find the matrix R
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			R[i+index]=0;
		}
		for(i=0;i<n;i++)
		{
			for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
				R[j+index]+= Q[j/n*n+i+index] * PrevM[i*n+j%n+index];
			}
		}
		__syncthreads();
	//STEP 14: Find the transpose of matrix Q and store int TransposeOfQ
		//threadIdx.x%n = column
		//threadIdx.x/n = row (integer division)
		// # -> #%n*n+#/n
		// for n=4		0->0  1->4  2->8   3->12
		//			4->1  5->5  6->9   7->13
		//			8->2  9->6  10->10 11->14
		//			12->3 13->7 14->11 15->15
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			z[i%n*n+i/n+index] = Q[i+index];
		}
		__syncthreads();
	//STEP 14.5: Multiply matrices eigenvector and TransposeOfQ and store in eigenvector(use NewM as a temporary matrix)
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			NewM[i+index]=0;
			for(j=0;j<n;j++){
				NewM[i+index]+= eigenvector[i/n*n+j+index] * z[j*n+i%n+index];
			}
		}
		__syncthreads();
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			eigenvector[i+index]=NewM[i+index];
		}
		__syncthreads();
	//STEP 15: Multiply matrices R and TransposeOfQ and store in NewM matrix
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			NewM[i+index]=0;
			for(j=0;j<n;j++)
			{
				NewM[i+index]+= R[i/n*n+j+index] * z[j*n+i%n+index];
			}
		}
	//STEP 16: Check for Convergence of New Matrix (Newm)
		converged[blockIdx.x] = 1;
		__syncthreads();
		//threadIdx.x%n = column
		//threadIdx.x/n = row (integer division)
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){	
			if(i/n==i%n&&(PrevM[i/n*n+i%n+index]/NewM[i/n*n+i%n+index]>1||PrevM[i/n*n+i%n+index]/NewM[i/n*n+i%n+index]<1)){
				converged[blockIdx.x] = 0;
			}
		}
		__syncthreads();
	//STEP 17: Set up for next iteration if converged is 0
		if(converged[blockIdx.x]==0){
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z[i+index] = NewM[i+index];
				z1[i+index] = NewM[i+index];
				Q[i+index] = NewM[i+index];
				PrevM[i+index] = NewM[i+index];
			}
		}
		__syncthreads();
}while(converged[blockIdx.x]==0&&iteration<200);
//put eigenvalues into vector
if(threadIdx.x<n){
       vector[threadIdx.x+vectindex]=NewM[threadIdx.x+threadIdx.x*n+index];
}
	if(threadIdx.x==0){
	//Sort Eigenvalues low to high and swap eigenvectors to match eigenvalues
	//Simple Bubble Sort
		int i1,i2,i3;
		for(i1=vectindex;i1<n-1+vectindex;i1++){
			for(i2=i1+1;i2<n+vectindex;i2++){
				if(vector[i1]>vector[i2]){
					float tmp = vector[i1];
					vector[i1] = vector[i2];
					vector[i2] = tmp;
					for(i3 = 0;i3<n;i3++){
						float tmp = eigenvector[i3*n+(i1-vectindex)%n+index];
						eigenvector[i3*n+(i1-vectindex)%n+index] = eigenvector[i3*n+(i2-vectindex)%n+index];
						eigenvector[i3*n+(i2-vectindex)%n+index] = tmp;
					}
				}
			}
		}
	}
}
}

//Number of matrices, matrix data, size of matrix array, matrix index data, eigenvalue index data , widths data, empty vector for eigenvalues
extern "C" void BlockQR_HOST(const int n_mat, float *matrix, const size_t matrix_size, const int *index, const int *eigenvaluesindex, const int *sizes, float *eigenvalues ) {
	//Set up timing variables
		struct timeval start, finish;
        	struct timezone tz;
	//Set variables for CUDA use
		int* d_converged = NULL;
		int* d_sizes = NULL;
		int* d_eigenvaluesindex = NULL;
		int* d_index = NULL;	
		float* d_PrevM=NULL;
		float* d_NewM=NULL;
		float* d_Q=NULL;
		float* d_NewQ=NULL;
		float* d_R = NULL;
		float* d_vector = NULL;
		float* d_vector1 = NULL;
		float* d_z = NULL;
		float* d_z1 = NULL;
		float* d_eigenvector = NULL;
	//Calculate vector size needed for all matrices(add all the widths together)
	int vector_size = 0;
	int i;
	for(i=0;i<n_mat;i++){
		vector_size+=sizes[i];
	}
		cudaMalloc((void **)&d_converged, sizeof(int) * n_mat);
		cudaMalloc((void **)&d_sizes, sizeof(int) * n_mat);
		cudaMalloc((void **)&d_eigenvaluesindex, sizeof(int) * n_mat);
		cudaMalloc((void **)&d_index, sizeof(int) * n_mat);
		cudaMalloc((void **)&d_vector, sizeof(float) * vector_size);                
		cudaMalloc((void **)&d_vector1, sizeof(float) * vector_size);
		cudaMalloc((void **)&d_PrevM, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_NewM, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_Q, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_NewQ, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_R, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_z, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_z1, sizeof(float) * matrix_size);
		cudaMalloc((void **)&d_eigenvector, sizeof(float) * matrix_size);
	//Get CUDA device properties
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);
		int threads = props.maxThreadsPerBlock;
	//copy input matrix data into z
		cudaMemcpy(d_z, matrix, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_sizes, sizes, sizeof(int)*n_mat, cudaMemcpyHostToDevice);
		cudaMemcpy(d_index, index, sizeof(int)*n_mat, cudaMemcpyHostToDevice);
		cudaMemcpy(d_eigenvaluesindex, eigenvaluesindex, sizeof(int)*n_mat, cudaMemcpyHostToDevice);
	//Time and run kernel
		gettimeofday(&start, &tz);
		block_QR <<<n_mat, threads>>>(d_z,d_z1,d_vector,d_vector1,d_Q, d_NewQ, d_R, d_PrevM, d_NewM, d_converged, d_eigenvector, d_sizes, d_index, d_eigenvaluesindex);
		gettimeofday(&finish, &tz);
	//copy back data
		cudaMemcpy(eigenvalues, d_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(matrix, d_eigenvector, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
     	/*
        // Debugg
		// Vector is corrupt
                float* vec = (float*)malloc(vector_size*sizeof(float));
                cudaMemcpy(vec, d_vector, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
                printf("VEC\n");
                for (i = 0; i < vector_size; i++)
                        printf("%f ", vec[i]);
		// Vector1 is corrupt
		float* vec1 = (float*)malloc(vector_size*sizeof(float));
		cudaMemcpy(vec1, d_vector1, sizeof(float) * vector_size, cudaMemcpyDeviceToHost);
		printf("VEC1\n");
		for (i = 0; i < vector_size; i++)
			printf("%f ", vec1[i]);
		// Q is corrupt
		float* q = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(q, d_Q, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("Q\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", q[i]);
		// NewQ is corrupt
		float* newq = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(newq, d_NewQ, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nNEWQ\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", newq[i]);
		// R is corrupt
		float* r= (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(r, d_R, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nR\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", r[i]);
		// Z is corrupt
		float* z = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(z, d_z, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nZ\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", z[i]);
		// Z1 is corrupt
		float* z1 = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(z1, d_z1, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nZ1\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", z1[i]);
		// PrevM is not corrupt
		float* prevm = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(prevm, d_PrevM, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nPREVM\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", prevm[i]);
		// NewM is corrupt
		float* newm = (float*)malloc(matrix_size*sizeof(float));
		cudaMemcpy(newm, d_NewM, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
		printf("\nNEWM\n");
		for (i = 0; i < matrix_size; i++)
			printf("%f ", newm[i]);
	*/
	//Print Time of run
		double elapsed = ( finish.tv_sec - start.tv_sec ) * 1000.0 + ( finish.tv_usec - start.tv_usec ) / 1000.0;
       		printf("Time to cluster: %lf ms\n", elapsed);
	// Cleanup
	cudaFree(d_eigenvector);
	cudaFree(d_sizes);
	cudaFree(d_eigenvaluesindex);
	cudaFree(d_index);
	cudaFree(d_converged);
	cudaFree(d_PrevM);
	cudaFree(d_NewM);
	cudaFree(d_Q);
	cudaFree(d_NewQ);
	cudaFree(d_R);
	cudaFree(d_vector);
	cudaFree(d_vector1);
	cudaFree(d_z);
	cudaFree(d_z1);
	cudaDeviceReset();
}
