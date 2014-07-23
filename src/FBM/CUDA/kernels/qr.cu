//#include<vector>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/*
__device__ void run8_2_2( float *matA, const int length, int p, int q ) {
	q = q - 1;
	int n = length - p - q;
	float c, s;
	float tnn = matA[( p + n - 1 ) * length + p + n - 1];
	float tn1n1 = matA[length * ( p + n - 2 ) + p + n - 2];
	float tnn1 = matA[length * ( p + n - 2 ) + p + n - 1];
	float d = ( tn1n1 - tnn ) / 2;
	float mu = tnn - tnn1 * tnn1 / ( d + ( d > 0.0f ? 1.0f : -1.0f ) * sqrt( d * d + tnn1 * tnn1 ) );
	float x = matA[p * length + p] - mu;
	float z = matA[p * length + p + 1];
	for( int k = 0; k < n - 1; k++ ) {
		// "givens" function:
		if( z == 0 ) {
			c = 1;
			s = 0;
		} else {
			if( abs( z ) > abs( x ) ) {
				const float tau = -x / z;
				s = 1 / sqrt( 1 + tau * tau );
				c = s * tau;
			} else {
				const float tau = -z / x;
				c = 1 / sqrt( 1 + tau * tau );
				s = c * tau;
			}
		}
		// T = GtTG, G = G(k,k+1,omega) (givens rotation)
		float t11 = matA[( p + k ) * length + p + k];
		float t12 = matA[( p + k ) * length + p + k + 1];
		float t22 = matA[( p + k + 1 ) * length + p + k + 1];
		if( k < n - 2 ) {
			matA[( p + k )*length + p + k + 2] = -s * matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 1 )*length + p + k + 2] = c * matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 2 )*length + p + k + 1] = matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 2 )*length + p + k] = matA[( p + k ) * length + p + k + 2];
		}
		if( k != 0 ) {
			matA[( p + k - 1 )*length + p + k] = c * matA[( p + k - 1 ) * length + p + k] - s * matA[( p + k - 1 ) * length + p + k + 1];
			matA[( p + k )*length + p + k + 1] = matA[( p + k - 1 ) * length + p + k];
			matA[( p + k - 1 )*length + p + k + 1] = 0.0f;
			matA[( p + k + 1 )*length + p + k - 1] = 0.0f;
		}

		matA[( p + k )*length + p + k] = c * c * t11 - 2 * s * c * t12 + s * s * t22;
		matA[( p + k )*length + p + k + 1] = c * s * t11 + ( c * c - s * s ) * t12 - s * c * t22;
		matA[( p + k + 1 )*length + p + k] = matA[( p + k ) * length + p + k + 1];
		matA[( p + k + 1 )*length + p + k + 1] = s * s * t11 + 2 * s * c * t12 + c * c * t22;

		if( k < n - 2 ) {
			x = matA[( p + k + 1 ) * length + p + k];
			z = matA[( p + k + 2 ) * length + p + k];
		}
	}
}

__global__ void run8_2_2_global( float *matA, const int length, int p, int q ) {
	q = q - 1;
	int n = length - p - q;
	float c, s;
	float tnn = matA[( p + n - 1 ) * length + p + n - 1];
	float tn1n1 = matA[length * ( p + n - 2 ) + p + n - 2];
	float tnn1 = matA[length * ( p + n - 2 ) + p + n - 1];
	float d = ( tn1n1 - tnn ) / 2;
	float mu = tnn - tnn1 * tnn1 / ( d + ( d > 0.0f ? 1.0f : -1.0f ) * sqrt( d * d + tnn1 * tnn1 ) );
	float x = matA[p * length + p] - mu;
	float z = matA[p * length + p + 1];
	for( int k = 0; k < n - 1; k++ ) {
		// "givens" function:
		if( z == 0 ) {
			c = 1;
			s = 0;
		} else {
			if( abs( z ) > abs( x ) ) {
				const float tau = -x / z;
				s = 1 / sqrt( 1 + tau * tau );
				c = s * tau;
			} else {
				const float tau = -z / x;
				c = 1 / sqrt( 1 + tau * tau );
				s = c * tau;
			}
		}
		// T = GtTG, G = G(k,k+1,omega) (givens rotation)
		float t11 = matA[( p + k ) * length + p + k];
		float t12 = matA[( p + k ) * length + p + k + 1];
		float t22 = matA[( p + k + 1 ) * length + p + k + 1];
		if( k < n - 2 ) {
			matA[( p + k )*length + p + k + 2] = -s * matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 1 )*length + p + k + 2] = c * matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 2 )*length + p + k + 1] = matA[( p + k + 1 ) * length + p + k + 2];
			matA[( p + k + 2 )*length + p + k] = matA[( p + k ) * length + p + k + 2];
		}
		if( k != 0 ) {
			matA[( p + k - 1 )*length + p + k] = c * matA[( p + k - 1 ) * length + p + k] - s * matA[( p + k - 1 ) * length + p + k + 1];
			matA[( p + k )*length + p + k + 1] = matA[( p + k - 1 ) * length + p + k];
			matA[( p + k - 1 )*length + p + k + 1] = 0.0f;
			matA[( p + k + 1 )*length + p + k - 1] = 0.0f;
		}

		matA[( p + k )*length + p + k] = c * c * t11 - 2 * s * c * t12 + s * s * t22;
		matA[( p + k )*length + p + k + 1] = c * s * t11 + ( c * c - s * s ) * t12 - s * c * t22;
		matA[( p + k + 1 )*length + p + k] = matA[( p + k ) * length + p + k + 1];
		matA[( p + k + 1 )*length + p + k + 1] = s * s * t11 + 2 * s * c * t12 + c * c * t22;

		if( k < n - 2 ) {
			x = matA[( p + k + 1 ) * length + p + k];
			z = matA[( p + k + 2 ) * length + p + k];
		}
	}
}

extern "C" void QRStep822( float *matrix, const size_t matrix_size ) {
	float *matrix_d;
	cudaMalloc( ( void ** )&matrix_d, sizeof( float )*matrix_size );
	cudaMemcpy( matrix_d, matrix, sizeof( float )*matrix_size, cudaMemcpyHostToDevice );

	dim3 threads( 16, 16, 1 );
	dim3 blocks( 1, 1 );

	run8_2_2_global <<< blocks, threads >>>( matrix_d, matrix_size, 0, 0 );

	// Copy Data Back
	cudaMemcpy( ( void ** )&matrix, matrix_d, sizeof( float )*matrix_size, cudaMemcpyDeviceToHost );

	// Cleanup
	cudaFree( matrix_d );
}

// Assuming these things:
// A symmetric positive matrix (from the hessian)
// The matrix will be small (40 to 200)
__device__ void hessian_qrf( const int n, float *A, const int t_width ) {
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	float sum;
	float vTv;
	float pTv;
	float mu;
	const float eps = 0.0000000001f;
	int k, i, cur1, cur2, l, startX, startY;
	__shared__ float v[1024];
	float *p;
	float *w;
	p = v + n;
	w = v + 2 * n;
	// Ensure we are in matrix (we don't have to worry about this anymore)
	if( inx * t_width < n && iny * t_width < n ) {
		// householder Tridiagonalization Algorithm 8.3.1
		// 1. for loop
		for( i = 0; i < n - 2; i++ ) {
			// Keep these to tell if we are in the right spot
			cur1 = i / t_width;
			cur2 = ( i + 1 ) / t_width; // For when we need to check against i+1
			// find the start value for loops once
			startX = ( ( inx * t_width ) > i + 1 ) ? ( inx * t_width ) : ( i + 1 ) ;
			startY = ( ( iny * t_width ) > i + 1 ) ? ( iny * t_width ) : ( i + 1 ) ;
			// Reset these values
			sum = 0.0f;
			vTv = 0.0f;
			pTv = 0.0f;
			// 2. house(A,i); algorithm 5.1.1
			// length(v) = n-i
			// Check if in proper column and below diagonal
			if( iny == cur1 && inx >= cur2 )
				for( k = startX; k < ( inx + 1 )*t_width && k < n; k++ ) {
					v[k] = A[k * n + i];
				}
			__syncthreads();
			for( k = i + 1; k < n; k++ ) {
				sum += v[k] * v[k];
			}
			mu = sqrt( sum );
			// sum = beta
			sum = v[i + 1] + mu * ( v[i + 1] > 0.0f ? 1.0f : -1.0f );
			if( iny == cur1 && inx >= cur2 )
				for( k = startX; k < ( inx + 1 )*t_width && k < n; k++ ) {
					v[k] = v[k] / sum;
				}
			if( inx == cur2 && iny == cur1 ) {
				v[i + 1] = 1.0f;
			}
			__syncthreads();
			// END house
			// 2.5 find v**T v = length of v
			for( k = i + 1; k < n; k++ ) {
				vTv += v[k] * v[k];
			}
			// 3. find p = 2A(i+1:n,i+1:n)v/vTv
			if( inx >= cur2 && iny == cur1 ) {
				for( k = startX; k < ( inx + 1 )*t_width && k < n; k++ ) {
					p[k] = 0.0f;
					for( l = i + 1; l < n; l++ ) {
						p[k] = p[k] + 2.0f * A[( k ) * n + l] * v[l] / vTv;
					}
				}

			}
			__syncthreads();
			// 3.5 find pTv
			for( k = i + 1; k < n; k++ ) {
				pTv += p[k] * v[k];
			}
			__syncthreads();
			// 4. find w
			if( inx >= cur2 && iny == cur1 )
				for( k = i + 1; k < n && k < ( inx + 1 )*t_width; k++ ) {
					w[k] = p[k] - pTv * v[k] / vTv;
				}
			__syncthreads();
			// 5. find new A values
			if( inx >= cur2 && iny >= cur2 ) {
				for( k = startX; k < ( inx + 1 )*t_width && k < n; k++ )
					for( l = startY; l < ( iny + 1 )*t_width && l < n; l++ ) {
						A[k * n + l] = A[k * n + l] - v[k] * w[l] - w[k] * v[l];
					}
			}
			if( iny == cur1 && inx == cur1 ) {
				A[( i + 1 )*n + i] = mu;
				A[( i )*n + i + 1] = mu;
			}
			__syncthreads();
		}
		// QR Diagonalization
		// Algorithm 8.2.3 in Golub
		cur1 = 0; // cur1 = "p"
		cur2 = 0; // cur2 = "q"
		while( cur2 < n ) {
			//1 a[i+1,i] and a[i,i+1] = 0 if a[i,i+1] <= eps(a[i,i]+a[i+1,i+1])
			if( inx == iny ) {
				for( i = inx * t_width; i < ( inx + 1 )*t_width && i < n - 1; i++ ) {
					if( abs( A[i * n + i + 1] ) <= eps * ( abs( A[i * n + i] ) + abs( A[( i + 1 )*n + i + 1] ) ) ) {
						A[i * n + i + 1] = 0.0f;
						A[( i + 1 )*n + i] = 0.0f;
					}
				}
			}
			__syncthreads();
			//2 choose p,q such that T22 is unreduced(no zeros in subdiagonal)
			i = n - 1;
			while( abs( A[i * n + i - 1] ) < eps * ( abs( A[i * n + i] ) + abs( A[( i - 1 )*n + i - 1] ) ) ) {
				i--;
				if( i == 0 ) {
					break;
				}
			}
			cur2 = n - i;

			while( abs( A[i * n + i - 1] ) > eps * ( abs( A[i * n + i] ) + abs( A[( i - 1 )*n + i - 1] ) ) ) {
				i--;
				if( i == 0 ) {
					break;
				}

			}
			cur1 = i;
			//3 if q<n, do run8_2_2 on T22
			if( cur2 < n && iny == inx && inx == 0 ) {
				run8_2_2( A, n, cur1, cur2 );
			}
			__syncthreads();
		}
	}
}
*/
/* block_QR( const int n_mat, float *mat, const int* idxs, const int* sizes)
ARGUMENTS:
    n_mat: The number of matrices to be diagonalized
    mat: The matrices to be diagonalized, stored in consecutive memory
    idxs: Array containing indices of the starting point of each matrix
    sizes: Array containing the size of each matrix (total number of elements)
*/
/*
__global__ void block_QR( const int n_mat, float *mat, const int *idxs, const int *sizes ) {
	if( blockIdx.x < n_mat ) {
		int t_width = sizes[blockIdx.x] / blockDim.x + 1;
		hessian_qrf( sizes[blockIdx.x], mat + idxs[blockIdx.x], t_width );
	}
}
*/
//input: z
//(vector workspaces) size n: vector, vector1
//(matrix workspaces) size n*n: prevm, Newm, Q, NewQ, R, z, z1
//output: m
__global__ void block_QR(float* z, float* z1, float* vector, float* vector1, float* Q, float* NewQ, float* R, float* PrevM, float* NewM, int* converged, int n)
{
*converged = 0;
int iteration = 0;
if(threadIdx.x<n*n){
	int i;
	for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
		z1[i]=z[i];
		Q[i]=z[i];
		PrevM[i]=z[i];
	}
do{
	iteration++;
	int k, j, PowOf2;
	for(k=0;k<n-1;k++){
		//Householder Code
		//STEP 0: Get value of z[k*n+k] for use in step 4
			float NormCheck = z[k*n+k];
		//STEP 1: Find minor matrix of the input matrix z and sets it to z
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i%n==i/n&&i/n<k)z[i]=1;
				else if(i/n>=k&&i%n>=k)z[i]=z[i];
				else z[i]=0;
			}
			__syncthreads();
		//STEP 2: Find kTH column of z and set to vector
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					vector[i] = z[i*n+k];
				}
			}
		//STEP 3: Find the norm of the kTh column and set to NormOfKcol
			float NormOfKcol;
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					//Need a temporary for vector that we can change the values of since we need mcol later
					vector1[i] = vector[i];
					vector1[i] = vector1[i] * vector1[i];
				}
			}
			PowOf2 = 1;
			__syncthreads();
			//add all x's together, 2 at a time. O((log(n)) function
			for(i = 0;i < ((float)n)/2.0;i++){
			      for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
					if(j<n&&j%(PowOf2*2)==0&&j+PowOf2<n){
						vector1[j] = vector1[j] + vector1[j+PowOf2];
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
			NormOfKcol = sqrt(vector1[0]);
		//STEP 4: Make Norm Negative if NormCheck is > 0
			if(NormCheck > 0) NormOfKcol = -NormOfKcol;
		//STEPS 5+6 Combined: add NormOfKcol to tmp[k]
			if(k==threadIdx.x)vector[k]=vector[k]+NormOfKcol;
                        __syncthreads();
		//STEP 7: Finds the addition of the new kcol and stores it in tmp[0]
		//used in ||tmp|| 
		       for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i<n){
					vector1[i] = vector[i] * vector[i];
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
						vector1[j] = vector1[j] + vector1[j+PowOf2];
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
					vector[i] = vector[i]/(sqrt(vector1[0]));
				}
			}
			__syncthreads();
		//STEP 9: Multiply the Vdiv vector by its transverse and subtract that from I, store the resulting matrix in Vmul
		// Vmul = I - 2 * Vdiv * Vdiv^T
			//threadIdx.x%n = column
			//threadIdx.x/n = row (integer division)
		      for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				R[i/n*n+i%n] = -2 * vector[i/n] * vector[i%n];
		       }
			//if on the diagonal(row==column)
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				if(i/n==i%n){
					R[i/n*n+i%n] += 1;
				}
			}
			__syncthreads();
		//STEP 10: Multiply Vmul by input matrix z1 and store in VmulZ
		       for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z[i]=0;
				for(j=0;j<n;j++){
					z[i]+= R[i/n*n+j] * z1[j*n+i%n];
				}
			}	
		//STEP 11: if k!=0 Multiply Vmul by input matrix Q and set to NewQ
			if(k!=0){
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					NewQ[i]=0;
					for(j=0;j<n;j++)
					{
						NewQ[i]+= R[i/n*n+j] * Q[j*n+i%n];
					}
				}
			}
			__syncthreads();
		//STEP 12.1: If first iteration of k, set Q to vmul for use in next iteration of k
			if(k==0){
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					Q[i] = R[i];
				}
			}
		//STEP 12.2: If after first iteration of k, set Q to NewQ, which was found by multiplying the old Q by Vmul.
			else {
				for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
					Q[i] = NewQ[i];
				}
			}
		//STEP 12.3: Set z and z1 to VmulZ for use in the next iteration of k.
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z1[i] = z[i];
			}
			__syncthreads();
	}
	//Once for loop is completed:
	//STEP 13: Multiply matrices Q and m to find the matrix R
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			R[i]=0;
		}
		for(i=0;i<n;i++)
		{
			for(j=threadIdx.x;j<n*n;j=j+blockDim.x){
				R[j]+= Q[j/n*n+i] * PrevM[i*n+j%n];
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
			z[i%n*n+i/n] = Q[i];
		}
		__syncthreads();
	//STEP 15: Multiply matrices R and TransposeOfQ and store in NewM matrix
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
			NewM[i]=0;
			for(j=0;j<n;j++)
			{
			NewM[i]+= R[i/n*n+j] * z[j*n+i%n];
			}
		}
	//STEP 16: Check for Convergence of New Matrix (Newm)
		*converged = 1;
		__syncthreads();
		//threadIdx.x%n = column
		//threadIdx.x/n = row (integer division)
		for(i=threadIdx.x;i<n*n;i=i+blockDim.x){	
			if(i/n==i%n&&(PrevM[i/n*n+i%n]/NewM[i/n*n+i%n]>1.01||PrevM[i/n*n+i%n]/NewM[i/n*n+i%n]<0.99)){
				*converged = 0;
			}
		}
		__syncthreads();
	//STEP 17: Set up for next iteration if converged is 0
		if(*converged==0){
			for(i=threadIdx.x;i<n*n;i=i+blockDim.x){
				z[i] = NewM[i];
				z1[i] = NewM[i];
				Q[i] = NewM[i];
				PrevM[i] = NewM[i];
			}
		}
		__syncthreads();
}while(*converged==0&&iteration<200);
}
}

//Number of matrices, matrix data, size of matrix array, index data, size of index array, widths data, size of widths array
extern "C" void BlockQR_HOST(const int n_mat, float *matrix, const size_t matrix_size, const int *index, const size_t index_size, const int *sizes, const size_t sizes_size ) {
	/*
	float *matrix_d;
	cudaMalloc( ( void ** )&matrix_d, sizeof( float )*matrix_size );
	cudaMemcpy( matrix_d, matrix, sizeof( float )*matrix_size, cudaMemcpyHostToDevice );
	int *index_d;
	cudaMalloc( ( void ** )&index_d, sizeof( int )*index_size );
	cudaMemcpy( index_d, index, sizeof( int )*index_size, cudaMemcpyHostToDevice );
	int *sizes_d;
	cudaMalloc( ( void ** )&sizes_d, sizeof( int )*sizes_size );
	cudaMemcpy( sizes_d, sizes, sizeof( int )*sizes_size, cudaMemcpyHostToDevice );
	*/
	//Set variables for CUDA use
		int* d_converged = NULL;	
		float* d_PrevM=NULL;
		float* d_NewM=NULL;
		float* d_Q=NULL;
		float* d_NewQ=NULL;
		float* d_R = NULL;
		float* d_vector = NULL;
		float* d_vector1 = NULL;
		float* d_z = NULL;
		float* d_z1 = NULL;
		cudaMalloc((void **)&d_converged, sizeof(int));
		cudaMalloc((void **)&d_vector, sizeof(float) * sizes[0]);                
		cudaMalloc((void **)&d_vector1, sizeof(float) * sizes[0]);
		cudaMalloc((void **)&d_PrevM, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_NewM, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_Q, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_NewQ, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_R, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_z, sizeof(float) * sizes[0] * sizes[0]);
		cudaMalloc((void **)&d_z1, sizeof(float) * sizes[0] * sizes[0]);
	//Get CUDA device properties
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);
		int threads = props.maxThreadsPerBlock;
	//copy input matrix data into z
		cudaMemcpy(d_z, matrix, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
	//(float* z, float* z1, float* vector, float* vector1, float* Q, float* NewQ, float* R, float* PrevM, float* NewM, int* converged, int n)
		block_QR <<<n_mat, threads >>>(d_z,d_z1,d_vector,d_vector1,d_Q, d_NewQ, d_R, d_PrevM, d_NewM, d_converged, sizes[0]);
		cudaMemcpy(matrix, d_NewM, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
	// Copy Data Back
	printf( "%f\n", matrix[0] );
	// Cleanup
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
