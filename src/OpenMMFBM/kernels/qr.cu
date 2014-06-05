#include <vector>

__device__ void run8_2_2( float *matA, const int length, int p, int q ) {
	q = q - 1;
	int n = length - p - q;
	int k;
	float tau, c, s, t11, t12, t22;
	float tnn = matA[( p + n - 1 ) * length + p + n - 1];
	float tn1n1 = matA[length * ( p + n - 2 ) + p + n - 2];
	float tnn1 = matA[length * ( p + n - 2 ) + p + n - 1];
	float d = ( tn1n1 - tnn ) / 2;
	float mu = tnn - tnn1 * tnn1 / ( d + ( d > 0.0f ? 1.0f : -1.0f ) * sqrt( d * d + tnn1 * tnn1 ) );
	float x = matA[p * length + p] - mu;
	float z = matA[p * length + p + 1];
	for( k = 0; k < n - 1; k++ ) {
		// "givens" function:
		if( z == 0 ) {
			c = 1;
			s = 0;
		} else {
			if( abs( z ) > abs( x ) ) {
				tau = -x / z;
				s = 1 / sqrt( 1 + tau * tau );
				c = s * tau;
			} else {
				tau = -z / x;
				c = 1 / sqrt( 1 + tau * tau );
				s = c * tau;
			}
		}
		// T = GtTG, G = G(k,k+1,omega) (givens rotation)
		t11 = matA[( p + k ) * length + p + k];
		t12 = matA[( p + k ) * length + p + k + 1];
		t22 = matA[( p + k + 1 ) * length + p + k + 1];
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
	extern __shared__ float v[];
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

/* block_QR( const int n_mat, float *mat, const int* idxs, const int* sizes)
ARGUMENTS:
    n_mat: The number of matrices to be diagonalized
    mat: The matrices to be diagonalized, stored in consecutive memory
    idxs: Array containing indices of the starting point of each matrix
    sizes: Array containing the size of each matrix (total number of elements)
*/
__global__ void block_QR( const int n_mat, float *mat, const int *idxs, const int *sizes ) {
	if( blockIdx.x < n_mat ) {
		int t_width = sizes[blockIdx.x] / blockDim.x + 1;
		hessian_qrf( sizes[blockIdx.x], mat + idxs[blockIdx.x], t_width );
	}
}

extern "C" void BlockQR_HOST( const int n_mat, float *matrix, const size_t matrix_size, const int *index, const size_t index_size, const int *sizes, const size_t sizes_size ) {
	float *matrix_d;
	cudaMalloc( ( void ** )&matrix_d, sizeof( float )*matrix_size );
	cudaMemcpy( matrix_d, matrix, sizeof( float )*matrix_size, cudaMemcpyHostToDevice );

	int *index_d;
	cudaMalloc( ( void ** )&index_d, sizeof( int )*index_size );
	cudaMemcpy( index_d, index, sizeof( int )*index_size, cudaMemcpyHostToDevice );

	int *sizes_d;
	cudaMalloc( ( void ** )&sizes_d, sizeof( int )*sizes_size );
	cudaMemcpy( sizes_d, sizes, sizeof( int )*sizes_size, cudaMemcpyHostToDevice );

	dim3 threads( 16, 16, 1 );
	dim3 blocks( n_mat + 1, 1 );

	block_QR <<< blocks, threads >>>( n_mat, matrix_d, index_d, sizes_d );

	printf( "%f ", matrix[0] );

	// Copy Data Back
	cudaMemcpy( ( void ** )&matrix, matrix_d, sizeof( float )*matrix_size, cudaMemcpyDeviceToHost );

	printf( "%f\n", matrix[0] );

	// Cleanup
	cudaFree( matrix_d );
	cudaFree( index_d );
	cudaFree( sizes_d );
}
