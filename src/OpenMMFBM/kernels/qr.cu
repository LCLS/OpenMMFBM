__device__ void run8_2_2_eVects(float* C,float* v,const int length,int p,int q, float* A,int t_width){
    if(q != 0)
	q = q-1;
    int n=length-p-q;
    int k, startX, startY, i;
    int inx = threadIdx.x, iny = threadIdx.y;
    startX = inx*t_width;
    startY = iny*t_width;
    float tau,c,s,t11,t12,t22;
    t22 = C[p+n-1];
    t11 = C[p+n-2];
    t12 = v[p+n-1];
    float d = (t11 - t22)/2;
    float mu = t22 - t12*t12/(d+(d>0.0f ? 1.0f:-1.0f)*sqrt(d*d + t12*t12));
    float x = C[p] - mu;
    float z = v[p];
    // We are repurposing d, so set it to zero
    d = 0.0f;
    for(k=0;k<n-1;k++){
	// "givens" function:
	if(z == 0.0f){
	    c = 1.0f;
	    s = 0.0f;
    	}
	else{
	    if(abs(z) > abs(x)){
    		tau = -x/z;
	    	s = 1/sqrt(1+tau*tau);
		c = s*tau;
	    }
	    else{
    		tau = -z/x;
		c = 1/sqrt(1+tau*tau);
		s = c*tau;
	    }
	}
	// T = GtTG, G = G(k,k+1,omega) (givens rotation)
	// Before we start changing stuff, sync up
	if(k < n-2)
    	    z = -s*v[p+k+1];
	__syncthreads();
	if(inx == iny && inx == 1){
	    t11 = C[p+k];
	    t12 = v[p+k];
	    t22 = C[p+k+1];
	    if(k != 0){
		v[p+k-1] = c*v[p+k-1]-s*d;
	    }
	    if(k < n-2){
		v[p+k+1] = c*v[p+k+1];
	    }
	    C[p+k] = c*c*t11 - 2*s*c*t12 + s*s*t22;
	    v[p+k] = c*s*t11 + (c*c-s*s)*t12 - s*c*t22;
	    C[p+k+1] = s*s*t11 + 2*s*c*t12 + c*c*t22;
	}
	else if(inx == 0 && iny == 0){
	    // use these to hold things while we change them
	    t11 = A[length*(p+k)+p+k];
	    tau = A[length*(p+k+1)+p+k]; // Here, tau is t21 - no symmetry
	    t12=A[length*(p+k)+p+k+1];
	    t22=A[length*(p+k+1)+p+k+1];
	    // Now, make changes
	    A[length*(p+k)+p+k]=c*c*t11 + s*c*tau - s*c*t12 - s*s*t22;
	    A[length*(p+k+1)+p+k]=-s*c*t11 + c*c*tau + s*s*t12 - s*c*t22;
	    A[length*(p+k)+p+k+1]=s*c*t11 + s*s*tau + c*c*t12 + s*c*t22;
	    A[length*(p+k+1)+p+k+1]=-s*s*t11 + c*s*tau -s*c*t12 + c*c*t22;
	    for(i=startX+2;i<startX+t_width;i++){
		t11 = A[length*(p+k) + (p+k+i)%length];
		tau = A[length*(p+k+1) + (p+k+i)%length];
		A[length*(p+k) + (p+k+i)%length] = c*t11 + s*tau;
		A[length*(p+k+1) + (p+k+i)%length] = -s*t11 + c*tau;
	    }
	    for(i=startY+2;i<startY+t_width;i++){
		t11 = A[length*((p+k+i)%length) + p+k];
		t12 = A[length*((p+k+i)%length) + p+k+1];
		A[length*((p+k+i)%length) + p+k] = c*t11 - s*t12;
		A[length*((p+k+i)%length) + p+k+1] = s*t11 + c*t12;
	    }
	}
	else if(inx == 0){
	    // In order to keep the previous loop in the middle, we have to do
	    // some indexing tricks.  Basically, we just wrap around.
	    for(i=startX;i<startX+t_width && i<length;i++){
		t11 = A[length*(p+k) + (p+k+i)%length];
		tau = A[length*(p+k+1) + (p+k+i)%length];
		A[length*(p+k) + (p+k+i)%length] = c*t11 + s*tau;
		A[length*(p+k+1) + (p+k+i)%length] = -s*t11 + c*tau;
	    }
	}
	else if(iny == 0){
	    for(i=startY;i<startY+t_width && i<length;i++){
		t11 = A[length*((p+k+i)%length) + p+k];
		tau = A[length*((p+k+i)%length) + p+k+1];
		A[length*((p+k+i)%length) + p+k] = c*t11 - s*t12;
		A[length*((p+k+i)%length) + p+k+1] = s*t11 + c*t12;
	    }
	}

	__syncthreads();
    	if(k < n-2){
	    x = v[p+k];
	    d = z;
	}
    }
}

__device__ void run8_2_2(float* matA,const int length,int p,int q){
    if(q != 0)
	q = q-1;
    int n=length-p-q;
    int k;
    float tau,c,s,t11,t12,t22;
    t22 = matA[(p+n-1)*length+p+n-1];
    t11 = matA[length*(p+n-2)+p+n-2];
    t12 = matA[length*(p+n-2)+p+n-1];
    float d = (t11 - t22)/2;
    float mu = t22 - t12*t12/(d+(d>0.0f ? 1.0f:-1.0f)*sqrt(d*d + t12*t12));
    float x = matA[p*length+p] - mu;
    float z = matA[p*length+p+1];
    for(k=0;k<n-1;k++){
	// "givens" function:
	if(z == 0){
	    c = 1;
	    s = 0;
    	}
	else{
	    if(abs(z) > abs(x)){
    		tau = -x/z;
	    	s = 1/sqrt(1+tau*tau);
		c = s*tau;
	    }
	    else{
    		tau = -z/x;
		c = 1/sqrt(1+tau*tau);
		s = c*tau;
	    }
	}
	// T = GtTG, G = G(k,k+1,omega) (givens rotation)
	t11 = matA[(p+k)*length+p+k];
	t12 = matA[(p+k)*length+p+k+1];
	t22 = matA[(p+k+1)*length+p+k+1];
	if(k < n-2){
	    matA[(p+k)*length+p+k+2] = -s*matA[(p+k+1)*length+p+k+2];
	    matA[(p+k+1)*length+p+k+2] = c*matA[(p+k+1)*length+p+k+2];
    	    matA[(p+k+2)*length+p+k+1] = matA[(p+k+1)*length+p+k+2];
    	    matA[(p+k+2)*length+p+k] = matA[(p+k)*length+p+k+2];
	}
	if(k != 0){
	    matA[(p+k-1)*length+p+k] = c*matA[(p+k-1)*length+p+k]-s*matA[(p+k-1)*length+p+k+1];
	    matA[(p+k)*length+p+k+1] = matA[(p+k-1)*length+p+k];
	    matA[(p+k-1)*length+p+k+1]=0.0f;
	    matA[(p+k+1)*length+p+k-1]=0.0f;
	}

	matA[(p+k)*length+p+k] = c*c*t11 - 2*s*c*t12 + s*s*t22;
	matA[(p+k)*length+p+k+1] = c*s*t11 + (c*c-s*s)*t12 - s*c*t22;
	matA[(p+k+1)*length+p+k] = matA[(p+k)*length+p+k+1];
	matA[(p+k+1)*length+p+k+1] = s*s*t11 + 2*s*c*t12 + c*c*t22;

	if(k < n-2){
	    x = matA[(p+k+1)*length+p+k];
	    z = matA[(p+k+2)*length+p+k];
	}
    }
}

// Assuming these things:
// A symmetric positive matrix (from the hessian)
// The matrix will be small (40 to 200)
// This function returns the eigenvalues and eigenvectors
__device__ void hessian_qrf( const int n, float *A, const int t_width, float *C, const float eps){
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    float sum;
    float vTv;
    float pTv;
    float mu;
    int k,i,cur1,cur2,l,startX,startY;
    extern __shared__ float v[];
    float * p;
    float * w;
    p = v + n;
    w = v + 2*n;
    // Ensure we are in matrix (we don't have to worry about this anymore)
    if(inx*t_width<n && iny*t_width<n){
	// householder Tridiagonalization Algorithm 8.3.1
	// 1. for loop
	for(i=0;i<n-2;i++){
	    // Keep these to tell if we are in the right spot
	    cur1 = i/t_width;
	    cur2 = (i+1)/t_width; // For when we need to check against i+1
	    // find the start value for loops once
	    startX =((inx*t_width) > i+1) ? (inx*t_width) : (i+1) ;
	    startY =((iny*t_width) > i+1) ? (iny*t_width) : (i+1) ;
	    // Reset these values
	    sum=0.0f;
	    vTv=0.0f;
    	    pTv=0.0f;
	    // 2. house(A,i); algorithm 5.1.1
	    // length(v) = n-i
	    // Check if in proper column and below diagonal
	    if(iny == cur1 && inx >= cur2)
		for(k = startX;k<(inx+1)*t_width && k<n;k++)
		    v[k] = A[k*n+i];
	    __syncthreads();
    	    for(k=i+1;k<n;k++)
		sum+=v[k]*v[k];
	    mu = sqrt(sum);
	    // sum = beta
	    sum = v[i+1]+mu*(v[i+1]>0.0f ? 1.0f:-1.0f);
	    if(iny == cur1 && inx >= cur2)
		for(k = startX;k<(inx+1)*t_width && k<n;k++)
		    v[k] = v[k]/sum;
	    if(inx==cur2 && iny==cur1)
		v[i+1]=1.0f;
	    __syncthreads();
	    // END house
	    // 2.5 find v**T v = length of v
	    for(k=i+1;k<n;k++)
    		vTv += v[k]*v[k];
	    // 3. find p = 2A(i+1:n,i+1:n)v/vTv
	    if(inx>=cur2 && iny==cur1){
		for(k = startX;k<(inx+1)*t_width && k<n;k++){
		    p[k]=0.0f;
		    for(l=i+1;l<n;l++)
			p[k]=p[k]+2.0f*A[(k)*n+l]*v[l]/vTv;
		}
	    }
	    __syncthreads();
	    // 3.5 find pTv
	    for(k=i+1;k<n;k++)
    		pTv += p[k]*v[k];
	    __syncthreads();
	    // 4. find w
	    if(inx>=cur2 && iny==cur1)
		for(k=startX;k<n && k<(inx+1)*t_width;k++)
		    w[k]=p[k]-pTv*v[k]/vTv;
	    __syncthreads();
	    // 5. find new A values
	    if(inx>=cur2 && iny>=cur2)
	    {
		for(k=startX;k<(inx+1)*t_width && k<n;k++)
		    for(l=startY;l<(iny+1)*t_width && l<n;l++)
			A[k*n+l]=A[k*n+l]-v[k]*w[l] - w[k]*v[l];
	    }
	    // Track the subdiagonal values
	    if( iny == cur1 && inx==cur1)
		A[i*n + i+1] = mu;
	    // Store the householder vectors in the matrix 
	    if(inx>=cur1 && iny == cur1)
		for(k=startX;k<(inx+1)*t_width && k<n;k++)
		    A[k*n+i] = v[k];
	    __syncthreads();
	}
	// Here, we need to move the diagonals over to C...
	// And store the subdiagonals into p.
	// Do it with the threads in the middle.
	if(inx == iny){
	    for(k=inx*t_width;k<n && k<(inx+1)*t_width;k++){
		C[k] = A[k*n+k];
		if(k<n-1)
		    p[k] = A[k*n+k+1];
		A[k*n+k]=1.0f;
	    }
	}
	__syncthreads();
	// Here, we accumulate Q from householder
	// Important to get this done AFTER copying diagonals
	for(i=n-1;i>0;i--){
	    // Page 199 of Golub: ( A begins as identity)
	    // v[i] = 1; v[i+1:n] = A[i+1:n]
	    // Page 197 of Golub: (row.house)
	    // beta = -2/vTv ; w = beta ATv; A = A+vwT
	    startX = (inx*t_width > i) ? inx*t_width : i;
	    startY = (iny*t_width > i) ? iny*t_width : i;
	    if(inx*t_width >= i && iny== i/t_width){
		for(k = startX; k<n && k<(inx+1)*t_width; k++){
		    if(k!=i){
			v[k] = A[k*n+i];
			// Set the values equal to zero
			A[k*n+i] = 0.0f;
			A[i*n+k] = 0.0f;
		    }
		}
	    }

	    __syncthreads();
	    // Find vTv in each thread
	    vTv = 0.0f;
	    for(k=i;k<n;k++)
		vTv = vTv + v[k]*v[k];
	    // Find w cooperatively
	    if(inx*t_width >= i && iny*t_width >= i){
		// w = beta * AT * v
		for(k = startX; k<n && k<(inx+1)*t_width; k++)
		    w[k]=0.0f;
		    for(l=startY; l<n && l<(iny+1)*t_width; l++)
			w[k] = w[k]-2/vTv * A[l*n+k] * v[l];
		__syncthreads();
		// Calculate next A(bottom right matrix)
		// A = A + v wT
		for(k = startX; k<n && k<(inx+1)*t_width; k++)
		    for(l=startY; l<n && l<(iny+1)*t_width; l++)
			A[l*n+k] = A[l*n+k]+ v[l] * w[k];
		__syncthreads();
	    }
	}
	// QR Diagonalization
	// Algorithm 8.2.3 in Golub
	cur1 = 0; // cur1 = "p"
	cur2 = 0; // cur2 = "q"
	while(cur2<n){
	    //1 a[i+1,i] and a[i,i+1] = 0 if a[i,i+1] <= eps(a[i,i]+a[i+1,i+1])
	    //2 choose p,q such that T22 is unreduced(no zeros in subdiagonal)
	    i=n-1;
	    while(i>0 && abs(p[i-1])<eps*(abs(C[i-1])+abs(C[i]))){
		if(iny == 0 && inx == 0)
		    p[i-1] = 0.0f;
		i--;
	    }

	    cur2 = n-i;

	    while(i>0 && abs(p[i-1])>=(eps*(abs(C[i-1])+abs(C[i]))))
		i--;
	    cur1=i;
	    //3 if q<n, do run8_2_2 on T22
	    // Also, calculate eigenvectors
	    if(cur2<n)
    		run8_2_2_eVects(C,p,n,cur1,cur2,A,t_width);
	    __syncthreads();
	}
    }
}

// Assuming these things:
// A symmetric positive matrix (from the hessian)
// The matrix will be small (40 to 200)
// This function returns the eigenvalues in the C matrix
__device__ void hessian_qrd( const int n, float *A, const int t_width, float *C, const float eps)
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    float sum;
    float vTv;
    float pTv;
    float mu;
    int k,i,cur1,cur2,l,startX,startY;
    extern __shared__ float v[];
    float * p;
    float * w;
    p = v + n;
    w = v + 2*n;
    // Ensure we are in matrix (we don't have to worry about this anymore)
    if(inx*t_width<n && iny*t_width<n){
	// householder Tridiagonalization Algorithm 8.3.1
	// 1. for loop
	for(i=0;i<n-2;i++){
	    // Keep these to tell if we are in the right spot
	    cur1 = i/t_width;
	    cur2 = (i+1)/t_width; // For when we need to check against i+1
	    // find the start value for loops once
	    startX =((inx*t_width) > i+1) ? (inx*t_width) : (i+1) ;
	    startY =((iny*t_width) > i+1) ? (iny*t_width) : (i+1) ;
	    // Reset these values
	    sum=0.0f;
	    vTv=0.0f;
    	    pTv=0.0f;
	    // 2. house(A,i); algorithm 5.1.1
	    // length(v) = n-i
	    // Check if in proper column and below diagonal
	    if(iny == cur1 && inx >= cur2)
		for(k = startX;k<(inx+1)*t_width && k<n;k++)
		    v[k] = A[k*n+i];
	    __syncthreads();
    	    for(k=i+1;k<n;k++)
		sum+=v[k]*v[k];
	    mu = sqrt(sum);
	    // sum = beta
	    sum = v[i+1]+mu*(v[i+1]>0.0f ? 1.0f:-1.0f);
	    __syncthreads();
	    if(iny == cur1 && inx >= cur2)
		for(k = startX;k<(inx+1)*t_width && k<n;k++)
		    v[k] = v[k]/sum;
	    if(inx==cur2 && iny==cur1)
		v[i+1]=1.0f;
	    __syncthreads();
	    // END house
	    // 2.5 find v**T v = length of v
	    for(k=i+1;k<n;k++)
    		vTv += v[k]*v[k];
	    // 3. find p = 2A(i+1:n,i+1:n)v/vTv
	    if(inx>=cur2 && iny==cur1){
		for(k = startX;k<(inx+1)*t_width && k<n;k++){
		    p[k]=0.0f;
		    for(l=i+1;l<n;l++)
			p[k]=p[k]+2.0f*A[(k)*n+l]*v[l]/vTv;
		}

	    }
	    __syncthreads();
	    // 3.5 find pTv
	    for(k=i+1;k<n;k++)
    		pTv += p[k]*v[k];
	    __syncthreads();
	    // 4. find w
	    if(inx>=cur2 && iny==cur1)
		for(k=i+1;k<n && k<(inx+1)*t_width;k++)
		    w[k]=p[k]-pTv*v[k]/vTv;
	    __syncthreads();
	    // 5. find new A values
	    if(inx>=cur2 && iny>=cur2)
	    {
		for(k=startX;k<(inx+1)*t_width && k<n;k++)
		    for(l=startY;l<(iny+1)*t_width && l<n;l++)
			A[k*n+l]=A[k*n+l]-v[k]*w[l] - w[k]*v[l];
	    }
	    if( iny == cur1 && inx==cur1){
		A[(i+1)*n + i] = mu;
		A[(i)*n + i+1] = mu;
	    }
	    __syncthreads();
	}
	// QR Diagonalization
	// Algorithm 8.2.3 in Golub
	cur1 = 0; // cur1 = "p"
	cur2 = 0; // cur2 = "q"
	while(cur2<n){
	    //1 a[i+1,i] and a[i,i+1] = 0 if a[i,i+1] <= eps(a[i,i]+a[i+1,i+1])
	    //2 choose p,q such that T22 is unreduced(no zeros in subdiagonal)
	    i=n-1;
	    while(i>0){
		if(abs(A[i*n+i-1])<=eps*(abs(A[i*n+i])+abs(A[(i+1)*n+i+1])))
		    A[i*n+i-1] = 0.0f;
		else
		    break;
		i--;
	    }

	    cur2 = n-i;

	    while(i>0 && abs(A[i*n+i-1])>=(eps*(abs(A[i*n+i])+abs(A[(i+1)*n+i+1]))))
		i--;
	    cur1=i;
	    //3 if q<n, do run8_2_2 on T22
	    if(cur2<n && iny==inx && inx==0)
		run8_2_2(A,n,cur1,cur2);
	    __syncthreads();
	}
	if(iny == inx){
	    for(i=inx*t_width; i<n && i<(inx+1)*t_width; i++){
		C[i] = A[i*n+i];
	    }
	}
    }
}

/* block_QR( const int n_mat, float *mat, const int* idxs, const int* sizes)
ARGUMENTS:
    n_mat: The number of matrices to be diagonalized
    mat: The matrices to be diagonalized, stored in consecutive memory
    idxs: Array containing indices of the starting point of each matrix
    sizes: Array containing the size of each matrix (total number of elements)
    evals: array containing the space to store the final eigenvectors.
    EVECTS: boolean telling us whether to collect the eigenvectors.
*/
__global__ void block_QR( const int n_mat, float *mat,const int* idxs, const int* sizes, float* evals, const int* eval_idxs,const int EVECTS, const float eps){
    if(blockIdx.x < n_mat){
	int t_width = sizes[blockIdx.x]/blockDim.x+1;
	if(EVECTS<0)
	    hessian_qrd(sizes[blockIdx.x],mat+idxs[blockIdx.x],t_width,evals+eval_idxs[blockIdx.x], eps);
	else
	    hessian_qrf(sizes[blockIdx.x],mat+idxs[blockIdx.x],t_width,evals+eval_idxs[blockIdx.x], eps);
    }
}

__global__ void block_QR_EVal_only( const int n_mat, float *mat, const int* idxs, const int* sizes, const float eps){
	float* evals=mat;
	int t_width = sizes[blockIdx.x]/blockDim.x+1;
    	hessian_qrd(sizes[blockIdx.x],mat+idxs[blockIdx.x],t_width,evals, eps);
}
