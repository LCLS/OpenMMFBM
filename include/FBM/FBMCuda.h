/* Class FBMCuda
 * Purpose: Provide the Cuda implementation of all
 *          inherited methods from FBMAbstract
 *          Note we have extended our comments to
 *          incorporate our general strategy for
 *          parallelization on the GPUs
 */

#ifndef FBMCUDA_H
#define FBMCUDA_H

#include <vector>

#include "jama_eig.h"

#include "OpenMM.h"

#include "OpenMMFBM/FBMAbstract.h"

#include "openmm/Platform.h"
#include "CudaPlatform.h"
//#include "Diagonalize.h"
//#include "BlockDiagonalize.h"

class FBMCuda : public FBMAbstract, public OpenMM::CudaPlatform {
	public:
		FBMCuda( OpenMM::Context &c, OpenMM::Context &bC, OpenMMFBM::FBMParameters &p );
		//FBMCuda(FBMParameters &p);

		virtual void getBlockHessian( std::vector<std::vector<double > > &blockHessianVectors ) const;

		virtual void getBlockEigenvectors( std::vector<std::vector<double > > &blockVectors ) const;

		virtual void getProjectionMatrix( std::vector<std::vector<double > > &projMatrix ) const;

		virtual void getHE( std::vector<std::vector<double > > &HE ) const;

		virtual void getCoarseGrainedHessian( std::vector<std::vector<double > > &coarseGrainedHessian ) const;


	private:
		//CudaContext& context;
		//CudaContext& blockContext;
		CudaPlatform::PlatformData *data;
		CudaPlatform::PlatformData *blockData;
		float *blockHessian; // TMC I chose to make this a linear array because of how Kevin's kernel was set up
		float *blockEigenvalues;
		float *blockEigenvectors; // TMC See above.
		float *Et;
		float *E;
		float *HE;
		float *S;
		float *dS;
		float *Q;
		float *masses;

		int *blocknums;
		int *blocksizes;
		int *hessiannums;
		int *hessiansizes;

		int numBlocks;
		int numThreads;
		int _3N;

		void initialize() { }

		// void formBlocks()
		// 1. Allocate an NxN block Hessian matrix (blockHessian)
		//    - On the GPU using cudaMalloc()
		// 2. Compute the individual blocks using numerical differentiation
		//    - Assign thread (i, j) to block i, degree of freedom j
		//      We can perturb these degrees of freedom in parallel
		//    - Forces can be calculated on the GPU using OpenMM's
		//      calculators after the forward and backward perturbations
		// 3. Place these blocks on the diagonal of blockHessian
		//    - This step performs a vector subtraction, which can
		//      be done trivially with a linear array of GPU threads
		void formBlocks();

		// void diagonalizeBlocks()
		// 1. Take the diagonal blocks from the blockHessian matrices
		// 2. Diagonalize each of these blocks
		//    - We will use the parallel QR algorithm here developed by
		//      McShane.  This can do blocks in parallel, as well as
		//      individual portions of blocks
		//    - However, we also will provide the option of using the CPU and MKL
		//      Note if this option is chosen, we will have to copy data back
		// 3. Allocate and populate blockEigenvalues and blockEigenvectors
		//     a. Compute geometric degrees of freedom
		//        - Assign thread (i, j) to block i, degree of freedom j (2D block)
		//     b. Orthogonalization
		//        - Assign thread (i, j, k) to block i,
		//          vector j that you're orthogonalizing,
		//          vector k that you're orthogonalizing against
		// 4. Call sortEigenvectors() to sort eigenvectors by eigenvalue
		// 5. Allocate and populate the E matrix
		//        - The way it is implemented now is trivially parallelizable
		//          in a 2D Block of threads, assign E[i][j] to the appropriate
		//          entry in the eigenvectors using thread (i, j).
		//          We will try to make it better by
		//          incorporating this into step 4.
		void diagonalizeBlocks( /*BlockDiagonalize* bd*/ );

		// void sortEigenvectors()
		// 1. Sort the passed eigenvectors by eigenvalue
		//    - We can do a GPU parallel sort here
		//      One possibility is the odd-even mergesort:
		//      http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter46.html
		void sortEigenvectors( float *eigenvalues, float **eigenvectors );

		void formProjectionMatrix();
		// void computeHE()
		// 1. We need this function to avoid computing S=E^T H E directly
		//    which requires the full Hessian H.
		//    Instead, we compute HE using these steps:
		//      a. perturb positions by eps E M^(-1/2)
		//         - Similar to computing blocks, we can assign thread
		//           (i) to degree of freedom i and peturb
		//         - Use OpenMM for the force calculations between the
		//           perturbations
		//      b. run numerical differentiation
		//         - GPU vector subtraction of forward and backward forces
		//           Use a 1D array of threads
		void computeHE();

		// void computeS()
		// 1. Simple matrix multiplication, S = E^T (HE)
		//    - Follow the CUDA programmer's guide, use their example
		//      Let thread (i, j) handle S[i][j], use a loop internally
		void computeS();

		// void diagonalizeS()
		// 1. Diagonalize the S matrix
		//     - We will give the user a choice here
		//     - For small sizes of S, we have observed that it can sometimes
		//     -   be cheaper to diagonalize using MKL on the CPU
		//     - For larger sizes, will will offer Cula as an option
		//         Note Cula is not open-source, so that is further motivation
		//         to give the user a choice
		// 2. Allocate and populate dS and Q
		// 3. Call sortEigenvectors() to sort Q by the eigenvalues in dS
		void diagonalizeS( /*Diagonalize* d*/ );

		// void computeModes()
		// 1. Simple matrix multiplication, U = EQ
		//    - Follow the CUDA programmer's guide, use their example
		//      Let thread (i, j) handle S[i][j], use a loop internally
		// 2. Note this will allocate and populate the passed modes array,
		//    which is allocated by the user
		void computeModes( std::vector<double> &eigenvalues,
						   std::vector<std::vector<OpenMM::Vec3> > &modes );


	private:
		int numHessBlocks;
		int largestBlockSize;
		int m;
		void makeBlocksAndThreads( int );
};

#endif
