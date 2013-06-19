/* Class FBMAbstract
 * Purpose: Provide an interface for implementations of
 *          the Flexible Block Method on different platforms
 *          Its private member functions can be overloaded
 *          to operate on any desired platform (Cuda, OpenCL, etc.)
 *          Its run() function is inherited by all child classes
 *          and calls each of these private functions.
 */

#ifndef FBMABSTRACT_H
#define FBMABSTRACT_H

#include <iostream>
#include <string>
#include <vector>

#include "OpenMM.h"

#include "OpenMMFBM/FBMParameters.h"
//#include "OpenMMFBM/BlockDiagonalize.h"
//#include "OpenMMFBM/Diagonalize.h"

/*
 * Until we sort out the exact design...

#include "OpenMMFBM/CulaDiagonalize.h"
#include "OpenMMFBM/MKLDiagonalize.h"
#include "OpenMMFBM/MKLBlockDiagonalize.h"
#include "OpenMMFBM/QRBlockDiagonalize.h"
*/

class FBMAbstract {
   public:
 FBMAbstract(OpenMM::Context &c, OpenMM::Context &bC, FBMParameters &p) : context(c), blockContext(bC), params(p) {}
      // This will be implemented here, and will simply call the pure virtual
      // functions below with CPU 'glue code' in between.
      virtual void run(std::vector<double> &eigenvalues, std::vector<std::vector<OpenMM::Vec3> > &modes, std::string blockdiag, std::string diag) {
	std::cout << "I'm running!" << std::endl;

	/*
	 * Until we sort out the exact design ...

           BlockDiagonalize* bd;
	   Diagonalize* d;

#ifdef INTEL_MKL
	   if (blockdiag == "CPU")
	      bd = new MKLBlockDiagonlize();
#endif

#ifdef HAVE_QR
	   if (blockdiag == "GPU")
	      bd = new QRBlockDiagonlize();
#endif

#ifdef INTEL_MKL
	   if (diag == "CPU")
	      d = new MKLDiagonalize();
#endif

#ifdef CULA
	   if (blockdiag == "GPU")
	      d = new CulaDiagonalize();
#endif
	*/

	initialize();
	formBlocks();  // blockHessian
	/*
	diagonalizeBlocks();
	formProjectionMatrix(); // E
	computeHE();  // HE
	computeS();   // S
	diagonalizeS();  // Q
	computeModes(eigenvalues, modes); // U, or modes
	*/

	   /*
	   delete bd;
	   delete d;
	   */
      }

      // void getBlockHessian()
      // Populates the given vector with the block hessian
      virtual void getBlockHessian(std::vector<std::vector<double > >& blockHessianVectors) const = 0;

      virtual void getBlockEigenvectors(std::vector<std::vector<double > >& blockEigenvectors) const = 0;

      virtual void getProjectionMatrix(std::vector<std::vector<double > >& projectionMatrix) const = 0;

      virtual void getHE(std::vector<std::vector<double > >& HE) const = 0;

      virtual void getCoarseGrainedHessian(std::vector<std::vector<double > >& coarseGrainedHessian) const = 0;

   protected:
      // Initialized in the constructor
      OpenMM::Context &context;
      OpenMM::Context &blockContext;
      FBMParameters &params;

   private:
      // Each of these functions will be implemented in our two child classes
      // of FBMAbstract, which will be FBMRefernce and FBMCuda
      // These in turn will respectively implement CPU code and GPU code using
      // the CUDA platform.
      // We have outlined below what each function will be performing:
      
      // void initialize()
      // Perform any needed initialization for this run
      virtual void initialize()=0;

      // void formBlocks()
      // 1. Allocate an NxN block Hessian matrix (blockHessian
      // 2. Compute the individual blocks using numerical differentiation
      // 3. Place these blocks on the diagonal of blockHessian
      virtual void formBlocks()=0;

      // void diagonalizeBlocks(String platform)
      // 1. Take the diagonal blocks from the blockHessian matrices
      // 2. Diagonalize each of these blocks
      // 3. Allocate and populate blockEigenvalues and blockEigenvectors
      //     a. Compute geometric degrees of freedom
      //     b. Orthogonalization
      // 4. Call sortEigenvectors() to sort eigenvectors by eigenvalue
      // 5. Allocate and populate the E matrix
      virtual void diagonalizeBlocks()=0;

      // void sortEigenvectors()
      // 1. Sort the passed eigenvectors by eigenvalue
      // TMC 8/14: We have since moved this to the child classes
      // since implementation of the two arrays will vary
      //virtual void sortEigenvectors(float* eigenvalues, float** eigenvectors)=0;

      // void computeProjectionMatrix()
      // 1. Populates the E matrix
      virtual void formProjectionMatrix()=0;

      // void computeHE()
      // 1. We need this function to avoid computing S=E^T H E directly
      //    which requires the full Hessian H.
      //    Instead, we compute HE using these steps:
      //      a. perturb positions by eps E M^(-1/2)
      //      b. run numerical differentiation
      virtual void computeHE()=0;
      
      // void computeS()
      // 1. Simple matrix multiplication, S = E^T (HE)
      virtual void computeS()=0;

      // void diagonalizeS()
      // 1. Diagonalize the S matrix
      // 2. Allocate and populate dS and Q
      // 3. Call sortEigenvectors() to sort Q by the eigenvalues in dS
      virtual void diagonalizeS()=0;

      // void computeModes()
      // 1. Simple matrix multiplication, U = EQ
      // 2. Note this will allocate and populate the passed modes array,
      //    which is allocated by the user
      virtual void computeModes(std::vector<double>& eigenvalues,
                                std::vector<std::vector<OpenMM::Vec3> >& modes)=0;

};

#endif
