/* Class FBM
 * Purpose: Initialize and run the Flexible Block Method
 * Usage: A user should be able to allocate an FBM object
 *        and call its run() function
 *        This in turn should accept two Contexts for the
 *        platforms the user wants along with FBM parameters,
 *        and populate the passed C array of doubles for the 
 *        modes.
 */

#ifndef FBM_H
#define FBM_H

#include <string>

#include <vector>

#include "OpenMM.h"

#include "OpenMMFBM/FBMAbstract.h"

class FBM {
   public:
      // Initialize an FBM object
      // Accept appropriate parameters
 FBM(FBMParameters &params);

      // Based on the platform used by the two contexts:
      //    Create the appropriate concrete object of FBMAbstract
      //    Call run() on that object, passing the modes
 void run(OpenMM::Context &context, OpenMM::Context &blockContext, std::vector<std::vector<OpenMM::Vec3> > &modes, std::vector<double> &eigenvalues, std::string fbmPlatform);

 void getBlockHessian(std::vector<std::vector<double > >& blockHessianVectors) const {
   implementation->getBlockHessian(blockHessianVectors);
 }

 void getBlockEigenvectors(std::vector<std::vector<double > >& blockEigenvectors) const {
   implementation->getBlockEigenvectors(blockEigenvectors);
 }

 void getProjectionMatrix(std::vector<std::vector<double > >& projectionMatrix) const {
   implementation->getProjectionMatrix(projectionMatrix);
 }

 void getHE(std::vector<std::vector<double > >& HE) const {
   implementation->getHE(HE);
 }

 void getCoarseGrainedHessian(std::vector<std::vector<double > >& coarseGrainedHessian) const {
   implementation->getCoarseGrainedHessian(coarseGrainedHessian);
 }

   private:
 FBMAbstract* implementationFactory(OpenMM::Context &context, OpenMM::Context &blockContext, std::string fbmPlatform);

 FBMParameters &myParameters;
 FBMAbstract* implementation;
      
};


#endif
