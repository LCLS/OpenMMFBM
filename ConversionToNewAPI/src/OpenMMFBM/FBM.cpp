#include <iostream>
#include <string>
#include <vector>

#include "OpenMM.h"


//#ifdef OPENMMFBM_CUDA
#include "OpenMMFBM/FBMCuda.h"
//#endif

#include "OpenMMFBM/FBMReference.h"
#include "OpenMMFBM/FBMAbstract.h"
#include "OpenMMFBM/FBM.h"


using namespace std;

FBM::FBM(FBMParameters &params) : myParameters(params) {

}

void FBM::run(OpenMM::Context &context, OpenMM::Context &blockContext, std::vector<std::vector<OpenMM::Vec3> > &modes, std::vector<double> &eigenvalues, std::string fbmPlatform) {

  cout << "In FBM::run" << endl;

  implementation = implementationFactory(context, blockContext, fbmPlatform);

  cout << "have my implementation!" << endl;

  implementation->run(eigenvalues, modes, "", "");
}

FBMAbstract* FBM::implementationFactory(OpenMM::Context &context, OpenMM::Context &blockContext, std::string fbmPlatform) {
  cout << "In our \"factory\"..." << endl;

  FBMAbstract* fbmImplementation = NULL;

  //#ifdef OPENMMFMB_CUDA
  if(fbmPlatform == "Cuda") {
    fbmImplementation = new FBMCuda(context, blockContext, myParameters);
  }
  //#endif

  if(fbmImplementation == NULL) {
    fbmImplementation = new FBMReference(context, blockContext, myParameters);
  }

  cout << "Have implementation!" << endl;
  return fbmImplementation;
}
