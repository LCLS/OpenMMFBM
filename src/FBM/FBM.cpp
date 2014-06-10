#include <iostream>
#include <string>
#include <vector>

#include "OpenMM.h"


#ifdef OPENMMFBM_CUDA
#include "FBM/FBMCuda.h"
#endif

#include "FBM/FBMReference.h"
#include "FBM/FBMAbstract.h"
#include "FBM/FBM.h"

FBM::FBM( OpenMMFBM::FBMParameters &params ) : myParameters( params ) {

}

void FBM::run( OpenMM::Context &context, OpenMM::Context &blockContext, std::vector<std::vector<OpenMM::Vec3> > &modes, std::vector<double> &eigenvalues, std::string fbmPlatform ) {
	std::cout << "In FBM::run" << std::endl;

	implementation = implementationFactory( context, blockContext, fbmPlatform );

	std::cout << "have my implementation!" << std::endl;

	implementation->run( eigenvalues, modes, "", "" );
}

FBMAbstract *FBM::implementationFactory( OpenMM::Context &context, OpenMM::Context &blockContext, std::string fbmPlatform ) {
	std::cout << "In our \"factory\"..." << std::endl;

	FBMAbstract *fbmImplementation = NULL;

#ifdef OPENMMFMB_CUDA
	if( fbmPlatform == "Cuda" ) {
		fbmImplementation = new FBMCuda( context, blockContext, myParameters );
	}
#endif

	if( fbmImplementation == NULL ) {
		fbmImplementation = new OpenMMFBM::FBMReference( context, blockContext, myParameters );
	}

	std::cout << "Have implementation!" << std::endl;
	return fbmImplementation;
}
