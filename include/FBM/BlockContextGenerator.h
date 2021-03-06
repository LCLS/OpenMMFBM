#ifndef BLOCKCONTEXTGENERATOR_H
#define BLOCKCONTEXTGENERATOR_H

#include <vector>

#include "OpenMM.h"

#include "FBM/FBMParameters.h"

namespace OpenMM {
	namespace FBM {
		class BlockContextGenerator {
			public:
				BlockContextGenerator( OpenMMFBM::FBMParameters &p ) : params( p ) {}

				OpenMM::Context *generateBlockContext( OpenMM::Context &context );

			private:
				void addBondForce( const int forceIndex, const std::vector<int> &blockSizes, const OpenMM::System &system, OpenMM::System *blockSystem );

				void addAngleForce( const int forceIndex, const std::vector<int> &blockSizes, const OpenMM::System &system, OpenMM::System *blockSystem );

				void addPeriodicTorsionForce( const int forceIndex, const std::vector<int> &blockSizes, const OpenMM::System &system, OpenMM::System *blockSystem );

				void addRBTorsionForce( const int forceIndex, const std::vector<int> &blockSizes, const OpenMM::System &system, OpenMM::System *blockSystem );

				void addNonbondedForce( const int forceIndex, const std::vector<int> &blockSizes, const OpenMM::System &system, OpenMM::System *blockSystem );

				void determineBlockSizes( int residuesPerBlock, const std::vector<int> &residueSizes, std::vector<int> &blockSizes );

				unsigned int blockNumber( const std::vector<int> &blockSizes, const int p ) const;

				bool inSameBlock( const std::vector<int> &blockSizes, const int p1, const int p2, const int p3, const int p4 ) const;

				OpenMMFBM::FBMParameters &params;
		};
	}
}

#endif //BLOCKCONTEXTGENERATOR_H
