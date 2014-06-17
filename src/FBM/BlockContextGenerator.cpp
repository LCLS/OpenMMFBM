#include <iostream>
#include <map>
#include <vector>

#include "OpenMM.h"

#include "FBM/BlockContextGenerator.h"

using namespace std;
using namespace OpenMM;

namespace OpenMM {
	namespace FBM {

unsigned int BlockContextGenerator::blockNumber( const vector<int> &blockSizes, const int p ) const {
	unsigned int block = 0;
	while( block != blockSizes.size() && blockSizes[block] <= p ) {
		block++;
	}
	return block - 1;
}

bool BlockContextGenerator::inSameBlock( const vector<int> &blockSizes, const int p1, const int p2, const int p3 = -1, const int p4 = -1 ) const {
	if( blockNumber( blockSizes, p1 ) != blockNumber( blockSizes, p2 ) ) {
		return false;
	}

	if( p3 != -1 && blockNumber( blockSizes, p3 ) != blockNumber( blockSizes, p1 ) ) {
		return false;
	}

	if( p4 != -1 && blockNumber( blockSizes, p4 ) != blockNumber( blockSizes, p1 ) ) {
		return false;
	}

	return true;   // They're all the same!
}



void BlockContextGenerator::determineBlockSizes( int residuesPerBlock, const vector<int> &residueSizes, vector<int> &blockSizes ) {
	int block_start = 0;
	int mLargestBlockSize = 0;
	blockSizes.clear();
	for( int i = 0; i < params.residue_sizes.size(); i++ ) {
		if( i % params.res_per_block == 0 ) {
			blockSizes.push_back( block_start );
		}
		block_start += params.residue_sizes[i];
	}

	for( int i = 1; i < blockSizes.size(); i++ ) {
		int block_size = blockSizes[i] - blockSizes[i - 1];
		if( block_size > mLargestBlockSize ) {
			mLargestBlockSize = block_size;
		}
	}

	mLargestBlockSize *= 3; // degrees of freedom in the largest block
	cout << "blocks " << blockSizes.size() << endl;
	cout << blockSizes[blockSizes.size() - 1] << endl;
}



void BlockContextGenerator::addBondForce( const int forceIndex, const vector<int> &blockSizes, const System &system, System *blockSystem ) {
	// Create a new harmonic bond force.
	// This only contains pairs of atoms which are in the same block.
	// I have to iterate through each bond from the old force, then
	// selectively add them to the new force based on this condition.
	HarmonicBondForce *hf = new HarmonicBondForce();
	const HarmonicBondForce *ohf = dynamic_cast<const HarmonicBondForce *>( &system.getForce( params.forces[forceIndex].index ) );
	for( int i = 0; i < ohf->getNumBonds(); i++ ) {
		// For our system, add bonds between atoms in the same block
		int particle1, particle2;
		double length, k;
		ohf->getBondParameters( i, particle1, particle2, length, k );
		if( inSameBlock( blockSizes, particle1, particle2 ) ) {
			hf->addBond( particle1, particle2, length, k );
		}
	}
	blockSystem->addForce( hf );
}

void BlockContextGenerator::addAngleForce( int forceIndex, const vector<int> &blockSizes, const System &system, System *blockSystem ) {
	// Same thing with the angle force....
	HarmonicAngleForce *af = new HarmonicAngleForce();
	const HarmonicAngleForce *ahf = dynamic_cast<const HarmonicAngleForce *>( &system.getForce( params.forces[forceIndex].index ) );
	for( int i = 0; i < ahf->getNumAngles(); i++ ) {
		// For our system, add bonds between atoms in the same block
		int particle1, particle2, particle3;
		double angle, k;
		ahf->getAngleParameters( i, particle1, particle2, particle3, angle, k );
		if( inSameBlock( blockSizes, particle1, particle2, particle3 ) ) {
			af->addAngle( particle1, particle2, particle3, angle, k );
		}
	}
	blockSystem->addForce( af );

}

void BlockContextGenerator::addPeriodicTorsionForce( int forceIndex, const vector<int> &blockSizes, const System &system, System *blockSystem ) {
	// And the dihedrals....
	PeriodicTorsionForce *ptf = new PeriodicTorsionForce();
	const PeriodicTorsionForce *optf = dynamic_cast<const PeriodicTorsionForce *>( &system.getForce( params.forces[forceIndex].index ) );
	for( int i = 0; i < optf->getNumTorsions(); i++ ) {
		// For our system, add bonds between atoms in the same block
		int particle1, particle2, particle3, particle4, periodicity;
		double phase, k;
		optf->getTorsionParameters( i, particle1, particle2, particle3, particle4, periodicity, phase, k );
		if( inSameBlock( blockSizes, particle1, particle2, particle3, particle4 ) ) {
			ptf->addTorsion( particle1, particle2, particle3, particle4, periodicity, phase, k );
		}
	}
	blockSystem->addForce( ptf );
}

void BlockContextGenerator::addRBTorsionForce( int forceIndex,  const vector<int> &blockSizes, const System &system, System *blockSystem ) {
	// And the impropers....
	RBTorsionForce *rbtf = new RBTorsionForce();
	const RBTorsionForce *orbtf = dynamic_cast<const RBTorsionForce *>( &system.getForce( params.forces[forceIndex].index ) );
	for( int i = 0; i < orbtf->getNumTorsions(); i++ ) {
		// For our system, add bonds between atoms in the same block
		int particle1, particle2, particle3, particle4;
		double c0, c1, c2, c3, c4, c5;
		orbtf->getTorsionParameters( i, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
		if( inSameBlock( blockSizes, particle1, particle2, particle3, particle4 ) ) {
			rbtf->addTorsion( particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
		}
	}
	blockSystem->addForce( rbtf );
}

void BlockContextGenerator::addNonbondedForce( int forceIndex, const vector<int> &blockSizes, const System &system, System *blockSystem ) {
	// This is a custom nonbonded pairwise force and
	// includes terms for both LJ and Coulomb.
	// Note that the step term will go to zero if block1 does not equal block 2,
	// and will be one otherwise.
	CustomBondForce *cbf = new CustomBondForce( "4*eps*((sigma/r)^12-(sigma/r)^6)+138.935456*q/r" );
	const NonbondedForce *nbf = dynamic_cast<const NonbondedForce *>( &system.getForce( params.forces[forceIndex].index ) );

	cbf->addPerBondParameter( "q" );
	cbf->addPerBondParameter( "sigma" );
	cbf->addPerBondParameter( "eps" );

	// store exceptions
	// exceptions[p1][p2] = params
	map<int, map<int, vector<double> > > exceptions;

	for( int i = 0; i < nbf->getNumExceptions(); i++ ) {
		int p1, p2;
		double q, sig, eps;
		nbf->getExceptionParameters( i, p1, p2, q, sig, eps );
		if( inSameBlock( blockSizes, p1, p2 ) ) {
			vector<double> params;
			params.push_back( q );
			params.push_back( sig );
			params.push_back( eps );
			if( exceptions.count( p1 ) == 0 ) {
				map<int, vector<double> > pair_exception;
				pair_exception[p2] = params;
				exceptions[p1] = pair_exception;
			} else {
				exceptions[p1][p2] = params;
			}
		}
	}

	// add particle params
	// TODO: iterate over block dimensions to reduce to O(b^2 N_b)
	for( int i = 0; i < nbf->getNumParticles() - 1; i++ ) {
		for( int j = i + 1; j < nbf->getNumParticles(); j++ ) {
			if( !inSameBlock( blockSizes, i, j ) ) {
				continue;
			}
			// we have an exception -- 1-4 modified interactions, etc.
			if( exceptions.count( i ) == 1 && exceptions[i].count( j ) == 1 ) {
				vector<double> params = exceptions[i][j];
				cbf->addBond( i, j, params );
			}
			// no exception, normal interaction
			else {
				vector<double> params;
				double q1, q2, eps1, eps2, sigma1, sigma2, q, eps, sigma;

				nbf->getParticleParameters( i, q1, sigma1, eps1 );
				nbf->getParticleParameters( j, q2, sigma2, eps2 );

				q = q1 * q2;
				sigma = 0.5 * ( sigma1 + sigma2 );
				eps = sqrt( eps1 * eps2 );

				params.push_back( q );
				params.push_back( sigma );
				params.push_back( eps );

				cbf->addBond( i, j, params );
			}
		}
	}

	blockSystem->addForce( cbf );
}


Context *BlockContextGenerator::generateBlockContext( Context &context ) {
	vector<int> blockSizes;

	determineBlockSizes( params.res_per_block, params.residue_sizes, blockSizes );

	vector<Vec3> positions = context.getState( State::Positions ).getPositions();

	// Store Particle Information
	const unsigned int mParticleCount = positions.size();

	cout << "n particles " << mParticleCount << endl;

	vector<double> mParticleMass;
	mParticleMass.reserve( mParticleCount );
	for( unsigned int i = 0; i < mParticleCount; i++ ) {
		mParticleMass.push_back( context.getSystem().getParticleMass( i ) );
	}

	System *blockSystem = new System();

	cout << "res per block " << params.res_per_block << endl;
	for( int i = 0; i < mParticleCount; i++ ) {
		blockSystem->addParticle( mParticleMass[i] );
	}

	// Creating a whole new system called the blockSystem.
	// This system will only contain bonds, angles, dihedrals, and impropers
	// between atoms in the same block.
	// Also contains pairwise force terms which are zeroed out for atoms
	// in different blocks.
	// This necessitates some copying from the original system, but is required
	// because OpenMM populates all data when it reads XML.
	// Copy all atoms into the block system.

	// Copy the center of mass force.
	cout << "adding forces..." << endl;
	for( int i = 0; i < params.forces.size(); i++ ) {
		string forcename = params.forces[i].name;
		cout << "Adding force " << forcename << " at index " << params.forces[i].index << endl;
		if( forcename == "Bond" ) {
			addBondForce( i, blockSizes, context.getSystem(), blockSystem );
		} else if( forcename == "Angle" ) {
			addAngleForce( i, blockSizes, context.getSystem(), blockSystem );
		} else if( forcename == "Dihedral" ) {
			addPeriodicTorsionForce( i, blockSizes, context.getSystem(), blockSystem );
		} else if( forcename == "Improper" ) {
			addRBTorsionForce( i, blockSizes, context.getSystem(), blockSystem );
		} else if( forcename == "Nonbonded" ) {
			addNonbondedForce( i, blockSizes, context.getSystem(), blockSystem );
		} else {
			cout << "Unknown Force: " << forcename << endl;
		}
	}

	cout << "done." << endl;

	VerletIntegrator *integ = new VerletIntegrator( 0.000001 );

	Context *blockContext;

	switch( params.blockPlatform ) {
		case OpenMMFBM::Preference::Reference:
			blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "Reference" ) );
			break;
		case OpenMMFBM::Preference::OpenCL:
			blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "OpenCL" ) );
			break;
		case OpenMMFBM::Preference::CUDA:
			blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "CUDA" ) );
			break;
	}

	blockContext->setPositions( positions );

	return blockContext;
}
}
}
