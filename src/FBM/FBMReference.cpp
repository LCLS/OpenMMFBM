#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/Vec3.h"
#include <sys/time.h>
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include <algorithm>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "OpenMM.h"
#include "FBM/Math.h"
#include "FBM/FBMReference.h"

#include "tnt_array2d_utils.h"

using namespace OpenMM;
using namespace std;

namespace OpenMMFBM {
	const unsigned int ConservedDegreesOfFreedom = 6;

	// Function Implementations
	unsigned int FBMReference::blockNumber( const int p ) const {
		unsigned int block = 0;
		while( block != blockSizes.size() && blockSizes[block] <= p ) {
			block++;
		}
		return block - 1;
	}

	bool FBMReference::inSameBlock( const int p1, const int p2, const int p3 = -1, const int p4 = -1 ) const {
		if( blockNumber( p1 ) != blockNumber( p2 ) ) {
			return false;
		}

		if( p3 != -1 && blockNumber( p3 ) != blockNumber( p1 ) ) {
			return false;
		}

		if( p4 != -1 && blockNumber( p4 ) != blockNumber( p1 ) ) {
			return false;
		}

		return true;   // They're all the same!
	}

	void FBMReference::determineBlockSizes( int residuesPerBlock, const vector<int> &residueSizes ) {
		int block_start = 0;

		blockSizes.clear();
		cout << "residue sizes length " << params.residue_sizes.size() << endl;
		for( int i = 0; i < params.residue_sizes.size(); i++ ) {
			if( i % params.res_per_block == 0 ) {
				blockSizes.push_back( block_start );
			}
			block_start += params.residue_sizes[i];
		}

		for( int i = 1; i < blockSizes.size(); i++ ) {
			int block_size = blockSizes[i] - blockSizes[i - 1];
			if( block_size > largestBlockSize ) {
				largestBlockSize = block_size;
			}
		}

		largestBlockSize *= 3; // degrees of freedom in the largest block
		cout << "blocks " << blockSizes.size() << endl;
		cout << blockSizes[blockSizes.size() - 1] << endl;
	}


	bool sort_func( const EigenvalueColumn &a, const EigenvalueColumn &b ) {
		if( std::fabs( a.first - b.first ) < 1e-8 ) {
			if( a.second <= b.second ) {
				return true;
			}
		} else {
			if( a.first < b.first ) {
				return true;
			}
		}

		return false;
	}

	std::vector<EigenvalueColumn> FBMReference::sortEigenvalues( const EigenvalueArray &values ) const {
		std::vector<EigenvalueColumn> retVal;

		// Create Array
		retVal.reserve( values.dim() );
		for( unsigned int i = 0; i < values.dim(); i++ ) {
			retVal.push_back( std::make_pair( std::fabs( values[i] ), i ) );
		}

		// Sort Data
		std::sort( retVal.begin(), retVal.end(), sort_func );

		return retVal;
	}

	void FBMReference::initialize() {
		determineBlockSizes( params.res_per_block, params.residue_sizes );
		initialPositions = context.getState( State::Positions | State::Forces ).getPositions();
		particleCount = initialPositions.size();
		System &system = context.getSystem();
		mParticleMass.clear();
		for( unsigned int i = 0; i < particleCount; i++ ) {
			mParticleMass.push_back( system.getParticleMass( i ) );
		}
	}

	void FBMReference::formBlocks() {
#ifdef FIRST_ORDER
		blockContext.setPositions( initialPositions );
		vector<Vec3> block_start_forces = blockContext.getState( State::Forces ).getForces();
#endif

		const unsigned int n = 3 * particleCount;
		TNT::Array2D<double> h( n, n, 0.0 );
		vector<Vec3> blockPositions( initialPositions );
		cout << "largest block size " << largestBlockSize << endl;

		for( unsigned int i = 0; i < largestBlockSize; i++ ) {
			// Perturb the ith degree of freedom in EACH block
			// Note: not all blocks will have i degrees, we have to check for this
			for( unsigned int j = 0; j < blockSizes.size(); j++ ) {
				unsigned int dof_to_perturb = 3 * blockSizes[j] + i;
				unsigned int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

				// Cases to not perturb, in this case just skip the block
				if( j == blockSizes.size() - 1 && atom_to_perturb >= particleCount ) {
					continue;
				}
				if( j != blockSizes.size() - 1 && atom_to_perturb >= blockSizes[j + 1] ) {
					continue;
				}

				blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialPositions[atom_to_perturb][dof_to_perturb % 3] - params.blockDelta;
			}

			blockContext.setPositions( blockPositions );
			vector<Vec3> forces1 = blockContext.getState( State::Forces ).getForces();


#ifndef FIRST_ORDER
			// Now, do it again...
			for( int j = 0; j < blockSizes.size(); j++ ) {
				int dof_to_perturb = 3 * blockSizes[j] + i;
				int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

				// Cases to not perturb, in this case just skip the block
				if( j == blockSizes.size() - 1 && atom_to_perturb >= particleCount ) {
					continue;
				}
				if( j != blockSizes.size() - 1 && atom_to_perturb >= blockSizes[j + 1] ) {
					continue;
				}

				blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialPositions[atom_to_perturb][dof_to_perturb % 3] + params.blockDelta;
			}

			blockContext.setPositions( blockPositions );
			vector<Vec3> forces2 = blockContext.getState( State::Forces ).getForces();
#endif

			// revert block positions
			for( int j = 0; j < blockSizes.size(); j++ ) {
				int dof_to_perturb = 3 * blockSizes[j] + i;
				int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

				// Cases to not perturb, in this case just skip the block
				if( j == blockSizes.size() - 1 && atom_to_perturb >= particleCount ) {
					continue;
				}
				if( j != blockSizes.size() - 1 && atom_to_perturb >= blockSizes[j + 1] ) {
					continue;
				}

				blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialPositions[atom_to_perturb][dof_to_perturb % 3];

			}

			for( int j = 0; j < blockSizes.size(); j++ ) {
				int dof_to_perturb = 3 * blockSizes[j] + i;
				int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

				// Cases to not perturb, in this case just skip the block
				if( j == blockSizes.size() - 1 && atom_to_perturb >= particleCount ) {
					continue;
				}
				if( j != blockSizes.size() - 1 && atom_to_perturb >= blockSizes[j + 1] ) {
					continue;
				}

				int col = dof_to_perturb;

				int start_dof = 3 * blockSizes[j];
				int end_dof;
				if( j == blockSizes.size() - 1 ) {
					end_dof = 3 * particleCount;
				} else {
					end_dof = 3 * blockSizes[j + 1];
				}

				for( int k = start_dof; k < end_dof; k++ ) {
#ifdef FIRST_ORDER
					double blockscale = 1.0 / ( params.blockDelta * sqrt( mParticleMass[atom_to_perturb] * mParticleMass[k / 3] ) );
					h[k][col] = ( forces1[k / 3][k % 3] - block_start_forces[k / 3][k % 3] ) * blockscale;
#else
					double blockscale = 1.0 / ( 2 * params.blockDelta * sqrt( mParticleMass[atom_to_perturb] * mParticleMass[k / 3] ) );
					h[k][col] = ( forces1[k / 3][k % 3] - forces2[k / 3][k % 3] ) * blockscale;
#endif
				}
			}
		}

		// Make sure it is exactly symmetric.

		for( int i = 0; i < n; i++ ) {
			for( int j = 0; j < i; j++ ) {
				double avg = 0.5f * ( h[i][j] + h[j][i] );
				h[i][j] = avg;
				h[j][i] = avg;
			}
		}

		blockHessian = h.copy();
	}

	void FBMReference::diagonalizeBlocks() {

		// Diagonalize each block Hessian, get Eigenvectors
		// Note: The eigenvalues will be placed in one large array, because
		//       we must sort them to get k

		const unsigned int n = 3 * particleCount;
		TNT::Array1D<double> block_eigval( n, 0.0 );
		TNT::Array2D<double> block_eigvec( n, n, 0.0 );

		#pragma omp parallel for
		for( int i = 0; i < blockSizes.size(); i++ ) {
			diagonalizeBlock( i, blockHessian, initialPositions, block_eigval, block_eigvec );
		}

		blockEigenvectors = block_eigvec.copy();
		blockEigenvalues = block_eigval.copy();
	}

	void FBMReference::formProjectionMatrix() {
		const unsigned int n = 3 * particleCount;

		//***********************************************************
		// This section here is only to find the cuttoff eigenvalue.
		// First sort the eigenvectors by the absolute value of the eigenvalue.

		// sort all eigenvalues by absolute magnitude to determine cutoff
		std::vector<EigenvalueColumn> sortedEvalues = sortEigenvalues( blockEigenvalues );

		int max_eigs = params.bdof * blockSizes.size();
		double cutEigen = sortedEvalues[max_eigs].first;  // This is the cutoff eigenvalue

		// get cols of all eigenvalues under cutoff
		vector<int> selectedEigsCols;
		for( int i = 0; i < n; i++ ) {
			if( fabs( blockEigenvalues[i] ) < cutEigen ) {
				selectedEigsCols.push_back( i );
			}
		}

		// we may select fewer eigs if there are duplicate eigenvalues
		const int m = selectedEigsCols.size();

		// Inefficient, needs to improve.
		// Basically, just setting up E and E^T by
		// copying values from bigE.
		// Again, right now I'm only worried about
		// correctness plus this time will be marginal compared to
		// diagonalization.
		TNT::Array2D<double> E( n, m, 0.0 );
		TNT::Array2D<double> E_transposed( m, n, 0.0 );
		for( int i = 0; i < m; i++ ) {
			int eig_col = selectedEigsCols[i];
			for( int j = 0; j < n; j++ ) {
				E[j][i] = blockEigenvectors[j][eig_col];
				E_transposed[i][j] = blockEigenvectors[j][eig_col];
			}
		}

		projectionMatrix = E.copy();
		transposedProjectionMatrix = E_transposed.copy();
	}

	void FBMReference::computeHE() {
		const unsigned int n = 3 * particleCount;
		const unsigned int m = projectionMatrix.dim2();

		TNT::Array2D<double> HE( n, m, 0.0 );

		// Compute eps.
		const double eps = params.sDelta;

		// Make a temp copy of positions.
		vector<Vec3> tmppos( initialPositions );

#ifdef FIRST_ORDER
		vector<Vec3> forces_start = context.getState( State::Forces ).getForces();
#endif

		// Loop over i.
		for( unsigned int k = 0; k < m; k++ ) {

			// Perturb positions.
			int pos = 0;

			// forward perturbations
			for( unsigned int i = 0; i < particleCount; i++ ) {
				for( unsigned int j = 0; j < 3; j++ ) {
					tmppos[i][j] = initialPositions[i][j] + eps * projectionMatrix[3 * i + j][k] / sqrt( mParticleMass[i] );
					pos++;
				}
			}
			context.setPositions( tmppos );

			// Calculate F(xi).
			vector<Vec3> forces_forward = context.getState( State::Forces ).getForces();

#ifndef FIRST_ORDER
			// backward perturbations
			for( unsigned int i = 0; i < particleCount; i++ ) {
				for( unsigned int j = 0; j < 3; j++ ) {
					tmppos[i][j] = initialPositions[i][j] - eps * projectionMatrix[3 * i + j][k] / sqrt( mParticleMass[i] );
				}
			}
			context.setPositions( tmppos );

			// Calculate forces
			vector<Vec3> forces_backward = context.getState( State::Forces ).getForces();
#endif

			for( int i = 0; i < n; i++ ) {
#ifdef FIRST_ORDER
				const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 1.0 * eps;
				HE[i][k] = ( forces_forward[i / 3][i % 3] - forces_start[i / 3][i % 3] ) / scaleFactor;
#else
				const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 2.0 * eps;
				HE[i][k] = ( forces_forward[i / 3][i % 3] - forces_backward[i / 3][i % 3] ) / scaleFactor;
#endif
			}

			// restore positions
			for( unsigned int i = 0; i < particleCount; i++ ) {
				for( unsigned int j = 0; j < 3; j++ ) {
					tmppos[i][j] = initialPositions[i][j];
				}
			}
		}

		// *****************************************************************
		// restore unperturbed positions
		context.setPositions( initialPositions );

		productMatrix = HE.copy();
	}

	void FBMReference::computeS() {
		const unsigned int m = projectionMatrix.dim2();
		//*****************************************************************
		// Compute S, which is equal to E^T * H * E.
		TNT::Array2D<double> S( m, m, 0.0 );

		matrixMultiply( transposedProjectionMatrix, productMatrix, S );

		// make S symmetric
		for( unsigned int i = 0; i < S.dim1(); i++ ) {
			for( unsigned int j = i + 1; j < S.dim2(); j++ ) {
				double avg = 0.5 * ( S[i][j] + S[j][i] );
				S[i][j] = avg;
				S[j][i] = avg;
			}
		}


		reducedSpaceHessian = S.copy();
	}

	void FBMReference::diagonalizeS() {
		const unsigned int m = projectionMatrix.dim2();
		// Diagonalizing S by finding eigenvalues and eigenvectors...
		TNT::Array1D<double> dS( m, 0.0 );
		TNT::Array2D<double> q( m, m, 0.0 );
		diagonalizeSymmetricMatrix( reducedSpaceHessian, dS, q );

		reducedSpaceEigenvalues = dS.copy();
		reducedSpaceEigenvectors = q.copy();
	}

	void FBMReference::computeModes( std::vector<double> &eigenvalues, std::vector<std::vector<Vec3> > &modes ) {
		const unsigned int numModes = params.modes;

		// Sort by ABSOLUTE VALUE of eigenvalues.
		std::vector<EigenvalueColumn> sortedEvalues = sortEigenvalues( reducedSpaceEigenvalues );

		TNT::Array2D<double> Q( reducedSpaceEigenvectors.dim1(), numModes, 0.0 );
		for( int i = 0; i < numModes; i++ ) {
			for( int j = 0; j < reducedSpaceEigenvectors.dim1(); j++ ) {
				Q[j][i] = reducedSpaceEigenvectors[j][sortedEvalues[i].second];
			}
		}

		TNT::Array2D<double> modeMatrix( projectionMatrix.dim1(), numModes, 0.0 );
		matrixMultiply( projectionMatrix, Q, modeMatrix );

		modes.resize( numModes, vector<Vec3>( particleCount ) );
		for( unsigned int i = 0; i < numModes; i++ ) {
			for( unsigned int j = 0; j < particleCount; j++ ) {
				modes[i][j] = Vec3( modeMatrix[3 * j][i], modeMatrix[3 * j + 1][i], modeMatrix[3 * j + 2][i] );
			}
		}
		eigenvalues.resize( numModes, 0.0 );
		for( unsigned int i = 0; i < numModes; i++ ) {
			eigenvalues[i] = sortedEvalues[i].first;
		}
	}

	void FBMReference::diagonalizeBlock( const unsigned int block, const TNT::Array2D<double> &hessian,
										 const std::vector<Vec3> &positions, TNT::Array1D<double> &eval, TNT::Array2D<double> &evec ) {

		const unsigned int ConservedDegreesOfFreedom = 6;

		printf( "Diagonalizing Block: %d\n", block );

		// 1. Determine the starting and ending index for the block
		//    This means that the upper left corner of the block will be at (startatom, startatom)
		//    And the lower right corner will be at (endatom, endatom)
		const int startatom = 3 * blockSizes[block];
		int endatom = 3 * particleCount - 1 ;

		if( block != ( blockSizes.size() - 1 ) ) {
			endatom = 3 * blockSizes[block + 1] - 1;
		}

		const int size = endatom - startatom + 1;

		// 2. Get the block Hessian Hii
		//    Right now I'm just doing a copy from the big Hessian
		//    There's probably a more efficient way but for now I just want things to work..
		TNT::Array2D<double> h_tilde( size, size, 0.0 );
		for( int j = startatom; j <= endatom; j++ ) {
			for( int k = startatom; k <= endatom; k++ ) {
				h_tilde[k - startatom][j - startatom] = hessian[k][j];
			}
		}

		// 3. Diagonalize the block Hessian only, and get eigenvectors
		TNT::Array1D<double> di( size, 0.0 );
		TNT::Array2D<double> Qi( size, size, 0.0 );
		diagonalizeSymmetricMatrix( h_tilde, di, Qi );

		// sort eigenvalues by absolute magnitude
		std::vector<EigenvalueColumn> sortedPairs = sortEigenvalues( di );

		// find geometric dof
		TNT::Array2D<double> Qi_gdof( size, size, 0.0 );

		Vec3 pos_center( 0.0, 0.0, 0.0 );
		double totalmass = 0.0;

		for( int j = startatom; j <= endatom; j += 3 ) {
			double mass = mParticleMass[ j / 3 ];
			pos_center += positions[j / 3] * mass;
			totalmass += mass;
		}

		double norm = sqrt( totalmass );

		// actual center
		pos_center *= 1.0 / totalmass;

		// create geometric dof vectors
		// iterating over rows and filling in values for 6 vectors as we go
		for( int j = 0; j < size; j += 3 ) {
			double atom_index = ( startatom + j ) / 3;
			double mass = mParticleMass[atom_index];
			double factor = sqrt( mass ) / norm;

			// translational
			Qi_gdof[j][0]   = factor;
			Qi_gdof[j + 1][1] = factor;
			Qi_gdof[j + 2][2] = factor;

			// rotational
			// cross product of rotation axis and vector to center of molecule
			// x-axis (b1=1) ja3-ka2
			// y-axis (b2=1) ka1-ia3
			// z-axis (b3=1) ia2-ja1
			Vec3 diff = positions[atom_index] - pos_center;
			// x
			Qi_gdof[j + 1][3] =  diff[2] * factor;
			Qi_gdof[j + 2][3] = -diff[1] * factor;

			// y
			Qi_gdof[j][4]   = -diff[2] * factor;
			Qi_gdof[j + 2][4] =  diff[0] * factor;

			// z
			Qi_gdof[j][5]   =  diff[1] * factor;
			Qi_gdof[j + 1][5] = -diff[0] * factor;
		}

		// normalize first rotational vector
		double rotnorm = 0.0;
		for( int j = 0; j < size; j++ ) {
			rotnorm += Qi_gdof[j][3] * Qi_gdof[j][3];
		}

		rotnorm = 1.0 / sqrt( rotnorm );

		for( int j = 0; j < size; j++ ) {
			Qi_gdof[j][3] = Qi_gdof[j][3] * rotnorm;
		}

		// orthogonalize rotational vectors 2 and 3
		for( int j = 4; j < ConservedDegreesOfFreedom; j++ ) { // <-- vector we're orthogonalizing
			for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
				double dot_prod = 0.0;
				for( int l = 0; l < size; l++ ) {
					dot_prod += Qi_gdof[l][k] * Qi_gdof[l][j];
				}
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][j] = Qi_gdof[l][j] - Qi_gdof[l][k] * dot_prod;
				}
			}

			// normalize residual vector
			double rotnorm = 0.0;
			for( int l = 0; l < size; l++ ) {
				rotnorm += Qi_gdof[l][j] * Qi_gdof[l][j];
			}

			rotnorm = 1.0 / sqrt( rotnorm );

			for( int l = 0; l < size; l++ ) {
				Qi_gdof[l][j] = Qi_gdof[l][j] * rotnorm;
			}
		}

		// orthogonalize original eigenvectors against gdof
		// number of evec that survive orthogonalization
		int curr_evec = ConservedDegreesOfFreedom;
		for( int j = 0; j < size; j++ ) { // <-- vector we're orthogonalizing
			// to match ProtoMol we only include size instead of size + cdof vectors
			// Note: for every vector that is skipped due to a low norm,
			// we add an additional vector to replace it, so we could actually
			// use all size original eigenvectors
			if( curr_evec == size ) {
				break;
			}

			// orthogonalize original eigenvectors in order from smallest magnitude
			// eigenvalue to biggest
			int col = sortedPairs.at( j ).second;

			// copy original vector to Qi_gdof -- updated in place
			for( int l = 0; l < size; l++ ) {
				Qi_gdof[l][curr_evec] = Qi[l][col];
			}

			// get dot products with previous vectors
			for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
				// dot product between original vector and previously
				// orthogonalized vectors
				double dot_prod = 0.0;
				for( int l = 0; l < size; l++ ) {
					dot_prod += Qi_gdof[l][k] * Qi[l][col];
				}

				// subtract from current vector -- update in place
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] - Qi_gdof[l][k] * dot_prod;
				}
			}

			//normalize residual vector
			double norm = 0.0;
			for( int l = 0; l < size; l++ ) {
				norm += Qi_gdof[l][curr_evec] * Qi_gdof[l][curr_evec];
			}

			// if norm less than 1/20th of original
			// continue on to next vector
			// we don't update curr_evec so this vector
			// will be overwritten
			if( norm < 0.05 ) {
				continue;
			}

			// scale vector
			norm = sqrt( norm );
			for( int l = 0; l < size; l++ ) {
				Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] / norm;
			}

			curr_evec++;
		}

		// 4. Copy eigenpairs to big array
		//    This is necessary because we have to sort them, and determine
		//    the cutoff eigenvalue for everybody.
		// we assume curr_evec <= size
		for( int j = 0; j < curr_evec; j++ ) {
			int col = sortedPairs.at( j ).second;
			eval[startatom + j] = di[col];

			// orthogonalized eigenvectors already sorted by eigenvalue
			for( int k = 0; k < size; k++ ) {
				evec[startatom + k][startatom + j] = Qi_gdof[k][j];
			}
		}
	}
}
