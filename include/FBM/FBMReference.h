/* Class FBMReference
 * Purpose: Provide the pure CPU implementation of all
 *          inherited methods from FBMAbstract
 *          Note comments are essentially the same as
 *          in FBMAbstract
 */

#ifndef FBMREFERENCE_H
#define FBMREFERENCE_H


#include <vector>
#include <iostream>
using namespace std;
#include "jama_eig.h"

#include "OpenMM.h"


#include "FBM/FBMAbstract.h"
#include "FBM/FBMParameters.h"

typedef TNT::Array1D<double> EigenvalueArray;
typedef std::pair<double, int> EigenvalueColumn;

namespace OpenMMFBM {
	class FBMReference : public FBMAbstract {
		public:
			FBMReference( OpenMM::Context &c, OpenMM::Context &bC, OpenMMFBM::FBMParameters &p ) : FBMAbstract( c, bC, p ) { }

			virtual void getBlockHessian( std::vector<std::vector<double > > &blockHessianVectors ) const {
				const unsigned int N = 3 * particleCount;
				blockHessianVectors.resize( N, std::vector<double>( N, 0.0 ) );

				for( unsigned int i = 0; i < N; i++ ) {
					for( unsigned int j = 0; j < N; j++ ) {
						blockHessianVectors[i][j] = blockHessian[i][j];
					}
				}
			}

			virtual void getBlockEigenvectors( std::vector<std::vector<double > > &blockVectors ) const {
				const unsigned int N = 3 * particleCount;
				blockVectors.resize( N, std::vector<double>( N, 0.0 ) );
				cout << "BLOCKVECTOR SIZE: " << N << "x" << N << endl;
				cout << "BLOCKEIGENVECTOR SIZE: " << blockEigenvectors.dim1() << "x" << blockEigenvectors.dim2() << endl;
				for( unsigned int i = 0; i < N; i++ ) {
					for( unsigned int j = 0; j < N; j++ ) {
						blockVectors[i][j] = blockEigenvectors[i][j];
					}
				}
			}

			virtual void getProjectionMatrix( std::vector<std::vector<double > > &projMatrix ) const {
				const unsigned int nRows = projectionMatrix.dim1();
				const unsigned int nCols = projectionMatrix.dim2();

				projMatrix.resize( nRows, std::vector<double>( nCols, 0.0 ) );

				for( unsigned int i = 0; i < nRows; i++ ) {
					for( unsigned int j = 0; j < nCols; j++ ) {
						projMatrix[i][j] = projectionMatrix[i][j];
					}
				}
			}

			virtual void getHE( std::vector<std::vector<double > > &HE ) const {
				const unsigned int nRows = productMatrix.dim1();
				const unsigned int nCols = productMatrix.dim2();

				HE.resize( nRows, std::vector<double>( nCols, 0.0 ) );

				for( unsigned int i = 0; i < nRows; i++ ) {
					for( unsigned int j = 0; j < nCols; j++ ) {
						HE[i][j] = productMatrix[i][j];
					}
				}
			}

			virtual void getCoarseGrainedHessian( std::vector<std::vector<double > > &coarseGrainedHessian ) const {
				const unsigned int N = reducedSpaceHessian.dim1();

				coarseGrainedHessian.resize( N, std::vector<double>( N, 0.0 ) );

				for( unsigned int i = 0; i < N; i++ ) {
					for( unsigned int j = 0; j < N; j++ ) {
						coarseGrainedHessian[i][j] = reducedSpaceHessian[i][j];
					}
				}
			}


		private:
			// void initialize()
			void initialize();

			// void formBlocks()
			// 1. Allocate an NxN block Hessian matrix (blockHessian)
			// 2. Compute the individual blocks using numerical differentiation
			// 3. Place these blocks on the diagonal of blockHessian
			void formBlocks();

			// void diagonalizeBlocks()
			// 1. Take the diagonal blocks from the blockHessian matrices
			// 2. Diagonalize each of these blocks
			// 3. Allocate and populate blockEigenvalues and blockEigenvectors
			//     a. Compute geometric degrees of freedom
			//     b. Orthogonalization
			// 4. Call sortEigenvectors() to sort eigenvectors by eigenvalue
			// 5. Allocate and populate the E matrix
			void diagonalizeBlocks();

			// void formProjectionMatrix()
			// 1. Use eigenvectors below cutoff to form projection matrix
			void formProjectionMatrix();

			// void computeHE()
			// 1. We need this function to avoid computing S=E^T H E directly
			//    which requires the full Hessian H.
			//    Instead, we compute HE using these steps:
			//      a. perturb positions by eps E M^(-1/2)
			//      b. run numerical differentiation
			void computeHE();

			// void computeS()
			// 1. Simple matrix multiplication, S = E^T (HE)
			void computeS();

			// void diagonalizeS()
			// 1. Diagonalize the S matrix
			// 2. Allocate and populate dS and Q
			// 3. Call sortEigenvectors() to sort Q by the eigenvalues in dS
			void diagonalizeS();

			// void computeModes()
			// 1. Simple matrix multiplication, U = EQ
			// 2. Note this will resize and populate the passed modes and
			void computeModes( std::vector<double> &eigenvalues, std::vector<std::vector<OpenMM::Vec3> > &modes );

			// unsigned int blockNumber(const int p) const
			// Return block index for dof p.
			unsigned int blockNumber( const int p ) const;

			// bool inSameBlock(const int p1, const int p2, const int p3, const int p4) const
			// Determines if dof p1 ... p4 are in the same block.  p3 and p4 are optional.
			bool inSameBlock( const int p1, const int p2, const int p3, const int p4 ) const;

			// std::vector<EigenvalueColumn> sortEigenvalues(const EigenvalueArray& values) const;
			// 1. Sorts eigenvalues by absolute magnitude
			// 2. Returns std::pair<double, unsigned int> of eigenvalue and original index.
			std::vector<EigenvalueColumn> sortEigenvalues( const EigenvalueArray &values ) const;

			void determineBlockSizes( int residuesPerBlock, const std::vector<int> &residueSizes );


			// void diagonalizeBlock()
			// 1. Calls MKL on each block
			// 2. Computes Geometric Degrees of Freedom (GDOF)
			// 3. Orthogonalizes block eigenvectors against GDOF
			// 4. Returns orthogonalized set
			void diagonalizeBlock( const unsigned int block, const TNT::Array2D<double> &hessian,
								   const std::vector<OpenMM::Vec3> &positions, TNT::Array1D<double> &eval, TNT::Array2D<double> &evec );

			int particleCount;
			int largestBlockSize;
			std::vector<double> mParticleMass;
			std::vector<int> blockSizes;

			std::vector<OpenMM::Vec3> initialPositions;

			TNT::Array2D<double> blockHessian;

			TNT::Array1D<double> blockEigenvalues;
			TNT::Array2D<double> blockEigenvectors;

			TNT::Array2D<double> projectionMatrix;
			TNT::Array2D<double> transposedProjectionMatrix;

			TNT::Array2D<double> productMatrix;

			TNT::Array2D<double> reducedSpaceHessian;

			TNT::Array1D<double> reducedSpaceEigenvalues;
			TNT::Array2D<double> reducedSpaceEigenvectors;
	};
}

#endif
