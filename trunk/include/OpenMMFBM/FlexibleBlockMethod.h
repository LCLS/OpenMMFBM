#ifndef FLEXIBLE_BLOCK_METHOD_H_
#define FLEXIBLE_BLOCK_METHOD_H_

#include <map>
#include <utility>
#include <vector>

#include "OpenMM.h"
#include "jama_eig.h"
#include "OpenMMFBM/FBMParameters.h"

namespace OpenMMFBM {
    typedef TNT::Array1D<double> EigenvalueArray;
    typedef std::pair<double,int> EigenvalueColumn;
    
    class OPENMM_EXPORT FlexibleBlockMethod {
        public:
            FlexibleBlockMethod( const FBMParameters& params ) : mParticleCount( 0 ), mParams( params ), mLargestBlockSize( -1 ) {
                mInitialized = false;
                blockContext = NULL;
            }

            ~FlexibleBlockMethod() {
                if( blockContext ) {
                    delete blockContext;
                }
            }

            void diagonalize( OpenMM::Context &context );
        
            const std::vector<std::vector<OpenMM::Vec3> >& getEigenvectors() const {
                return eigenvectors;
            }

	    const std::vector<double>& getEigenvalues() const {
	      return eigenvalues;
	    }
        
            double getMaxEigenvalue() const {
                return maxEigenvalue;
            }

	    void getBlockHessian(std::vector<std::vector<double > >& blockHessianVectors) const {
	      const unsigned int N = 3 * mParticleCount;
	      blockHessianVectors.resize(N, std::vector<double>(N, 0.0));
 
	      for(unsigned int i = 0; i < N; i++)
		{
		  for(unsigned int j = 0; j < N; j++)
		    {
		      blockHessianVectors[i][j] = mBlockHessian[i][j];
		    }
		}
	    }

	    void getBlockEigenvectors(std::vector<std::vector<double > >& blockEigenvectors) const {
	      const unsigned int N = 3 * mParticleCount;
	      blockEigenvectors.resize(N, std::vector<double>(N, 0.0));
 
	      for(unsigned int i = 0; i < N; i++)
		{
		  for(unsigned int j = 0; j < N; j++)
		    {
		      blockEigenvectors[i][j] = mBlockEigenvectors[i][j];
		    }
		}
	    }

	    void getProjectionMatrix(std::vector<std::vector<double > >& projectionMatrix) const {
	      const unsigned int nRows = mProjectionMatrix.dim1();
	      const unsigned int nCols = mProjectionMatrix.dim2();

	      projectionMatrix.resize(nRows, std::vector<double>(nCols, 0.0));
 
	      for(unsigned int i = 0; i < nRows; i++)
		{
		  for(unsigned int j = 0; j < nCols; j++)
		    {
		      projectionMatrix[i][j] = mProjectionMatrix[i][j];
		    }
		}
	    }

	    void getHE(std::vector<std::vector<double > >& HE) const {
	      const unsigned int nRows = mHE.dim1();
	      const unsigned int nCols = mHE.dim2();

	      HE.resize(nRows, std::vector<double>(nCols, 0.0));
 
	      for(unsigned int i = 0; i < nRows; i++)
		{
		  for(unsigned int j = 0; j < nCols; j++)
		    {
		      HE[i][j] = mHE[i][j];
		    }
		}
	    }



	    void getCoarseGrainedHessian(std::vector<std::vector<double > >& coarseGrainedHessian) const {
	      const unsigned int N = mCoarseGrainedHessian.dim1();

	      coarseGrainedHessian.resize(N, std::vector<double>(N, 0.0));
 
	      for(unsigned int i = 0; i < N; i++)
		{
		  for(unsigned int j = 0; j < N; j++)
		    {
		      coarseGrainedHessian[i][j] = mCoarseGrainedHessian[i][j];
		    }
		}
	    }

	    const std::vector<std::vector<OpenMM::Vec3> >& getSPositions() const {
	      return mSPositions;
	    }

	    const std::vector<std::vector<OpenMM::Vec3> >& getSForces() const {
	      return mSForces;
	    }
        
        private:
            unsigned int blockNumber( int );

            bool inSameBlock( int, int, int, int );
            
            const TNT::Array2D<double> calculateU( const TNT::Array2D<double>& E, const TNT::Array2D<double>& Q ) const;
            static std::vector<EigenvalueColumn> sortEigenvalues( const EigenvalueArray& values );

            void initialize( OpenMM::Context &context );
            void diagonalizeBlock( const unsigned int block, const TNT::Array2D<double>& hessian, 
            const std::vector<OpenMM::Vec3>& positions, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec );


            unsigned int mParticleCount;
            std::vector<double> mParticleMass;
        
            const FBMParameters& mParams;
            
            int mLargestBlockSize;
            bool mInitialized;
            std::vector<std::vector<OpenMM::Vec3> > eigenvectors;
	    std::vector<double> eigenvalues;
            double maxEigenvalue;
            OpenMM::Context *blockContext;
            std::vector<int> blocks;

	    TNT::Array2D<double> mBlockHessian;
	    TNT::Array2D<double> mBlockEigenvectors;
	    TNT::Array2D<double> mProjectionMatrix;
	    TNT::Array2D<double> mCoarseGrainedHessian;
	    std::vector<std::vector<OpenMM::Vec3> > mSPositions;
	    std::vector<std::vector<OpenMM::Vec3> > mSForces;
	    TNT::Array2D<double> mHE;
    };
}

#endif // FLEXIBLE_BLOCK_METHOD_H_
