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
            std::vector<std::pair<int, int> > bonds;
            std::vector<std::vector<int> > particleBonds;
            std::vector<std::vector<double> > projection;
            std::vector<std::vector<OpenMM::Vec3> > eigenvectors;
	    std::vector<double> eigenvalues;
            double maxEigenvalue;
            OpenMM::Context *blockContext;
            std::vector<int> blocks;
    };
}

#endif // FLEXIBLE_BLOCK_METHOD_H_
