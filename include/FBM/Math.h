#ifndef FBM_MATH_H_
#define FBM_MATH_H_

#include "jama_eig.h"

void matrixMultiply( const TNT::Array2D<double> &a, const TNT::Array2D<double> &b, TNT::Array2D<double> &c );
void diagonalizeSymmetricMatrix( const TNT::Array2D<double> &matrix, TNT::Array1D<double> &values, TNT::Array2D<double> &vectors );

#endif // FBM_MATH_H_
