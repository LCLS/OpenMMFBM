/* Class CulaDiagonalize
 * Purpose: Run diagonalization of some matrix A on the GPU
 *          using Cula
 */

#ifndef CULADIAGONALIZE_H
#define CULADIAGONALIZE_H

#include "Diagonalize.h"

#ifdef HAVE_CULA

class CulaDiagonalize : public Diagonalize {
   public:
      // void run()
      // Populate eigval and eigvec using Cula routines
      // for matrix diagonalization
      // All of these can be GPU memory
      void run(float** A, float* eigval, float** eigvec);
}

#endif

#endif
