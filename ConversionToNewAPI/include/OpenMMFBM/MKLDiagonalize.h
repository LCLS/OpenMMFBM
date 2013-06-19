/* Class MKLDiagonalize
 * Purpose: Run diagonalization of some matrix A on the CPU
 *          but parallelized using multicore Intel MKL
 */

#ifndef MKLDIAGONALIZE_H
#define MKLDIAGONALIZE_H

#ifdef INTEL_MKL

#include "Diagonalize.h"

class MKLDiagonalize : public Diagonalize {
   public:

      // void run()
      // Populate eigval and eigvec by running MKL's
      // matrix diagonalization algorithm on matrix A
      // All three of these refer to CPU memory
      void run(float** A, float* eigval, float** eigvec);
}

#endif

#endif
