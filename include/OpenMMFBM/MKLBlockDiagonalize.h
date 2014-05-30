/* Class MKLBlockDiagonalize
 * Purpose: Run diagonalization of some matrix A on the CPU
 *          with blocks on its diagonal
 *          but parallelized using multicore Intel MKL
 */

#ifndef MKLBLOCKDIAGONALIZE_H
#define MKLBLOCKDIAGONALIZE_H

class MKLBlockDiagonalize : public BlockDiagonalize {
   public:
      // void run()
      // Diagonalize the blocks of a matrix A (these blocks are assumed to be
      // on the diagonal).  The ending degrees of freedom of every block are in
      // blockStop.  Diagonalize the blocks using MKL and populate eigval and
      // eigvec.
      virtual void run(float* A, int* blockStop, float* eigval, float* eigvec);
}


#endif
