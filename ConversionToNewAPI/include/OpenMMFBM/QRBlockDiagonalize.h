/* Class QRBlockDiagonalize
 * Purpose: Run diagonalization of some matrix A on the CPU
 *          with blocks on its diagonal
 *          Uses a QR algorithm parallelized on the GPU
 *          The algorithm is parallelized by blocks and portions
 *          of each block
 */

#ifndef QRBLOCKDIAGONALIZE_H
#define QRBLOCKDIAGONALIZE_H

class QRBlockDiagonalize {
   public:
      // void run()
      // Diagonalize the blocks of a matrix A (these blocks are assumed to be
      // on the diagonal).  The ending degrees of freedom of every block are in
      // blockStop.  Diagonalize the blocks using GPU QR and populate eigval and
      // eigvec.
      virtual void run(float* A, int* blockStop, float* eigval, float* eigvec);
}


#endif
