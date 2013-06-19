/* Class BlockDiagonalize
 * Purpose: Provide an abstract interface for CPU and GPU diagonalization for blocks.
 * This allows the user to specifically control the platform for diagonalization
 * and separate it from the platform for the Flexible Block Method itself.
 * The decision should be made independently, since the benefits of using the 
 * GPU may not be the same for diagonalization.
 */

#ifndef BLOCKDIAGONALIZE_H
#define BLOCKDIAGONALIZE_H

class BlockDiagonalize {
   public:
      // void run()
      // This method will be overridden in child classes
      // In general, A will be a matrix with blocks on the diagonal
      //             blockStop will be an array consisting of the last degree
      //                of freedom of each block, i.e. blockStop[0] contains the
      //                last degree of freedom of the first block which starts at
      //                [0][0] and ends at [blockStop[0]][blockStop[0]]
      //             eigval and eigvec are populated by the function
      virtual void run(float* A, int* blockStop, float* eigval, float* eigvec)=0;
}


#endif
