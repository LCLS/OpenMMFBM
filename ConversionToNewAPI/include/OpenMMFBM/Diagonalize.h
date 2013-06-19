/* Class Diagonalize
 * Purpose: Provide an abstract interface for CPU and GPU diagonalization
 * This allows the user to specifically control the platform for diagonalization
 * and separate it from the platform for the Flexible Block Method itself.
 * The decision should be made independently, since the benefits of using the 
 * GPU may not be the same for diagonalization.
 */

#ifndef DIAGONALIZE_H
#define DIAGONALIZE_H

class Diagonalize {
   public:
      // void run()
      // This method will be overridden in child classes and call
      // the appropriate libraries for matrix diagonalization
      // In general, A will be the matrix to diagonalize (input)
      // eigval and eigvec will be populated by the method itself
      // Note that eah of these can be either GPU or CPU memory
      virtual void run(float** A, float* eigval, float** eigvec)=0;
}


#endif
