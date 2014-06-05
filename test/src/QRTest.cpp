#include "QRTest.h"

#include <helper_cuda.h>

#include <cuda_runtime.h>
#include <cppunit/extensions/HelperMacros.h>

CPPUNIT_TEST_SUITE_REGISTRATION(FBM::QR::Test);

extern "C" void BlockQR_HOST( const int n_mat, float* matrix, const size_t matrix_size, const int* index, const size_t index_size, const int* sizes, const size_t sizes_size );

namespace FBM {
	namespace QR {
		void Test::SingleMatrix4() {
            float matrix[16] = {
                0.1734, 0.0605, 0.6569, 0.0155,
                0.0605, 0.3993, 0.6280, 0.9841,
                0.6569, 0.6280, 0.2920, 0.1672,
                0.0155, 0.9841, 0.1672, 0.1062
            };

			const float expected_values[16] = {
				-0.8622,  0.0000, 0.0000, 0.0000,
				 0.0000, -0.4136, 0.0000, 0.0000,
				 0.0000,  0.0000, 0.6232, 0.0000,
				 0.0000,  0.0000, 0.0000, 1.6235
			};

			const float expected_vectors[16] = {
				 0.2221,  0.6548,  0.6737, -0.2608,
				 0.6565, -0.1531, -0.3244, -0.6636,
				-0.3963, -0.5818,  0.5014, -0.5030,
				-0.6022,  0.4575, -0.4352, -0.4885
			};

            int index[1] = { 0 };
            int widths[1] = { 4 };

            BlockQR_HOST(1, matrix, 16, index, 1, widths, 1);

			for( size_t i = 0; i < 4; i++ ) {
				for( size_t j = 0; j < 4; j++ ) {
					printf("%f ", matrix[i*4+j]);
				}
				printf("\n");
			}

			cudaSetDevice(0);
        	cudaDeviceReset();
		}
	}
}
