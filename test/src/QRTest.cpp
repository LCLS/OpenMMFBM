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

			for( size_t i = 0; i < 16; i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( matrix[i], expected_values[i], 1e-5 );
			}
		}

		void Test::MultipleMatrix4() {
			float matrix[32] = {
				0.1734, 0.0605, 0.6569, 0.0155,
				0.0605, 0.3993, 0.6280, 0.9841,
				0.6569, 0.6280, 0.2920, 0.1672,
				0.0155, 0.9841, 0.1672, 0.1062,

				0.8147, 0.6324, 0.9575, 0.9572,
				0.6324, 0.0975, 0.9649, 0.4854,
				0.9575, 0.9649, 0.1576, 0.8003,
				0.9572, 0.4854, 0.8003, 0.1419
			};

			const float expected_values[32] = {
				-0.8622,  0.0000, 0.0000, 0.0000,
				 0.0000, -0.4136, 0.0000, 0.0000,
				 0.0000,  0.0000, 0.6232, 0.0000,
				 0.0000,  0.0000, 0.0000, 1.6235,

				-0.9117,  0.0000,  0.0000, 0.0000
				 0.0000, -0.5354,  0.0000, 0.0000
				 0.0000,  0.0000, -0.1103, 0.0000
				 0.0000,  0.0000,  0.0000, 2.7691
			};

			const float expected_vectors[32] = {
				 0.2221,  0.6548,  0.6737, -0.2608,
				 0.6565, -0.1531, -0.3244, -0.6636,
				-0.3963, -0.5818,  0.5014, -0.5030,
				-0.6022,  0.4575, -0.4352, -0.4885,

				 0.0817, -0.5371,  0.5823, 0.6047,
				 0.5700, -0.1942, -0.6850, 0.4101,
				-0.7773,  0.0679, -0.3596, 0.5117,
				 0.2536,  0.8180,  0.2496, 0.4520
			};

			int index[2] = { 0, 16 };
			int widths[2] = { 4, 4 };

			BlockQR_HOST(2, matrix, 32, index, 2, widths, 2);

			for( size_t i = 0; i < 32; i++ ) {
				CPPUNIT_ASSERT_DOUBLES_EQUAL( matrix[i], expected_values[i], 1e-5 );
			}
		}
	}
}
