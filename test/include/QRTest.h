#ifndef OPENMM_FBM_QRTEST_H_
#define OPENMM_FBM_QRTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace FBM {
	class QR : public CppUnit::TestFixture  {
		private:
			CPPUNIT_TEST_SUITE(QR);
			CPPUNIT_TEST(QRStep);
			CPPUNIT_TEST(SingleMatrix4);
			CPPUNIT_TEST(MultipleMatrix4);
			CPPUNIT_TEST_SUITE_END();
		public:
			void QRStep();
			void SingleMatrix4();
			void MultipleMatrix4();
	};
}

#endif  // OPENMM_FBM_QRTEST_H_
