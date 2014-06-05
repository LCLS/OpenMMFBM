#ifndef OPENMM_FBM_QRTEST_H_
#define OPENMM_FBM_QRTEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace FBM {
	namespace QR {
		class Test : public CppUnit::TestFixture  {
			private:
				CPPUNIT_TEST_SUITE(Test);
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
}

#endif  // OPENMM_FBM_QRTEST_H_
