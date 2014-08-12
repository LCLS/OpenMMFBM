#ifndef OPENMM_FBM_CUDATEST_H_
#define OPENMM_FBM_CUDATEST_H_

#include <cppunit/extensions/HelperMacros.h>

namespace FBM {
    class CUDA : public CppUnit::TestFixture  {
        private:
            CPPUNIT_TEST_SUITE(CUDA);
            CPPUNIT_TEST(MakeProjection);
            CPPUNIT_TEST(OddEvenSort);
            CPPUNIT_TEST(CopyFrom);
            CPPUNIT_TEST(BlockCopyFrom);
            CPPUNIT_TEST(CopyTo);
            CPPUNIT_TEST(BlockCopyTo);
            CPPUNIT_TEST(MatrixMultiplication);
            CPPUNIT_TEST(Symmetrize1D);
            CPPUNIT_TEST(Symmetrize2D);
            CPPUNIT_TEST(MakeEigenvalues);
            CPPUNIT_TEST(MakeBlockHessian);
            CPPUNIT_TEST(PerturbPositions);
            CPPUNIT_TEST(PerturbByE);
            CPPUNIT_TEST(ComputeNormsAndCenter);
            CPPUNIT_TEST(MakeHE);
            CPPUNIT_TEST_SUITE_END();
        public:
            void MakeProjection();
            void OddEvenSort();
            void CopyFrom();
            void BlockCopyFrom();
            void CopyTo();
            void BlockCopyTo();
            void MatrixMultiplication();
            void Symmetrize1D();
            void Symmetrize2D();
            void MakeEigenvalues();
            void MakeBlockHessian();
            void PerturbPositions();
            void PerturbByE();
            void ComputeNormsAndCenter();
            void MakeHE();
    };
}

#endif  // OPENMM_FBM_CUDATEST_H_
