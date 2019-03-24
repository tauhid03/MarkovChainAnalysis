#! /usr/bin/env python

from fail import As_fail, Y_fail
from scipy.stats import ortho_group
import unittest
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import reduce
from src.operations import invert
from src.kronprod import KronProd
class TestKronInv(unittest.TestCase):

    # add global stuff here
    def setUp(self):
        return

    # add global stuff here
    def setUp(self):
        return
	#Make some tests if the matrix IS invertible
    def testOnes_inv(self):
        A1 = [ np.array([[1., 1.], [1.,1.]]),
                    np.array([[1.,1.], [1.,1.]])]
        x1 = np.array([1.,1.,1.,1.])
        y1 = np.array([4,4,4,4])
        kp = KronProd(invert(A1))
        x = kp.dot(y1)
        np.testing.assert_almost_equal(x, x1, decimal=7, verbose=True)


    # this dimensionality pushes the limit of what full rank calc can do
    def testRandom_inv(self):
        n = 5 # number of factors
        p = 5 # dimension of factor
        r_As = [ortho_group.rvs(dim=p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        y = np.random.rand(p**n)

        big_A = reduce(np.kron, As)
        big_x = np.linalg.solve(big_A, y)
        print("full calc: ",big_x)

        kp = KronProd(invert(As))
        x = kp.dot(y)
        print("efficient calc: ", x)

        np.testing.assert_almost_equal(x, big_x, decimal=7, verbose=True)

    def testBig_inv(self):
        n = 2 # number of factors
        p = 100 # dimension of factor
        r_As = [ortho_group.rvs(dim=p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        y = np.random.rand(p**n)
        kp = KronProd(invert(As))
        x = kp.dot(y)
        print("efficient calc: ", x)

	#Make some tests if the matrix ISNT invertible

    def testInts_pInv(self):
        A1 = [ np.array([[1.0, 0.0], [0.0,0.0]]),
                    np.array([[1.,1.], [0.,0.]])]
        y1 = np.array([1.,2.,3.,4.])
        A1_inv = []
        for a in A1:
            A1_inv.append(np.linalg.pinv(a))
        big_A = reduce(np.kron, A1_inv)
        big_x = big_A.dot(y1)
        print("FOO")
        print("full calc: ",big_x)
        kp = KronProd(invert(A1))
        x = kp.dot(y1)
        print("efficient calc: ", x)
        print("BAR")
        self.assertSequenceEqual(list(x), list(big_x))


    # this dimensionality pushes the limit of what full rank calc can do
    def testRandom_pInv(self):
        n = 5 # number of factors
        p = 5 # dimension of factor
        r_As = [ortho_group.rvs(dim=p) for i in range(n)]
		#Make first and second row the same so that way it becomes a non-invertible matrix
        for A in r_As:
            A[1,:] = A[0,:]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        y = np.random.rand(p**n)
        As_inv = []
        for a in As:
            As_inv.append(np.linalg.pinv(a))

        big_A = reduce(np.kron, As_inv)
        big_x = big_A.dot(y)
        print("[test_kron_inv - testRandom_pInv] full calc: ",big_x)

        kp = KronProd(invert(As))
        x = kp.dot(y)
        print("[test_kron_inv - testRandom_pInv] efficient calc: ", x)

        np.testing.assert_almost_equal(big_x, x, decimal=7, verbose=True)

    def testBig_pInv(self):
        n = 2 # number of factors
        p = 100 # dimension of factor
        r_As = [np.random.rand(p,p) for i in range(n)]
		#Make first and second row the same so that way it becomes a non-invertible matrix
        for A in r_As:
            A[1,:] = A[0,:]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        x = np.random.rand(p**n)
        kp = KronProd(invert(As))
        Y = kp.dot(x)
        print("efficient calc: ", Y)

    def testFail(self):
        #For some reason this breaks the property of kron product - (A_1 kron A_2)^+ = A_1^+ kron A_2^+
        #The equivalent test for this is testRandom_pInv, but the reduce is performed after pinv. If it isn't then we get this error.
        #This error didn't seem to happen for n=p=3 or 2 but does occur sometimes when n=p=4 and somewhat frequently when n=p=5.
        n = 4
        p = 4
        As = As_fail
        y = Y_fail
        big_A = reduce(np.kron, As)
        big_A_inv = np.linalg.pinv(big_A)
        big_x = big_A_inv.dot(y)
        kp = KronProd(invert(As))
        x = kp.dot(y)
        if(np.allclose(x,big_x) == False):
            return
        else:
            self.fail("No equivalent!")




if __name__ == '__main__':
    unittest.main()
