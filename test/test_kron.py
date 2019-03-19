#! /usr/bin/env python

import unittest
from pathlib import Path

print(Path('/home/username').parent)
import numpy as np
from functools import reduce
from src.kronprod import KronProd

class TestKron(unittest.TestCase):

    # add global stuff here
    def setUp(self):
        return

    def testOnes(self):
        A1 = [ np.array([[1., 1.], [1.,1.]]),
                    np.array([[1.,1.], [1.,1.]])]
        x1 = np.array([1.,1.,1.,1.])
        y1 = np.array([4.,4.,4.,4.])
        kp = KronProd(list(reversed(A1)))
        y = kp.dot(x1)
        self.assertSequenceEqual(list(y), list(y1))

    def testInts(self):
        A1 = [ np.array([[1.0, 0.0], [0.0,0.0]]),
                    np.array([[1.,1.], [0.,0.]])]
        x1 = np.array([1.,2.,3.,4.])
        big_A = reduce(np.kron, A1)
        print(big_A)
        print(x1)
        big_y = np.matmul(big_A, x1)
        print("full calc: ",big_y)
        kp = KronProd(list(reversed(A1)))
        Y = kp.dot(x1)
        print("efficient calc: ", Y)
        self.assertSequenceEqual(list(Y), list(big_y))

    # this dimensionality pushes the limit of what full rank calc can do
    def testRandom(self):
        n = 5 # number of factors
        p = 5 # dimension of factor
        r_As = [np.random.rand(p,p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        x = np.random.rand(p**n)

        big_A = reduce(np.kron, As)
        big_y = np.matmul(big_A, x)
        print("full calc: ",big_y)

        kp = KronProd(list(reversed(As)))
        Y = kp.dot(x)
        print("efficient calc: ", Y)

        np.testing.assert_almost_equal(big_y, Y, decimal=7, verbose=True)

    def testBig(self):
        n = 2 # number of factors
        p = 100 # dimension of factor
        r_As = [np.random.rand(p,p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        x = np.random.rand(p**n)
        kp = KronProd(list(reversed(As)))
        Y = kp.dot(x)
        print("efficient calc: ", Y)


if __name__ == '__main__':
    unittest.main()
