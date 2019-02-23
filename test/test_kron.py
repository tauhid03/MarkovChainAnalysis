#! /usr/bin/env python

import unittest

from src.kronprod_sparse import *
from src.kronprod import KronProd
from hittingTimes import *

class TestKron(unittest.TestCase):

    # add global stuff here
    def setUp(self):
        return

    def testOnesDense(self):
        A1 = [ np.array([[1., 1.], [1.,1.]]),
                    np.array([[1.,1.], [1.,1.]])]
        X = np.array([1.,1.,1.,1.])
        y1 = np.array([4.,4.,4.,4.])

        kp = KronProd(A1)
        y = kp.dot(X)
        self.assertSequenceEqual(list(y), list(y1))

    def testOnesSparse(self):
        A1 = [ np.array([[1., 1.], [1.,1.]]),
                    np.array([[1.,1.], [1.,1.]])]
        X = np.array([1.,1.,1.,1.])
        y1 = np.array([4.,4.,4.,4.])

        A2 = np.concatenate([a.flatten() for a in A1], axis=None)
        A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
        A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))

        X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
        X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
        x_keys = X_dok.keys()
        x_keys = sorted(x_keys, key=itemgetter(1))
        a_keys = A2_dok.keys()
        a_keys = sorted(a_keys, key=itemgetter(1))
        kp = KronProdSparse(A1, A2, a_keys, x_keys, A2, X)
        y = kp.dot(A2, X)
        self.assertSequenceEqual(list(y), list(y1))
         
    def testInts(self):
        A1 = [ np.array([[1.0, 0.0], [0.0,0.0]]),
                    np.array([[1.,1.], [0.,0.]])]
        X = np.array([1.,2.,3.,4.])
        big_A = reduce(np.kron, A1)
        big_y = np.matmul(big_A, X)

        A2 = np.concatenate([a.flatten() for a in list(reversed(A1))], axis=None)
        A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
        A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))

        X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
        X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
        x_keys = X_dok.keys()
        x_keys = sorted(x_keys, key=itemgetter(1))
        a_keys = A2_dok.keys()
        a_keys = sorted(a_keys, key=itemgetter(1))
        kp = KronProdSparse(A1, A2, a_keys, x_keys, A2, X)
        Y = kp.dot(A2, X)

        self.assertSequenceEqual(list(Y), list(big_y))

    # this dimensionality pushes the limit of what full rank calc can do
    def testRandom(self):
        n = 5 # number of factors
        p = 7 # dimension of factor
        r_As = [np.random.rand(p,p) for i in range(n)]
        A1 = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        X = np.random.rand(p**n)

        big_A = reduce(np.kron, A1)
        big_y = np.matmul(big_A, X)

        A2 = np.concatenate([a.flatten() for a in list(reversed(A1))], axis=None)
        A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
        A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))

        X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
        X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
        x_keys = list(X_dok.keys())
        x_keys = sorted(x_keys, key=itemgetter(1))
        a_keys = list(A2_dok.keys())
        a_keys = sorted(a_keys, key=itemgetter(1))
        kp = KronProdSparse(A1, A2, a_keys, x_keys, A2, X)
        Y = kp.dot(A2, X)

        np.testing.assert_almost_equal(big_y, Y, decimal=7, verbose=True)

    # took ~150 seconds
    def testBig(self):
        n = 5 # number of factors
        p = 20 # dimension of factor
        r_As = [np.identity(p) for i in range(n)]
        A1 = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        X = np.random.rand(p**n)

        A2 = np.concatenate([a.flatten() for a in list(reversed(A1))], axis=None)
        A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
        A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))

        X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
        X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
        x_keys = list(X_dok.keys())
        x_keys.sort(key=itemgetter(1))
        a_keys = list(A2_dok.keys())
        a_keys.sort(key=itemgetter(1))
        kp = KronProdSparse(A1, A2, a_keys, x_keys, A2, X)
        Y = kp.dot(A2, X)

    def testHittingTimes(self):
        n = 2 # number of factors
        p = 4 # dimension of factor
        r_As = [np.random.rand(p,p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        hittingtime(As)


if __name__ == '__main__':
    unittest.main()
