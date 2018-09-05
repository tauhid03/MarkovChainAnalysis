#! /usr/bin/env python
# Code implementing "Efficient Computer Manipulation of Tensor Products..."
# Pereyra Scherer
# Assumes all factor matrices square, identical size
# TODO use pycontracts to enforce this ^

import numpy as np
from operator import mul
from functools import reduce
from operator import itemgetter
import scipy.sparse
import copy

DEBUG = True 

# TODO investigate LinearOperators for even moar fast
#from scipy.sparse.linalg import LinearOperator

class KronProd:
    def __init__(self, As, flat_As, akeys, xkeys,A_dok, X_dok):
        self.As = As
        self.flat_A = flat_As
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.A_dok = A_dok
      #  self.Y = np.empty(shape=self.N, dtype = np.float64)
      #  self.X = np.empty(shape=self.n[0]**self.nmat)
      #  self.Y = scipy.sparse.dok_matrix(self.Y)
      #  self.X = scipy.sparse.dok_matrix(self.X)
        print("N = {}, n[0] = {}, nmat = {}".format(self.N, self.n[0], self.nmat))
        self.Y = scipy.sparse.dok_matrix((1, self.N), dtype=np.float32)
        self.X = X_dok
        self.xkeys = xkeys
        self.akeys = akeys
        self.xkeys_full = copy.deepcopy(xkeys)
        self.akeys_full = copy.deepcopy(akeys)
        self.xval = 0 #Used in finding sum
        print("Shape is {}".format(self.n[0]**self.nmat))
        #self.Y = np.empty(shape=self.N, dtype = object)
        #self.X = x

    def contract(self, nk, mk, ki, A):
        ktemp = 0
        inic = ki*(nk*nk)
        if DEBUG:
			print("nk = {}, mk = {}, ki = {}".format(nk, mk, ki))
        for i in range(nk): # dim of matrix k
            self.xkeys = copy.deepcopy(self.xkeys_full)
            self.xval = 0
            pairs = []
            while(self.xval < len(self.xkeys_full)):
                print("Xval = {}, key size = {}".format(self.xval, len(self.xkeys_full)))
                pairs = self.getPairs(inic, nk, A, self.X)
                print("Pairs = {}".format(pairs))
                pair_sum = 0.0
                counter = 0
                for pair in pairs: # N / nk
                    foo = A[pair[0]]
                    bar = self.X[pair[1]]
                    pair_sum += foo * bar
                    print("A_val = {}, X_val = {}".format(A[pair[0]], self.X[pair[1]]))
                print("Sum = {}".format(pair_sum))
                self.Y[(0,ktemp)] = pair_sum
                ktemp += 1
                for pair in pairs:
                    self.xkeys.remove(pair[1])
            for pair in pairs:
                self.akeys.remove(pair[0]) 
            inic += nk
        #copy Y to X
        for key in self.Y.keys():
            self.X[key] = self.Y[key]
        print(self.Y)
        print("FINAL A KEYS = {}".format(self.akeys))

    def getPairs(self, INIC, nk, A, X):
        pairs = []
        for a in self.akeys:
            if (a[1] >= INIC+nk):
                break
            for x in self.xkeys:
                if(x[1] >= self.xval + nk):
                    break
                if(a[1] == ((x[1] % nk) + INIC)):
                    pairs.append( (a,x) )
        self.xval = self.xval +  nk
        return pairs

    def dot(self, a, x):
        k = self.nmat
        nk = self.n[k-1]
        mk = self.N/nk
        for ki in range(k):
            if DEBUG:
                print("IN CONTRACTION ",ki)
                print("mk: ", mk)
            mk = self.N/self.n[k-1-ki]
            self.contract(nk, mk, ki, a )
        return self.Y




if __name__ == '__main__':
    n = 2
    p = 4
    A1 = [A for i in range(n)]
    A2 = np.concatenate([a.flatten() for a in A1], axis=None)
    A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
    A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))
	
    X = np.random.rand(p**n)
    X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
    X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
    x_keys = X_dok.keys()
    x_keys.sort(key=itemgetter(1))
    a_keys = A2_dok.keys()
    a_keys.sort(key=itemgetter(1))
    print("X-keys = {}, A-keys = {}".format(x_keys, a_keys))
    kp = KronProd(A1, A2, a_keys, x_keys, A2_dok, X_dok)
    Y = kp.dot(A2_dok, X_dok)
    print("Y = ")
    y_keys = Y.keys()
    y_keys.sort(key=itemgetter(1))
    for key in y_keys:
        print(Y[key])
	


