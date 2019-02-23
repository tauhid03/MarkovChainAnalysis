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
import time

DEBUG = False

# TODO investigate LinearOperators for even moar fast
#from scipy.sparse.linalg import LinearOperator

class KronProdSparse:
    def __init__(self, As, flat_As, akeys, xkeys,A_dok, X_dok):
        self.As = As
        self.flat_A = flat_As
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.A_dok = A_dok
        self.Y = np.empty(shape=self.N, dtype = np.float64)
        self.X = X_dok
        self.xkeys = xkeys
        self.akeys = akeys
        self.akeys_full = copy.deepcopy(akeys)
        self.xval = 0 #Used in finding sum

        self.counter = 0 #Used to determine how many multiplications were performed
        if DEBUG:
            print("Shape is {}".format(self.n[0]**self.nmat))

    def contract(self, nk, mk, ki, A):
        time_A = 0.0
        ktemp = 0
        inic = ki*(nk*nk)
        if DEBUG:
            print("A = {}".format(A))
            print("X = {}".format(self.X))
            print("nk = {}, mk = {}, ki = {}".format(nk, mk, ki))
        for i in range(nk): # dim of matrix k
            self.xval = 0
            pairs = []
            while(self.xval < len(self.xkeys)):
                pairs = self.getPairs(inic, nk)
                #print("[DEBUG] Pairs = {}".format(pairs))
                pair_sum = 0.0
                counter = 0
                time_start_foo = time.time()
                for pair in pairs: # N / nk
                    foo = A[pair[0][1]]
                    bar = self.X[pair[1][1]]
                    pair_sum += foo * bar
                    self.counter += 1
                time_A += time.time()-time_start_foo
                self.Y[ktemp] = pair_sum
                ktemp += 1
                #Get highest value of list
                maxval = 0
            #    for pair in pairs:
            #        if (pair[1][1] > maxval):
            #            maxval = pair[1][1]
            #    for i in range(self.xkeys[0][1],self.xkeys[0][1]+nk):
             #       self.xkeys.remove((0,i))
            for pair in pairs:
                self.akeys.remove(pair[0])
            inic += nk
        np.copyto(self.X, self.Y)


    def getPairs(self, INIC, nk):
        pairs = []
        for a in self.akeys:
            if (a[1] >= INIC+nk):
                break
            pairs.append( (a, (0,(a[1]%nk) + self.xval)))
        self.xval = self.xval +  nk
        return pairs


    def dot(self, a, x):
        k = self.nmat
        nk = self.n[k-1]
        mk = self.N/nk
        if DEBUG:
            print("nk = {}, mk = {}, k = {}".format(nk,mk,k))
        for ki in range(k):
            if DEBUG:
                print("IN CONTRACTION ",ki)
                print("mk: ", mk)
            mk = self.N/self.n[k-1-ki]
            self.contract(nk, mk, ki, a )
        return self.Y

def benchmarkTestSparse(n,p):
    A1 = [np.identity(p) for i in range(n)]
    A2 = np.concatenate([a.flatten() for a in list(reversed(A1))], axis=None)
    A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
    A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))
        
    X = np.random.rand(p**n)
    X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
    X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
    x_keys = X_dok.keys()
    x_keys.sort(key=itemgetter(1))
    a_keys = A2_dok.keys()
    a_keys.sort(key=itemgetter(1))
    kp = KronProdSparse(A1, A2, a_keys, x_keys, A2_dok, X_dok)
    Y = kp.dot(A2_dok, X_dok)
    y_keys = Y.keys()
    y_keys.sort(key=itemgetter(1))

if __name__ == '__main__':
    n = 2
    p = 400
    A1 = [np.identity(p) for i in range(n)]
    A2 = np.concatenate([a.flatten() for a in list(reversed(A1))], axis=None)
    A2_csr = scipy.sparse.csr_matrix(A2) #For some reason I have to use this to reshape the DOK matrix
    A2_dok = scipy.sparse.dok_matrix(A2.reshape(A2_csr.shape))
    print("size of A = {}".format(A2_dok.shape))

        
    X = np.random.rand(p**n)
    X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
    X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))
    x_keys = X_dok.keys()
    x_keys.sort(key=itemgetter(1))
    a_keys = A2_dok.keys()
    a_keys.sort(key=itemgetter(1))
    kp = KronProdSparse(A1, A2, a_keys, x_keys, A2, X)
    time_start = time.time()
    Y = kp.dot(A2, X)
    print("Time to do dot = {}".format(time.time() - time_start))
  #  print("Y = ")
  #  y_keys = Y.keys()
        


