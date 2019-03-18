#! /usr/bin/env python
# Code implementing "Efficient Computer Manipulation of Tensor Products..."
# Pereyra Scherer
# Assumes all factor matrices square, identical size
# TODO use pycontracts to enforce this ^

from scipy.stats import ortho_group
import numpy as np
from operator import mul
from functools import reduce

DEBUG = False 

class KronProd:
    def __init__(self, As):
        self.As = As
        self.flat_A = np.concatenate([a.flatten() for a in self.As], axis=None)
        if DEBUG:
            print(self.flat_A)
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.Y = np.empty(shape=self.N, dtype = np.float64)
        self.X = np.empty(shape=self.n[0]**self.nmat)

    def contract(self, nk, mk, ki):
        ktemp = 0
        inic = ki*(nk*nk)
        if DEBUG:
            print("nk = {}, mk = {}, ki = {}".format(nk, mk, ki))
        for i in range(nk): # dim of matrix k
            J = 0
            for s in range(int(mk)): # N / nk
                I = inic
                sum = 0.0
                for t in range(nk): # dim of matrix k
#                    if(self.flat_A[I] == 1):
#                        print("A",I,J)
                #    if((I-J) % 4 == 0 and (I in [0,5,10,15,16,21,26,31])):
                #        print("B",I, J)
                #        sum = sum + (1-(self.flat_A[I]*self.X[J]))
                #    else:
                    sum = sum + (self.flat_A[I] * self.X[J])
                    if DEBUG:
                        print ("I = {}, J = {}".format(I,J))
                    I = I + 1
                    J = J + 1
            #    print("Sum = {}".format(sum))
                self.Y[ktemp] = sum
                if DEBUG:
                    print("Sum = {}".format(sum))
                    print("setting element",ktemp,"of Y")
                    print("Y is now", self.Y)
                ktemp = ktemp + 1
            inic = I
            if DEBUG:
                print("inic = ", inic)
        np.copyto(self.X, self.Y)

    def dot(self, x):
        np.copyto(self.X, x)
        k = self.nmat
        nk = self.n[k-1]
        mk = self.N/nk
        for ki in range(k):
            if DEBUG:
                print("IN CONTRACTION ",ki)
                print("mk: ", mk)
            mk = self.N/self.n[k-1-ki]
            self.contract(nk, mk, ki)
        print("________________RESULTS___________________")
        print("[DEBUG] Y = {}, sum = {}".format(self.Y, np.sum(self.Y)))

        return self.Y

# Example code
# ------------

if __name__ == '__main__':

    n = 5#number of factors
    p = 5 # dimension of factor
    r_As = [ortho_group.rvs(dim=p) for i in range(n)]
		#Make first and second row the same so that way it becomes a non-invertible matrix
    for A in r_As:
        A[1,:] = A[0,:]
    As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
    y = np.random.rand(p**n)

    big_A = reduce(np.kron, As)
    big_x = big_A.dot(y)
    print("[test_kron_inv - testRandom_pInv] full calc: ",big_x)

    kp = KronProd(list(reversed(As)))
    x = kp.dot(y)
    print("[test_kron_inv - testRandom_pInv] efficient calc: ", x)

    print(np.allclose(x,big_x))



