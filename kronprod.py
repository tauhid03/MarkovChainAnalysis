#! /usr/bin/env python
# Code implementing "Efficient Computer Manipulation of Tensor Products..."
# Pereyra Scherer
# Assumes all factor matrices square, identical size
# TODO use pycontracts to enforce this ^

import numpy as np
from operator import mul
from functools import reduce

DEBUG = False

# TODO investigate LinearOperators for even moar fast
#from scipy.sparse.linalg import LinearOperator

class KronProd:
    def __init__(self, As):
        self.As = As
        self.flat_A = np.concatenate([a.flatten() for a in self.As], axis=None)
        print(self.flat_A)
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.Y = np.empty(shape=self.N, dtype = np.float64)
        self.X = np.empty(shape=self.n[0]**self.nmat)
        #self.Y = np.empty(shape=self.N, dtype = object)
        #self.X = x

    def contract(self, nk, mk, ki):
        ktemp = 0
        inic = ki*(nk*nk)
        if DEBUG:
            print("nk = {}, mk = {}, ki = {}".format(nk, mk, ki))
        for i in range(nk): # dim of matrix k
            J = 0
            for s in range(int(mk)): # N / nk
                I = inic
                #sum = ""
                sum = 0.0
                for t in range(nk): # dim of matrix k
                    #sum = sum +"+"+ self.flat_A[I]+"*("+self.X[J]+")"
                    sum = sum + self.flat_A[I]*self.X[J]
           #         print("A_Val = {}, X_Val = {}".format(self.flat_A[I], self.X[J]))
                    if DEBUG:
                        pass
                        print ("I = {}, J = {}".format(I,J))
           #             print("elem",I,"of",self.flat_A)
           #             print("elem",J,"of",self.X)
           #             print("sum=", sum)
                    I = I + 1
                    J = J + 1
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
        return self.Y
if __name__ == '__main__':
    n = 2 # number of factors
    p = 4 # dimension of factor
    A = np.array([[.2,.4,0, .4],
              [.4, .2, .4, 0],
              [0, .4, .2, .4],
              [.4, 0, .4, .2]])


    r_As = [A for i in range(n)]
  #  As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
    x = np.random.rand(p**n)
    x = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.1,.2,.3,.4,.5,.6,.7])
    print("X= {}".format(x))

    kp = KronProd(list(reversed(r_As)))
    Y = kp.dot(x)
    print("Y = {}".format(Y))

    big_A = reduce(np.kron, r_As)
    big_y = np.matmul(big_A, x)
    print("full calc: ",big_y)


