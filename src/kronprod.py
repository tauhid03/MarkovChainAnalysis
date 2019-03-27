#! /usr/bin/env python
# Code implementing "Efficient Computer Manipulation of Tensor Products..."
# Pereyra Scherer
# Assumes all factor matrices square, identical size
# TODO use pycontracts to enforce this ^

from scipy.stats import ortho_group
import numpy as np
from operator import mul
from functools import reduce
from operator import itemgetter
import scipy.sparse
import copy
import time

TIMING_ANALYSIS = False
DEBUG = False 

class KronProd:
    def __init__(self, As, sparse_flag=False):
        if DEBUG:
            print("SparseFlag = ",sparse_flag)
        self.sparse = sparse_flag
        if(self.sparse):
            self.createSparse(As)
        else:
            self.As = list(reversed(As))
            self.flat_A = np.concatenate([a.flatten() for a in self.As], axis=None)
            if DEBUG:
                print(self.flat_A)
            self.nmat = len(self.As)
            self.n = [len(a) for a in self.As] # dimensions of factors
            self.N = reduce(mul, self.n, 1) # size of final vector y = A*x self.Y = np.empty(shape=self.N, dtype = np.float64)
            self.Y = np.empty(shape=self.N, dtype = np.float64)
            self.X = np.empty(shape=self.n[0]**self.nmat)
    def createSparse(self,As):
        self.As = As
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.Y = np.empty(shape=self.N, dtype = np.float64)
        self.xval = 0 #Used in finding sum
        self.X = None

        self.flat_As = np.concatenate([matrix.flatten() for matrix in list(reversed(self.As))], axis=None)
        #Make the matrices sparse
        markov_matrices_csr = scipy.sparse.csr_matrix(self.flat_As) #For some reason I have to use this to reshape the DOK matrix
        markov_matrices_dok = scipy.sparse.dok_matrix(self.flat_As.reshape(markov_matrices_csr.shape))
        #Get A keys and sort them.
        a_keys = list(markov_matrices_dok.keys())
#        a_keys.sort(key=itemgetter(1))
        a_keys = sorted(a_keys, key=itemgetter(1))

        self.akeys = a_keys
        self.akeys_full = copy.deepcopy((self.akeys))

        self.counter = 0 #Used to determine how many multiplications were performed
        if DEBUG:
            print("Shape is {}".format(self.n[0]**self.nmat))

    def updateAkeys(self):
        self.akeys = copy.deepcopy(self.akeys_full)

    def updateA(self,As):
        self.As = As
        self.nmat = len(self.As)
        self.n = [len(a) for a in self.As] # dimensions of factors
        self.N = reduce(mul, self.n, 1) # size of final vector y = A*x
        self.Y = np.empty(shape=self.N, dtype = np.float64)
        self.xval = 0 #Used in finding sum
        self.X = None

        self.flat_As = np.concatenate([matrix.flatten() for matrix in list(reversed(self.As))], axis=None)
        #Make the matrices sparse
        markov_matrices_csr = scipy.sparse.csr_matrix(self.flat_As) #For some reason I have to use this to reshape the DOK matrix
        markov_matrices_dok = scipy.sparse.dok_matrix(self.flat_As.reshape(markov_matrices_csr.shape))
        #Get A keys and sort them.
        a_keys = markov_matrices_dok.keys()
        a_keys.sort(key=itemgetter(1))

        self.akeys = a_keys
        self.akeys_full = copy.deepcopy(self.akeys)

    #This is where the bulk of the speedup comes from using sparse matrices. It was noticable from running the algorithm that for a given element index in A, nk, and inic a given x would be chosen as its "pair". This pattern is generalized as a-INIC = x % nk.  This can be transformed into x = (a % nk) + self.xval. Self.xval is used because the corresponding x val needs to be within a range of #*nk to (#+1)*nk where # is the amount of times getPairs has been called. The best way to derive this by hand would be to print some pair examples.    
    #By finding the corresponding pairs of (a,x) we are ablee to skip over some calculations that needed to be done.
    def getPairs(self, INIC, nk):
        pairs = []
        if DEBUG:
            print("INIC = {}, nk = {}, self.xval = {}".format(INIC, nk, self.xval))
        #We iterate over a_keys only within INIC + nk. This can be seen from the algorithm (or from printing out a few examples).
        for a in self.akeys:
            if (a[1] >= INIC+nk):
                break
            pairs.append( (a, (0,(a[1]%nk) + self.xval)))
        self.xval = self.xval +  nk
        return pairs

    #Contract is defined in the paper. This allows for memory saving. Going from (p^2)^n to np^2.
    #Getting the elements from the A and X matrix is odd. The DOK has its keys in the value of (0, location). So if we want to get the location of an A value you have to do A_key[1] and similarily if you want to get the location for an X value you need to do X_key[1]. In the getPairs this structures is held by returning a pair of value (a_key, (0,x_element)). 
    def contract_sparse(self, nk, mk, ki, A, xkeys):
        if DEBUG:
            print("A = {}".format(A))
            print("X = {}".format(self.X))
            print("nk = {}".format(nk))
            print("mk = {}".format(mk))
            print("xkeys = {}".format(xkeys))
        time_A = 0.0
        ktemp = 0
        inic = ki*(nk*nk)
        if DEBUG:
            print("nk = {}, mk = {}, ki = {}".format(nk, mk, ki))
        for i in range(nk): # dim of matrix k
            self.xval = 0
            pairs = []
            while(self.xval < self.X.shape[0]): #Iterate over all of x
                pairs = self.getPairs(inic, nk)
                if DEBUG:
                    print("[DEBUG] Pairs = {}".format(pairs))
                pair_sum = 0.0
                counter = 0#Used to count how many calculations were done to benchmark.
                if TIMING_ANALYSIS:
                    time_start_foo = time.time()
                for pair in pairs: # N / nk
                    pair_sum += A[pair[0][1]] * self.X[pair[1][1]]
                    self.counter += 1
                if TIMING_ANALYSIS:
                    time_A += time.time()-time_start_foo
                self.Y[ktemp] = pair_sum
                ktemp += 1

            #Remove all akeys that were used. 
            for pair in pairs:
                self.akeys.remove(pair[0])
            inic += nk
        np.copyto(self.X, self.Y)

        if TIMING_ANALYSIS:
            print("Time foo = {}".format(time_A))

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
                    sum = sum + (self.flat_A[I] * self.X[J])
                    if DEBUG:
                        print ("I = {}, J = {}".format(I,J))
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

    #Given a vector X, computes the dot product of A.X (A is given when the object initializes). This function takes the X given and converts it to a DOK matrix in order to get its' keys. The DOK matrix representation is never used because X tends to not be sparse and takes longer to access individual elements when compared to the regular numpy matrix. This function then runs the algorithm as given in the paper.   
    def dot_sparse(self, X):
        #self.printProperties()
        X = X.astype(float)
        if self.As == []:
            print("[Error] No A given")
        #Need to save value of X in class because it is used to store iterative calculations
        self.X = X
        #Create X as sparse matrix. It should be noted that we don't really use the sparse matrix for its sparseness (because it really isn't sparse), but its used to get the dictionary keys that have a nonzero element. These keys are used in the algorithm to solve the kronnecker product. 
        X_csr = scipy.sparse.csr_matrix(X) #For some reason I have to use this to reshape the DOK matrix
        X_dok = scipy.sparse.dok_matrix(X.reshape(X_csr.shape))

        #Get X keys
        x_keys = list(X_dok.keys())
      #  x_keys.sort(key=itemgetter(1))
        x_keys = sorted(x_keys, key=itemgetter(1))

        k = self.nmat
        nk = self.n[k-1]
        mk = self.N/nk
        for ki in range(k):
            if DEBUG:
                print("IN CONTRACTION ",ki)
                print("mk: ", mk)
            mk = self.N/self.n[k-1-ki]
            self.contract_sparse(nk, mk, ki,self.flat_As , x_keys)
        if DEBUG:
            print("Total operations = {}".format(self.counter))

        self.updateAkeys()
        if DEBUG:
            print("________________RESULTS___________________")
            print("[DEBUG] Y = {}, sum = {}".format(self.Y, np.sum(self.Y)))
        return self.Y

    def dot(self, x):
        if(self.sparse):
            self.dot_sparse(x)
        else:
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
            if DEBUG:
                print("________________RESULTS___________________")
                print("[DEBUG] Y = {}, sum = {}".format(self.Y, np.sum(self.Y)))

        return self.Y

# Example code
# ------------

if __name__ == '__main__':

    n = 4#number of factors
    p = 4 # dimension of factor
    r_As = [ortho_group.rvs(dim=p) for i in range(n)]
		#Make first and second row the same so that way it becomes a non-invertible matrix
    for A in r_As:
        A[1,:] = A[0,:]
    As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
    y = np.random.rand(p**n)

    big_A = reduce(np.kron, As)
    big_x = big_A.dot(y)
    print("[test_kron_inv - testRandom_pInv] full calc: ",big_x)

    kp = KronProd(As, False)
    x = kp.dot(y)
    print("[test_kron_inv - testRandom_pInv] efficient calc: ", x)

    print(np.allclose(x,big_x))



