#All of these operations can be found in Kathrin Schacke's "On the Kronecker Product" written on August 1, 2013. This paper provides a review of the properties of the kronecker product, as well its history and applications. This code will implement many of the properties given in Section 2 : The Kronecker Product
#All comments that describe the functions below are directly quoted from Kathrin Schacke's paper. 
#For all examples, c is a constant, and A and B are matrices
#I want these functions to be useful for when we have an arbitrarily sized list of matrices, and apply all of these operations to this list.
import numpy as np
from functools import reduce

def verifyInput(A):
    if type(A) == np.matrix:
        if len(A) == 0:
            assert "Can't apply operation, list of matrices is empty."

#Apply a constant to a matrix. When applying a constant to a kron product it doesn't matter where its applied, as long as its applied once. 
#(cA) kron B == A kron (cB) == c(A kron B)
def applyConstant(A,constant):
    verifyInput(A)
    if(type(A) == list):
        return A[0] * constant
    else:
        return A * constant
    
#Taking the transpose of the kron prod is the same thing as taking the kron prod after
#(A kron B).T = A.T kron B.T
def transpose(A):
    verifyInput(A)
    if type(A) == list:
        A_T = []
        for a in A:
            A_T.append(a.T)
        return A_T
    else:
        return A.T

#Taking the complex conjugate before carrying out the Kron product yields the same results as doing so afterwards
#(A kron B)* = A* kron B*
def complexConjugate(A):
    verifyInput(A)
    if type(A) == list:
        A_Conj = []
        for a in A:
            A_Conj.append(np.conj(a))
        return A_Conj
    else:
        return np.conj(A)
#The trace of the kron product of two matrices is the product of the traces of the matrices
#trace(A kron B) = trace(A)trace(B)
def trace(A):
    verifyInput(A)
    if type(A) == list:
        return reduce(lambda x, y: np.trace(x)*np.trace(y), A)
    else:
        return np.trace(A)

#det(A kron B) = (det(A))^n * (det(B))^m where A is mxm and B is nxn
def det(A):
    verifyInput(A)
    if type(A) == list:
        if len(A) == 1:
            return np.linalg.det(A[0])
        power1 = None
        if len(A) == 2:
            power1 = A[1].shape[0]
        else:
            power1 = reduce(lambda x, y: x.shape[0]*y.shape[0], A[1:]) 
        power2 = A[0].shape[0]
        return det(A[0])**(power1) * det(A[1:])**(power2)
    else:
        return np.linalg.det(A)


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def invert(A,allow_pinvert=True):
    verifyInput(A)
    if type(A) == list:
        A_inv = []
        for a in A:
            if(is_invertible(a)):
                A_inv.append(np.linalg.inv(a))
            elif(allow_pinvert == True):
                print("[Warning] Taking pinvert since matrix is not invertible")
                A_inv.append(pinvert(a))
            else:
                assert "Non invertible matrix given, and not allowed to take pinvert"
        return A_inv
    else:
        if(is_invertible(A)):
            return np.linalg.inv(A)
        elif(allow_pinvert == True):
            print("[Warning] Taking pinvert since matrix is not invertible")
            return pinvert(A)
        else:
            assert "Non invertible matrix given, and not allowed to take pinvert"

def pinvert(A):
    verifyInput(A)
    if type(A) == list:
        A_pinv = []
        for a in A:
            A_pinv.append(np.linalg.pinv(a))
        return A_pinv
    else:
        return np.linalg.pinv(A)



