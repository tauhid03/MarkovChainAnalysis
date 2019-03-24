#! /usr/bin/env python


#All of these operations can be found in Kathrin Schacke's "On the Kronecker Product" written on August 1, 2013. This paper provides a review of the properties of the kronecker product, as well its history and applications. This code will implement many of the properties given in Section 2 : The Kronecker Product
#All sections are split off by KRON# which is specified in the previous paper.
#In general there are different levels of restrictions to operations being done, but they are satisfied in general by randomly generated square matrices. Some properties require specific properties of the matrix to be held, but that will be done on a case by case basis.
import unittest
from scipy.stats import ortho_group
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import reduce
from src.operations import *

class TestOperations(unittest.TestCase):
#____________________HELPER FUNCTIONS________________
    # add global stuff here
    def setUp(self):
        self.p = 3

    def generateMatrices(self,n,p):
        if n < 0:
            assert "[TestOperations - generateMatrices] Can't generate negative amount of matrices"
        if p <= 0:
            assert "[TestOperations - generateMatrices] Can't generate matrix with p less than or equal to 0"
        
        r_As = [np.random.rand(p,p) for i in range(n)]
        As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
        x = np.random.rand(p**n)
        return As, x

#_________________KRON 1 ____________________
#Apply a constant to a matrix. When applying a constant to a kron product it doesn't matter where its applied, as long as its applied once. 
#(cA) kron B == A kron (cB) == c(A kron B)
    def test_applyConstant(self):
        As, x = self.generateMatrices(1,self.p)
        constant = 3.14159
        results_func = applyConstant(As[0], constant) 
        results_truth = As[0] * constant
        if(np.allclose(results_func,results_truth)):
            return
        else:
            self.fail("Results not equal")
    def test_KRON1(self):
        As, x = self.generateMatrices(2,self.p)
        constant = 3.14159
        results1 = reduce(np.kron, [applyConstant(As[0],constant), As[1]])
        results2 = reduce(np.kron, [As[0], applyConstant(As[1],constant)])
        results3 = applyConstant(reduce(np.kron, As), constant)
        if(np.allclose(results1, results2, results3)):
            return
        else:
            self.fail("Results not equal")
#________________KRON 2______________________
#Taking the transpose of the kron prod is the same thing as taking the kron prod after
#(A kron B).T = A.T kron B.T
    def test_KRON2(self):
        As, x = self.generateMatrices(2,2)
        results1 = transpose(reduce(np.kron, As))
        results2 = reduce(np.kron,transpose(As))
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#________________KRON 3____________________-
#Taking the complex conjugate before carrying out the Kron product yields the same results as doing so afterwards
#(A kron B)* = A* kron B*
    def test_KRON3(self):
        As, x = self.generateMatrices(2,self.p)
        results1 = complexConjugate(reduce(np.kron, As))
        results2 = reduce(np.kron,complexConjugate(As))
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#_______________KRON 4_________________________
#The kron product is associative
#(A kron B) kron C = A kron (B kron C)
    def test_KRON4(self):
        As, x = self.generateMatrices(3,self.p)
        results1 = reduce(np.kron,[reduce(np.kron,[As[0],As[1]]),As[2]])  
        results2 = reduce(np.kron,[As[0], reduce(np.kron,[As[1],As[2]])])  
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#______________KRON 5_________________________
#The kron product is right-distrutive
#(A+B) kron C = A kron C + B kron C
    def test_KRON5(self):
        [A,B,C], x = self.generateMatrices(3,self.p)
        results1 = reduce(np.kron,[(A+B),C])
        results2 = reduce(np.kron,[A,C]) + reduce(np.kron,[B,C])
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#______________KRON 6________________________
#The Kron product is left-distributive
#A kron (B+C) = A kron B + A kron C
    def test_KRON6(self):
        [A,B,C], x = self.generateMatrices(3,self.p)
        results1 = reduce(np.kron,[A,(B+C)])
        results2 = reduce(np.kron,[A,B]) + reduce(np.kron,[A,C])
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#______________KRON 7_______________________
#The product of two kron products yields another kron product
#(A kron B)(C kron D) = AC kron BD
    def test_KRON7(self):
        [A,B,C,D], x = self.generateMatrices(4,self.p)
        results1 = (reduce(np.kron,[A,B]))*(reduce(np.kron,[C,D]))
        results2 = reduce(np.kron,[(A*C),(B*D)])
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#_____________KRON 8 ____________________
#The trace of the kron product of two matrices is the product of the races of the matrices
#trace(A kron B) = trace(B kron A) = trace(A)*trace(B)
    def test_KRON8(self):
        [A,B], x = self.generateMatrices(2,self.p)
        results1 = trace(reduce(np.kron,[A,B]))
        results2 = trace([A,B])
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
#_____________KRON 9 _________________________
#det(A kron B) = det(B kron A) = (det(A))^n*(det(B))^m
    def test_KRON9(self):
        [A,B], x = self.generateMatrices(2,self.p)
        results1 = det(reduce(np.kron,[A,B]))
        results2 = det([A,B])
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")

#_____________KRON 10 ___________________________
#If A is mxm and B is nxn and are non singular then
#(A kron B) ^-1 = A^-1 kron B^-1
#This also applies for pinv
    def test_KRON10(self):
        A = ortho_group.rvs(dim=self.p)
        B = ortho_group.rvs(dim=self.p)
        results1 = invert(reduce(np.kron,[A,B])) 
        results2 = reduce(np.kron,invert([A,B]))
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
    #Test invert function when a singular matrix is given
    def test_KRON10_pinv(self):
        A = ortho_group.rvs(dim=self.p)
        B = ortho_group.rvs(dim=self.p)
        A[1,:] = A[0,:]
        B[1,:] = B[0,:]
        results1 = invert(reduce(np.kron,[A,B])) 
        results2 = reduce(np.kron,invert([A,B]))
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")
    #Test psued-invert function when a singular matrix is given
    def test_KRON10_pinv_function(self):
        A = ortho_group.rvs(dim=self.p)
        B = ortho_group.rvs(dim=self.p)
        A[1,:] = A[0,:]
        B[1,:] = B[0,:]
        results1 = pinvert(reduce(np.kron,[A,B])) 
        results2 = reduce(np.kron,pinvert([A,B]))
        if(np.allclose(results1,results2)):
            return
        else:
            self.fail("Results not equal")







if __name__ == '__main__':
    unittest.main()
