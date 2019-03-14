#In this experiment I will show the time comparison to run y=Ax
import time
import numpy as np
from scipy.stats import ortho_group
from operator import mul
from functools import reduce
from src.kronprod import KronProd
from src.kronprod_sparse import KronProdSparse
import math
#
# for y=A.Tx
def verifyCorrectness(l1, l2, l3):
    flag = 1
    #Check all lists are same length
    if(len(l1) != len(l2) != len(l3)):
        return 0
    #Check all lists are about the same
    for i in range(len(l1)):
        if(math.isclose(l1[i], l2[i]) and math.isclose(l2[i], l3[i])):
            pass
        else:
            print(l1[i],l2[i],l3[i])
            flag = 0
            break
    return flag


def transposeTest(A,x):
    foo = time.time()
    big_A = reduce(np.kron, A)
    big_A = big_A.T
    big_y1 = np.matmul(big_A, x)
    time_full = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(A[i].T)
    big_A = reduce(np.kron, newA)
    big_y2 = np.matmul(big_A, x)
    time_kron = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(A[i].T)
    kp1 = KronProd(list(reversed(newA)))
    Y1 = kp1.dot(x)
    time_kron_dyn = time.time() - foo

    if(verifyCorrectness(big_y1, big_y2, Y1)):
        print("Transpose Test Passed")
    else:
        print("Transpose Test Failed!")

    return time_full, time_kron, time_kron_dyn

# for y=A^-1x
def inverseTest(A,x):
    foo = time.time()
    big_A = reduce(np.kron, A)
    big_A = np.linalg.inv(big_A)
    big_y1 = np.matmul(big_A, x)
    time_full = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.linalg.inv(A[i]))
    big_A = reduce(np.kron, newA)
    big_y2 = np.matmul(big_A, x)
    time_kron = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.linalg.inv(A[i]))
    kp1 = KronProd(list(reversed(newA)))
    Y1 = kp1.dot(x)
    time_kron_dyn = time.time() - foo

    if(verifyCorrectness(big_y1, big_y2, Y1)):
        print("Inverse Test Passed")
    else:
        print("Inverse Test Failed!")

    return time_full, time_kron, time_kron_dyn
#
# for y=Ax
def dotTest(A,x):
    foo = time.time()
    big_A = reduce(np.kron, A)
    big_y1 = np.matmul(big_A, x)
    time_full = time.time() - foo


    foo = time.time()
    kp1 = KronProd(list(reversed(A)))
    Y1 = kp1.dot(x)
    time_kron_dyn = time.time() - foo

    if(verifyCorrectness(big_y1, big_y1, Y1)):
        print("Dot Test Passed")
    else:
        print("Dot Test Failed!")

    return time_full, time_kron_dyn
#
# for Complex conjugate
def complexTest(A,x):
    foo = time.time()
    big_A = reduce(np.kron, A)
    big_A = np.conj(big_A)
    big_y1 = np.matmul(big_A, x)
    time_full = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.conj(A[i]))
    big_A = reduce(np.kron, newA)
    big_y2 = np.matmul(big_A, x)
    time_kron = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.conj(A[i]))
    kp1 = KronProd(list(reversed(newA)))
    Y1 = kp1.dot(x)
    time_kron_dyn = time.time() - foo

    if(verifyCorrectness(big_y1, big_y2, Y1)):
        print("Conj Test Passed")
    else:
        print("Conj Test Failed!")

    return time_full, time_kron, time_kron_dyn
#
# for psuedoInv
def psuedoTest(A,x):
    foo = time.time()
    big_A = reduce(np.kron, A)
    big_A = np.linalg.pinv(big_A)
    big_y1 = np.matmul(big_A, x)
    time_full = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.linalg.pinv(A[i]))
    big_A = reduce(np.kron, newA)
    big_y2 = np.matmul(big_A, x)
    time_kron = time.time() - foo

    foo = time.time()
    newA = []
    for i in range(len(A)):
        newA.append(np.linalg.pinv(A[i]))
    kp1 = KronProd(list(reversed(newA)))
    Y1 = kp1.dot(x)
    time_kron_dyn = time.time() - foo

    if(verifyCorrectness(big_y1, big_y2, Y1)):
        print("PInverse Test Passed")
    else:
        print("PInverse Test Failed!")

    return time_full, time_kron, time_kron_dyn
#
#

if __name__ == "__main__":
    #Generate matrix and vector
    A1 = ortho_group.rvs(dim=100)
    A2 = ortho_group.rvs(dim=100)
    x = np.random.rand(100**2)
    print("Transpose Test")
    print(transposeTest([A1,A2],x))
    print("Inverse Test")
    print(inverseTest([A1, A2], x))
    print("PInverse Test")
    print(psuedoTest([A1, A2], x))
    print("Complex Test")
    print(complexTest([A1,A2],x))
    print("Dot Test")
    print(dotTest([A1,A2],x))
