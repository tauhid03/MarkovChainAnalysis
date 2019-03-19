
from scipy.stats import ortho_group
from pathlib import Path
try: 
    print(Path('/home/username').parent)
    from src.kronprod import KronProd
except Exception as e:
    from kronprod import KronProd
import numpy as np
from operator import mul
#from fail import As_fail
#from fail import Y_fail 
from functools import reduce

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


class KronProdInv(KronProd):
    def __init__(self, As):
        if len(As) == 0:
            assert "[KronProdInv] Empty list of matrix given to KronProdInv"
        print("size",len(As))
        As_inv = []
        inv_flag = 1
        for A in As:
            if(is_invertible(A) == False):
                inv_flag = 0
                print("[KronProdInv] A given matrix in not invertible, will use psuedo inverse")
                break
        #Invert the As if possible
        if(inv_flag):
            for A in As:
                As_inv.append(np.linalg.inv(A))
        else:
            for A in As:
                As_inv.append(np.linalg.pinv(A))
        #If it isn't use psuedoInverse
        super().__init__(As_inv)

def runTest():
    n = 4#number of factors
    p = 4 # dimension of factor
    r_As = [ortho_group.rvs(dim=p) for i in range(n)]
		#Make first and second row the same so that way it becomes a non-invertible matrix
    for A in r_As:
        A[1,:] = A[0,:]
    As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
    print("As = ", As)
    y = np.random.rand(p**n)
    print("Y = ", y)

    big_A = reduce(np.kron, As)
    big_A_inv = np.linalg.pinv(big_A)
    big_x = big_A_inv.dot(y)
 #   print("[test_kron_inv - testRandom_pInv] full calc: ",big_x)

    kp = KronProdInv(list(reversed(As)))
    x = kp.dot(y)
 #   print("[test_kron_inv - testRandom_pInv] efficient calc: ", x)
    print("All_close=",np.allclose(x,big_x))

    return(np.allclose(x,big_x))

def fail():
    n = 4#number of factors
    p = 4 # dimension of factor
    As = As_fail
    y = Y_fail
    As2 = []
    for a in As:
        As2.append(np.linalg.pinv(a))
        

    big_A = reduce(np.kron, As2)
    big_x = big_A.dot(y)
    print("[test_kron_inv - testRandom_pInv] full calc: ",big_x)

    kp = KronProdInv(list(reversed(As)))
    x = kp.dot(y)
    print("[test_kron_inv - testRandom_pInv] efficient calc: ", x)
    print("All_close=",np.allclose(x,big_x))



if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    fail()
#    for i in range(50000):
#        try:
#            results = runTest()
#        except Exception as e:
#            print(e)
#            results = True
#        if(results == False):
#            break
