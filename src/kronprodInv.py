from src.kronprod import KronProd
import numpy as np
from operator import mul
from functools import reduce

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

class KronProdInv(KronProd):
    def __init__(self, As):
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


if __name__ == '__main__':
    n = 3 # number of factors
    p = 4 # dimension of factor
    A = np.array([[.2,.4,0, .4],
              [.4, .2, .4, 0],
              [0, .4, .2, .4],
              [.4, 0, .4, .2]])


    r_As = [A for i in range(n)]
  #  As = [m/m.sum(axis=1)[:,None] for m in r_As] # normalize each row
    x = np.random.rand(p**n)
    print("X= {}".format(x))

    kp = KronProdInv(list(reversed(r_As)))
    Y = kp.dot(x)
    print("Y = {}".format(Y))

    big_A = reduce(np.kron, r_As)
    big_y = np.linalg.solve(big_A, x)
    print("full calc: ",big_y)
    print(x.shape, big_A.shape, big_y.shape)

