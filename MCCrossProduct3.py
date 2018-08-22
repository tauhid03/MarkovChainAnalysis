import numpy as np 
import math

def main():


    PQ = np.matrix([[ 0.06,0.24,0.14,0.56], [0.12,0.18,0.28,0.42],[ 0.02,0.08,0.18,0.72],[0.04,0.06,0.36,0.54]])
    R = np.matrix([[0.15, 0.85], [0.25, 0.75]])

    lPQ = np.zeros((len(PQ),len(PQ)))
    lR = np.zeros((len(R),len(R)))
    
    for i in range(len(PQ)):
        for j in range(len(PQ)):
            lPQ[i][j] = math.log(PQ.item((i, j)),2)
             
    for i in range(len(R)):
        for j in range(len(R)):
            lR[i][j] = math.log(R.item((i, j)),2)
    
    
    print 'Product of the first two Markov chains'
    print PQ
    print 'Third Markov chain'
    print R
    lPQR=np.zeros((len(PQ)*len(R),len(PQ)*len(R)))
    
    n=len(lPQ)
    for k in range(len(lR)):
        for l in range(len(lR)):
            for i in range (len(lPQ)): 
                for j in range (len(lPQ)):
                    if lPQ[i][j] != 0 and lR[k][l] != 0:
                        #lPQR[i+n*k][j+n*l]= lPQ[i][j] + lR[k][l]
                        lPQR[i+n*k][j+n*l]= math.pow(2,(lPQ[i][j] + lR[k][l]))


    print 'Cross product of three Markov chains'                
    print lPQR
         
    for i in range(len(lPQR)):
        sum=0
        for j in range(len(lPQR[i])):
            sum+=lPQR[i][j]

        print 'row', (i+1), 'adds up to', sum

   
if __name__ == '__main__': main()




                      
