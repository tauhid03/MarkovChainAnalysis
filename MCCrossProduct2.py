import numpy as np
import math


def main():
    P = np.matrix([[0.2, 0.8], [0.4, 0.6]])
    Q = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    lP = np.zeros((len(P),len(P)))
    lQ = np.zeros((len(Q),len(Q)))
    
    for i in range(len(P)):
        for j in range(len(P)):
            lP[i][j] = math.log(P.item((i, j)),2)
            lQ[i][j] = math.log(Q.item((i, j)),2)
            
    print 'First Markov chain'
    print P
    print 'Second Markov chain'
    print Q
    lPQ=np.zeros((len(P)*len(Q),len(P)*len(Q)))
    n=len(lQ)
    for k in range(len(lQ)):
        for l in range(len(lQ)):
            for i in range (len(lP)): 
                for j in range (len(lP)):
                    if lP[i][j]!=0 and lQ[k][l] !=0:
                        #lPQ[i+n*k][j+n*l]= lP[i][j] + lQ[k][l]
                        lPQ[i+n*k][j+n*l]= math.pow(2,(lP[i][j] + lQ[k][l]))
    
    print 'Cross product of two Markov chains'                
    
    print lPQ

    for i in range(len(lPQ)):
        sum=0
        for j in range(len(lPQ[i])):
            sum+=lPQ[i][j]
            #print math.pow(2,lPQ[i][j]) 
        print 'row', (i+1), 'adds up to', sum

                      

if __name__ == '__main__': main()

