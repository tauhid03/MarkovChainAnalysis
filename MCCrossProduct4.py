import numpy as np 
import math

def main():
    PQR = np.matrix([[ 0.009,0.036,0.021,0.084,0.051,0.204,0.119,0.476],[0.018,0.027,0.042,0.063,0.102,0.153,0.238,0.357], [ 0.003,  0.012,  0.027,  0.108,  0.017,  0.068,  0.153,  0.612],[ 0.006,  0.009,  0.054,  0.081,  0.034,  0.051,  0.306,  0.459],[ 0.015,  0.06,   0.035,  0.14,   0.045,  0.18,   0.105,  0.42 ], [ 0.03,   0.045,  0.07,   0.105,  0.09,   0.135,  0.21,   0.315], [ 0.005,  0.02,   0.045,  0.18,   0.015,  0.06,   0.135,  0.54 ], [ 0.01,   0.015,  0.09,   0.135,  0.03,   0.045,  0.27,   0.405]] )

    S = np.matrix([[0.1, 0.9], [0.05, 0.95]])

    lPQR = np.zeros((len(PQR),len(PQR)))
    lS = np.zeros((len(S),len(S)))
    
    for i in range(len(PQR)):
        for j in range(len(PQR)):
            lPQR[i][j] = math.log(PQR.item((i, j)),2)
             
    for i in range(len(S)):
        for j in range(len(S)):
            lS[i][j] = math.log(S.item((i, j)),2)

    print 'Product of the first three Markov chains'
    print PQR
    print 'Fourth Markov chain'
    print S


    lPQRS=np.zeros((len(PQR)*len(S),len(PQR)*len(S)))
    
    n=len(lPQR)
    for k in range(len(lS)):
        for l in range(len(lS)):
            for i in range (len(lPQR)): 
                for j in range (len(lPQR)):
                    if lPQR[i][j] != 0 and lS[k][l] != 0:
                        #lPQRS[i+n*k][j+n*l]= lPQR[i][j] + lS[k][l]
                        lPQRS[i+n*k][j+n*l]= math.pow(2,(lPQR[i][j] + lS[k][l]))
                    

    print 'Cross product of four Markov chains'                
    print lPQRS
         
    for i in range(len(lPQRS)):
        sum=0
        for j in range(len(lPQRS[i])):
            sum+=lPQRS[i][j]

        print 'row', (i+1), 'adds up to', sum


                      

if __name__ == '__main__': main()

