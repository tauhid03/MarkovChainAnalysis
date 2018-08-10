import numpy as np 


def main():
    P = np.matrix([[0.2, 0.8], [0.4, 0.6]])
    Q = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    print 'First Markov chain'
    print P
    print 'Second Markov chain'
    print Q
    PQ=np.zeros((len(P)*len(Q),len(P)*len(Q)))
    
    n=len(Q)
    for l in range(len(Q)):
        for k in range(len(Q)):
            for j in range (len(P)): 
                for i in range (len(P)):
                    PQ[k+n*l][i+n*j]=P.item((k, i))* Q.item((l, j)) 


    print 'Cross product of two Markov chains'                
    print PQ 

                      

if __name__ == '__main__': main()
