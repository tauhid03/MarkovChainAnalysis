import numpy as np 


def main():
    P = np.matrix([[0.2, 0.8], [0.4, 0.6]])
    Q = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    R = np.matrix([[0.15, 0.85], [0.25, 0.75]])
    S = np.matrix([[0.1, 0.9], [0.05, 0.95]])
    print 'First Markov chain'
    print P
    print 'Second Markov chain'
    print Q
    print 'Third Markov chain'
    print R
    print 'Fourth Markov chain'
    print S

    PQRS=np.zeros((len(P)*len(Q)*len(R)*len(S),len(P)*len(Q)*len(R)*len(S)))
    
    
    n=len(P)
    for s in range(len(S)):
        for t in range(len(S)):
            for q in range(len(R)):
                for r in range(len(R)):
                    for k in range(len(Q)):
                        for l in range(len(Q)):
                            for i in range (len(P)): 
                                for j in range (len(P)):
                                    PQRS[i+n*k+(n*n)*q+(n*n*n)*s][j+n*l+(n*n)*r+(n*n*n)*t]=P.item((i, j))* Q.item((k, l))*R.item((q, r))*S.item((s, t)) 


    print 'Cross product of four Markov chains'                
    print PQRS

    #for i in range(len(PQRS)):
    #    sum=0
    #    for j in range(len(PQRS[i])):
    #        sum+=PQRS[i][j]
    #    print 'row', (i+1), 'adds up to', sum

                      

if __name__ == '__main__': main()
