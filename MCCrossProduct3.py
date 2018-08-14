import numpy as np 


def main():
    P = np.matrix([[0.2, 0.8], [0.4, 0.6]])
    Q = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    R = np.matrix([[0.15, 0.85], [0.25, 0.75]])
    print 'First Markov chain'
    print P
    print 'Second Markov chain'
    print Q
    print 'Third Markov chain'
    print R
    
    PQR=np.zeros((len(P)*len(Q)*len(R),len(P)*len(Q)*len(R)))
    
    n=len(P)
    for q in range(len(R)):
        for r in range(len(R)):
            for k in range(len(Q)):
                for l in range(len(Q)):
                    for i in range (len(P)): 
                        for j in range (len(P)):
                            PQR[i+n*k+(n*n)*q][j+n*l+(n*n)*r]=P.item((i, j))* Q.item((k, l))*R.item((q, r)) 


    print 'Cross product of three Markov chains'                
    print PQR

    #for i in range(len(PQR)):
    #    sum=0
    #    for j in range(len(PQR[i])):
    #        sum+=PQR[i][j]
    #    print 'row', (i+1), 'adds up to', sum

if __name__ == '__main__': main()




                      
