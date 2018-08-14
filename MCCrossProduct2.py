import numpy as np 


def main():
    P = np.matrix([[0.2, 0.8], [0.4, 0.6]])
    Q = np.matrix([[0.3, 0.7], [0.1, 0.9]])
    print 'First Markov chain'
    print P
    print 'Second Markov chain'
    print Q
    PQ=np.zeros((len(P)*len(Q),len(P)*len(Q)))
    
    n=len(P)
    for k in range(len(Q)):
        for l in range(len(Q)):
            for i in range (len(P)): 
                for j in range (len(P)):
                    PQ[i+n*k][j+n*l]=P.item((i, j))* Q.item((k, l)) 


    print 'Cross product of two Markov chains'                
    print PQ

    #for i in range(len(PQ)):
    #    sum=0
    #    for j in range(len(PQ[i])):
    #        sum+=PQ[i][j]
    #    print 'row', (i+1), 'adds up to', sum
                      

if __name__ == '__main__': main()
