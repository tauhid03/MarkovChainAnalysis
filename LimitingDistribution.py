import numpy as np 
from pylab import *
from discreteMarkovChain import markovChain

def calculateSteadyStateDistribution(matrix):
    """Returns the Steady State Distribution, or Fixed Vector, or a transition matrix"""
    matrix = matrix.T
    vals = eig(matrix)
    vector = None
    i = -1
    # Find the eigenvector corresponding to the eigenvalue '1'
    for x in range(len(vals[0])):
        if 1.0 - vals[0][x] < .001:
            i = x
            vector = vals[1][:,x]
    # Normalize it so that the vector's components sum to 1
    return vector / sum(vector)

def main():
    #Aperiodic, irreducible and Double stochastic Markov Chain (e.g., random walk on a cyclic graph)
    P= np.matrix([[0, 0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2, 0]])
    mc=markovChain(P)
    #Calculate the limiting distribution or the steady state distribution of a Markov chain
    mc.computePi('linear') #We can also use 'power', 'krylov' or 'eigen'
    print mc.pi      
    print calculateSteadyStateDistribution(P)   

if __name__ == '__main__': main()

