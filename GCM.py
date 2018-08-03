from math import *
import numpy as np
import networkx as nx
import io
from tarjan import tarjan
from discreteMarkovChain import markovChain
from scipy import *
from pylab import *
import csv


def tc(g):
    """ Given a graph @g, returns the transitive closure of @g """
    ret = {}
    for scc in tarjan(g):
            ws = set()
            ews = set()
            #print scc
            for v in scc:
                    ws.update(g[v])
            for w in ws:
                    assert w in ret or w in scc
                    ews.add(w)
                    ews.update(ret.get(w,()))
            if len(scc) > 1:
                    ews.update(scc)
            ews = tuple(ews)
        #ews=list(ews)
            for v in scc:
                    ret[v] = ews                
    return ret
    
        
def main():
    print ("Attractors and Basins of Attractions")
    graph={}
    PGlist=[[]] 
    PGR=0
    try:
        pgfile = open('PersistentGroup.txt','w')
    except IOError:
        print ('Cannot open PersistentGroup.txt')
        quit()    
     
    try:
        tgfile = open('TransientGroup.txt','w')
    except IOError:
        print ('Cannot open TransientGroup.txt')
        quit()   
          


    P = []
    # read the transition matrix from the file and create the graph
    with open('transitionmatrix.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            P.append(row)
            row=list(map(float, row))
            indices = np.nonzero(row)[0]
            graph.update({len(P)-1: indices})
            
    # compute the strongly connected component of the graph        
    SCC=tarjan(graph)
    graphnodes=graph.keys()
    print 'Total no. of strongly connected component of the graph is ',len(SCC)
    # compute the transitive closure of the graph        
    TC=tc(graph)

    for i  in range(0, len(SCC)):
        sum=0
        tempSCC=sorted(SCC[i])
        for j in range(0,len(SCC[i])):
            tempTC=np.array(sorted(TC[SCC[i][j]]))
            if np.array_equiv(tempSCC, tempTC):
                sum=sum+1
        if(sum==len(SCC[i])):
            PGlist.append([])
            #print ((i+1),'is a persistent group')
            PGlist[PGR]=np.array(sorted(SCC[i]))
            #print  PGlist[PGR]
            pgfile.write(str(i+1)+' is a persistant group with length '+str(len(SCC[i]))+'\n\n')
            pgfile.write(str(PGlist[PGR])+'\n\n')
            PGR=PGR+1
        else:
            #print ((i+1),'is a transient group')
            #print np.array(sorted(SCC[i]))
            tg=np.array(sorted(SCC[i]))
            tgfile.write(str(i+1)+' is a transient group with length '+str(len(SCC[i]))+'\n\n')
            tgfile.write(str(tg)+'\n\n')
     
    ## find transient cells, single or multiple domiciles
    persistentcelllist=[] 
    for i in range(PGR):
        for  j in range(len(PGlist[i])):
            persistentcelllist.append(PGlist[i][j])  
        
    transientcelllist=list(set(graphnodes) - set(persistentcelllist))

    transientcell={}
    
    ##single and multiple-domicile determination
    for i in range (0,PGR):
        tempgrlist= PGlist[i]
        for l in range(0,len(transientcelllist)):
            tempTC=list(TC[transientcelllist[l]])
            noofpaths=0
            for k in range(0,len(tempTC)):
                if tempTC[k] in tempgrlist:
                    noofpaths=noofpaths+1
            if noofpaths>0:
                transientcell.setdefault(transientcelllist[l], [])
                transientcell[transientcelllist[l]].append(i+1) 

    print 'Total no. of persistent groups', PGR
    print 'Transient cells with their domiciles',transientcell
    tgfile.write('Transient cells with their domiciles'+'\n\n')
    tgfile.write(str(transientcell)+'\n\n')
          
        
         
if __name__ == '__main__': main()
  
