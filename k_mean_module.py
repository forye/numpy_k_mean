'''
Created on Nov 23, 2014

@author: Idan
'''
#from sklearn.cluster.k_means_ import k_means

if __name__ == '__main__':
    pass

#from numpy import size, argmin, empty, repeat as rp

import numpy as np
from numpy.matlib import randn, rand

# import matplotlib.pyplot as plt

from  k_mean_module_draft import *



test_set = randn(80,2)
test_set[  :20] = 2* test_set[:20]+[2,3]
test_set[20:40] = 2* test_set[:20]+[3,2]
test_set[40:60] = 2* test_set[:20]+[4,1]
test_set[60:80] = 2* test_set[:20]+[-1,0]


x1= test_set
K1 =4
T1=50




i=0
S=50
while i<S:
    res1 = my_k_means(x1, K1, T1)
    i+=1
    print "returend " + str(i) + " times...."



'''    
    K-means clustering algorithm in Python 
    
    with the following inputs:
        a. Samples: (matrix of double, NxM) where N = samples, M=attributes
        b. K: number of classes
        c. P: number of random repetitions
    And the following outputs:    
        A. Idx: clustering of each sample into one of K clusters
        B. C: center of each cluster
        
    for consumer physics task4idan
    '''
    
'''
    1.create K groups- and indication array of size K containing the centroids, y (k x size(Samples)[1])
    1.5 - create a result vector- X, containing indexes,seized (Samples), create K counters sized "Samples"
    2.choose and assign y[k] (centroids- prediction center estimators) 
    3.do for P itterations:
     
    4.    for each Sample in Samples sized - size(Samples)[0] 
    5.        argmin(y-Samples[sample]) -> k
    6.        counter[k]+=1
    7.        x[sample]=k          
          #here sum(counter)= size( Samples)[0]
    8.    for each kth centroid in y  #update prediction centers
    9.        if counter(k) is empty:
    10.              y(k) = Samples(argmax(x[samples]-y[centroids]))
              else:
    11.              y[k] = mean(group - k)# - y[k] = sum(x where x[sample]==k) / counters(k)
    
    '''
    
    
    
    
    
    
    
    
    
    
    
    
#     def cluster1(Samples,Centers,guess):
#         #primal
#         P ,N = size( Samples)
#         K ,_ = size( Centers)
#         
#         Samps = Samples.reshape(P,N,1)
#         Cents = Centers.reshape(1,N,K)
#         guess = argmin( (Samps-Cents), axis = 2)
        
        #Cen =  numpy.repeat(Centers,K,axis = 1)# expand the centers to colums 
        #guess = argmin ( Samples  - Cen, axis = 0 ) # min the rowas

'''
        for p in xrange( P):
            #for n in xrange( N):
            guess[ p] = argmin(Samples[ p] - Centers)
            
        guess[ p] [p] = argmin( Samples[ p] - Centers)
        
        guess Px1 = argmin ( Samples PxN - Centers NxK )

       ??? 
                 
       #for p in xrange( P):
        #   guess[p]= argmin(Samples[p] - Centers, axis=1)
'''        
        
    