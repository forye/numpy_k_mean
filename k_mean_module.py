'''
Created on Nov 23, 2014

@author: Idan
'''
    
'''
what is more consuming?
reshaping Centers and Samples?

or approacing the 3d matrics via Cents[0,:,k] or Samples[guess == k,:,0]
'''    


if __name__ == '__main__':
    pass


import numpy as np
from numpy.matlib import randn, rand

import matplotlib.pyplot as plt

#from sklearn.cluster.k_means_ import k_means





def my_k_means(Samples, K, Iter):
    
    Samples=np.array(Samples)    
    _ ,N = np.shape(Samples)
    
    Centers = np.empty([K,N])
    
    #init groups 
    for k in range(K):
        Centers[k,:] = Samples[np.floor( N* rand(1)[0,0] ).astype(int) ,: ] 
    # K ,_ = np.shape( Centers)
    
    guess,Centers = cluster1(Samples, Centers, Iter)
    
    return guess,Centers


def cluster1(Samples,Centers,Iter):
    P ,N = np.shape( Samples)
    K ,_ = np.shape( Centers)
    
    #Samps = np.array(np.transpose(Samples)).reshape(1,P,N)
    Samps = Samples.reshape(P,N,1)
    Cents = Centers.reshape(1,N,K)
    
    guess = fit_groups(Samps,Cents)  
    
          
    for _ in xrange(Iter-1):            
        '''
        change Cent to Global? No,didnt prove itself
        '''
        counter = count_samples_in_group(guess,K)
        
        Cents = fix_empty_groups(counter,Samps,Cents)
        Cents_old= np.array(Cents)
        Cents = new_centroids(guess,Samples,Cents)
         
        guess = fit_groups(Samps,Cents)    
        
        if np.all(Cents_old == Cents):
#             print "success!"
            break
#     print "\nFinished after " +str(t) +" itterations"
#     print "\n FINAL guess: " +str(guess)
    return guess,Cents[0]

        

def fit_groups(Samps,Cents):
    return np.argmin( np.linalg.norm(Samps-Cents, axis=1) , axis =1)
    #return np.argmin( np.linalg.norm(Samples.reshape(P,N,1)-Centers.reshape(1,N,K), axis=1) , axis =1)


def count_samples_in_group(guess,K):
    result =[0]*K
    for k in range(K):
        result[k] = np.sum( guess == k)
    return result


def fix_empty_groups(counter,Samps,Cents):
    
#     acc =0
    for k,c in enumerate(counter):
        
        if not c:        
            Cents[0,:,k] = Samps[ ( np.argmax(np.linalg.norm(Samps[:,:,0] - 
                                np.nanmean(Cents,axis=2), axis=1))) , : , 0]
#             print "new centroid artificially generated Cent: " + str(Cents[0,:,k])

    return Cents


def fix_empty_group(counter,Samps,Cents):
    return Samps[ ( np.argmax(np.linalg.norm(Samps[:,:,0] - 
                                np.nanmean(Cents,axis=2), axis=1))) , : , 0]
  

def new_centroids(guess,Samps,Cents):
    K = Cents.shape[2]
    for k in xrange(K):
        #Cents[0,:,k] = np.transpose( np.mean(  Samps[guess == k,:,0], axis = 0 ) )
        Cents[0,:,k] = np.nanmean( Samps[guess == k], axis = 0 )  #changed to nanmean to no mean on empty slice
        if np.any(np.isnan( Cents[0,:,k])):
            Cents[0,:,k]= Samps[0,:]
    return Cents
  
  
def plot_results(x1,res1,centroids):
    colors = ['red','blue','green','magenta','cyan','black']
    col=[0]*len(res1)
    for p in range(len(res1)):
        col[p] = colors[res1[p]%6]
    plt.scatter(x1[:,0], x1[:,1], c=col )#facecolors='none', edgecolors='r')    
    plt.scatter(centroids[0], centroids[1], s=180, facecolors=colors, edgecolors='white')    
    plt.show()
    

def create_test_set1(P,N=2,mu =1,sig =1):
    qP = P/4
        
    test_set = randn(P,N)
    if N==2:    
        test_set[  :qP] =sig* test_set[:qP]+[2,3]
        test_set[qP:2*qP] =sig *test_set[qP:2*qP]+[3,2]
        test_set[2*qP:3*qP] =2*sig *test_set[2*qP:3*qP]+[4,1]
        test_set[3*qP:4*qP] =2*sig *test_set[3*qP:4*qP]+[-1,0]
    else:
        test_set =np.array(  np.sqrt(sig) *randn(P,N) ) + mu *np.array( range(N) )
    return test_set

def sort_cents(cents):
    '''
    sorts a numpy array accoding to the 1sth index
    '''
    A=np.array(cents)
    n = A.shape[0]
    for k in xrange(n):
        minimal = k
        for j in xrange(k + 1, n):
            if (A[minimal,0] > A[j,0]):
                minimal = j
        temp1 = np.array( A[k,:] )        
        A[k,:] =A[minimal,:]
        A[minimal,:] = temp1   
    
    
    return A

def sort_groups_by_centroids(cents):
    '''
    sorts centroids
    '''
    A=np.transpose( np.array(cents) )
    n = A.shape[0]
    B=np.array( range(n) )
    for k in xrange(n):
        minimal = k        
        for j in xrange(k + 1, n):
            if (A[minimal,0] > A[j,0]):
                minimal = j        
        temp1 = np.array( A[k,:] )
        temp2 = B[k]
        A[k,:] =A[minimal,:]
        B[k] = B[minimal]
        A[minimal,:] = temp1
        B[minimal] = temp2
    return B


# def distance_from_means( centroids, mus= None ):
#     '''
#     returns the l2 norm of of the mu's from the centroids
#     '''
#     if mus==None:
#         mus = np.array( [ [2,3],[3,2],[4,1],[-1,0] ] )
#     else:
#         mus = np.array(mus)    
#     c = np.transpose(centroids)
#     return np.linalg.norm(sort_cents(mus)- sort_cents(c))


def mean_distance_from_means( centroids, mus= None ):
    '''
    returns the l2 norm of of the mu's from the centroids
    '''
    if mus==None:
        mus = np.array( [ [2,3],[3,2],[4,1],[-1,0] ] )
    else:
        mus = np.array(mus)
    
    c = np.array( np.transpose(centroids) )
    return np.mean(np.abs(sort_cents(mus)- sort_cents(c) ) )
#     return np.mean( np.mean(np.abs(sort_cents(mus)- sort_cents(c)) ))


def mean_varience_diff( centroids,X,res, sig, stds= None ):
    '''
    stds
    '''
    if stds==None:
        stds = np.array( [ [sig,sig],[sig,sig],[2*sig,2*sig],[2*sig,2*sig] ] )
    else:
        stds = np.array(stds)
    stds1=sort_cents(stds)
    group = sort_groups_by_centroids(centroids)
    measured = np.empty(stds.shape)  
    for k in range(len(stds)):
        measured[k] = np.abs( stds1[k] -  np.std(X[res==group[k]] , axis=0 ))
    return np.mean(measured)

#     return np.mean(np.abs(sort_cents(stds)- sort_cents(c) ) )
#     return np.mean( np.mean(np.abs(sort_cents(mus)- sort_cents(c)) ))


def recount_samples(res1,K): 
        return sum( np.sum(np.array([res1 == k for k in range( K )]),axis=1))
    
 