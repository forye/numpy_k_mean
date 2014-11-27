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
    '''
    gets:
    Samples:PxN list/Array/Matrix type,
    K: number of groups
    Iter: maximum desired iterations
    
    returns:
    guess: 1xN int array classifying each sample to group
    Centers: NxK array representing the centroids of each group
    '''
    # initialize the samples to an warray
    Samples=np.array(Samples)    
    #initialize N as the number of attributes 
    _ ,N = np.shape(Samples)
    #create an empty array for the centroids
    Centers = np.empty([K,N])
    
    #init groups in randomly chosen samples 
    for k in range(K):
        Centers[k,:] = Samples[np.floor( N* rand(1)[0,0] ).astype(int) ,: ] 
    
    #run and return the k-mean algorithm
    return start_k_mean(Samples, Centers, Iter)


def start_k_mean(Samples,Centers,Iter):
    '''
    gets:
    Samples:PxN list/Array/Matrix type,
    Centers:randomly initialized centroids
    Iter: maximum desired iterations
    
    returns:
    guess: 1xN int array classifying each sample to group
    Centers: NxK array representing the centroids of each group
    
    '''
    # initialize P in Smaples number, N as attributes number and K in groups number
    P ,N = np.shape( Samples)
    K ,_ = np.shape( Centers)
    
    #reshape Samples and Centers to 3d ndarrays, for quicker computation if the future
    Samps = Samples.reshape(P,N,1)
    Cents = Centers.reshape(1,N,K)
    
    #fits the samples to initial centroids
    guess = fit_groups(Samps,Cents)  
    
    #itterate 'Iter' times      
    for i in xrange(Iter-1):            
        #creates a counter, 1xK counts the amount of samples for each group
        counter = count_samples_in_group(guess,K)
        
        #if a group is empty, initialize the groups centroid to the most far values from the mean of the centroids
        Cents = fix_empty_groups(counter,Samps,Cents)
       
        #remember the initial centroids, for stopping condition
        Cents_old= np.array(Cents)
        
        # calculate the new centroids according to the old centroids guess 
        Cents = new_centroids(guess,Samples,Cents)
        
        # fit again according to the new centroids
        guess = fit_groups(Samps,Cents)    
        
        # if the centroids havent changed in the last itteration, finish process
        if np.all(Cents_old == Cents):
#             print "success!"
            break
        
        #print status before finishing
#     print "\nFinished after " +str(t) +" itterations"
#     print "\n FINAL guess: " +str(guess)

#     print "number of itterarions: " + str(i)
    #retrun results (cents is still a 3d array!)
    return guess,Cents[0]

        

def fit_groups(Samps,Cents):
    '''
    calculate the centroids from the samples to the centroids using numpy,
    array broadcasting and linear algebra- should be very quick and efficient
    
    taking the index of each group representing the minimal normed(l2) 
    difference between the samples and the centroids
    
    Samples: 1xPxN Array type
    Centers: NxKx1 array representing the centroids of each group

    retruns guesses list: 1xP array int array classifying each sample to group
    '''
    return np.argmin( np.linalg.norm(Samps-Cents, axis=1) , axis =1)
    

def count_samples_in_group(guess,K):
    '''
    counts the numbers of samples classified to each group
     guesses list: 1xP array int array classifying each sample to group
    K- number of clusters
    returns a list (why not 1xK array?) containes number of samples for each group
    '''
    #for each group, sum all the guesses 
    return [ np.sum( guess == k) for k in range(K)]
    
    
    #create a list for result. each group defined by its index
#     result =[0]*K

#     for k in range(K):
#         result[k] = np.sum( guess == k)
#     return result

# '''
# from
# np.mean(np.array([]))
# got
# C:\Python27\lib\site-packages\numpy\core\_methods.py:59: RuntimeWarning: Mean of empty slice.
#   warnings.warn("Mean of empty slice.", RuntimeWarning)
# nan
# C:\Python27\lib\site-packages\numpy\core\_methods.py:71: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
# 
# Idan
# '''



def fix_empty_groups(counter,Samps,Cents):
    '''
    for each empty group, initialize its centroid to the most far sample from the other
    centroids mean 
    
    gets:
    counter: a list (why not 1xK array?) containes number of samples for each group
    Samples: 1xPxN Array type
    Cents: NxKx1 array representing the centroids of each group    
    retruns:
    Cents: NxKx1 array representing the NEW centroids, after updating the empty groups centroids    
    '''    
    for k,c in enumerate(counter):
        if not c:        
            Cents[0,:,k] = Samps[  np.argmax(  np.linalg.norm(Samps[:,:,0] - 
                                          np.nanmean(Cents,axis=2), axis=1)  ), : , 0]
#             print "new centroid artificially generated Cent: " + str(Cents[0,:,k])

    return Cents


def fix_empty_group(counter,Samps,Cents):
    '''
    for ONLY 1 GROUP! 
    initialize its centroid to the most far sample from the other
    centroids mean 
    
    gets:
    counter: a list (why not 1xK array?) containes number of samples for each group
    Samples: 1xPxN Array type
    Cents: NxKx1 array representing the centroids of each group    
    retruns:
    Cents: NxKx1 array representing the NEW centroids, after updating the empty groups centroids    
    '''    
    return Samps[ ( np.argmax(np.linalg.norm(Samps[:,:,0] - 
                                np.nanmean(Cents,axis=2), axis=1))) , : , 0]
  

def new_centroids(guess,Samps,Cents):
    '''
    updates the new centroids to the (new) groups classification guesses mean
    gets:
    counter: a list (why not 1xK array?) containes number of samples for each group
    Samples: 1xPxN Array type
    Cents: NxKx1 array representing the centroids of each group    
    retruns:
    Cents: NxKx1 array representing the NEW centroids, after updating the empty groups centroids
    
    '''
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


def create_test_set2(P,N=2,sig1=0.5, sig2=0.5,q=0.5):#create_test_set2(P,N=2,sig1 =1,sig2 =1,q=0.5):
    qP = q*P
        
    test_set = randn(P,N)
    if N==2:    
        test_set[  :qP] =sig1* test_set[:qP]+[0,3]
        test_set[qP:P] =sig2 *test_set[qP:P]+[3,0]        
    return test_set



def sort_cents(cents):
    '''
    sorts a numpy array according to the 1sth index
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
    sorts centroids group indexes
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
    returns the abs norm of of the mu's from the centroids
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
    '''
    returns the sum samples classified to each group
    '''
    return sum( np.sum(np.array([res1 == k for k in range( K )]),axis=1))
    
 