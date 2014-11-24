# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.matlib import randn, rand

def my_k_means(Samples, K, T):
    
    Samples=np.array(Samples)    
    P ,N = np.shape(Samples)
    
    Centers = np.empty([K,N])
    
    #init groups 
    for k in range(K):
        Centers[k,:] = Samples[np.floor( N* rand(1)[0,0] ).astype(int) ,: ] 
    # K ,_ = np.shape( Centers)
    
    guess,Centers = cluster1(Samples, Centers, T)
    
    return guess,Centers


def cluster1(Samples,Centers,T):
    #primal
    P ,N = np.shape( Samples)
    K ,_ = np.shape( Centers)
    
    #Samps = np.array(np.transpose(Samples)).reshape(1,P,N)
    Samps = Samples.reshape(P,N,1)
    Cents = Centers.reshape(1,N,K)
    
    guess = fit_groups(Samps,Cents)  
    
    #print "\nguess: " +str(guess)
    #print "\nCents: "+ str(Cents)
          
    for _ in xrange(T-1):            
        '''
        change Cent to Global?
        '''
        counter = count_samples_in_group(guess,K)
        
        Cents = fix_empty_groups(counter,Samps,Cents)
        Cents_old= np.array(Cents)
        Cents = new_centroids(guess,Samples,Cents)
         
        guess = fit_groups(Samps,Cents)    
        
        #print Cents
       # print "\nguess: " +str(guess)
       # print "\nCents: "+ str(Cents)
       # print "\ncounter: " +str(counter)
        #print Cents
        
        if np.all(Cents_old == Cents):
            print "succes!"
            break
    print "\n FINAL guess: " +str(guess)
    return guess,Cents[0]
    
'''
what is more consuming?
reshaping Centers and Samples?

or approacing the 3d matrics via Cents[0,:,k] or Samples[guess == k,:,0]
'''    
        

def fit_groups(Samps,Cents):
    return np.argmin( np.linalg.norm(Samps-Cents, axis=1) , axis =1)
    #return np.argmin( np.linalg.norm(Samples.reshape(P,N,1)-Centers.reshape(1,N,K), axis=1) , axis =1)

def count_samples_in_group(guess,K):
    result =[0]*K
    for k in range(K):
        result[k] = np.sum( guess == k)
    return result

def fix_empty_groups(counter,Samps,Cents):
    
 
    acc =0
    for k,c in enumerate(counter):
        
        if not c:# or k==2:
            #print  "\n\ncentroid leads to empty a group : ("+str(k)+")  Centroids are: " +str(np.transpose(Cents)) +"\n"+ "conter is : "+str(counter) 

            #find the Sample (1xN) that is the most far from each Centroid- 
            #mu = np.mean(Cents,axis=2)
            #d = Samps[:,:,0]-mu
            #p_farest = np.argmax(np.linalg.norm(d, axis=1))
            #Cents[0,:,k] = Samps[p_farest,:,0] 
            Cents[0,:,k] = Samps[ ( np.argmax(np.linalg.norm(Samps[:,:,0]-np.mean(Cents,axis=2), axis=1)) +acc) %Samps.shape[0] , : , 0]
            acc +=1
            print "new centroid artificially generated Cent: "# +str(Cents) 

    return Cents


def new_centroids(guess,Samps,Cents):
    K = Cents.shape[2]
    for k in xrange(K):
        #Cents[0,:,k] = np.transpose( np.mean(  Samps[guess == k,:,0], axis = 0 ) )
        Cents[0,:,k] = np.mean( Samps[guess == k], axis = 0 ) 
        if np.any(np.isnan( Cents[0,:,k])):#not all(
#             Cents[0,:,k]= np.mean( Samps, axis = 0 )# + 0.25*np.std(Samps, axis = 0 )*(rand(1) -0.5)
            Cents[0,:,k]= Samps[0,:]
        #Cents[k,:] =  np.mean(  Samps[guess == k,:], axis = 0 ) 
    return Cents

'''
first entrance to function (t=0)
third itereation of k, k=2

pydev debugger: starting (pid: 12388)
Cents
[[[  1.02215624   4.83881998   7.43870332   3.09284475]
  [  1.02215624   4.83881998  15.08969593   3.09284475]]]
'''
''
#Cents[k] = Samps[np.argmax(np.linalg.norm(Samps-Cents, axis=1) , axis =1),:,0]
#np.linalg.norm(Samps-Cents, axis=1)            
#            Cents[k] = Samps[np.argmax( np.linalg.norm(Samps[:,:,0]-np.mean(Cents[0], axis=1), axis=1), axis=1),:,0]


    
        
        
        #guess = np.argmin(np.sum(np.abs(Samps-Cents, axis = 1 ) axis=1)  , axis = 2)
        #np.sum(np.abs(Samps-Cents, axis = 1 ) axis=1) 
        #guess = np.argmin( (Samps-Cents), axis = 2)
        