# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def cluster1(Samples,Centers,guess,T):
    #primal
    P ,N = np.size( Samples)
    K ,_ = np.size( Centers)
    
    Samps = Samples.reshape(P,N,1)
    Cents = Centers.reshape(1,N,K)
    
    guess = fit_groups(Samps,Cents)    
    
    for _ in xrange(T-1):                
        counter = count_samples_in_group(guess,K)
        counter = fix_empty_groups(counter,Samps,Cents)
        
        Cents = new_centroids(guess,Samples) 
        guess = fit_groups(Samps,Cents)    
        
    return guess
    
'''
what is more consuming?
reshaping Centers and Samples?

or approacing the 3d matrics via Cents[0,:,k] or Samples[guess == k,:,0]
'''    
        

def fit_groups(Samps,Cents):
    return np.argmin( np.linalg.norm(Samps-Cents, axis=1) , axis =1)
    #return np.argmin( np.linalg.norm(Samples.reshape(P,N,1)-Centers.reshape(1,N,K), axis=1) , axis =1)

def count_samples_in_group(guess,K):
    result =[]*K
    for k in enumerate(result):
        result[k] = np.sum( guess == k)
    return result

def fix_empty_groups(counter,Samps,Cents):
    for c in counter:
        if not c:
            c = Samps[np.argmax(np.linalg.norm(Samps-Cents, axis=1) , axis =1),:,0]
    return counter

def new_centroids(guess,Samps,Cents):
    K = Cents.shape[2]
    for k in xrange(K):
        Cents[0,:,k] = np.transpose( np.mean(  Samps[guess == k,:,0], axis = 0 ) )
        
        #Cents[k,:] =  np.mean(  Samps[guess == k,:], axis = 0 ) 
    return Cents
    
        
        
        #guess = np.argmin(np.sum(np.abs(Samps-Cents, axis = 1 ) axis=1)  , axis = 2)
        #np.sum(np.abs(Samps-Cents, axis = 1 ) axis=1) 
        #guess = np.argmin( (Samps-Cents), axis = 2)
        