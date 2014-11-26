'''
Created on Nov 24, 2014

@author: Idan
'''
import unittest

import k_mean_module 
from src.k_mean_module import mean_varience_diff

K_STATIC = 4
ITER_STATIC = 50
SIG=0.5
MU=1

class Kmean_Test(unittest.TestCase):
    
    def setUp(self,P=80):
        '''
        #for fixture
        define the fixure
        '''
        self.P=P
        self.x1 = k_mean_module.create_test_set1(P,N=2,mu=MU,sig=SIG)
        pass


    def tearDown(self):
        '''
        #for fixture
        free the fixture
        '''

        del self.x1
        del self.P

        pass


    def test1(self,K = K_STATIC,Iter = ITER_STATIC):
        '''
        conduct test1 here
        
        fails if the distance from the mean is high (1?)
        '''                
        _, centroids = k_mean_module.my_k_means(self.x1, K, Iter)
        d = k_mean_module.mean_distance_from_means(centroids,[ [2,3],[3,2],[4,1],[-1,0] ])
#         print "\ndistance of centroids and means is: " +str(round(d,2))
        self.failUnlessAlmostEqual(0, d, places=0)
        
    def test2(self,K = K_STATIC,Iter = ITER_STATIC):
        '''
        conduct test2 here
        fails if the number of guesses is smaller then the number of Samples
        '''
        res1, _ = k_mean_module.my_k_means(self.x1, K, Iter)
        self.failUnlessEqual( self.P , k_mean_module.recount_samples(res1,K) ,'hi!2')     
#         print "\nguesses== samples"

    def test3(self,K = K_STATIC,Iter = ITER_STATIC):
        '''
        conduct the test3 here
        '''
        res, centroids = k_mean_module.my_k_means(self.x1, K, Iter)
        d2 = mean_varience_diff(centroids, self.x1, res, sig=SIG, stds = None)
        self.failUnlessAlmostEqual(0, d2, places=0)
        #         print "\ndistance of cariences and means is: " +str(round(d2,2))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()