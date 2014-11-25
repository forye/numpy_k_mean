'''
Created on Nov 24, 2014

@author: Idan
'''
import unittest

import k_mean_module 

class Kmean_Test(unittest.TestCase):
    
    


    def setUp(self,P=80):
        '''
        #for fixture
        define the fixure
        '''
        self.P=P
        self.x1 = k_mean_module.create_test_set1(P,N=2,sig=0.25)
        pass


    def tearDown(self):
        '''
        #for fixture
        free the fixture
        '''

        del self.x1
        del self.P

        pass


    def test1(self,K=2,Iter=50):
        '''
        conduct test1 here
        
        fails if the distance from the mean is high (1?)
        '''                
        res1, centroids = k_mean_module.my_k_means(self.x1, K, Iter)
        d = k_mean_module.distance_from_means(centroids,[ [2,3],[3,2],[4,1],[-1,0] ])
        self.failUnlessAlmostEqual(0, d, places=0)
        
    def test2(self,K=2,Iter=50):
        '''
        conduct test2 here
        
        fails if the number of guesses is smaller then the number of Samples
        
        '''
        res1, centroids = k_mean_module.my_k_means(self.x1, K, Iter)
                
        self.failUnlessEqual( self.P , k_mean_module.recount_samples(res1,K) )
     

        
    def test3(self):
        '''
        conduct the test3 here
        '''
        self.failIfAlmostEqual(1.1, 3.3-2.0, places=1)

        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()