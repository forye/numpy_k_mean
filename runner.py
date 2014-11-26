'''
Created on Nov 25, 2014

@author: Idan
'''

if __name__ == '__main__':
    pass
'''
P = number of samples
K1 =number of clusters
Iter1=number if itterations to finish
S=times of reexecution
'''
from k_mean_module import create_test_set1,my_k_means,plot_results,create_test_set2

P = 1000
K1 =2
Iter1=100
S=1
    
# x1= create_test_set1(P,N=2,sig=0.5)
x1= create_test_set2(P,N=2,sig1=0.5, sig2=0.5,q=0.95)



i=0# iterratuib start
S=S-i# itteration end+1

while i<S:
    res1, centroids = my_k_means(x1, K1, Iter1)
#     d = distance_from_means(centroids)
#     a= distance_from_means(centroids,[ [2,3],[3,2],[4,1],[-1,0] ])
#     print d 
    i+=1
plot_results(x1,res1,centroids)