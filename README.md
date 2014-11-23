numpy_k_mean
============

this is a k-mean realization (only in numoy) - multivariate and with arbitary k in python
using numpy and linear algebra

1. T- number of itterations
2. plot is able up to 6 groups
3. emphesis on using vector computation
4. each empty group, initialized with the most far sample from the avrage of samples

to be done:
1. fix the double dealing with empty group
2. measuere times and compare
3. unerror weird data types, and nan handling
3. normalize weighs of samples
4. create and option of hypothesis
5. create a filtered input (remove outliers)
6. PCA- remove redudndant groups




first remark:

question in python (numpy)

what is more consuming?

reshaping Centers and Samples each time?
each itteration, reshaping again, meaning recaculating the reshape

or
reshaping Centers and Samples only once?
approacing the 3d matrics via Cents[0,:,k] or Samples[guess == k,:,0]
pro - reshaping only once
con - neet to approche the matrics using ":" and i=0 exc..

'''    
        
