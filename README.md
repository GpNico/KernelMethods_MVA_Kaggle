# KernelMethods_MVA_Kaggle
Code associated to the MVA Kaggle Challenge of the course Kernel Methods.

# Suggested Plan

a) implement a linear classifier to use the _mat100.csv  files (e.g., logistic regression). You can also start with a ridge regression estimator used as a classifier (regression on labels -1, or +1), since this is very easy to implement. # Nicolas (21/02)

b) move to a nonlinear classifier by using a Gaussian kernel (still using the *_mat100.csv files).  Kernel ridge regression is a good candidate, but then you may move to a support vector machine (use a QP solver). Wny starting with kernel ridge regression => because it can be implemented in a few lines of code. # Jean (21/02)

c) then start working on the raw sequences. Time to design a good kernel for this data! # We will see...

# Expected behavior of the files :

'''
python linear_regression.py -d dataset/my_dataset.csv
'''
