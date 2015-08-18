#!/usr/bin/env python
# -------------- one-vs-all logistic regression-------- #

import numpy as np
import scipy.io as scio
import scipy.stats as spstats
import scipy.optimize as scopt
import pandas as pd
import matplotlib.pyplot as plt
# -------------- import modules ------------------ #
# this module defines the hypothesis
# and the cost function
# for a general case
import logistic as lgs
# -------------- xxx ----------- modules


# -------------- data files -------------- #
# the train data file is modified in the inputs.py file
# inputs.py is a separate py file (and not a module)
# so, to modify the train set further
# change num_cases to desired value in inputs.py
# and run the inputs.py file separately
# it saves the chopped train data as a numpy array
# we are now reading in the modified input files
traindata = np.load('trainfirst150dig5250.npz')
X = traindata['X'] # feature matrix to be constructed from X
Y = traindata['Y']  # classifier matrix (or labels)

"""
# we treat the last half of the data points
# in the training data set as our testdata
train_original = pd.read_csv('train.csv')
num = train_original.shape[0]
train_half = train_original.ix[0:num/2,:]
testdata = train_original.ix[num/2+1:num,:]
test = testdata.as_matrix()[:,1:]
"""
# ------------------ xxx -------------- read files


# ---------- preparing feature matrix from data file ------------ #
# construct the feature matrix X
# scale it properly as needed
# most importantly, the 1st column of X has to be 1
Xmat = np.insert(X,0,1,axis=1)
#Xtestmat = np.insert(test,0,1,axis=1)
#Ytestmat = testdata.ix[:,0].as_matrix()
# --------------------- xxx ----------------- feature matrix


# --------------------- classification matrices ----------------- #
# its a "one-vs-all" classification problem
# in this case, there are 10 different classifiers
# they are numbers 1,2,3,...10
# however, since we are tweaking the data set in inputs.py file
# we check how many classifiers we are left with
# and what they are!

# first sort the labels in ascending order
Ysort = np.sort(Y) 
tmp = Ysort[0]
labels = [tmp]
k = 1
for i in xrange(Ysort.shape[0]):
    if Ysort[i] > tmp:
        tmp = Ysort[i]
        k+=1 
        labels.append(tmp)

# construct the "k" number of classifiers
# we will store the "k" classifier matrices in this list
Yclasses = []
for i in xrange(k):
    # create the "k" different classifier matrices
    # (k=10 in this case)....code is for unknown k
    # initialize each of them
    Ymat = np.zeros(Y.shape[0],int)
    for j in xrange(Y.shape[0]):
        if Y[j] == labels[i]:
            Ymat[j] = 1
    Yclasses.append(Ymat)
# -------------------- xxx ------------------ matrix Y


# -------------- one-vs-all logistic regression-------- #
# we will have k different hypothesis parameter matrices
# each of size Xmat.shape[1]
# initialize every theta with a single matrix
# with proper initial guess
# that will potentially lead to faster minimization
theta = np.zeros(Xmat.shape[1],float)

# minimizing the cost function for all the k cases
# and obtain the corresponding hypothesis parameters
# we store the parameters in a matrix
# where every row correspond to each case
thetaresults = np.ones((k,Xmat.shape[1]),float) #Xmat.shape[1]
for i in xrange(k):
    Ymat = Yclasses[i]
    Xmat = Xmat
    reg = 0.0 # regularization parameter (our choice)
    # using the cost function as defined in the module logistic
    Jcost = lgs.J(Xmat,Ymat,reg)
    Jcostprime = lgs.Jprime(Xmat,Ymat)
    res =  lgs.findtheta(Jcost,Jcostprime,theta)
    thetaresults[i,:] = res.x

# hypothesis predicton for each image
# i.e. each row of the feature matrix X
# each row in "predictions" contains all the 
# k predictions for every image example in Xmat
predictions = lgs.g( np.dot( Xmat,np.transpose(thetaresults) ) )
# -------------- xxx -------------- logistic regression

