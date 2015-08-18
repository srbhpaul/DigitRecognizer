#!/usr/bin/env python

# not a module, is run once to get the desired data file
# inputs for handwriting recognition
# prepare the input file (train.csv) by chopping it
# to a sizable extent for quick computation

"""
Contains:
1> input training file train.csv

The training file contains 42000 examples.
The first column is the digit (labels => Y matrix),
while the remaining 784 (28x28) are the pixel values
for that corresponding digit (X matrix). 
"""

import pandas as pd
import numpy as np

# -------------- reading data files -------------- #
# we use the last half of the training set as test data
traindata = pd.read_csv('train.csv')
# ------------------ xxx -------------- read files


# ------------- scaling the pixel values---------- #
length = ( traindata.shape[0], traindata.shape[1] )
X = np.zeros( (length[0],length[1]-1),float )
X+= traindata.ix[:,1:]
Xmean = np.mean( X, axis=1 )
Xstd = np.std( X, axis=1 )
for i in xrange(X.shape[0]):
    X[i,:] = (X[i,:]-Xmean[i])/(Xstd[i])
# -------------- xxx ---------- scaling


# ------------------ classifiers ----------------- #
# some given facts about the data sets
# number of classifiers ( or digits in this case )
k = 10                 # constant
# and the digits themselves
digits = np.arange(10) # constant
# ------------------ xxx ------------- classifiers


# --------------- creating a smaller data set -------- #
# enter the number of cases (num_cases) for every digit instance
# for a digit 'd', we choose a subset of training examples
# such that there are more number of examples corresponding
# to digit 'd', we call this number extra
# the base number of other background digits, we call base
extra = 100 #100 
base = 150
num_cases = base*np.ones(k,int)
num_cases[9]+= extra
tot_cases = np.sum(num_cases)

# we initialize a numpy array of k elements
# whose elements keep count of the
# number of times a particular digit appears in traindata
# we stop when we have reached num_cases for each data
# this array is syncronized with the list "digits"
# this is an expensive loop!
digit_count = np.zeros(k,int)

# we do the manipulations using numpy arrays
# since we no longer need the labels or indices 
train_chopped = np.zeros( (tot_cases,length[1]-1), float )
labels_chopped = np.zeros(tot_cases,int)

# storing 100 or so random instances of every digit
# from the entire training set train.csv
digit_count = np.zeros(k,int)
# storing the first 100 instances of every digit
# from the training set train.csv
count = 0
for i in xrange(traindata.shape[0]): #traindata.shape[0]
    for j in xrange(k):
        if traindata.label[i] == digits[j] and digit_count[j] < num_cases[j]:
            digit_count[j]+=1
            train_chopped[count,:] = X[i,:]
            labels_chopped[count] = traindata.label[i]
            count+=1
Yfirst = labels_chopped
Xfirst = train_chopped
filename = 'trainfirst' + str(base) + 'dig9' + str(base+extra) + '.npz'
#filename = 'trainfirst' + str(base) + '.npz'
np.savez(filename, X=Xfirst, Y=Yfirst)
# --------------- xxx ------------- smaller file



