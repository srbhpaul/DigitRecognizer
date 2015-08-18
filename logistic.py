# module for logistic regression

"""
Contains:
1> hypothesis for logistic regression
2> cost function for logistic regression
3> INPUTS: feature matrix X (1st column has to be 1), size MxN
           classifier matrix Y (as row vector), size 1xM
           hypothesis parameter matrix theta of size X.shape[1]
           theta has to be a row vector, size 1xN
"""
import numpy as np
import scipy.io as scio
import scipy.stats as spstats
import scipy.optimize as scopt
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- logistic regression -------------- #
# define the sigmoid function
def g(z):
    return 1.0/(1.0 + np.exp(-z))

# define the hypothesis function
# INPUT theta and feature matrix X as numpy arrays
# the first element in X has to be 1
# theta matrix has the same size as no of features in X (after adding 1) 
class hypothesis:
    def __init__(self,theta):
        self.theta = theta
    def fn(self,X):
        l = self.theta.shape[0]
        prod = np.dot( X,np.transpose(self.theta) )
        return g(prod)
    # find out the probability outcome
    # for a given X data, X = ([1,X1,X2,...])
    def prob(self,X):
        return np.sum( self.fn(X) )
    def __call__(self,X):
        return self.fn(X)

# regularized cost function
# reg is the regularization parameter
# INPUT theta, X, Y as numpy arrays
# theta, X follow same rule as for hypothesis part
# enter Y as a row vector (NOT a column vector)
class J:
    def __init__(self,X,Y,reg):
        self.X = X
        self.Y = Y
        self.reg = reg
    # define the cost function as a function of theta
    def fn(self,theta):
        # theta is the hypothesis parameter matrix
        # cost function construction
        X = self.X
        Y = self.Y
        reg = self.reg
        leni = Y.shape[0]
        lenj = X.shape[1]
        h = hypothesis(theta)
        arr1 = np.log( h(X) )
        arr2 = np.log( 1 - h(X) )
        regarr = (reg/(2.0*leni))*( np.sum( np.square(theta) ) - theta[0]**2)
        arr = -np.dot(Y,arr1) - np.dot( (1 - Y),arr2 ) + regarr
        costfn = (1./leni)*np.sum( arr )
        return costfn
    # this call statement returns the cost function 
    # which behaves as a function of theta
    def __call__(self,theta):
        cost = self.fn(theta)
        return cost

# define the derivative of the cost function
# required for quicker minimization using CG method
class Jprime:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def fnprime(self,theta):
        X = self.X
        Y = self.Y
        h = hypothesis(theta)
        leni = Y.shape[0]
        lenj = X.shape[1]
        tmpmat1 = ( h(X)-np.transpose(Y) )
        tmpmat2 = np.transpose(X)
        costfnprime = (1./leni)*np.dot(tmpmat2,tmpmat1)
        costfnprime = costfnprime.flatten()
        return costfnprime
    # this call statement returns the derivative
    # of the cost function 
    # which behaves as a function of theta
    def __call__(self,theta):
        costprime = self.fnprime(theta)
        return costprime

# minimize the logistic cost function
# and get the hypothesis parameter matrix theta
def findtheta(J,Jprime,theta):
    res = scopt.minimize(J,theta,method='CG',jac=Jprime)
    #res = scopt.fmin_cg(J,theta,fprime=Jprime)
    #options={'gtol': 1e-6,'disp':True}
    return res
# -------------------- xxx -------------- logistic regress 
