# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:59:25 2020

@author: kwc57
"""

# Import package dependencies
import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




class RadialBasisFunctions():
    
    def __init__(self, node_count):
        # Initilize class variables
        self.node_count = node_count
        self.lr = .01
    
    # Generates a sin wave and square wave over specified interval
    def generate_sin_and_square(self, xrange, step):
        # Whether or not to display generated data
        plot = False
        # Generate numpy arrays of x, sin, and square
        x = np.arange(xrange[0],xrange[1],step)
        sin = np.sin(2*x)       
        square = np.sign(sin)
        
        if plot: # Show plots
            plt.plot(x,sin)
            plt.plot(x,square)
            plt.show()
            
        return x, sin, square
    
    def build_phi(self, x, mu_vec, std_vec):
        # x is size 1xN | mu_vec & std_vec are size 1xn
        # Reshape all input vectors to Nxn arrays
        x_arr = np.repeat(x.reshape(-1,1),mu_vec.size,axis=1)
        mu_arr = np.repeat(mu_vec.reshape(1,-1),x.size,axis=0)
        std_arr = np.repeat(std_vec.reshape(1,-1),x.size,axis=0)       
        # Calculate array of transfer functions for all combos of x & mu+std
        phi = self.gauss_transfer_function(x_arr, mu_arr, std_arr)
        return phi        
    
    def least_squares(self, phi, f):
        # Calculate least squares of weight vector - ignore other returns
        w, _, _, _ =  np.linalg.lstsq(phi,f, rcond=None)
        return w
    
    
    def delta_rule(self, x, f, w, phi):
        # x and f are size 1x1 | mu_vec & std_vec are size 1xn
        if x.size != 1 or f.size != 1:
            print("error - only pass one point at at time to delta_rule func")
            return 'error'
        # Calculate weight updates (column vector)
        dw = self.lr*(f - np.dot(phi.reshape(1,-1),w))*phi.reshape(-1,1)
        return dw
        
    ##### Support functions ######
    
    
    # Shuffle data for delta learning
    def shuffle_data(self, x, y1, y2):
        #
        # Ensure proper shape of x, y1, & y2 is 1xN
        x = x.reshape(-1)
        y1 = y1.reshape(-1)
        y2 = y2.reshape(-1)
        
        
        
        return x, y1, y2
    
    # Calcualte the gaussian transfer function
    def gauss_transfer_function(self, x, mu, std):
        return np.exp(-1*np.divide(np.square(x-mu),(2*np.square(std))))
    
    # Calculate mean square error between function and its approximation
    def MSE(self, f, f_hat):
        return np.mean(np.square(f - f_hat))
    
    # Calculate absolute residual error between function and its approximation
    def ARE(self, f, f_hat):
        return np.mean(np.abs(f - f_hat))
    
    # Adds random gaussian noise to array x
    def add_gauss_noise(self, x, mu, std):
        return x + np.random.randn(x.size).reshape(x.shape)*std + mu
        