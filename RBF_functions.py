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
    
    
    def delta_learning(self, f_train, phi_train, epochs, plot_result_per_epoch = True, f_test = None, phi_test = None):
        # initialize random weights as column
        w = np.random.randn(self.node_count).reshape(-1,1)
        # Choose random order of points for weight update 
        # Initialize vectors for storing errors
        ARE_train = np.zeros(epochs)
        ARE_test = np.zeros(epochs)
        # Iteratively call delta rule function and update weights
        for i in range(epochs):
            rand_ids = np.random.permutation(f_train.size)     

            for idx_training_point in rand_ids:
                w += self.delta_rule(f_train[idx_training_point], w, phi_train[idx_training_point])
            
            if plot_result_per_epoch:
                # Flatten w to 1-D for dot product - just used for ARE trend/plot
                w_temp = w.flatten()
                # Calculate predicted outputs
                fhat_train = np.dot(phi_train, w_temp)
                fhat_test = np.dot(phi_test, w_temp)
                # Measure absolute residual error
                ARE_train[i] = self.ARE(f_train, fhat_train)
                ARE_test[i] = self.ARE(f_test, fhat_test)
        
        if plot_result_per_epoch:
            plt.plot(ARE_train, 'k',label='Training ARE')
            plt.plot(ARE_test, '--c', label='Testing ARE')
            title = str(self.node_count) + " hidden nodes"
            plt.xlabel("epochs")
            plt.ylabel("ARE")
            plt.title(title)
            plt.legend()
            plt.show()
        
        return w
    
    
    def delta_rule(self, f_point, w, phi_x):
        """
        @param f_point are size 1x1 | w & phi are size 1xn_hidden_nodes
        
        @return column vector with the updated weights of each RBF node
        """
    
        if f_point.size != 1:
            raise Exception("only pass one point at at time to delta_rule func")
        
        # Calculate weight updates (column vector)
        dw = self.lr*(f_point - np.dot(phi_x, w))*phi_x.reshape(-1,1)
        return dw
    
    
    def competitive_learning_1D(self, x, mu_vec):
        rand_ids = np.random.permutation(x.size)     
        # limit number of iterations??
        for iteration in range(len(rand_ids)):
            idx_training_sample = rand_ids[iteration]
            mu_vec =  self.update_mu_CL(x[idx_training_sample], mu_vec)
            
        return mu_vec
    
    
    def update_mu_CL(self, x, mu_vec):
        # Find index of closest mu (gaussian center) a.k.a. the 'winner'
        
        # euclidean distance 1D
        idx_winning_rbf = np.argmin(np.abs(x-mu_vec))
        
        # Update winner [shift towards x]
        mu_vec[idx_winning_rbf] += self.lr*(x - mu_vec[idx_winning_rbf])
        return mu_vec
 
        
    ##### Support functions ######
    
    
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
    
    ##### Unused functions ######

#    # Shuffle data for delta learning
#    def shuffle_data(self, x, y1, y2):
#        #
#        # Ensure proper shape of x, y1, & y2 is 1xN
#        x = x.reshape(-1)
#        y1 = y1.reshape(-1)
#        y2 = y2.reshape(-1)
#        
#        rand_ids = np.random.permutation(x.size)     
#        x = x[rand_ids]
#        y1 =y1[rand_ids]
#        y2 = y2[rand_ids]
#        
#        return x, y1, y2
        