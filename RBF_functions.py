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
import random 



class RadialBasisFunctions():
    
    def __init__(self, node_count, lr=0.01):
        # Initilize class variables
        self.node_count = node_count
        self.lr = lr
    
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
    
    
    def build_phi(self, x, mu_RBF, std):
        """
        @param x: rows = number of training samples, columns = dimensions of input data 
        @param mu_RBF: rows = number of RBF nodes, columns = dimensions of input data
        @param std: decimal number 
        @return phi: rows = number of training samples, columns = number of RBF nodes
        """
        n_training_samples = len(x)
        n_RBF_nodes = len(mu_RBF)
        phi = np.zeros((n_training_samples,n_RBF_nodes))
        for idx_rbf_node in range(n_RBF_nodes):
            # Calculate array of transfer functions for all combos of x & mu+std
            phi[:,idx_rbf_node] = self.gauss_transfer_function(x, mu_RBF[idx_rbf_node], std)
        return phi        
    
    
    def least_squares(self, phi, f):
        # Calculate least squares of weight vector - ignore other returns
        w = np.linalg.inv(np.transpose(phi) @ phi) @ np.transpose(phi) @ f
        
        return w   
    
    
    def delta_learning(self, f_train, phi_train, epochs, plot_result_per_epoch = False, f_test = None, phi_test = None, randomize_samples=False):
        # initialize random weights as column
        w = np.random.randn(self.node_count, f_train.shape[1]).reshape(-1,f_train.shape[1])
        # Choose random order of points for weight update 
        # Initialize vectors for storing errors
        ARE_train = np.zeros(epochs)
        ARE_test = np.zeros(epochs)
        # Iteratively call delta rule function and update weights
        for i in range(epochs):
            if randomize_samples:
                rand_ids = np.random.permutation(f_train.shape[0])     
            else:
                rand_ids = range(f_train.shape[0])
                
            for idx_training_point in rand_ids:
                f_point = f_train[idx_training_point]
                f_point= f_point.reshape(1,-1)
                w += self.delta_rule(f_point, w, phi_train[idx_training_point])
            
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
        
        if f_point.shape[0] != 1:
            raise Exception("Only pass one point at at time to delta_rule func")
        
        # Calculate weight updates (column vector)
        error = f_point - np.dot(phi_x, w)
        dw = self.lr*(error)*phi_x.reshape(-1,1)
        return dw
    
    """
    def competitive_learning(self, x_train, mu_vec, n_epochs=100):
        for epoch in range(n_epochs):
            rand_ids = np.random.permutation(x_train.size)     
            # limit number of iterations??
            for iteration_training_point in range(len(rand_ids)):
                # idx_training_sample = random.randint(0,len(x_train)-1)
                idx_training_sample = rand_ids[iteration_training_point]
                mu_vec =  self.update_mu_CL(x_train[idx_training_sample], mu_vec)
                
        return mu_vec
    """
    
    def update_mu_CL(self, x_train_point, mu_vec, n_winners):
        """
        Update the mu of the winning RBF node (the one most near x_training_point)
        """
        # Find index of closest mu (gaussian center) a.k.a. the 'winner'
        
        # euclidean distance 1D
        idx_winning_rbf = np.argsort(self.compute_euclidean_distance(x_train_point, mu_vec))[:n_winners]
        # Update winner [shift towards x]
        lr_exp = np.arange(1, n_winners + 1, 1)
        mu_vec[idx_winning_rbf] += np.power(self.lr, lr_exp)[:, None]*(x_train_point - mu_vec[idx_winning_rbf])
        return mu_vec
 
        
    
    def gauss_transfer_function(self, x, mu_RBF_node, std):
        """
        @param x: rows = number of training samples, columns = dimensions of input data 
        @param mu_RBF_node: vector with length = dimensions of input data
        @param std: decimal number 
        @return vector with the gaussian transfer function of each training point to the RBF node (length=number of training samples)
        """
        # square euclidean distance 
        r_square = np.square(self.compute_euclidean_distance(x, mu_RBF_node))
        gauss_transfer_function_train_samples = np.exp(-1*np.divide(r_square,(2*np.square(std))))
        return gauss_transfer_function_train_samples
    
    
    def compute_euclidean_distance(self, x, mu_RBF_node):
        sqrt_term = np.sum(np.square(x-mu_RBF_node), axis = 1)
        euclidean_distance = np.sqrt(sqrt_term)
        return euclidean_distance
    
    # Calculate mean square error between function and its approximation
    def MSE(self, f, f_hat):
        return np.mean(np.square(f - f_hat))
    
    # Calculate absolute residual error between function and its approximation
    def ARE(self, f, f_hat):
        return np.mean(np.abs(f - f_hat))
    
    

    # Adds random gaussian noise to array x
    def add_gauss_noise(self, x, mu, std):
        noise = np.random.normal(mu, std, x.size)
        return x + noise
    
    
    def plot_results(self, x, f, fhat, print_mu_centers=False, mu_vec=None, mu_vec_init=None):
        plt.plot(x,f,'k',label='Real')
        plt.plot(x,fhat, '--c', label='Predicted')
        if print_mu_centers:
            old_and_new_mu_vec = np.concatenate([mu_vec,mu_vec_init])
            classes = np.zeros(len(old_and_new_mu_vec))
            classes[len(mu_vec):] = 1 
            plt.scatter(mu_vec_init,np.zeros(len(mu_vec_init)), label="initial mu", alpha=0.3) 
            plt.scatter(mu_vec,np.zeros(len(mu_vec)), label="mu after CL", alpha=0.5) 

        title = str(self.node_count) + " hidden nodes"
        plt.title(title)
        plt.legend()
        plt.show()



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
        