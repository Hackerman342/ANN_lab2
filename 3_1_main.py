# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:36 2020

@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt

# Import necessary classes from script with functions
from RBF_functions import RadialBasisFunctions


def transform_RBF_square(values):
    return [1 if i > 0 else -1 for i in values]


def RBF_NN(n_nodes, sin_or_square="sin", std = 1, tranf_test=False, rand_std=False, plot_train=False, plot_test=True, verbose=True, plot_mu_rbf=True):
        
    # Initialize class w/ node count
    rbf = RadialBasisFunctions(n_nodes)
            
    # Set parameters for RBF


    # Generate datasets
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    step = 0.1
    

    # Call functions to generate train and test dataseta
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)
    
    if sin_or_square == 'sin':
        f_train = sin_train
        f_test = sin_test
    elif sin_or_square == 'square':
        f_train = square_train
        f_test = square_test
        
    # Set parameters RBF 
    mu_range = [0, round(2*math.pi,1)]
    #mu_range = [-5, 5]
    mu_RBF = np.linspace(mu_range[0], mu_range[1], rbf.node_count) #rows=number of RBF nodes, cols=number of dimensions
    #mu_RBF = np.random.normal(size=rbf.node_count)

    
    mu_RBF = mu_RBF.reshape(len(mu_RBF),-1)
    # reshape to have rows = n_training samples and cols= n_dimensions
    x_train = x_train.reshape(len(x_train),-1)
    x_test = x_test.reshape(len(x_test),-1)
    
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_RBF, std)
    phi_test = rbf.build_phi(x_test, mu_RBF, std)     
    
    # Call least squares functoin to calc ls weights
    w = rbf.least_squares(phi_train, f_train)
    
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)
    
    
    ARE_train = rbf.ARE(f_train, fhat_train)
    ARE_test = rbf.ARE(f_test, fhat_test)
    if verbose:
        print("Training ARE: ", ARE_train)
        print("Training visual fit")
        
        print("Testing ARE: ", ARE_test)
        print("Testing visual fit")
        
    if plot_train:
        plt.plot(x_train,f_train,'k',label='Real')
        plt.plot(x_train,fhat_train, '--c', label='Predicted')
        plt.legend()
        text_title = str(n_nodes) + " hidden nodes over train data"
        plt.title(text_title)
        plt.show()
        
    if plot_test:
        plt.plot(x_test,f_test,'k',label='Real')
        plt.plot(x_test,fhat_test, '--c', label='Predicted')
        if plot_mu_rbf:
            plt.scatter(mu_RBF.reshape(-1,), np.zeros(len(mu_RBF)), marker="x", c="r", label="RBF centers")
        plt.legend()
        text_title = str(n_nodes) + " hidden nodes over test data"
        plt.title(text_title)
        plt.show()
        
    return ARE_test

if __name__ == "__main__":
    RBF_NN(50, sin_or_square="square", std = 0.08, tranf_test=False, rand_std=False, plot_train=True)
    
