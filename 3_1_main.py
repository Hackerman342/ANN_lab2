# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:36 2020

@author: kwc57
"""

# Import package dependencies
import sys
import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Import necessary classes from script with functions
from RBF_functions import RadialBasisFunctions


def transform_RBF_square(values):
    return [1 if i > 0 else -1 for i in values]


def RBF_NN(n_nodes, sin_or_square="sin", std = 1, tranf_test=False, rand_std=False, plot=False, verbose=True):
        
    # Initialize class w/ node count
    rbf = RadialBasisFunctions(n_nodes)
            
    # Set parameters for RBF
    mu_range = [0, round(2*math.pi,1)]
    #std = 1

    # Set which input function to approximate

    #sin_or_square = 'sin'
    #sin_or_square = 'square'
    
    # Boolean for whether or not to use random standard deviations
    #rand_std = True
    #rand_std = False 
    
    # Set parameters for data
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    step = 0.1
    #tranform RBF output to square
    tranf_train = False
    tranf_test = False
    
    # Call functions to generate train and test dataseta
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)
    
    if sin_or_square == 'sin':
        f_train = sin_train
        f_test = sin_test
    elif sin_or_square == 'square':
        f_train = square_train
        f_test = square_test
        

    mu_vec = np.linspace(mu_range[0], mu_range[1],rbf.node_count)

    if rand_std:    
        std_vec = std*np.random.rand(rbf.node_count)
    else:
        std_vec = std*np.ones((1,rbf.node_count))
    
    # Square wave function approximation
#    mu_vec = np.array([1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,
#                       4,4,4,4,5,5,5,5,5,5,5,5])
#    std_vec = np.array([1., 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 1., 1/3, 1/5,
#                        1/7, 1/9, 1/11, 1/13, 1/15, 1., 1/3, 1/5, 1/7, 1/9, 1/11,
#                        1/13, 1/15, 1., 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15, 
#                        1., 1/3, 1/5, 1/7, 1/9, 1/11, 1/13, 1/15])
     
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_vec, std_vec)
    phi_test = rbf.build_phi(x_test, mu_vec, std_vec)     
    # Call least squares functoin to calc ls weights
    w = rbf.least_squares(phi_train, f_train)
    
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)
    
    if tranf_train:
        fhat_train = transform_RBF_square(fhat_train)
    if tranf_test:
        fhat_test = transform_RBF_square(fhat_test)
        
    
    ARE_train = rbf.ARE(f_train, fhat_train)
    ARE_test = rbf.ARE(f_test, fhat_test)
    if verbose:
        print("Training ARE: ", ARE_train)
        print("Training visual fit")
    if plot:
        plt.plot(x_train,f_train)
        plt.plot(x_train,fhat_train)
        plt.show()
    
    if verbose:
        print("Testing ARE: ", ARE_test)
        print("Testing visual fit")
    if plot:
        plt.plot(x_test,f_test,'k',label='Real')
        plt.plot(x_test,fhat_test, '--c', label='Predicted')
        title = str(n_nodes) + " hidden nodes"
        plt.title(title)
        plt.legend()
        plt.show()
    return ARE_test

if __name__ == "__main__":
    RBF_NN(11, sin_or_square="sin", std = 1, tranf_test=False, rand_std=False, plot=True)
    
    '''
    sigma_list = np.linspace(0.001, 1, 1000)
    for sigma in sigma_list:
        for nodes in range(1, 140):
            print("Sigma " + str(sigma) + ". Nodes " + str(nodes) + "\n")
            ARE = RBF_NN(nodes, sin_or_square='square', std = sigma, plot=False, verbose=False)
            if ARE <= 0.001:
                break
    '''

