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



if __name__ == "__main__":
    
    # Initialize class w/ node count
    rbf = RadialBasisFunctions(10000)
            
    # Set parameters for RBF
    mu_range = [-5, 5]
    std = 1

    # Set which input function to approximate
    #sin_or_square = 'sin'
    sin_or_square = 'square'
    
    # Set parameters for data
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
        

    mu_vec = np.linspace(mu_range[0], mu_range[1],rbf.node_count)
    #std_vec = std*np.ones((1,rbf.node_count))    
    std_vec = std*np.random.rand(rbf.node_count)
    #std_vec = np.linspace(.2,std,rbf.node_count)
    
     
    ls_weights, phi_train = rbf.least_squares(x_train, mu_vec, std_vec, f_train)
    _, phi_test = rbf.least_squares(x_test, mu_vec, std_vec, f_test)
    
    fhat_train = np.dot(phi_train, ls_weights)
    fhat_test = np.dot(phi_test, ls_weights)
    
    ARE_train = rbf.ARE(f_train, fhat_train)
    ARE_test = rbf.ARE(f_test, fhat_test)
    
    print("Training ARE: ", ARE_train)
    print("Training visual fit")
    plt.plot(x_train,f_train)
    plt.plot(x_train,fhat_train)
    plt.show()

    print("Testing ARE: ", ARE_test)
    print("Testing visual fit")
    plt.plot(x_test,f_test)
    plt.plot(x_test,fhat_test)
    plt.show()

