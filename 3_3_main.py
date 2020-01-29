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
    rbf = RadialBasisFunctions(10)
    # Set parameters for RBF
    mu_range = [0, round(2*math.pi,1)]
    std = 1
    rbf.lr = .01
    epochs = 100
    
    # Set which input function to approximate
    sin_or_square = 'sin'
    sin_or_square = 'square'
    
    # Boolean for whether or not to use random standard deviations
    rand_std = True
    #rand_std = False    
    
    # Boolean for whether or not to add gaussian noise
    add_noise = True
    add_noise = False
    
    # Set parameters for data
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    step = 0.1

    
    # Call functions to generate train and test dataseta
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)

    
    if add_noise:
        # Add zero mean, low variance noise to data
        sin_train = rbf.add_gauss_noise(sin_train, 0, 0.1)
        square_train = rbf.add_gauss_noise(square_train, 0, 0.1)
        sin_test = rbf.add_gauss_noise(sin_test, 0, 0.1)
        square_test = rbf.add_gauss_noise(square_test, 0, 0.1)

    
    if sin_or_square == 'sin':
        f_train = sin_train
        f_test = sin_test
    elif sin_or_square == 'square':
        f_train = square_train
        f_test = square_test
        
    # Build equally spaced mu_vec to initialize
    mu_vec = np.linspace(mu_range[0], mu_range[1],rbf.node_count)
    #mu_vec = np.array([-1.,2.,3.,-4.,5.,6.8,7.8,8.2,9.,0.])
    print(mu_vec)    
    # Optimize mu_vec centers with competitive learning
    rand_ids = np.random.permutation(x_train.size)     
    for j in rand_ids:
        mu_vec =  rbf.competitive_learning_1D(x_train[j], mu_vec)
    print(mu_vec)    

        
    if rand_std:    
        std_vec = std*np.random.rand(rbf.node_count)
    else:
        std_vec = std*np.ones((1,rbf.node_count))
    
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_vec, std_vec)
    phi_test = rbf.build_phi(x_test, mu_vec, std_vec)  
    
    # initialize random weights as column
    w = np.random.randn(rbf.node_count).reshape(-1,1)
    # Choose random order of points for weight update 
    rand_ids = np.random.permutation(x_train.size)     
    # Initialize vectors for storing errors
    ARE_train = np.zeros((1,epochs)).flatten()
    ARE_test = np.zeros((1,epochs)).flatten()
    # Iteratively call delta rule function and update weights
    for i in range(epochs):
        for j in rand_ids:
            w += rbf.delta_rule(x_train[j], f_train[j], w, phi_train[j,:])
        # Flatten w to 1-D for dot product - just used for ARE trend/plot
        w_temp = w.flatten()
        # Calculate predicted outputs
        fhat_train = np.dot(phi_train, w_temp)
        fhat_test = np.dot(phi_test, w_temp)
        # Measure absolute residual error
        ARE_train[i] = rbf.ARE(f_train, fhat_train)
        ARE_test[i] = rbf.ARE(f_test, fhat_test)
    # Show ARE trend
    plt.plot(ARE_train, label = "Training ARE")
    plt.plot(ARE_test, label = "Testing ARE")
    plt.legend()
    plt.show()
    # Flatten w to 1-D for dot product
    w = w.flatten()    

    # Calculate predicted outputs
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)
    
    # Measure absolute residual error
    ARE_train = rbf.ARE(f_train, fhat_train)
    ARE_test = rbf.ARE(f_test, fhat_test)
    
    print("Training ARE: ", ARE_train)
    print("Training visual fit")
    plt.plot(x_train,f_train)
    plt.plot(x_train,fhat_train)
    plt.scatter(mu_vec,np.zeros(mu_vec.shape))
    plt.show()

    print("Testing ARE: ", ARE_test)
    print("Testing visual fit")
    plt.plot(x_test,f_test)
    plt.plot(x_test,fhat_test)
    plt.show()
    
    
        

