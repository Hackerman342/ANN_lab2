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

np.random.seed(123)
def RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'ls', eta = 0.01, epochs=100, add_noise = False, rand_std=False, plot=False, verbose=True):
    rbf = RadialBasisFunctions(n_nodes)
    # Set parameters for RBF
    mu_range = [0, round(2*math.pi,1)]
    #std = 0.08
    rbf.lr = eta
    #epochs = 100
    
    # Set which input function to approximate
    #sin_or_square = 'sin'
    #sin_or_square = 'square'
    
    # Set which input function to approximate
    #ls_or_delta = 'ls'
    #ls_or_delta = 'delta'
    
    # Boolean for whether or not to use random standard deviations
    #rand_std = True
    #rand_std = False    
    
    # Boolean for whether or not to add gaussian noise
    #add_noise = True
    #add_noise = False
    
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
        

    mu_vec = np.linspace(mu_range[0], mu_range[1], rbf.node_count)
    np.random.uniform(low=mu_range[0], high=mu_range[1], size=rbf.node_count)
        
    if rand_std:    
        std_vec = std*np.random.rand(rbf.node_count)
    else:
        std_vec = std*np.ones((1,rbf.node_count))
    
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_vec, std_vec)
    phi_test = rbf.build_phi(x_test, mu_vec, std_vec)  
    
    if ls_or_delta == 'ls': 
        # Call least squares functoin to calc ls weights
        w = rbf.least_squares(phi_train, f_train)#.reshape(-1,1)
    
    elif ls_or_delta == 'delta':
        w = rbf.delta_learning(f_train, phi_train, epochs, plot_result_per_epoch = False, f_test = f_test, phi_test = phi_test)
        # Flatten w to 1-D for dot product
        w = w.flatten()    

    # Calculate predicted outputs
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)
    
    # Measure absolute residual error
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
        plt.plot(x_test,f_test)
        plt.plot(x_test,fhat_test)
        plt.show()
    return ARE_test

if __name__ == "__main__":
    
    '''
    etas = np.linspace(0, 1, 100)
    results = []
    for e in etas:
        results.append(RBF_NN(60, sin_or_square="square", std = 0.03, eta = e, ls_or_delta = 'delta', epochs=100, add_noise = True, rand_std=True, plot=False, verbose=False))
    #print(results)
    
    plt.plot(etas, results)
    plt.xlabel('Learning rate')
    plt.ylabel('ARE')
    plt.show()
    '''
    
    ls_error = np.Infinity
    ls_sigma = 0
    ls_nodes = 0
    delta_error = np.Infinity
    delta_sigma = 0
    delta_nodes = 0
    sigmas = np.linspace(0.001, 1, 50)
    for sigma in sigmas:#[0.005, 0.01, 0.05, 0.08, 0.1 , 0.3, 0.6, 0.8, 1]:
        nodes_list = np.arange(2, 70, 1)
        print(sigmas)
        for nodes in nodes_list:#[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
            print("Sigma " + str(sigma) + ". Nodes " + str(nodes) + "\n")
            #err1 = RBF_NN(nodes, sin_or_square="square", std = sigma, ls_or_delta = 'ls', epochs=100, add_noise = True, rand_std=False, plot=False, verbose=False)
            if err1 < ls_error:
                ls_error = err1
                ls_sigma = sigma
                ls_nodes = nodes
            #print(err1)
            print("\n")
            err2 = RBF_NN(nodes, sin_or_square="sin", std = sigma, ls_or_delta = 'delta', eta = 0.01, epochs=100, add_noise = True, rand_std=False, plot=False, verbose=False)
            if err2 < delta_error:
                delta_error = err2
                delta_sigma = sigma
                delta_nodes = nodes
            print(err2)
            print("\n---------------------------------")
    print("Sigma " + str(ls_sigma) + ". Nodes " + str(ls_nodes) + " Error "+ str(ls_error))
    print("Sigma " + str(delta_sigma) + ". Nodes " + str(delta_nodes) + " Error "+ str(delta_error))
    