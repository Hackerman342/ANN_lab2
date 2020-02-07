# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:36 2020

@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt
import time 
# Import necessary classes from script with functions
from RBF_functions import RadialBasisFunctions

np.random.seed(123)


def RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'ls', eta = 0.01, epochs=100, add_noise = False, plot_train=False, plot_test = True, verbose=True, plot_mu_rbf=True):
    rbf = RadialBasisFunctions(n_nodes,eta)
    # Set parameters for RBF
    mu_range = [0, round(2*math.pi,1)]
    
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
    
    mu_vec = mu_vec.reshape(len(mu_vec),-1)
    x_train = x_train.reshape(len(x_train),-1)
    x_test = x_test.reshape(len(x_test),-1)
    #mu_vec = np.random.uniform(low=mu_range[0], high=mu_range[1], size=rbf.node_count)
        
   
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_vec, std)
    phi_test = rbf.build_phi(x_test, mu_vec, std)  
    
    start_time = time.time()
    if ls_or_delta == 'ls': 
        # Call least squares functoin to calc ls weights
        w = rbf.least_squares(phi_train, f_train)#.reshape(-1,1)
    

    elif ls_or_delta == 'delta':
        w = rbf.delta_learning(f_train, phi_train, epochs, f_test = f_test, phi_test = phi_test, randomize_samples=True)
        # Flatten w to 1-D for dot product
        w = w.flatten()    
    print("--- %s seconds ---" % (time.time() - start_time))

    # Calculate predicted outputs
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)
    
    
    # Measure absolute residual error
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
            plt.scatter(mu_vec.reshape(-1,), np.zeros(len(mu_vec)), marker="x", c="r", label="RBF centers")
        plt.legend()
        text_title = str(n_nodes) + " hidden nodes over test data"
        plt.title(text_title)
        plt.show()
        
    return ARE_test


 """
    #### Batch learning with noise 
    n_nodes=6
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'ls', epochs=100, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=8
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'ls', epochs=100, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=11
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'ls', epochs=100, add_noise = True, plot_train=True, plot_test = True)
    
    n_nodes=50
    RBF_NN(n_nodes, sin_or_square="square", std =0.08, ls_or_delta = 'ls', epochs=100, add_noise = True, plot_train=True, plot_test = True)
   
a = []
b = []
c = []

for i in range(30):    
    #### Online learning without noise 
    n_nodes=6
    a.append(RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=False, plot_test = False, verbose=False))

    n_nodes=8
    b.append(RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=False, plot_test = False, verbose=False))

    n_nodes=11
    c.append(RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=False, plot_test = False, verbose=False))
    

a = np.array(a)
np.mean(a)
np.std(a)

b = np.array(b)
np.mean(b)
np.std(b)

c = np.array(c)
np.mean(c)
np.std(c)


    n_nodes=80
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100, add_noise = False, plot_train=True, plot_test = True)
    

    n_nodes=6
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=True, plot_test = True)

    n_nodes=8
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=True, plot_test = True)

    
    n_nodes=11
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = False, plot_train=True, plot_test = True)
    
    


    #### Online learning with noise    
    n_nodes=6
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=8
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=11
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100, add_noise = True, plot_train=True, plot_test = True)
    
    n_nodes=50
    RBF_NN(n_nodes, sin_or_square="square", std =0.08, ls_or_delta = 'delta', epochs=100, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=6
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=8
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = True, plot_train=True, plot_test = True)

    n_nodes=11
    RBF_NN(n_nodes, sin_or_square="sin", std = 1, ls_or_delta = 'delta', epochs=100000, add_noise = True, plot_train=True, plot_test = True)
    
""" 

if __name__ == "__main__":
  
    etas = np.linspace(0, 1, 100)[1:]
    results = []
    epochs = 1000
    for e in etas:
        results.append(RBF_NN(11, sin_or_square="sin", std = 1, eta = e, ls_or_delta = 'delta', epochs=epochs, add_noise = True, plot_test=False, verbose=False))
    #print(results)
    title_txt = "Error over the test set epochs=" + str(epochs)
    plt.title(title_txt)
    #plt.plot(etas, results)
    a=results[:73]
    plt.plot(etas[:73], a, "c")
    plt.xlabel('Learning rate')
    plt.ylabel('ARE')
    plt.show()
    
    mas_epochs = a 
    results = []
    for e in etas:
        results.append(RBF_NN(11, sin_or_square="sin", std = 1, eta = e, ls_or_delta = 'delta', epochs=100, add_noise = True, plot_test=False, verbose=False))
    a=results[:73]
    title_txt = "Error over the test set with delta learning"
    plt.title(title_txt)
    plt.plot(etas[:73], a, "k", label="epochs=100")
    plt.plot(etas[:73], mas_epochs, "--c", label="epochs=1000")
    plt.xlabel('Learning rate')
    plt.ylabel('ARE')
    plt.legend()
    plt.show()
    
    """
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
            err1 = RBF_NN(nodes, sin_or_square="square", std = sigma, ls_or_delta = 'ls', epochs=100, add_noise = True, rand_std=False, plot=False, verbose=False)
            if err1 < ls_error:
                ls_error = err1
                ls_sigma = sigma
                ls_nodes = nodes
            print(err1)
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

    #print(RBF_NN(55, sin_or_square="sin", std = 0.7145, ls_or_delta = 'delta', eta = 0.01, epochs=100, add_noise = True, rand_std=False, plot=False, verbose=False))
    """