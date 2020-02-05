# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:36 2020

@author: kwc57
"""

# Import package dependencies
import math
import numpy as np 
import matplotlib.pyplot as plt
import random 

random.seed(123)

# Import necessary classes from script with functions
from RBF_functions import RadialBasisFunctions

NORMALIZATION_TERM = 6.2

def initialize_rbf_parameters(rbf, x_train, initialize_from_data_points=True, do_linespace=False, mu_range=[0,2*math.pi]):
    if initialize_from_data_points:
        idx_points_for_mu = np.random.choice(len(x_train), n_hidden_nodes)    
        # this should be randomly depending on n_nodes
        idx_points_for_mu = np.array([13, 59, 61, 44,  9, 43, 20, 55, 24, 46, 32])
        mu_vec_init = x_train[idx_points_for_mu]
    elif do_linespace:
        mu_vec_init = np.linspace(mu_range[0], mu_range[1],rbf.node_count)
    
    return mu_vec_init

"""
 def competitive_learning_1D(self, x_train, mu_vec, n_epochs=100):
        for epoch in range(n_epochs):
            rand_ids = np.random.permutation(x_train.size)     
            # limit number of iterations??
            for iteration_training_point in range(len(rand_ids)):
                # idx_training_sample = random.randint(0,len(x_train)-1)
                idx_training_sample = rand_ids[iteration_training_point]
                mu_vec =  self.update_mu_CL(x_train[idx_training_sample], mu_vec)
                
        return mu_vec

"""

def get_NN_predictions(rbf, x_train, f_train, f_test, x_test, mu_vec, std_vec, epochs_NN):
    # Build phi arrays
    phi_train = rbf.build_phi(x_train, mu_vec, std_vec)
    phi_test = rbf.build_phi(x_test, mu_vec, std_vec)  
        
    w = rbf.delta_learning(f_train, phi_train, epochs_NN, f_test = f_test, phi_test = phi_test, plot_result_per_epoch=False)
        
    # Calculate predicted outputs
    fhat_train = np.dot(phi_train, w)
    fhat_test = np.dot(phi_test, w)

    return fhat_train, fhat_test



def get_results_with_CL(rbf, n_iterations, x_train, x_test, f_train, f_test, mu_vec, std_vec, epochs_NN, plot_results=False):
    ARE_train_list = [] 
    ARE_test_list =  []
    for iteration in range(n_iterations):
        """rand_ids = np.random.permutation(x_train.size)
        for iteration_training_point in range(len(rand_ids)):
            # idx_training_sample = random.randint(0,len(x_train)-1)
            idx_training_sample = rand_ids[iteration_training_point]
            mu_vec =  rbf.update_mu_CL(x_train[idx_training_sample], mu_vec)
        """
        # Without shuffling the training data in each epoch 
        for idx_training_sample in range(len(x_train)):
            # idx_training_sample = random.randint(0,len(x_train)-1)
            mu_vec = rbf.update_mu_CL(x_train[idx_training_sample], mu_vec)
        
        
        """fhat_train, fhat_test = get_NN_predictions(rbf, x_train, f_train, f_test, x_test, mu_vec, std_vec, epochs_NN)
        ARE_train = rbf.ARE(f_train, fhat_train)
        ARE_test = rbf.ARE(f_test, fhat_test)
        ARE_train_list.append(ARE_train)
        ARE_test_list.append(ARE_test)"""
        #print("Training ARE: ", ARE_train)      
        #print("---ITERATION: " + str(iteration) + "---")
        """idx_training_sample = random.randint(0,len(x_train)-1) 
        mu_vec =  rbf.update_mu_CL(x_train[idx_training_sample], mu_vec)"""
        if iteration%100==0:
            fhat_train, fhat_test = get_NN_predictions(rbf, x_train, f_train, f_test, x_test, mu_vec, std_vec, epochs_NN)
            ARE_train = rbf.ARE(f_train, fhat_train)
            ARE_test = rbf.ARE(f_test, fhat_test)
            ARE_train_list.append(ARE_train)
            ARE_test_list.append(ARE_test)
        #print("Training ARE: ", ARE_train)
        #print("Testing ARE: ", ARE_test)"""
    
    if plot_results:
        x = range(len(ARE_train_list))
        plt.plot(x,ARE_train_list,'k',label='Train')
        plt.plot(x,ARE_test_list, '--c', label='Test')
        plt.xlabel("Number iterations CL")
        plt.ylabel("ARE")
        plt.legend()
        plt.show()
    
    
    
    

def train_RBF_network(n_hidden_nodes, use_cl, std = 1, learning_rate = .1, epochs_NN = 100, epochs_CL=100, sin_or_square = 'sin', initialize_from_data_points=True, add_noise = False, do_linespace=False, mu_range=[None,None], mu_vec_init=None, plot_results=True, normalize_training_data=True):
    
    # Initialize class w/ node count
    rbf = RadialBasisFunctions(n_hidden_nodes)
    rbf.lr = learning_rate
    
    
    # Generate data 
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    step = 0.1
    # Call functions to generate train and test dataseta
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)
    
    
    x_train = x_train.reshape(len(x_train),-1)
    x_test = x_test.reshape(len(x_test),-1)
    
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
       
    if normalize_training_data:
        x_train = x_train/NORMALIZATION_TERM
    
    ### RBF parameters initialization
    # std 
    mu_vec_init= initialize_rbf_parameters(rbf, x_train)
    
    mu_vec = mu_vec_init.copy()
    
    # Optimize mu_vec centers with competitive learning
    if use_cl:
        #mu_vec = rbf.competitive_learning_1D(x_train, mu_vec, epochs_CL)  
 
        #if np.all(mu_vec_init == mu_vec):
        #    raise Exception("CL is not performing good")
        
        n_iterations=10000
        get_results_with_CL(rbf, n_iterations, x_train, x_test, f_train, f_test, mu_vec, std, epochs_NN, plot_results=True)

        
        
    else:
        """
        # Measure absolute residual error
        ARE_train = rbf.ARE(f_train, fhat_train)
        ARE_test = rbf.ARE(f_test, fhat_test)
        
        print("Training ARE: ", ARE_train)
        print("Testing ARE: ", ARE_test)
        
        if plot_results:
            rbf.plot_results(x_train,f_train, fhat_train, print_mu_centers=True, mu_vec=mu_vec, mu_vec_init=mu_vec_init)
        
            rbf.plot_results(x_test,f_test, fhat_test)
        """
        pass
        
        
    #return ARE_train,ARE_test
    
    
    
    
    
    
   
    

def train_RBF_network_op2(n_hidden_nodes, use_cl, std = 1, learning_rate = .1, epochs_NN = 100, epochs_CL=10000, sin_or_square = 'sin', initialize_from_data_points=True, add_noise = False, do_linespace=False, mu_range=[None,None], mu_vec_init=None, plot_results=True, normalize_training_data=True):
    
    # Initialize class w/ node count
    rbf = RadialBasisFunctions(n_hidden_nodes)
    rbf.lr = learning_rate
    
    
    # Generate data 
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    step = 0.1
    # Call functions to generate train and test dataseta
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)
    
    
    x_train = x_train.reshape(len(x_train),-1)
    x_test = x_test.reshape(len(x_test),-1)
    
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
       
    
    ### RBF parameters initialization
    # std 
    mu_vec_init= initialize_rbf_parameters(rbf, x_train)
    
    mu_vec = mu_vec_init.copy()
    
          
    n_iterations=10000
    get_results_with_CL_op2(rbf, n_iterations, x_train, x_test, f_train, f_test, mu_vec, std, epochs_NN, plot_results=True)
        
    #return ARE_train,ARE_test
        
  


def get_results_with_CL_op2(rbf, epochs_CL, x_train, x_test, f_train, f_test, mu_vec, std, epochs_NN, plot_results=False):
    ARE_train_list = [] 
    ARE_test_list =  []
    for epoch in range(epochs_CL):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices) 
        new_x_train = x_train[indices]
        new_f_train = f_train[indices]
        
        mu_vec =  rbf.update_mu_CL(new_x_train[0], mu_vec)
        
        if epoch%100==0:
            fhat_train, fhat_test = get_NN_predictions(rbf, new_x_train, new_f_train, f_test, x_test, mu_vec, std, epochs_NN=1)
            
            ARE_train = rbf.ARE(new_f_train, fhat_train)
            ARE_test = rbf.ARE(f_test, fhat_test)
            
            ARE_train_list.append(ARE_train)
            ARE_test_list.append(ARE_test)
        #print("Training ARE: ", ARE_train)
        #print("Testing ARE: ", ARE_test)"""
    
    if plot_results:
        x = range(len(ARE_train_list))
        plt.plot(x,ARE_train_list,'k',label='Train')
        plt.plot(x,ARE_test_list, '--c', label='Test')
        plt.xlabel("Number iterations CL")
        plt.ylabel("ARE")
        plt.legend()
        plt.show()
    
    
        
    
    
    
  
if __name__ == "__main__":    
    #sin_or_square = 'square'
           
    n_hidden_nodes=11
    
    use_cl=True

    std = 1
    
    
    train_RBF_network_op2(n_hidden_nodes, use_cl, std, plot_results=False, normalize_training_data=False)
    
    
    
    #mu_range = [0, round(2*math.pi,1)]
    #mu_range = [-5, 5]
    
    #mu_vec_init = np.array([-0.04844062,  0.43508019, -1.43848543,  0.52651128,  0.63813337,
    #    1.22463682, -0.38844132, -0.00855284,  1.36392453,  0.3646959 ,
    #   -0.36336789])#,  0.43839091])
    
    
    # Random mu between 0 and 2pi
    #mu_vec_init = np.random.uniform(low=0, high=2*math.pi, size=(rbf.node_count,))    
    #mu_vec_init = np.array([4.46506106, 0.07038114, 0.49323453, 3.47258945, 2.58994933,3.83851232, 0.71430821, 3.77674251, 4.19862989, 1.77314162, 4.7771036 ])
    # np.random.randn(rbf.node_count)
    
    #train_RBF_network(n_hidden_nodes, use_cl, std = 0.8, epochs_CL=100, plot_results=False)
        
    #train_RBF_network(n_hidden_nodes, use_cl, std = 0.5, epochs_CL=1000, mu_range=mu_range, plot_results=True,do_linespace=False, mu_vec_init=mu_vec_init)

    
    #train_RBF_network(n_hidden_nodes, use_cl=False, std = 0.5,  mu_range=mu_range, plot_results=False,do_linespace=True)
    """
    ARE_train_list = [] 
    ARE_test_list =  []
    
    # No Competitive Learning
    ARE_train, ARE_test = train_RBF_network(n_hidden_nodes, use_cl=False, std = 0.5, plot_results=False)
    
    ARE_train_list.append(ARE_train)
    ARE_test_list.append(ARE_test)
    
    
    ################################################
    ## GRID SEARCH NUMBER OF EPOCHS CL 
    grid_search_epochs_CL = [20, 50, 100, 500, 1000, 5000, 8000, 10000, 20000, 40000, 50000, 100000]
    
    for epochs_CL in grid_search_epochs_CL:
        print("Number of epochs: " + str(epochs_CL))
        
        ARE_train,ARE_test = train_RBF_network(n_hidden_nodes, use_cl, std = 0.5, epochs_CL=epochs_CL, plot_results=False)
        
        ARE_train_list.append(ARE_train)
        ARE_test_list.append(ARE_test)
    
    x = range(len(ARE_train_list))
    plt.plot(x,ARE_train_list,'k',label='Train')
    plt.plot(x,ARE_test_list, '--c', label='Test')
    plt.xlabel("Number epochs CL")
    plt.ylabel("ARE")
    plt.legend()
    plt.show()
    ################################################"""