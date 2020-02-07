# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:05:56 2020

"""
# Import package dependencies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
import sys
from RBF_functions import RadialBasisFunctions
import time 

# Import other project scripts/functions
#from lab1 import generate_data
#from lab1_part2 import remove_points_class, remove_points_classA

np.random.seed(123)

class TwoLayer():

    def __init__(self):

        # Define parameters
        self.h_nodes = 5 # Number of nodes in hidden layer (h)
        self.l_rate = .01 # Learning rate
        self.epochs = 10000 # Number of epochs
        self.printstep = 500 # number of epochs between plots
        self.alpha = .9 # For momentum weight update | DO NOT SET = 1

        # Percentage of training data to remove - Ignored for autoencoder
        self.percent_remove_classA = 0
        self.percent_remove_classB = 0
        self.percent_remove_gauss = 50

        # Set ONLY one of the following modes to True
        self.classifier = False # Binary classification
        self.autoencoder = False # Auto-encoder
        self.function_approx = True # Function approximation

        # Ensure one and only one mode was selected
        if self.classifier and (self.autoencoder or self.function_approx):
            sys.exit("Only select ONE mode to execute: Classification, Encoder, or Function Approximation")
        if self.autoencoder and self.function_approx:
            sys.exit("Only select ONE mode to execute: Classification, Encoder, or Function Approximation")
        if not(self.classifier or self.autoencoder or self.function_approx):
            sys.exit("Must select ONE mode to execute: Classification, Encoder, or Function Approximation")



        # Set mode-specific parameters
        if self.classifier:
            self.linsep = False # linearly separable(True) or non-lin-sep(False)
            self.sequential = True # True for sequential updates (batch otherwise)
            self.n = 100 # number of samples per class
            self.mA = [1.0,0.3]
            self.mB = [0.0, -0.1]
            self.sigmaA = 0.2
            self.sigmaB = 0.3

        if self.autoencoder:
            self.vars = 8 # Number of variables in encoded string
            # This is the same as the number of strings (i.e. 8x8 input)
            self.h_nodes = 3 # Number of nodes in hidden layer (h)

        if self.function_approx:
            self.res = 21 # Resolution (x,y step size) for 3d gaussian data



    def generate_3d_gauss(self): # Generate and plot 3d Gaussian data
        # Unpack necessary class variables
        res = self.res

        # Build true function
        x = np.linspace(-5,5, res)
        y = np.linspace(-5,5, res)
        xx, yy = np.meshgrid(x,y)
        z = np.exp(-0.1*(np.multiply(xx,xx)+np.multiply(yy,yy)))-0.5

        # Plot true function
        print("\n True surface plot")
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xx, yy, z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

        # Reshape (flatten) for proper format
        patterns = np.concatenate((xx.reshape(1,-1), yy.reshape(1,-1)), axis=0)
        targets = z.reshape(1,-1)

        # Return grids (patterns) and true function values (targets)
        return patterns, targets

    def split_gauss_data(self, patterns, targets):

        # patterns are ordered and classes are separate when split_data is called
        n = patterns.shape[1]
        # Number of samples per class to use in training
        train_count = n * (100-self.percent_remove_gauss)//100
        # Generate random indices of training samples (without replacement)
        train_ids = np.sort(np.random.choice(n, train_count, replace = False))

        # Pull input samples for training data
        train_patterns = patterns[:, train_ids]

        # Pull output samples for training data
        train_targets = targets[:, train_ids]

        return  train_patterns, train_targets




    def generate_encode_data(self):

        # Generate all possible digits
        patterns = 2*np.eye(self.vars) - np.ones((self.vars,self.vars))
        targets = patterns
        self.enc_pos = np.array(range(self.vars))

        # Old encoder function for generating many input samples

        # patterns = -np.ones((self.vars,self.n))
        # self.enc_pos = np.random.randint(0, self.vars, self.n)
        # for i in range(self.n):
        #     patterns[self.enc_pos[i],i] = 1
        # targets = patterns
        # print(targets)

        return patterns, targets




    def generate_sep_data(self):

        # Unpack necessary class variables
        n = self.n
        mA = self.mA
        mB = self.mB
        sigmaA = self.sigmaA
        sigmaB = self.sigmaB


        x_dim = len(mA)
        # Patterns contains (x,y) coordinates - input
        patterns_classA = np.zeros((x_dim,n))
        patterns_classB = np.zeros((x_dim,n))
        # Targets contains correct classification
        # Class A has as target -1 and class B has target 1.
        targets_classA = -np.ones(n)
        targets_classB = np.ones(n)

        # Build patterns for both classes
        for i in range(x_dim):
            patterns_classB[i] = np.random.randn(1, n) * sigmaB + mB[i]
            if i == 0 and not self.linsep:
                patterns_classA[i][0:round(.5*n)] = np.random.randn(1, round(.5*n)) *sigmaA - mA[i]
                patterns_classA[i][round(.5*n):] = np.random.randn(1, n-round(.5*n)) *sigmaA + mA[i]
            else:
                patterns_classA[i] = np.random.randn(1, n) * sigmaA + mA[i]

        # Call split_data to separate data into training and validation sets
        X_train, X_test, Y_train, Y_test = self.split_data(patterns_classA, targets_classA, patterns_classB, targets_classB)


        # Combine patterns and targets to shuffle without losing correlations
        # Random shuffling is important for the sequential learning
        merged_train = np.concatenate((X_train, Y_train.reshape(1,-1)), axis=0)
        rand_idx_cols = np.random.permutation(merged_train.shape[1])
        merged_train_shuffled = merged_train[:,rand_idx_cols]
        X_train_shuffled = merged_train_shuffled[:-1]
        Y_train_shuffled = merged_train_shuffled[-1]

        merged_test = np.concatenate((X_test, Y_test.reshape(1,-1)), axis=0)
        rand_idx_cols2 = np.random.permutation(merged_test.shape[1])
        merged_test_shuffled = merged_test[:,rand_idx_cols2]
        X_test_shuffled = merged_test_shuffled[:-1]
        Y_test_shuffled = merged_test_shuffled[-1]

        # Plot classified data
        X = np.concatenate((X_train_shuffled,X_test_shuffled), axis=1)
        Y = np.concatenate((Y_train_shuffled,Y_test_shuffled), axis=0)
        print("\n Classified input data")
        plt.scatter(X[0],X[1], marker='x', c=Y)
        plt.show()

        return X_train_shuffled, X_test_shuffled, Y_train_shuffled, Y_test_shuffled

    def split_data(self, patterns_A, targets_A, patterns_B, targets_B):
        # patterns are ordered and classes are separate when split_data is called
        n_A = patterns_A.shape[1]
        n_B = patterns_B.shape[1]

        # Number of samples per class to use in training
        train_A_count = n_A * (100-self.percent_remove_classA)//100
        train_B_count = n_B * (100-self.percent_remove_classB)//100
        # Generate random indices of training samples (without replacement)
        train_ids_classA = np.sort(np.random.choice(n_A, train_A_count, replace = False))
        train_ids_classB = np.sort(np.random.choice(n_B, train_B_count, replace = False))
        # Remaining indices are for test (validation) set
        test_ids_classA = np.arange(0,n_A)[np.isin(np.arange(0,n_A),train_ids_classA,invert=True)]
        test_ids_classB = np.arange(0,n_B)[np.isin(np.arange(0,n_B),train_ids_classB,invert=True)]

        # Pull and merge input samples for training data
        train_X_classA = patterns_A[:, train_ids_classA]
        train_X_classB = patterns_B[:, train_ids_classB]
        train_patterns = np.concatenate((train_X_classA, train_X_classB), axis=1)
        # Pull and merge input samples for test data
        test_X_classA = patterns_A[:, test_ids_classA]
        test_X_classB = patterns_B[:, test_ids_classB]
        test_patterns = np.concatenate((test_X_classA, test_X_classB), axis=1)

        # Pull and merge output samples for training data
        train_Y_classA = targets_A[train_ids_classA]
        train_Y_classB = targets_B[train_ids_classB]
        train_targets = np.concatenate((train_Y_classA, train_Y_classB), axis=0)
        # Pull and merge output samples for test data
        test_Y_classA = targets_A[test_ids_classA]
        test_Y_classB = targets_B[test_ids_classB]
        test_targets = np.concatenate((test_Y_classA, test_Y_classB), axis=0)

        return  train_patterns, test_patterns, train_targets, test_targets


    def centered_sigmoid(self, array):
        return (2/(1 + np.exp(-array)) - 1)

    def classif_plot(self, X, out, i):

        # Print performance metrics
        print("\n ------------------- \n")
        print("Training Data")
        print("epoch: ", i+1)
        print("Mean square error: ", "{0:.5f}".format(self.MSE[i]))
        print("Misclassified points: ", self.misclass[i].astype(int))
        print("Misclassified percentage: %", 100*self.misclass[i]/X.shape[1])

        # Define binary vector for classification
        col = out.reshape(out.size,)
        col[col<0] = 0
        col[col>0] = 1

        # Plot classified points
        plt.scatter(X[0],X[1], marker='x', c=col)
        plt.show()

    def classif_test_plot(self, X, out, i):

        # Print performance metrics
        print("\n ------------------- \n")
        print("Validation Data")
        print("epoch: ", i+1)
        print("Mean square error: ", "{0:.5f}".format(self.MSE_test[i]))
        print("Misclassified points: ", self.misclass_test[i].astype(int))
        print("Misclassified percentage: %", 100*self.misclass_test[i]/X.shape[1])
        # Define binary vector for classification
        col = out.reshape(out.size,)
        col[col<0] = 0
        col[col>0] = 1

        # Plot classified points
        plt.scatter(X[0],X[1], marker='x', c=col)
        plt.show()


    def encode_plot(self, X, out, i):

        # Print performance metrics
        print("\n ------------------- \n")
        print("epoch: ", i+1)
        print("Mean square error: ", "{0:.5f}".format(self.MSE[i]))
        print("Misclassified points: ", self.misclass[i].astype(int))

        # Define binary vector for classification
        #col = out.reshape(out.size,)
        #col[col<0] = 0
        #col[col>0] = 1

        # Plot classified points
        #plt.scatter(X[0],X[1], marker='x', c=col)
        #plt.show()

    def func_plot(self, X, out, i):

        # Print performance metrics
        print("\n ------------------- \n")
        print("epoch: ", i+1)
        print("Mean square error: ", "{0:.5f}".format(self.MSE_test[i]))



        # Plot estimated surface from current model
        print("Modeled surface plot")
        
        plt.figure()
        #print(X[0, :])
        #print(out)

        X = np.array(X)
        inp = X[0].reshape(1,-1)
        out = np.array(out).reshape(1,-1)
        plt.plot(X[0], out[0])
        
        #plt.set_xlabel('x')
        #ax.set_ylabel('y')
        plt.show()

    def func_plot_test_train(self, X, out, i):

        # Print performance metrics
        print("\n ------------------- \n")
        print("epoch: ", i+1)
        print("Mean square error: ", "{0:.5f}".format(self.MSE[i]))

        # Reshape arrays for plotting
        xx = X[0].reshape(round(sqrt(out.size)),round(sqrt(out.size)))
        yy = X[1].reshape(round(sqrt(out.size)),round(sqrt(out.size)))
        zz = out.reshape(round(sqrt(out.size)),round(sqrt(out.size)))

        # Plot estimated surface from current model
        print("Modeled surface plot")
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xx,yy, zz, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def two_layer_network(self, X, Y):

        # Unpack necessary class variables
        h_nodes = self.h_nodes
        l_rate = self.l_rate
        epochs = self.epochs
        alpha = self.alpha
        printstep = self.printstep

        # Determine number of input parametrs (x) and samples (N)
        if X.shape[0] == X.size: # X is 1-dimensional
            x_count = 1
            N = X.shape[0]
        else: # X is multi-dimensional
            x_count = X.shape[0]
            N = X.shape[1]

        # Determine number of output parametrs (y)
        if Y.shape[0] == Y.size: # Y is 1-dimensional
            y_count = 1
        else: # Y is multi-dimensional
            y_count = Y.shape[0]

        # Add bias row to X
        X = np.concatenate((X, np.ones(N).reshape(1,-1)), axis=0)

        # Initialize weight matrices
        V = np.random.randn(h_nodes, x_count+1)
        W = np.random.randn(y_count, h_nodes+1)
        if self.autoencoder:
            V /= self.vars # Divide weight matrices by number of input variables
            W /= self.vars

        # Initialize weight matrix update values
        dv = np.zeros(V.shape)
        dw = np.zeros(W.shape)

        # Initialize performance metric vectors
        self.MSE = np.zeros(epochs)
        if not self.function_approx:
            # Do not track misclassiifed points for function mode
            self.misclass = np.zeros(epochs)
        
        start_time = time.time()

        # Run optimization
        for i in range(epochs):

            if self.classifier and self.sequential:
                out = np.zeros(Y.shape)
                for j in range(N):
                    xj = X[:,j]
                    if y_count > 1:
                        yj = Y[:,j]
                    else:
                        yj = Y[j]


                    # Forward pass
                    # Layer 1
                    hin = np.dot(V,xj)
                    hout = self.centered_sigmoid(hin)
                    # Add bias row to hout
                    hout = np.append(hout, 1)
                    # Layer 2
                    oin = np.dot(W,hout)
                    out[j] = self.centered_sigmoid(oin)

                    # Backward pass
                    if y_count > 1:
                        delta_o = 0.5*np.multiply(out[j] - yj, np.multiply(1+out[j], 1-out[j]))
                    else:
                        delta_o = 0.5*np.multiply(out[j] - yj, np.multiply(1+out[j], 1-out[j]))
                    delta_h = 0.5*np.multiply(np.dot(W.T,delta_o),
                                              np.multiply(1+hout.reshape(-1,1), 1-hout.reshape(-1,1)))

                    # Calculate updates for weight matrices
                    dv = dv*alpha - np.dot(delta_h[:-1],X[:,j].reshape(1,-1))*(1-alpha)
                    # ignore bias row of delta_h with [:-1,:]
                    dw = dw*alpha - np.dot(delta_o.reshape(-1,1),hout.reshape(1,-1))*(1-alpha)

                    # Apply updates
                    V += dv*l_rate
                    W += dw*l_rate

            else: # Batch update

                # Forward pass
                # Layer 1
                hin = np.dot(V,X)
                hout = self.centered_sigmoid(hin)
                # Add bias row to hout
                hout = np.concatenate((hout, np.ones(N).reshape(1,-1)), axis=0)
                # Layer 2
                oin = np.dot(W,hout)
                out = self.centered_sigmoid(oin)

                # Backward pass
                delta_o = 0.5*np.multiply(out - Y, np.multiply(1+out, 1-out))
                delta_h = 0.5*np.multiply(np.dot(W.T,delta_o),
                                          np.multiply(1+hout, 1-hout))

                # Calculate updates for weight matrices
                dv = dv*alpha - np.dot(delta_h[:-1,:],X.T)*(1-alpha)
                # ignore bias row of delta_h with [:-1,:]
                dw = dw*alpha - np.dot(delta_o,hout.T)*(1-alpha)

                # Apply updates
                V += dv*l_rate
                W += dw*l_rate

            # Update performance metrics
            self.MSE[i] = np.mean(np.abs(Y-out))
            if self.classifier:
                self.misclass[i] = sum(np.multiply(Y,out.reshape(out.size,))<0)
            if self.autoencoder:
                guess = np.argmax(out,axis=0)
                self.misclass[i] = np.sum(self.enc_pos != guess)
#                print("guess: \n", out)
#                print("out: \n", guess)
#                print("pos: \n", self.enc_pos)

                #print(np.round(out))
                #self.misclass[i] =
            # Plot estimate of function and print metrics every printstep
            if (i+1)%printstep == 0:
                if self.classifier:
                    self.classif_plot(X, out, i)
                if self.autoencoder:
                    self.encode_plot(X, out, i)
                if self.function_approx:
                    self.func_plot(X, out, i)

        print("--- %s seconds ---" % (time.time() - start_time))

        # Final results plots
        plt.plot(self.MSE)
        plt.title("Mean Square error")
        plt.xlabel("epoch")
        plt.show()
        if not self.function_approx:
            plt.plot(self.misclass)
            plt.title("Misclassified points")
            plt.xlabel("epoch")
            plt.show()
        if self.autoencoder:
            print("Unique encoding (pseudo-binary representation):")
            print(np.sign(hout[:-1]))
            print("Error: \n", np.round(Y-out,1))

    def two_layer_train_test_network(self, X_train, X_test, Y_train, Y_test):
        # Call train data X & Y to avoid changing lines
        X = X_train
        Y = Y_train


        # Unpack necessary class variables
        h_nodes = self.h_nodes
        l_rate = self.l_rate
        epochs = self.epochs
        alpha = self.alpha
        printstep = self.printstep

        # Determine number of input parametrs (x) and samples (N)
        if X.shape[0] == X.size: # X is 1-dimensional
            x_count = 1
            N = X.shape[0]
            N_test = X_test.shape[0]
        else: # X is multi-dimensional
            x_count = X.shape[0]
            N = X.shape[1]
            N_test = X_test.shape[1]

        # Determine number of output parametrs (y)
        if Y.shape[0] == Y.size: # Y is 1-dimensional
            y_count = 1
        else: # Y is multi-dimensional
            y_count = Y.shape[0]

        # Add bias row to X
        #print(N)
        #print(X.shape)
        X = np.concatenate((X, np.ones(N).reshape(1,-1)), axis=0)
        # Add bias row to X_test
        X_test = np.concatenate((X_test, np.ones(N_test).reshape(1,-1)), axis=0)

        # Initialize weight matrices
        V = np.random.randn(h_nodes, x_count+1)
        W = np.random.randn(y_count, h_nodes+1)
        if self.autoencoder:
            V /= self.vars # Divide weight matrices by number of input variables
            W /= self.vars

        # Initialize weight matrix update values
        dv = np.zeros(V.shape)
        dw = np.zeros(W.shape)

        # Initialize performance metric vectors
        self.MSE = np.zeros(epochs)
        self.MSE_test = np.zeros(epochs)
        if not self.function_approx:
            # Do not track misclassiifed points for function mode
            self.misclass = np.zeros(epochs)
            self.misclass_test = np.zeros(epochs)

        start_time = time.time()
     
        # Run optimization
        for i in range(epochs):

            if self.classifier and self.sequential:
                out = np.zeros(Y.shape)
                for j in range(N):
                    xj = X[:,j]
                    if y_count > 1:
                        yj = Y[:,j]
                    else:
                        yj = Y[j]


                    # Forward pass
                    # Layer 1
                    hin = np.dot(V,xj)
                    hout = self.centered_sigmoid(hin)
                    # Add bias row to hout
                    hout = np.append(hout, 1)
                    # Layer 2
                    oin = np.dot(W,hout)
                    out[j] = self.centered_sigmoid(oin)

                    # Backward pass
                    if y_count > 1:
                        delta_o = 0.5*np.multiply(out[j] - yj, np.multiply(1+out[j], 1-out[j]))
                    else:
                        delta_o = 0.5*np.multiply(out[j] - yj, np.multiply(1+out[j], 1-out[j]))
                    delta_h = 0.5*np.multiply(np.dot(W.T,delta_o),
                                              np.multiply(1+hout.reshape(-1,1), 1-hout.reshape(-1,1)))

                    # Calculate updates for weight matrices
                    dv = dv*alpha - np.dot(delta_h[:-1],X[:,j].reshape(1,-1))*(1-alpha)
                    # ignore bias row of delta_h with [:-1,:]
                    dw = dw*alpha - np.dot(delta_o.reshape(-1,1),hout.reshape(1,-1))*(1-alpha)

                    # Apply updates
                    V += dv*l_rate
                    W += dw*l_rate

            else: # Batch update
                # Forward pass
                # Layer 1
                hin = np.dot(V,X)
                hout = self.centered_sigmoid(hin)
                # Add bias row to hout
                hout = np.concatenate((hout, np.ones(N).reshape(1,-1)), axis=0)
                # Layer 2
                oin = np.dot(W,hout)
                out = self.centered_sigmoid(oin)

                # Backward pass
                delta_o = 0.5*np.multiply(out - Y, np.multiply(1+out, 1-out))
                delta_h = 0.5*np.multiply(np.dot(W.T,delta_o),
                                          np.multiply(1+hout, 1-hout))

                # Calculate updates for weight matrices
                dv = dv*alpha - np.dot(delta_h[:-1,:],X.T)*(1-alpha)
                # ignore bias row of delta_h with [:-1,:]
                dw = dw*alpha - np.dot(delta_o,hout.T)*(1-alpha)

                # Apply updates
                V += dv*l_rate
                W += dw*l_rate

            # Update training performance metrics
            self.MSE[i] = np.mean(np.abs(Y-out))
            hin_test = np.dot(V,X_test)
            hout_test = self.centered_sigmoid(hin_test)
            # Add bias row to hout
            hout_test = np.concatenate((hout_test, np.ones(N_test).reshape(1,-1)), axis=0)
            # Layer 2
            oin_test = np.dot(W,hout_test)
            out_test = self.centered_sigmoid(oin_test)
            
            #self.MSE_test[i] = np.mean(np.square(Y_test-out_test))
            self.MSE_test[i] = np.mean(np.abs(Y_test-out_test))
            if self.classifier:
                self.misclass[i] = sum(np.multiply(Y,out.reshape(out.size,))<0)


            
        print("--- %s seconds ---" % (time.time() - start_time))

    
        hin_test = np.dot(V,X_test)
        hout_test = self.centered_sigmoid(hin_test)
        # Add bias row to hout
        hout_test = np.concatenate((hout_test, np.ones(N_test).reshape(1,-1)), axis=0)
        # Layer 2
        oin_test = np.dot(W,hout_test)
        out_test = self.centered_sigmoid(oin_test)
        #print(out_test)
        if self.function_approx:
                    self.func_plot(X_test, out_test, i)
        # Final results plots
        plt.plot(self.MSE)
        plt.title("Training Mean Square error")
        plt.xlabel("epoch")
        plt.show()
        plt.plot(self.MSE_test)
        plt.title("Test Mean Absolute error")
        plt.xlabel("epoch")
        plt.show()
        if not self.function_approx:
            plt.plot(self.misclass)
            plt.title("Training Misclassified points")
            plt.xlabel("epoch")
            plt.show()
            plt.plot(self.misclass_test[self.misclass_test > 0])
            plt.title("Validation Misclassified points")
            plt.xlabel("epoch")
            plt.show()
        print("Train error: "+ str(self.MSE[-1]))
        print("Test error: "+ str(self.MSE_test[-1]))

if __name__ == "__main__":
    n_nodes = 50
    
    # Initialize class
    multi = TwoLayer()
    multi.h_nodes = n_nodes
    train_range = [0 , 2*math.pi]
    test_range = [0.05 , 2*math.pi + 0.05]
    
    rbf = RadialBasisFunctions(n_nodes)
    step = 0.1
    
    x_train, sin_train, square_train = rbf.generate_sin_and_square(train_range,step)
    x_test, sin_test, square_test = rbf.generate_sin_and_square(test_range,step)
    
    sin_train = rbf.add_gauss_noise(sin_train, 0, 0.1)
    square_train = rbf.add_gauss_noise(square_train, 0, 0.1)
    sin_test = rbf.add_gauss_noise(sin_test, 0, 0.1)
    square_test = rbf.add_gauss_noise(square_test, 0, 0.1)
    
    if multi.function_approx:
        # Run network
        #X_test, Y_test = multi.generate_3d_gauss()
        #X_train, Y_train = multi.split_gauss_data(X_test, Y_test)
        #print(np.asmatrix(x_train).shape)
        multi.two_layer_train_test_network(np.asmatrix(x_train), np.asmatrix(x_test), np.asmatrix(sin_train), np.asmatrix(sin_test))
        #X, Y = multi.generate_3d_gauss()
