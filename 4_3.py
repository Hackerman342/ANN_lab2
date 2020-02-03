#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:30:59 2020

@author: flaviagv
"""

import numpy as np
import SOM
import matplotlib.pyplot as plt 


plt.rc('grid', linestyle="dotted", color='grey')


def get_votes_data(n_samples=349, n_dimensions=31):
    f=open("data/votes.dat", "r")
    contents = f.read().split(',')
    props = np.zeros((n_samples, n_dimensions))
    for index in range(n_samples):
        props[index, :] = contents[index*n_dimensions : (index+1)*n_dimensions]
    return props




def get_extra_data(file_name):
    f=open(file_name, "r")
    data = f.read().split('\n\t')
    data_clean = [int(text.strip()) for text in data]
    return np.array(data_clean)


    

def get_gender_node_grid(MPs_that_neuron, gender_data):
    gender_data_node = gender_data[MPs_that_neuron]
    # get the most frequent gender in that node
    most_freq_gender = np.bincount(gender_data_node).argmax()
    return most_freq_gender
    
    
def get_district_node_grid():
    return 0 
    
    
    
def get_party_node_grid():
    return 0




def plot_grid__________(votes_best_neuron_arr, n_rows_grid=10, get_gender=False, get_district=False, get_party=False, plot_legend=False):
    
    
    gender_data = get_extra_data("data/mpsex.dat")
    district_data = get_extra_data("data/mpdistrict.dat")
    party_data = get_extra_data("data/mpparty.dat") 
    
    coordinates_scatter = np.zeros((n_rows_grid*n_rows_grid,2))
    class_list = []
    
    ## set color for each class depending in the dataset
    """
    if get_gender:
    
    elif get_district:
            
    elif get_party:
    """  
        
    # iterate over the best neuron indexes 
    for best_neuron_idx in range(n_rows_grid*n_rows_grid):
        #for best_neuron in len(votes_best_neuron_arr):
        coord_x = best_neuron_idx%n_rows_grid
        coord_y = best_neuron_idx//n_rows_grid
        
        
        MPs_that_neuron = np.argwhere(votes_best_neuron_arr == best_neuron_idx).reshape(-1,)
        
        # If that node identifies a sample we plot it, if not we don't 
        if len(MPs_that_neuron!=0):
            coordinates_scatter[best_neuron_idx,0] = coord_x
            coordinates_scatter[best_neuron_idx,1] = coord_y
        
            if get_gender:
                most_frequent = get_gender_node_grid(MPs_that_neuron, gender_data)
            #elif get_district:
                
            #elif get_party:
        class_list.append(most_frequent)
    
    # plot most_frequent in (coord_x, coord_y)
    sc = plt.scatter(coordinates_scatter[:,0], coordinates_scatter[:,1], marker = 'o', c=class_list, cmap="bwr_r")
    plt.grid()
    if plot_legend:
        #plot legend
        plt.colorbar(sc)
    plt.title("SOM nodes grid based on gender")
    plt.show()

    return 0 





   
if __name__ == "__main__":

    votes_data = get_votes_data()
    
    
    weights  = SOM.SOM_train(votes_data)
    n_samples = votes_data.shape[0]
    votes_best_neuron_arr = -np.ones(n_samples)
    
    for sample_idx in range(n_samples):
        nearest_idx = SOM.get_best_matching_neuron(weights, votes_data[sample_idx, :])
        votes_best_neuron_arr[sample_idx] = nearest_idx
    
    



         
        
    
    