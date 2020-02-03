# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:00:11 2020

@author: matteo
"""
import numpy as np

def get_neighbours_index(index, offset, n_nodes, circular_offset=False):
    if circular_offset:
        neighbours_indexes = []
        lower_neighbour = (index - offset)%n_nodes
        n_steps = offset*2 + 1
        for index in range(n_steps):
            neighbours_indexes.append((index+lower_neighbour)%n_nodes)
        return neighbours_indexes
            
    else:
        lower_neighbour = (index - offset) if (index - offset) > 0 else 0
        upper_neighbour = (index + offset) if (index + offset) < n_nodes else n_nodes-1
        neighbours_indexes = np.arange(lower_neighbour, upper_neighbour + 1, 1)
        return neighbours_indexes


def get_best_matching_neuron(weights, sample):
    distances = np.sum(np.square(weights - sample), axis = 1)
    best_matching_neuron = np.argmin(distances) 
    return best_matching_neuron


def SOM_train(dataset, n_nodes=100, n_epochs = 20, initial_neighbourhood_size=50, circular_offset=False, learning_rate=0.2):
    dimension_dataset = dataset.shape[1]
    
    # Initialize weights
    weights = np.random.uniform(size=(n_nodes, dimension_dataset))

    for epoch in range(n_epochs):
        for sample in dataset:
            best_matching_neuron = get_best_matching_neuron(weights, sample)
            
            offset = initial_neighbourhood_size - int(np.round((epoch+1)/n_epochs*(initial_neighbourhood_size)))
            print(offset)
            
            neighbours_indexes = get_neighbours_index(best_matching_neuron, offset, n_nodes, circular_offset=circular_offset)
            weights[neighbours_indexes, :] += learning_rate*(sample - weights[neighbours_indexes,:])
    
    return weights
  