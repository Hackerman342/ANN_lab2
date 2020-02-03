# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:00:11 2020

@author: matte
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


def SOM_train(dataset, n_nodes=100, n_epochs = 20, std_w=1, initial_neighbourhood_size=50, circular_offset=False, eta=0.2 ):
    ROWS = dataset.shape[0]
    COL = dataset.shape[1]
    W = np.random.normal(0, std_w, (n_nodes, COL))
    for e in range(n_epochs):
        for sample in range(ROWS):
            distances = np.sum(np.square(W - dataset[sample, :]), axis = 1)
            nearest_idx = np.argmin(distances) 
            
            offset = initial_neighbourhood_size - int(np.round((e+1)/n_epochs*(initial_neighbourhood_size)))
            print(offset)
            neighbours_indexes = get_neighbours_index(nearest_idx, offset, n_nodes, circular_offset=circular_offset)
            W[neighbours_indexes, :] += eta*(dataset[sample,:] - W[neighbours_indexes,:])
    return W
  