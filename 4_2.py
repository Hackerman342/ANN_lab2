# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:04:52 2020

@author: matte
"""
import numpy as np
from SOM import SOM_train
import matplotlib.pyplot as plt

def get_cities_dataset(N_CITIES=10, COL=2):
    f=open("data/cities.dat", "r")
    file_lines = f.readlines()
    cities_coord = np.zeros((N_CITIES, COL))
    index = 0
    for city in file_lines:
        city_coords = city.replace('\n', '').replace(';', '').split(',')
        cities_coord[index, 0] = float(city_coords[0])
        cities_coord[index, 1] = float(city_coords[1])
        index += 1
    return cities_coord

def plot_path(dataset, W):
    coords = []
    for city in range(dataset.shape[0]):
        distances = np.sum(np.square(W - dataset[city, :]), axis = 1)
        nearest_idx = np.argmin(distances) 
        coords.append(W[nearest_idx,:])
    coords.append(coords[0])
    coords = np.asarray(coords)
    
    plt.scatter(dataset[:,0], dataset[:,1], marker='x', color='red')
    plt.plot(coords[:,0],coords[:,1], color='blue')

if __name__ == "__main__":
    cities_dataset = get_cities_dataset()
    W = SOM_train(cities_dataset, n_epochs=100, learning_rate=0.6, exp_descrease=True, alpha=0.8, circular_offset=True, n_nodes=10, initial_neighbourhood_size=2)
    plot_path(cities_dataset, W)