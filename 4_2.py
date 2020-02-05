# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:04:52 2020

@author: matte
"""
import numpy as np
from SOM import SOM_train
import matplotlib.pyplot as plt
import SOM

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

def sort_cities(weights, cities_dataset):
    
    cities_indexes_dict = {}
    n_cities = cities_dataset.shape[0]
    for city_index in range(n_cities):
        nearest_idx = SOM.get_best_matching_neuron(weights, cities_dataset[city_index, :])
        cities_indexes_dict[city_index] = nearest_idx
        
    sorted_cities_index = dict(sorted(cities_indexes_dict.items(), key=lambda item: item[1]))
    return sorted_cities_index.keys()

def plot_path(dataset, ordered_cities):
    coords = []
    for city_index in ordered_cities:
        coords.append(dataset[city_index])
    coords.append(coords[0])
    coords = np.asarray(coords)
    plt.title("Minimum path among cities")
    plt.scatter(dataset[:,0], dataset[:,1], marker='x', color='red')
    plt.plot(coords[:,0],coords[:,1], color='blue')

if __name__ == "__main__":
    cities_dataset = get_cities_dataset()
    W = SOM_train(cities_dataset, n_epochs=20, learning_rate=0.2, exp_descrease=False, alpha=1, circular_offset=True, n_nodes=10, initial_neighbourhood_size=2)
    sorted_cities = sort_cities(W, cities_dataset)
    plot_path(cities_dataset, sorted_cities)
