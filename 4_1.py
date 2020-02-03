# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:56:24 2020

@author: matte
"""

import numpy as np
import SOM

def get_animals_dataset(n_animals = 32, n_dimensions = 84):
    f=open("data/animals.dat", "r")
    contents = f.read().split(',')
    props = np.zeros((n_animals, n_dimensions))
    for index in range(n_animals):
        props[index, :] = contents[index*n_dimensions : (index+1)*n_dimensions]
    return props


def get_name_animals():
    """
    @return list with the name of each animal
    """
    f=open("data/animalnames.txt", "r")
    file_lines = f.readlines()
    animal_names = [] 
    for name in file_lines:
        animal_names.append(name.replace("'", "").replace("\t","").replace("\n",""))
    
    return animal_names
  

def sort_animals(weights, animals_dataset):
    animals_names = get_name_animals()
    
    animals_indexes_dict = {}
    n_animals = animals_dataset.shape[0]
    for animal_index in range(n_animals):
        nearest_idx = SOM.get_best_matching_neuron(weights, animals_dataset[animal_index, :])
        animals_indexes_dict[animal_index] = nearest_idx
        
    sorted_animals_index = sorted(animals_indexes_dict.items(), key=lambda item: item[1])
    animals_sorted = []
    for animal_idx in sorted_animals_index:
        animals_sorted.append(animals_names[animal_idx[0]])
    return animals_sorted

    
if __name__ == "__main__":
    animals_dataset = get_animals_dataset()
    weights  = SOM.SOM_train(animals_dataset)
    print(sort_animals(weights, animals_dataset))