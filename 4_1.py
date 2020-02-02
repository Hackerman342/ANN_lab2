# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:56:24 2020

@author: matte
"""

import numpy as np
from SOM import SOM_train

def get_animals_dataset(N_ANIMALS = 32, COL = 84):
    f=open("data/animals.dat", "r")
    contents = f.read().split(',')
    props = np.zeros((N_ANIMALS, COL))
    for index in range(N_ANIMALS):
        props[index, :] = contents[index*COL : (index+1)*COL]
    return props

def get_name_animals():
    f=open("data/animalnames.txt", "r")
    file_lines = f.readlines()
    animal_names = [] 
    for name in file_lines:
        animal_names.append(name.replace("'", ""))
    return animal_names
  
def sort_animals(weights, props):
    animals_names = get_name_animals()
    
    animals_indexes_dict = {}
    N_ANIMALS = props.shape[0]
    for animal_index in range(N_ANIMALS):
        distances = np.sum(np.square(weights - props[animal_index, :]), axis = 1)
        nearest_idx = np.argmin(distances) 
        animals_indexes_dict[animal_index] = nearest_idx
    sorted_animals_index = sorted(animals_indexes_dict.items(), key=lambda item: item[1])
    animals_sorted = []
    for animal_idx in sorted_animals_index:
        animals_sorted.append(animals_names[animal_idx[0]])
    return animals_sorted
    
if __name__ == "__main__":
    animals_dataset = get_animals_dataset()
    W  = SOM_train(animals_dataset)
    print(sort_animals(W, animals_dataset))