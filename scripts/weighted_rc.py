import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
from tqdm import tqdm
from copy import deepcopy


path = "..\\data\\averaged\\"
G = os.listdir(path)[0]
path_to_file = path+G+"\\interactions_resD7_1.csv"
arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
data = arr[1:, 1:].astype(float)


def weighted_rich_club(data, k = 3, iterations = 100):
    graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    graph_core = graph.k_core(k)
    observed_richness = np.sum(graph_core.es["weight"])
    random_richness = []
    for i in tqdm(range(iterations)):
        ## shuffling array while keeping the symmetric structure of the array
        values = np.triu(data)
        idx = values != 0
        shuffled_values = values[idx]
        np.random.shuffle(shuffled_values)
        shuffled_data = np.zeros(data.shape)
        for j in range(data.shape[0]):
            for k in range(j, data.shape[0]):
                shuffled_data[j, k] = shuffled_values[np.ravel_multi_index((j,k), data.shape)]
                shuffled_data[k, j] = shuffled_values[np.ravel_multi_index((j,k), data.shape)]

        random_graph= ig.Graph.Weighted_Adjacency(shuffled_data, mode='undirected')
        random_core = random_graph.k_core(k)
        random_richness.append(np.sum(random_core.es["weight"]))
    normalization_factor = np.sum(np.array(random_richness))
    return observed_richness/normalization_factor
    
weighted_rich_club(data)
    
        
    
