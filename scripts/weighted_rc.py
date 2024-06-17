import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
from tqdm import tqdm
from copy import deepcopy

def mnn_cut(arr, nn = 2):
    """
    Cuts the edges of the graph in the input array by keeping only the edges that are mutual neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
    assert nn > 1, "nn should be bigger than 1."
    nn -= 1
    mnn_arr = np.zeros_like(arr)  
    neighbors_i = np.argsort(-arr, 1) #computing the nearest neighbors for local nn estimation
    neighbors_j = np.argsort(-arr, 0)
    for i in range(arr.shape[1]):
        for j in range(i, arr.shape[1]):
            if any(np.isin(neighbors_i[i, 0:nn], j)) and any(np.isin(neighbors_j[0:nn, j], i)): #local nearest distance estimation
                mnn_arr[i, j] += arr[i, j]
                mnn_arr[j, i] += arr[j, i]
    return mnn_arr


path = "..\\data\\averaged\\"
G = os.listdir(path)[0]
path_to_file = path+G+"\\interactions_resD7_1.csv"
arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
data = arr[1:, 1:].astype(float)
data = mnn_cut(data, 5)



def weighted_rich_club(data, k = 3, iterations = 100):
    graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    graph_core = graph.k_core(k)
    fig, ax = plt.subplots(1, 1)
    ig.plot(graph_core, target = ax)
    observed_richness = np.sum(graph_core.es["weight"])
    random_richness = []
    for i in tqdm(range(iterations)):
        ## shuffling array while keeping the symmetric structure of the array
        values = np.triu(data, 1)
        idx = np.triu_indices(data.shape[0], 1, data.shape[1])
        shuffled_values = values[idx]
        np.random.shuffle(shuffled_values)
        shuffled_data = np.zeros(data.shape)
        counter = 0
        for m in range(data.shape[0]):
            for n in range(m, data.shape[0]):
                if m != n:
                    shuffled_data[m, n] = shuffled_values[counter]
                    shuffled_data[n, m] = shuffled_values[counter]
                    counter += 1
        random_graph = ig.Graph.Weighted_Adjacency(shuffled_data, mode='undirected')
        random_core = random_graph.k_core(k)
        random_richness.append(np.sum(random_core.es["weight"]))
        
    normalization_factor = np.sum(np.array(random_richness))/iterations
    print(observed_richness)
    print(normalization_factor)
    return observed_richness/normalization_factor
    
rc = weighted_rich_club(data, 3, 1000)
    
        
    
