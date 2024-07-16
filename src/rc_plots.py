import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
sys.path.append('..\\src\\')
from read_graph import read_graph, k_core_weights, display_graph

def day_to_day_interactions(dirname, mnn = 4, deg = 3):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\social network matrices 3days\\"+dirname+"\\"
    fig, axs = plt.subplots(5, 1, figsize=(5, 10)) #putting all results directly on same figure
    for i in range(5):
        file = "interactions_resD3_"+str(i+1)+".csv"
        display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "k-core", deg = deg, layout = "circle", node_labels = [])#, idx = [1,1,1,0,0,0,1,1,1,0])
    fig.suptitle(dirname, fontsize=16)
    plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_interactions.png")
    plt.show()

def day_to_day_approach(dirname, mnn = 4, deg = 2):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\social network matrices 3days\\"+dirname+"\\"
    fig, axs = plt.subplots(5, 1, figsize=(5, 10)) #putting all results directly on same figure
    for i in range(5):
        try:
            file = "approaches_resD3_"+str(i+1)+".csv"
            display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "k-core", deg = deg, layout = "circle", node_labels = [])#, idx = [1,1,1,0,0,0,1,1,1,0])
        except:
            pass
    axs[0].set_title(dirname, fontsize=16)
    plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_approach.png")
    plt.show()


def day_to_day_approachprop(dirname, mnn = 4, deg = 2):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\social network matrices 3days\\"+dirname+"\\"
    fig, axs = plt.subplots(5, 1, figsize=(5, 10)) #putting all results directly on same figure
    for i in range(5):
        try:
            file = "approach_prop_resD3_"+str(i+1)+".csv"
            display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "k-core", deg = deg, layout = "circle", node_labels = [])#, idx = [1,1,1,0,0,0,1,1,1,0])
        except:
            pass
    axs[0].set_title(dirname, fontsize=16)
    plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_approachprop.png")
    plt.show()

for graph in ["G1"]:#, "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10"]:
    day_to_day_approachprop(graph, 5, 1)