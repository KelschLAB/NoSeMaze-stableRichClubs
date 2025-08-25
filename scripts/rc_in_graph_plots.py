import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import pandas as pd
sys.path.append('..\\src\\')
from read_graph import read_graph, k_core_weights, display_graph, display_graph_3d
from utils import get_category_indices

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def graph_in_time_3d(graph_idx, var = "interactions", mnn = 3, show_mut = True):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"
    
    fig, ax = plt.subplots(1, 1, figsize=(2, 2)) #putting all results directly on same figure
    ax = fig.add_subplot(111, projection='3d')
    
    files = []
    for t in range(5):
        if os.path.exists(datapath+f"{var}_resD3_"+str(t+1)+".csv"):
            files.append(datapath+f"{var}_resD3_"+str(t+1)+".csv")
        else:
            print("missing day, ignoring")
            
    if show_mut:
        mutants_idx, _, _, _, RFIDs = get_category_indices(graph_idx, "interactions", 3)
        mutants = np.ones(len(RFIDs))
        mutants[mutants_idx] = 0
        display_graph_3d(files, ax, mnn = mnn, node_metric = "rich-club", deg = mnn,
                         layout = "circle", node_labels = None, node_size = 8, default_edge_width = 2, idx = mutants,
                         scale_edge_width = False, between_layer_edges = False)
        
    else:
        display_graph_3d(files, ax, mnn = mnn, node_metric = "rich-club", deg = mnn,
                         layout = "circle", node_labels = None, node_size = 8, default_edge_width = 2,
                         scale_edge_width = False, between_layer_edges = False)
    
    # fig.suptitle(labels[graph_idx], fontsize=12)
    # ax.set_frame_on(False)
    plt.tight_layout()
    plt.show()

def graph_in_time_2d(graph_idx, var = "interactions", mnn = 3, deg = 3, mutual = True, show_mut = True, rm_weak = False):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"
    fig, axs = plt.subplots(5, 1, figsize=(2, 7)) #putting all results directly on same figure
    for i in range(5):
        try:
            file = f"{var}_resD3_"+str(i+1)+".csv"
            if show_mut:
                mutants_idx, _, _, _, RFIDs = get_category_indices(graph_idx, "interactions", 3, rm_weak_histo=rm_weak)
                mutants = np.ones(len(RFIDs))
                mutants[mutants_idx] = 0
                display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "rich-club", deg = deg, mutual = mutual, layout = "circle",
                              node_size = 12, node_labels = None, idx = mutants, edge_width = 1.5, scale_edge_width = False)
            else:
                display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "rich-club", deg = deg, mutual = mutual,
                              layout = "circle", node_size = 12, node_labels = None, edge_width = 1.5, scale_edge_width = False)
        except:
            pass
    try:
        data = read_graph([datapath+file], return_ig = False)
        total = np.sum(data[0].flatten())
    except:
        total = np.nan
    plt.tight_layout()
    # axs[0].set_title(labels[graph_idx]+", total approaches: "+str(total), fontsize=16)
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
            # file = "approach_prop_resD3_1.csv"
            display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "k-core", deg = deg, layout = "circle", node_labels = [])#, idx = [1,1,1,0,0,0,1,1,1,0])
        except:
            pass
    axs[0].set_title(dirname, fontsize=16)
    plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_approachprop.png")
    plt.show()

## Main & supplement
# Supplement for k stability under different variables
# graph_in_time_2d(2, "interactions", mnn = 3, deg = 3, mutual = True, show_mut = True)
# graph_in_time_2d(2, "approaches", mnn = 3, deg = 3, mutual = True, show_mut = True)

# Supplement for club stability under k
graph_to_plot = 8 # Index of the group to display
# graph_in_time_2d(graph_to_plot, "interactions", mnn = 2, deg = 2, mutual = True, show_mut = True)
# graph_in_time_2d(graph_to_plot, "interactions", mnn = 3, deg = 3, mutual = True, show_mut = True)
# graph_in_time_2d(graph_to_plot, "interactions", mnn = 4, deg = 4, mutual = True, show_mut = True)
graph_in_time_2d(graph_to_plot, "interactions", mnn = 5, deg = 4, mutual = True, show_mut = True)


