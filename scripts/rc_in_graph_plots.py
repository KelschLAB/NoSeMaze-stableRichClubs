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

def day_to_day_interactions_3d(graph_idx, mnn = 3, deg = 3):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"
    
    fig, ax = plt.subplots(1, 1, figsize=(2, 2)) #putting all results directly on same figure
    ax = fig.add_subplot(111, projection='3d')
    
    files = []
    for t in range(5):
        if os.path.exists(datapath+"interactions_resD3_"+str(t+1)+".csv"):
            files.append(datapath+"interactions_resD3_"+str(t+1)+".csv")
        else:
            print("missing day, ignoring")
            
    mutants_idx, _, _, _, RFIDs = get_category_indices(graph_idx, "interactions", 3)
    mutants = np.ones(len(RFIDs))
    mutants[mutants_idx] = 0
    
    display_graph_3d(files, ax, mnn = mnn, node_metric = "rich-club", deg = deg, layout = "circle", node_labels = None, idx = mutants, node_size = 0.1, default_edge_width = 1)
    
    # ax.set_axis_off()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # ax.set_zlabel('')
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # fig.suptitle(labels[graph_idx], fontsize=12)
    # ax.set_frame_on(False)
    #plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_interactions.png")
    plt.savefig("..\\plots\\paper\\rc_presentation\\graphs\\"+labels[graph_idx]+"_mnn"+str(mnn)+"_interactions.svg")
    plt.tight_layout()
    plt.show()

def day_to_day_approach(dirname, mnn = 4, deg = 2, mutual = True):
    """
    Plots and saves the graph of the rich club for the interaction graph given in 'datapath'
    after mnn cuting and k-core computation. Saves on svg format.
    """
    datapath = "..\\data\\social network matrices 3days\\"+dirname+"\\"
    fig, axs = plt.subplots(5, 1, figsize=(5, 10)) #putting all results directly on same figure
    for i in range(5):
        try:
            file = "approaches_resD3_"+str(i+1)+".csv"
            display_graph([datapath+file], axs[i], mnn = mnn, node_metric = "k-core", deg = deg, mutual = mutual, layout = "circle", node_labels = [])#, idx = [1,1,1,0,0,0,1,1,1,0])
        except:
            pass
    try:
        data = read_graph([datapath+file], return_ig = False)
        total = np.sum(data[0].flatten())
    except:
        total = np.nan
    axs[0].set_title(dirname+", total approaches: "+str(total), fontsize=16)
    # if mutual:
    #     plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_mnn"+str(mnn)+"_deg"+str(deg)+"_approach.png")
    # else:
    #     plt.savefig("C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\plots\\params_exploration\\"+dirname+"_nn"+str(mnn)+"_deg"+str(deg)+"_approach.png")
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

# for graph in ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10"]:
#     day_to_day_approach(graph, 4, 6, False)
for i in range(len(labels)):
    day_to_day_interactions_3d(i, mnn = 3, deg = 3)
# day_to_day_interactions_3d(1, mnn = 3, deg = 3)
