import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from weighted_rc import weighted_rich_club

sys.path.append('..\\src\\')
from read_graph import read_graph

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def format_plot(ax, bp):
    """ Sets the x-axis the RC/mutants/Others, changes the color of the bars in the boxplot."""
    xticklabels = ["RC", "Mutants", "Others"]
    ax.set_xticklabels(xticklabels, fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')

def boxplot_chasing():
    """
    Compares the number of chasings made by RC members, mutants and others.
    """
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [2, 4], [5,6], [0, 1], [3,4,6], [3, 5, 7], [6, 7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    chasings_rc, chasings_mutants, chasings_others = [], [], [] 
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        for rc in all_rc[idx]:
            chasings_rc.append(np.sum(data[rc, :]))
        for mutant in all_mutants[idx]:
            chasings_mutants.append(np.sum(data[mutant, :]))
        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), all_rc[idx]), ~np.isin(np.arange(10), all_mutants[idx]))]
        for other in others:
            chasings_others.append(np.sum(data[other, :]))
    data = [chasings_rc, chasings_mutants, chasings_others]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_ylabel("Outgoing chasings", fontsize = 20)
    format_plot(ax, bp) # set x_axis, and colors of each bar
    # bottom, top = ax.get_ylim()
    plt.show()
    
def boxplot_approaches():
    """
    Compares the number of approaches made by RC members, mutants and others.
    """
    all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_rc, approaches_mutants, approaches_others = [], [], [] 
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        for rc in all_rc[idx]:
            approaches_rc.append(np.sum(data[:, rc]))
        for mutant in all_mutants[idx]:
            approaches_mutants.append(np.sum(data[mutant, :]))
        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), all_rc[idx]), ~np.isin(np.arange(10), all_mutants[idx]))]
        for other in others:
            try:
                approaches_others.append(np.sum(data[other, :]))
            except:
                pass
            
    data = [approaches_rc, approaches_mutants, approaches_others]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_ylabel("Outgoing approaches", fontsize = 20)
    format_plot(ax, bp) # set x_axis, and colors of each bar
    # bottom, top = ax.get_ylim()
    # ax.set_title("Week 1", fontsize = 25, weight='bold')
    plt.show()
    
def boxplot_interactions():
    """
    Compares the number of interactions made by RC members, mutants and others.
    """
    all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    interactions_rc, interactions_mutants, interactions_others = [], [], [] 
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]
        for rc in all_rc[idx]:
            interactions_rc.append(np.sum(data[rc, :]))
        for mutant in all_mutants[idx]:
            interactions_mutants.append(np.sum(data[mutant, :]))
        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), all_rc[idx]), ~np.isin(np.arange(10), all_mutants[idx]))]
        for other in others:
            try:
                interactions_others.append(np.sum(data[other, :]))
            except:
                pass
    data = [interactions_rc, interactions_mutants, interactions_others]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_ylabel("Total interactions", fontsize = 20)
    format_plot(ax, bp) # set x_axis, and colors of each bar
    # bottom, top = ax.get_ylim()
    plt.show()
    
if __name__ == "__main__":
    # boxplot_chasing()
    # boxplot_approaches() 
    boxplot_interactions()
    