import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from weighted_rc import weighted_rich_club
from scipy import stats


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
    
def add_significance(data, ax, bp):
    """Computes and adds p-value significance to input ax and boxplot"""
    # Initialise a list of combinations of groups that are significantly different
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for combination in combinations:
        data1 = data[combination[0] - 1]
        data2 = data[combination[1] - 1]
        # Significance
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([combination, p])
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        # Get the y-axis limits
        bottom, top = ax.get_ylim()
        y_range = top - bottom
        bar_height = (y_range * 0.07 * level) + top
        bar_tips = bar_height - (y_range * 0.02)
        plt.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
        )
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (y_range * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

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
    add_significance(data, ax, bp)
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
    add_significance(data, ax, bp)
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
    add_significance(data, ax, bp)
    # bottom, top = ax.get_ylim()
    plt.show()
    
if __name__ == "__main__":
    boxplot_chasing()
    # boxplot_approaches() 
    # boxplot_interactions()
    