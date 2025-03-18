import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
from scipy import stats
import pandas as pd

sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

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
        # U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        # U, p = stats.ttest_ind(data1, data2,equal_var=False)
        result = stats.permutation_test((data1, data2), statistic)
        U = result.statistic  # This gives the test statistic
        p = result.pvalue      # This gives the p-value

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

## Graph theory measures
all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], [], [0, 2]]
# all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9], [3, 9]] #took out mutants with weak histology
all_mutants = [[4, 6], [2, 6], [6], [0, 5], [3, 5], [7, 9], [0, 5], [2, 3], [2, 3], [0, 2, 3, 6], [4, 5, 6, 7, 9], [2, 3, 9], [1,3,9], [0,2,3, 6], [2,3,8,9], [3, 9]] #full histology

# labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10"]
labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def measures(graph_idx, day, measure = "hub"):
    """
    Returns the measurement specified by input argument for mutants, rc and in wt in graph specified by input index.
    """
    datapath = "..\\data\\social network matrices 1day\\"+labels[graph_idx]+"\\interactions_resD1_"+str(day)+".csv"
    if graph_idx > 8:
        datapath = "..\\data\\validation cohort\\1day\\"+labels[graph_idx]+"\\interactions_resD1_"+str(day)+".csv"
    try:
        data = read_graph([datapath], percentage_threshold = 0)[0]
    except Exception as e:
        print(e)
        return [np.nan], [np.nan], [np.nan], [np.nan]    
    data = (data + np.transpose(data))/2 # ensure symmetry
    g = ig.Graph.Weighted_Adjacency(data, mode='undirected')

    # g = ig.Graph.Weighted_Adjacency(data, mode='directed')
    # node_labels = read_labels(datapath+"\\approaches_resD7_1.csv")
    graph_length = len(g.vs())
    mutants = all_mutants[graph_idx]
    rc = all_rc[graph_idx]
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), all_rc[graph_idx]), ~np.isin(np.arange(graph_length), all_mutants[graph_idx]))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), all_mutants[graph_idx])]
    if measure == "hub":
        scores_mutants = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in wt]
    elif measure == "pagerank":
        scores_mutants = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in wt]
    elif measure == "authority":
        scores_mutants = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in wt]
    elif measure == "eigenvector_centrality":
        scores_mutants = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in wt]
    elif measure == "harmonic_centrality":
        scores_mutants = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in wt]
    elif measure == "betweenness":
         scores_mutants = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
    elif measure == "closeness":
         scores_mutants = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
    else:
        raise Exception("Unknown or misspelled input measurement.") 

    return scores_mutants, scores_rc, scores_rest, scores_wt

def plot_measure_timeseries(measure = "hub", separate_wt = False):
    """
    Box plot of graph theory measurement specified by input argument for all graphs in dataset, separated in mutants and WT, or 
    mutants, RC and others (using separate_wt argument).
    """
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            scores = measures(j, d+1, measure)
            scores_mutants.append(np.nanmean(scores[0]))
            scores_rc.append(np.nanmean(scores[1]))
            scores_rest.append(np.nanmean(scores[2]))
            scores_wt.append(np.nanmean(scores[3]))
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        std_mutants.append(np.nanstd(scores_mutants))
        std_rc.append(np.nanstd(scores_rc))
        std_rest.append(np.nanstd(scores_rest))
        std_wt.append(np.nanstd(scores_wt))
    t = np.arange(1, 16)
    value_mutants, value_rc, value_rest, value_wt = np.array(value_mutants), np.array(value_rc), np.array(value_rest), np.array(value_wt)
    std_mutants, std_rc, std_rest, std_wt = np.array(std_mutants), np.array(std_rc), np.array(std_rest), np.array(std_wt) 
    plt.figure()
    if separate_wt:
        plt.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        plt.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        plt.plot(t, value_rc, lw = 2, c = "green", label = "RC members")
        plt.fill_between(t, value_rc - std_rc, value_rc + std_rc, color="green", alpha=0.2)
        plt.plot(t, value_rest, lw = 2, c = "blue", label = "Others")
        plt.fill_between(t, value_rest - std_rest, value_rest + std_rest, color="blue", alpha=0.2)
    else:
        plt.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        plt.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        plt.plot(t, value_wt, lw = 2, c = "green", label = "WT")
        plt.fill_between(t, value_wt - std_wt, value_wt + std_wt, color="green", alpha=0.2)
        
    plt.legend()
    plt.title(measure, fontsize = 18)
    plt.xlabel("Day", fontsize = 15)
    plt.ylabel("Value", fontsize = 15)
    plt.xticks(range(1, 16, 2)) 
    plt.tick_params(axis='both', which='major', labelsize=15)  # Adjust major tick labels
    plt.show()
    
if __name__ == "__main__":
    # boxplot_chasing(False)
    # boxplot_approaches() 
    # boxplot_interactions()
    # time_in_arena(True)
    # social_time()
    # boxplot_measures("hub")
    # plot_measure_timeseries("authority")
    # plot_measure_timeseries("pagerank")
    # plot_measure_timeseries("hub")
    plot_measure_timeseries("eigenvector_centrality")
    # plot_measure_timeseries("harmonic_centrality")

    # boxplot_measures("harmonic_centrality")
    # boxplot_measures("closeness")




    