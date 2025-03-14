import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from weighted_rc import weighted_rich_club
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

def format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Others"]):
    """ Sets the x-axis the RC/mutants/Others, changes the color of the bars in the boxplot."""
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

def boxplot_chasing(out = True):
    """
    Compares the number of chasings made by RC members, mutants and others.
    """
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [2, 4], [5,6], [0, 1], [3,4,6], [3, 5, 7], [6, 7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    chasings_rc, chasings_mutants, chasings_others = [], [], [] 
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        if out:
            for rc in all_rc[idx]:
                chasings_rc.append(np.sum(data[rc, :]))
            for mutant in all_mutants[idx]:
                chasings_mutants.append(np.sum(data[mutant, :]))
            others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), all_rc[idx]), ~np.isin(np.arange(10), all_mutants[idx]))]
            for other in others:
                chasings_others.append(np.sum(data[other, :]))
        else:
            for rc in all_rc[idx]:
                chasings_rc.append(np.sum(data[:, rc]))
            for mutant in all_mutants[idx]:
                chasings_mutants.append(np.sum(data[:, mutant]))
            others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), all_rc[idx]), ~np.isin(np.arange(10), all_mutants[idx]))]
            for other in others:
                chasings_others.append(np.sum(data[:, other]))
            
    data = [chasings_rc, chasings_mutants, chasings_others]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    if out:
        ax.set_ylabel("Outgoing chasings", fontsize = 20)
    else:
        ax.set_ylabel("Ingoing chasings", fontsize = 20)

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
    
def social_time():
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\meta-data_full.xlsx"
    
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    df2 = pd.read_excel(path_to_second_cohort)
    
    rc2 = df2.loc[:, "RC"]
    mutants2 = df2.loc[:, "mutant"]

    rc = np.concatenate([rc1, rc2])
    mutants = np.concatenate([mutants1, mutants2])
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    time_in_arena = np.concatenate((time_in_arena1, time_in_arena2))
    social_time = np.concatenate((social_time1, social_time2))
    
    # print(time_in_arena[~rc]**2)
    nan = np.isnan(social_time)
    rc_mems = np.array([time_in_arena[~nan*rc], social_time[~nan*rc]])
    non_rc = np.array([time_in_arena[~nan*~rc], social_time[~nan*~rc]])
    wt = np.array([time_in_arena[~nan*~mutants], social_time[~nan*~mutants]])
    muts = np.array([time_in_arena[~nan*mutants], social_time[~nan*mutants]])
    
    # plot params
    data = [social_time[~nan*rc], 
            social_time[~nan*mutants], 
            social_time[~nan*~mutants*~rc]]

    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    format_plot(ax, bp) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
    
 #   plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xticklabels = ["RC", "Mutants", "Others"]
    ax.set_xticklabels(xticklabels, fontsize = 20)
    ax.set_ylabel("Avg. social time", fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, 3000, fr'$n = {sample_size}$', ha='center', size='x-small')
    plt.tight_layout()
    
def time_in_arena(plot_both_cohorts = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\meta-data_full.xlsx"
    
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    df2 = pd.read_excel(path_to_second_cohort)
    
    rc2 = df2.loc[:, "RC"]
    mutants2 = df2.loc[:, "mutant"]

    rc = np.concatenate([rc1, rc2])
    mutants = np.concatenate([mutants1, mutants2])
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    time_in_arena = np.concatenate((time_in_arena1, time_in_arena2))
    social_time = np.concatenate((social_time1, social_time2))
    
    # print(time_in_arena[~rc]**2)
    nan = np.isnan(social_time)
    rc_mems = np.array([time_in_arena[~nan*rc], social_time[~nan*rc]])
    non_rc = np.array([time_in_arena[~nan*~rc], social_time[~nan*~rc]])
    wt = np.array([time_in_arena[~nan*~mutants], social_time[~nan*~mutants]])
    muts = np.array([time_in_arena[~nan*mutants], social_time[~nan*mutants]])
    
    # plot params
    data = [time_in_arena[~nan*rc], 
            time_in_arena[~nan*mutants], 
            time_in_arena[~nan*~mutants*~rc]]

    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    format_plot(ax, bp) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
 #   plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xticklabels = ["RC", "Mutants", "Others"]
    ax.set_xticklabels(xticklabels, fontsize = 20)
    ax.set_ylabel("Avg. time in arena", fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, 15000, fr'$n = {sample_size}$', ha='center', size='x-small')
        
        
## Graph theory measures
all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], [], [0, 2]]
# all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9], [3, 9]] #took out mutants with weak histology
all_mutants = [[4, 6], [2, 6], [6], [0, 5], [3, 5], [7, 9], [0, 5], [2, 3], [2, 3], [0, 2, 3, 6], [4, 5, 6, 7, 9], [2, 3, 9], [1,3,9], [0,2,3, 6], [2,3,8,9], [3, 9]] #full histology

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def measures(graph_idx, measure = "hub"):
    """
    Returns the measurement specified by input argument for mutants, rc and in wt in graph specified by input index.
    """
    datapath = "..\\data\\averaged\\"+labels[graph_idx]
    data = read_graph([datapath+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph([datapath+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]
    data = (data + np.transpose(data))/2 # ensure symmetry
    g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    node_labels = read_labels(datapath+"\\approaches_resD7_1.csv")
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

def boxplot_measures(measure = "hub", separate_wt = False):
    """
    Box plot of graph theory measurement specified by input argument for all graphs in dataset, separated in mutants and WT, or 
    mutants, RC and others (using separate_wt argument).
    """
    scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
    for j in range(10):#range(len(labels)):
        scores = measures(j, measure)
        scores_mutants.extend(scores[0])
        scores_rc.extend(scores[1])
        scores_rest.extend(scores[2])
        scores_wt.extend(scores[3])
    plt.figure()
    ax = plt.axes()
    # sns.catplot([scores_mutants, scores_wt])
    # plt.hist(scores_mutants, bins = 10, alpha = 0.5)
    # plt.hist(scores_wt, bins = 10, alpha = 0.5)
    if separate_wt:
        bp = ax.boxplot([scores_rc, scores_mutants, scores_wt], widths=0.6, patch_artist=True)
        format_plot(ax, bp) # set x_axis, and colors of each bar
        add_significance([scores_rc, scores_mutants, scores_wt], ax, bp)
    else:
        bp = ax.boxplot([scores_mutants, scores_wt], widths=0.6, patch_artist=True)
        format_plot(ax, bp, ["Mutants", "WT"]) # set x_axis, and colors of each bar
        add_significance([scores_mutants, scores_wt], ax, bp)
    plt.title(measure)
    plt.show()
    
if __name__ == "__main__":
    # boxplot_chasing(False)
    # boxplot_approaches() 
    # boxplot_interactions()
    # time_in_arena(True)
    # social_time()
    # boxplot_measures("hub")
    # boxplot_measures("authority")
    # boxplot_measures("pagerank")
    # boxplot_measures("pagerank", True)
    # boxplot_measures("eigenvector_centrality")
    # boxplot_measures("eigenvector_centrality", True)

    boxplot_measures("harmonic_centrality")
    # boxplot_measures("closeness")




    