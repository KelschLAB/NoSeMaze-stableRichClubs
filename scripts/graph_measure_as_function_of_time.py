import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.stats import sem
from tqdm import tqdm

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
        print(p)
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

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def measures(graph_idx, day, window = 3, measure = "hub", variable = "approaches", mnn = None, mutual = True):
    """
    Returns the measurement specified by input argument for mutants, rc and in wt in graph specified by input index.
    graph_idx specifies the cohort number, day specifies the specific timepoint (depends on window) and window specifies 
    the size of the window of averaging for the graph representation.        
    """
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    if window == 1:
        datapath = "..\\data\\both_cohorts_1day\\"+labels[graph_idx]+"\\"+variable+"_resD1_"+str(day)+".csv"
    elif window == 3:
        datapath = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"+variable+"_resD3_"+str(day)+".csv"
    elif window == 7:
        datapath = "..\\data\\averaged\\"+labels[graph_idx]+"\\"+variable+"_resD7_"+str(day)+".csv"
    else:
        print("Incorrect time window")
        return
    try:
        data = read_graph([datapath], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
        arr = np.loadtxt(datapath, delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)
    except Exception as e:
        print(e)
        return [np.nan], [np.nan], [np.nan], [np.nan]    
    if variable == "interactions":
        mode = 'undirected'
        data = (data + np.transpose(data))/2 # ensure symmetry
    elif variable == "approaches":
        mode = 'directed'
    else:
        raise NameError("Incorrect input argument for 'variable'.")
        
    data = np.where(data > 0.01, data, 0)

    g = ig.Graph.Weighted_Adjacency(data, mode=mode)

    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    is_mutant = []#[True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
    for rfid in RFIDs:
        if len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
            histology = metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "histology"].values[0]
            if histology != 'weak':
                # mouse is a mutant and histology is strong or unknown
                is_mutant.append(True)
            else:
                is_mutant.append(False)
        elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and not metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
            is_mutant.append(False)
        elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) == 0: 
            print("Mouse not found, treat as WT")
            is_mutant.append(False)

    is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
    graph_length = len(g.vs())
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
    
    if measure == "hub":
        scores_mutants = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        
    elif measure == "indegree":
        scores_mutants = [g.degree(mode="in")[i] for i in mutants]
        scores_rc = [g.degree(mode="in")[i] for i in rc]
        scores_rest = [g.degree(mode="in")[i] for i in others]
        scores_wt = [g.degree(mode="in")[i] for i in wt]    
        
    elif measure == "outdegree":
        scores_mutants = [g.degree(mode="out")[i] for i in mutants]
        scores_rc = [g.degree(mode="out")[i] for i in rc]
        scores_rest = [g.degree(mode="out")[i] for i in others]
        scores_wt = [g.degree(mode="out")[i] for i in wt]    
        
    elif measure == "summed insubcomponent":
        scores_mutants = [np.sum(g.subcomponent(i, mode="in")) for i in mutants]
        scores_rc = [np.sum(g.subcomponent(i, mode="in")) for i in rc]
        scores_rest = [np.sum(g.subcomponent(i, mode="in")) for i in others]
        scores_wt = [np.sum(g.subcomponent(i, mode="in")) for i in wt]    
        
    elif measure == "summed outsubcomponent":
        scores_mutants = [np.sum(g.subcomponent(i, mode="out")) for i in mutants]
        scores_rc = [np.sum(g.subcomponent(i, mode="out")) for i in rc]
        scores_rest = [np.sum(g.subcomponent(i, mode="out")) for i in others]
        scores_wt = [np.sum(g.subcomponent(i, mode="out")) for i in wt]    

    elif measure == "summed injaccard":
        scores_mutants = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in wt]    
        
    elif measure == "summed outjaccard":
        scores_mutants = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in wt]     
        
    elif measure == "instrength":
        scores_mutants = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in mutants]
        scores_rc = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in rc]
        scores_rest = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in others]
        scores_wt = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in wt]    
        
    elif measure == "outstrength":
        scores_mutants = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in mutants]
        scores_rc = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in rc]
        scores_rest = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in others]
        scores_wt = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in wt]    
        
    elif measure == "outcloseness":
        scores_mutants = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in wt]       
    elif measure == "incloseness":
        scores_mutants = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in wt]  
            
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
         
    elif measure == "constraint":
         scores_mutants = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
         
    elif measure == "summed cocitation":
         scores_mutants = [np.sum(g.cocitation()[i]) for i in mutants]
         scores_rc = [np.sum(g.cocitation()[i]) for i in rc]
         scores_rest = [np.sum(g.cocitation()[i]) for i in others]  
         scores_wt = [np.sum(g.cocitation()[i]) for i in wt]  
         
    elif measure == "summed bibcoupling":
         scores_mutants = [np.sum(g.bibcoupling()[i]) for i in mutants]
         scores_rc = [np.sum(g.bibcoupling()[i]) for i in rc]
         scores_rest = [np.sum(g.bibcoupling()[i]) for i in others]  
         scores_wt = [np.sum(g.bibcoupling()[i]) for i in wt]  
         
    elif measure == "transitivity":
         scores_mutants = [g.transitivity_local_undirected()[i] for i in mutants]
         scores_rc = [g.transitivity_local_undirected()[i] for i in rc]
         scores_rest = [g.transitivity_local_undirected()[i] for i in others]  
         scores_wt = [g.transitivity_local_undirected()[i] for i in wt]  
    else:
        raise Exception("Unknown or misspelled input measurement.") 

    return scores_mutants, scores_rc, scores_rest, scores_wt, RFIDs

def plot_measure_timeseries(measure = "hub", window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    Box plot of graph theory measurement specified by input argument for all graphs in dataset, separated in mutants and WT, or 
    mutants, RC and others (using separate_wt argument).
    Inputs:
        - measure: the type of graph measurment to display. Default to hub score.
        - window: how many days of averaging should be used to display the results. 
            Defaults to 3, meaning one graph is the average of three days of experiment. 
        - variable: which kind of variable should be displayed. Shoud be "interactions" or "approaches".
        - separate_wt: whether or not the display of WT should be separated in the RC and others.
    """
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            scores = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_mutants.extend(scores[0])
            scores_rc.extend(scores[1])
            scores_rest.extend(scores[2])
            scores_wt.extend(scores[3])
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        scores_mutants, scores_rc, scores_rest, scores_wt = np.array(scores_mutants), np.array(scores_rc), np.array(scores_rest), np.array(scores_wt)
        std_mutants.append(sem(scores_mutants[~np.isnan(scores_mutants)]))
        std_rc.append(sem(scores_rc[~np.isnan(scores_rc)]))
        std_rest.append(sem(scores_rest[~np.isnan(scores_rest)]))
        std_wt.append(sem(scores_wt[~np.isnan(scores_wt)]))
        
    t = np.arange(1, 1+15//window)
    value_mutants, value_rc, value_rest, value_wt = np.array(value_mutants), np.array(value_rc), np.array(value_rest), np.array(value_wt)
    std_mutants, std_rc, std_rest, std_wt = np.array(std_mutants), np.array(std_rc), np.array(std_rest), np.array(std_wt) 
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if separate_wt:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_rc, lw = 2, c = "green", label = "RC members")
        ax.fill_between(t, value_rc - std_rc, value_rc + std_rc, color="green", alpha=0.2)
        ax.plot(t, value_rest, lw = 2, c = "blue", label = "Others")
        ax.fill_between(t, value_rest - std_rest, value_rest + std_rest, color="blue", alpha=0.2)
    else:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_wt, lw = 2, c = "green", label = "WT")
        ax.fill_between(t, value_wt - std_wt, value_wt + std_wt, color="green", alpha=0.2)
        
    ax.set_title(measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window+1, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    if ax is None:
        ax.legend()
        
def plot_derivative_timeseries(measure = "hub", window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    Box plot of graph theory measurement specified by input argument for all graphs in dataset, separated in mutants and WT, or 
    mutants, RC and others (using separate_wt argument).
    Inputs:
        - measure: the type of graph measurment to display. Default to hub score.
        - window: how many days of averaging should be used to display the results. 
            Defaults to 3, meaning one graph is the average of three days of experiment. 
        - variable: which kind of variable should be displayed. Shoud be "interactions" or "approaches".
        - separate_wt: whether or not the display of WT should be separated in the RC and others.
    """
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_t2 = measures(j, d+2, window, measure, variable, mnn, mutual)
            if scores_t1 is None or scores_t2 is None:
                continue
            scores_mutants.extend(np.array(scores_t2[0]) - np.array(scores_t1[0]))
            scores_rc.extend(np.array(scores_t2[1]) - np.array(scores_t1[1]))
            scores_rest.extend(np.array(scores_t2[2]) - np.array(scores_t1[2]))
            scores_wt.extend(np.array(scores_t2[3]) - np.array(scores_t1[3]))
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        scores_mutants, scores_rc, scores_rest, scores_wt = np.array(scores_mutants), np.array(scores_rc), np.array(scores_rest), np.array(scores_wt)
        std_mutants.append(sem(scores_mutants[~np.isnan(scores_mutants)]))
        std_rc.append(sem(scores_rc[~np.isnan(scores_rc)]))
        std_rest.append(sem(scores_rest[~np.isnan(scores_rest)]))
        std_wt.append(sem(scores_wt[~np.isnan(scores_wt)]))
        
    t = np.arange(1, 1+15//window)
    value_mutants, value_rc, value_rest, value_wt = np.array(value_mutants), np.array(value_rc), np.array(value_rest), np.array(value_wt)
    std_mutants, std_rc, std_rest, std_wt = np.array(std_mutants), np.array(std_rc), np.array(std_rest), np.array(std_wt) 
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if separate_wt:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_rc, lw = 2, c = "green", label = "RC members")
        ax.fill_between(t, value_rc - std_rc, value_rc + std_rc, color="green", alpha=0.2)
        ax.plot(t, value_rest, lw = 2, c = "blue", label = "Others")
        ax.fill_between(t, value_rest - std_rest, value_rest + std_rest, color="blue", alpha=0.2)
    else:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_wt, lw = 2, c = "green", label = "WT")
        ax.fill_between(t, value_wt - std_wt, value_wt + std_wt, color="green", alpha=0.2)
        
    ax.set_title("Time derivative of "+ measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    if ax is None:
        ax.legend()
        
def plot_persistance(out = True, window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    Computes the average persistance of the animal over all graphs defined in the different cohort.
    The persistance rewards active edges that are preserved, meaning that an absence of edge does NOT 
    influence the value of the persistance. The persistance of a node is the sum of preserved existing edges
    over two time points. Again, an absence of edge does NOT influence this value.
    """
    
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in tqdm(range(1, 15//window)):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            if window == 1:
                datapath_t1 = "..\\data\\both_cohorts_1day\\"+labels[j]+"\\"+variable+"_resD1_"+str(d)+".csv"
                datapath_t2 = "..\\data\\both_cohorts_1day\\"+labels[j]+"\\"+variable+"_resD1_"+str(d+1)+".csv"
            elif window == 3:
                datapath_t1 = "..\\data\\both_cohorts_3days\\"+labels[j]+"\\"+variable+"_resD3_"+str(d)+".csv"
                datapath_t2 = "..\\data\\both_cohorts_3days\\"+labels[j]+"\\"+variable+"_resD3_"+str(d+1)+".csv"
            elif window == 7:
                datapath_t1 = "..\\data\\averaged\\"+labels[j]+"\\"+variable+"_resD7_"+str(d)+".csv"
                datapath_t2 = "..\\data\\averaged\\"+labels[j]+"\\"+variable+"_resD7_"+str(d+1)+".csv"
            else:
                print("Incorrect time window")
                return
            try:
                data_t1 = read_graph([datapath_t1], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
                data_t2 = read_graph([datapath_t2], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
                arr_t1 = np.loadtxt(datapath_t1, delimiter=",", dtype=str)
                arr_t2 = np.loadtxt(datapath_t2, delimiter=",", dtype=str)
                RFIDs = arr_t1[0, 1:].astype(str)
            except Exception as e:
                continue    
            if variable == "interactions":
                mode = 'undirected'
                data_t1 = (data_t1 + np.transpose(data_t1))/2 # ensure symmetry
                data_t2 = (data_t2 + np.transpose(data_t2))/2 # ensure symmetry
            elif variable == "approaches":
                mode = 'directed'
            else:
                raise NameError("Incorrect input argument for 'variable'.")
    
            if mnn is not None:
                data_t1 = np.where(data_t1 > 0.01, 1, 0) # replacing remaining connections by 1
                data_t2 = np.where(data_t2 > 0.01, 1, 0)
            else:
                data_t1 = np.where(data_t1 > 0.01, data_t1, 0) # replacing remaining connections by 1
                data_t2 = np.where(data_t2 > 0.01, data_t2, 0)
                
            axis = 1 if out else 0
            # persistance = np.sum((2*np.abs(data_t2 - data_t1)/(data_t1+data_t2)) <= 0.5, axis = axis)
            cond = np.abs(data_t2 - data_t1) <= 0.5*data_t1
            persistance = np.sum(cond, axis = axis)
 


            curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[j][1:]), :]
            # figuring out index of true mutants in current group
            is_mutant = []#[True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
            for rfid in RFIDs:
                if len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                    histology = metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "histology"].values[0]
                    if histology != 'weak':
                        # mouse is a mutant and histology is strong or unknown
                        is_mutant.append(True)
                    else:
                        is_mutant.append(False)
                elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and not metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                    is_mutant.append(False)
                elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) == 0: 
                    print("Mouse not found, treat as WT")
                    is_mutant.append(False)

            is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
            graph_length = data_t1.shape[0]
            mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
            rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
            others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
            wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
            
            scores_mutants = [persistance[i] for i in mutants]
            scores_rc = [persistance[i] for i in rc]
            scores_rest = [persistance[i] for i in others]
            scores_wt = [persistance[i] for i in wt]
            
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        scores_mutants, scores_rc, scores_rest, scores_wt = np.array(scores_mutants), np.array(scores_rc), np.array(scores_rest), np.array(scores_wt)
        std_mutants.append(sem(scores_mutants[~np.isnan(scores_mutants)]))
        std_rc.append(sem(scores_rc[~np.isnan(scores_rc)]))
        std_rest.append(sem(scores_rest[~np.isnan(scores_rest)]))
        std_wt.append(sem(scores_wt[~np.isnan(scores_wt)]))
    
    t = np.arange(1, 15//window)
    value_mutants, value_rc, value_rest, value_wt = np.array(value_mutants), np.array(value_rc), np.array(value_rest), np.array(value_wt)
    std_mutants, std_rc, std_rest, std_wt = np.array(std_mutants), np.array(std_rc), np.array(std_rest), np.array(std_wt) 
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if separate_wt:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_rc, lw = 2, c = "green", label = "RC members")
        ax.fill_between(t, value_rc - std_rc, value_rc + std_rc, color="green", alpha=0.2)
        ax.plot(t, value_rest, lw = 2, c = "blue", label = "Others")
        ax.fill_between(t, value_rest - std_rest, value_rest + std_rest, color="blue", alpha=0.2)
    else:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_wt, lw = 2, c = "green", label = "WT")
        ax.fill_between(t, value_wt - std_wt, value_wt + std_wt, color="green", alpha=0.2)
    
    name = "Persistance out" if out else "Persistance in"
    ax.set_title(name, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    ax.legend()
    plt.show()
    
def format_measure(measure = "hub", window = 3):
    """Formats the measures made on the graphs into a csv file readable by matlab or R to run LME analysis
    """
    genotype_info_path = "..\\data\\genotype_info.csv"
    genotype_df = pd.read_csv(genotype_info_path)
    all_scores, RFIDs, genotype, day = [], [], [], []
    for d in range(15//window):
        for j in range(len(labels)):
            if window == 1:
                datapath = "..\\data\\both_cohorts_1day\\"+labels[j]+"\\approaches_resD1_"+str(d+1)+".csv"
            elif window == 3:
                datapath = "..\\data\\both_cohorts_3days\\"+labels[j]+"\\approaches_resD3_"+str(d+1)+".csv"
            elif window == 7:
                datapath = "..\\data\\averaged\\"+labels[j]+"\\approaches_resD7_"+str(d+1)+".csv"
            else:
                print("Incorrect time window")
                return
            try:
                data = read_graph([datapath], percentage_threshold = 0)[0]
                arr = np.loadtxt(datapath, delimiter=",", dtype=str)
                tags = arr[0, 1:].astype(str)
            except Exception as e:
                print(e)
                continue   
            
            # data = (data + np.transpose(data))/2 # ensure symmetry
            # g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
            g = ig.Graph.Weighted_Adjacency(data, mode='directed')
            # 
            graph_len = len(g.vs())
            if measure == "hub":
                scores = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in range(graph_len)]
            elif measure == "pagerank":
                scores = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in range(graph_len)]
            elif measure == "eigenvector_centrality":
                scores = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in range(graph_len)]
            elif measure == "betweenness":
                scores = [g.netweenness(weights = [e['weight'] for e in g.es()])[i] for i in range(graph_len)]
            elif measure == "closeness":
                scores = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in range(graph_len)]
            else:
                raise Exception("Unknown or misspelled input measurement.") 
                return
            scores = np.array(scores)
            
            matched_RFID = np.array([any(genotype_df["Mouse_RFID"] == t) for t in tags]) # checking if there are misnamed RFIDs that need to be filtered out
            all_scores.extend(scores[matched_RFID])
            RFIDs.extend(tags[matched_RFID]) # RFIDs tags of this cohort
            genotype.extend([genotype_df.loc[genotype_df["Mouse_RFID"] == t, 'genotype'].values[0] for t in tags[matched_RFID]])
            day.extend([d for i in range(np.sum(matched_RFID))])
            
    d = {"Mouse_RFID":RFIDs, "%s"%measure: all_scores, "genotype":genotype, "timepoint":day}
    df = pd.DataFrame(data=d)
    df.to_csv("C:\\Users\\Corentin offline\\Downloads\\"+str(measure)+"_data_"+str(window)+"d.csv")
            
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
    # plot_measure_timeseries("hub", 3)
    # format_measure("pagerank", 1)
    
    
    unweighted_features = ["indegree", "outdegree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]

    mnn = 5
    mutual = False
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, ax in enumerate(axs.flatten()):
    #     try:
    #         plot_measure_timeseries(unweighted_features[idx], 1, 'approaches', False, mnn = mnn, mutual = mutual, ax = ax)
    #         # plot_derivative_timeseries(unweighted_features[idx], 1, 'approaches', False, mnn = mnn, mutual = mutual, ax = ax)
    #     except:
    #         continue
    # # plt.tight_layout()
    # title = f"mnn = {mnn}" if mutual else f"nn = {mnn}"
    # fig.suptitle(title)
    # plot_derivative_timeseries("instrength", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    
    plot_measure_timeseries("summed injaccard", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    plot_derivative_timeseries("summed injaccard", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    
    plot_measure_timeseries("outstrength", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    plot_derivative_timeseries("outstrength", 1, 'approaches', False, mnn = mnn, mutual = mutual)

    
    plot_persistance(True, 1, "approaches", False, None, False, None)
    plot_persistance(False, 1, "approaches", False, None, False, None)

    # plot_persistance(False,1, "approaches", False, None, False, None)
    
    # plot_derivative_timeseries("incloseness", 3, 'approaches', False, mnn = None, mutual = False)

    # plt.savefig(f"..\\plots\\graph_features\\unweighted_mnn_{mnn}_mutual_{mutual}.png", dpi = 150)
    # plot_measure_timeseries("hub", 3, 'approaches', True) # ingoing approaches


    # boxplot_measures("harmonic_centrality")
    # boxplot_measures("closeness")




    