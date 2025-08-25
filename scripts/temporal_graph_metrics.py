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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
from tqdm import tqdm 
from sklearn.manifold import TSNE
from sklearn import manifold
sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels
from utils import get_category_indices, spread_points_around_center, add_group_significance
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats
from graph_metrics import add_significance

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def default_on_error(graph_idx, variable, window):
    """
    If the data for a graph on a specific day is not available because of detection issue, nans should be returned. 
    To preserve the structure of the data array, the number of returned nans should reflect the number of mutants, rc members and wt in the cohort. 
    """
    mutants, rc, others, wt, RFIDs = get_category_indices(graph_idx, "approaches", window) # load data from exp day 1 as a template, because all group were properly detected.
    return [np.nan]*len(mutants), [np.nan]*len(rc), [np.nan]*len(others), \
        [np.nan]*len(wt), [np.nan]*len(RFIDs), \
        {"Mouse_RFID": [np.nan]*len(RFIDs), "mutant": [np.nan]*len(RFIDs), "RC": [np.nan]*len(RFIDs), "Group_ID": int(labels[graph_idx][1:])}

def time_measures(measure, graph_idx, window = 1, variable = "approaches",
                  mnn = None, mutual = True, weighted = False, threshold = 0.0,
                  summation = "mean", normalization = None, in_group_norm = False, logscale = False):
    """
    Computes specified time graph-theoretical metric for a given group, and time window.

    This function processes a time series of adjacency matrices (graphs) representing animal interactions,
    computes a variety of metrics (e.g., burstiness, inter-contact interval), and returns values for
    different animal subgroups (mutants, wild-type, RC, etc.).

    Parameters:
        - measure (str): Name of the metric to compute. Supported values are:
            - "summed outICI", "summed inICI"
            - "summed outburstiness", "summed inburstiness"
            - "summed outburstiness rank", "summed inburstiness rank"
            - "summed outNEF", "summed inNEF"
            - "summed outNEF rank", "summed inNEF rank"

        - graph_idx (int): Index of the group to analyze (corresponds to an entry in the global 'labels' list).
        - window (int, optional): Size of the time window for graph aggregation. Supported values are 1, 3, or 7 (days). Default is 1.
        - variable (str, optional): Type of interaction to analyze. "approaches" (directed graph) or "interactions" (undirected graph).
        - mnn (int or None, optional): Minimum number of neighbors for graph cutting. If None, no neighbor filtering is applied.
        - mutual (bool, optional): If True, uses mutual nearest neighbors when building graphs. Ignored if `mnn` is None.
        - weighted (bool, optional): If True, preserves edge weights in graph; otherwise binarizes the graph.
        - threshold (float, optional): Percentage threshold for edge pruning in the adjacency matrix. Default is 0.0.
        - summation (str, optional): Aggregation function across time:
             "mean": Computes the mean value per node.
             "median": Computes the median value per node.
        - normalization (str or None, optional): Type of normalization to apply to some metrics. Options include:
             "Poisson", "poisson", "CV" and "median CV". If None, no normalization is applied.
        - in_group_norm (bool, optional): If True, normalizes final metric values by the group maximum.
        - logscale (bool, optional): If True, applies logarithmic scaling to the final metric values.

    Returns:
        tuple: (scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, metadata)
            - scores_mutants (list): Metric values for mutant mice.
            - scores_rc (list): Metric values for rich-club (RC) mice.
            - scores_rest (list): Metric values for mice that are neither mutants nor RC.
            - scores_wt (list): Metric values for wild-type mice.
            - scores_all (list): Metric values for all mice in the group.
            - metadata (dict): Metadata dictionary for the group containing:
                - "Mouse_RFID" (list of str): RFID identifiers for all mice in the group.
                - "mutant" (list of bool): Boolean list indicating mutant status.
                - "RC" (list of bool): Boolean list indicating RC status.
                - "Group_ID" (int): Numerical group identifier.
    """

    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    if window not in [1, 3, 7]:
        print("Incorrect time window")
        return 
    
    ## figuring out position of mutants, wt and rc within the indexing of the csv file
    datapath = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_1.csv"
    try:
        data_ref = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
        arr = np.loadtxt(datapath, delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)
    except Exception as e:
        print(e)
        return default_on_error(graph_idx, variable, window)
    if variable == "interactions":
        mode = 'undirected'
        data_ref = (data_ref + np.transpose(data_ref))/2 # ensure symmetry
    elif variable == "approaches":
        mode = 'directed'
    else:
        raise NameError("Incorrect input argument for 'variable'.")
        
    data_ref = np.where(data_ref > 0.01, data_ref, 0)

    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    mutant_map = curr_metadata_df.set_index('Mouse_RFID')['mutant'].to_dict()
    is_mutant = [mutant_map.get(rfid, False) for rfid in RFIDs] # if RFID is missing, animal is assumed to be neurotypical
    

    is_RC = [] # no list comprehension allowed because of deprenciation warning due to RFID mismatch
    for rfid in RFIDs:
        if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values.size > 0:
            if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values[0]:
                is_RC.append(True)
            else:
                is_RC.append(False)
        else:
                is_RC.append(False)
        
    graph_length = data_ref.shape[0]
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_RC)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]

    graphs = []
    ## extracting graphs for each experimental day/session
    for day in np.arange(1, 16, window):
        datapath = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_"+str(day)+".csv"
        try:
            data = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
            arr = np.loadtxt(datapath, delimiter=",", dtype=str)
            RFIDs = arr[0, 1:].astype(str)
        except Exception as e:
            print(e)
            data = data_ref*np.nan
        if variable == "interactions":
            mode = 'undirected'
            data = (data + np.transpose(data))/2 # ensure symmetry
        elif variable == "approaches":
            mode = 'directed'
        else:
            raise NameError("Incorrect input argument for 'variable'.")
        
        if weighted:
            data = np.where(data > 0.01, data, 0)
        else:
            data = np.where(data > 0.01, 1, 0)

       # g = ig.Graph.Weighted_Adjacency(data, mode=mode)
        graphs.append(data)
        
    if measure == "summed outICI" or measure == "summed inICI": # summed average inter conter interval
        all_data = np.array(graphs)
        num_mice = all_data.shape[1]
        time_std, time_mean = np.zeros((num_mice, num_mice)), np.zeros((num_mice, num_mice))
        for i in range(num_mice):
            for j in range(num_mice):
                try:
                    time_std[i, j] = np.nanstd(np.where(all_data[:, i, j] > 0)[0][1:] - np.where(all_data[:, i, j] > 0)[0][:-1])
                except:
                    time_std[i, j] = np.nan
                try:
                    time_mean[i, j] = np.nanmean(np.where(all_data[:, i, j] > 0)[0][1:] - np.where(all_data[:, i, j] > 0)[0][:-1])
                except:
                    time_mean[i, j] = np.nan    
        all_scores = time_mean

    elif measure == "summed outburstiness" or measure == "summed inburstiness" or \
        measure == "summed outburstiness rank" or measure == "summed inburstiness rank":
        all_data = np.array(graphs)
        num_mice = all_data.shape[1]
        time_std, time_mean = np.zeros((num_mice, num_mice)), np.zeros((num_mice, num_mice))
        for i in range(num_mice):
            for j in range(num_mice):
                try:
                    time_std[i, j] = np.nanstd(np.where(all_data[:, i, j] > 0)[0][1:] - np.where(all_data[:, i, j] > 0)[0][:-1])
                except:
                    time_std[i, j] = np.nan
                try:
                    time_mean[i, j] = np.nanmean(np.where(all_data[:, i, j] > 0)[0][1:] - np.where(all_data[:, i, j] > 0)[0][:-1])
                except:
                    time_mean[i, j] = np.nan    

        all_scores = (time_std - time_mean)/(time_std + time_mean)


    elif measure == "summed outNEF" or measure == "summed inNEF" or \
        measure == "summed outNEF rank" or measure == "summed inNEF rank":
        all_data = np.array(graphs)
        time_std, time_mean, time_median = np.nanstd(all_data, axis = 0), np.nanmean(all_data, axis = 0), np.nanmedian(all_data, axis = 0)
        if normalization == "Poisson" or normalization == "poisson": # variant of the CV aimed at studying poissonicity
            all_scores = (time_std - time_mean)/(time_std + time_mean)
        elif normalization == "CV":
            all_scores = time_std/time_mean
        elif normalization == "median CV":
            all_scores = time_std/time_median
        else:
            all_scores = time_std

    else:
        raise Exception("Unknown or misspelled input measurement.") 
        return
    
    if "out" in measure:
        if summation == "mean":
            all_scores = np.nanmean(all_scores, axis = 1)
        if summation == "median":
            all_scores = np.nanmedian(all_scores, axis = 1)

        if logscale:
            all_scores = np.log(all_scores)
        if "rank" in measure:
            all_scores = all_scores.argsort().argsort()
        if in_group_norm:
            all_scores = all_scores/np.max(all_scores)
        metric_mutants = [all_scores[i] for i in mutants]
        metric_rc = [all_scores[i] for i in rc]
        metric_wt = [all_scores[i] for i in wt]
        metric_others = [all_scores[i] for i in others]
        metric_all = [all_scores[i] for i in range(len(RFIDs))]

    elif "in" in measure:
        if summation == "mean":
            all_scores = np.nanmean(all_scores, axis = 0)
        if summation == "median":
            all_scores = np.nanmedian(all_scores, axis = 0)
        if logscale:
            all_scores = np.log(all_scores)
            
        if "rank" in measure:
            all_scores = all_scores.argsort().argsort()
        if in_group_norm:
            all_scores = all_scores/np.max(all_scores)
        metric_mutants = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in mutants]
        metric_rc = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in rc]
        metric_wt = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in wt]
        metric_others = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in others]
        metric_all = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in range(len(RFIDs))]

    return metric_mutants, metric_rc, metric_others, metric_wt, metric_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}

def get_time_metric_df(measure, window = 1, variable = "approaches", mnn = None, mutual = True, weighted = False):
    scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, window, variable, mnn, mutual, weighted)
        scores_mutants.extend(res[0])
        scores_rc.extend(res[1])
        scores_others.extend(res[2])
        scores_wt.extend(res[3])
        scores_all.extend(res[4])
        RFIDs.extend(res[5]["Mouse_RFID"])
        mutants.extend(res[5]["mutant"])
        RCs.extend(res[5]["RC"])
        
    df = pd.DataFrame()
    df[measure] = scores_all
    df["mutant"] = mutants
    df["RC"] = RCs
    df["Mouse_RFID"] = RFIDs
    return df

def bp_metric(measure, window = 1, variable = "approaches",
                      mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean", ax = None):
    """
    Plots and compares the specified graph metric between RC and non mutant mice across all groups.

    Generates a boxplot (and optional swarm/violin plots) to visualize differences in a time-resolved 
    graph-theoretical measure, with statistical testing between groups. For more details on input parameters, 
    see the documentation of the time_measures function. 
    """
    scores_mutants, scores_RC, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, window, variable, mnn, mutual,
                            weighted, threshold, summation, normalization, in_group_norm, logscale)
        scores_mutants.extend(res[0])
        scores_RC.extend(res[1])
        scores_others.extend(res[2])
        scores_wt.extend(res[3])
        scores_all.extend(res[4])
        RFIDs.extend(res[5]["Mouse_RFID"])
        mutants.extend(res[5]["mutant"])
        RCs.extend(res[5]["RC"])
        
    df = pd.DataFrame()
    df[measure] = scores_all
    df["mutant"] = mutants
    df["RC"] = RCs
    df["Mouse_RFID"] = RFIDs
    
    data = [df.loc[df["RC"] == True, ['Mouse_RFID', measure]], df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', measure]],
            df.loc[np.logical_and(df["mutant"] == True, df["RC"] == False), ['Mouse_RFID', measure]] ]
    
    data[0][data[0] == np.inf] = np.nan
    data[1][data[1] == np.inf] = np.nan
    data[2][data[2] == np.inf] = np.nan
    data[0].dropna()
    data[1].dropna()
    data[2].dropna()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 60, 0.4
    positions = [0.75, 1.25, 1.75]
    bp = ax.boxplot([data[0][measure].dropna(), data[1][measure].dropna(), data[2][measure].dropna()], positions = positions, labels=["sRC", "Non-members WT", "mutants"],
                    showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    
    x_RC, scores_RC = spread_points_around_center(scores_RC, center=positions[0], bin_width = 0.1, interpoint=0.04)
    ax.scatter(x_RC, scores_RC, alpha=alpha, s=size, color="blue",edgecolor='none')
    x_others, scores_others = spread_points_around_center(scores_others, center=positions[1], bin_width = 0.1, interpoint=0.03)
    ax.scatter(x_others, scores_others, alpha=alpha, s=size, color="gray", label="Non-member",edgecolor='none')
    x_mutants, scores_mutants = spread_points_around_center(scores_mutants, center=positions[2], bin_width = 0.1, interpoint=0.03)
    ax.scatter(x_mutants, scores_mutants, alpha=alpha, s=size, color="red", label="Non-member",edgecolor='none')

    vp = plt.violinplot([data[0][measure], data[1][measure], data[2][measure]], positions, widths = [0.25, 0.38, 0.25], showextrema = False, showmedians=True)
    vp['bodies'][0].set_facecolor('blue')
    vp['bodies'][1].set_facecolor('gray')
    vp['bodies'][2].set_facecolor('red')

    if 'cmedians' in vp:  # Safety check
        vp['cmedians'].set_linewidth(5)  # Directly set width on LineCollection
        vp['cmedians'].set_color('k')  # Set color
        vp['cmedians'].set_linestyle('-')  # Ensure solid line


    add_significance(data, measure, ax, bp, stat)
    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization}, inGroupNorm = {in_group_norm}\n logscale = {logscale}, permutation test on the {stat}."
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return bp
    

def bp_metric_RC(measure, window = 1, variable = "approaches",
                      mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean", swarmplot = True, ax = None):
    """
    Plots and compares the specified graph metric between RC and non mutant mice across all groups.

    Generates a boxplot (and optional swarm/violin plots) to visualize differences in a time-resolved 
    graph-theoretical measure, with statistical testing between groups. For more details on input parameters, 
    see the documentation of the time_measures function. 
    """
    scores_mutants, scores_RC, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, window, variable, mnn, mutual,
                            weighted, threshold, summation, normalization, in_group_norm, logscale)
        scores_mutants.extend(res[0])
        scores_RC.extend(res[1])
        scores_others.extend(res[2])
        scores_wt.extend(res[3])
        scores_all.extend(res[4])
        RFIDs.extend(res[5]["Mouse_RFID"])
        mutants.extend(res[5]["mutant"])
        RCs.extend(res[5]["RC"])
        
    df = pd.DataFrame()
    df[measure] = scores_all
    df["mutant"] = mutants
    df["RC"] = RCs
    df["Mouse_RFID"] = RFIDs
    
    data = [ df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', measure]], 
            df.loc[df["RC"] == True, ['Mouse_RFID', measure]] ]
    
    data[0][data[0] == np.inf] = np.nan
    data[1][data[1] == np.inf] = np.nan
    data[0].dropna()
    data[1].dropna()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 60, 0.4
    positions = [0.75, 1.25]
    bp = ax.boxplot([data[0][measure].dropna(), data[1][measure].dropna()], positions = positions, labels=["Non-members WT", "sRC"],
                    showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    if swarmplot:
        x_others, scores_others = spread_points_around_center(scores_others, center=positions[0], bin_width = 0.1, interpoint=0.03)
        ax.scatter(x_others, scores_others, alpha=alpha, s=size, color="gray", label="Non-member",edgecolor='none')
        x_RC, scores_R = spread_points_around_center(scores_RC, center=positions[1], bin_width = 0.1, interpoint=0.04)
        ax.scatter(x_RC, scores_RC, alpha=alpha, s=size, color="blue",edgecolor='none')

    else:
        ax.scatter([positions[0] + np.random.normal()*0.05 for i in range(len(scores_others))], 
                    scores_others, alpha = alpha, s = size, color = "gray", label = "Non-member",edgecolor='none')
        ax.scatter([positions[1] + np.random.normal()*0.05 for i in range(len(scores_mutants))], 
                    scores_mutants, alpha = alpha, s = size, color = "blue",edgecolor='none', label = "sRC"); 
        
    vp = plt.violinplot([data[0][measure], data[1][measure]], positions, widths = [0.38, 0.25], showextrema = False, showmedians=True)
    vp['bodies'][0].set_facecolor('gray')
    vp['bodies'][1].set_facecolor('blue')
    if 'cmedians' in vp:  # Safety check
        vp['cmedians'].set_linewidth(5)  # Directly set width on LineCollection
        vp['cmedians'].set_color('k')  # Set color
        vp['cmedians'].set_linestyle('-')  # Ensure solid line


    add_significance(data, measure, ax, bp, stat)
    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization}, inGroupNorm = {in_group_norm}\n logscale = {logscale}, permutation test on the {stat}."
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return bp
    

def bp_metric_mutants(measure, window = 1, variable = "approaches",
                      mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean", swarmplot = True, ax = None):
    """
    Plots and compares the specified graph metric between mutant and non-mutant mice across all groups.

    Generates a boxplot (and optional swarm/violin plots) to visualize differences in a time-resolved 
    graph-theoretical measure, with statistical testing between groups. For more details on input parameters, 
    see the documentation of the time_measures function. 
    """
    scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, window, variable, mnn, mutual,
                            weighted, threshold, summation, normalization, in_group_norm, logscale)
        scores_mutants.extend(res[0])
        scores_rc.extend(res[1])
        scores_others.extend(res[2])
        scores_wt.extend(res[3])
        scores_all.extend(res[4])
        RFIDs.extend(res[5]["Mouse_RFID"])
        mutants.extend(res[5]["mutant"])
        RCs.extend(res[5]["RC"])
        
    df = pd.DataFrame()
    df[measure] = scores_all
    df["mutant"] = mutants
    df["RC"] = RCs
    df["Mouse_RFID"] = RFIDs
    
    data = [ df.loc[np.logical_and(df["mutant"], df["RC"] == False), ['Mouse_RFID', measure]], 
            df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', measure]] ]
    
    data[0][data[0] == np.inf] = np.nan
    data[1][data[1] == np.inf] = np.nan
    data[0].dropna()
    data[1].dropna()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 60, 0.4
    positions = [0.75, 1.25]
    bp = ax.boxplot([data[1][measure].dropna(), data[0][measure].dropna()], positions = positions, labels=["Non-members WT", "OXTRΔAON"],
                    showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    if swarmplot:
        x_mutants, scores_mutants = spread_points_around_center(scores_mutants, center=positions[1], bin_width = 0.1, interpoint=0.04)
        ax.scatter(x_mutants, scores_mutants, alpha=alpha, s=size, color="red",edgecolor='none')
        x_others, scores_others = spread_points_around_center(scores_others, center=positions[0], bin_width = 0.1, interpoint=0.03)
        ax.scatter(x_others, scores_others, alpha=alpha, s=size, color="gray", label="Non-member",edgecolor='none')
    else:
        ax.scatter([positions[1] + np.random.normal()*0.05 for i in range(len(scores_mutants))], 
                    scores_mutants, alpha = alpha, s = size, color = "red",edgecolor='none'); 
        ax.scatter([positions[0] + np.random.normal()*0.05 for i in range(len(scores_others))], 
                    scores_others, alpha = alpha, s = size, color = "gray", label = "Non-member",edgecolor='none')
        
    vp = plt.violinplot([data[1][measure], data[0][measure]], positions, widths = [0.38, 0.25], showextrema = False, showmedians=True)
    vp['bodies'][1].set_facecolor('lightcoral')
    vp['bodies'][0].set_facecolor('gray')
    if 'cmedians' in vp:  # Safety check
        vp['cmedians'].set_linewidth(5)  # Directly set width on LineCollection
        vp['cmedians'].set_color('k')  # Set color
        vp['cmedians'].set_linestyle('-')  # Ensure solid line


    add_significance(data, measure, ax, bp, stat)
    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization}, inGroupNorm = {in_group_norm}\n logscale = {logscale}, permutation test on the {stat}."
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return bp
    

if __name__ == "__main__":    
## Main
    bp_metric_mutants("summed outNEF", mnn = None, mutual = True, weighted = True, threshold = 0, 
                      summation = "mean", normalization="CV", logscale = False, stat = "median", swarmplot  = True)

    bp_metric_mutants("summed inNEF", mnn = None, mutual = True, weighted = True, threshold = 0, 
                      summation = "mean", normalization="CV", logscale = False, stat = "median")

## Supplement
    bp_metric_mutants("summed inburstiness", mnn = 7, mutual = True, weighted = False, threshold = 0, 
                              summation = "mean", normalization=None, in_group_norm = False, logscale = False, stat = "median", swarmplot = True)
    
    bp_metric_mutants("summed outburstiness", mnn = 7, mutual = True, weighted = False, threshold = 0, 
                              summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "median", swarmplot = True)
   
    bp_metric("summed outNEF", mnn = None, mutual = True, weighted = True, threshold = 0, 
                      summation = "mean", normalization="CV", logscale = False, stat = "median")
    
    bp_metric("summed inNEF", mnn = None, mutual = True, weighted = True, threshold = 0, 
                      summation = "mean", normalization="CV", logscale = False, stat = "median")
    
    

