## script for plotting boxplots and scatter plots of simple stats such as count of approaches, count of interactions, chasings etc...

import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import pandas as pd
import os
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, ttest_1samp, sem
from scipy import stats
import seaborn as sns
from tqdm import tqdm
from scipy.stats import rankdata
from scipy.stats import ks_2samp
from scipy.stats import median_abs_deviation
from scipy.stats import iqr
from statsmodels.robust.scale import qn_scale
from pathlib import Path
import os
import seaborn as sns
import ptitprince as pt

sys.path.append('C:\\Users\\corentin.nelias\\Documents\\GitHub\\sRC_backup\\src\\')
from read_graph import read_graph
from utils import format_plot, add_group_significance, spread_points_around_center, add_LME_significance

filter_type = "Rsgolay"
frame_len = 120
datapath = f"C:\\Users\\corentin.nelias\\Desktop\\DLC_outputs\\sarahs_data\\distance_threshold_analysis_{frame_len}f\\"
labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def mean_abs_deviation_median(data, axis = 0):
    data = np.array(data)
    med_delta = np.median(data, axis = axis)
    return np.nanmean(np.abs(data - med_delta), axis = axis)

def default_on_error(graph_idx, variable, dthresh, window):
    """
    If the data for a graph on a specific day is not available because of detection issue, nans should be returned. 
    To preserve the structure of the data array, the number of returned nans should reflect the number of mutants, rc members and wt in the cohort. 
    """
    mutants, rc, others, wt, RFIDs = get_category_indices(graph_idx, "approaches", dthresh, window) # load data from exp day 1 as a template, because all group were properly detected.
    return [np.nan]*len(mutants), [np.nan]*len(rc), [np.nan]*len(others), \
        [np.nan]*len(wt), [np.nan]*len(RFIDs), \
        {"Mouse_RFID": [np.nan]*len(RFIDs), "mutant": [np.nan]*len(RFIDs), "RC": [np.nan]*len(RFIDs), "Group_ID": int(labels[graph_idx][1:])}

def get_category_indices(graph_idx, variable, dthresh, window):
    """
    Returns the indices of mutant, RC (rich club), other, and wild-type (WT) mice for a given group.

    Parameters:
        graph_idx (int): Index of the group (corresponds to an entry in the 'labels' list).
        variable (str): The variable name used to construct the data file path (e.g., "approaches", "interactions", "chasing"...).

    Returns:
        mutants (np.ndarray): Indices of mutant mice in the group.
        rc (np.ndarray): Indices of RC (rich club) mice in the group.
        others (np.ndarray): Indices of mice that are neither mutant nor RC.
        wt (np.ndarray): Indices of wild-type mice in the group.
        RFIDs (np.ndarray): Array of Mouse RFID strings corresponding to the group.

    Notes:
        - The function reads metadata and group data to determine group membership.
        - Mice not found in the metadata are treated as wild-type.
    """
    metadata_path = "C:\\Users\\corentin.nelias\\Documents\\GitHub\\sRC_backup\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)

    datapath = f"C:\\Users\\corentin.nelias\\Desktop\\DLC_outputs\\sarahs_data\\distance_threshold_analysis_{frame_len}f\\"+filter_type+ \
    "\\d"+dthresh+"\\"+labels[graph_idx]+"\\7day_basis\\approaches_resD7_1.csv"
    if window not in [1, 3, 7]:
        print("Incorrect time window")
        return 
    
    arr = np.loadtxt(datapath, delimiter=",", dtype=str)
    RFIDs = arr[0, 1:].astype(str)
    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    mutant_map = curr_metadata_df.set_index('Mouse_RFID')['mutant'].to_dict()
    is_mutant = [mutant_map.get(rfid, False) for rfid in RFIDs] # if RFID is missing, animal is assumed to be neurotypical

    graph_length = len(RFIDs)
    # is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
    is_RC = [
        True if (curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values.size > 0 and
             curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values[0]) else False
        for rfid in RFIDs
    ]
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_RC)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
    return mutants, rc, others, wt, RFIDs

        
def time_measures(measure, graph_idx, dthresh = "1.5", window = 1, variable = "approaches",
                  mnn = None, mutual = True, weighted = False, threshold = 0.0,
                  summation = "mean", normalization = None, in_group_norm = False, logscale = False):

    metadata_path = "C:\\Users\\corentin.nelias\\Documents\\GitHub\\sRC_backup\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    if window not in [1, 3, 7]:
        print("Incorrect time window")
        return 
        
    datapath = f"C:\\Users\\corentin.nelias\\Desktop\\DLC_outputs\\sarahs_data\\distance_threshold_analysis_{frame_len}f\\{filter_type}\d"+dthresh+"\\"+labels[graph_idx]+"\\approaches_D2.csv"
    
    try:
        data_ref = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
        arr = np.loadtxt(datapath, delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)
    except Exception as e:
        return default_on_error(graph_idx, variable, dthresh, window)
    if variable == "interactions" or variable == "t_mean":
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
        datapath = f"C:\\Users\\corentin.nelias\\Desktop\\DLC_outputs\\sarahs_data\\distance_threshold_analysis_{frame_len}f\\{filter_type}\\d"+dthresh+"\\"+labels[graph_idx]+"\\"+variable+f"_D"+str(day)+".csv"
        try:
            data = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
            arr = np.loadtxt(datapath, delimiter=",", dtype=str)
            RFIDs = arr[0, 1:].astype(str)
        except Exception as e:
            # print(e)
            data = data_ref*np.nan
        if variable == "interactions" or variable == "t_mean" or variable == "mean_dist" or variable == "t_summed":
            mode = 'undirected'
            data = (data + np.transpose(data))/2 # ensure symmetry
        elif variable == "approaches" or "HWI_t":
            mode = 'directed'
        else:
            raise NameError("Incorrect input argument for 'variable'.")
        
        if weighted:
            data = np.where(data > 0.01, data, 0)
        else:
            data = np.where(data > 0.01, 1, 0)

        graphs.append(data)
        
    if measure == "summed outNEF" or measure == "summed inNEF" or \
        measure == "summed outNEF rank" or measure == "summed inNEF rank":
        all_data = np.array(graphs)
        if in_group_norm:
            all_data = all_data/np.mean(all_data, axis = (0, 1, 2), keepdims=True)    
            # all_scores = all_scores/np.sum(all_scores)
        time_std, time_mean, time_median = np.nanstd(all_data, axis = 0), np.nanmean(all_data, axis = 0), np.nanmedian(all_data, axis = 0)
        time_std[time_std == 0] = np.nan # if time_std = 0, it means the animals was never detected over the whole experiment
        if normalization == "Poisson" or normalization == "poisson": # variant of the CV aimed at studying poissonicity
            all_scores = (time_std - time_mean)/(time_std + time_mean)
        elif normalization == "CV":
            all_scores = time_std/time_mean
        elif normalization == "median CV":
            all_scores = time_std/time_median
        elif normalization == "mdm CV":
            all_scores = mean_abs_deviation_median(all_data, axis = 0)/time_median
        elif normalization == "iqr CV":
            all_scores = iqr(all_data, axis = 0, nan_policy="omit")/time_median
        elif normalization == "qn CV":
            # all_scores = np.apply_along_axis(Sn, axis=0, arr=all_data, finite_corr=False)/time_median
            all_scores = qn_scale(all_data)/time_median
        elif normalization == "none":
            print("No normalization applied")
            all_scores = time_std
        else:
            raise TypeError("Input normalization type incorrect")
        all_scores[all_scores == np.inf] = np.nan

            
    elif measure == "summed outICI" or measure == "summed inICI": # summed average inter conter interval
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
        if logscale:
            all_scores += 1.0001
            
    else:
        raise Exception("Unknown or misspelled input measurement.") 
        return
    
    if "out" in measure:
        if summation == "mean":
            all_scores[np.isinf(all_scores)] = np.nan
            all_scores = np.nanmean(all_scores, axis = 1)
        if summation == "median":
            all_scores = np.nanmedian(all_scores, axis = 1)

        if logscale:
            all_scores = np.log(all_scores)
        if "rank" in measure:
            all_scores = all_scores.argsort().argsort()
        # if in_group_norm:
        #     all_scores = all_scores/np.max(all_scores)
       
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
        # if in_group_norm:
        #     all_scores = all_scores/np.max(all_scores)
        metric_mutants = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in mutants]

        metric_rc = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in rc]
        metric_wt = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in wt]
        metric_others = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in others]
        metric_all = [all_scores[i] if all_scores[i] != np.inf else np.nan for i in range(len(RFIDs))]

    return metric_mutants, metric_rc, metric_others, metric_wt, metric_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}


def bp_metric_RC_vs_OXTR(measure, dthresh = "3.1", mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean", swarmplot = True, split_view = False, ax = None):
    """
    Plots and compares the specified graph metric between mutant and non-mutant mice across all groups.

    Generates a boxplot (and optional swarm/violin plots) to visualize differences in a time-resolved 
    graph-theoretical measure, with statistical testing between groups. For more details on input parameters, 
    see the documentation of the time_measures function. 
    """
    scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, dthresh, 1, "approaches", mnn, mutual,
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
    
        
    data = [ df.loc[np.logical_and(df["mutant"] == False, df["RC"] == True), ['Mouse_RFID', measure]], 
            df.loc[np.logical_and(df["mutant"], df["RC"] == False), ['Mouse_RFID', measure]] ] 
    
    data[0][data[0] == np.inf] = np.nan
    data[1][data[1] == np.inf] = np.nan
    data[0].dropna()
    data[1].dropna()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 60, 0.4
    positions = [0.75, 1.25]
    bp = ax.boxplot([data[1][measure].dropna(), data[0][measure].dropna()], positions = positions, labels=["OXTRΔAON","non-OXTRΔAON sRC"],
                    showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    if swarmplot:
        x_mutants, scores_mutants = spread_points_around_center(scores_mutants, center=positions[0], bin_width = 0.1, interpoint=0.04)
        ax.scatter(x_mutants, scores_mutants, alpha=alpha, s=size, color="red",edgecolor='none')
        x_rc, scores_rc = spread_points_around_center(scores_rc, center=positions[1], bin_width = 0.1, interpoint=0.03)
        ax.scatter(x_rc, scores_rc, alpha=alpha, s=size, color="blue", label="Non-member",edgecolor='none')
    else:
        ax.scatter([positions[0] + np.random.normal()*0.05 for i in range(len(scores_mutants))], 
                    scores_mutants, alpha = alpha, s = size, color = "red",edgecolor='none'); 
        ax.scatter([positions[1] + np.random.normal()*0.05 for i in range(len(scores_others))], 
                    scores_others, alpha = alpha, s = size, color = "blue", label = "Non-member",edgecolor='none')
    
    # add_significance(data, measure, ax, bp, stat)
    if stat == "LME":
        add_LME_significance(data, measure, ax, bp)
    else:
        add_group_significance([data[0][measure].values, data[1][measure].values], [data[0]["Mouse_RFID"], data[1]["Mouse_RFID"]], ax, bp, stat)
    
    if logscale:
        data[0].replace([np.inf, -np.inf], np.nan, inplace=True)
        data[1].replace([np.inf, -np.inf], np.nan, inplace=True)
    vp = plt.violinplot([data[1][measure].dropna(), data[0][measure].dropna()], positions, widths = [0.35, 0.23], showextrema = False, showmedians=True)
    vp['bodies'][0].set_facecolor('red')
    vp['bodies'][1].set_facecolor('blue')
    if 'cmedians' in vp:  # Safety check
        vp['cmedians'].set_linewidth(5)  # Directly set width on LineCollection
        vp['cmedians'].set_color('k')  # Set color
        vp['cmedians'].set_linestyle('-')  # Ensure solid line

    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization},\
    inGroupNorm = {in_group_norm}\n logscale = {logscale}, permutation test on the {stat}\n"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return bp
    
    
def bp_metric_approaches(measure, dthresh = "3.1", mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean", swarmplot = True, include_RC = False, ax = None):
    """
    Plots and compares the specified graph metric between mutant and non-mutant mice across all groups.

    Generates a boxplot (and optional swarm/violin plots) to visualize differences in a time-resolved 
    graph-theoretical measure, with statistical testing between groups. For more details on input parameters, 
    see the documentation of the time_measures function. 
    """
    scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, dthresh, 1, "approaches", mnn, mutual,
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
    
    if include_RC:
        data = [ df.loc[np.logical_and(df["mutant"], df["RC"] == False), ['Mouse_RFID', measure]], 
                pd.concat([df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', measure]], df.loc[np.logical_and(df["mutant"] == False, df["RC"] == True), ['Mouse_RFID', measure]]]) ]
    else:
        
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
    if include_RC:
        bp = ax.boxplot([data[1][measure].dropna(), data[0][measure].dropna()], positions = positions, labels=["Controls", "OXTRΔAON"],
                        showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    else:
        bp = ax.boxplot([data[1][measure].dropna(), data[0][measure].dropna()], positions = positions, labels=["Non-members WT", "OXTRΔAON"],
                            showfliers = False, meanline=False, showmeans = False, medianprops={'visible': False})
    if swarmplot:
        x_mutants, scores_mutants = spread_points_around_center(scores_mutants, center=positions[1], bin_width = 0.1, interpoint=0.04)
        ax.scatter(x_mutants, scores_mutants, alpha=alpha, s=size, color="red",edgecolor='none')
        if not include_RC:
            x_others, scores_others = spread_points_around_center(scores_others, center=positions[0], bin_width = 0.1, interpoint=0.03)
            ax.scatter(x_others, scores_others, alpha=alpha, s=size, color="gray", label="Non-member",edgecolor='none')
        else:
            x_others, scores_others = spread_points_around_center(scores_others+scores_rc, center=positions[0], bin_width = 0.1, interpoint=0.03)
            ax.scatter(x_others, scores_others, alpha=alpha, s=size, color="gray", label="Non-member",edgecolor='none')
    else:
        ax.scatter([positions[1] + np.random.normal()*0.05 for i in range(len(scores_mutants))], 
                    scores_mutants, alpha = alpha, s = size, color = "red",edgecolor='none'); 
        ax.scatter([positions[0] + np.random.normal()*0.05 for i in range(len(scores_others))], 
                    scores_others, alpha = alpha, s = size, color = "gray", label = "Non-member",edgecolor='none')
    
    # add_significance(data, measure, ax, bp, stat)
    if stat == "LME":
        add_LME_significance(data, measure, ax, bp)
    else:
        add_group_significance([data[0][measure].values, data[1][measure].values], [data[0]["Mouse_RFID"], data[1]["Mouse_RFID"]], ax, bp, stat)
    
    if logscale:
        data[0].replace([np.inf, -np.inf], np.nan, inplace=True)
        data[1].replace([np.inf, -np.inf], np.nan, inplace=True)
    vp = plt.violinplot([data[1][measure].dropna(), data[0][measure].dropna()], positions, widths = [0.35, 0.23], showextrema = False, showmedians=True)
    vp['bodies'][1].set_facecolor('lightcoral')
    vp['bodies'][0].set_facecolor('gray')
    if 'cmedians' in vp:  # Safety check
        vp['cmedians'].set_linewidth(5)  # Directly set width on LineCollection
        vp['cmedians'].set_color('k')  # Set color
        vp['cmedians'].set_linestyle('-')  # Ensure solid line

    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization},\
    inGroupNorm = {in_group_norm}\n logscale = {logscale}, permutation test on the {stat}\n"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return bp
    
    
def tuning_curve(measure, dthresh, mnn = None, mutual = True, weighted = False, threshold = 0.0,
                      summation = "mean", normalization = None, in_group_norm = False, logscale = False, stat = "mean"):
    """
    Plots the approach tuning curve of significance for specified graph metric between mutant and non-mutant mice across all groups.
    For more details on input parameters, see the documentation of the time_measures function. 
    """
    p_values, deltas = [], []

    for th in dthresh:
        scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
        RFIDs, mutants, RCs = [], [], []
        for graph_idx in range(len(labels)):
            res = time_measures(measure, graph_idx, th, 1, "approaches", mnn, mutual,
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
    
        if stat != "LME":
            p = add_group_significance([data[0][measure].values, data[1][measure].values], [data[0]["Mouse_RFID"], data[1]["Mouse_RFID"]], stat = stat)
        elif stat == "LME":
            p, result = add_LME_significance(data, measure)
            # plot_lme_diagnostics(result, [0]*len(data[0].dropna())+[1]*len(data[1].dropna()))
        p_values.append(p)
        deltas.append(np.nanmean(data[0][measure].values) - np.nanmean(data[1][measure].values))

    title = f"{measure}\n mnn = {mnn} thresh = {threshold}, summation = {summation}\n norm. = {normalization},\
    inGroupNorm = {in_group_norm}\n logscale = {logscale}, "
    if stat == "LME":
        title += "statistical test on LME"
    else:
        title += f"permutation test on {stat}"
        
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(np.round(dthresh.astype(float), 2), p_values, marker='o', c = "k", linewidth=2, markersize=4, label='p-value')
    if "out" in measure:
        axs[0].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
    else:
        axs[0].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')

    axs[0].set_xlabel('Distance Threshold', fontsize=12)
    axs[0].set_ylabel('p-value', fontsize=12)
    axs[0].legend(fontsize=11)
    axs[1].plot(np.round(dthresh.astype(float), 2),deltas, marker='o', c = "k", linewidth=2, markersize=4, label='p-value')
    axs[1].set_xlabel('Distance Threshold', fontsize=12)
    axs[1].set_ylabel(r'μ_{OXTR} - μ_{WT}', fontsize=12)
    fig.suptitle(title)
    return 

    
if __name__ == "__main__":
## Main
    # NEF comparisons
    bp_metric_approaches(f"summed outNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
                        summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu")
    
    bp_metric_approaches(f"summed inNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
                        summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu")
    
    # NEF tuning curves
    # dths = np.round(np.arange(2.5, 4.5, 0.2), 1).astype(str) # approach distance threshold
    
    # tuning_curve(f"summed inNEF", dths, mnn = None, mutual = False, weighted = True, threshold = 0, 
    #                     summation ="mean", normalization="mdm CV", logscale = False, stat = "LME")
    
    # tuning_curve(f"summed outNEF", dths, mnn = None, mutual = False, weighted = True, threshold = 0, 
    #                     summation ="mean", normalization="meDM CV", logscale = False, stat = "LME")

    
# Supplement
    # NEF comparison of OXTR to all control mice
    # bp_metric_approaches(f"summed outNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
    #                     summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu", include_RC = True)
    # bp_metric_approaches(f"summed inNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
                        # summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu", include_RC = True)
                        
    # NEF comparison of OXTR to sRC control mice
    # bp_metric_RC_vs_OXTR(f"summed outNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
    #                     summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu")
    # bp_metric_RC_vs_OXTR(f"summed inNEF", "3.0", mnn = None, mutual = False, weighted = True, threshold = 0, 
    #                     summation ="mean", normalization="mdm CV", logscale = True, stat = "mannwhitneyu")
    
   


 