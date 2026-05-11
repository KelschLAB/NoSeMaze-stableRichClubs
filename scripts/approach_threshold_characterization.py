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

def approach_x_threshold(out = True, show_rc = False, thresholds = ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3"], sep = True):
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """
    fig, axs = plt.subplots(1, 1, figsize = (5, 6))
    
    wt_mean, oxtr_mean, src_mean, nonsrc_mean, all_mean = [], [], [], [], []
    wt_std, oxtr_std, src_std, nonsrc_std, all_std = [], [], [], [], []
    for dthresh in thresholds:
        approaches_wt, approaches_mutants, approaches_src, approaches_nonrc, approach_all = [], [], [], [], []
        for idx, g in enumerate(labels):
            this_mutants, this_rc, _, this_wt, _ = get_category_indices(idx, "approaches", dthresh, 7)
    
            data = read_graph([datapath+filter_type+"\\d"+dthresh+"\\"+g+"\\7day_basis"+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + \
            read_graph([datapath+filter_type+"\\d"+dthresh+"\\"+g+"\\7day_basis"+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0]
    
            # helper depending on direction
            def val(i):
                return np.sum(data[:, i]) if out else np.sum(data[i, :])
            
            approach_all.append(np.mean([val(i) for i in range(len(this_wt)+len(this_mutants))]))
            if not show_rc:
                approaches_wt.append(np.mean([val(wt) for wt in this_wt]))
                approaches_mutants.append(np.mean([val(mutant) for mutant in this_mutants]))
            else:
                approaches_nonrc.append(
                    np.mean(
                        [val(nonrc) for nonrc in this_wt.tolist()+this_mutants.tolist() if nonrc not in this_rc]
                    )
                )
            approaches_src.append(np.mean([val(rc) for rc in this_rc]))
        
        all_mean.append(np.mean(approach_all))
        all_std.append(sem(approach_all))
        if not show_rc:
            wt_mean.append(np.mean(approaches_wt))
            oxtr_mean.append(np.mean(approaches_mutants))
            wt_std.append(sem(approaches_wt))
            oxtr_std.append(sem(approaches_mutants))
        else:
            nonsrc_mean.append(np.mean(approaches_nonrc))
            nonsrc_std.append(sem(approaches_nonrc))
            src_mean.append(np.nanmean(approaches_src))
            src_std.append(sem(approaches_src, nan_policy = "omit"))

    threshold_values = [float(t) for t in thresholds]
    
    
    if not show_rc:
        axs.errorbar(threshold_values, wt_mean, yerr=wt_std, 
                     marker='o', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, label='WT', color='gray')
        
        axs.errorbar(threshold_values, oxtr_mean, yerr=oxtr_std,
                     marker='s', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, label='Oxtr', color='red')
    elif sep:
        axs.errorbar(threshold_values, nonsrc_mean, yerr=nonsrc_std, 
                     marker='o', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, label='non-sRC', color='gray')
        
        axs.errorbar(threshold_values, src_mean, yerr= src_std,
                     marker='s', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, label='sRC', color='blue')
    else:
        axs.errorbar(threshold_values, all_mean, yerr=all_std, 
                     marker='o', linestyle='-', linewidth=2, markersize=8,
                     capsize=5, capthick=2, color='k')
    
    plt.tight_layout()
    plt.show()
    
    return fig, threshold_values, [src_mean, src_std], [nonsrc_mean, nonsrc_std]
        
def boxplot_approaches_review(out = True, dthresh = "1.5", save_path = None):
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """
    fig, axs = plt.subplots(1, 2, figsize = (5, 6))

    approaches_wtrc, approaches_wt, approaches_mutants, approaches_mutantsrc = [], [], [], []
    RFIDs_RC, RFIDs_nonRC = [], []
    RFIDs_WTnonRC, RFIDs_OXTRnonRC = [], []

    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, this_wt, _ = get_category_indices(idx, "approaches", dthresh, 7)

        data = read_graph([datapath+filter_type+"\\d"+dthresh+"\\"+g+"\\7day_basis"+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + \
        read_graph([datapath+filter_type+"\\d"+dthresh+"\\"+g+"\\7day_basis"+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0]
        arr = np.loadtxt(datapath+filter_type+"\\d"+dthresh+"\\"+g+"\\7day_basis"+"\\approaches_resD7_1.csv", delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)

        # helper depending on direction
        def val(i):
            return np.sum(data[:, i]) if out else np.sum(data[i, :])

        # mutants
        for mutant in this_mutants:
            if mutant in this_rc:
                approaches_mutantsrc.append(val(mutant))
                RFIDs_RC.append(RFIDs[mutant])
            else:
                approaches_mutants.append(val(mutant))
                RFIDs_nonRC.append(RFIDs[mutant])
                RFIDs_OXTRnonRC.append(RFIDs[mutant])

        # wild-types
        for wt in this_wt:
            if wt in this_rc:
                approaches_wtrc.append(val(wt))
                RFIDs_RC.append(RFIDs[wt])
            else:
                approaches_wt.append(val(wt))
                RFIDs_nonRC.append(RFIDs[wt])
                RFIDs_WTnonRC.append(RFIDs[wt])

    data = [approaches_wtrc + approaches_mutantsrc, approaches_wt + approaches_mutants]

    bp1 = axs[0].boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
    plt.setp(bp1['medians'], color='k')
    axs[0].set_xticks([1,2], ["sRC", "non-sRC"])
    add_group_significance([data[0], data[1]], [RFIDs_RC, RFIDs_nonRC], ax = axs[0], bp = bp1, stat = "median")

    alpha, size  = 1, 40
    colors = ["gray", "red", "lightgray", "red"]
    axs[0].scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wtrc))], 
                approaches_wtrc, alpha = alpha, c = colors[0], s = size); 
    axs[0].scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_mutantsrc))], 
                approaches_mutantsrc, alpha = alpha, c = colors[1], s = size); 
    axs[0].scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_wt))], 
                approaches_wt, alpha = alpha, c = colors[2], s = size); 
    axs[0].scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutants))], 
                approaches_mutants, alpha = alpha, c = colors[3], s = size);
    axs[0].set_ylabel("Outgoing approaches" if out else "Ingoing approaches",
                  fontsize=20)
    axs[0].set_title("dist threshold = "+dthresh)
    
    data_genotype = [approaches_wt, approaches_mutants]

    bp2 = axs[1].boxplot(data_genotype, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
    plt.setp(bp2['medians'], color='k')
    axs[1].set_xticks([1,2], ["non-sRC WT", "non-sRC OXTR"])
    add_group_significance([data_genotype[0], data_genotype[1]], [RFIDs_WTnonRC, RFIDs_OXTRnonRC], stat = "median")

    alpha, size  = 1, 40
    colors = ["lightgray", "red"]
    axs[1].scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wt))], 
                approaches_wt, alpha = alpha, c = colors[0], s = size); 
    axs[1].scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutants))], 
                approaches_mutants, alpha = alpha, c = colors[1], s = size);
    axs[1].set_ylabel("Outgoing approaches" if out else "Ingoing approaches",
                  fontsize=20)
    axs[1].set_title("dist threshold = "+dthresh)

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        
def activity_histogram(dthresh = "1.5", variable = "approaches", out = True, bins = 20,
                  mnn = None, mutual = True, weighted = True, threshold = 0.0,
                  summation = "mean", normalization = None, in_group_norm = False, logscale = False):

    metadata_path = "C:\\Users\\corentin.nelias\\Documents\\GitHub\\sRC_backup\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    out_scores_mutants, out_scores_wt, in_scores_mutants, in_scores_wt  = [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        print("Graph "+str(graph_idx))
        datapath = f"C:\\Users\\corentin.nelias\\Desktop\\DLC_outputs\\sarahs_data\\distance_threshold_analysis_{frame_len}f\\{filter_type}\d"+dthresh+"\\"+labels[graph_idx]+"\\approaches_D2.csv"

        try:
            data_ref = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
            arr = np.loadtxt(datapath, delimiter=",", dtype=str)
            RFIDs = arr[0, 1:].astype(str)
        except Exception as e:
            return default_on_error(graph_idx, variable, dthresh, 1)
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
        for day in np.arange(1, 16, 1):
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
            all_data = np.array(graphs)
            
            for i in mutants:
                out_scores_mutants.extend(all_data[:, i, :].flatten())
            for wt_mouse in wt:
                out_scores_wt.extend(all_data[:, wt_mouse, :].flatten())
            for i in mutants:
                in_scores_mutants.extend(all_data[:, :, i].flatten())
            for wt_mouse in wt:
                in_scores_wt.extend(all_data[:, :, wt_mouse].flatten())
    
    plt.figure()
    if out:
        plt.hist(np.array(out_scores_mutants)**(1/3), density = True, label = "OXTR", bins = bins)
        plt.hist(np.array(out_scores_wt)**(1/3), density= True, label = "WT", alpha = 0.6, bins = bins)
    if not out:
        plt.hist(np.array(in_scores_mutants)**(1/3), density = True, label = "OXTR", bins = bins)
        plt.hist(np.array(in_scores_wt)**(1/3), density= True, label = "WT", alpha = 0.6, bins = bins)
        
    plt.show()

    if out:
        plt.title("outgoing")
    else:
        plt.title("incoming")
    plt.yticks(np.arange(0, 2, 0.1))
    plt.grid()
    plt.legend()

    return



    
if __name__ == "__main__":
# Supplement
    # evolution of approach numbers as function of threshold 
    dths = np.round(np.arange(1, 6, 0.2), 1).astype(str) # approach distance threshold
    fig, thresh, src, nonsrc = approach_x_threshold(True, True, dths)     
    # approach_x_threshold(False, False, dths)      
    # approach_x_threshold(True, True, dths, sep = False)     
    # approach_x_threshold(False, True, dths, sep = False)  
    


 