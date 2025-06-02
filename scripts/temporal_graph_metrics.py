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
from utils import get_category_indices
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats
from graph_metrics import add_significance

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def default_on_error(graph_idx, variable, window, rm_weak_histo = True):
    """
    If the data for a graph on a specific day is not available because of detection issue, nans should be returned. 
    To preserve the structure of the data array, the number of returned nans should reflect the number of mutants, rc members and wt in the cohort. 
    """
    mutants, rc, others, wt, RFIDs = get_category_indices(graph_idx, "approaches", window, rm_weak_histo) # load data from exp day 1 as a template, because all group were properly detected.
    return [np.nan]*len(mutants), [np.nan]*len(rc), [np.nan]*len(others), \
        [np.nan]*len(wt), [np.nan]*len(RFIDs), \
        {"Mouse_RFID": [np.nan]*len(RFIDs), "mutant": [np.nan]*len(RFIDs), "RC": [np.nan]*len(RFIDs), "Group_ID": int(labels[graph_idx][1:])}

def time_measures(measure, graph_idx, window = 1, variable = "approaches", mnn = None, mutual = True, weighted = False, rm_weak_histo = True, threshold = 0.0):
    """
    Computes specified time graph-theoretical metric for a given cohort, and time window.

    Parameters:
        graph_idx (int): Index of the group/cohort to analyze (corresponds to an entry in the 'labels' list).
        window (int, optional): Size of the time window for averaging the graph representation (default is 3).
        metric (str, optional): Name of the graph-theoretical measure to compute (e.g., "hub", "degree", "pagerank").
        variable (str, optional): Type of interaction to analyze ("approaches" or "interactions").
        mnn (int or None, optional): Minimum number of neighbors for graph thresholding (used for unweighted measures).
        mutual (bool, optional): Whether to use mutual nearest neighbors for graph construction.
            if mnn is None (used for weighted metrics), mutual is ignored.
        rm_weak_histo (bool, optional): If True, removes mutants with weak histology from analysis.

    Returns:
        tuple: (scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, metadata)
            - scores_mutants: List of measure values for mutant mice.
            - scores_rc: List of measure values for RC (rich club) mice.
            - scores_rest: List of measure values for other mice (not mutant or RC).
            - scores_wt: List of measure values for wild-type mice.
            - scores_all: List of measure values for all mice in the group.
            - metadata: Dictionary with keys "Mouse_RFID", "mutant", "RC", and "Group_ID" for the current group.
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
        return default_on_error(graph_idx, variable, window, rm_weak_histo)
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
    if not rm_weak_histo:
        is_mutant = [True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
    else:
        is_mutant = []
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
                # print("Mouse not found, treat as WT")
                is_mutant.append(False)

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

    elif measure == "summed outburstiness" or measure == "summed inburstiness":
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

        time_std, time_mean = np.nanstd(all_data, axis = 0), np.nanmean(all_data, axis = 0)
        all_scores = (time_std - time_mean)/(time_std + time_mean)
        all_Scores = time_std
        
    elif measure == "summed outNEF" or measure == "summed inNEF": # NEF = normalized edge fluctuations
        all_data = np.array(graphs)
        time_std, time_mean = np.nanstd(all_data, axis = 0), np.nanmean(all_data, axis = 0)
        all_scores = time_std
        
    else:
        raise Exception("Unknown or misspelled input measurement.") 
        return

    if "out" in measure:
        metric_mutants = [np.nansum(all_scores[i, :]) for i in mutants]
        metric_rc = [np.nansum(all_scores[i, :]) for i in rc]
        metric_wt = [np.nansum(all_scores[i, :]) for i in wt]
        metric_others = [np.nansum(all_scores[i, :]) for i in others]
        metric_all = [np.nansum(all_scores[i, :]) for i in range(len(RFIDs))]
        
    elif "in" in measure:
        metric_mutants = [np.nansum(all_scores[:, i]) for i in mutants]
        metric_rc = [np.nansum(all_scores[:, i]) for i in rc]
        metric_wt = [np.nansum(all_scores[:, i]) for i in wt]
        metric_others = [np.nansum(all_scores[:, i]) for i in others]
        metric_all = [np.nansum(all_scores[:, i]) for i in range(len(RFIDs))]

    
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

def bp_metric_mutants(measure, window = 1, variable = "approaches", mnn = None, mutual = True, weighted = False, threshold = 0.0, ax = None):
    scores_mutants, scores_rc, scores_wt, scores_others, scores_all = [], [], [], [], []
    RFIDs, mutants, RCs = [], [], []
    for graph_idx in range(len(labels)):
        res = time_measures(measure, graph_idx, window, variable, mnn, mutual, weighted, True, threshold)
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
    
    data = [df.loc[np.logical_and(df["mutant"], df["RC"] == False), ['Mouse_RFID', measure]], 
            df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', measure]] ]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    size, alpha = 40, 0.5
    bp = ax.boxplot([scores_mutants, scores_others], labels=["Mutants", "Non-members WT"], showfliers = False)
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(scores_mutants))], 
                scores_mutants, alpha = alpha, s = size, color = "red"); 
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(scores_others))], 
                scores_others, alpha = alpha, s = size, color = "gray", label = "Non-member")
    add_significance(data, measure, ax, bp)
    title = f"weighted {measure}, mnn = {mnn}, thresh = {threshold}%" if weighted else f"Unweighted {measure}, mnn = {mnn}, thresh = {threshold}%"
    ax.set_title(title)
    plt.show()
    
temporal_metrics = ["summed outNEF", "summed inNEF", "summed outICI", "summed inICI", "summed outburstiness", "summed inburstiness"]


if __name__ == "__main__":
    # bp_metric_mutants("summed outburstiness", mnn = None, mutual = True, weighted = False, threshold=5)    
    bp_metric_mutants("summed outNEF", mnn = None, mutual = False, weighted = False, threshold = 5)

    
