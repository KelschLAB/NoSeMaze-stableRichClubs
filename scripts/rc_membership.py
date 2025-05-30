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
from read_graph import read_graph, read_labels, rich_club_weights
from utils import get_category_indices
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats
from collections import Counter
from termcolor import colored

import warnings

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def get_rc(graph_idx, variable = "interactions", window = 3, deg = 3, nn = 3, mutual = True, rm_weak_histo = True):
    """Detects the stable rich club members in the graph indexed by 'graph_idx' after the cut specified by the input params.
    print the RFIDs of the detected members.

    Parameters:
        graph_idx (int): Index of the group/cohort to analyze (corresponds to an entry in the 'labels' list).
        day (int): The day or timepoint for which to compute the measure (depends on the window size).
        window (int, optional): Size of the time window for averaging the graph representation (default is 3).
        variable (str, optional): Type of interaction to analyze ("approaches" or "interactions").
        deg (int, optional): Degree of the rich club to look for. If mutual is True, this should be at most the 
            value of 'mnn'. (default is 3).
        mnn (int or None, optional): Minimum number of neighbors for graph thresholding (used for unweighted measures).
        mutual (bool, optional): Whether to use mutual nearest neighbors for graph construction.
            if mnn is None (used for weighted metrics), mutual is ignored.
        rm_weak_histo (bool, optional): If True, removes mutants with weak histology from analysis.
    
    Returns:
        tags (list): Lists of RFIDs tags of the stable RC members.
        mutant_count: number of mutants in stable RC.
    """
    assert window in [1, 3, 7], "Incompatible time window, should be 1, 3 or 7."
    assert variable == "interactions" or variable == "approaches" or variable == "t_mean"
    
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    tags = [] # list to append the RFIDs tag of the RC members before stability detection
    days = np.arange(1, 15//window+1)
    recorded_days = 0 # counter to know the actual number of days recorded (for some groups, some days are missing).
    for day in days:
        datapath = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_"+str(day)+".csv"
    
        try: # try loading the data. if it is missing, ignore and continue loop
            data = read_graph([datapath], percentage_threshold = 0, mnn = nn, mutual = mutual)[0]
            arr = np.loadtxt(datapath, delimiter=",", dtype=str)
            RFIDs = arr[0, 1:].astype(str)
            
        except Exception as e:
            continue
        
        if variable == "interactions" or variable == "t_mean":
            mode = 'undirected'
            data = (data + np.transpose(data))/2 # ensure symmetry
        elif variable == "approaches":
            mode = 'directed'

        #data = np.where(data > 0.01, data, 0)
        g = ig.Graph.Weighted_Adjacency(data, mode=mode)
        rc = rich_club_weights(g, deg, 0.01)
        rc = np.where(np.array(rc) > 0.2)
        tags.extend(RFIDs[rc])
        recorded_days += 1 
        
    tag_counts = Counter(tags)
    unique_RFIDs = np.array(list(tag_counts.keys()))
    times_in_rc = np.array(list(tag_counts.values()))
    sRC = unique_RFIDs[np.where(times_in_rc >=  0.8*len(days))] # RC member if it is in rich club in at least 80% of experimental days
        
    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    if not rm_weak_histo:
        is_mutant = [True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
    else:
        is_mutant = []
        for rfid in RFIDs:
            # if this RFID is known in the metadata and corresponding animal is a mutant
            if len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                histology = metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "histology"].values[0]
                if histology != 'weak': # mouse is a mutant and histology is strong or unknown
                    is_mutant.append(True)
                else:
                    is_mutant.append(False)
            # if RFID is known in metadata but corresponding animal is not a mutant
            elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and not metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                is_mutant.append(False)
            # else, treat animal as WT
            elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) == 0: 
                is_mutant.append(False)
                
    mutants_RFIDs = RFIDs[is_mutant]
    mutant_count = 0
    for sRC_member in sRC:
        if sRC_member in mutants_RFIDs:
            mutant_count += 1
            
    return sRC, mutant_count

def process_all_cohorts(variable = "interactions", window = 3, deg = 3, nn = 3, mutual = True, rm_weak_histo = True, verbose = True):
    """
    Computes the sRC for all cohorts 

    """
    sRC, mut_count = [], 0
    
    variable, window = "approaches",  3
    deg, nn, mutual = 3, 3, True
    rm_weak_histo = True
    
    print(colored("------------------------  sRC list --------------------------", "green"))
    params =  f"------ Variable: {variable}, time window: {window},  mnn: {nn} ------" if mutual else f"Variable: {variable}, time window: {window}, deg:{deg}, nn:{nn}"
    print(colored(params, "green"))
   
    for graph_idx in range(len(labels)):
        res = get_rc(graph_idx, variable, window, deg, nn, mutual, rm_weak_histo)
        sRC.extend(res[0])
        mut_count += res[1]
        print(f"{labels[graph_idx]}: {res[0]}")

        
    print(f"{len(sRC)} members")
    print(colored(f"{mut_count} mutants detected in the stable rich club!", "red"))

if __name__ == "__main__":
    

    
      
    
    