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
                print("Mutant in metadata not found in matrix, treat as MUTANT anyways")
                is_mutant.append(True)
                
    mutants_RFIDs = RFIDs[is_mutant]
    mutant_count = 0
    for sRC_member in sRC:
        if sRC_member in mutants_RFIDs:
            mutant_count += 1
            
    return sRC, mutant_count, mutants_RFIDs         

def process_all_cohorts(variable = "interactions", window = 3, deg = 3, nn = 3, mutual = True, rm_weak_histo = True, verbose = True):
    """
    Computes the sRC for all cohorts 

    """
    sRC, sRC_count, mut_count, mutants = [], 0, 0, []
        
    print(colored("------------------------  sRC list --------------------------", "green"))
    params =  f"------ Variable: {variable}, time window: {window},  mnn: {nn} ------" if mutual else f"Variable: {variable}, time window: {window}, deg:{deg}, nn:{nn}"
    print(colored(params, "green"))
   
    for graph_idx in range(len(labels)):
        res = get_rc(graph_idx, variable, window, deg, nn, mutual, rm_weak_histo)
        sRC.append(res[0])
        sRC_count += len(res[0])
        mut_count += res[1]
        mutants.append(res[2])
        print(f"{labels[graph_idx]}: {res[0]}")

        
    print(f"{sRC_count} members")
    print(colored(f"{mut_count} mutants detected in the stable rich club!", "red"))
    return sRC, mut_count, mutants

def get_results_significance(sRC,  mut_count, rm_weak = True):
    number_of_hits = []
    not_applicable = [0]
    single_mutants = np.array([1])
    double_mutants = np.array([1, 2])
    triple_mutants = np.array([1,2,3])
    quadruple_mutants = np.array([1,2,3,4])
    quintuple_mutants = np.array([1,2,3,4,5])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    if rm_weak:
        # for mnn = 3
        mutants = [single_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, single_mutants,
                              quadruple_mutants, double_mutants, not_applicable, not_applicable, quadruple_mutants, not_applicable, double_mutants]
        # for mnn = 4
        # mutants = [single_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, single_mutants,
        #                       quadruple_mutants, double_mutants, not_applicable, not_applicable, quadruple_mutants, triple_mutants, double_mutants]
    else:
        mutants = [double_mutants, double_mutants,double_mutants,double_mutants,double_mutants,double_mutants,double_mutants, double_mutants, double_mutants, 
                              quadruple_mutants, quintuple_mutants, triple_mutants, triple_mutants, triple_mutants]

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(1000)):
        hits = 0
        for idx, rc in enumerate(sRC):
            np.random.shuffle(shuffled_arr)
            for j in mutants[idx]:
                if j in shuffled_arr[:len(rc)]:
                    hits += 1
        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    # plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(mut_count, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.cumsum(h[0]), 4)[mut_count]), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    sRC, mut_count, mutants = process_all_cohorts("approaches", 3, 3, 3, True, True)
    get_results_significance(sRC, 1)
    
      
    
    