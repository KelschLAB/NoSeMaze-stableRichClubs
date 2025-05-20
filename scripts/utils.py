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
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def get_category_indices(graph_idx, variable, rm_weak_histo = True):
    """
    Returns the indices of mutant, RC (rich club), other, and wild-type (WT) mice for a given group.

    Parameters:
        graph_idx (int): Index of the group (corresponds to an entry in the 'labels' list).
        variable (str): The variable name used to construct the data file path (e.g., "approaches", "interactions", "chasing"...).
        rm_weak_histo (bool, optional): If True, excludes mutants with weak histology from the mutant group (default: True).

    Returns:
        mutants (np.ndarray): Indices of mutant mice in the group.
        rc (np.ndarray): Indices of RC (rich club) mice in the group.
        others (np.ndarray): Indices of mice that are neither mutant nor RC.
        wt (np.ndarray): Indices of wild-type mice in the group.
        RFIDs (np.ndarray): Array of Mouse RFID strings corresponding to the group.

    Notes:
        - The function reads metadata and group data to determine group membership.
        - If rm_weak_histo is True, only mutants with strong or unknown histology are included as mutants.
        - Mice not found in the metadata are treated as wild-type.
    """
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    if variable == "chasing":
        datapath = "..\\data\\chasing\\single\\"+labels[graph_idx]+"_single_chasing.csv"
    else:
        datapath = "..\\data\\both_cohorts_1day\\"+labels[graph_idx]+"\\"+variable+"_resD1_1.csv" # Day 1 as template
    
    arr = np.loadtxt(datapath, delimiter=",", dtype=str)
    RFIDs = arr[0, 1:].astype(str)
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
                print("Mouse not found, treat as WT")
                is_mutant.append(False)
            
    graph_length = len(RFIDs)
    is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
    return mutants, rc, others, wt, RFIDs