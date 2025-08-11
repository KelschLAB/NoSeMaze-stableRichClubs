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

def mean(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def median(x, y, axis):
    return np.median(x, axis=axis) - np.median(y, axis=axis)

def format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Others"]):
    """ Sets the x-axis the RC/mutants/Others, changes the color of the bars in the boxplot."""
    ax.set_xticklabels(xticklabels, fontsize = 20)
    if len(xticklabels) == 2:
        colors = ["darkgray", "firebrick"]
    elif len(xticklabels) == 3:
        colors = ["cornflowerblue", "darkgray", "firebrick"]
    else:
        raise("Tick labels should have length 2 or 3.")
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    
def add_significance(data, ax, bp, stat = mean):
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
        result = stats.permutation_test((data1, data2), stat)
        U = result.statistic  # This gives the test statistic
        p = result.pvalue      # This gives the p-value
        print(p)
        # if p < 0.05:
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
            text_height = bar_height - (y_range * 0.03)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize = 20)
        elif p < 0.01:
            sig_symbol = '**'
            text_height = bar_height - (y_range * 0.03)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize = 20)
        elif p < 0.05:
            sig_symbol = '*'
            text_height = bar_height - (y_range * 0.03)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k', fontsize = 20)
        else:
            sig_symbol = f"p = {np.round(p, 3)}"
            text_height = bar_height + (y_range * 0.01)
            plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
            
def statistic(x, y):
    x = np.concatenate(x)
    y = np.concatenate(y)
    return np.mean(x) - np.mean(y)            

def add_group_significance(data, rfids, ax, bp):
    """Computes and adds p-value significance to input ax and boxplot"""
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    rfids1 = rfids[0]
    rfids2 = rfids[1]
    data1, data2 = data[0], data[1]

    # Group data by RFID
    grouped_data1 = [[data1[i] for i, r in enumerate(rfids1) if r == rfid] for rfid in np.unique(rfids1)]
    grouped_data1 = [list(d) for d in grouped_data1]
    grouped_data2 = [[data2[i] for i, r in enumerate(rfids2) if r == rfid] for rfid in np.unique(rfids2)]
    grouped_data2 = [list(d) for d in grouped_data2]

    combined_data = grouped_data1 + grouped_data2
    labels = [0] * len(grouped_data1) + [1] * len(grouped_data2)

    ## custom permutation test
    observed_stat = statistic(grouped_data1, grouped_data2)

    # Generate permutations
    permuted_stats = []
    t = 0
    for _ in tqdm(range(10000)):
        np.random.shuffle(labels)
        permuted_group1 = [combined_data[i] for i in range(len(labels)) if labels[i] == 0]
        permuted_group2 = [combined_data[i] for i in range(len(labels)) if labels[i] == 1]
        permuted_stats.append(statistic(permuted_group1, permuted_group2))

    # Compute p-value
    permuted_stats = np.array(permuted_stats)
    p = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    print(p)

    x1 = 1
    x2 = 2
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    bar_height = (y_range * 0.05 * 1) + top #- 20
    bar_tips = bar_height - (y_range * 0.02)
    ax.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )

    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = f"p = {np.round(p, 3)}"
    text_height = bar_height + (y_range * 0.01)
    ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')


labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def get_category_indices(graph_idx, variable, window, rm_weak_histo = True):
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
        datapath = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_1.csv"
        if window not in [1, 3, 7]:
            print("Incorrect time window")
            return 
    
    arr = np.loadtxt(datapath, delimiter=",", dtype=str)
    RFIDs = arr[0, 1:].astype(str)
    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    if not rm_weak_histo:
        mutant_map = curr_metadata_df.set_index('Mouse_RFID')['mutant'].to_dict()
        is_mutant = [mutant_map.get(rfid, False) for rfid in RFIDs] # if RFID is missing, animal is assumed to be neurotypical
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
    # is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
    is_RC = [
        True if (curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values.size > 0 and
             curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values[0]) else False
        for rfid in RFIDs
    ]
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
    return mutants, rc, others, wt, RFIDs



