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
from scipy.stats import ttest_ind, ttest_1samp
from scipy import stats
import seaborn as sns
from scipy.stats import iqr

sys.path.append('..\\src\\')
from read_graph import read_graph
from utils import get_category_indices, format_plot, add_significance, add_group_significance

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Arial"


path = "..\\data\\reduced_data.xlsx"
df = pd.read_excel(path)

metadata_path = r"C:\Users\corentin.nelias\Documents\GitHub\sRC_backup\data\metadata_controls.csv"

def boxplot_approaches(out = True):
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """

    labels = ["GJ1", "GJ2", "GD6"]
    
    approaches_rc, approaches = [], []
    rfids_rc, rfids = [], []
    group_rc, group = [], []
    
    for idx, g in enumerate(labels):
        metadata_df = pd.read_csv(metadata_path)
        this_group_sRC = metadata_df.loc[metadata_df["Group"] == g, "RC"].values
        RFIDs = metadata_df.loc[metadata_df["Group"] == g, "Mouse_RFID"].values

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] +\
            read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]

        def val(i):
            return np.sum(data[:, i]) if out else np.sum(data[i, :])

        for mouse_idx, rc in enumerate(this_group_sRC):
            if not rc:
                approaches.append(val(mouse_idx))
                rfids.append(RFIDs[mouse_idx])
                group.append(g)

            else:
                approaches_rc.append(val(mouse_idx))
                rfids_rc.append(RFIDs[mouse_idx])
                group_rc.append(g)

    data = [approaches_rc, approaches]
    RFIDs = [rfids_rc, rfids]

    fig, ax = plt.subplots(1, 1, figsize = (5, 6))

    bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1, showmeans=True)
    plt.setp(bp['medians'], color='k')
    ax.set_xticks([1,2], ["sRC", "non-sRC"])
    add_group_significance(data, RFIDs, ax, bp, "mannwhitneyu")

    alpha, size  = 1, 40
    colors = ["gray", "red", "lightgray", "red"]
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_rc))], 
                approaches_rc, alpha = alpha, c = colors[0], s = size); 
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches))], 
                approaches, alpha = alpha, c = colors[2], s = size); 
    label_y = "Outgoing approaches per round" if out else "Incoming approaches per round"
    ax.set_ylabel(label_y, fontsize = 20)

    plt.tight_layout()
    plt.show()
    print("iqr RC: "+str(iqr(data[0])))
    print("iqr non: "+str(iqr(data[1])))

def boxplot_interactions():
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """

    labels = ["GJ1", "GJ2", "GD6"]
    
    approaches_rc, approaches = [], []
    rfids_rc, rfids = [], []
    group_rc, group = [], []
    
    for idx, g in enumerate(labels):
        metadata_df = pd.read_csv(metadata_path)
        this_group_sRC = metadata_df.loc[metadata_df["Group"] == g, "RC"].values
        RFIDs = metadata_df.loc[metadata_df["Group"] == g, "Mouse_RFID"].values

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]

        for mouse_idx, rc in enumerate(this_group_sRC):
            if not rc:
                approaches.append(np.sum(data[mouse_idx, :]))
                try:
                    rfids.append(RFIDs[mouse_idx])
                except:
                    pass
                group.append(g)

            else:
                approaches_rc.append(np.sum(data[mouse_idx, :]))
                rfids_rc.append(RFIDs[mouse_idx])
                group_rc.append(g)

    data = [approaches_rc, approaches]
    RFIDs = [rfids_rc, rfids]

    fig, ax = plt.subplots(1, 1, figsize = (5, 6))

    bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1, showmeans=True)
    plt.setp(bp['medians'], color='k')
    ax.set_xticks([1,2], ["sRC", "non-sRC"])
    add_group_significance(data, RFIDs, ax, bp, "mannwhitneyu")

    alpha, size  = 1, 40
    colors = ["gray", "red", "lightgray", "red"]
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_rc))], 
                approaches_rc, alpha = alpha, c = colors[0], s = size); 
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches))], 
                approaches, alpha = alpha, c = colors[2], s = size); 
    ax.set_ylabel("Total interactions", fontsize = 20)

    plt.tight_layout()
    plt.show()
    print("iqr RC: "+str(iqr(data[0])))
    print("iqr non: "+str(iqr(data[1])))

    
def boxplot_interaction_durations():
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """

    labels = ["GJ1", "GJ2", "GD6"]
    
    approaches_rc, approaches = [], []
    rfids_rc, rfids = [], []
    group_rc, group = [], []
    
    for idx, g in enumerate(labels):
        metadata_df = pd.read_csv(metadata_path)
        this_group_sRC = metadata_df.loc[metadata_df["Group"] == g, "RC"].values
        RFIDs = metadata_df.loc[metadata_df["Group"] == g, "Mouse_RFID"].values

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\time_mean_resD7_1.csv"], percentage_threshold = 0)[0]/2 +\
                  read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\time_mean_resD7_2.csv"], percentage_threshold = 0)[0]/2
                  
        data = np.where(data != 0, data, np.nan)        
        for mouse_idx, rc in enumerate(this_group_sRC):
            if not rc:
                approaches.append(np.nanmean(data[mouse_idx, :])/2 + np.nanmean(data[:, mouse_idx])/2)
                try:
                    rfids.append(RFIDs[mouse_idx])
                except:
                    pass
                group.append(g)

            else:
                approaches_rc.append(np.nanmean(data[mouse_idx, :])/2 + np.nanmean(data[:, mouse_idx])/2)
                rfids_rc.append(RFIDs[mouse_idx])
                group_rc.append(g)
    
    approaches, rfids =np.array(approaches), np.array(rfids)
    rfids = rfids[~np.isnan(approaches)]
    approaches = approaches[~np.isnan(approaches)]
    data = [approaches_rc, approaches]
    RFIDs = [rfids_rc, rfids]

    fig, ax = plt.subplots(1, 1, figsize = (5, 6))

    bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1, showmeans=True)
    plt.setp(bp['medians'], color='k')
    ax.set_xticks([1,2], ["sRC", "non-sRC"])
    add_group_significance(data, RFIDs, ax, bp, "mannwhitneyu")

    alpha, size  = 1, 40
    colors = ["gray", "red", "lightgray", "red"]
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_rc))], 
                approaches_rc, alpha = alpha, c = colors[0], s = size); 
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches))], 
                approaches, alpha = alpha, c = colors[2], s = size); 
    ax.set_ylabel("Mean interaction duration (s)", fontsize = 20)

    plt.tight_layout()
    plt.show()
    print("iqr RC: "+str(iqr(data[0])))
    print("iqr non: "+str(iqr(data[1])))
    

    
if __name__ == "__main__":
    boxplot_interactions()
    boxplot_interaction_durations()
    boxplot_approaches(True)
    boxplot_approaches(False)
    

