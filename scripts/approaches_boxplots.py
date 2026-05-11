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

sys.path.append('..\\src\\')
from read_graph import read_graph
from utils import get_category_indices, format_plot, add_significance, add_group_significance

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Arial"


path = "..\\data\\reduced_data.xlsx"
df = pd.read_excel(path)

    
def boxplot_approaches_review(out = True, show_RC = True):
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype, the distance threshold used here is 3
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_wtrc, approaches_wt, approaches_mutants, approaches_mutantsrc = [], [], [], []
    rfids_wtrc, rfids_wt, rfids_mutants, rfids_mutantsrc = [], [], [], []
    group_wtrc, group_wt, group_mutants, group_mutantsrc = [], [], [], []

    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, this_wt, RFIDs = get_category_indices(idx, "approaches", 7)

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]

        # helper depending on direction
        def val(i):
            return np.sum(data[:, i]) if out else np.sum(data[i, :])

        # mutants
        for mutant in this_mutants:
            if mutant in this_rc:
                approaches_mutantsrc.append(val(mutant))
                rfids_mutants.append(RFIDs[mutant])
                group_mutants.append(g)
            else:
                approaches_mutants.append(val(mutant))
                rfids_mutantsrc.append(RFIDs[mutant])
                group_mutantsrc.append(g)

        # wild-types
        for wt in this_wt:
            if wt in this_rc:
                approaches_wtrc.append(val(wt))
                rfids_wt.append(RFIDs[wt])
                group_wt.append(g)
            else:
                approaches_wt.append(val(wt))
                rfids_wtrc.append(RFIDs[wt])
                group_wtrc.append(g)
            
    if show_RC:
        data = [approaches_wtrc + approaches_mutantsrc, approaches_wt + approaches_mutants]
        RFIDs = [rfids_wtrc + rfids_mutantsrc, rfids_wt + rfids_mutants]
    else:
        data = [approaches_wtrc + approaches_wt, approaches_mutantsrc + approaches_mutants]
        RFIDs = [rfids_wtrc + rfids_wt, rfids_mutantsrc + rfids_mutants]

    fig, ax = plt.subplots(1, 1, figsize = (5, 6))

    bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
    plt.setp(bp['medians'], color='k')
    if show_RC:
        ax.set_xticks([1,2], ["sRC", "non-sRC"])
    else:
        ax.set_xticks([1,2], ["WT", "OXTR"])

    add_group_significance(data, RFIDs, ax, bp)

    alpha, size  = 1, 40
    if show_RC:
        colors = ["lightgray", "red", "lightgray", "red"]
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wtrc))], 
                    approaches_wtrc, alpha = alpha, c = colors[0], s = size); 
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_mutantsrc))], 
                    approaches_mutantsrc, alpha = alpha, c = colors[1], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_wt))], 
                    approaches_wt, alpha = alpha, c = colors[2], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutants))], 
                    approaches_mutants, alpha = alpha, c = colors[3], s = size);
    else:
        colors = ["darkgray", "firebrick"]
        print(f"Approaches out: {out}")
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                    data[0], alpha = alpha, c = colors[0], s = size); 
        print(f"median of oxtr: {np.nanmedian(data[0])}")

        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                    data[1], alpha = alpha, c = colors[1], s = size); 
        print(f"median of WT: {np.nanmedian(data[1])}")


    ax.set_ylabel("Outgoing approaches" if out else "Ingoing approaches",
                      fontsize=20)
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":    
    ## Figure 4
    boxplot_approaches_review(True, show_RC= False)
    boxplot_approaches_review(False, show_RC= False)
    
    ## Figure 5
    # boxplot_approaches_review(True, show_RC= True)
    # boxplot_approaches_review(False, show_RC= True)
    
    ## code to format source data:
    # RFIDs = rfids_wtrc + rfids_mutantsrc + rfids_wt + rfids_mutants
    # Groups = group_wtrc + group_mutantsrc + group_wt + group_mutants
    # OXTR = [False]*len(approaches_wtrc) + [True]*len(approaches_mutantsrc) +\
    # [False]*len(approaches_wt) + [True]*len(approaches_mutants)
    # sRC = [True]*len(approaches_wtrc) + [True]*len(approaches_mutantsrc) +\
    # [False]*len(approaches_wt) + [False]*len(approaches_mutants)
    # data = approaches_wtrc + approaches_mutantsrc + approaches_wt + approaches_mutants
    
    # df = pd.DataFrame({
    #     "RFID": pd.Series(RFIDs),
    #     "Group": pd.Series(Groups),
    #     "OXTR":pd.Series(OXTR),
    #     "sRC": pd.Series(sRC),
    #     "data": pd.Series(data),    
    # })
    
    # df.to_excel(
    #     r"C:\Users\corentin.nelias\Documents\GitHub\sRC_backup\data\source data\5e_interactions.xlsx",
    #     index=False
    # )
