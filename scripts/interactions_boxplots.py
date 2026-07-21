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
# sns.catplot(data=df, x = "mutant", y = "rank_by_tube", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "rank_by_tube", size=5, hue = "mutant")
# plt.title("Tube rank")
# plt.tight_layout()

# sns.catplot(data=df, x = "mutant",y = "rank_by_chasing", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "rank_by_chasing", size=5, hue = "mutant")
# plt.title("Chasing rank")
# plt.tight_layout()

# sns.catplot(data=df, x = "mutant",y = "time_in_arena_average", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "time_in_arena_average", size=5, hue = "mutant")
# oxtype = df.loc[:, "mutant"].to_numpy()
# times = df.loc[:, "time_in_arena_average"].to_numpy()
# dist_wt = times[~oxtype]
# dist_mu = times[oxtype]
# t, p = ttest_ind(dist_wt[~np.isnan(dist_wt)], dist_mu[~np.isnan(dist_mu)])
# plt.title(f"time_in_arena_average\n p-value = {p}")
# plt.tight_layout()

# sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed", size=5, hue = "mutant")
# plt.title("cs_plus_detection_speed")
# plt.tight_layout()

# sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", size=5, hue = "mutant")
# plt.title("cs_plus_detection_speed_crossreversal_shaping")
# plt.tight_layout()


# sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", size=5, hue = "mutant")
# plt.title("cs_plus_detection_speed_crossreversal_shaping")
# plt.tight_layout()

# sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_intraphase_shaping", color=".9", kind="box", width = 0.33)
# sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_intraphase_shaping", size=5, hue = "mutant")
# plt.title("cs_plus_detection_speed_intraphase_shaping")
# plt.tight_layout()

def boxplot_chasing(out = True):
    """
    Compares the number of chasings made by RC members, mutants and others.
    """
    # all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [2, 4], [5,6], [0, 1], [3,4,6], [3, 5, 7], [6, 7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    # all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    chasings_rc, chasings_mutants, chasings_others = [], [], [] 
    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, _, _ = get_category_indices(idx, "chasing")
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        if out:
            for rc in this_rc:
                chasings_rc.append(np.sum(data[rc, :]))
            for mutant in this_mutants:
                chasings_mutants.append(np.sum(data[mutant, :]))
            others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
            for other in others:
                chasings_others.append(np.sum(data[other, :]))
        else:
            for rc in this_rc:
                chasings_rc.append(np.sum(data[:, rc]))
            for mutant in this_mutants:
                chasings_mutants.append(np.sum(data[:, mutant]))
            others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
            for other in others:
                chasings_others.append(np.sum(data[:, other]))
            
    data = [chasings_rc, chasings_mutants, chasings_others]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    if out:
        ax.set_ylabel("Outgoing chasings", fontsize = 20)
    else:
        ax.set_ylabel("Ingoing chasings", fontsize = 20)

    format_plot(ax, bp) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
    # bottom, top = ax.get_ylim()
    plt.show()
        
def boxplot_interactions(show_RC = False):
    """
    Compares the number of interations made by RC and non-RC, color coding for genotype
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_wtrc, approaches_wt, approaches_mutants, approaches_mutantsrc = [], [], [], []
    rfids_wtrc, rfids_wt, rfids_mutants, rfids_mutantsrc = [], [], [], []
    group_wtrc, group_wt, group_mutants, group_mutantsrc = [], [], [], []
    
    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, this_wt, RFIDs = get_category_indices(idx, "approaches", 7)

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]

        for mutant in this_mutants:
            if ~np.isin(mutant, this_rc):
                approaches_mutants.append(np.sum(data[mutant, :]))
                rfids_mutants.append(RFIDs[mutant])
                group_mutants.append(g)
            else:
                approaches_mutantsrc.append(np.sum(data[mutant, :]))
                rfids_mutantsrc.append(RFIDs[mutant])
                group_mutantsrc.append(g)

        for wt in this_wt:
            if ~np.isin(wt, this_rc):
                approaches_wt.append(np.sum(data[wt, :]))
                rfids_wt.append(RFIDs[wt])
                group_wt.append(g)

            else:
                approaches_wtrc.append(np.sum(data[wt, :]))
                rfids_wtrc.append(RFIDs[wt])
                group_wtrc.append(g)

    if show_RC:
        data = [approaches_wtrc + approaches_mutantsrc, approaches_wt + approaches_mutants]
        RFIDs = [rfids_wtrc + rfids_mutantsrc, rfids_wt + rfids_mutants]
    
        fig, ax = plt.subplots(1, 1, figsize = (5, 6))
    
        bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
        plt.setp(bp['medians'], color='k')
        ax.set_xticks([1,2], ["sRC", "non-sRC"])
        add_group_significance(data, RFIDs, ax, bp)
    
        alpha, size  = 1, 40
        colors = ["gray", "red", "lightgray", "red"]
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wtrc))], 
                    approaches_wtrc, alpha = alpha, c = colors[0], s = size); 
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_mutantsrc))], 
                    approaches_mutantsrc, alpha = alpha, c = colors[1], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_wt))], 
                    approaches_wt, alpha = alpha, c = colors[2], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutants))], 
                    approaches_mutants, alpha = alpha, c = colors[3], s = size);
        ax.set_ylabel("Total interactions", fontsize = 20)
    
        plt.tight_layout()
        plt.show()
    else:
        data = [approaches_wtrc + approaches_wt, approaches_mutantsrc + approaches_mutants]
        RFIDs = [rfids_wtrc + rfids_wt, rfids_mutantsrc  + rfids_mutants]
    
        fig, ax = plt.subplots(1, 1, figsize = (5, 6))
    
        bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
        plt.setp(bp['medians'], color='k')
        ax.set_xticks([1,2], ["Controls", "OXTRAON"])
        add_group_significance(data, RFIDs, ax, bp)
    
        alpha, size  = 1, 40
        colors = ["gray", "red"]
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wtrc))], 
                    approaches_wtrc, alpha = alpha, c = colors[0], s = size); 
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(approaches_wt))], 
                    approaches_wt, alpha = alpha, c = colors[0], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutantsrc))], 
                    approaches_mutantsrc, alpha = alpha, c = colors[1], s = size); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(approaches_mutants))], 
                    approaches_mutants, alpha = alpha, c = colors[1], s = size);
        ax.set_ylabel("Total interactions", fontsize = 20)
    
        plt.tight_layout()
        plt.show()
    
def boxplot_interaction_durations(show_RC = False, all_wt = True):
    """
    Compares the number of approaches made by RC members, mutants and others.
    sep (bool): whether to show the rich club in a separate box
    all_wt (boot): if true, rc will be included in normal wt, otherwise, only show non member WT
    histo: if true, plots the results as a histogram
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    if not show_RC:
    
        approaches_rc, approaches_mutants, approaches_others = [], [], [] 
        for idx, g in enumerate(labels):
            this_mutants, this_rc, _, _, _ = get_category_indices(idx, "approaches", 7)
    
            data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\t_mean_resD7_1.csv"], percentage_threshold = 0)[0]/2 + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\t_mean_resD7_2.csv"], percentage_threshold = 0)[0]/2
            for rc in this_rc:
                approaches_rc.append(np.nanmean(data[:, rc])/2 + np.nanmean(data[rc, :])/2)
    
            for mutant in this_mutants:
                approaches_mutants.append(np.nanmean(data[mutant, :])/2 + np.nanmean(data[:, mutant])/2)
    
            others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
            for other in others:
                try:
                    approaches_others.append(np.nanmean(data[other, :])/2 + np.nanmean(data[:, other])/2)
                except:
                    pass
    
        if not all_wt:
            data = [approaches_others, approaches_mutants]
        else:
            data = [approaches_others + approaches_rc, approaches_mutants]
    
        fig, ax = plt.subplots(1, 1, figsize = (5, 6))
       
        bp = ax.boxplot(data, widths=0.4, patch_artist=False, showfliers = False, zorder=1)
        plt.setp(bp['medians'], color='k')
        add_significance(data, ax, bp)
    
        alpha, size  = 1, 40
        colors = ["darkgray", "firebrick"]
        print("Mean interaction time")
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                    data[0], alpha = alpha, c = colors[0], s = size, label = "mutant", zorder=2); 
        print(f"median of oxtr: {np.nanmedian(data[0])}")

        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                    data[1], alpha = alpha, c = colors[1], s = size, label = "non-member", zorder=2); 
        print(f"median of WT: {np.nanmedian(data[1])}")
      
        ax.set_ylabel("Mean interaction duration", fontsize = 20)

        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
        plt.tight_layout()
        plt.show()

    else:
        dur_wtrc, dur_mutrc = [], []
        dur_wt, dur_mut = [], []
        rfids_wtrc, rfids_wt, rfids_mutants, rfids_mutantsrc = [], [], [], []
        group_wtrc, group_wt, group_mutants, group_mutantsrc = [], [], [], []
    
        for idx, g in enumerate(labels):
            this_mutants, this_rc, _, this_wt, RFIDs = get_category_indices(idx, "approaches", 7)
    
            data = (
                read_graph([f"..\\data\\both_cohorts_7days\\{g}\\t_mean_resD7_1.csv"], percentage_threshold=0)[0]/2 +
                read_graph([f"..\\data\\both_cohorts_7days\\{g}\\t_mean_resD7_2.csv"], percentage_threshold=0)[0]/2
            )
    
            def val(i):
                return (np.nanmean(data[i, :]) + np.nanmean(data[:, i])) / 2
    
            for m in this_mutants:
                if m in this_rc:   
                    dur_mutrc.append(val(m))
                    rfids_mutantsrc.append(RFIDs[m])
                    group_mutantsrc.append(g)
                else:              
                    dur_mut.append(val(m))
                    rfids_mutants.append(RFIDs[m])
                    group_mutants.append(g)
    
            for w in this_wt:
                if w in this_rc:   
                    dur_wtrc.append(val(w))
                    rfids_wtrc.append(RFIDs[w])
                    group_wtrc.append(g)
                else:              
                    dur_wt.append(val(w))
                    rfids_wt.append(RFIDs[w])
                    group_wt.append(g)
    
        data_plot = [
            dur_wtrc + dur_mutrc,     # sRC
            dur_wt + dur_mut          # non-sRC
        ]
        RFIDs = [rfids_wtrc + rfids_mutantsrc, rfids_wt + rfids_mutants]
        
        print(f"median of wt: {np.nanmedian(data[0])}")
        print(f"median of oxtr: {np.nanmedian(data[1])}")
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        bp = ax.boxplot(data_plot, widths=0.4, patch_artist=False,
                        showfliers=False, zorder=1)
    
        plt.setp(bp['medians'], color='k')
        ax.set_xticks([1, 2], ["sRC", "non-sRC"])
        add_group_significance(data_plot, RFIDs, ax, bp)
    
        # point colors (WT vs Mut)
        colors = {"wtrc": "gray", "mutrc": "red", "wt": "lightgray", "mut": "red"}
    
        alpha, size = 1, 40
    
        ax.scatter([1 + np.random.normal()*0.05 for _ in dur_wtrc],
                   dur_wtrc, alpha=alpha, c=colors["wtrc"], s=size)
        ax.scatter([1 + np.random.normal()*0.05 for _ in dur_mutrc],
                   dur_mutrc, alpha=alpha, c=colors["mutrc"], s=size)
    
        ax.scatter([2 + np.random.normal()*0.05 for _ in dur_wt],
                   dur_wt, alpha=alpha, c=colors["wt"], s=size)
        ax.scatter([2 + np.random.normal()*0.05 for _ in dur_mut],
                   dur_mut, alpha=alpha, c=colors["mut"], s=size)
    
        ax.set_ylabel("Mean interaction duration (s)", fontsize=20)
    
        plt.tight_layout()
        plt.show()
    

def time_in_arena(sep = False, all_wt = False, plot_both_cohorts = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    groups = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
    
    df1 = pd.read_excel(path_to_first_cohort)
    df1 = df1[df1["group"].isin(groups)]
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    
    df2 = pd.read_excel(path_to_second_cohort)
    df2 = df2[df2["group"].isin(groups)]
    rc2 = df2.loc[:, "RC"]
    mutants2 = df2.loc[:, "mutant"]

    rc = np.concatenate([rc1, rc2])
    mutants = np.concatenate([mutants1, mutants2])
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    time_in_arena = np.concatenate((time_in_arena1, time_in_arena2))
    social_time = np.concatenate((social_time1, social_time2))
    
    # print(time_in_arena[~rc]**2)
    nan = np.isnan(social_time)
    rc_mems = np.array([time_in_arena[~nan*rc], social_time[~nan*rc]])
    non_rc = np.array([time_in_arena[~nan*~rc], social_time[~nan*~rc]])
    wt = np.array([time_in_arena[~nan*~mutants], social_time[~nan*~mutants]])
    muts = np.array([time_in_arena[~nan*mutants], social_time[~nan*mutants]])
    
    # plot params
    data = [time_in_arena[~nan*rc], 
            time_in_arena[~nan*mutants], 
            time_in_arena[~nan*~mutants*~rc]]
    
    if sep:
        pass
    else:
        if not all_wt:
            data = [data[1], data[2]]
        else:
            data = [data[1], np.concatenate((data[2], data[0]))]
            
    fig, ax = plt.subplots(1, 1)
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    add_significance(data, ax, bp)

    alpha, size  = 1, 40
    if not sep:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                    data[0], alpha = alpha, s = size, label = "mutant", zorder=2); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                    data[1], alpha = alpha, s = size, label = "non-member", zorder=2); 
    
    elif sep:
         ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                     data[1], alpha = alpha, s = size, label = "mutant"); 
         ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[2]))], 
                     data[2], alpha = alpha, s = size, label = "non-member"); 
         ax.scatter([3 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                     data[0], alpha = alpha, s = size, label = "RC"); 

    ax.set_ylabel("Time spent in arena", fontsize = 20)
    if sep:
        format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Non-members"]) # set x_axis, and colors of each bar
    else:
        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
    return data
        
def social_time(sep = False, all_wt = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    groups = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]

    df1 = pd.read_excel(path_to_first_cohort)
    df1 = df1[df1["group"].isin(groups)]
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    
    df2 = pd.read_excel(path_to_second_cohort)
    df2 = df2[df2["group"].isin(groups)]
    rc2 = df2.loc[:, "RC"]
    mutants2 = df2.loc[:, "mutant"]

    rc = np.concatenate([rc1, rc2])
    mutants = np.concatenate([mutants1, mutants2])
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    time_in_arena = np.concatenate((time_in_arena1, time_in_arena2))
    social_time = np.concatenate((social_time1, social_time2))
    
    # print(time_in_arena[~rc]**2)
    nan = np.isnan(social_time)
    rc_mems = np.array([time_in_arena[~nan*rc], social_time[~nan*rc]])
    non_rc = np.array([time_in_arena[~nan*~rc], social_time[~nan*~rc]])
    wt = np.array([time_in_arena[~nan*~mutants], social_time[~nan*~mutants]])
    muts = np.array([time_in_arena[~nan*mutants], social_time[~nan*mutants]])
    
    data = [social_time[~nan*rc], 
            social_time[~nan*mutants], 
            social_time[~nan*~mutants*~rc]]
    if sep:
        pass
    else:
        if not all_wt:
            data = [data[1], data[2]]
        else:
            data = [data[1], np.concatenate((data[2], data[0]))]   
            
    fig, ax = plt.subplots(1, 1)
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    add_significance(data, ax, bp)

    alpha, size  = 1, 40
    if not sep:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                    data[0], alpha = alpha, s = size, label = "mutant", zorder=2); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                    data[1], alpha = alpha, s = size, label = "non-member", zorder=2); 
    
    elif sep:
         ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                     data[1], alpha = alpha, s = size, label = "mutant"); 
         ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[2]))], 
                     data[2], alpha = alpha, s = size, label = "non-member"); 
         ax.scatter([3 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                     data[0], alpha = alpha, s = size, label = "RC"); 

    ax.set_ylabel("Avg. social time", fontsize = 20)
    if sep:
        format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Non-members"]) # set x_axis, and colors of each bar
    else:
        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
    plt.tight_layout()
    plt.show()
    return data

def arenaTime_x_socialTime(sep= False):
    data_arena = time_in_arena(sep = sep, all_wt = True)
    data_social = social_time(sep = sep, all_wt = True)
    plt.close("all")
    if sep:
        plt.figure()
        plt.scatter(data_arena[1], data_social[1]); plt.title("OXTR"); plt.xlabel("Arena"); plt.ylabel("social")
        plt.figure()
        plt.scatter(data_arena[0], data_social[0]); plt.title("RC (controls)"); plt.xlabel("Arena"); plt.ylabel("social")
        plt.figure()
        plt.scatter(data_arena[2], data_social[2]); plt.title("non-members (controls)"); plt.xlabel("Arena"); plt.ylabel("social")
    else:
        plt.figure()
        plt.scatter(data_arena[0], data_social[0]); plt.title("OXTR"); plt.xlabel("Arena"); plt.ylabel("social")
        plt.figure()
        plt.scatter(data_arena[1], data_social[1]); plt.title("controls"); plt.xlabel("Arena"); plt.ylabel("social")

    
def tube_rank(sep = False, all_wt = False):
    path_to_first_cohort = "..\\data\\summary_data_Day1to14.csv"

    df1 = pd.read_csv(path_to_first_cohort)
    mutants1 = df1.loc[:, "mutant"]
    ranks1 = np.array(df1.Rank_Competition.values)
    ranks_wt1, ranks_mutants1 = ranks1[~mutants1], ranks1[mutants1]
    rfids_wt, rfids_mutants = df1.loc[~df1["mutant"], "Mouse_RFID"].values, df1.loc[df1["mutant"], "Mouse_RFID"].values
    ranks_wt = ranks_wt1[~np.isnan(ranks_wt1)]
    ranks_mutants = ranks_mutants1[~np.isnan(ranks_mutants1)]

    data = [ranks_mutants, ranks_wt]
    fig, ax = plt.subplots(1, 1)
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    add_significance(data ,ax, bp)

    fig, ax = plt.subplots(1, 1)
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    add_group_significance(data, [rfids_mutants, rfids_wt] ,ax, bp)
    alpha, size = 0.5, 30
    
    plt.figure()
    plt.hist(ranks_wt, bins = np.arange(1, 12), color = "gray", density=False, rwidth = 0.9)
    plt.hist(ranks_mutants, bins = np.arange(1, 12), color = "darkred", density=False, rwidth=0.75)
    ax.set_xlabel("Tube rank", fontsize = 20)
    ax.set_ylabel("Count", fontsize = 20)
    plt.tight_layout()
    plt.show()
  

    
if __name__ == "__main__":
    # Figure 4
    # boxplot_interactions(False)
    # boxplot_interaction_durations(False, all_wt = True)

    ## Figure 5
    boxplot_interactions(True)
    # boxplot_interaction_durations(True, all_wt = True)

    
    ## Figure 6
    # tube_rank()
    
    ## Supplement
    # mut, wt = time_in_arena(False, True)
    # mut, wt = social_time(False, True)
    # arenaTime_x_socialTime(sep= False)
    # age_distribution()
    # activity_x_age()
    # activity_x_age("approaches")

    # boxplot_interaction_durations(True, all_wt = True)
    # approaches_scatter_plot(True)
    
