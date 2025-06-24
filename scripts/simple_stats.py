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
from scipy.stats import ttest_ind
from scipy import stats
import seaborn as sns

sys.path.append('..\\src\\')
from read_graph import read_graph
from utils import get_category_indices, format_plot, add_significance

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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
    
def boxplot_approaches(out = True, sep = False, all_wt = False):
    """
    Compares the number of approaches made by RC members, mutants and others.
    out (bool): if true, plots outgoing approaches, else ingoing
    sep (bool): whether to show the rich club in a separate box
    all_wt (boot): if true, rc will be included in normal wt, otherwise, only show non member WT
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_rc, approaches_mutants, approaches_others = [], [], [] 
    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, _, _ = get_category_indices(idx, "approaches", 7)

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        for rc in this_rc:
            if out:
                approaches_rc.append(np.sum(data[:, rc]))
            else:
                approaches_rc.append(np.sum(data[rc, :]))
        for mutant in this_mutants:
            if out:
                approaches_mutants.append(np.sum(data[mutant, :]))
            else:
                approaches_mutants.append(np.sum(data[:, mutant]))
        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
        for other in others:
            try:
                if out:
                    approaches_others.append(np.sum(data[other, :]))
                else:
                    approaches_others.append(np.sum(data[:, other]))
            except:
                pass
    if sep:
        data = [approaches_rc, approaches_mutants, approaches_others]
    else:
        if not all_wt:
            data = [approaches_mutants, approaches_others]
        else:
            data = [approaches_mutants, approaches_others + approaches_rc]

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
         ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                     data[0], alpha = alpha, s = size, label = "RC"); 
         ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                     data[1], alpha = alpha, s = size, label = "mutant"); 
         ax.scatter([3 + np.random.normal()*0.05 for i in range(len(data[2]))], 
                     data[2], alpha = alpha, s = size, label = "non-member"); 

    if out:
        ax.set_ylabel("Outgoing approaches", fontsize = 20)
    else:
        ax.set_ylabel("Ingoing approaches", fontsize = 20)
    if sep:
        format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Non-members"]) # set x_axis, and colors of each bar
    else:
        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
    plt.show()
    
    
def boxplot_interactions(sep = False, all_wt = False):
    """
    Compares the number of approaches made by RC members, mutants and others.
    sep (bool): whether to show the rich club in a separate box
    all_wt (boot): if true, rc will be included in normal wt, otherwise, only show non member WT
    histo: if true, plots the results as a histogram
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_rc, approaches_mutants, approaches_others = [], [], [] 
    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, _, _ = get_category_indices(idx, "approaches", 7)

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]
        for rc in this_rc:
                approaches_rc.append(np.sum(data[:, rc])/2 + np.sum(data[rc, :])/2)

        for mutant in this_mutants:
                approaches_mutants.append(np.sum(data[mutant, :])/2 + np.sum(data[:, mutant])/2)

        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
        for other in others:
            try:
                approaches_others.append(np.sum(data[other, :])/2 + np.sum(data[:, other])/2)
            except:
                pass
    if sep:
        data = [approaches_rc, approaches_mutants, approaches_others]
    else:
        if not all_wt:
            data = [approaches_mutants, approaches_others]
        else:
            data = [approaches_mutants, approaches_others + approaches_rc]

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
         ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                     data[0], alpha = alpha, s = size, label = "RC"); 
         ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                     data[1], alpha = alpha, s = size, label = "mutant"); 
         ax.scatter([3 + np.random.normal()*0.05 for i in range(len(data[2]))], 
                     data[2], alpha = alpha, s = size, label = "non-member"); 

    ax.set_ylabel("Total interactions", fontsize = 20)
    if sep:
        format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Non-members"]) # set x_axis, and colors of each bar
    else:
        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
            
    plt.tight_layout()
    plt.show()
    
def boxplot_interaction_durations(sep = False, all_wt = False):
    """
    Compares the number of approaches made by RC members, mutants and others.
    sep (bool): whether to show the rich club in a separate box
    all_wt (boot): if true, rc will be included in normal wt, otherwise, only show non member WT
    histo: if true, plots the results as a histogram
    """

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_rc, approaches_mutants, approaches_others = [], [], [] 
    for idx, g in enumerate(labels):
        this_mutants, this_rc, _, _, _ = get_category_indices(idx, "approaches", 7)

        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\t_mean_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+"\\t_mean_resD7_2.csv"], percentage_threshold = 0)[0]
        for rc in this_rc:
                approaches_rc.append(np.sum(data[:, rc])/2 + np.sum(data[rc, :])/2)

        for mutant in this_mutants:
                approaches_mutants.append(np.sum(data[mutant, :])/2 + np.sum(data[:, mutant])/2)

        others = np.arange(10)[np.logical_and(~np.isin(np.arange(10), this_rc), ~np.isin(np.arange(10), this_mutants))]
        for other in others:
            try:
                approaches_others.append(np.sum(data[other, :])/2 + np.sum(data[:, other])/2)
            except:
                pass
    if sep:
        data = [approaches_rc, approaches_mutants, approaches_others]
    else:
        if not all_wt:
            data = [approaches_mutants, approaches_others]
        else:
            data = [approaches_mutants, approaches_others + approaches_rc]

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
         ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                     data[0], alpha = alpha, s = size, label = "RC"); 
         ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                     data[1], alpha = alpha, s = size, label = "mutant"); 
         ax.scatter([3 + np.random.normal()*0.05 for i in range(len(data[2]))], 
                     data[2], alpha = alpha, s = size, label = "non-member"); 


    ax.set_ylabel("Mean interaction duration", fontsize = 20)
    if sep:
        format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Non-members"]) # set x_axis, and colors of each bar
    else:
        if not all_wt:
            format_plot(ax, bp, xticklabels = ["Mutants", "Non-members"]) # set x_axis, and colors of each bar
        else:
            format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
    plt.tight_layout()
    plt.show()
    
def approaches_scatter_plot(show_rc = False, symmetry_parameter = 1.5):
    """
    Compares the number of ingoing approaches to outgoing via a scatterplot.
    """
    
    data_ingoing, data_outgoing = [], []
    data_ingoing_rc, data_outgoing_rc = [], []
    all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        for mouse in range(data.shape[0]):
            if mouse in all_rc[idx]:
                data_ingoing_rc.append(np.sum(data[mouse, :]))
                data_outgoing_rc.append(np.sum(data[:, mouse]))
            else:
                data_ingoing.append(np.sum(data[mouse, :]))
                data_outgoing.append(np.sum(data[:, mouse]))
    data_ingoing, data_outgoing = np.array(data_ingoing), np.array(data_outgoing) 
    data_ingoing_rc, data_outgoing_rc = np.array(data_ingoing_rc), np.array(data_outgoing_rc)
    
    
    plt.figure(figsize=(4, 3))
    # Define the x range to show symmetric approaches
    x_min = np.min(data_outgoing) - 100  # Extend slightly below the minimum
    x_max = np.max(data_outgoing_rc) + 1000  # Extend slightly above the maximum
    x = np.linspace(x_min, x_max, 500)  # Create a smooth range of x values
    # _, intercept = np.polyfit(np.concatenate((data_outgoing, data_outgoing_rc)), np.concatenate((data_ingoing, data_ingoing_rc)), 1)
    plt.fill_between(x, 0.5*x, 1.5*x, color = 'gray', alpha=0.2, label = "Symmetric approaches")
    
    plt.grid(alpha = 0.2)
    plt.scatter(data_outgoing, data_ingoing, c = 'k', alpha = 0.6, label = "Non-RC members")
    
    if show_rc:
        edge_colors = ['red' if np.abs(data_ingoing_rc[i] - v)/v > 0.5  else 'green' for i, v in enumerate(data_outgoing_rc)]
        plt.scatter(data_outgoing_rc, data_ingoing_rc, c = 'limegreen', alpha = 0.8, label = "RC members")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.legend(fontsize = 8)
    else:
        plt.scatter(data_outgoing_rc, data_ingoing_rc, c = 'k', alpha = 0.4)
    
    plt.xlabel("Total outgoing approaches", fontsize = 15)
    plt.ylabel("Total ingoing approaches", fontsize = 15)
    plt.title("approaches_scatter_plot function in ./src/simple_stats.py", fontsize = 5)
    plt.tight_layout()
    plt.xlim(x_min, x_max)
    plt.show()
    

def time_in_arena(sep = False, all_wt = False, plot_both_cohorts = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    
    df2 = pd.read_excel(path_to_second_cohort)
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
    plt.tight_layout()
    plt.show()
        
def social_time(sep = False, all_wt = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    df2 = pd.read_excel(path_to_second_cohort)
    
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
    
def tube_rank(sep = False, all_wt = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    
    # first cohort
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1.loc[:, "RC"]
    mutants1 = df1.loc[:, "mutant"]
    ranks1 = np.array(df1.rank_by_tube.values)
    ranks_wt1, ranks_mutants1 = ranks1[~mutants1], ranks1[mutants1]
    
    #second cohort
    df2 = pd.read_excel(path_to_second_cohort)
    rc2 = df2.loc[:, "RC"]
    mutants2 = df2.loc[:, "mutant"]
    ranks2 = np.array(df2.rank_by_tube.values)
    ranks_wt2, ranks_mutants2 = ranks2[~mutants2], ranks2[mutants2]

    ranks_wt = np.concatenate((ranks_wt1, ranks_wt2))
    ranks_mutants = np.concatenate((ranks_mutants1, ranks_mutants2))

    data = [ranks_mutants, ranks_wt]
    
    fig, ax = plt.subplots(1, 1)
    # bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    # add_significance(data, ax, bp)

    alpha, size = 0.5, 30
    # ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
    #             data[0], alpha = alpha, s = size, label = "mutant", zorder=2); 
    # ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
    #             data[1], alpha = alpha, s = size, label = "non-member", zorder=2); 
    
    plt.hist(ranks_wt, bins = 10, color = "green", alpha = alpha, density=False, rwidth = 0.95)
    plt.hist(ranks_mutants, bins = 10, color = "red", alpha = alpha, density=False, rwidth=0.95)

    
    ax.set_ylabel("Tube rank", fontsize = 20)
    # format_plot(ax, bp, xticklabels = ["Mutants", "WT"]) # set x_axis, and colors of each bar
    plt.tight_layout()
    plt.show()
  
def rc_size():
    rcs = [2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2]
    plt.hist(rcs, bins = [1.5, 2.5, 3.5], rwidth = 0.8)
    plt.xticks([1, 2, 3,4 ,5, 6], labels = [1, 2, 3, 4, 5, 6])
    plt.xlabel("Rich club size")
    plt.ylabel("Count")
    plt.show()
    
if __name__ == "__main__":
    tube_rank()
# rc_size()
    # time_in_arena(False, True)
    # social_time(False, True)
# chasings()
    # boxplot_approaches(True, True, all_wt = True)
    # boxplot_approaches(False, True, all_wt = True)
    # boxplot_interactions(True, all_wt = True)
    # boxplot_interaction_durations(True, all_wt = True)
# approaches_scatter_plot(True)