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


sys.path.append('..\\src\\')
from read_graph import read_graph

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# path = "..\\data\\reduced_data.xlsx"
# df = pd.read_excel(path)
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
# plt.title("time_in_arena_average\n p-value = 0.28")
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


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Others"]):
    """ Sets the x-axis the RC/mutants/Others, changes the color of the bars in the boxplot."""
    ax.set_xticklabels(xticklabels, fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    
def add_significance(data, ax, bp):
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
        result = stats.permutation_test((data1, data2), statistic)
        U = result.statistic  # This gives the test statistic
        p = result.pvalue      # This gives the p-value

        if p < 0.05:
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
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (y_range * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')

def chasings(out = True):
    """
    Compares the number of chasings made by mutants vs WT
    """
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    chasings_wt, chasings_mutants= [], []
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        if out:
            for mutant in all_mutants[idx]:
                chasings_mutants.append(np.sum(data[mutant, :]))
            others = np.arange(10)[~np.isin(np.arange(10), all_mutants[idx])]
            for other in others:
                chasings_wt.append(np.sum(data[other, :]))
        else:
            for mutant in all_mutants[idx]:
                chasings_mutants.append(np.sum(data[:, mutant]))
            others = np.arange(10)[~np.isin(np.arange(10), all_mutants[idx])]
            for other in others:
                chasings_wt.append(np.sum(data[:, other]))
            
    data = [chasings_wt, chasings_mutants]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    if out:
        ax.set_ylabel("Outgoing chasings", fontsize = 20)
    else:
        ax.set_ylabel("Ingoing chasings", fontsize = 20)

    format_plot(ax, bp, xticklabels = ["WT", "Mutants"]) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
    # bottom, top = ax.get_ylim()
    plt.show()
    
def approaches():
    """
    Compares the number of approaches made by WT and mutants.
    """
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    approaches_mutants, approaches_wt = [], [] 
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        for mutant in all_mutants[idx]:
            approaches_mutants.append(np.sum(data[mutant, :]))
        others = np.arange(10)[~np.isin(np.arange(10), all_mutants[idx])]
        for other in others:
            try:
                approaches_wt.append(np.sum(data[other, :]))
            except:
                pass
            
    data = [approaches_wt, approaches_mutants]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_ylabel("Outgoing approaches", fontsize = 20)
    format_plot(ax, bp, xticklabels = ["WT", "Mutants"]) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
    # bottom, top = ax.get_ylim()
    # ax.set_title("Week 1", fontsize = 25, weight='bold')
    plt.show()
    
def approaches_scatter_plot():
    """
    Compares the number of ingoing approaches to outgoing via a scatterplot.
    """
    
    data_ingoing, data_outgoing = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        for mouse in range(data.shape[0]):
            data_ingoing.append(np.sum(data[mouse, :]))
            data_outgoing.append(np.sum(data[:, mouse]))
    
    plt.figure(figsize=(4, 3))
    plt.scatter(data_ingoing, data_outgoing, c = 'k', alpha = 0.5)
    plt.xlabel("Total outgoing approaches", fontsize = 15)
    plt.ylabel("Total ingoing approaches", fontsize = 15)
    plt.title("approaches_scatter_plot function in ./src/simple_stats.py", fontsize = 5)
    plt.tight_layout()
    plt.show()
    
def interactions():
    """
    Compares the number of interactions made by RC members, mutants and others.
    """
    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    interactions_mutants, interactions_wt = [], []
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\averaged\\"+g+"\\interactions_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\averaged\\"+g+"\\interactions_resD7_2.csv"], percentage_threshold = 0)[0]
        for mutant in all_mutants[idx]:
            interactions_mutants.append(np.sum(data[mutant, :]))
        others = np.arange(10)[~np.isin(np.arange(10), all_mutants[idx])]
        for other in others:
            try:
                interactions_wt.append(np.sum(data[other, :]))
            except:
                pass
    data = [interactions_wt, interactions_mutants]
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
    ax.set_ylabel("Total interactions", fontsize = 20)
    format_plot(ax, bp, xticklabels = ["WT", "Mutants"]) # set x_axis, and colors of each bar
    add_significance(data, ax, bp)
    # bottom, top = ax.get_ylim()
    plt.show()


## Plot time spent in arena as function of RC/mutant/none
def time_in_arena(plot_both_cohorts = False):
    # data first cohort
    rc_index_in_excel1 = np.array([7, 12, 14, 15, 25, 26, 31, 34, 35, 45, 53, 59, 67, 69, 70, 76, 78, 95, 98]) - 2 # RC
    mutants_index_in_excel1 = np.array([6, 9, 13, 21, 27, 28, 33, 41, 51, 54, 58, 65, 66, 73, 75, 92, 94]) - 2 # Mutants
    other_index_in_excel1 = np.arange(99)[~np.isin(np.arange(99), np.concatenate((rc_index_in_excel1, mutants_index_in_excel1)))] # Others
    arena_time_rc1 = pd.read_excel(r"..\data\reduced_data.xlsx", 
                                   sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 32].astype(float)
    arena_time_rc1 = arena_time_rc1[~np.isnan(arena_time_rc1)]
    arena_time_mutants1 = pd.read_excel(r"..\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][mutants_index_in_excel1, 32].astype(float)
    arena_time_mutants1 = arena_time_mutants1[~np.isnan(arena_time_mutants1)]
    arena_time_others1 = pd.read_excel(r"..\data\reduced_data.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel1, 32].astype(float)
    arena_time_others1 = arena_time_others1[~np.isnan(arena_time_others1)]
    # data second cohort 
    rc_index_in_excel2 = np.array([9, 11, 12, 14, 50, 51]) - 2
    mutants_index_in_excel2 = np.array([3, 5, 6, 7, 17, 19, 22, 28, 29, 34, 40, 43, 46]) - 2
    other_index_in_excel2 = np.arange(109)[~np.isin(np.arange(109), np.concatenate((rc_index_in_excel2, mutants_index_in_excel2)))] # Others
    arena_time_rc2 = pd.read_excel(r"..\data\meta-data_validation.xlsx", 
                                   sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 32].astype(float)
    arena_time_rc2 = arena_time_rc2[~np.isnan(arena_time_rc2)]
    arena_time_mutants2 = pd.read_excel(r"..\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][mutants_index_in_excel1, 32].astype(float)
    arena_time_mutants2 = arena_time_mutants2[~np.isnan(arena_time_mutants2)]
    arena_time_others2 = pd.read_excel(r"..\data\meta-data_validation.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel1, 32].astype(float)
    arena_time_others2 = arena_time_others2[~np.isnan(arena_time_others2)]
    # plot params
    if plot_both_cohorts:
        data = [np.concatenate((arena_time_rc1, arena_time_rc2)), 
                np.concatenate((arena_time_mutants1, arena_time_mutants2)), 
                np.concatenate((arena_time_others1, arena_time_others2))]
    else:
        data = [arena_time_rc1, arena_time_mutants1, arena_time_others1]

    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
  #  plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xticklabels = ["RC", "Mutants", "Others"]
    ax.set_xticklabels(xticklabels, fontsize = 20)
    ax.set_ylabel("Avg. time in arena", fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, 15000, fr'$n = {sample_size}$', ha='center', size='x-small')
        
def social_time():
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\meta-data_full.xlsx"
    
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
    data = [social_time[~nan*rc], 
            social_time[~nan*mutants], 
            social_time[~nan*~mutants*~rc]]

    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.6, patch_artist=True)
  #  plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xticklabels = ["RC", "Mutants", "Others"]
    ax.set_xticklabels(xticklabels, fontsize = 20)
    ax.set_ylabel("Avg. social time", fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, 3000, fr'$n = {sample_size}$', ha='center', size='x-small')
    plt.tight_layout()
    plt.show()
        

def rc_size():
    rcs = [2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2]
    plt.hist(rcs, bins = [1.5, 2.5, 3.5], rwidth = 0.8)
    plt.xticks([1, 2, 3,4 ,5, 6], labels = [1, 2, 3, 4, 5, 6])
    plt.xlabel("Rich club size")
    plt.ylabel("Count")
    plt.show()
    
# rc_size()
# time_in_arena(True)
# social_time()
# chasings()
# approaches()
# interactions()
approaches_scatter_plot()