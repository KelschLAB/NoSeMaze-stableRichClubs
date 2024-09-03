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


## Plot time spent in arena as function of RC/mutant/none
def time_in_arena(plot_both_cohorts = False):
    # data first cohort
    rc_index_in_excel1 = np.array([7, 12, 14, 15, 25, 26, 31, 34, 35, 45, 53, 59, 67, 69, 70, 76, 78, 95, 98]) - 2 # RC
    mutants_index_in_excel1 = np.array([6, 9, 13, 21, 27, 28, 33, 41, 51, 54, 58, 65, 66, 73, 75, 92, 94]) - 2 # Mutants
    other_index_in_excel1 = np.arange(99)[~np.isin(np.arange(99), np.concatenate((rc_index_in_excel1, mutants_index_in_excel1)))] # Others
    arena_time_rc1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                   sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 32].astype(float)
    arena_time_rc1 = arena_time_rc1[~np.isnan(arena_time_rc1)]
    arena_time_mutants1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][mutants_index_in_excel1, 32].astype(float)
    arena_time_mutants1 = arena_time_mutants1[~np.isnan(arena_time_mutants1)]
    arena_time_others1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel1, 32].astype(float)
    arena_time_others1 = arena_time_others1[~np.isnan(arena_time_others1)]
    # data second cohort 
    rc_index_in_excel2 = np.array([9, 11, 12, 14, 50, 51]) - 2
    mutants_index_in_excel2 = np.array([3, 5, 6, 7, 17, 19, 22, 28, 29, 34, 40, 43, 46]) - 2
    other_index_in_excel2 = np.arange(109)[~np.isin(np.arange(109), np.concatenate((rc_index_in_excel2, mutants_index_in_excel2)))] # Others
    arena_time_rc2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                   sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 32].astype(float)
    arena_time_rc2 = arena_time_rc2[~np.isnan(arena_time_rc2)]
    arena_time_mutants2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][mutants_index_in_excel1, 32].astype(float)
    arena_time_mutants2 = arena_time_mutants2[~np.isnan(arena_time_mutants2)]
    arena_time_others2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
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
    plt.rc('text', usetex=True)
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

time_in_arena(True)
