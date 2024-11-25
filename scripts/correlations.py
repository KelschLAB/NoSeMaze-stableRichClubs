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
from weighted_rc import weighted_rich_club
sys.path.append('..\\src\\')
from read_graph import read_graph
from collections import Counter
from scipy import stats


plt.rcParams.update({
"text.usetex": True,
"font.family": "Helvetica"
})

def tube_rank_vs_chasing_rank():
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    tube_ranks = df1.loc[:, "rank_by_tube"].to_numpy()
    chasing_ranks = df1.loc[:, "rank_by_chasing"].to_numpy()
    points = list(zip(tube_ranks, chasing_ranks))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(tube_ranks, chasing_ranks, alpha = 0.5, c = 'k', s = sizes, label = "Pearson coeff = "+str(np.round(np.corrcoef(tube_ranks, chasing_ranks)[0,1], 2)))
    plt.xlabel("Tube rank", fontsize = 15)
    plt.ylabel("Chasing rank", fontsize = 15)
    plt.legend()
    plt.tight_layout()
    plt.show()


rc_index_in_excel1 = np.array([7, 12, 14, 15, 25, 26, 31, 34, 35, 45, 53, 59, 67, 69, 70, 76, 78, 95, 98]) - 2 # RC
rc_index_in_excel2 = np.array([9, 11, 12, 14, 50, 51]) - 2
def chasingRank_david_vs_sortingRC(datapath = "..\\data\\chasing\\single\\"):
    """plots the chasing rank of RC members vs their corresponding rank if we rank them by ho much chasing they do w.r.t to other mice."""
    chasingrank_rc1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 11].astype(float)
    names1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[rc_index_in_excel1, :1]
    chasingrank_rc2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel2, 11].astype(float)
    names2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[rc_index_in_excel2, :1]
    david_ranks, names_xlsx = np.concatenate((chasingrank_rc1, chasingrank_rc2)), np.concatenate((names1[:, 0], names2[:, 0]))      
    
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [5, 6], [0, 1], [3, 4, 6], [5, 7], [7, 8], [5, 8], [0, 2], [2, 8, 9]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10", "G11", "G12", "G15"]
    
    sorting_ranks, names_csv = [], [] # total chasing from RC (obtained by sorting them according to how much chasing they do)
    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        group_names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        rank = np.argsort(-np.sum(data, axis = 1))
        sorting_ranks.extend(rank[all_rc[idx]])
        names_csv.extend(group_names[all_rc[idx]])
        
    filter_xlsx = np.isin(names_xlsx, names_csv)
    filter_csv = np.isin(names_csv, names_xlsx)

    sorting_ranks, david_ranks = np.array(sorting_ranks), np.array(david_ranks)

    plt.figure()
    plt.scatter(sorting_ranks[filter_csv], david_ranks[filter_xlsx], c = "k")
    plt.title("Correlation between chasings rank and david score\n for rich-club members")
    plt.xlabel("Chasing rank (Sorted)", fontsize = 15)
    plt.ylabel("Chasing score (David)", fontsize = 15)
    plt.show()
    
def chasingRank_david_vs_sortingALL(datapath = "..\\data\\chasing\\single\\", both = False):
    """plots the chasing rank of all members vs their corresponding rank if we rank them by ho much chasing they do w.r.t to other mice."""
    chasingrank1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][:, 11].astype(float)
    names1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, :1]
    chasingrank2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][:71, 11].astype(float) #Indexing until 71 because the excel files contains groups that are not part of the camera tracked study (beyond G17)
    names2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:71, :1]
    david_ranks, names_xlsx = np.concatenate((chasingrank1, chasingrank2)), np.concatenate((names1[:, 0], names2[:, 0]))      
    
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
    
    sorting_ranks, names_csv = [], [] # total chasing from RC (obtained by sorting them according to how much chasing they do)
    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        group_names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        if both:
            rank = np.argsort(-np.sum(data, axis = 1) - np.sum(data, axis = 0))
        else:
            rank = np.argsort(-np.sum(data, axis = 1))
        sorting_ranks.extend(rank)
        names_csv.extend(group_names)
        
    filter_xlsx = np.isin(names_xlsx, names_csv)
    filter_csv = np.isin(names_csv, names_xlsx)
    names_csv, names_xlsx = np.array(names_csv)[filter_csv], np.array(names_xlsx)[filter_xlsx]
    sorting_indices_xlsx, sorting_indices_csv, counter = [], [], 0
    for i, name_csv in enumerate(names_csv):
        for j, name_xlsx in enumerate(names_xlsx[counter:]):
            if name_csv == name_xlsx:
                sorting_indices_xlsx.append(j+counter)
                sorting_indices_csv.append(i)
                break
        counter += 1
    
    sorting_ranks, david_ranks = np.array(sorting_ranks), np.array(david_ranks)
    sorting_ranks, david_ranks = sorting_ranks[filter_csv][sorting_indices_csv], david_ranks[filter_xlsx][sorting_indices_xlsx]
    points = list(zip(sorting_ranks, david_ranks))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    # correlation_matrix = np.corrcoef(sorting_ranks, david_ranks)
    # pearson_coefficient = correlation_matrix[0, 1]
    p = stats.pearsonr(sorting_ranks, david_ranks)
    print(p)
    rng = np.random.default_rng()
    print(p.confidence_interval(95))

    plt.figure()
    plt.scatter(sorting_ranks, david_ranks, c = "k", s = sizes, alpha = 0.4, edgecolors='none', label = "Pearson = "+str(np.round(p.statistic, 2)))
    if both:
        plt.xlabel(r"Chasing order (rank by total chasings)", fontsize = 15)
    else:
        plt.xlabel(r"Chasing order (rank by total outgoing chasings)", fontsize = 15)
    plt.ylabel(r"Chasing score (David)", fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()
    
def chasingRank_david_vs_approachesALL(datapath = "..\\data\\averaged\\", both = False):
    """plots the chasing rank of all members vs their corresponding rank if we rank them by ho much approach they do w.r.t to other mice."""
    chasingrank1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][:, 11].astype(float)
    names1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, :1]
    chasingrank2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][:71, 11].astype(float) #Indexing until 71 because the excel files contains groups that are not part of the camera tracked study (beyond G17)
    names2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:71, :1]
    david_ranks, names_xlsx = np.concatenate((chasingrank1, chasingrank2)), np.concatenate((names1[:, 0], names2[:, 0]))      
    
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
    
    sorting_ranks, names_csv = [], [] # total chasing from RC (obtained by sorting them according to how much chasing they do)
    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"\\approaches_resD7_1.csv"])[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"])[0]
        group_names = np.loadtxt(datapath+g+"\\approaches_resD7_1.csv", delimiter=",", dtype=str)[0, :][1:]
        if both:
            rank = np.argsort(-np.sum(data, axis = 1) - np.sum(data, axis = 0))
        else:
            rank = np.argsort(-np.sum(data, axis = 1))
        sorting_ranks.extend(rank)
        names_csv.extend(group_names)
        
    filter_xlsx = np.isin(names_xlsx, names_csv)
    filter_csv = np.isin(names_csv, names_xlsx)
    names_csv, names_xlsx = np.array(names_csv)[filter_csv], np.array(names_xlsx)[filter_xlsx]
    sorting_indices_xlsx, sorting_indices_csv, counter = [], [], 0
    for i, name_csv in enumerate(names_csv):
        for j, name_xlsx in enumerate(names_xlsx[counter:]):
            if name_csv == name_xlsx:
                sorting_indices_xlsx.append(j+counter)
                sorting_indices_csv.append(i)
                break
        counter += 1
    
    sorting_ranks, david_ranks = np.array(sorting_ranks), np.array(david_ranks)
    sorting_ranks, david_ranks = sorting_ranks[filter_csv][sorting_indices_csv], david_ranks[filter_xlsx][sorting_indices_xlsx]
    points = list(zip(sorting_ranks, david_ranks))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    p = stats.pearsonr(sorting_ranks, david_ranks)

    plt.figure()
    plt.scatter(sorting_ranks, david_ranks, c = "k", s = sizes, alpha = 0.4, edgecolors='none', label = "Pearson = "+str(np.round(p.statistic, 2)))
    if both:
        plt.xlabel(r"Approach order (rank by total approaches)", fontsize = 15)
    else:
        plt.xlabel(r"Appraoch order (rank by total outgoing approaches)", fontsize = 15)
    plt.ylabel(r"Chasing score (David)", fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()
    

def time_in_arena_correlation():
    path_to_first_cohort = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    path_to_second_cohort = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\meta-data_full.xlsx"
    df1 = pd.read_excel(path_to_first_cohort)
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    df2 = pd.read_excel(path_to_second_cohort)
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    time_in_arena = np.concatenate((time_in_arena1, time_in_arena2))
    social_time = np.concatenate((social_time1, social_time2))
    plt.scatter(time_in_arena, social_time, c = 'k')
    plt.xlabel("Time in arena", fontsize = 20)
    plt.ylabel("Social time", fontsize = 20)
    plt.title("Both cohorts", fontsize = 20)
    
def interaction_vs_social_rank_correlation():
    path_to_first_cohort = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\reduced_data.xlsx"
    path_to_second_cohort = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\meta-data_full.xlsx"
    df1 = pd.read_excel(path_to_first_cohort)
    tube_rank1 = np.array(df1.rank_by_tube.values)
    chasing_rank1 = np.array(df1.rank_by_tube.values)
    time_in_arena1 = np.array(df1.time_in_arena_average.values)
    social_time1 = np.array(df1.ratio_social_to_total_time_average.values)*time_in_arena1
    
    df2 = pd.read_excel(path_to_second_cohort)
    tube_rank2 = np.array(df2.rank_by_tube.values)
    chasing_rank2 = np.array(df2.rank_by_tube.values)
    time_in_arena2 = np.array(df2.time_in_arena_average.values)
    social_time2 = np.array(df2.ratio_social_to_total_time_average.values)*time_in_arena2
    tube_rank = np.concatenate((tube_rank1, tube_rank2))
    chasing_rank = np.concatenate((chasing_rank1, chasing_rank2))
    social_time = np.concatenate((social_time1, social_time2))
    filt = ~np.isnan(social_time)
    plt.scatter(social_time[filt], tube_rank[filt], c = 'b', alpha = 0.8, label = "Tube rank", s = 55)
    plt.scatter(social_time[filt], chasing_rank[filt], c = 'r', alpha = 0.8, label = "Chasing rank", s = 55)
    # plt.yscale('log')
    plt.legend()
    plt.ylabel("Ranking", fontsize = 20)
    plt.xlabel("Social time", fontsize = 20)
    plt.title("Both cohorts", fontsize = 20)
    plt.tight_layout()
    plt.show()
    
# chasingRank_david_vs_sortingRC()
# chasingRank_david_vs_sortingALL(both = True)
# chasingRank_david_vs_approachesALL(both = True)
# chasingRank_david_vs_approachesALL()
# interaction_vs_social_rank_correlation()
tube_rank_vs_chasing_rank()