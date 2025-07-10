import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns

from scipy import stats
import pandas as pd
from HierarchiaPy import Hierarchia

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..\\src\\')
#sys.path.append(os.getcwd())
from read_graph import read_graph
from collections import Counter
# from weighted_rc import weighted_rich_club



def plot_reshuffled_tuberank_corr():
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    
    tube_first_others, tube_second_others = [], []
    tube_first_rc, tube_second_rc = [], []

    
    # first cohort
    df = pd.read_excel(path_to_first_cohort)
    group_counts = df.groupby('Mouse_RFID')['group'].nunique()
    reshuffled_mice = group_counts[group_counts > 1].index.tolist() # creates a list of the RFIDs that were reshuffled at leasta once
    reshuffled_rc = df["Mouse_RFID"].isin(reshuffled_mice) & df["RC"]
    reshuffled_others = df["Mouse_RFID"].isin(reshuffled_mice) & ~df["RC"]
    RFID_reshuffled_rc = df["Mouse_RFID"][reshuffled_rc]
    RFID_reshuffled_others = df["Mouse_RFID"][reshuffled_others]
    ranks = np.array(df.rank_by_tube.values)
    
    for tag in np.unique(RFID_reshuffled_rc):
        tube_first_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][0]])
        tube_second_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
        if len(np.where(df["Mouse_RFID"] == tag)[0]) >= 3:
            tube_first_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
            tube_second_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][2]])

    for tag in np.unique(RFID_reshuffled_others):
        tube_first_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][0]])
        tube_second_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
        if len(np.where(df["Mouse_RFID"] == tag)[0]) >= 3:
            tube_first_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
            tube_second_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][2]])

    #second cohort
    df = pd.read_excel(path_to_first_cohort)
    group_counts = df.groupby('Mouse_RFID')['group'].nunique()
    reshuffled_mice = group_counts[group_counts > 1].index.tolist() # creates a list of the RFIDs that were reshuffled at leasta once
    reshuffled_rc = df["Mouse_RFID"].isin(reshuffled_mice) & df["RC"]
    reshuffled_others = df["Mouse_RFID"].isin(reshuffled_mice) & ~df["RC"]
    RFID_reshuffled_rc = df["Mouse_RFID"][reshuffled_rc]
    RFID_reshuffled_others = df["Mouse_RFID"][reshuffled_others]
    ranks = np.array(df.rank_by_tube.values)
    
    for tag in np.unique(RFID_reshuffled_rc):
        tube_first_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][0]])
        tube_second_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
        if len(np.where(df["Mouse_RFID"] == tag)[0]) >= 3:
            tube_first_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
            tube_second_rc.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][2]])

    for tag in np.unique(RFID_reshuffled_others):
        tube_first_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][0]])
        tube_second_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
        if len(np.where(df["Mouse_RFID"] == tag)[0]) >= 3:
            tube_first_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][1]])
            tube_second_others.append(ranks[np.where(df["Mouse_RFID"] == tag)[0][2]])

    plt.figure(figsize = (4.5, 4.5))
    # plt.grid(alpha = 0.1)

    
    points = list(zip(tube_first_others, tube_second_others))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(tube_first_others, tube_second_others, s = sizes, c = 'gray', alpha = 0.5, label = "Non-members")
    
    points = list(zip(tube_first_rc, tube_second_rc))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(tube_first_rc, tube_second_rc, s = sizes, c = 'blue', alpha = 0.4, label = "RC")
    
    plt.xlabel("Tube rank, round 1", fontsize = 17)
    plt.ylabel("Tube rank, round 2", fontsize = 17)
    slope, intercept = np.polyfit(tube_first_rc + tube_first_others, tube_second_rc + tube_second_others, 1)
    x = np.arange(np.min(tube_first_others), np.max(tube_first_others)+1)
    correlation_matrix = np.corrcoef(tube_first_rc + tube_first_others, tube_second_rc + tube_second_others)
    pearson_corr = correlation_matrix[0, 1] 
    print(pearson_corr)
    plt.plot(x, slope * x + intercept, color='gray', linestyle='--', label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_reshuffled_outchasing_corr(fraction = False):
    path_to_first_cohort = "..\\data\\reduced_data.xlsx"
    path_to_second_cohort = "..\\data\\validation_cohort_full.xlsx"
    
    # first cohort
    df1 = pd.read_excel(path_to_first_cohort)
    rc1 = df1["Mouse_RFID"][df1["RC"]].tolist()
    df2 = pd.read_excel(path_to_second_cohort)
    rc2 = df2["Mouse_RFID"][df2["RC"]].tolist()
    RFID_rc = rc1 + rc2

    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasings_second = [], []    
    chasings_first_rc, chasings_second_rc = [], []
    
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 1)
        if fraction:
            chasings = chasings/np.sum(np.sum(data))
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                if names[c] in RFID_rc:
                    chasings_first_rc.append(chasings[c])
                else:
                    chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                if names[c] in RFID_rc:
                    chasings_second_rc.append(chasings[c])
                else:
                    chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names, filter_names_rc = [], []
    for name in already_seen_first:
        if name not in already_seen_second: # if mouse wasn't reshuffled, exclude it from plot
            if name in RFID_rc:
                filter_names_rc.append(False)
            else:
                filter_names.append(False)
        else:
            if name in RFID_rc:
                filter_names_rc.append(True)
            else:
                filter_names.append(True)

    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure(figsize = (4.5, 4.5))

    plt.scatter(chasings_first[filter_names], chasings_second, alpha = 0.7, c = 'k', label = "Never-members")
    plt.scatter(np.array(chasings_first_rc)[filter_names_rc], chasings_second_rc, alpha = 0.7, c = 'blue', label="RC")

    plt.xlabel("Total outgoing chasings, round 1", fontsize = 17)
    plt.ylabel("Total outgoing chasings, round 2", fontsize = 17)
    if fraction:
        plt.xlabel("Normalized outgoing chasings, round 1", fontsize = 17)
        plt.ylabel("Normalized outgoing chasings, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    # correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    # pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_inchasing_corr(fraction = False):
    
    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 0)
        if fraction:
            chasings = chasings/np.sum(np.sum(data))
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k')
    plt.xlabel("Total ingoing chasings, round 1", fontsize = 17)
    plt.ylabel("Total ingoing chasings, round 2", fontsize = 17)
    if fraction:
        plt.xlabel("Normalized ingoing chasings, round 1", fontsize = 17)
        plt.ylabel("Normalized ingoing chasings, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)  # 1 means linear
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_outapproach_corr(fraction = False):
    datapath = "..\\data\\averaged\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt(datapath+g+"\\approaches_resD7_1.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 1)
        if fraction:
            chasings = chasings/np.sum(np.sum(data))
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k')
    plt.xlabel("Total outgoing approaches, round 1", fontsize = 17)
    plt.ylabel("Total outgoing approaches, round 2", fontsize = 17)
    if fraction:
        plt.xlabel("Normalized outgoing approaches, round 1", fontsize = 17)
        plt.ylabel("Normalized outgoing approaches, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_chasingOrder_corr(both=False):
    
    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        outchasings = np.sum(data, axis = 1)
        inchasings = np.sum(data, axis = 0)
        if both:
            rank = np.argsort(-outchasings-inchasings)
        else:
            rank = np.argsort(-outchasings)
        
        for c in range(len(rank)):
            if names[c] not in already_seen_first:
                chasings_first.append(rank[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(rank[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.xlabel("Chasing order, round 1", fontsize = 17)
    plt.ylabel("Chasing order, round 2", fontsize = 17)
    if both:
        plt.title("Rank on outgoing + ingoing chasings")
    else:
        plt.title("Rank on outgoing chasings")
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(chasings_first[filter_names], chasings_second))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()


def plot_reshuffled_approachRank_corr(both_cohorts = False):
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    if both_cohorts: # where to stop itarating. if both cohorts are considered, this number is 18.
        stop = 18
    else:
        stop = np.max(groups1) + 1
    names, ranks = [], []
    for group_idx in range(1, stop): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = list(hier_mat.davids_score().values())
        approach_rank = np.argsort(-np.array(approach_david_score))
        for idx, mouse_name in enumerate(names_in_approach_matrix ):
            where_mouse = np.where(np.array(list(hier_mat.davids_score().keys())) == mouse_name)
            names.append(mouse_name)
            ranks.append(approach_rank[where_mouse][0])

    names = np.array(names) # to allow numpy operations

    rank_first, rank_second = [], []
    for idx, name in enumerate(names):
        if np.sum(names[idx+1:] == name) >= 1:
            next_idx = np.where(names[idx+1:] == name)[0][0] + idx + 1
            rank_first.append(ranks[idx])
            rank_second.append(ranks[next_idx])

    plt.figure()
    plt.xlabel("Approach rank, round 1", fontsize = 17)
    plt.ylabel("Approach rank, round 2", fontsize = 17)
    if both_cohorts:
        plt.title("Both cohorts")
    else:
        plt.title("First cohort")
    correlation_matrix = np.corrcoef(rank_first, rank_second)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(rank_first, rank_second))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(rank_first, rank_second, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()

def plot_reshuffled_approachOrder_corr(both_cohorts = False):
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    if both_cohorts: # where to stop itarating. if both cohorts are considered, this number is 18.
        stop = 18
    else:
        stop = np.max(groups1) + 1
    names, ranks = [], []
    for group_idx in range(1, stop): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        
        approach_rank = np.argsort(-np.sum(approach_matrix, 1))
        for idx, mouse_name in enumerate(names_in_approach_matrix ):
            names.append(mouse_name)
            ranks.append(list(approach_rank).index(idx))

    names = np.array(names) # to allow numpy operations

    rank_first, rank_second = [], []
    for idx, name in enumerate(names):
        if np.sum(names[idx+1:] == name) >= 1:
            next_idx = np.where(names[idx+1:] == name)[0][0] + idx + 1
            rank_first.append(ranks[idx])
            rank_second.append(ranks[next_idx])

    plt.figure()
    plt.xlabel("Approach order, round 1", fontsize = 17)
    plt.ylabel("Approach order, round 2", fontsize = 17)
    if both_cohorts:
        plt.title("Both cohorts")
    else:
        plt.title("First cohort")
    correlation_matrix = np.corrcoef(rank_first, rank_second)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(rank_first, rank_second))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(rank_first, rank_second, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 
    plt.yticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()

def plot_reshuffled_chasingRank_corr():
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    df1 = pd.read_excel(path_cohort1)
    # tube_ranks = df1.loc[:, "rank_by_tube"].to_numpy()
    chasing_ranks = df1.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    chasings_first, chasings_second = [], []
    for idx, name in enumerate(RFIDs1):
        if np.sum(RFIDs1[idx+1:] == name) >= 1:
            next_idx = np.where(RFIDs1[idx+1:] == name)[0][0] + idx + 1
            chasings_first.append(chasing_ranks[idx])
            chasings_second.append(chasing_ranks[next_idx])
    plt.figure()
    plt.xlabel("Chasing rank, round 1", fontsize = 17)
    plt.ylabel("Chasing rank, round 2", fontsize = 17)
    correlation_matrix = np.corrcoef(chasings_first, chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(chasings_first, chasings_second))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(chasings_first, chasings_second, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()

# plot_reshuffled_tuberank_corr()
# plot_reshuffled_inchasing_corr(True)
plot_reshuffled_outchasing_corr(True)
# plot_reshuffled_chasingRank_corr(True)
# plot_reshuffled_outapproach_corr(True)
# plot_reshuffled_chasingRank_corr()
# plot_reshuffled_approachRank_corr(True)
# plot_reshuffled_approachOrder_corr(True)