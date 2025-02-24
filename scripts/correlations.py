from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from HierarchiaPy import Hierarchia
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..\\src\\')
sys.path.append(os.getcwd())
from read_graph import read_graph
from collections import Counter
from scipy import stats
import statsmodels.formula.api as smf
import pandas as pd


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

def tube_rank_vs_chasing_order(out = True):
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    tube_ranks = df1.loc[:, "rank_by_tube"].to_numpy()
    groups = df1.loc[:, "group"].to_numpy()
    RFIDs = df1.loc[:, "Mouse_RFID"].to_numpy()
    ranks, orders = [], []
    for group_idx in range(1, np.max(groups)+1): #iterate over groups
        mouse_indices = np.where(groups == group_idx) # find out indices of mice from current group
        current_ranks = tube_ranks[mouse_indices]
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        outchasings = np.sum(chasing_matrix, axis = 1)
        inchasings = np.sum(chasing_matrix, axis = 0)
        if out:
            current_orders = np.argsort(-outchasings)
        else:
            current_orders = np.argsort(-inchasings)
        for idx, mouse_name in enumerate(RFIDs[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_name):
                continue
            else:
                where_mouse = np.where(names_in_chasing_matrix == mouse_name)
                ranks.append(current_ranks[idx])
                orders.append(list(current_orders).index([where_mouse][0]))

    points = list(zip(ranks, orders))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(ranks, orders, alpha = 0.5, c = 'k', s = sizes, label = "Pearson coeff = "+str(np.round(np.corrcoef(ranks, orders)[0,1], 2)))
    plt.xlabel("Tube rank", fontsize = 15)
    plt.ylabel("Chasing order", fontsize = 15)
    plt.legend()
    plt.tight_layout()
    plt.show()

def ChasingRank_vs_approachRank():
    """This function computes the correlation between the chasing rank (david's score based)
    and the approach rank (also david's score based). This can give information about the
    meaning of the chasing behavior. This is based on outgoing interactions only.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    chasing_ranks = df1.loc[:, "rank_by_chasing"].to_numpy()
    groups1 = df1.loc[:, "group"].to_numpy()
    RFIDs = df1.loc[:, "Mouse_RFID"].to_numpy()
    ranks, orders = [], []
    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        current_ranks = chasing_ranks[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = list(hier_mat.davids_score().values())
        approach_rank = np.argsort(-np.array(approach_david_score))
        for idx, mouse_name in enumerate(RFIDs[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_approach_matrix == mouse_name):
                continue
            else:
                where_mouse = np.where(np.array(list(hier_mat.davids_score().keys())) == mouse_name)
                ranks.append(current_ranks[idx])
                orders.append(approach_rank[where_mouse][0])

    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    df2 = pd.read_excel(path_cohort2)
    chasing_ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()
    for group_idx in range(np.max(groups1)+1, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        current_ranks = chasing_ranks2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = list(hier_mat.davids_score().values())
        approach_rank = np.argsort(-np.array(approach_david_score))
        for idx, mouse_name in enumerate(RFIDs2[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_approach_matrix == mouse_name):
                continue
            else:
                where_mouse = np.where(np.array(list(hier_mat.davids_score().keys())) == mouse_name)
                ranks.append(current_ranks[idx])
                orders.append(approach_rank[where_mouse][0])

    points = list(zip(ranks, orders))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(ranks, orders, alpha = 0.4, c = 'k', s = sizes, label = "Pearson coeff = "+str(np.round(np.corrcoef(ranks, orders)[0,1], 2)))
    plt.xlabel("Chasing rank", fontsize = 15)
    plt.ylabel("Approach rank", fontsize = 15)
    plt.legend()
    plt.tight_layout()
    plt.show()

def TubeRank_vs_approachRank():
    """This function computes the correlation between the Tube rank (david's score based)
    and the approach rank (also david's score based). This can give information about the
    meaning of the chasing behavior. This is based on outgoing interactions only.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    chasing_ranks = df1.loc[:, "rank_by_tube"].to_numpy()
    groups1 = df1.loc[:, "group"].to_numpy()
    RFIDs = df1.loc[:, "Mouse_RFID"].to_numpy()
    ranks, orders = [], []
    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        current_ranks = chasing_ranks[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = list(hier_mat.davids_score().values())
        approach_rank = np.argsort(-np.array(approach_david_score))
        for idx, mouse_name in enumerate(RFIDs[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_approach_matrix == mouse_name):
                continue
            else:
                where_mouse = np.where(np.array(list(hier_mat.davids_score().keys())) == mouse_name)
                ranks.append(current_ranks[idx])
                orders.append(approach_rank[where_mouse][0])

    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    df2 = pd.read_excel(path_cohort2)
    chasing_ranks2 = df2.loc[:, "rank_by_tube"].to_numpy()
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()
    for group_idx in range(np.max(groups1)+1, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        current_ranks = chasing_ranks2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = list(hier_mat.davids_score().values())
        approach_rank = np.argsort(-np.array(approach_david_score))
        for idx, mouse_name in enumerate(RFIDs2[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_approach_matrix == mouse_name):
                continue
            else:
                where_mouse = np.where(np.array(list(hier_mat.davids_score().keys())) == mouse_name)
                ranks.append(current_ranks[idx])
                orders.append(approach_rank[where_mouse][0])

    points = list(zip(ranks, orders))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(ranks, orders, alpha = 0.4, c = 'k', s = sizes, label = "Pearson coeff = "+str(np.round(np.corrcoef(ranks, orders)[0,1], 2)))
    plt.xlabel("Tube rank", fontsize = 15)
    plt.ylabel("Approach rank", fontsize = 15)
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

    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"


    sorting_ranks, david_ranks = [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    ranks1 = df1.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    # store whether the one with the higher rank chases more the one with the lower rank in diadic interaction

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        outchasings = np.sum(chasing_matrix, axis = 1)
        current_orders = np.argsort(-outchasings)
        current_ranks = ranks1[mouse_indices]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs1[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
                david_ranks.append(current_ranks[idx_a])
                sorting_ranks.append(list(current_orders).index([where_a][0]))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        outchasings = np.sum(chasing_matrix, axis = 1)
        current_orders = np.argsort(-outchasings)
        current_ranks = ranks2[mouse_indices]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs2[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
                david_ranks.append(current_ranks[idx_a])
                sorting_ranks.append(list(current_orders).index([where_a][0]))
            
    # sorting_ranks, david_ranks = np.array(sorting_ranks), np.array(david_ranks)
    # sorting_ranks, david_ranks = sorting_ranks[filter_csv][sorting_indices_csv], david_ranks[filter_xlsx][sorting_indices_xlsx]
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
        plt.xlabel(r"Chasing order (by outgoing chasings)", fontsize = 15)
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
    
def statistic(x, y, axis):
    xm = np.mean(x, axis = axis)
    ym = np.mean(y, axis = axis)
    return np.linalg.norm(np.array([x[0]-xm[0], x[1]-xm[1]]), axis=axis) - np.linalg.norm(np.array([y[0]-ym[0], y[1]-ym[1]]), axis=axis)

def time_in_arena_correlation(sep = False):
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
    
    
    # groups = ["rc" if r else "non-rc" for r in rc[~nan]]
    # data = {'X': time_in_arena[~nan], 'Y': social_time[~nan], 'Group': groups}
    # df = pd.DataFrame(data)
    # model = smf.ols('Y ~ X * C(Group)', data=df).fit()
    # print(model.summary())
    
    
    # groups = ["rc" for r in range(np.sum(rc[~nan]))]
    # groups.extend(["wt" for r in range(np.sum(~nan*~mutants))])
    # X = np.concatenate((rc_mems[0], wt[0]))
    # Y = np.concatenate((rc_mems[1], wt[1]))
    # data = {'X': X, 'Y': Y, 'Group': groups}
    # df = pd.DataFrame(data)
    # model = smf.ols('Y ~ X * C(Group)', data=df).fit()
    # print(model.summary())
    
    groups = ["rc" for r in range(np.sum(rc[~nan]))]
    groups.extend(["mutant" for r in range(np.sum(~nan*mutants))])
    X = np.concatenate((rc_mems[0], muts[0]))
    Y = np.concatenate((rc_mems[1], muts[1]))
    data = {'X': X, 'Y': Y, 'Group': groups}
    df = pd.DataFrame(data)
    model = smf.ols('Y ~ X * C(Group)', data=df).fit()
    print(model.summary())
    
    # groups = ["muts" if r else "wt" for r in mutants[~nan]]
    # data = {'X': time_in_arena[~nan], 'Y': social_time[~nan], 'Group': groups}
    # df = pd.DataFrame(data)
    # model = smf.ols('Y ~ X * C(Group)', data=df).fit()
    # print(model.summary())
    
    
    
    slope_A, intercept_A, _, _, SE_A = stats.linregress(rc_mems[0], rc_mems[1])
    slope_B, intercept_B, _, _, SE_B = stats.linregress(non_rc[0], non_rc[1])
    slope_C, intercept_C, _, _, SE_C = stats.linregress(wt[0], wt[1])
    slope_D, intercept_D, _, _, SE_D = stats.linregress(muts[0], muts[1])

    diff_slope = slope_A - slope_B
    SE_diff = np.sqrt(SE_A**2 + SE_B**2)
    z_stat = diff_slope / SE_diff
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    print(f"P-value rc vs non-rc: {p_value}")
    
    diff_slope = slope_A - slope_C
    SE_diff = np.sqrt(SE_A**2 + SE_C**2)
    z_stat = diff_slope / SE_diff
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    print(f"P-value rc vs wt: {p_value}")
    
    diff_slope = slope_A - slope_D
    SE_diff = np.sqrt(SE_A**2 + SE_D**2)
    z_stat = diff_slope / SE_diff
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    print(f"P-value rc vs mutants: {p_value}")
    
    diff_slope = slope_D - slope_C
    SE_diff = np.sqrt(SE_D**2 + SE_C**2)
    z_stat = diff_slope / SE_diff
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    print(f"P-value mutants vs wildtype: {p_value}")
    

    if sep:
        _, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        ax[0, 0].scatter(time_in_arena[rc], social_time[rc], c = 'k', alpha = 0.5, label = "RC members")
        ax[0, 0].annotate('RC members', xy=(0.93, 0.9), xycoords='axes fraction',
            size=11, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        ax[0, 1].scatter(time_in_arena[~rc], social_time[~rc], c = 'k', alpha = 0.5, label = "Non RC members")
        ax[0, 1].annotate('Non RC members', xy=(0.93, 0.9), xycoords='axes fraction',
            size=11, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        ax[1, 0].scatter(time_in_arena[~mutants], social_time[~mutants], c = 'k', alpha = 0.5, label = "WT")
        ax[1, 0].annotate('WT', xy=(0.93, 0.9), xycoords='axes fraction',
            size=11, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        ax[1, 1].scatter(time_in_arena[mutants], social_time[mutants], c = 'k', alpha = 0.5, label = "Mutants")
        ax[1, 1].annotate('Mutants', xy=(0.93, 0.9), xycoords='axes fraction',
            size=11, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
        ax[1, 0].set_xlabel("Time in arena", fontsize = 20)
        ax[1, 1].set_xlabel("Time in arena", fontsize = 20)
        ax[0, 0].set_ylabel("Social time", fontsize = 20)
        ax[1, 0].set_ylabel("Social time", fontsize = 20)

    else:
        plt.scatter(time_in_arena[~mutants], social_time[~mutants], s = 70, c = 'blue', alpha = 0.5, label = "WT")
        plt.scatter(time_in_arena[mutants], social_time[mutants], s = 70, c = 'red', alpha = 0.5, label = "mutants")
        plt.scatter(time_in_arena[~rc], social_time[~rc], c = 'gray', alpha = 1, label = "Non RC members")
        plt.scatter(time_in_arena[rc], social_time[rc], c = 'green', alpha = 1, label = "RC members")
        plt.xlabel("Time in arena", fontsize = 18)
        plt.ylabel("Social time", fontsize = 18)
        plt.legend()
    plt.show()
    
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

def chasingOrder_vs_approachOrder():
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"

    approachOrders, chasingOrders = [], []
    for group_idx in range(1, 18): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        approach_order = np.argsort(-np.sum(approach_matrix, 1))
        chasing_order = np.argsort(-np.sum(chasing_matrix, 1))
        for idx, mouse_name in enumerate(names_in_approach_matrix):
            if names_in_chasing_matrix[idx] == mouse_name:
                approachOrders.append(list(approach_order).index(idx))
                chasingOrders.append(list(chasing_order).index(idx))

    approachOrders, chasingOrders = np.array(approachOrders), np.array(chasingOrders)

    plt.figure()
    plt.xlabel("Approach order (outgoing)", fontsize = 17)
    plt.ylabel("Chasing order (outgoing)", fontsize = 17)
    correlation_matrix = np.corrcoef(approachOrders, chasingOrders)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(approachOrders, chasingOrders))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(approachOrders, chasingOrders, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 
    plt.yticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.show()
    
# chasingRank_david_vs_sortingRC()
# chasingRank_david_vs_sortingALL()
# chasingRank_david_vs_approachesALL()
# chasingRank_david_vs_approachesALL()
# interaction_vs_social_rank_correlation()
# tube_rank_vs_chasing_rank()
# tube_rank_vs_chasing_order()
# chasingRank_david_vs_sortingALL()
# ChasingRank_vs_approachRank()
# TubeRank_vs_approachRank()
# chasingOrder_vs_approachOrder()
time_in_arena_correlation(False)