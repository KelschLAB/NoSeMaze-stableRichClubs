import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..\\scripts\\')
sys.path.append(os.getcwd())
from read_graph import read_graph
import pandas as pd
from HierarchiaPy import Hierarchia
from collections import Counter
from scipy import stats
from utils import add_group_significance
plt.rcParams["font.family"] = "Arial"

def rich_club_piecharts():
    fig, ax = plt.subplots(1, 2)
    plt.style.use('seaborn-v0_8-whitegrid')
    sizes = [8/9, 1/9]
    labels = ["Rich club", "No stable club"]
    ax[0].pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    ax[0].set_title("First cohort")
    sizes =  sizes = [4/7, 3/7]
    labels = ["Rich club", "No stable club"]
    ax[1].pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    ax[1].set_title("2nd cohort")
    plt.show()

def rich_club_piechart_both():
    fig, ax = plt.subplots(1, 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    sizes = [13/16, 3/16]
    labels = ["Stable \nrich club", "No stable \nrich club"]
    ax.pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    plt.tight_layout()
    plt.show()

def approach_order_mutants(out = True, both = False):
    """ Histogram of the approach order for mutants (i.e., if we rank them by according to how much total approaches they perform.)
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    approach_order_out, approach_order_in = [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    mutants1 = df1.loc[:, "mutant"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, mutant in enumerate(mutants1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mutant:
                    mutant_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(mutant_idx))
                    approach_order_in.append(list(current_orders_in).index(mutant_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    mutants2 = df2.loc[:, "genotype"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, mutant in enumerate(mutants2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mutant == "Oxt":
                    mutant_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(mutant_idx))
                    approach_order_in.append(list(current_orders_in).index(mutant_idx))
        
    plt.figure()
    if both:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Outgoing", alpha = 0.5)
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Ingoing", alpha = 0.5)
        plt.legend()
    elif out:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')
    else:
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')

    plt.xticks(np.arange(0, 10), labels=[1,2,3,4,5,6,7,8,9,10]) 

    if both:
        plt.xlabel(r"Approach order", fontsize = 15)
    elif out:
        plt.xlabel(r"Approach order (outgoing)", fontsize = 15)
    else:
        plt.xlabel(r"Approach order (ingoing)", fontsize = 15)
    plt.ylabel(r"Count", fontsize = 15)
    plt.title("Appraoch order of mutants", fontsize = 18)
    plt.show()
    

def approach_order_sep(out = True):
    """ Histogram of approach order discriminating between RC, mutants and others.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    approach_order_rc, approach_order_mut, approach_order_others = [], [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    genotype1 = df1.loc[:, "mutant"]
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) 
        names_in_approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        
        approaches_order = np.sum(approach_matrix, axis = 1)
        if not out:
            approaches_order = np.sum(approach_matrix, axis = 0)
        current_orders = np.argsort(-approaches_order)

        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype1[idx]:
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    genotype2 = df2.loc[:, "genotype"]
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        
        approaches_order = np.sum(approach_matrix, axis = 1)
        if not out:
            approaches_order = np.sum(approach_matrix, axis = 0)
        current_orders = np.argsort(-approaches_order)
        
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype2[idx] == "Oxt":
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))
        
    plt.figure(figsize = (5, 5))
    
    plt.hist(approach_order_rc, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C0', edgecolor='black', alpha = 0.8, label = "RC")
    plt.hist(approach_order_mut, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C1', edgecolor='black', alpha = 0.8, label = "Mutants")
    plt.hist(approach_order_others, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C2', edgecolor='black', alpha = 0.8, label = "Others")

    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"], fontsize = 15) 
    plt.yticks(np.linspace(0, 0.4, 5), fontsize = 15)  # Ensure ticks are centered on 0 through 9
    
    plt.xlabel(r"Approach order (outgoing)", fontsize = 20)
    if not out:
        plt.xlabel(r"Approach order (ingoing)", fontsize = 20)

    plt.ylabel(r"Density", fontsize = 19)
    plt.legend(fontsize = 15)

    plt.show()
    
def chasing_order_sep(out = True):
    """ Histogram of chasings order discriminating between RC, mutants and others.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"
    
    approach_order_rc, approach_order_mut, approach_order_others = [], [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    genotype1 = df1.loc[:, "mutant"]
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                     delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                             delimiter=",", dtype=str)[0, :][1:]
        
        approaches_order = np.sum(approach_matrix, axis = 1)
        if not out:
            approaches_order = np.sum(approach_matrix, axis = 0)
        current_orders = np.argsort(-approaches_order)

        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype1[idx]:
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    genotype2 = df2.loc[:, "genotype"]
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        
        approaches_order = np.sum(approach_matrix, axis = 1)
        if not out:
            approaches_order = np.sum(approach_matrix, axis = 0)
        current_orders = np.argsort(-approaches_order)
        
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype2[idx] == "Oxt":
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))
        
    plt.figure(figsize = (5, 5))
    
    plt.hist(approach_order_rc, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C0', edgecolor='black', alpha = 0.8, label = "RC")
    plt.hist(approach_order_mut, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C1', edgecolor='black', alpha = 0.8, label = "Mutants")
    plt.hist(approach_order_others, bins = np.arange(-0.5, 10.5), density = True, rwidth= 0.8, align='mid', color = 'C2', edgecolor='black', alpha = 0.8, label = "Others")

    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"], fontsize = 15) 
    plt.yticks(np.linspace(0, 0.4, 5), fontsize = 15)  # Ensure ticks are centered on 0 through 9
    
    plt.xlabel(r"Chasing order (outgoing)", fontsize = 20)
    if not out:
        plt.xlabel(r"Chasing order (ingoing)", fontsize = 20)

    plt.ylabel(r"Density", fontsize = 19)
    plt.legend(fontsize = 15)
    plt.show()
    

def approachRank_mutants():
    """ Histogram of the approach david score for mutants.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    mutants1 = df1.loc[:, "mutant"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    scores = []

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        for idx, mutant in enumerate(mutants1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mutant:
                    mutant_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    # where_in_david = 
                    scores.append(list(hier_mat.davids_score().keys()).index(mouse_names[idx]))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    mutants2 = df2.loc[:, "genotype"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        for idx, mutant in enumerate(mutants2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mutant == "Oxt":
                    scores.append(list(hier_mat.davids_score().keys()).index(mouse_names[idx]))
        
    plt.figure()
    plt.hist(scores, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')
    plt.xticks(np.arange(0, 10))  # Ensure ticks are centered on 0 through 9
    plt.xlabel(r"Approach rank", fontsize = 15)
    plt.ylabel(r"Count", fontsize = 15)
    plt.title(r"Approach rank of mutants", fontsize = 18)
    plt.show()
    
def tuberank_vs_rc(ax = None, norm = False):
    """Plots the tube rank of rich clulb members for both cohorts"""
    
    ranks = []
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"

    df1 = pd.read_excel(path_cohort1)
    RCs1 = df1.loc[:, "RC"].to_numpy()
    Ranks1 = df1.loc[:, "rank_by_tube"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2 = df2.loc[:, "RC"].to_numpy()
    Ranks2 = df2.loc[:, "rank_by_tube"].to_numpy()
    
    arr = np.concatenate((Ranks1[RCs1], Ranks2[RCs2]))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.6, color = "blue", density = norm) 
    ax.set_xlabel("Tube rank", fontsize = 15)
    ax.set_ylabel("Count", fontsize = 15); 
    ticklabels = [i for i in range(1, 11)]
    ax.set_xticks([i + 0.5 for i in range(1, 11)], ticklabels, fontsize = 12)
    return arr

def tuberank_vs_exrc(ax = None):
    """Plots the tube ranks of mice that were once in the rich club, but got out after reshuffling."""
    ranks = []
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    RCs1_ids = np.unique(df1.loc[df1.loc[:, "RC"], "Mouse_RFID"])
    mask = np.logical_and(df1["Mouse_RFID"].isin(RCs1_ids), ~df1["RC"])
    Ranks1 = df1.loc[mask, "rank_by_tube"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2_ids = np.unique(df2.loc[df2.loc[:, "RC"], "Mouse_RFID"])
    mask = np.logical_and([id in RCs2_ids for id in df2["Mouse_RFID"]], ~df2["RC"])
    Ranks2 = df2.loc[mask, "rank_by_tube"].to_numpy()
    
    arr = np.concatenate((Ranks1, Ranks2))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
        ax.legend()
    ax.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.8, color = "gray", alpha = 0.7, label = "ex sRC") 
    ax.set_xlabel("Tube rank", fontsize = 15)
    ax.set_ylabel("Count", fontsize = 15); 
    ticklabels = [i for i in range(1, 11)]
    ax.set_xticks([i + 0.5 for i in range(1, 11)], ticklabels, fontsize = 12)
    plt.title("Tube rank of ex sRC members")
    return arr

    
def chasingrank_vs_rc(ax = None):
    """Plots the chasing rank of rich club members for both cohorts"""

    ranks = []
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"

    df1 = pd.read_excel(path_cohort1)
    RCs1 = df1.loc[:, "RC"].to_numpy()
    Ranks1 = df1.loc[:, "rank_by_chasing"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2 = df2.loc[:, "RC"].to_numpy()
    Ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    
    arr = np.concatenate((Ranks1[RCs1], Ranks2[RCs2]))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.6, color = "blue") 
    ax.set_xlabel("Chasing rank", fontsize = 15)
    ax.set_ylabel("Count", fontsize = 15); 
    ticklabels = [i for i in range(1, 11)]
    ax.set_xticks([i + 0.5 for i in range(1, 11)], ticklabels, fontsize = 12)
    
def chasingrank_vs_exrc(ax = None):
    """Plots the tube ranks of mice that were once in the rich club, but got out after reshuffling."""
    ranks = []
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"

    df1 = pd.read_excel(path_cohort1)
    RCs1_ids = np.unique(df1.loc[df1.loc[:, "RC"], "Mouse_RFID"])
    mask = np.logical_and([id in RCs1_ids for id in df1["Mouse_RFID"]], ~df1["RC"])
    Ranks1 = df1.loc[mask, "rank_by_chasing"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2_ids = np.unique(df2.loc[df2.loc[:, "RC"], "Mouse_RFID"])
    mask = np.logical_and([id in RCs2_ids for id in df2["Mouse_RFID"]], ~df2["RC"])
    Ranks2 = df2.loc[mask, "rank_by_chasing"].to_numpy()
    
    arr = np.concatenate((Ranks1, Ranks2))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', alpha = 0.7, rwidth = 0.8, color = "gray") 
    ax.set_xlabel("Chasing rank", fontsize = 15)
    ax.set_ylabel("Count", fontsize = 15); 
    ticklabels = [i for i in range(1, 11)]
    ax.set_xticks([i + 0.5 for i in range(1, 11)], ticklabels, fontsize = 12)
    
def tuberank_vs_nonrc_WT(ax = None, norm = False):
    """Plots the tube rank of nonrich club, non mutants members for both cohorts"""
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    RCs1 = df1.loc[:, "RC"].to_numpy()
    mut1 = df1.loc[:, "mutant"].values
    Ranks1 = df1.loc[:, "rank_by_tube"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2 = df2.loc[:, "RC"].to_numpy()
    mut2 = df2["genotype"].values == "Oxt"
    Ranks2 = df2.loc[:, "rank_by_tube"].to_numpy()
    
    arr = np.concatenate((Ranks1[np.logical_and(~RCs1, ~mut1)], Ranks2[np.logical_and(~RCs2, mut2)]))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.7, color = "gray", density = norm) 
    ax.set_xlabel("Tube rank", fontsize = 15)
    ax.set_ylabel("Count", fontsize = 15); 
    ticklabels = [i for i in range(1, 11)]
    ax.set_xticks([i + 0.5 for i in range(1, 11)], ticklabels, fontsize = 12)
    return arr
    
def chasingrank_vs_nonrc_WT(norm = False):
    """Plots the chasing rank of nonrich club members for both cohorts"""
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    RCs1 = df1.loc[:, "RC"].to_numpy()
    mut1 = df1.loc[:, "mutant"].values
    Ranks1 = df1.loc[:, "rank_by_chasing"].to_numpy()

    df2 = pd.read_excel(path_cohort2)
    RCs2 = df2.loc[:, "RC"].to_numpy()
    mut2 = df2["genotype"].values == "Oxt"
    Ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    
    arr = np.concatenate((Ranks1[np.logical_and(~RCs1, ~mut1)], Ranks2[np.logical_and(~RCs2, mut2)]))
    
    plt.figure()
    plt.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.9, color = "gray", density = norm) 
    plt.xlabel("Chasing rank", fontsize = 15) 
    plt.ylabel("Number of observations", fontsize = 15); 
    ticklabels = [i for i in range(1, 12)]
    plt.xticks([i + 0.5 for i in range(1, 12)], ticklabels)
    plt.title("Chasing rank of non rich-club members")
    plt.show()

def approach_order_WT(out = True, both = False):
    """ Histogram of the approach order for mutants (i.e., if we rank them by according to how much total approaches they perform.)
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    approach_order_out, approach_order_in = [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    mutants1 = df1.loc[:, "mutant"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, mutant in enumerate(mutants1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if ~mutant:
                    wt_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(wt_idx))
                    approach_order_in.append(list(current_orders_in).index(wt_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    mutants2 = df2.loc[:, "genotype"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, mutant in enumerate(mutants2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mutant != "Oxt":
                    wt_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(wt_idx))
                    approach_order_in.append(list(current_orders_in).index(wt_idx))
        
    plt.figure()
    if both:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Outgoing", alpha = 0.5)
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Ingoing", alpha = 0.5)
        plt.legend()
    elif out:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')
    else:
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')

    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 

    if both:
        plt.xlabel(r"Appraoch order", fontsize = 15)
    elif out:
        plt.xlabel(r"Appraoch order (outgoing)", fontsize = 15)
    else:
        plt.xlabel(r"Appraoch order (ingoing)", fontsize = 15)
    plt.ylabel(r"Count", fontsize = 15)
    plt.title("Approach order of WT", fontsize = 18)
    plt.show()

def approach_order_RC(out = True, both = False):
    """ Histogram of the approach order for RC members (i.e., if we rank them by according to how much total approaches they perform.)
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    approach_order_out, approach_order_in = [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) 
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(rc_idx))
                    approach_order_in.append(list(current_orders_in).index(rc_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(rc_idx))
                    approach_order_in.append(list(current_orders_in).index(rc_idx))
        
    plt.figure()
    if both:
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Ingoing", alpha = 0.5)
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Outgoing", alpha = 0.5)
        plt.legend()
    elif out:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')
    else:
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')

    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"], fontsize = 15) 
    plt.yticks(np.arange(0, 12, 2), fontsize = 15)  # Ensure ticks are centered on 0 through 9

    if both:
        plt.xlabel(r"Approach order", fontsize = 19)
    elif out:
        plt.xlabel(r"Approach order (outgoing)", fontsize = 20)
    else:
        plt.xlabel(r"Approach order (ingoing)", fontsize = 20)
    plt.ylabel(r"Count", fontsize = 19)
    # plt.title("Approach order of RC members", fontsize = 18)
    plt.tight_layout()
    plt.show()

def approachRank_RC():
    """ Histogram of the approach david score for RC members.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    approach_dir = "..\\data\\both_cohorts_7days\\"

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    scores = []

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    scores.append(list(hier_mat.davids_score().keys()).index(mouse_names[idx]))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "genotype"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                        delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    scores.append(list(hier_mat.davids_score().keys()).index(mouse_names[idx]))
        
    plt.figure()
    plt.hist(scores, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')#
    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"]) 
    plt.xlabel(r"Approach rank", fontsize = 15)
    plt.ylabel(r"Count", fontsize = 15)
    plt.title(r"Appraoch rank of RC", fontsize = 18)
    plt.show()

def chasingOrder_RC(out = True, both = False):

    """ Histogram of the chasing order for RC members (i.e., if we rank them by according to how much total approaches they perform.)
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"

    approach_order_out, approach_order_in = [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]

        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(rc_idx))
                    approach_order_in.append(list(current_orders_in).index(rc_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        approaches_out = np.sum(approach_matrix, axis = 1)
        approaches_in = np.sum(approach_matrix, axis = 0)
        current_orders_out = np.argsort(-approaches_out)
        current_orders_in = np.argsort(-approaches_in)
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(current_orders_out).index(rc_idx))
                    approach_order_in.append(list(current_orders_in).index(rc_idx))
       
    plt.figure(figsize=(5,4))

    if both:
        # plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Ingoing", alpha = 0.5)
        # plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), stacked = True, rwidth= 0.8, align='mid', edgecolor='black', label = "Outgoing", alpha = 0.5)
        # plt.legend()
        points = list(zip(approach_order_in, approach_order_out))
        counts = Counter(points)
        sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
        plt.scatter(approach_order_in, approach_order_out, alpha = 0.3, c = "k", s = sizes)
        plt.xlabel(r"Chasing order (in)", fontsize = 15)
        plt.ylabel(r"Chasing order (out)", fontsize = 15)
        plt.title("Chasing order for RC", fontsize = 20)
        return
    elif out:
        plt.hist(approach_order_out, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')
    else:
        plt.hist(approach_order_in, bins = np.arange(-0.5, 10.5), rwidth= 0.8, align='mid', color = 'gray', edgecolor='black')

    plt.xticks(np.arange(0, 10), labels=["1","2","3","4","5","6","7","8","9","10"], fontsize = 15) 
    plt.yticks(np.arange(0, 12, 2), fontsize = 15)  # Ensure ticks are centered on 0 through 9

    if out:
        plt.xlabel(r"Chasing order (outgoing)", fontsize = 20)
        plt.ylabel(r"Count", fontsize = 19)
    else:
        plt.xlabel(r"Chasing order (ingoing)", fontsize = 20)
        plt.ylabel(r"Count", fontsize = 19)
    # plt.title("Approach order of RC members", fontsize = 18)
    plt.show()
   
def chasingFraction_RC(out = True, ax = None):
    """ 
    Histogram of the fraction of total chasings in a group performed by RC members
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"

    approach_order_out, approach_order_in = [], []
    RFIDs = []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]

        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, rc in enumerate(RCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(chasings_out)[rc_idx])
                    approach_order_in.append(list(chasings_in)[rc_idx])
                    RFIDs.append(mouse_names[idx])

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
       
        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, rc in enumerate(RCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(chasings_out)[rc_idx])
                    approach_order_in.append(list(chasings_in)[rc_idx])
                    RFIDs.append(mouse_names[idx])
                      
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5,4))

    if out:
        ax.hist(approach_order_out, bins = np.linspace(0, 0.5, 11), rwidth= 0.6, align='mid', color = 'blue')
    else:
        ax.hist(approach_order_in, bins = np.linspace(0, 0.5, 11), rwidth= 0.6, align='mid', color = 'blue')

    if out:
        ax.set_xlabel(r"Chasing fraction (outgoing)", fontsize = 20)
    else:
        ax.set_xlabel(r"Chasing fraction (incoming)", fontsize = 20)
    ax.set_ylabel(r"Count", fontsize = 19)
    if out:
        return approach_order_out, RFIDs
    else:
        return approach_order_in, RFIDs

    
def chasingFraction_exRC(out = True, ax = None):
    """
    Histogram of the chasing fraction for ex RC members
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"

    approach_order_out, approach_order_in = [], []
    RFIDs = []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()    
    RCs1_ids = np.unique(df1.loc[df1.loc[:, "RC"], "Mouse_RFID"])
    exRCs1 = np.logical_and([id in RCs1_ids for id in df1["Mouse_RFID"]], ~df1["RC"]).to_numpy()
    
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]

        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, rc in enumerate(exRCs1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(chasings_out)[rc_idx])
                    approach_order_in.append(list(chasings_in)[rc_idx])
                    RFIDs.append(mouse_names[idx])

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2_ids = np.unique(df2.loc[df2.loc[:, "RC"], "Mouse_RFID"])
    exRCs2 = np.logical_and([id in RCs2_ids for id in df2["Mouse_RFID"]], ~df2["RC"]).to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
       
        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, rc in enumerate(exRCs2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if rc:
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out.append(list(chasings_out)[rc_idx])
                    approach_order_in.append(list(chasings_in)[rc_idx])
                    RFIDs.append(mouse_names[idx])
                           
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5,4))

    if out:
        ax.hist(approach_order_out, bins = np.linspace(0, 0.5, 11), rwidth= 0.8, align='mid', color = 'gray', alpha = 0.7)
    else:
        ax.hist(approach_order_in, bins = np.linspace(0, 0.5, 11), rwidth= 0.8, align='mid', color = 'gray', alpha = 0.7)

    if out:
        ax.set_xlabel(r"Chasing fraction (outgoing)", fontsize = 15)
    else:
        ax.set_xlabel(r"Chasing fraction (incoming)", fontsize = 15)
    ax.set_ylabel(r"Count", fontsize = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    if out:
        return approach_order_out, RFIDs
    else:
        return approach_order_in, RFIDs
    
def tuberanks_sep():
    """ Histogram of tube ranks discriminating between RC, mutants and others.
    """
    # first cohort
    ranks = []
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    RCs1 = df1.loc[:, "RC"].to_numpy()
    mut1 = df1.loc[:, "mutant"].values
    Ranks1 = df1.loc[:, "rank_by_tube"].to_numpy()
    RFIDs1 = df1["Mouse_RFID"]

    df2 = pd.read_excel(path_cohort2)
    RCs2 = df2.loc[:, "RC"].to_numpy()
    mut2 = df2["genotype"].values == "Oxt"
    Ranks2 = df2.loc[:, "rank_by_tube"].to_numpy()
    RFIDs2 = df2["Mouse_RFID"]
    
    tube_ranks_mut = np.concatenate((Ranks1[mut1], Ranks2[mut2]))
    tube_ranks_others = np.concatenate((Ranks1[np.logical_and(~RCs1, ~mut1)], Ranks2[np.logical_and(~RCs2, mut2)]))
    tube_ranks_rc = np.concatenate((Ranks1[RCs1], Ranks2[RCs2]))
    RFIDs_rc = np.concatenate((RFIDs1[RCs1], RFIDs2[RCs2]))
    RFIDs_rest = np.concatenate((RFIDs1[np.logical_and(~RCs1, ~mut1)], RFIDs2[np.logical_and(~RCs2, mut2)]))
    RFIDs_mut = np.concatenate((RFIDs1[mut1], RFIDs2[mut2]))

    
    plt.figure(figsize = (5, 5))
    
    plt.hist(tube_ranks_rc, bins = np.arange(0.5, 11.5), density = True, rwidth= 0.8, align='mid', color = 'blue', edgecolor='black', alpha = 0.8, label = "RC")
    plt.hist(tube_ranks_mut, bins = np.arange(0.5, 11.5), density = True, rwidth= 0.8, align='mid', color = 'darkred', edgecolor='black', alpha = 0.8, label = "Mutants")
    plt.hist(tube_ranks_others, bins = np.arange(0.5, 11.5), density = True, rwidth= 0.8, align='mid', color = 'grey', edgecolor='black', alpha = 0.8, label = "Others")

    plt.xticks(np.arange(1, 11), labels=["1","2","3","4","5","6","7","8","9","10"], fontsize = 15) 
    plt.yticks(np.linspace(0, 0.4, 5), fontsize = 15)  # Ensure ticks are centered on 0 through 9
    
    plt.xlabel(r"Tube ranks", fontsize = 20)
    plt.ylabel(r"Density", fontsize = 19)
    plt.legend(fontsize = 15)
    print("p RC vs Others:")
    add_group_significance([tube_ranks_rc, tube_ranks_others], [RFIDs_rc, RFIDs_rest])
    print("p mutants vs Others:")
    add_group_significance([tube_ranks_mut, tube_ranks_others], [RFIDs_mut, RFIDs_rest])

    plt.show()
    
def chasingFraction_sep(out = True):
    """Histogram of chasing fractions discriminating between RC, mutants and others.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"
  
    chasing_frac_rc, chasing_frac_mutants, chasing_frac_others = [], [], []
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()    
    RCs1_ids = np.unique(df1.loc[df1.loc[:, "RC"], "Mouse_RFID"])
    mut1_ids = np.unique(df1.loc[df1.loc[:, "mutant"], "Mouse_RFID"])
    RFIDs_rc, RFIDs_mut, RFIDs_others = [], [], []

    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        # mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        
        if out:
            chasings_out = np.sum(approach_matrix, axis = 1)
            chasings_out = chasings_out/np.sum(chasings_out)
        else:
            chasings_out = np.sum(approach_matrix, axis = 0)
            chasings_out = chasings_out/np.sum(chasings_out)
        for idx, name in enumerate(names_in_approach_matrix):
            if name in RCs1_ids:
                rc_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_rc.append(list(chasings_out)[rc_idx])
                RFIDs_rc.append(name)
            elif name in mut1_ids:
                mut1_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_mutants.append(list(chasings_out)[mut1_idx])
                RFIDs_mut.append(name)
            else:
                other_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_others.append(list(chasings_out)[other_idx])
                RFIDs_others.append(name)

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2_ids = np.unique(df2.loc[df2.loc[:, "RC"], "Mouse_RFID"])
    mut2_ids = np.unique(df1.loc[df2.loc[:, "genotype"] == "Oxt", "Mouse_RFID"])
    # RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(11, 18): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        # mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        if out:
            chasings_out = np.sum(approach_matrix, axis = 1)
            chasings_out = chasings_out/np.sum(chasings_out)
        else:
            chasings_out = np.sum(approach_matrix, axis = 0)
            chasings_out = chasings_out/np.sum(chasings_out)
        for idx, name in enumerate(names_in_approach_matrix):
            if name in RCs2_ids:
                rc_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_rc.append(list(chasings_out)[rc_idx])
                RFIDs_rc.append(name)
            elif name in mut2_ids:
                mut1_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_mutants.append(list(chasings_out)[mut1_idx])
                RFIDs_mut.append(name)
            else:
                other_idx = np.where(name == names_in_approach_matrix)[0][0]
                chasing_frac_others.append(list(chasings_out)[other_idx])
                RFIDs_others.append(name)

    fig, ax = plt.subplots(1, 1)
    
    h = ax.hist(chasing_frac_rc, bins = np.linspace(0, 0.5, 11), density = True, rwidth= 0.8, align='mid', color = 'blue', edgecolor='black', alpha = 0.8, label = "RC")
    ax.hist(chasing_frac_mutants, bins = np.linspace(0, 0.5, 11), density = True, rwidth= 0.8, align='mid', color = 'darkred', edgecolor='black', alpha = 0.8, label = "Mutants")
    ax.hist(chasing_frac_others, bins = np.linspace(0, 0.5, 11), density = True, rwidth= 0.8, align='mid', color = 'grey', edgecolor='black', alpha = 0.8, label = "Others")
    
    if out:
        ax.set_xlabel(r"Fraction of total chases in group", fontsize = 20)
    else:
        ax.set_xlabel(r"Fraction of total being chased", fontsize = 20)

    ax.set_ylabel(r"Density", fontsize = 19)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    plt.legend(fontsize = 15)
    plt.tight_layout()
    
    print("p RC vs Others:")
    add_group_significance([chasing_frac_rc, chasing_frac_others], [RFIDs_rc, RFIDs_others])
    print("p mutants vs Others:")
    add_group_significance([chasing_frac_mutants, chasing_frac_others], [RFIDs_mut, RFIDs_others])

    plt.show()
    
    
def chasingFraction_MutVsWT(out = True, ax = None):
    """ 
    Histogram of the chasing fraction for mutants vs WT mice
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"

    approach_order_out_Mut, approach_order_in_Mut = [], []
    approach_order_out_WT, approach_order_in_WT = [], []

    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    Mut1 = df1.loc[:, "mutant"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in np.arange(1, 11): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs1[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]

        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, mut in enumerate(Mut1[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mut:
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out_Mut.append(list(chasings_out)[mut_idx])
                    approach_order_in_Mut.append(list(chasings_in)[mut_idx])
                else:
                    wt_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out_WT.append(list(chasings_out)[wt_idx])
                    approach_order_in_WT.append(list(chasings_in)[wt_idx])

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[~df2["Group_ID"].isin([16, 20, 21]), "Group_ID"].to_numpy()
    Mut2 = np.array(df2.loc[~df2["Group_ID"].isin([16, 20, 21]), "genotype"] == "Oxt")
    RFIDs2 = df2.loc[~df2["Group_ID"].isin([16, 20, 21]), "Mouse_RFID"].to_numpy()

    for group_idx in [11,12,13,14,15,17,18,19]: #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        mouse_names = RFIDs2[mouse_indices]
        approach_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
       
        chasings_out = np.sum(approach_matrix, axis = 1)
        chasings_out = chasings_out/np.sum(chasings_out)
        chasings_in = np.sum(approach_matrix, axis = 0)
        chasings_in = chasings_in/np.sum(chasings_in)
        for idx, mut in enumerate(Mut2[mouse_indices]):
            if np.any(mouse_names[idx] == names_in_approach_matrix):
                if mut:
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out_Mut.append(list(chasings_out)[mut_idx])
                    approach_order_in_Mut.append(list(chasings_in)[mut_idx])
                else:
                    wt_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_out_WT.append(list(chasings_out)[wt_idx])
                    approach_order_in_WT.append(list(chasings_in)[wt_idx])
                    

                      
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (5,4))

    if out:
        ax.hist(approach_order_out_WT, bins = np.linspace(0, 0.5, 11), rwidth= 0.9, align='mid', color = 'gray')
        ax.hist(approach_order_out_Mut, bins = np.linspace(0, 0.5, 11), rwidth= 0.75, align='mid', color = 'darkred')

    else:
        ax.hist(approach_order_in_WT, bins = np.linspace(0, 0.5, 11), rwidth= 0.9, align='mid', color = 'gray')
        ax.hist(approach_order_in_Mut, bins = np.linspace(0, 0.5, 11), rwidth= 0.75, align='mid', color = 'darkred')

    if out:
        ax.set_xlabel(r"Chasing fraction (outgoing)", fontsize = 17)
    else:
        ax.set_xlabel(r"Chasing fraction (ingoing)", fontsize = 17)
        
    ax.set_ylabel(r"Count", fontsize = 17)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # fig.tight_layout()
    
    fig2, ax2 = plt.subplots(1, 1)
    mutants = np.concatenate((Mut1, Mut2))
    RFIDs = np.concatenate((RFIDs1, RFIDs2))
    rfids_wt, rfids_mut = RFIDs[~mutants], RFIDs[mutants]
    data = [approach_order_out_WT, approach_order_out_Mut]
    bp = ax2.boxplot(data, widths=0.6, patch_artist=True, showfliers = False, zorder=1)
    ax2.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]))], 
                data[0], label = "mutant", zorder=2); 
    ax2.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]))], 
                data[1], label = "non-member", zorder=2); 
    add_group_significance(data, [rfids_wt, rfids_mut], ax2, bp)


if __name__ == "__main__":
## Main figure 5
    rich_club_piechart_both()

    # fig, ax = plt.subplots(1, 1)
    # rankex = np.array(tuberank_vs_exrc(ax))
    # rankrc = np.array(tuberank_vs_rc(ax))
    # t_stat, p_value = stats.ttest_ind(rankex, rankrc)
    # print(f"Paired t-test: p = {p_value}")

    # fig, ax = plt.subplots(1, 1, figsize = (3, 4))
    # chasingFraction_exRC(True, ax)
    # chasingFraction_RC(True, ax)
    # add_group_significance([frac_ex, frac_rc], [rfids_ex, rfids_rc], stat = "mean")

 ## Supplement
    # fig, ax = plt.subplots(1, 1, figsize = (3, 4))
    # frac_ex, rfids_ex = chasingFraction_exRC(False, ax)
    # frac_rc, rfids_rc = chasingFraction_RC(False, ax)
    # add_group_significance([frac_ex, frac_rc], [rfids_ex, rfids_rc], stat = "mean")
    # tuberanks_sep() # tube rank of sRC, mutants and non-sRC WT.
    # approach_order_sep(True) # outgoing approach order of sRC, mutants and non-sRC WT.
    # approach_order_sep(False) # ingoing approach order of sRC, mutants and non-sRC WT.
    # chasingFraction_sep(out = True)
    # chasingFraction_sep(out = False)
    
    # fig, ax = plt.subplots(1, 1)
    # chasingrank_vs_exrc(ax)
    # chasingrank_vs_rc(ax)



    