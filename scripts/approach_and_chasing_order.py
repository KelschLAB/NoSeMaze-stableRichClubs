import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..\\src\\')
sys.path.append(os.getcwd())
from read_graph import read_graph
import pandas as pd
from HierarchiaPy import Hierarchia


all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]

def approach_order(out = True):
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
    histo1 = df1.loc[:, "histology"]
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
                if rc == "VRAI":
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype1[idx] == "VRAI" and histo1[idx] == "strong":
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))

    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    genotype2 = df2.loc[:, "genotype"]
    histo2 = df2.loc[:, "histology"]
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
                if rc == "VRAI":
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype2[idx] == "Oxt" and histo2[idx] == "strong":
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
    
def chasing_order(out = True):
    """ Histogram of chasings order discriminating between RC, mutants and others.
    """
    path_cohort1 = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\reduced_data.csv"
    path_cohort2 = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\validation_cohort.csv"
    chasing_dir = "..\\data\\chasing\\single\\"

    approach_order_rc, approach_order_mut, approach_order_others = [], [], []
    # first cohort
    df1 = pd.read_csv(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    genotype1 = df1.loc[:, "mutant"]
    histo1 = df1.loc[:, "histology"]
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
                if rc == "VRAI":
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype1[idx] == "VRAI" and histo1[idx] == "strong":
                    mut_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_mut.append(list(current_orders).index(mut_idx))
                else:
                    other_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_others.append(list(current_orders).index(other_idx))

    df2 = pd.read_csv(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    RCs2 = df2.loc[:, "RC"].to_numpy()
    genotype2 = df2.loc[:, "genotype"]
    histo2 = df2.loc[:, "histology"]
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
                if rc == "VRAI":
                    rc_idx = np.where(mouse_names[idx] == names_in_approach_matrix)[0][0]
                    approach_order_rc.append(list(current_orders).index(rc_idx))
                elif genotype2[idx] == "Oxt" and histo2[idx] == "strong":
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
    

    
# approach_order() # outgoing
# approach_order(False) # ingoing
# chasing_order() # outgoing
# chasing_order(False) # ingoing