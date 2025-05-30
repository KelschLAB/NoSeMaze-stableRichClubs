import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HierarchiaPy import Hierarchia
import matplotlib
matplotlib.style.use("seaborn-v0_8-whitegrid")
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from weighted_rc import weighted_rich_club
from scipy import stats
sys.path.append('..\\src\\')
from read_graph import read_graph
from collections import Counter


def chasing_asymmetry_by_tube():
    """Computes whether or not the social rank is a predictor for the asymmetry in chasing, i.e. if an animal is higher in hierarchy,
    will it tend to chase more animals that are lower in the hierarchy? This is done by (binarily) counting the number of time an animal 
    with a higher rank chased more an animal with a lower rank that it interacted with, and divide by the total number of diadic interactions.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    ranks1 = df1.loc[:, "rank_by_tube"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    # store whether the one with the higher rank chases more the one with the lower rank in diadic interaction
    direction = []  # 1 for yes 0 and for no

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        current_ranks = ranks1[mouse_indices]
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs1[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for idx_b, mouse_b in enumerate(RFIDs1[mouse_indices]):
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a] and current_ranks[idx_a] < current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
                    continue
                elif chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a] and current_ranks[idx_a] > current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                else:
                    direction.append(0)
                    # print("Weak -> Strong")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")

    # second cohort
    #    df2 = pd.read_excel(path_cohort2)
    # groups2 = df2.loc[:, "Group_ID"].to_numpy()
    # rank2 = df1.loc[:, "rank_by_tube"].to_numpy()
                    
    direction = np.array(direction)
    print(np.sum(direction)/len(direction))
 
def chasing_asymmetry_by_chasing():
    """Computes whether or not the chasing rank is a predictor for the asymmetry in chasing, i.e. if an animal is higher in hierarchy,
    will it tend to chase more animals that are lower in the hierarchy? This is done by (binarily) counting the number of time an animal 
    with a higher rank chased more an animal with a lower rank that it interacted with, and divide by the total number of diadic interactions.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    ranks1 = df1.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    # store whether the one with the higher rank chases more the one with the lower rank in diadic interaction
    direction = []  # 1 for yes 0 and for no

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        current_ranks = ranks1[mouse_indices]
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs1[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for idx_b, mouse_b in enumerate(RFIDs1[mouse_indices]):
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a] and current_ranks[idx_a] < current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
                    continue
                elif chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a] and current_ranks[idx_a] > current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                else:
                    direction.append(0)
                    # print("Weak -> Strong")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")

    # second cohort
    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(np.min(groups2)+1, np.max(groups2)+1): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        current_ranks = ranks2[mouse_indices]
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs2[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for idx_b, mouse_b in enumerate(RFIDs2[mouse_indices]):
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a] and current_ranks[idx_a] < current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
                    continue
                elif chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a] and current_ranks[idx_a] > current_ranks[idx_b]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                else:
                    direction.append(0)
                    # print("Weak -> Strong")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                    
    direction = np.array(direction)
    print(np.sum(direction)/len(direction))

def chasing_asymmetry_by_chasingOrder(out=True):
    """Computes whether or not the chasing order (i.e. rank the animals by how much chasing they do) is a predictor for the asymmetry in chasing, 
    i.e. if an animal is higher in hierarchy, will it tend to chase more animals that are lower in the hierarchy? 
    This is done by (binarily) counting the number of time an animal with a higher rank chased more an animal with a lower rank that it interacted with, 
    and divide by the total number of diadic interactions.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    ranks1 = df1.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    # store whether the one with the higher rank chases more the one with the lower rank in diadic interaction
    direction = []  # 1 for yes 0 and for no

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        outchasings = np.sum(chasing_matrix, axis = 1)
        inchasings = np.sum(chasing_matrix, axis = 0)
        if out:
            current_orders = np.argsort(-outchasings)
        else:
            current_orders = np.argsort(-inchasings)
        current_ranks = ranks1[mouse_indices]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs1[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for idx_b, mouse_b in enumerate(RFIDs1[mouse_indices]):
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if current_orders[where_a] < current_orders[where_b] and chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
                    continue
                elif current_orders[where_a] > current_orders[where_b] and chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                else:
                    direction.append(0)
                    # print("Weak -> Strong")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")

    # second cohort
    df2 = pd.read_excel(path_cohort2)
    groups2 = df2.loc[:, "Group_ID"].to_numpy()
    ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    for group_idx in range(np.min(groups2)+1, np.max(groups2)+1): #iterate over groups
        mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
        current_ranks = ranks2[mouse_indices]
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        outchasings = np.sum(chasing_matrix, axis = 1)
        current_orders = np.argsort(-outchasings)
        current_ranks = ranks2[mouse_indices]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(RFIDs2[mouse_indices]):
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for idx_b, mouse_b in enumerate(RFIDs2[mouse_indices]):
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if current_orders[where_a] < current_orders[where_b] and chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
                    continue
                elif current_orders[where_a] > current_orders[where_b] and chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a]:
                    direction.append(1)
                    # print("Strong -> Weak")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                else:
                    direction.append(0)
                    # print("Weak -> Strong")
                    # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
                    #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                    
    direction = np.array(direction)
    print(np.sum(direction)/len(direction))

def chasing_asymmetry_by_approachRank():
    """Computes whether or not the approach rank (i.e. david score based on approach matrix) is a predictor for the asymmetry in chasing, 
    i.e. if an animal is higher in approach hierarchy, will it tend to chase more animals that are lower in the hierarchy? 
    This is done by (binarily) counting the number of time an animal with a higher rank chased more an animal with a lower rank that it interacted with, 
    and divide by the total number of diadic interactions.
    """
    chasing_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"
    # first cohort
    direction = []  # 1 for yes 0 and for no

    for group_idx in range(1, 18): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        hier_mat = Hierarchia(approach_matrix, name_seq=names_in_approach_matrix)
        approach_david_score = hier_mat.davids_score()
        hier_mat = Hierarchia(chasing_matrix, name_seq=names_in_chasing_matrix)
        chasing_david_score = hier_mat.davids_score()
        for mouse_a in names_in_approach_matrix:
            # testing if mouse_a is found in chasing matrix
            if ~np.any(names_in_chasing_matrix == mouse_a):
                continue
            else:
                where_a = np.where(names_in_chasing_matrix == mouse_a)
            for mouse_b in names_in_approach_matrix:
                # testing if mouse_b is also found in chasing matrix
                if ~np.any(names_in_chasing_matrix == mouse_b):
                    continue
                else:
                    where_b = np.where(names_in_chasing_matrix == mouse_b)
                # testing which chases most and has higher rank
                if approach_david_score[mouse_a] > approach_david_score[mouse_b] and chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a]:
                    direction.append(1)
                elif mouse_a == mouse_b: # ignore self-interaction
                    continue
                elif approach_david_score[mouse_a] < approach_david_score[mouse_b] and chasing_david_score[mouse_a] < chasing_david_score[mouse_b]:
                    direction.append(1)
                else:
                    direction.append(0)
                   
    direction = np.array(direction)
    print(np.sum(direction)/len(direction))

def chasing_direction_RC():
    """Tests whether or not RC members preferentially chase other RC members.
    """
    path_cohort1 = "..\\data\\reduced_data.xlsx"
    chasing_dir = "..\\data\\chasing\\single\\"
    path_cohort2 = "..\\data\\validation_cohort.xlsx"
    # first cohort
    df1 = pd.read_excel(path_cohort1)
    groups1 = df1.loc[:, "group"].to_numpy()
    RCs1 = df1.loc[:, "RC"].to_numpy()
    RFIDs1 = df1.loc[:, "Mouse_RFID"].to_numpy()
    # store whether the one with the higher rank chases more the one with the lower rank in diadic interaction
    direction = []  # 1 for yes 0 and for no

    for group_idx in range(1, np.max(groups1)+1): #iterate over groups
        mouse_indices = np.where(groups1 == group_idx) # find out indices of mice from current group
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        max_idx = np.argmax(chasing_matrix, axis = 0)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        for idx_a, mouse_a in enumerate(names_in_chasing_matrix):
            if ~np.any(mouse_a == RFIDs1[mouse_indices]): # testing if mouse_a is found in chasing matrix
                continue
            where_a = np.where(mouse_a == RFIDs1[mouse_indices])
            if RCs1[mouse_indices][where_a]:
                where_chased = np.where(names_in_chasing_matrix[max_idx[idx_a]] == RFIDs1[mouse_indices])
                if len(where_chased[0]) == 0: # test if the chased mouse is also in excel sheet
                    continue
                if RCs1[where_chased]:
                    direction.append(1)
                else:
                    direction.append(0)

    # # second cohort
    # df2 = pd.read_excel(path_cohort2)
    # groups2 = df2.loc[:, "Group_ID"].to_numpy()
    # ranks2 = df2.loc[:, "rank_by_chasing"].to_numpy()
    # RFIDs2 = df2.loc[:, "Mouse_RFID"].to_numpy()

    # for group_idx in range(np.min(groups2)+1, np.max(groups2)+1): #iterate over groups
    #     mouse_indices = np.where(groups2 == group_idx) # find out indices of mice from current group
    #     current_ranks = ranks2[mouse_indices]
    #     chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
    #                                 delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
    #     outchasings = np.sum(chasing_matrix, axis = 1)
    #     current_orders = np.argsort(-outchasings)
    #     current_ranks = ranks2[mouse_indices]
    #     names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
    #                                         delimiter=",", dtype=str)[0, :][1:]
    #     names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
    #                                         delimiter=",", dtype=str)[0, :][1:]
    #     for idx_a, mouse_a in enumerate(RFIDs2[mouse_indices]):
    #         # testing if mouse_a is found in chasing matrix
    #         if ~np.any(names_in_chasing_matrix == mouse_a):
    #             continue
    #         else:
    #             where_a = np.where(names_in_chasing_matrix == mouse_a)
    #         for idx_b, mouse_b in enumerate(RFIDs2[mouse_indices]):
    #             # testing if mouse_b is also found in chasing matrix
    #             if ~np.any(names_in_chasing_matrix == mouse_b):
    #                 continue
    #             else:
    #                 where_b = np.where(names_in_chasing_matrix == mouse_b)
    #             # testing which chases most and has higher rank
    #             if current_orders[where_a] < current_orders[where_b] and chasing_matrix[where_a, where_b] > chasing_matrix[where_b, where_a]:
    #                 direction.append(1)
    #                 # print("Strong -> Weak")
    #                 # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
    #                 #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
    #             elif current_ranks[idx_a] == current_ranks[idx_b]: # ignore self-interaction
    #                 continue
    #             elif current_orders[where_a] > current_orders[where_b] and chasing_matrix[where_a, where_b] < chasing_matrix[where_b, where_a]:
    #                 direction.append(1)
    #                 # print("Strong -> Weak")
    #                 # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
    #                 #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
    #             else:
    #                 direction.append(0)
    #                 # print("Weak -> Strong")
    #                 # print("Name_a: "+mouse_a+" rank: "+str(current_ranks[idx_a])+" chases "+str(chasing_matrix[where_a, where_b])+\
    #                 #       " Name_b: "+mouse_b+" rank: "+str(current_ranks[idx_b])+" chases "+str(chasing_matrix[where_b, where_a])+"\n")
                    
    direction = np.array(direction)
    print(np.sum(direction)/len(direction))
    
def chasing_asymmetry(threshold = 2, qtl = 0.0, rc = False):
    """Shows that chasing is a highly asymmetric behavior, by showing the total lack of correlation between ingoing and
    outgoing chasings.
    inputs:
        - treshold: how many times an approaches has to be bigger 
            than the reciprocal one to consider it is not symmetric
        - qtl: quantile filter to cut the graph before computing the symmetry. 
            If set to 0.5, all approaches under the value of median approach are cut.
            Defaults to 0, to keep all data.
        - rc (bool): if true, only the appaoches of RC members towards other RC are considered.
    """
    chasing_dir = "..\\data\\chasing\\single\\"
    ingoings, outgoings = [], []
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [2, 4], [5,6], [0, 1], [3,4,6], [3, 5, 7], [6, 7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    cohorts_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16] if rc else range(1, 17)
    for rc_list_idx, group_idx in enumerate(cohorts_list): #iterate over groups
        chasing_matrix = np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                    delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_chasing_matrix =  np.loadtxt(chasing_dir+"G"+str(group_idx)+"_single_chasing.csv",
                                            delimiter=",", dtype=str)[0, :][1:]
        
        chasing_cut = np.quantile(chasing_matrix[:], qtl)#np.max(chasing_matrix[:])
        
        for idx_a, mouse_a in enumerate(names_in_chasing_matrix):
            for idx_b, mouse_b in enumerate(names_in_chasing_matrix):
                if idx_a != idx_b and (chasing_matrix[idx_a, idx_b] > chasing_cut or chasing_matrix[idx_b, idx_a] > chasing_cut):
                    if rc and (np.isin(idx_a, all_rc[rc_list_idx]) and np.isin(idx_b, all_rc[rc_list_idx])):
                        outgoings.append(chasing_matrix[idx_a, idx_b])
                        ingoings.append(chasing_matrix[idx_b, idx_a])
                    elif not rc:
                        outgoings.append(chasing_matrix[idx_a, idx_b])
                        ingoings.append(chasing_matrix[idx_b, idx_a])
        # plt.xscale('log')
        # plt.yscale('log')
    ingoings = np.array(ingoings)
    outgoings = np.array(outgoings)
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    asym_count = np.sum(ingoings > threshold * outgoings) + np.sum(outgoings > threshold * ingoings)
    asym_ratio = asym_count / len(ingoings)
    sizes = [asym_ratio, 1 - asym_ratio]
    labels = ["Asymmetric chasings", "Symmetric chasings"]
    colors = ['dimgray', 'lightgray']
    explode = (0.05, 0)  # Slight explode for asymmetric slice
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}\%', 
            colors=colors, explode=explode, startangle=90, 
            wedgeprops=dict(width=0.6, edgecolor='k'),
            textprops=dict(color="black", fontsize=16))
    title = "Symmetry of chasing \n from RC towards RC" if rc else "Symmetry of chasing \n whole population"
    plt.title(title, fontsize=18)
    # plt.tight_layout()
    plt.show()
    
def approach_symmetry(threshold = 2, qtl = 0.0, rc = False):
    """Shows that approach is a symmetric behavior, by showing the correlation between ingoing and
    outgoing approaches. If rc is True, only the approaches of RC members to themsleves will be compared.
    inputs:
        - treshold: how many times an approaches has to be bigger 
            than the reciprocal one to consider it is not symmetric
        - qtl: quantile filter to cut the graph before computing the symmetry. 
            If set to 0.5, all approaches under the value of median approach are cut.
            Defaults to 0, to keep all data.
        - rc (bool): if true, only the appaoches of RC members towards other RC are considered.
    """
    approach_dir = "..\\data\\averaged\\"
    ingoings, outgoings = [], []

    all_rc = [[0,6], [3, 8, 9], [2, 3, 6], [2, 4], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8], [5, 8], [0, 2], [], [], [2, 8, 9], []]
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    cohorts_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16] if rc else range(1, 17)
    for rc_list_idx, group_idx in enumerate(cohorts_list): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                     delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                     delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                                 delimiter=",", dtype=str)[0, :][1:]
                
        approach_cut = np.quantile(approach_matrix[:], qtl)#np.max(chasing_matrix[:])
        
        for idx_a, mouse_a in enumerate(names_in_approach_matrix):
            for idx_b, mouse_b in enumerate(names_in_approach_matrix):
                if idx_a != idx_b and (approach_matrix[idx_a, idx_b] > approach_cut or approach_matrix[idx_b, idx_a] > approach_cut):
                    if rc and (np.isin(idx_a, all_rc[rc_list_idx]) and np.isin(idx_b, all_rc[rc_list_idx])):
                        outgoings.append(approach_matrix[idx_a, idx_b])
                        ingoings.append(approach_matrix[idx_b, idx_a])
                    elif not rc:
                        outgoings.append(approach_matrix[idx_a, idx_b])
                        ingoings.append(approach_matrix[idx_b, idx_a])
        # plt.xscale('log')
        # plt.yscale('log')
    ingoings = np.array(ingoings)
    outgoings = np.array(outgoings)
    print(np.corrcoef(ingoings, outgoings))
    print((np.sum(ingoings > threshold*outgoings) + np.sum(outgoings > threshold*ingoings))/len(ingoings))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.figure()
    plt.pie([(np.sum(ingoings > threshold*outgoings) + np.sum(outgoings > threshold*ingoings))/len(ingoings),
              1 - (np.sum(ingoings > threshold*outgoings) + np.sum(outgoings > threshold*ingoings))/len(ingoings)], 
            labels = ["Asymmetric approaches", "Symmetric approaches"] , autopct=lambda frac: f'{frac :.1f}%', colors=['gray', 'silver'])
    title="Symmetry of approaches from RC towards RC" if rc else "Symmetry of approaches"
    plt.title(title)
    plt.tight_layout()
    # plt.scatter(outgoings, ingoings, c = 'k')
    plt.show()

    
    
def approach_symmetry_mutants(threshold = 1.5, qtl = 0.0):
    """Shows the symmetry of the approach behavior for mutants
    inputs:
        - treshold: how many times an approaches has to be bigger 
            than the reciprocal one to consider it is not symmetric
        - qtl: quantile filter to cut the graph before computing the symmetry. 
            If set to 0.5, all approaches under the value of median approach are cut.
            Defaults to 0, to keep all data.
    """
    approach_dir = "..\\data\\averaged\\"
    ingoings, outgoings = [], []

    all_mutants = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    cohorts_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    for rc_list_idx, group_idx in enumerate(cohorts_list): #iterate over groups
        approach_matrix = np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                     delimiter = ",", dtype=str)[1:, 1:].astype(np.int16) + np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_2.csv",
                                     delimiter = ",", dtype=str)[1:, 1:].astype(np.int16)
        names_in_approach_matrix =  np.loadtxt(approach_dir+"G"+str(group_idx)+"\\approaches_resD7_1.csv",
                                                 delimiter=",", dtype=str)[0, :][1:]
                
        approach_cut = np.quantile(approach_matrix[:], qtl)#np.max(chasing_matrix[:])
        
        for idx_a, mouse_a in enumerate(names_in_approach_matrix):
            for idx_b, mouse_b in enumerate(names_in_approach_matrix):
                if idx_a != idx_b and (approach_matrix[idx_a, idx_b] > approach_cut or approach_matrix[idx_b, idx_a] > approach_cut):
                    outgoings.append(approach_matrix[idx_a, idx_b])
                    ingoings.append(approach_matrix[idx_b, idx_a])
                    if np.isin(idx_a, all_mutants[rc_list_idx]) and np.isin(idx_b, all_mutants[rc_list_idx]):
                        outgoings.append(approach_matrix[idx_a, idx_b])
                        ingoings.append(approach_matrix[idx_b, idx_a])
                    
    ingoings = np.array(ingoings)
    outgoings = np.array(outgoings)
    plt.figure()
    plt.pie([(np.sum(ingoings > threshold*outgoings) + np.sum(outgoings > threshold*ingoings))/len(ingoings),
             1 - (np.sum(ingoings > threshold*outgoings) + np.sum(outgoings > threshold*ingoings))/len(ingoings)], 
            labels = ["Asymmetric approaches", "Symmetric approaches"] , autopct=lambda frac: f'{frac :.1f}%', colors=['gray', 'silver'])
    title = "Symmetry of approaches between mutants"
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def chasing_towards_piechart(normed = False, filter_big_chasers = False):
    # """
    # if both, animals are ranked by outgoing+ingoing chasings. else, they are ranked by outgoing chasings.
    # """
    # datapath = "..\\data\\chasing\\single\\"
    # # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # # performed
    # all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [5, 6], [0, 1], [3, 4, 6], [5, 7], [7, 8], [5, 8], [0, 2], [2, 8, 9]]
    # labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10", "G11", "G12", "G15"]
    
    # rc_chased = [] # total chasing from RC for thres 10%

    # for idx, g in enumerate(labels):
    #     data = read_graph([datapath+g+"_single_chasing.csv"])[0]
    #     for member in all_rc[idx]:
    #         members = np.array(all_rc[idx])
    #         avg_total_chasing = np.quantile(np.sum(data, axis = 1), 0.8)
    #         # avg_total_chasing = np.mean(np.sum(data, axis = 1))

    #         if not filter_big_chasers:
    #             if (np.any(np.argmax(data[member, :]) == members)) and (np.argmax(data[member, :]) != member):
    #                 rc_chased.append(1)
    #             else:
    #                 rc_chased.append(0)
    #         elif np.sum(data[member, :]) > avg_total_chasing:
    #             if (np.any(np.argmax(data[member, :]) == members)) and (np.argmax(data[member, :]) != member):
    #                 rc_chased.append(1)
    #             else:
    #                 rc_chased.append(0)
                
    # total_rc_chasings = np.array(rc_chased)
    # num_chasings = len(total_rc_chasings)
    # in_frac = np.sum(total_rc_chasings)
    # out_frac = num_chasings - np.sum(total_rc_chasings)
    
    in_frac, out_frac, num_chasings = 11, 14, 25 # hand counted
    
    in_frac = 100*(in_frac/num_chasings)
    out_frac = 100*(out_frac/num_chasings)
    if normed:
        in_frac = in_frac/2.5
        out_frac = out_frac/7.5
        in_frac = 100*(in_frac/(in_frac+out_frac))
        out_frac = 100*(out_frac/(in_frac+out_frac))
    
    labels = "Towards RC", "Outside RC"
    sizes = [in_frac, out_frac]
    fig, ax = plt.subplots()
    # ax.pie(sizes, labels=labels, autopct='%1.1f\%%')
    explode = (0.0, 0.0)  # Slight explode for asymmetric slice
    plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}\%', 
            explode=explode, startangle=90, 
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=5, alpha=0.7),
            textprops=dict(color="black", fontsize=16))    
    
    if normed:
        ax.set_title("Direction of maximal chasings\nfrom RC members (normalized)")
    else:
        ax.set_title("Direction of maximal chasings\nfrom RC members")
    plt.tight_layout()
    plt.show()
    
def chasing_manhanttan_plot(out = True):
    
    datapath = "..\\data\\chasing\\single\\"
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [5, 6], [0, 1], [3, 4, 6], [5, 7], [7, 8], [5, 8], [0, 2], [2, 8, 9]]
    labels = ["G1", "G2", "G3", "G5", "G6", "G7", "G8","G10", "G11", "G12", "G15"]
    
    width = 0.05  # the width of the bars
    x = np.arange(11)  # the label locations

    fig, ax = plt.subplots(layout='constrained', figsize =  (12, 6))
    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        chasings = [] 
        colors = []
        for member in range(len(data)):
            try:
                if out:
                    chasings.append(np.sum(data[member, :]))
                else:
                    chasings.append(np.sum(data[:, member]))
    
                if np.any(member == np.array(all_rc[idx])):
                    colors.append("green")
                else:
                    colors.append("blue")
            except:
                chasings.append(0)
                colors.append("blue")
        multiplier = 0
        for i in range(len(chasings)):
            offset = width * multiplier
            rects = ax.bar(x[idx] + offset, chasings[i], width, label = g, color = colors[i])
            multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if out:
        ax.set_ylabel('Total outgoing chasings', fontsize = 20)
    else:
        ax.set_ylabel('Total ingoing chasings', fontsize = 20)

    ax.set_xticks(x + width, labels)
    custom_handles = [
        plt.Line2D([0], [0], color='green', lw=2, label='RC'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Non-RC')
    ]
    plt.legend(handles=custom_handles, fontsize = 16)    
    plt.tight_layout()
    plt.show()
    
def cumulative_chasings():
    """
    Plot the cumulative distribution of (normalized) chasings as a function of the rank of the animals
    ranked by how much chasing they do. Normalized by the max chasing observe in groups.
    """
    datapath = "..\\data\\chasing\\single\\"
    cumulative_chasings_out = []
    cumulative_chasings_in = []

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G15"]

    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        rank_out = np.argsort(-np.sum(data, axis = 1))
        summed_chasings_out = np.sum(data, axis = 1) # getting the summed chasing for all animals
        rank_in = np.argsort(-np.sum(data, axis = 0))
        summed_chasings_in = np.sum(data, axis = 0) # getting the summed chasing for all animals

        normalization_out = np.sum(summed_chasings_out)
        total_chasings_out = np.sum(data[rank_out, :], axis = 1)/normalization_out
        cumulative_sum_out = np.cumsum(total_chasings_out)
        cumulative_chasings_out.append(cumulative_sum_out)
        normalization_in = np.sum(summed_chasings_in)
        total_chasings_in = np.sum(data[rank_in, :], axis = 0)/normalization_in
        cumulative_sum_in = np.cumsum(total_chasings_in)
        cumulative_chasings_in.append(cumulative_sum_in)
        

    avg_cumsum_out = np.mean(np.array(cumulative_chasings_out), axis = 0)
    std_cumsum_out = np.std(np.array(cumulative_chasings_out), axis = 0)
    avg_cumsum_in = np.mean(np.array(cumulative_chasings_in), axis = 0)
    std_cumsum_in = np.std(np.array(cumulative_chasings_in), axis = 0)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    plt.plot(avg_cumsum_out, marker = "o",  c = "k", label = "Outgoing")
    plt.plot(avg_cumsum_in, marker = "o", label = "Ingoing", c = "blue")
    plt.errorbar(np.arange(10), avg_cumsum_out, std_cumsum_out, c = 'k')
    plt.errorbar(np.arange(10), avg_cumsum_in, std_cumsum_in, c = "blue")
    plt.xlabel("Ranking by total performed chasing", fontsize = 14)
    plt.xticks(np.arange(10), labels = np.arange(1, 11))
    plt.ylabel("Average chasing cumulative sum\n normalized by total group chasings", fontsize = 15)
    plt.xticks(np.arange(10))
    plt.legend()
    plt.show()
    
def cumulative_approaches():
    """
    Plot the cumulative distribution of (normalized) chasings as a function of the rank of the animals
    ranked by how much chasing they do. Normalized by the max chasing observe in groups.
    """
    datapath = "..\\data\\averaged\\"
    cumulative_chasings_out = []
    cumulative_chasings_in = []

    labels = ["G1", "G2", "G4", "G5", "G6", "G8", "G11", "G12","G14", "G15", "G16", "G17"]

    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"\\approaches_resD7_1.csv"])[0] + read_graph([datapath+g+"\\approaches_resD7_1.csv"])[0]
        rank_out = np.argsort(-np.sum(data, axis = 1))
        rank_in = np.argsort(-np.sum(data, axis = 0))

        summed_chasings = np.sum(data) # getting the summed chasing for all animals, in order to normalize results
        normalization = np.sum(summed_chasings)
        total_chasings_out = np.sum(data[rank_out, :], axis = 1)/normalization
        cumulative_sum_out = np.cumsum(total_chasings_out)
        cumulative_chasings_out.append(cumulative_sum_out)
        total_chasings_in = np.sum(data[rank_in, :], axis = 0)/normalization
        cumulative_sum_in = np.cumsum(total_chasings_in)
        cumulative_chasings_in.append(cumulative_sum_in)
        print(len(cumulative_sum_out))
        

    avg_cumsum_out = np.mean(np.array(cumulative_chasings_out), axis = 0)
    std_cumsum_out = np.std(np.array(cumulative_chasings_out), axis = 0)
    avg_cumsum_in = np.mean(np.array(cumulative_chasings_in), axis = 0)
    std_cumsum_in = np.std(np.array(cumulative_chasings_in), axis = 0)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    plt.plot(avg_cumsum_out, marker = "o",  c = "k", label = "Outgoing")
    plt.plot(avg_cumsum_in, marker = "o", label = "Ingoing", c = "blue")
    plt.errorbar(np.arange(10), avg_cumsum_out, std_cumsum_out, c = 'k')
    plt.errorbar(np.arange(10), avg_cumsum_in, std_cumsum_in, c = "blue")
    plt.xlabel("Ranking by total performed approaches", fontsize = 14)
    plt.xticks(np.arange(10), labels = np.arange(1, 11))
    plt.ylabel("Average chasing cumulative sum\n normalized by total group approaches", fontsize = 15)
    plt.xticks(np.arange(10))
    plt.legend()
    plt.show()
    
    
chasing_towards_piechart(True)
# cumulative_approaches()
# chasing_manhanttan_plot(False)
# cumulative_chasings()

# plot normalized cumulative plots of chasings for ranked animals (by chasing), in and out
# plot chasings ranking for ingoing.



# chasing_asymmetry_by_tube()
# chasing_asymmetry_by_chasing()
# chasing_asymmetry_by_chasingOrder(out=True)
# chasing_asymmetry_by_approachRank()
# chasing_direction_RC()
# chasing_asymmetry(1.5, 0.0, True)
# chasing_asymmetry(1.5, 0.0, False)

# approach_symmetry(1.5, 0.0, rc = True)

# approach_symmetry_mutants()

