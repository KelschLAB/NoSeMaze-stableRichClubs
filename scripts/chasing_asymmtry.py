import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HierarchiaPy import Hierarchia


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

                
# chasing_asymmetry_by_tube()
# chasing_asymmetry_by_chasing()
chasing_asymmetry_by_chasingOrder(out=True)
# chasing_asymmetry_by_approachRank()



