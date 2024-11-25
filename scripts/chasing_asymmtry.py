import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_cohort1 = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\reduced_data.xlsx"
chasing_dir = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\chasing\\single\\"
path_cohort2 = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\validation_cohort.xlsx"
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
                                delimiter = ",", dtype=str)[1:, 1:].astype(np.int8)
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
                
direction = np.array(direction)
print(np.sum(direction)/len(direction))
                
            
    
# second cohort
# df2 = pd.read_excel(path_cohort2)
# groups2 = df2.loc[:, "Group_ID"].to_numpy()
# rank2 = df1.loc[:, "rank_by_tube"].to_numpy()


