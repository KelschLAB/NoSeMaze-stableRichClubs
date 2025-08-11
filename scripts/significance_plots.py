import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def chasing_towards(rc_size = 3, total_chasings = 100):
    """
    Computes the distribution of observed chasings towards sRC members, if chasings were
    distributed by random chance.
    """
    percentage_list = []
    for i in tqdm(range(10000)):
        shuffled_arr = np.array([1,2,3,4,5,6,7,8,9,10])
        hits = 0
        for chasing in range(total_chasings): # assuming 500 chasing events per cohort
            np.random.shuffle(shuffled_arr)
            if shuffled_arr[0] in [1,2,3]:
                hits += 1
        percentage_list.append(100*(hits/total_chasings))
        
    fig = plt.figure()
    h = plt.hist(percentage_list, bins = np.arange(0, 100, 2), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(percentage_list, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(percentage_list, 97.5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(44, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.sum(h[0][22:]), 5)), (9, 0.05), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.xlabel("Percentage of chasing towards other sRC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    plt.show()

# mutants exclusion analysis
def mutants_in_rc_mnn2_k2(iterations = 10000):
    # extracting meta data for mnn = 2 and k = 2
    metadata_path = "..\\data\\meta_data_exclusion_in_second\\meta_data_k2.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    # getting number of mutants, size of sRC for each group and actual observation
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    mutants_number = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["mutant"]) for g in group_list] # & (metadata_df["histology"] != "weak")
    observed = np.sum(metadata_df["mutant"] & metadata_df["RC"]) # number of mutants observed in the sRC
 
    # permutation test
    number_of_hits = []   
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # pre-alocating memory for shuffling
    for _ in tqdm(range(iterations)):
        hits = 0
        for m, r in zip(mutants_number, rc_sizes): # iterating through all groups
            np.random.shuffle(shuffled_arr)
            selected = shuffled_arr[:r]
            hits += np.sum([idx in selected for idx in range(1, m + 1)])
        number_of_hits.append(hits)
    
    fig = plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) <= observed)
    print(p_value)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.75*max(h[0])), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in stable rich-club\n mnn = 2, k = 2")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    
def mutants_in_rc_mnn3_k3(iterations = 10000): #mnn = 3, rc = 3
    metadata_path = "..\\data\\meta_data_exclusion_in_second\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)

    # getting number of mutants, size of sRC for each group and actual observation
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    mutants_number = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["mutant"]) for g in group_list] # & (metadata_df["histology"] != "weak")
    observed = np.sum(metadata_df["mutant"] & metadata_df["RC"]) # number of mutants observed in the sRC

    # permutation test
    number_of_hits = []
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for _ in tqdm(range(iterations)):
        hits = 0
        for m, r in zip(mutants_number, rc_sizes): # iterating through all groups
            np.random.shuffle(shuffled_arr)
            selected = shuffled_arr[:r]
            hits += np.sum([idx in selected for idx in range(1, m + 1)])
        number_of_hits.append(hits)
    
    fig = plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) <= observed)
    print(p_value)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.75*max(h[0])), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in stable rich-club\n mnn = 3, k = 3")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    
def mutants_in_rc_mnn4_k4(weak = True, iterations = 10000): # mnn4 and rc 4
    metadata_path = "..\\data\\meta_data_exclusion_in_second\\meta_data_k4.csv"
    metadata_df = pd.read_csv(metadata_path)

    # getting number of mutants, size of sRC for each group and actual observation
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    mutants_number = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["mutant"]) for g in group_list] # & (metadata_df["histology"] != "weak")
    observed = np.sum(metadata_df["mutant"] & metadata_df["RC"]) # number of mutants observed in the sRC
    
    # permutation test
    number_of_hits = []
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for _ in tqdm(range(iterations)):
        hits = 0
        for m, r in zip(mutants_number, rc_sizes): # iterating through all groups
            np.random.shuffle(shuffled_arr)
            selected = shuffled_arr[:r]
            hits += np.sum([idx in selected for idx in range(1, m + 1)])
        number_of_hits.append(hits)
        
    plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(17), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) <= observed)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.75*max(h[0])), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in stable rich-club\n mnn = 4, k = 4")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    print(str(np.round(p_value, 4)))
    plt.legend()
    
# limiting the analysis for groups that have AT MOST 2 mutants
def mutants_in_rc_subcohort(iterations = 9999):
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)

    # extracting relevant information for permutation test
    group_list = [g for g in np.arange(1, 18)  # list of groups for which a sRC was identified and there were at most 2 mutants in them
                  if np.sum(metadata_df.loc[metadata_df["Group_ID"] == g, "mutant"]) < 3 
                  and np.sum(metadata_df.loc[metadata_df["Group_ID"] == g, "RC"]) > 0] 
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    mutants_number = [np.sum(metadata_df.loc[metadata_df["Group_ID"] == g, "mutant"]) for g in group_list] # & (metadata_df["histology"] != "weak")
    observed = np.sum(metadata_df["mutant"] & metadata_df["RC"] & metadata_df["Group_ID"].isin(group_list)) # number of mutants observed in the sRC
    
    # permutation test
    number_of_hits = []
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for _ in tqdm(range(iterations)):
        hits = 0
        for m, r in zip(mutants_number, rc_sizes): # iterating through all groups
            np.random.shuffle(shuffled_arr)
            selected = shuffled_arr[:r]
            hits += np.sum([idx in selected for idx in range(1, m + 1)])
        number_of_hits.append(hits)
        
    fig = plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(15),  density = True, align = 'left', label = "Expected by random chance")
    p_value = np.mean(np.array(number_of_hits) <= observed)
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.75*max(h[0])), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n groups with n <= 2 mutants \n k = 3")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    print(np.round(np.cumsum(h[0]), 4)[observed])
    plt.show()
    
    
# littermates analysis   
"""
Extracting littermates indices, that will be shuffled to compute the permutation test
"""
littermates_path = "..\\data\\littermates_info.xlsx"
litter_cohort1 = pd.read_excel(littermates_path, sheet_name='cohort 1') # groups 1 to 10
shuffled_arr1 = litter_cohort1.loc[litter_cohort1["group 1"].notna(), "litter"].values
shuffled_arr2 = litter_cohort1.loc[litter_cohort1["group 2"].notna(), "litter"].values
shuffled_arr3 = litter_cohort1.loc[litter_cohort1["group 3"].notna(), "litter"].values
shuffled_arr4 = litter_cohort1.loc[litter_cohort1["group 4"].notna(), "litter"].values
shuffled_arr5 = litter_cohort1.loc[litter_cohort1["group 5"].notna(), "litter"].values
shuffled_arr6 = litter_cohort1.loc[litter_cohort1["group 6"].notna(), "litter"].values
shuffled_arr7 = litter_cohort1.loc[litter_cohort1["group 7"].notna(), "litter"].values
shuffled_arr8 = litter_cohort1.loc[litter_cohort1["group 8"].notna(), "litter"].values
shuffled_arr10 = litter_cohort1.loc[litter_cohort1["group 10"].notna(), "litter"].values
littermates_path = "..\\data\\littermates_info.xlsx"
litter_cohort2 = pd.read_excel(littermates_path, sheet_name='cohort 2') # groups 11 to 17
shuffled_arr11 = litter_cohort2.loc[litter_cohort2["group 11"].notna(), "litter"].values
shuffled_arr12 = litter_cohort2.loc[litter_cohort2["group 12"].notna(), "litter"].values
shuffled_arr15 = litter_cohort2.loc[litter_cohort2["group 15"].notna(), "litter"].values
shuffled_arr17 = litter_cohort2.loc[litter_cohort2["group 17"].notna(), "litter"].values

def littermates_in_rc_mnn2_k2(iterations = 10000):
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 2 and k = 2.
    """
    # Extracting sRC sizes
    metadata_path = "..\\data\\meta_data_k2.csv"
    metadata_df = pd.read_csv(metadata_path)
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    
    number_of_hits = []
    for i in tqdm(range(iterations)):
        hits = 0
        for i, r in zip(group_list, rc_sizes):
            current_arr = globals().get(f"shuffled_arr{i}")
            np.random.shuffle(current_arr)
            if len(np.unique(current_arr[:r])) < r: # if unique numbers are less than rc size, it means littermates are in sRC
                hits += 1
        number_of_hits.append(hits)
   
    plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(2, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) >= 2)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()

def littermates_in_rc_mnn3_k3(iterations = 10000):
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 3 and k = 3.
    """
    # Extracting sRC sizes
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    
    number_of_hits = []
    for i in tqdm(range(iterations)):
        hits = 0
        for i, r in zip(group_list, rc_sizes):
            current_arr = globals().get(f"shuffled_arr{i}")
            np.random.shuffle(current_arr)
            if len(np.unique(current_arr[:r])) < r: # if unique numbers are less than rc size, it means littermates are in sRC
                hits += 1
        number_of_hits.append(hits)
        
    plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(3, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) >= 3)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()
    
def littermates_in_rc_mnn4_k4(iterations = 10000):
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 4 and k = 4.
    """
    
    # Extracting sRC sizes
    metadata_path = "..\\data\\meta_data_k4.csv"
    metadata_df = pd.read_csv(metadata_path)
    group_list = [g for g in np.arange(18) if np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) > 0] # list of groups for which a sRC was identified
    rc_sizes = [np.sum((metadata_df["Group_ID"] == g) & metadata_df["RC"]) for g in group_list] # size of the sRC in each group
    
    number_of_hits = []
    for i in tqdm(range(iterations)):
        hits = 0
        for i, r in zip(group_list, rc_sizes):
            current_arr = globals().get(f"shuffled_arr{i}")
            np.random.shuffle(current_arr)
            if len(np.unique(current_arr[:r])) < r: # if unique numbers are less than rc size, it means littermates are in sRC
                hits += 1
        number_of_hits.append(hits)
        
    plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(4, color = 'red', label = "Experimentally observed")
    p_value = np.mean(np.array(number_of_hits) >= 4)
    plt.annotate("p-value = "+str(np.round(p_value, 4)), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()

# precedence effect analyis, i.e. what are the chance to be in sRC if animal was previously in
def reshuffled_rc_mnn2_k2(iterations = 10000, cumulative = False):
    """
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data_k2.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(iterations)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
             
    plt.figure()
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(14), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(14), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(7, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[7], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.xlim([0, 14])
    plt.legend(loc='upper left')
    plt.show()        
    
def reshuffled_rc_mnn3_k3(iterations = 10000, cumulative = False):
    """
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """ 
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(iterations)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
        
    plt.figure()
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(15), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[7], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.axvline(7, color = 'red', label = "Experimentally observed")
    plt.xlim([0, 15])
    plt.legend(loc='upper left')
    plt.show()    
    
def reshuffled_rc_mnn4_k4(iterations = 10000, cumulative = False):
    """ 
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data_k4.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(iterations)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
    
    plt.figure()
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(17), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(17), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[8], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.axvline(8, color = 'red', label = "Experimentally observed")
    plt.xlim([0, 16])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()    


# Mutants excluded from sRC
mutants_in_rc_mnn2_k2()
mutants_in_rc_mnn3_k3()
mutants_in_rc_mnn4_k4()
# mutants_in_rc_subcohort() # significance of mutants excluded from sRC, for groups with at most 2 mutants

# Absence of littermate effect
# littermates_in_rc_mnn2_k2()    
# littermates_in_rc_mnn3_k3()
# littermates_in_rc_mnn4_k4()

# Dynamic emergence of sRC
# reshuffled_rc_mnn2_k2()
# reshuffled_rc_mnn3_k3()
# reshuffled_rc_mnn4_k4()

# chasing_towards()
