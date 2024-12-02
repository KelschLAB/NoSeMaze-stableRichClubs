import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from weighted_rc import weighted_rich_club
import sys
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..\\src\\')
sys.path.append(os.getcwd())
from read_graph import read_graph
from HierarchiaPy import Hierarchia

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"


## Validation cohort dataframe
validation_AON = np.array([2,3,2,1,1,1,2,2,3,3,1,2])
validation_AON2 = np.array([2,3,2,1,1,1,2,1,2,2,1,1])
brain_area = ["AON", "AON","AON","AON","AON","AON","AON","AON","AON","AON","AON","AON",
              "AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2"]
validation_tr = np.array([6, 9, 5, 8, 4, 5, 7, 4, 7.5, 1, 7.5, 7.5]) #tube rank
validation_cr = np.array([9, 8.5, 2, 6, 7, 7, 3.5, 2, 9.5, 7, 2, 2]) # chasing rank
validation_cin = np.array([8, 8, 8, 8,8,8,8.5, 7, 8, 7, 8, 6.5]) # chasing in
validation_cout = np.array([5, 7, 9, 8, 8, 8, 9, 9, 2.5, 9, 8.5, 8.5]) # chasing out
# val_dict = {'tr':validation_tr, 'cr': validation_cr, 'cin': validation_cin, 'cout':validation_cout, 'aon':validation_AON, 'aon2':validation_AON2}
val_dict = {'Tube rank':np.concatenate((validation_tr,validation_tr)), 'Chasing rank': np.concatenate((validation_cr,validation_cr)),
                                'cin': np.concatenate((validation_cin,validation_cin)), 'Chasing out':np.concatenate((validation_cout,validation_cout)),
                                'Gene deletion':np.concatenate((validation_AON,validation_AON2)), 'Brain area': brain_area}
validation_df = pd.DataFrame(data=val_dict)

## First cohort dataframe
first_brain_area = ["AON", "AON","AON","AON","AON","AON","AON",
              "AON2","AON2","AON2","AON2","AON2","AON2","AON2"]
first_AON = np.array([0,0,1,2,3,3,3])
first_AON2 = np.array([0,0,0,1,2,2,3])
first_time = np.array([3114, 1950, 2480, 2337, 3228, 5224, 2507]) # time in arena
first_cr = np.array([3, 4.5, 8.5, 9, 10, 7, 2.5]) # chasing rank
first_tr = np.array([5.5, 8, 6, 8, 6, 3.5, 9])
first_dict = {'Tube rank':np.concatenate((first_tr, first_tr)), 'Chasing rank': np.concatenate((first_cr, first_cr)),
                            'Time in arena':np.concatenate((first_time, first_time)), 'Gene deletion':np.concatenate((first_AON, first_AON2)), 'Brain area': first_brain_area}
first_df = pd.DataFrame(data=first_dict)

## Both cohort combined
both_brain_area = first_brain_area
both_brain_area.extend(brain_area) #extend is in-place operation
both_dict = {'Tube rank':np.concatenate((first_tr, first_tr, validation_tr, validation_tr)), 
            'Chasing rank': np.concatenate((first_cr, first_cr, validation_cr, validation_cr)),
            'Gene deletion':np.concatenate((first_AON, first_AON2, validation_AON, validation_AON2)), 
            'Brain area': both_brain_area}
both_df = pd.DataFrame(data=both_dict)

def mutants_in_club_piechart():
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots(1, 1)
    plt.style.use('seaborn')
    sizes = [25/26, 1/26]
    labels = ["WT", "Mutant"]
    ax.pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    plt.tight_layout()
    plt.show()

def mutants_tube_rank(yaxis = "Chasing rank", dataset = "both", hue = True):
    sns.set(font_scale = 1.5)
    if dataset == "both":
        if hue:
            sns.catplot(data=both_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=both_df, x="Gene deletion", y=yaxis, kind="bar", color = "gray")
            plt.title("Average effect over both brain areas")
            plt.tight_layout()

    elif dataset == "validation":
        if hue:
            sns.catplot(data=validation_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=validation_df, x="Gene deletion", y=yaxis, kind = 'bar', color = "gray")
            plt.title("Average over AON & AON2")
            plt.tight_layout()

    elif dataset == "first":
        if hue:
            sns.catplot(data=first_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=first_df, x="Gene deletion", y=yaxis, kind = 'bar', color = 'gray')
            
def histogram_chasing_mutants():
    # all_rc = [[4, 6], [2, 6], [2, 6], [0, 5], [2, 4], [7, 9], [0, 5], [2, 3], [2, 3]]
    all_rc = [[6], [2], [6], [5], [2, 4], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    random_chances = [10, 10, 10, 20, 20, 10, 20, 10, 10, 33, 20, 33, 20, 33, 40]
    
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    for idx, g in enumerate(labels):
        data_10 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        data_30 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 5)[0]
        data_50 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 10)[0]
        
        tot10.append(np.sum(np.sum(data_10, axis = 1)))
        tot30.append(np.sum(np.sum(data_30, axis = 1)))
        tot50.append(np.sum(np.sum(data_50, axis = 1)))
        
        sum10, sum30, sum50 = 0,0,0
        for rc in all_rc[idx]:
            sum10 += np.sum(data_10[rc,:])
            sum30 += np.sum(data_30[rc,:])
            sum50 += np.sum(data_50[rc,:])
            
        arr10.append(sum10)
        arr30.append(sum30)
        arr50.append(sum50)
    
    tot10, tot30, tot50 = np.array(tot10), np.array(tot30), np.array(tot50)
    arr10, arr30, arr50 = np.array(arr10), np.array(arr30), np.array(arr50)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    x = np.arange(len(labels))*1
    width = 0.35
    fig, ax = plt.subplots()
    
    # colors = ["#ffeda0", "#feb24c", "#f03b20"] # reds
    colors = ["#edf8b1", "#7fcdbb", "#2c7fb8"] # veridis
    # colors = ['#f0f0f0','#bdbdbd','#636363'] # grays
    # colors = ['#e0ecf4','#9ebcda','#8856a7'] # purples
    # colors = ['#a1dab4','#41b6c4','#225ea8'] # veridis dark
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='0\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='5\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='10\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total chasings fraction (\%)')
    ax.set_xlabel('Both cohorts')
    ax.set_title('Mutants chasings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.ylim([0, 50])
    plt.show()
    
def histogram_approaches_mutants():
    # all_rc = [[4, 6], [2, 6], [4], [0, 5], [3, 5], [7, 9], [0, 5], [2, 3], [2, 3], [0,2,3], [5,7], [0,2,3], [3, 9]] 
    all_rc = [[6], [2], [4], [5], [3, 5], [7], [0, 5], [3], [2], [0,2,3], [5,7], [2,3,9], [3,9], [0,2,3], [2,3,8,9]] #took out mutants with weak histology

    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    random_chances = [10, 10, 10, 20, 20, 10, 20, 10, 10, 33, 20, 33, 20, 33, 40]
    
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    for idx, g in enumerate(labels):
        data_10 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 5)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 5)[0]
        data_30 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 10)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 10)[0]
        data_50 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 20)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 20)[0]
        
        tot10.append(np.sum(np.sum(data_10, axis = 1)))
        tot30.append(np.sum(np.sum(data_30, axis = 1)))
        tot50.append(np.sum(np.sum(data_50, axis = 1)))
        
        sum10, sum30, sum50 = 0,0,0
        for rc in all_rc[idx]:
            sum10 += np.sum(data_10[rc,:])
            sum30 += np.sum(data_30[rc,:])
            sum50 += np.sum(data_50[rc,:])
            
        arr10.append(sum10)
        arr30.append(sum30)
        arr50.append(sum50)
    
    tot10, tot30, tot50 = np.array(tot10), np.array(tot30), np.array(tot50)
    arr10, arr30, arr50 = np.array(arr10), np.array(arr30), np.array(arr50)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    x = np.arange(len(labels))*1
    width = 0.35
    fig, ax = plt.subplots()
    
    # colors = ["#ffeda0", "#feb24c", "#f03b20"] # reds
    colors = ["#edf8b1", "#7fcdbb", "#2c7fb8"] # veridis
    # colors = ['#f0f0f0','#bdbdbd','#636363'] # grays
    # colors = ['#e0ecf4','#9ebcda','#8856a7'] # purples
    # colors = ['#a1dab4','#41b6c4','#225ea8'] # veridis dark
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='5\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='10\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='20\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total approaches fraction (\%)')
    ax.set_xlabel('Both cohort')
    ax.set_title('Mutants approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.ylim([0, 70])
    plt.show()

def approach_order_mutants(out = True, both = False):
    """ Histogram of the approach order for mutants (i.e., if we rank them by according to how much total approaches they perform.)
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

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

def approachRank_mutants():
    """ Histogram of the approach david score for mutants.
    """
    path_cohort1 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\reduced_data.xlsx"
    path_cohort2 = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\validation_cohort.xlsx"
    approach_dir = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\data\\averaged\\"

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
    plt.title(r"Appraoch rank of mutants", fontsize = 18)
    plt.show()

            
if __name__ == "__main__":
    # y = "Chasing rank"
    # mutants_tube_rank(y, dataset = "both", hue = True)
    # mutants_tube_rank(y, dataset = "both", hue = False)
    # histogram_chasing_mutants()
    # histogram_approaches_mutants()
    # approach_order_mutants(both = True)
    # approachRank_mutants()
    mutants_in_club_piechart()

