import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from weighted_rc import weighted_rich_club
import sys
sys.path.append('..\\src\\')
from read_graph import read_graph


# datapath = "..\\data\\chasing\\single\\"
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
    all_rc = [[4, 6], [2, 6], [2, 6], [0, 5], [2, 4], [7, 9], [0, 5], [2, 3], [2, 3]]
    labels = ["G1", "G2", "G3", "G4","G5", "G6", "G7", "G8","G10"]
    random_chances = [20, 20, 20, 20, 20, 20, 20, 20, 20]
    
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
    ax.set_xlabel('First cohort')
    ax.set_title('Mutants chasings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7,8], [0,0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.ylim([0, 40])
    plt.show()
    
def histogram_approaches_mutants():
    all_rc = [[4, 6], [2, 6], [6], [0, 5], [3, 5], [7, 9], [0, 5], [2, 3], [2, 3]]
    labels = ["G1", "G2", "G3", "G4","G5", "G6", "G7", "G8","G10"]
    random_chances = [20, 20, 20, 20, 20, 20, 20, 20, 20]
    
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
    ax.set_xlabel('First cohort')
    ax.set_title('Mutants approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7,8], [0,0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.ylim([0, 40])
    plt.show()
            
if __name__ == "__main__":
    # y = "Chasing rank"
    # mutants_tube_rank(y, dataset = "both", hue = True)
    # mutants_tube_rank(y, dataset = "both", hue = False)
    # histogram_chasing_mutants()
    histogram_approaches_mutants()

