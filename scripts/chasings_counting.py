import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
from weighted_rc import weighted_rich_club

sys.path.append('..\\src\\')
from read_graph import read_graph

# datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"


def histogram_chasings(graph, rc_idx):
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    # plt.rcParams["figure.autolayout"] = True
    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "Helvetica"
    # })
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    data_10 = read_graph([datapath+graph], percentage_threshold = 10)[0]
    data_30 = read_graph([datapath+graph], percentage_threshold = 30)[0]
    data_50 = read_graph([datapath+graph], percentage_threshold = 50)[0]
    
    tot10.append(np.sum(data_10[:] > 0.5))
    tot30.append(np.sum(data_30[:]> 0.5))
    tot50.append(np.sum(data_50[:]> 0.5))
    
    sum10, sum30, sum50 = 0,0,0
    for rc in rc_idx:
        sum10 += np.sum(data_10[rc,:]> 0.5)
        sum30 += np.sum(data_30[rc,:]> 0.5)
        sum50 += np.sum(data_50[rc,:]> 0.5)
        
    arr10.append(sum10)
    arr30.append(sum30)
    arr50.append(sum50)
    
    tot10, tot30, tot50 = np.array(tot10), np.array(tot30), np.array(tot50)
    ar10, arr30, arr50 = np.array(arr10), np.array(arr30), np.array(arr50)


    labels = ['G1']
    x = np.arange(len(labels))*1
    width = 0.35
    fig, ax = plt.subplots()
    
    # colors = ["#ffeda0", "#feb24c", "#f03b20"] # reds
    colors = ["#edf8b1", "#7fcdbb", "#2c7fb8"] # veridis
    # colors = ['#f0f0f0','#bdbdbd','#636363'] # grays
    # colors = ['#e0ecf4','#9ebcda','#8856a7'] # purples
    # colors = ['#a1dab4','#41b6c4','#225ea8'] # veridis dark
    
    rects1 = ax.bar(x - width, 100*(arr10/tot10), 2*width/3, label="10%", align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*(arr30/tot30), 2*width/3, label='30%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*(arr50/tot50), 2*width/3, label='50%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total chasings fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members chasings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.hlines(25, x[0] - width, x[-1] + width, linestyle = "--", color = "k", alpha = 0.5)  # 25 because there are 2 to 3 members of stable rich club
    ax.annotate('Random chance', xy=(4.5, 25), xytext=(4 + width/10, 70), ha = 'left',      # 25 because there are 2 to 3 members of stable rich club
            arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 5), textcoords='data', xycoords='data')
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
def persistence_chasings(graph, rc_idx):
    
    arr = []
    total = []
    for i in range(1, 101, 1):
        data = read_graph([datapath+graph], percentage_threshold = i)[0]
        total.append(np.sum(data[:] > 0.5))
        sum_rc = 0
        for rc in rc_idx:
            sum_rc += np.sum(data[rc,:]> 0.5)
        arr.append(sum_rc)
    arr = np.array(arr)
    total = np.array(total)

    fig, ax = plt.subplots()
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Total chasings fraction (%)")
    ax.set_xlabel("Threshold (%)")
    ax.set_title('Rich-club members chasings')
    plt.plot(range(1, 101, 1), arr/total, c = 'k')
    # ax.hlines(25, x[0] - width, x[-1] + width, linestyle = "--", color = "k", alpha = 0.5)  # 25 because there are 2 to 3 members of stable rich club
    # ax.annotate('Random chance', xy=(4.5, 25), xytext=(4 + width/10, 70), ha = 'left',      # 25 because there are 2 to 3 members of stable rich club
    #         arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 5), textcoords='data', xycoords='data')
    plt.show()
    
def histogram_chasing_new():
    all_rc = [[0,6], [3,8,9], [3,4, 8],[5,6],[0, 1], [3,4,6], [3, 5, 7, 8], [6, 7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10"]
    random_chances = [20, 33, 33, 20, 20, 33, 33, 20]
    
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    for idx, g in enumerate(labels):
        data_10 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 10)[0]
        data_30 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 30)[0]
        data_50 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 50)[0]
        
        tot10.append(np.sum(np.sum(data_10, axis = 1)))
        tot30.append(np.sum(np.sum(data_30, axis = 1)))
        tot50.append(np.sum(np.sum(data_50, axis = 1)))
        
        sum10, sum30, sum50 = 0, 0, 0
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
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='10\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='30\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='50\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total chasings fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members chasings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7], [0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
def histogram_shared_passages():
    all_rc = [[0,6], [3,8,9], [3,4, 8], [5,6], [0, 1], [3,4,6], [3, 5, 7, 8], [6, 7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10"]
    random_chances = [20, 33, 33, 20, 20, 33, 33, 20]
    
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    for idx, g in enumerate(labels):
        data_10 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 10)[0]
        data_30 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 30)[0]
        data_50 = read_graph([datapath+g+"_single_chasing.csv"], percentage_threshold = 50)[0]
        
        tot10.append(np.sum(np.sum(data_10, axis = 0)) + np.sum(np.sum(data_10, axis = 1)))
        tot30.append(np.sum(np.sum(data_30, axis = 0)) + np.sum(np.sum(data_30, axis = 1)))
        tot50.append(np.sum(np.sum(data_50, axis = 0)) + np.sum(np.sum(data_50, axis = 1)))
        
        sum10, sum30, sum50 = 0, 0, 0
        for rc in all_rc[idx]:
            sum10 += np.sum(data_10[rc,:]) + np.sum(data_10[:, rc])
            sum30 += np.sum(data_30[rc,:]) + np.sum(data_30[:, rc]) 
            sum50 += np.sum(data_50[rc,:]) + np.sum(data_50[:, rc])
            
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
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='10\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='30\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='50\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total shared passages fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members shared passages')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7], [0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
    
def histogram_approaches():
    all_rc =  [[0,6], [3, 8, 9], [2, 3, 6], [1, 6], [0, 1], [3, 4, 6], [3, 5, 7, 8], [7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8", "G10"]
    random_chances = [20, 33, 33, 20, 20, 33, 33, 20]
    
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%

    for idx, g in enumerate(labels):
        data_10 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 10)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 10)[0]
        data_30 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 30)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 30)[0]
        data_50 = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 50)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 50)[0]
        
        tot10.append(np.sum(np.sum(data_10, axis = 1)))
        tot30.append(np.sum(np.sum(data_30, axis = 1)))
        tot50.append(np.sum(np.sum(data_50, axis = 1)))
        
        sum10, sum30, sum50 = 0, 0, 0
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
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='10\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='30\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='50\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total approaches fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.vlines([0,1,2,3,4,5,6,7], [0,0,0,0,0,0,0,0], random_chances, color = "k", label = "Random chance")  # 25 because there are 2 to 3 members of stable rich club
    # plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
def rank_by_outgoing_chasings():
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3,8,9], [3,4, 8],[5,6],[0, 1], [3,4,6], [5, 7, 8], [6, 7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10"]
    
    all_ranks = [] # total chasing from RC for thres 10%

    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        # data[data <= 1.00e-02] = 0
        rank = np.argsort(-np.sum(data, axis = 1))
        all_ranks.extend(rank[all_rc[idx]])
    
    all_ranks = np.array(all_ranks)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    plt.xticks([0, 1,2,3,4,5,6,7,8,9], labels = [1,2,3,4,5,6,7,8,9,10])#)["10", "9", "8","7", "6","5", "4", "3", "2", "1"])
    plt.yticks([0,1,2,3,4,5])
    plt.xlim([-1,10])
    plt.xlabel("Rank of RC members by total number of outgoing chasings", fontsize = 12)
    plt.ylabel("Count", fontsize = 12)

    plt.hist(all_ranks, bins = np.arange(10), align = "left", rwidth = 0.95, color = 'gray')
    plt.show()
    
def rank_by_outgoing_approaches(outgoing = True):
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3, 8], [2, 3, 6], [1, 6], [0, 1], [4], [3, 5, 7, 8], [7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8", "G10"]
    
    all_ranks = [] # total chasing from RC for thres 10%

    for idx, g in enumerate(labels):
        data1 = read_graph([datapath+g+"\\approaches_resD7_1.csv"])[0]
        data2 = read_graph([datapath+g+"\\approaches_resD7_2.csv"])[0]
        data = data1 + data2
        # data[data <= 1.00e-02] = 0
        if outgoing:
            rank = np.argsort(np.sum(-data, axis = 1))
        else:
            rank = np.argsort(np.sum(-data, axis = 0))

        all_ranks.extend(rank[all_rc[idx]])
    
    all_ranks = np.array(all_ranks)
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    plt.xticks([0,1,2,3,4,5,6,7,8,9], labels = [1,2,3,4,5,6,7,8,9,10])#["10", "9", "8","7", "6","5", "4", "3", "2", "1"])
    # plt.yticks([0,1,2,3,4,5])
    # plt.xlim([8.9,-1])
    if outgoing:
        plt.xlabel("Rank of RC members by total number of outgoing approaches", fontsize = 12)
    else:
        plt.xlabel("Rank of RC members by total number of incoming approaches", fontsize = 12)

    plt.ylabel("Count", fontsize = 12)

    plt.hist(all_ranks, bins = np.arange(11), align = "left", rwidth = 0.95, color = 'gray')
    plt.show()
    

if __name__ == "__main__":
    # histogram_chasings("G1_single_chasing.csv", [0, 6])
    # persistence_chasings("G1_single_chasing.csv", [0, 6])
    # persistence_chasings("G3_single_chasing.csv", [3,4,8])
    # persistence_chasings("G5_single_chasing.csv", [5, 6])!
    # persistence_chasings("G7_single_chasing.csv", [3, 4, 6])
    # histogram_chasing_new()
    # histogram_shared_passages()
    # rank_by_outgoing_chasings()
    rank_by_outgoing_approaches()
    
    # histogram_approaches()
