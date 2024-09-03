import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
from weighted_rc import weighted_rich_club

sys.path.append('..\\src\\')
from read_graph import read_graph

datapath = "..\\data\\chasing matrices\\"
datapath = "..\\data\\social network matrices 1day\\"

def day_to_day_approaches(outgoing = True):
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3, 8], [2, 3, 6], [1, 6], [0, 1], [4], [3, 5, 7, 8], [7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8", "G10"]
    
    all_fractions = np.zeros(15) # day by day fractions of outgoing approaches from RC
    
    for t in range(15):
        fraction = []
        for idx, g in enumerate(labels):
            try:
                data = read_graph([datapath+g+"\\approaches_resD1_"+str(t+1)+".csv"])[0]
            except:
                print("Graph "+g+"_approaches_resD1_"+str(t+1)+" not found.")    
            if outgoing:
                approaches = np.sum(data, axis = 1)
            else:
                approaches = np.sum(data, axis = 0)
            total = np.sum(approaches)
    
            fraction.append(np.sum(approaches[all_rc[idx]])/total)
        fraction = np.array(fraction)
        all_fractions[t] = np.mean(fraction)
            
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    ax = plt.subplot(111)
    ax.set_xlabel("Day", fontsize = 12)
    if outgoing:
        ax.set_ylabel("Avrage RC outgoing approaches (\%)", fontsize = 12)
    else:
        ax.set_ylabel("Avrage RC incoming approaches (\%)", fontsize = 12)
    # plt.xticks(np.arange(1, 16))

    ax.grid(alpha = 0.3)
    ax.plot(np.arange(1, 16), all_fractions*100, "-o", color = 'black')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
def day_to_day_chasing(outgoing = True):
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3, 8], [2, 3, 6], [1, 6], [0, 1], [4], [3, 5, 7, 8], [7, 8]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8", "G10"]
    
    all_fractions = np.zeros(15) # day by day fractions of outgoing approaches from RC
    
    for t in range(15):
        fraction = []
        for idx, g in enumerate(labels):
            try:
                data = read_graph([datapath+g+"\\chasing_resD1_"+str(t+1)+".csv"])[0]
            except:
                print("Graph "+g+"_chasing_resD1_"+str(t+1)+" not found.")    
            if outgoing:
                approaches = np.sum(data, axis = 1)
            else:
                approaches = np.sum(data, axis = 0)
            total = np.sum(approaches)
    
            fraction.append(np.sum(approaches[all_rc[idx]])/total)
        fraction = np.array(fraction)
        all_fractions[t] = np.mean(fraction)
            
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    ax = plt.subplot(111)
    ax.set_xlabel("Day", fontsize = 12)
    if outgoing:
        ax.set_ylabel("Avrage RC outgoing chasings (\%)", fontsize = 12)
    else:
        ax.set_ylabel("Avrage RC incoming chasings (\%)", fontsize = 12)
    # plt.xticks(np.arange(1, 16))

    ax.grid(alpha = 0.2)
    ax.plot(np.arange(1, 16), all_fractions*100, "-o", color = 'black')
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
if __name__ == "__main__":
    day_to_day_approaches()
    # day_to_day_chasing(True)