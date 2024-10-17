# this file intends to show that the RC members are heterogeneous in their chasing, and that they like to chase other RC members.
import matplotlib.pyplot as plt
import numpy as np
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


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
    
def chasing_towards_piechart(normed = False, filter_big_chasers = False):
    """
    if both, animals are ranked by outgoing+ingoing chasings. else, they are ranked by outgoing chasings.
    """
    datapath = "..\\data\\chasing\\single\\"
    # plots the rank of RC members, ranking each mice of each group by the total number of outgoing chases it 
    # performed
    all_rc = [[0,6], [3, 8, 9], [3, 4, 8], [5, 6], [0, 1], [3, 4, 6], [5, 7], [7, 8], [5, 8], [0, 2], [2, 8, 9]]
    labels = ["G1", "G2", "G3","G5", "G6", "G7", "G8","G10", "G11", "G12", "G15"]
    
    rc_chased = [] # total chasing from RC for thres 10%

    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"_single_chasing.csv"])[0]
        for member in all_rc[idx]:
            members = np.array(all_rc[idx])
            avg_total_chasing = np.quantile(np.sum(data, axis = 1), 0.8)
            # avg_total_chasing = np.mean(np.sum(data, axis = 1))

            if not filter_big_chasers:
                if (np.any(np.argmax(data[member, :]) == members)) and (np.argmax(data[member, :]) != member):
                    rc_chased.append(1)
                else:
                    rc_chased.append(0)
            elif np.sum(data[member, :]) > avg_total_chasing:
                if (np.any(np.argmax(data[member, :]) == members)) and (np.argmax(data[member, :]) != member):
                    rc_chased.append(1)
                else:
                    rc_chased.append(0)
                

    total_rc_chasings = np.array(rc_chased)
    num_chasings = len(total_rc_chasings)
    in_frac = np.sum(total_rc_chasings)

    out_frac = num_chasings - np.sum(total_rc_chasings)
    in_frac = 100*(in_frac/num_chasings)
    out_frac = 100/(out_frac/num_chasings)
    if normed:
        in_frac = in_frac/2.5
        out_frac = out_frac/7.5
        in_frac = 100*(in_frac/(in_frac+out_frac))
        out_frac = 100*(out_frac/(in_frac+out_frac))
    
    labels = "Towards RC", "Outside RC"
    sizes = [in_frac, out_frac]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f\%%')
    plt.show()
    
chasing_towards_piechart(True, True)