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
    ax.pie(sizes, labels=labels, autopct='%1.1f\%%')
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

    labels = ["G1", "G2", "G4", "G5", "G6", "G7", "G8", "G10", "G11","G12","G13","G14", "G15", "G16", "G17"]

    for idx, g in enumerate(labels):
        data = read_graph([datapath+g+"\\approaches_resD7_1.csv"])[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"])[0]
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
    
    
# chasing_towards_piechart(True)
cumulative_approaches()
# chasing_manhanttan_plot(False)
# cumulative_chasings()

# plot normalized cumulative plots of chasings for ranked animals (by chasing), in and out
# plot chasings ranking for ingoing.