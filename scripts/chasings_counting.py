import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
from weighted_rc import weighted_rich_club

sys.path.append('..\\src\\')
from read_graph import read_graph

datapath = "..\\data\\chasing\\single\\"

def histogram_chasings(graph):
    tot10 = [] # total chasing from RC for thres 10%
    tot30 = [] # total chasing from RC for thres 30%
    tot50 = [] # chasing from RC for thres 50%
    
    arr10 = [] # total chasing from RC for thres 10%
    arr30 = [] # total chasing from RC for thres 30%
    arr50 = [] # chasing from RC for thres 50%
    
    
    data_10 = read_graph(datapath+graph, percentage_threshold = 10)
    data_30 = read_graph(datapath+graph, percentage_threshold = 30)
    data_50 = read_graph(datapath+graph, percentage_threshold = 50)
    
    tot10.append(np.sum(data_10[:])+np.sum(data_10[:]))
    tot30.append(np.sum(data_10[:])+np.sum(data_10[:]))
    tot50.append(np.sum(data_10[:])+np.sum(data_10[:]))
    
    arr10.append(np.sum(data_10[0,:])+np.sum(data_10[6,:]))
    arr30.append(np.sum(data_10[0,:])+np.sum(data_10[6,:]))
    arr50.append(np.sum(data_10[0,:])+np.sum(data_10[6,:]))
    
    tot10, tot30, tot50 = np.array(tot10), np.array(tot30), np.araray(tot50)
    ar10, arr30, arr50 = np.array(arr10), np.array(arr30), np.araray(arr50)


    labels = ['G1']
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
    ax.hlines(25, x[0] - width, x[-1] + width, linestyle = "--", color = "k", alpha = 0.5)  # 25 because there are 2 to 3 members of stable rich club
    ax.annotate('Random chance', xy=(4.5, 25), xytext=(4 + width/10, 70), ha = 'left',      # 25 because there are 2 to 3 members of stable rich club
            arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 5), textcoords='data', xycoords='data')
    plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
histogram_chasings("G1")