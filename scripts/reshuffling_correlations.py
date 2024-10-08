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
    
def plot_reshuffled_outchasing_corr():
    
    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 1)
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k')
    plt.xlabel("Total outgoing chasings, round 1", fontsize = 17)
    plt.ylabel("Total outgoing chasings, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_inchasing_corr():
    
    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 0)
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k')
    plt.xlabel("Total ingoing chasings, round 1", fontsize = 17)
    plt.ylabel("Total ingoing chasings, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)  # 1 means linear
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_outapproach_corr():
    datapath = "..\\data\\averaged\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt(datapath+g+"\\approaches_resD7_1.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph([datapath+g+"\\approaches_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph([datapath+g+"\\approaches_resD7_2.csv"], percentage_threshold = 0)[0]
        chasings = np.sum(data, axis = 1)
        for c in range(len(chasings)):
            if names[c] not in already_seen_first:
                chasings_first.append(chasings[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(chasings[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k')
    plt.xlabel("Total outgoing approaches, round 1", fontsize = 17)
    plt.ylabel("Total outgoing approaches, round 2", fontsize = 17)
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    # plt.plot(x, slope * x + intercept, color='k', linestyle='--', label = "Pearson = "+str(pearson_corr))
    # plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_reshuffled_chasingRank_corr(both=False):
    
    datapath = "..\\data\\chasing\\single\\"
    chasings_first, chasings_second = [], []
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
    
    already_seen_first, already_seen_second = [], []
    chasings_first, chasing_second = [], []
    for idx, g in enumerate(labels):
        names = np.loadtxt("..\\data\\chasing\\single\\"+g+"_single_chasing.csv", delimiter=",", dtype=str)[0, :][1:]
        data = read_graph(["..\\data\\chasing\\single\\"+g+"_single_chasing.csv"], percentage_threshold = 0)[0]
        outchasings = np.sum(data, axis = 1)
        inchasings = np.sum(data, axis = 0)
        if both:
            rank = np.argsort(-outchasings-inchasings)
        else:
            rank = np.argsort(-outchasings)
        
        for c in range(len(rank)):
            if names[c] not in already_seen_first:
                chasings_first.append(rank[c])
                already_seen_first.append(names[c])
            elif names[c] not in already_seen_second:
                chasings_second.append(rank[c])
                already_seen_second.append(names[c])
    
    filter_names = []
    for name in already_seen_first:
        if name not in already_seen_second:
            filter_names.append(False)
        else:
            filter_names.append(True)
            
    chasings_first, chasings_second = np.array(chasings_first), np.array(chasings_second)
    plt.figure()
    plt.xlabel("Chasing order, round 1", fontsize = 17)
    plt.ylabel("Chasing order, round 2", fontsize = 17)
    if both:
        plt.title("Rank on outgoing + ingoing chasings")
    else:
        plt.title("Rank on outgoing chasings")
    slope, intercept = np.polyfit(chasings_first[filter_names], chasings_second, 1)
    x = np.arange(np.min(chasings_first), np.max(chasings_first))
    correlation_matrix = np.corrcoef(chasings_first[filter_names], chasings_second)
    pearson_corr = correlation_matrix[0, 1] 
    points = list(zip(chasings_first[filter_names], chasings_second))
    counts = Counter(points)
    sizes = [counts[(xi, yi)] * 80 for xi, yi in points]  # Scale size
    plt.scatter(chasings_first[filter_names], chasings_second, c = 'k', s = sizes, alpha = 0.5, label = "Pearson = "+str(np.round(pearson_corr, 2)))
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.show()

# plot_reshuffled_inchasing_corr()
# plot_reshuffled_outchasing_corr()
# plot_reshuffled_chasingRank_corr()
# plot_reshuffled_outapproach_corr()