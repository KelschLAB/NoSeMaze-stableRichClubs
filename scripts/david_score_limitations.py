from HierarchiaPy import Hierarchia
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import pandas as pd
import os
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from weighted_rc import weighted_rich_club
sys.path.append('..\\src\\')
from read_graph import read_graph
from collections import Counter
from scipy import stats
from copy import copy

path = "C:\\Users\\Agarwal Lab\\Corentin\\Python\\NoSeMaze\\data\\chasing\\single\\G1_single_chasing.csv"
raw_data = read_graph([path])[0]
raw_data[raw_data < 0.5] = 0
names = np.loadtxt(path, delimiter=",", dtype=str)[0, :][1:]

hier_mat = Hierarchia(raw_data, names)
davids_scores = hier_mat.davids_score("Dij")
print("__________________________")
print(davids_scores)

no_rc_data = copy(raw_data)
no_rc_data[0, :] = np.random.randint(0, 15, 10) 
no_rc_data[:, 0] = np.random.randint(0, 15, 10) 
hier_mat = Hierarchia(no_rc_data, names)
davids_scores = hier_mat.davids_score()
print("__________________________")
print(davids_scores)

ds_rank = 0
for i in range(100):
    exagerated_rc_data = copy(raw_data)
    exagerated_rc_data[0, :] = np.random.randint(0, 15, 10) 
    exagerated_rc_data[:, 0] = np.random.randint(0, 15, 10) 
    exagerated_rc_data[0, 6] = 500
    hier_mat = Hierarchia(exagerated_rc_data, names)
    davids_scores = hier_mat.davids_score()
    rank = np.argsort(list(davids_scores.values()))#[::-1]
    np_names = np.array(list(davids_scores.keys()))[rank]
    ds_rank += np.where(np_names == '0007AC2B02')[0][0]
print(ds_rank/100)

# exagerated_rc_data[:, 0] = 0
