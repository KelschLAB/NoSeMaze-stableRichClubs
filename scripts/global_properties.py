import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import pandas as pd
import seaborn as sns
from weighted_rc import weighted_rich_club
from scipy import stats
sys.path.append('..\\src\\')
from read_graph import read_graph, isSymmetric, rich_club_size, k_core_size
from collections import Counter
import matplotlib
matplotlib.use("TkAgg")

def extract_graph_measures(path, k_degree = 3, mnn = 3, plot = True):
    """
    Processes graphs from nosemaze data and extract global network properties.
    if paths is a list, the final plots will compare the results of the different 
    subfolders, treating them as different experiments.
    """

    data = read_graph([path], percentage_threshold = 0)[0]

    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    edge_betweenness = g.betweenness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
    edge_betweenness = ig.rescale(edge_betweenness)

    edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
    edge_strength = ig.rescale(edge_strength)

    edge_closeness = g.closeness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
    edge_closeness = ig.rescale(edge_closeness)

    edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
    edge_hub = ig.rescale(edge_hub)

    edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
    edge_authority = ig.rescale(edge_authority)

    edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
    edge_evc = ig.rescale(edge_evc)

    edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
    edge_pagerank = ig.rescale(edge_pagerank)

    # need to cut graph to extract meaningful global properties
    data_cut = read_graph([path], mnn= mnn)[0]
    if isSymmetric(data_cut):
        g_cut = ig.Graph.Weighted_Adjacency(data_cut, mode='undirected')
    else:
        g_cut = ig.Graph.Weighted_Adjacency(data_cut, mode='directed')
        
    club_size = rich_club_size(g_cut, k_degree)
    core_size = k_core_size(g_cut, k_degree)
    
    path_length = g.average_path_length(weights = [e['weight'] for e in g.es()])
    
    # entirely remove edges that have been cut to make usage of igraph functions
    data_cut = read_graph([path], mnn= mnn)[0]
    data_cut = np.where(data_cut > 0.01, data_cut, 0)
    if isSymmetric(data_cut):
        g_cut = ig.Graph.Weighted_Adjacency(data_cut, mode='undirected')
    else:
        g_cut = ig.Graph.Weighted_Adjacency(data_cut, mode='directed')
        
    largest_clique = max([len(clique) for clique in g_cut.maximal_cliques()])
    independence_number = g_cut.independence_number()
    
    results = {"edge_betweenness": edge_betweenness, "edge_strength": edge_strength,
               "edge_closeness": edge_closeness, "edge_hub": edge_hub,
               "edge_authority": edge_authority, "edge_evc": edge_evc,
               "edge_pagerank": edge_pagerank, "club_size": club_size,
               "core_size": core_size, "path_length": path_length,
               "largest_clique":largest_clique, "independence_number":independence_number}
   
    return results

def batch_process_data(synthetic_data_path = None):
    """
    Processes all interactions graphs from nosemaze data and extract global network properties.
    Also processes data from additional folders, to be compared against experimental data.
    This is specified in 'synthetic_data_path'.
    """
    cohorts = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G10'] 
    
    # Initialize lists to store the results
    origin = [] # to store where data came from
    edge_betweenness_list = []
    edge_strength_list = []
    edge_closeness_list = []
    edge_hub_list = []
    edge_authority_list = []
    edge_evc_list = []
    edge_pagerank_list = []
    club_size_list = []
    core_size_list = []
    path_length_list = []
    largest_clique_list = []
    indepedence_list = []
    

    for group in cohorts:
        file_path = "..\\data\\averaged\\" + group + "\\interactions_resD7_1.csv"
        print(f"Processing {file_path}...")
        results = extract_graph_measures(file_path)
        # local (node-based) measurments
        edge_betweenness_list.extend(results["edge_betweenness"])
        edge_strength_list.extend(results["edge_strength"])
        edge_closeness_list.extend(results["edge_closeness"])
        edge_hub_list.extend(results["edge_hub"])
        edge_authority_list.extend(results["edge_authority"])
        edge_evc_list.extend(results["edge_evc"])
        edge_pagerank_list.extend(results["edge_pagerank"])
        # global (graph-wide) variables and measurements
        origin.extend(["NoseMaze"]*len(results["edge_betweenness"]))
        club_size_list.extend([results["club_size"]])
        club_size_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
        core_size_list.extend([results["core_size"]])
        core_size_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
        path_length_list.extend([results["path_length"]])
        path_length_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
        largest_clique_list.extend([results["largest_clique"]])
        largest_clique_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
        indepedence_list.extend([results["independence_number"]])
        indepedence_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
    
    for folder in os.listdir(synthetic_data_path):
        for run in os.listdir(os.path.join(synthetic_data_path, folder)):
            file_path = os.path.join(synthetic_data_path, folder, run, "gridworld_interactions.csv")
            print(f"Processing {file_path}...")
            results = extract_graph_measures(file_path)
            # local (node-based) measurments
            edge_betweenness_list.extend(results["edge_betweenness"])
            edge_strength_list.extend(results["edge_strength"])
            edge_closeness_list.extend(results["edge_closeness"])
            edge_hub_list.extend(results["edge_hub"])
            edge_authority_list.extend(results["edge_authority"])
            edge_evc_list.extend(results["edge_evc"])
            edge_pagerank_list.extend(results["edge_pagerank"])
            # global (graph-wide) variables and measurements
            origin.extend([folder]*len(results["edge_betweenness"]))
            club_size_list.extend([results["club_size"]])
            club_size_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
            core_size_list.extend([results["core_size"]])
            core_size_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
            path_length_list.extend([results["path_length"]])
            path_length_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
            largest_clique_list.extend([results["largest_clique"]])
            largest_clique_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
            indepedence_list.extend([results["independence_number"]])
            indepedence_list.extend([np.nan]*(len(results["edge_betweenness"])-1)) # adding NaNs to make global variables have same length as local variables
    
    
    # Create a DataFrame for easier plotting
    data = pd.DataFrame({
        'origin':origin,
        'Edge Betweenness': edge_betweenness_list,
        'Edge Strength': edge_strength_list,
        'Edge Closeness': edge_closeness_list,
       # 'Edge Hub': edge_hub_list,
       # 'Edge Authority': edge_authority_list,
        'Edge EVC': edge_evc_list,
       # 'Edge PageRank': edge_pagerank_list,
        'Club Size': club_size_list,
       # 'Core Size' : core_size_list,
       # 'Avg Path Length': path_length_list,
        'Largest Clique': largest_clique_list,
        'Independence Number': indepedence_list
    })
     
    # Plot the results
    features = list(data.columns.values)
    features.pop(features.index('origin')) # origin is not a feature, so it should be removed from list
    
    num_features = len(features)
    fig, axes = plt.subplots(2, (num_features + 1) //2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        # sns.histplot(data, x = feature, ax=axes[i], hue = 'origin', bins=8, stat = 'percent', multiple='dodge', rw)
        # sns.stripplot(data=data, x="origin", y=feature, ax=axes[i], hue='origin', alpha=0.7, jitter=True)
        sns.boxplot(data=data, x="origin", y=feature, ax=axes[i], hue='origin')
        sns.swarmplot(data=data, x="origin", y=feature, ax=axes[i], color = "k", alpha=0.7)


        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # axes[i].set_title(feature)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Value")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Distribution of Network Properties Across All Cohorts', fontsize=16)
    plt.show()

batch_process_data("C:\\Users\\Corentin offline\\Documents\\GitHub\\MARL_modeling\\results\\")