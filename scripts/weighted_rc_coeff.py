import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import sem
from scipy.stats import ttest_ind, ttest_1samp

def mnn_cut(arr, nn = 2):
    """
    Cuts the edges of the graph in the input array by keeping only the edges that are mutual neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
    assert nn > 1, "nn should be bigger than 1."
    mnn_arr = np.zeros_like(arr)  
    neighbors_i = np.argsort(-arr, 1) #computing the nearest neighbors for local nn estimation
    neighbors_j = np.argsort(-arr, 0)
    for i in range(arr.shape[1]):
        for j in range(i, arr.shape[1]):
            if any(np.isin(neighbors_i[i, 0:nn], j)) and any(np.isin(neighbors_j[0:nn, j], i)): #local nearest distance estimation
                mnn_arr[i, j] += arr[i, j]
                mnn_arr[j, i] += arr[j, i]
    return mnn_arr

# def richness_from_data(arr, k):
#     """
#     This is a custom function to compute the weighted richness of the graph.
#     We cannot use the k_core in this case as it requires interconnectedness. Instead,
#     we use the definition of richness (see paper on rich clubs)
#     """
#     richness = []
#     for i in range(arr.shape[0]):
#         if np.sum(arr[i, :] > 0) >= k: #is this node rich?
#             richness.append(np.sum(arr[i, :]))
#     return np.sum(np.array(richness))

def richness_from_data(g, k):
    """Returns sum of edge weights where both endpoints have degree > k"""
    # deg_source = np.array([g.degree(e.source) for e in g.es])
    # deg_target = np.array([g.degree(e.target) for e in g.es])
    # weights = np.array(g.es["weight"])
    # mask = (deg_source >= k) & (deg_target >= k)
    # return np.sum(weights[mask])
    return 2*sum(e['weight'] for e in g.es 
                      if g.degree(e.source) >= k and g.degree(e.target) >= k)
    # return 2*sum(1 for e in g.es 
    #                   if g.degree(e.source) >= k and g.degree(e.target) >= k)

def random_richness(data, k, iterations = 100):
    random_richness = []
    for i in range(iterations):
        # cut_arr = (mnn_cut(data, k) > 0)*1
        cut_arr = mnn_cut(data, k)
        g_rand = ig.Graph.Weighted_Adjacency(cut_arr, mode='undirected')
        original_weights = g_rand.es['weight']
        g_rand.rewire(mode="simple", n=1000)
        shuffled_weights = original_weights.copy()
        np.random.shuffle(shuffled_weights)
        g_rand.es['weight'] = shuffled_weights
        cut_shuffled_arr = np.array(g_rand.get_adjacency(attribute="weight"))
        # n = data.shape[0]
        # triu_indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
        # upper_values = data[triu_indices]
        # np.random.shuffle(upper_values)
        # shuffled_data = np.zeros_like(data)
        # shuffled_data[triu_indices] = upper_values          
        # shuffled_data.T[triu_indices] = upper_values  
        # cut_shuffled_arr = (mnn_cut(shuffled_data, k) > 0)*1
        where_rand_rc = np.where(np.sum(cut_shuffled_arr > 0, 0) >= k)
        rand_normalization = np.sum(cut_shuffled_arr[where_rand_rc, :])
        # g_rand = ig.Graph.Weighted_Adjacency(cut_shuffled_arr, mode='undirected')
        # rand_richness = richness_from_data(g_rand, k)
        random_richness.append(richness_from_data(g_rand, k)/rand_normalization)
    rand_richnesses = np.array(random_richness)
    return np.sum(rand_richnesses[np.where(rand_richnesses != np.inf)])/iterations

def weighted_rich_club(data, k = 3, iterations = 50):
    # observed_richness = richness_from_data(data, k)
    # cut_arr = (mnn_cut(data, k) > 0)*1
    cut_arr = mnn_cut(data, k)
    where_rc = np.where(np.sum(cut_arr > 0, 0) >= k)
    normalization = np.sum(cut_arr[where_rc, :])
    g = ig.Graph.Weighted_Adjacency(cut_arr, mode='undirected')
    observed_richness = richness_from_data(g, k)/normalization

    rand_richness = random_richness(data, k, iterations)
    return observed_richness/rand_richness

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
def rc_coefficients_all(variable, k, window = 3):
    """
    Plots the rc coefficients of all groups, at the expense of showing the distribution of value (non notion of significance.)
    """
    all_coeffs = []
    avg_rc_coeffs, std_rc_coeffs = [], []
    fig, ax = plt.subplots(1, 1)
    for day in tqdm(np.arange(1, 15//window+1)):
        coefficients = []
        for graph_idx in range(len(labels)):
            try:
                path_to_file = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_"+str(day)+".csv"
                arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
                data = arr[1:, 1:].astype(float)
                # data = mnn_cut(data, k)
                rc = weighted_rich_club(data, k, 50)
                coefficients.append(rc)
            except Exception as e:
                print(e)
        all_coeffs.append(coefficients)
        coefficients = np.array(coefficients)
        avg_rc_coeffs.append(np.nanmean(coefficients))
        std_rc_coeffs.append(np.nanstd(coefficients[~np.isnan(coefficients)]))
        plt.scatter([day]*len(coefficients), coefficients, alpha = 0.5, s = 100, color = "k")
        plt.boxplot(coefficients[~np.isnan(coefficients)], positions = [day], showfliers= False)
        t_statistic, p_value = ttest_1samp(coefficients[~np.isnan(coefficients)], popmean=1, alternative='greater') 
        print(p_value)
        if p_value < 0.001:
            plt.text(day, 2.2, "***", fontsize="20",horizontalalignment='center')
        elif p_value < 0.01:
            plt.text(day, 2.2, "**", fontsize="20",horizontalalignment='center')
        elif p_value < 0.05:
            plt.text(day, 2.2, "*", fontsize="20",horizontalalignment='center')


    ax.spines[['right', 'top']].set_visible(False)   
    t = np.arange(1, 15//window+1)
    avg_rc_coeffs, std_rc_coeffs = np.array(avg_rc_coeffs), np.array(std_rc_coeffs)
    # plt.figure(figsize=(4, 3.5))
    # for line in range(len(all_coeffs[0])):
    #     plt.plot(t, [all_coeffs[idx][line] for idx in range(len(all_coeffs))])
    plt.plot(t, avg_rc_coeffs, '-o', lw = 2, c = "gray", label = "Average across groups")
    # # plt.scatter(t, avg_rc_coeffs, lw = 2, c = "k", label = "Rich club coefficient")
    # plt.fill_between(t, avg_rc_coeffs - std_rc_coeffs, avg_rc_coeffs + std_rc_coeffs, color="k", alpha=0.2)
    plt.hlines(1, t[0], t[-1], ls = "--", color = "k", label = "Randomized graph")
    plt.xlabel("Day", fontsize = 18)
    plt.ylabel("Richness", fontsize = 18)
    # plt.xticks(range(1, 16//window), fontsize = 14) 
    plt.yticks(np.linspace(0, 2.5, 6), fontsize = 14)
    # plt.ylim(0.3, 2.5)
    plt.legend()
    plt.tight_layout()
    
    return avg_rc_coeffs, std_rc_coeffs


def rc_coefficient_selected(group, variable, k, window = 3):
    plt.figure()
    all_coeffs = []
    avg_rc_coeffs, sem_rc_coeffs = [], []
    print(labels[group])
    for day in tqdm(np.arange(1, 15//window+1)):
        coefficients = []
        try:
            path_to_file = f"..\\data\\both_cohorts_{window}days\\"+labels[group]+"\\"+variable+f"_resD{window}_"+str(day)+".csv"
            arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
            data = arr[1:, 1:].astype(float)
            for i in range(100):
                rc = weighted_rich_club(data, k, 1)
                coefficients.append(rc)
        except Exception as e:
            print(e)
        all_coeffs.append(coefficients)
        coefficients = np.array(coefficients)
        med_val = np.nanmedian(coefficients) if np.nanmedian(coefficients) != np.inf else 0
        avg_rc_coeffs.append(med_val)
        SEM = sem(coefficients[~np.isnan(coefficients)])
        SEM = SEM if (~np.isinf(SEM) and ~np.isnan(SEM)) else 0
        sem_rc_coeffs.append(SEM)
        # plt.scatter([day]*len(coefficients), coefficients, alpha = 0.5, s = 100, color = "k")
      #  plt.boxplot(coefficients, positions = [day], showfliers= False)
        
    t = np.arange(1, 15//window+1)
    avg_rc_coeffs, sem_rc_coeffs = np.array(avg_rc_coeffs), np.array(sem_rc_coeffs)
    # plt.figure(figsize=(4, 3.5))
    # for line in range(len(all_coeffs[0])):
    #     plt.plot(t, [all_coeffs[idx][line] for idx in range(len(all_coeffs))])
    plt.plot(t, avg_rc_coeffs, '-o', lw = 2, c = "gray", label = "Median value")
    # # plt.scatter(t, avg_rc_coeffs, lw = 2, c = "k", label = "Rich club coefficient")
    plt.fill_between(t, avg_rc_coeffs - sem_rc_coeffs, avg_rc_coeffs + sem_rc_coeffs, color="k", alpha=0.2)
    plt.hlines(1, t[0], t[-1], ls = "--", color = "k", label = "Randomized graph")
    plt.xlabel("Day", fontsize = 18)
    plt.ylabel("Richness", fontsize = 18)
    plt.xticks(range(1, 16//window + 1), fontsize = 14) 
    # plt.yticks(np.linspace(1, 3, 3), fontsize = 14)
    # plt.ylim(0.5, 2)
    plt.legend()
    plt.tight_layout()
    
    return avg_rc_coeffs, sem_rc_coeffs

if __name__ == "__main__":
    a, s = rc_coefficients_all('interactions', 3, 3)

    # a, s = rc_coefficient_selected(5, 'interactions', 3, 3)
    
