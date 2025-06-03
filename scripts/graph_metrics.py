import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import seaborn as sns
from scipy import stats
import pandas as pd
from scipy.stats import sem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
from tqdm import tqdm 
from sklearn.manifold import TSNE
from sklearn import manifold
sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels
from utils import get_category_indices
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats
import warnings

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"

def rescale(arr):
    if len(arr) > 0:
        normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    else:
        return 
    return normalized_arr

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def default_on_error(graph_idx, variable, window, rm_weak_histo = True):
    """
    If the data for a graph on a specific day is not available because of detection issue, nans should be returned. 
    To preserve the structure of the data array, the number of returned nans should reflect the number of mutants, rc members and wt in the cohort. 
    """
    mutants, rc, others, wt, RFIDs = get_category_indices(graph_idx, "approaches", window, rm_weak_histo) # load data from exp day 1 as a template, because all group were properly detected.
    return [np.nan]*len(mutants), [np.nan]*len(rc), [np.nan]*len(others), \
        [np.nan]*len(wt), [np.nan]*len(RFIDs), \
        {"Mouse_RFID": [np.nan]*len(RFIDs), "mutant": [np.nan]*len(RFIDs), "RC": [np.nan]*len(RFIDs), "Group_ID": int(labels[graph_idx][1:])}

def measures(graph_idx, day, window = 3, measure = "hub", variable = "approaches", mnn = None, mutual = True, rm_weak_histo = True, threshold = 0.0):
    """
    Computes specified graph-theoretical metric for a given cohort, day, and time window.

    Parameters:
        graph_idx (int): Index of the group/cohort to analyze (corresponds to an entry in the 'labels' list).
        day (int): The day or timepoint for which to compute the measure (depends on the window size).
        window (int, optional): Size of the time window for averaging the graph representation (default is 3).
        metric (str, optional): Name of the graph-theoretical measure to compute (e.g., "hub", "degree", "pagerank").
        variable (str, optional): Type of interaction to analyze ("approaches" or "interactions").
        mnn (int or None, optional): Minimum number of neighbors for graph thresholding (used for unweighted measures).
        mutual (bool, optional): Whether to use mutual nearest neighbors for graph construction.
            if mnn is None (used for weighted metrics), mutual is ignored.
        rm_weak_histo (bool, optional): If True, removes mutants with weak histology from analysis.

    Returns:
        tuple: (scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, metadata)
            - scores_mutants: List of measure values for mutant mice.
            - scores_rc: List of measure values for RC (rich club) mice.
            - scores_rest: List of measure values for other mice (not mutant or RC).
            - scores_wt: List of measure values for wild-type mice.
            - scores_all: List of measure values for all mice in the group.
            - metadata: Dictionary with keys "Mouse_RFID", "mutant", "RC", and "Group_ID" for the current group.
    """
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
    datapath = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_"+str(day)+".csv"
    if window not in [1, 3, 7]:
        print("Incorrect time window")
        return 
    try:
        data = read_graph([datapath], percentage_threshold = threshold, mnn = mnn, mutual = mutual)[0]
        arr = np.loadtxt(datapath, delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)
    except Exception as e:
        return default_on_error(graph_idx, variable, window, rm_weak_histo)
    if variable == "interactions":
        mode = 'undirected'
        data = (data + np.transpose(data))/2 # ensure symmetry
    elif variable == "approaches":
        mode = 'directed'
    else:
        raise NameError("Incorrect input argument for 'variable'.")
        
    data = np.where(data > 0.01, data, 0)
    
    g = ig.Graph.Weighted_Adjacency(data, mode=mode)

    curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[graph_idx][1:]), :]
    # figuring out index of true mutants in current group
    if not rm_weak_histo:
        is_mutant = [True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
    else:
        is_mutant = []
        for rfid in RFIDs:
            if len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                histology = metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "histology"].values[0]
                if histology != 'weak':
                    # mouse is a mutant and histology is strong or unknown
                    is_mutant.append(True)
                else:
                    is_mutant.append(False)
            elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) != 0 and not metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values[0]:
                is_mutant.append(False)
            elif len(metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values) == 0: 
                # print("Mouse not found, treat as WT")
                is_mutant.append(False)

    is_RC = [] # no list comprehension allowed because of deprenciation warning due to RFID mismatch
    for rfid in RFIDs:
        if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values.size > 0:
            if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values[0]:
                is_RC.append(True)
            else:
                is_RC.append(False)
        else:
                is_RC.append(False)
        
    graph_length = len(g.vs())
    mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
    rc = np.where(is_RC)[0] if len(np.where(is_RC)) != 0 else [] # indices of RC in this group
    others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
    wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
    
    if measure == "hub":
        scores_mutants = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.hub_score(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "indegree":
        scores_mutants = [g.degree(mode="in")[i] for i in mutants]
        scores_rc = [g.degree(mode="in")[i] for i in rc]
        scores_rest = [g.degree(mode="in")[i] for i in others]
        scores_wt = [g.degree(mode="in")[i] for i in wt]    
        scores_all = [g.degree(mode="in")[i] for i in range(len(RFIDs))]
        
    elif measure == "outdegree":
        scores_mutants = [g.degree(mode="out")[i] for i in mutants]
        scores_rc = [g.degree(mode="out")[i] for i in rc]
        scores_rest = [g.degree(mode="out")[i] for i in others]
        scores_wt = [g.degree(mode="out")[i] for i in wt]    
        scores_all = [g.degree(mode="out")[i] for i in range(len(RFIDs))]
        
    elif measure == "degree":
        scores_mutants = [g.degree(mode="all")[i] for i in mutants]
        scores_rc = [g.degree(mode="all")[i] for i in rc]
        scores_rest = [g.degree(mode="all")[i] for i in others]
        scores_wt = [g.degree(mode="all")[i] for i in wt]    
        scores_all = [g.degree(mode="all")[i] for i in range(len(RFIDs))]

    elif measure == "summed insubcomponent":
        scores_mutants = [np.sum(g.subcomponent(i, mode="in")) for i in mutants]
        scores_rc = [np.sum(g.subcomponent(i, mode="in")) for i in rc]
        scores_rest = [np.sum(g.subcomponent(i, mode="in")) for i in others]
        scores_wt = [np.sum(g.subcomponent(i, mode="in")) for i in wt]
        scores_all = [np.sum(g.subcomponent(i, mode="in")) for i in range(len(RFIDs))]

    elif measure == "summed outsubcomponent":
        scores_mutants = [np.sum(g.subcomponent(i, mode="out")) for i in mutants]
        scores_rc = [np.sum(g.subcomponent(i, mode="out")) for i in rc]
        scores_rest = [np.sum(g.subcomponent(i, mode="out")) for i in others]
        scores_wt = [np.sum(g.subcomponent(i, mode="out")) for i in wt]    
        scores_all = [np.sum(g.subcomponent(i, mode="out")) for i in range(len(RFIDs))]

    elif measure == "summed injaccard":
        scores_mutants = [np.sum(g.similarity_jaccard(mode="all")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="all")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="all")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="all")[i]) for i in wt]  
        scores_all = [np.sum(g.similarity_jaccard(mode="all")[i]) for i in range(len(RFIDs))]

    elif measure == "summed outjaccard":
        scores_mutants = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in wt]     
        scores_all = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in range(len(RFIDs))]
        
    elif measure == "summed jaccard":
        scores_mutants = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in wt]     
        scores_all = [np.sum(g.similarity_jaccard(mode="out")[i]) for i in range(len(RFIDs))]
        
    elif measure == "instrength":
        scores_mutants = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in mutants]
        scores_rc = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in rc]
        scores_rest = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in others]
        scores_wt = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in wt]  
        scores_all = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in range(len(RFIDs))]
        
    elif measure == "outstrength":
        scores_mutants = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in mutants]
        scores_rc = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in rc]
        scores_rest = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in others]
        scores_wt = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in wt]    
        scores_all = [np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in range(len(RFIDs))]
        
    elif measure == "normed instrength":
        scores_mutants = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in mutants])
        scores_rc = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in rc])
        scores_rest = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in others])
        scores_wt = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in wt])
        scores_all = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="in")]) for i in range(len(RFIDs))])
        
    elif measure == "normed outstrength":
        scores_mutants = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in mutants])
        scores_rc = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in rc])
        scores_rest = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in others])
        scores_wt = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in wt])    
        scores_all = rescale([np.sum([g.es[edge]["weight"] for edge in g.incident(i, mode="out")]) for i in range(len(RFIDs))])

    elif measure == "outcloseness":
        scores_mutants = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.closeness(mode="out", weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]
        
    elif measure == "incloseness":
        scores_mutants = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in wt]  
        scores_all = [g.closeness(mode="in", weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]
        
    elif measure == "closeness":
        scores_mutants = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in wt]  
        scores_all = [g.closeness(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]
                
    elif measure == "pagerank":
        scores_mutants = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.pagerank(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "authority":
        scores_mutants = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.authority_score(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "eigenvector_centrality":
        scores_mutants = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "harmonic_centrality":
        scores_mutants = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in mutants]
        scores_rc = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in rc]
        scores_rest = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in others]
        scores_wt = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in wt]
        scores_all = [g.harmonic_centrality(weights = [e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "betweenness":
         scores_mutants = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
         scores_all = [g.betweenness(weights = [1/e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "constraint":
         scores_mutants = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
         scores_all = [g.constraint(weights = [1/e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

    elif measure == "summed cocitation":
         scores_mutants = [np.sum(g.cocitation()[i]) for i in mutants]
         scores_rc = [np.sum(g.cocitation()[i]) for i in rc]
         scores_rest = [np.sum(g.cocitation()[i]) for i in others]  
         scores_wt = [np.sum(g.cocitation()[i]) for i in wt]  
         scores_all = [np.sum(g.cocitation()[i]) for i in range(len(RFIDs))]

    elif measure == "summed bibcoupling":
         scores_mutants = [np.sum(g.bibcoupling()[i]) for i in mutants]
         scores_rc = [np.sum(g.bibcoupling()[i]) for i in rc]
         scores_rest = [np.sum(g.bibcoupling()[i]) for i in others]  
         scores_wt = [np.sum(g.bibcoupling()[i]) for i in wt]  
         scores_all = [np.sum(g.bibcoupling()[i]) for i in range(len(RFIDs))]
         
    elif measure == "summed common neighbors":
         n = len(RFIDs)
         scores_mutants = [np.sum(g.bibcoupling()[i]) + np.sum(g.cocitation()[i]) for i in mutants]
         scores_rc = [np.sum(g.bibcoupling()[i]) + np.sum(g.cocitation()[i]) for i in rc]
         scores_rest = [np.sum(g.bibcoupling()[i]) + np.sum(g.cocitation()[i]) for i in others]  
         scores_wt = [np.sum(g.bibcoupling()[i]) + np.sum(g.cocitation()[i]) for i in wt]  
         scores_all = [np.sum(g.bibcoupling()[i]) + np.sum(g.cocitation()[i]) for i in range(n)]
        
    elif measure == "transitivity":
         scores_mutants = [g.transitivity_local_undirected()[i] for i in mutants]
         scores_rc = [g.transitivity_local_undirected()[i] for i in rc]
         scores_rest = [g.transitivity_local_undirected()[i] for i in others]  
         scores_wt = [g.transitivity_local_undirected()[i] for i in wt]  
         scores_all = [g.transitivity_local_undirected()[i] for i in range(len(RFIDs))]
         
         
         
    elif measure == "out persistence" or measure == "in persistence":
        datapath2 = f"..\\data\\both_cohorts_{window}days\\"+labels[graph_idx]+"\\"+variable+f"_resD{window}_"+str(day+1)+".csv"
        if window not in [1, 3, 7]:
            print("Incorrect time window")
            return 
        try:
            data2 = read_graph([datapath2], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
            arr2 = np.loadtxt(datapath, delimiter=",", dtype=str)
            RFIDs = arr[0, 1:].astype(str)
        except Exception as e:
            scores_mutants = [np.nan for i in mutants]
            scores_rc = [np.nan for i in rc]
            scores_rest = [np.nan for i in others]  
            scores_wt = [np.nan for i in wt]  
            scores_all = [np.nan for i in range(len(RFIDs))]
            return scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}
        if variable == "interactions":
            mode = 'undirected'
            data = (data + np.transpose(data))/2 # ensure symmetry
        elif variable == "approaches":
            mode = 'directed'
        else:
            raise NameError("Incorrect input argument for 'variable'.")

        data= np.where(data > 0.01, data, 0) 
        data2 = np.where(data2 > 0.01, data2, 0)
            
        axis = 1 if measure[:3] == 'out' else 0
        cond1 = np.abs(data2 - data) + data <= 1.5*data
        persistance = np.sum(cond1, axis = axis)

        scores_mutants = [persistance[i] for i in mutants]
        scores_rc = [persistance[i] for i in rc]
        scores_rest = [persistance[i] for i in others]  
        scores_wt = [persistance[i] for i in wt]  
        scores_all = [persistance[i] for i in range(len(RFIDs))]

    else:
        raise Exception("Unknown or misspelled input measurement.") 
    
    return scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}
    
def mk_measures_dataframe(variable = "approaches", window = 3, mnn = 4, mutual = False, rm_weak_histo = True):
    """
    Formats the measures made on the graphs into a pandas dataframe and saves to a csv file
    """

    unweighted_features = ["indegree", "outdegree", "degree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed common neighbors", "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard",
        "summed jaccard"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "normed instrength", "normed outstrength", "instrength", "outstrength", "out persistance", "in persistance"]
    
    features_df = pd.DataFrame()
    
    for feature in weighted_features:
        print(f"Processing {feature}...")
        all_scores, day, RFIDs, Group_ID, mutants, RCs = [], [], [], [], [], []
        for d in range(15//window):
            for j in range(len(labels)):
                res = measures(j, d+1, window, feature, variable, None, True, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                if res is None:
                    continue
                scores, metadata = res[4], res[5]
                tags, is_mutant, is_RC, group = metadata["Mouse_RFID"], metadata["mutant"], metadata["RC"], metadata["Group_ID"]
                all_scores.extend(scores)
                day.extend([d]*len(tags))
                RFIDs.extend(tags)
                mutants.extend(is_mutant)
                RCs.extend(is_RC)
                Group_ID.extend([group]*len(tags))
                
        features_df["Mouse_RFID"] = RFIDs
        features_df["Group_ID"] = Group_ID
        features_df["mutant"] = mutants
        features_df[f"Timestamp_base{window}"] = day
        features_df["RC"] = RCs
        features_df[feature] = all_scores
        
    for feature in unweighted_features:
        print(f"Processing {feature}...")
        all_scores, day, RFIDs, mutants, RCs = [], [], [], [], []
        for d in range(15//window):
            for j in range(len(labels)):
                res = measures(j, d+1, window, feature, variable, mnn, mutual, rm_weak_histo) # unweighted edge measure NEED graph cut : mnn = mnn
                if res is None:
                    continue
                scores, metadata = res[4], res[5]
                tags, is_mutant, is_RC = metadata["Mouse_RFID"], metadata["mutant"], metadata["RC"]
                all_scores.extend(scores)
                day.extend([d]*len(tags))
                RFIDs.extend(tags)
                mutants.extend(is_mutant)
                RCs.extend(is_RC)
                
        features_df["Mouse_RFID"] = RFIDs
        features_df["mutant"] = mutants
        features_df[f"Timestamp_base{window}"] = day
        features_df["RC"] = RCs
        features_df[feature] = all_scores
            
    save_name = f"graph_features_{variable}_d{window}_mnn{mnn}" if mutual else f"graph_features_{variable}_d{window}_nn{mnn}"
    save_name = f"{save_name}_rm_weak.csv" if rm_weak_histo else f"{save_name}_all_muts.csv" 
    features_df.to_csv(f"..\\data\\processed_metrics\\{save_name}", index = False)
    return features_df

def mk_derivatives_dataframe(variable = "approaches", window = 3, mnn = 4, mutual = False, rm_weak_histo = True):
    """
    Formats the derivative of the measures made on the graphs as a function of time into a pandas dataframe and saves to a csv file
    """

    unweighted_features = ["indegree", "outdegree", "degree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed common neighbors", "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard",
        "summed jaccard", "out persistance"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "normed instrength", "normed outstrength","instrength", "outstrength", "out persistance", "in persistance"]
    
    features_df = pd.DataFrame()
    
    for feature in weighted_features:
        print(f"Processing {feature}...")
        all_scores, day, RFIDs, Group_ID, mutants, RCs = [], [], [], [], [], []
        for d in range(15//window):
            for j in range(len(labels)):
                res_t1 = measures(j, d+1, window, feature, variable, None, True, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                res_t2 = measures(j, d+2, window, feature, variable, None, True, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                if res_t1 is None or res_t2 is None:
                    continue
                scores_t1, metadata = np.array(res_t1[4]), res_t2[5]
                scores_t2, _ = np.array(res_t2[4]), res_t1[5]
                scores = scores_t2 - scores_t1
                if feature == "in persistance" or feature == "out persistance": # persistance is already a derivative!
                    scores = scores_t1
                tags, is_mutant, is_RC, group = metadata["Mouse_RFID"], metadata["mutant"], metadata["RC"], metadata["Group_ID"]
                all_scores.extend(scores)
                day.extend([d]*len(tags))
                RFIDs.extend(tags)
                mutants.extend(is_mutant)
                RCs.extend(is_RC)
                Group_ID.extend([group]*len(tags))
                
        features_df["Mouse_RFID"] = RFIDs
        features_df["Group_ID"] = Group_ID
        features_df["mutant"] = mutants
        features_df[f"Timestamp_base{window}"] = day
        features_df["RC"] = RCs
        features_df[feature] = all_scores
        
    for feature in unweighted_features:
        print(f"Processing {feature}...")
        all_scores, day, RFIDs, mutants, RCs = [], [], [], [], []
        for d in range(15//window):
            for j in range(len(labels)):
                res_t1 = measures(j, d+1, window, feature, variable, mnn, mutual, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                res_t2 = measures(j, d+2, window, feature, variable, mnn, mutual, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                if res_t1 is None or res_t2 is None:
                    continue
                scores_t1, metadata = np.array(res_t1[4]), res_t2[5]
                scores_t2, _ = np.array(res_t2[4]), res_t1[5]
                scores = scores_t2 - scores_t1
                tags, is_mutant, is_RC = metadata["Mouse_RFID"], metadata["mutant"], metadata["RC"]
                all_scores.extend(scores)
                day.extend([d]*len(tags))
                RFIDs.extend(tags)
                mutants.extend(is_mutant)
                RCs.extend(is_RC)
                
        features_df["Mouse_RFID"] = RFIDs
        features_df["mutant"] = mutants
        features_df[f"Timestamp_base{window}"] = day
        features_df["RC"] = RCs
        features_df[feature] = all_scores
            
    save_name = f"metrics_derivative_{variable}_d{window}_mnn{mnn}" if mutual else f"metrics_derivative_{variable}_d{window}_nn{mnn}"
    save_name = f"{save_name}_rm_weak.csv" if rm_weak_histo else f"{save_name}_all_muts.csv" 
    features_df.to_csv(f"..\\data\\processed_metrics\\{save_name}", index = False)
    return features_df

def get_metric_df(metric, variable = "approaches", derivative = False, window = 1, mnn = 4, mutual = False, rm_weak_histo = True, threshold = 0.0):
    """
    Extracts the values of the input metric as a function of time and returns it as a pandas dataframe 
    along with useful meta data used for plotting.
    """
    unweighted_features = ["indegree", "outdegree", "degree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed common neighbors", "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard",
        "summed jaccard", "out persistence"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "closeness",  "betweenness", "normed instrength", "normed outstrength","instrength", "outstrength", "out persistance", "in persistance"]
    
    is_weighted = True if metric in weighted_features else False
    if is_weighted:
        mnn = None
    
    features_df = pd.DataFrame()
    
    all_scores, day, RFIDs, Group_ID, mutants, RCs = [], [], [], [], [], []
    time_range = np.arange(7, 16, window) if metric not in ["instrength", "outstrength"] else np.arange(2, 16, window)
    for d in time_range:
        for j in range(len(labels)):
            res_t1 = measures(j, d+1, window, metric, variable, mnn, mutual, rm_weak_histo, threshold) 
            if res_t1 is None:
                continue
            scores, metadata = np.array(res_t1[4]), res_t1[5]
            if derivative:
                res_t2 = measures(j, d+2, window, metric, variable, mnn, mutual, rm_weak_histo, threshold) # weighted edge measures do not need graph cut : mnn = None
                scores_t1, metadata = np.array(res_t1[4]), res_t1[5]
                scores_t2, _ = np.array(res_t2[4]), res_t1[5]
                scores = scores_t2 - scores_t1
                if metric == "in persistence" or metric == "out persistence": # persistance is already a derivative!
                    scores = scores_t1
                
            tags, is_mutant, is_RC, group = metadata["Mouse_RFID"], metadata["mutant"], metadata["RC"], metadata["Group_ID"]
            all_scores.extend(scores)
            day.extend([d]*len(tags))
            RFIDs.extend(tags)
            mutants.extend(is_mutant)
            RCs.extend(is_RC)
            Group_ID.extend([group]*len(tags))
            
    features_df["Mouse_RFID"] = RFIDs
    features_df["Group_ID"] = Group_ID
    features_df["mutant"] = mutants
    features_df[f"Timestamp_base{window}"] = day
    features_df["RC"] = RCs
    features_df[metric] = all_scores
        
    return features_df

def manual_plot(metrics, variable = "approaches", mnn = None, mutual = False, dim = 2, show_RC = True, std = True):
    """
    Reads the dataframe, and plots the input variables specified in 'variables' against each other.
    Var should contain at least 2 variables. Then, a PCA round is applied to transform the data,
    remove potential covariance and rescale the data by variance, which makes the plots look nicer.
    
    Args:
        dim (int): dimension for the PCA reduction / covariance suppression
        variables (list): list of variable names (strings) of grpah metrices.
        show_RC (bool): whether or not to color the points associated to the RC.
        std (bool): whether or not to group the repeated points associated to each day by 
            their std, i.e. whether or not we plot one point per day, or if the points 
            represent the fluctuations in the timeseries.
    """
    from tmp_timegraph_metrics import get_time_metric_df
    temporal_metrics = ["summed outNEF", "summed inNEF", "summed outICI",
                        "summed inICI", "summed outburstiness", "summed inburstiness"]
    df = pd.DataFrame()
    for var in metrics:
        if var not in temporal_metrics:
            curr_df = get_metric_df(var, variable, True, 1, mnn, mutual, True) # derivative of variables
            
            if std:
                curr_df = curr_df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
                  'mutant': 'first',  # Keep the first value (or use another aggregation function)
                  'RC': 'first',      # Keep the first value (or use another aggregation function)
                  **{col: 'std' for col in curr_df.select_dtypes(include='number').columns if col not in ['Group_ID']}
                  })
                curr_df = curr_df.dropna()
            df["Mouse_RFID"] = curr_df["Mouse_RFID"]
            df["mutant"] = curr_df["mutant"]
            df["RC"] = curr_df["RC"]
            df[var] = curr_df[var]
            
        else:
            curr_df = get_time_metric_df(var, 1, variable, mnn = mnn, mutual = mutual, weighted = True)
            df["Mouse_RFID"] = curr_df["Mouse_RFID"]
            df["mutant"] = curr_df["mutant"]
            df["RC"] = curr_df["RC"]
            df[var] = curr_df[var]
        
    results = df[metrics].values

    if dim is not None:
        assert dim == 2 or dim == 3, "Dim argument for PCA should be 2 or 3 for visualization"
        scaler = StandardScaler()
        results = scaler.fit_transform(results)
        pca = PCA(n_components=dim)
        results = pca.fit_transform(results)
    
    if results.shape[1] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        if not show_RC:
            ax.scatter(results[(df["mutant"] == False).values, 0],
                       results[(df["mutant"] == False).values, 1],
                                 c="blue", alpha=0.9, s = 100, label = "control")
        if show_RC:
            other_idx = np.logical_and((df["mutant"] == False).values, df["RC"] == False) 
            RC_idx = np.logical_and((df["mutant"] == False).values, df["RC"] == True)
            ax.scatter(results[other_idx, 0], results[other_idx, 1],
                                 c="k", alpha=0.9, s = 100, label = "non-members")
            ax.scatter(results[RC_idx, 0], results[RC_idx, 1],
                                 c="green", alpha=0.9, s = 100, label = "RC")
            
        ax.scatter(results[df["mutant"].values == True, 0], results[df["mutant"].values == True, 1],
                             c="red", alpha=0.9, s = 100, label = "mutant")
        
    if results.shape[1] == 3:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        if not show_RC:
            ax.scatter(results[(df["mutant"] == False).values, 0], 
                       results[(df["mutant"] == False).values, 1], 
                       results[(df["mutant"] == False).values, 2],
                                 c="blue", alpha=0.9, s = 100, label = "control")
        if show_RC:
            other_idx = np.logical_and(df["mutant"] == False, df["RC"] == False) 
            RC_idx = np.logical_and(df["mutant"] == False, df["RC"] == True)
            ax.scatter(results[other_idx, 0], results[other_idx, 1], results[other_idx, 2],
                                 c="k", alpha=0.9, s = 100, label = "non-members")
            ax.scatter(results[RC_idx, 0], results[RC_idx, 1], results[RC_idx, 2],
                                 c="green", alpha=0.9, s = 100, label = "RC")
            
        ax.scatter(results[df["mutant"].values == True, 0], 
                   results[df["mutant"].values == True, 1], 
                   results[df["mutant"].values == True, 2],
                             c="red", alpha=0.9, s = 100, label = "mutant")
    
    ax.set_xlabel(f"Transformed {metrics[0]}", fontsize=12)
    ax.set_ylabel(f"Transformed {metrics[1]}", fontsize=12)
    if len(metrics) == 3:
        ax.set_zlabel(f"Transformed {metrics[2]}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()

def statistic(x, y):
    x = np.concatenate(x)
    y = np.concatenate(y)
    return np.mean(x) - np.mean(y)
    # return np.median(x) - np.median(y)

def add_significance(data, var, ax, bp):
    """Computes and adds p-value significance to input ax and boxplot"""
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    rfids1 = data[0]["Mouse_RFID"].values
    rfids2 = data[1]["Mouse_RFID"].values
    data1, data2 = data[0], data[1]

    # Group data by RFID
    grouped_data1 = [data1.loc[data1["Mouse_RFID"] == rfid, var].values for rfid in np.unique(rfids1)]
    grouped_data1 = [list(d) for d in grouped_data1]
    grouped_data2 = [data2.loc[data2["Mouse_RFID"] == rfid, var].values for rfid in np.unique(rfids2)]
    grouped_data2 = [list(d)for d in grouped_data2]
    
    combined_data = grouped_data1 + grouped_data2
    labels = [0] * len(grouped_data1) + [1] * len(grouped_data2)

    ## custom permutation test
    observed_stat = statistic(grouped_data1, grouped_data2)

    # Generate permutations
    permuted_stats = []
    for _ in tqdm(range(10000)):
        np.random.shuffle(labels)
        permuted_group1 = [combined_data[i] for i in range(len(labels)) if labels[i] == 0]
        permuted_group2 = [combined_data[i] for i in range(len(labels)) if labels[i] == 1]
        permuted_stats.append(statistic(permuted_group1, permuted_group2))

    # Compute p-value
    permuted_stats = np.array(permuted_stats)
    p = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    print(p)

    x1 = 1
    x2 = 2
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    bar_height = (y_range * 0.05 * 1) + top #- 20
    bar_tips = bar_height - (y_range * 0.02)
    ax.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )

    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = f"p = {np.round(p, 3)}"
    text_height = bar_height + (y_range * 0.01)
    ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
    
def boxplot_metric_mutant(var, derivative = False, aggregation = None, window = 1, mnn = 4, mutual = False,
                          rm_RC = True, overlay_RC = False, threshold = 0.0, ax = None):
    """
     Plots a boxplot comparing the specified graph-theoretical metric between mutant and wild-type mice. By Default, RC members
     are excluded from the comparison unless 'overlay_RC' is True. The function aggregates the metric per mouse using the specified method, 
     then plots the distribution for mutants and wild-type mice, and performs a statistical test to indicate significance.
    
     Parameters:
         measure (str): The graph-theoretical metric to plot (e.g., "hub", "degree", "outstrength").
         window (int): The time window size used for averaging the graph representation.
         aggregation: Aggregation method for repeated measures per mouse. Can be None, "mean" or "std".
     """
     
    df = get_metric_df(var, "approaches", derivative, window, mnn, mutual, True, threshold)
    df = df.dropna()
    
    if var == "instrength" or var == "outstrength" or var == "normed outstrength" or var == "normed outstrength":
        df = df[df[f"Timestamp_base{window}"] != 0]  # first time point is strong outlier for strength, for some reason
        df = df[df[var] < 150]
        warnings.warn("Strength behaves very strangely on first day,removing it.\n Also removing outliers > 150")

    if aggregation == "CV" and ~derivative:
        # Compute mean and std per animal
        grouped = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
            'mutant': 'first',
            'RC': 'first',
            var: ['mean', 'std']
        })
        # Flatten MultiIndex columns
        grouped.columns = ['Mouse_RFID', 'Group_ID', 'mutant', 'RC', 'mean', 'std']
        # Compute coefficient of variation
        grouped[var] = grouped['std'] / grouped['mean']
        # Remove inf/nan values (mean=0)
        grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=[var])
        df = grouped

    elif aggregation is not None:
        assert aggregation == "std" or aggregation == "mean", "aggregation should be mean or std"
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: aggregation for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
        
    # RC are exluded for the comparison, unless overlay RC is True
    if rm_RC:
        data = [df.loc[np.logical_and(df["mutant"], df["RC"] == False), ['Mouse_RFID', var]], 
                df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), ['Mouse_RFID', var]] ]
    else:
        data = [df.loc[df["mutant"],  ['Mouse_RFID', var]],  df.loc[df["mutant"] == False,  ['Mouse_RFID', var]] ]

    if overlay_RC: # if we compare whole population, and show RC as green overlay
        data = [df.loc[df["mutant"], ['Mouse_RFID', var]],   df.loc[df["mutant"] == False, ['Mouse_RFID', var]]]
        
    # convert the dataframe data to a list of np.array that can be passed to plt.boxplot
    numeric_data = [data[i].drop("Mouse_RFID", axis = 1, inplace = False).values for i in range(2)]
    numeric_data = [np.concatenate(d) for d in numeric_data]
            
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 100, 0.5
    
    if rm_RC:
        bp = ax.boxplot(numeric_data, labels=["Mutants", "Non-members WT"], showfliers = False)
    else:
        bp = ax.boxplot(numeric_data, labels=["Mutants", "WT"], showfliers = False)

    add_significance(data, var, ax, bp)
    if rm_RC:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"], df["RC"] == False),var].values))], 
                    df.loc[np.logical_and(df["mutant"], df["RC"] == False),var], alpha = alpha, s = size, color = "red", linewidths = 0); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), var].values))], 
                    df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), var], alpha = alpha, s = size, color = "gray", label = "Non-member", linewidths = 0); 
    elif not rm_RC:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0][var].values))], 
                    data[0][var].values, alpha = alpha, s = size, color = "red"); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1][var].values))], 
                    data[1][var].values, alpha = alpha, s = size, color = "gray", label = "WT"); 
    if overlay_RC: # Overlay the dots associated to RC on top of plot
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"], df["RC"]), var].values))], 
                    df.loc[np.logical_and(df["mutant"], df["RC"]), var], alpha = alpha, s = size, color = "green", label = "RC member"); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"] == False, df["RC"]), var].values))], 
                    df.loc[np.logical_and(df["mutant"] == False, df["RC"]), var], alpha = alpha, s = size, color = "green", label = "RC member"); 
    if derivative:
        if aggregation == "std":
            ax.set_ylabel("σ (dV(t)/dt)")
        elif aggregation == "mean":
            ax.set_ylabel("μ (dV(t)/dt)")
        elif aggregation is None:
            ax.set_ylabel("dV(t)/dt")
    else:
        if aggregation == "std":
            ax.set_ylabel("σ [V(t)]")
        elif aggregation == "mean":
            ax.set_ylabel("μ [V(t)]")
        elif aggregation == "CV":
            ax.set_ylabel("CV[V(t)]")
        elif aggregation is None:
            ax.set_ylabel("V(t)")
    ax.set_title(f"{var}, threshold: {threshold}%")
    plt.tight_layout()
    
def boxplot_metric_RC(var, derivative = False, aggregation = None, window = 1, mnn = 4, mutual = False, threshold = 0.0, ax = None):
    """
    Plots a boxplot comparing the specified graph-theoretical metric between RC (rich club) members and non-members. 
    Non distinction of mutant or WT mice is made here. The function aggregates the metric per mouse using the specified method, 
    then plots the distribution for RC members and non-members, and performs a statistical test to indicate significance.
    
    Parameters:
        measure (str): The graph-theoretical metric to plot (e.g., "hub", "degree", "outstrength").
        window (int): The time window size used for averaging the graph representation.
        aggregation: Aggregation method for repeated measures per mouse. Can be None, "mean" or "std".
    """
    
    df = get_metric_df(var, "approaches", derivative, window, mnn, mutual, True, threshold)
    df = df.dropna()
    
    if var == "instrength" or var == "outstrength" or var == "normed outstrength" or var == "normed outstrength":
        df = df[df[f"Timestamp_base{window}"] != 0]  # first time point is strong outlier for strength, for some reason

    if aggregation is not None:
        assert aggregation == "std" or aggregation == "mean", "aggregation should be mean or std"
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: aggregation for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
    data = [df.loc[df["RC"],  ['Mouse_RFID', var]],  df.loc[~df["RC"],  ['Mouse_RFID', var]]]
    
    # convert the dataframe data to a list of np.array that can be passed to plt.boxplot
    numeric_data = [data[i].drop("Mouse_RFID", axis = 1, inplace = False).values for i in range(2)]
    numeric_data = [np.concatenate(d) for d in numeric_data]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 100, 0.5
    bp = ax.boxplot(numeric_data, labels=["RC", "Non-members"], showfliers = False)
    add_significance(data, var, ax, bp)
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(numeric_data[0]))], 
                numeric_data[0], alpha = alpha, s = size, color = "green"); 
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(numeric_data[1]))], 
                numeric_data[1], alpha = alpha, s = size, color = "gray", label = "Non-member"); 
    if derivative:
        if aggregation == "std":
            ax.set_ylabel("σ (dV(t)/dt)")
        elif aggregation == "mean":
            ax.set_ylabel("μ (dV(t)/dt)")
        elif aggregation is None:
            ax.set_ylabel("dV(t)/dt")
    else:
        if aggregation == "std":
            ax.set_ylabel("σ [V(t)]")
        elif aggregation == "mean":
            ax.set_ylabel("μ [V(t)]")
        elif aggregation is None:
            ax.set_ylabel("V(t)")
    ax.set_title(f"{var}")
      
def fluctuation_rank(var, window = 1, mnn = 4, mutual = False, rm_RC = True, overlay_RC = False, threshold = 0.0, ax = None):
    """
     Plots a boxplot comparing the specified graph-theoretical metric between mutant and wild-type mice. By Default, RC members
     are excluded from the comparison unless 'overlay_RC' is True. The function aggregates the metric per mouse using the specified method, 
     then plots the distribution for mutants and wild-type mice, and performs a statistical test to indicate significance.
    
     Parameters:
         measure (str): The graph-theoretical metric to plot (e.g., "hub", "degree", "outstrength").
         window (int): The time window size used for averaging the graph representation.
         aggregation: Aggregation method for repeated measures per mouse. Can be None, "mean" or "std".
     """
     
    df = get_metric_df(var, "approaches", True, window, mnn, mutual, True, threshold)
    df = df.dropna()
    
    if var == "instrength" or var == "outstrength" or var == "normed outstrength" or var == "normed outstrength":
        df = df[df[f"Timestamp_base{window}"] != 0]  # first time point is strong outlier for strength, for some reason

    df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
        'mutant': 'first',  # Keep the first value (or use another aggregation function)
        'RC': 'first',      # Keep the first value (or use another aggregation function)
        **{col: 'std' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
        })
    
    # --- RANKING STEP ---
    df['rank'] = df.groupby('Group_ID')[var].rank(method='first') - 1  # 0-based rank
    # df['rank'] = df.groupby('Group_ID')[var].rank(method='first', ascending=True) - 1
        
    
    # RC are exluded for the comparison, unless overlay RC is True
    if rm_RC:
        data = [df.loc[np.logical_and(df["mutant"], df["RC"] == False),  ['Mouse_RFID', 'rank']],  
                df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False),  ['Mouse_RFID', 'rank']] ]

    else:
        data = [df.loc[df["mutant"],  ['Mouse_RFID', 'rank']],  
                df.loc[df["mutant"] == False,  ['Mouse_RFID', 'rank']] ]
    
    if overlay_RC:
        data = [df.loc[df["mutant"], ['Mouse_RFID', 'rank']],   
                df.loc[df["mutant"] == False, ['Mouse_RFID', 'rank']]]
            
    # convert the dataframe data to a list of np.array that can be passed to plt.boxplot
    numeric_data = [data[i]['rank'].values for i in range(2)]
    #numeric_data = [np.concatenate(d) for d in numeric_data]
            
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    size, alpha = 100, 0.5
    
    if rm_RC:
        bp = ax.boxplot(numeric_data, labels=["Mutants", "Non-members WT"], showfliers = False)
    else:
        bp = ax.boxplot(numeric_data, labels=["Mutants", "WT"], showfliers = False)

    add_significance(data, 'rank', ax, bp)
    if rm_RC:
        mutants = df.loc[np.logical_and(df["mutant"] == True, df["RC"] == False), 'rank'].values
        non_mem = df.loc[np.logical_and(df["mutant"] == False, df["RC"] == False), 'rank'].values
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(mutants))], 
                    mutants, alpha = alpha, s = size, color = "red"); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(non_mem))], 
                    non_mem, alpha = alpha, s = size, color = "gray", label = "Non-member"); 
        fig, ax2 = plt.subplots(1,1)
        ax2.hist(mutants, alpha = 0.5)
        ax2.hist(non_mem, alpha = 0.5)
    elif not rm_RC:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(data[0]['rank'].values))], 
                    data[0]['rank'].values, alpha = alpha, s = size, color = "red"); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(data[1]['rank'].values))], 
                    data[1]['rank'].values, alpha = alpha, s = size, color = "gray", label = "WT"); 
    if overlay_RC: # Overlay the dots associated to RC on top of plot
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"], df["RC"]), 'rank'].values))], 
                    df.loc[np.logical_and(df["mutant"], df["RC"]), 'rank'], alpha = alpha, s = size, color = "green", label = "RC member"); 
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"] == False, df["RC"]), 'rank'].values))], 
                    df.loc[np.logical_and(df["mutant"] == False, df["RC"]), 'rank'], alpha = alpha, s = size, color = "green", label = "RC member"); 

    ax.set_ylabel("Rank")
    ax.set_title(f"Rank of {var} fluctuations\nthreshold: {threshold}%")
    
if __name__ == "__main__":
    # df = mk_measures_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)
    # df = mk_derivatives_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)
    # df = mk_derivatives_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)

    unweighted_features = ["indegree", "outdegree", "degree", "summed cocitation", "summed bibcoupling", 
                          "summed common neighbors", "summed injaccard", "summed outjaccard", "summed jaccard"]
    
    weighted_features = ["eigenvector_centrality", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]
    
    temporal_metrics = ["summed outNEF", "summed inNEF", "summed outICI", 
                        "summed inICI", "summed outburstiness", "summed inburstiness"]

    
# (metrics, variable = "approaches", mnn = None, mutual = False, dim = 2, show_RC = True, std = True)

    # manual_plot( ["summed outburstiness", "summed injaccard", "indegree"], "approaches", 
    #             mnn =  5, mutual = False, dim = 2, show_RC = False)
    
    # manual_plot( ["summed outNEF", "summed injaccard", "indegree"], "approaches", 
    #             mnn =  5, mutual = False, dim = 2, show_RC = False)
    
    # manual_plot( ["summed outburstiness", "summed injaccard", "outdegree"], "approaches", 
    #             mnn =  5, mutual = False, dim = 2, show_RC = False)
    
    # manual_plot( ["indegree", "outdegree", "summed injaccard"], "approaches", 
    #             mnn =  5, mutual = False, dim = None, show_RC = False)
    

    

    # manual_plot( ["summed injaccard", "outdegree", "summed outNEF"], "approaches", 
    #             mnn =  5, mutual = False, dim = None, show_RC = False)
    # manual_plot( ["indegree", "outdegree"], "approaches", 
    #             mnn =  7, mutual = False, dim = None, show_RC = False)
    
        
 
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, ax in enumerate(axs.flatten()):
    #     boxplot_metric_mutant(unweighted_features[idx], derivative = True, aggregation = 'std', window = 1, mnn = 3, mutual = False, rm_RC = True, overlay_RC = False, ax = ax)
        
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, ax in enumerate(axs.flatten()):
    #     boxplot_metric_mutant(unweighted_features[idx], derivative = True, aggregation = 'std', window = 1, mnn = 4, mutual = False, rm_RC = True, overlay_RC = False, ax = ax)    
        
    # for mnn in range(2, 4):
    #     fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    #     for idx, ax in enumerate(axs.flatten()):
    #         boxplot_metric_mutant(unweighted_features[idx], derivative = True, aggregation = 'std', window = 1, mnn = mnn, mutual = True, rm_RC = True, overlay_RC = False, ax = ax)
        
    # mnn = 3

    # boxplot_metric_mutant("summed injaccard", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False, threshold=0)
    # boxplot_metric_mutant("summed outjaccard", derivative = True, aggregation = 'std', window = 1, mnn = 4, mutual = False, rm_RC = True, overlay_RC = False)

    # boxplot_metric_mutant("outdegree", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False, threshold = 0)

    # boxplot_metric_mutant("indegree", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False, threshold = 10)
    # boxplot_metric_mutant("outdegree", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False, threshold = 5)
    
    # boxplot_metric_mutant("instrength", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False)
    # boxplot_metric_mutant("outstrength", derivative = False, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False)


    # boxplot_metric_mutant("out persistence", derivative = False, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False)
    # boxplot_metric_mutant("in persistence", derivative = False, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False)
    
    # boxplot_metric_mutant("outdegree", derivative = True, aggregation = 'std', window = 1, mnn = None,
    #                       mutual = False, rm_RC = True, overlay_RC = False, threshold = 0.0)
    
    # boxplot_metric_mutant("outdegree", derivative = False, aggregation = 'mean', window = 1, mnn = None,
    #                       mutual = False, rm_RC = True, overlay_RC = False, threshold = 0.0)
    
    # boxplot_metric_mutant("indegree", derivative = True, aggregation = 'std', window = 1, mnn = None,
    #                       mutual = False, rm_RC = True, overlay_RC = False, threshold = 5)
    
    # boxplot_metric_mutant("out persistence", derivative = True, aggregation = 'std', window = 1, mnn = None, mutual = False, rm_RC = True, overlay_RC = False)
 
    # boxplot_metric_mutant("eigenvector_centrality", derivative = False, aggregation = None, window = 7, mnn = 4, mutual = False, rm_RC = False, overlay_RC = False)
    # boxplot_metric_mutant("pagerank", derivative = False, aggregation = None, window = 7, mnn = 4, mutual = False, rm_RC = False, overlay_RC = False)
    # boxplot_metric_RC("outstrength", derivative = False, aggregation = None, window = 7, mnn = 4, mutual = False)

    # fluctuation_rank("outstrength", window = 1, mnn = 4, mutual = False, rm_RC = True, overlay_RC = False)
    
    # fluctuation_rank("outdegree", window = 1, mnn = 5, mutual = False, rm_RC = True, overlay_RC = False)
    # fluctuation_rank("outdegree", window = 1, mnn = 6, mutual = False, rm_RC = True, overlay_RC = False)
    # fluctuation_rank("outdegree", window = 1, mnn = None, mutual = False,
                     # rm_RC = True, overlay_RC = False, threshold = 2.5, ax = None)
    # fluctuation_rank("indegree", window = 1, mnn = 7, mutual = False, rm_RC = True, overlay_RC = False)
