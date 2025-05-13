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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm 
from sklearn.manifold import TSNE
from sklearn import manifold
sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels
import mpl_toolkits.mplot3d  # noqa: F401

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def measures(graph_idx, day, window = 3, measure = "hub", variable = "approaches", mnn = None, mutual = True, rm_weak_histo = True):
    """
    Returns the measurement specified by input argument for mutants, rc and in wt in graph specified by input index.
    graph_idx specifies the cohort number, day specifies the specific timepoint (depends on window) and window specifies 
    the size of the window of averaging for the graph representation.        
    """
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    if window == 1:
        datapath = "..\\data\\both_cohorts_1day\\"+labels[graph_idx]+"\\"+variable+"_resD1_"+str(day)+".csv"
    elif window == 3:
        datapath = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"+variable+"_resD3_"+str(day)+".csv"
    elif window == 7:
        datapath = "..\\data\\averaged\\"+labels[graph_idx]+"\\"+variable+"_resD7_"+str(day)+".csv"
    else:
        print("Incorrect time window")
        return
    try:
        data = read_graph([datapath], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
        arr = np.loadtxt(datapath, delimiter=",", dtype=str)
        RFIDs = arr[0, 1:].astype(str)
    except Exception as e:
        print(e)
        return None
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
    rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
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
        scores_mutants = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in mutants]
        scores_rc = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in rc]
        scores_rest = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in others]
        scores_wt = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in wt]  
        scores_all = [np.sum(g.similarity_jaccard(mode="in")[i]) for i in range(len(RFIDs))]

    elif measure == "summed outjaccard":
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

    elif measure == "closeness":
         scores_mutants = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in mutants]
         scores_rc = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in rc]
         scores_rest = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in others]  
         scores_wt = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in wt]  
         scores_all = [g.closeness(weights = [1/e['weight'] for e in g.es()])[i] for i in range(len(RFIDs))]

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

    elif measure == "transitivity":
         scores_mutants = [g.transitivity_local_undirected()[i] for i in mutants]
         scores_rc = [g.transitivity_local_undirected()[i] for i in rc]
         scores_rest = [g.transitivity_local_undirected()[i] for i in others]  
         scores_wt = [g.transitivity_local_undirected()[i] for i in wt]  
         scores_all = [g.transitivity_local_undirected()[i] for i in range(len(RFIDs))]

    else:
        raise Exception("Unknown or misspelled input measurement.") 

    return scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}

def plot_measure_timeseries(measure = "hub", window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    Box plot of graph theory measurement specified by input argument for all graphs in dataset, separated in mutants and WT, or 
    mutants, RC and others (using separate_wt argument).
    Inputs:
        - measure: the type of graph measurment to display. Default to hub score.
        - window: how many days of averaging should be used to display the results. 
            Defaults to 3, meaning one graph is the average of three days of experiment. 
        - variable: which kind of variable should be displayed. Shoud be "interactions" or "approaches".
        - separate_wt: whether or not the display of WT should be separated in the RC and others.
    """
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            scores = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_mutants.extend(scores[0])
            scores_rc.extend(scores[1])
            scores_rest.extend(scores[2])
            scores_wt.extend(scores[3])
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        scores_mutants, scores_rc, scores_rest, scores_wt = np.array(scores_mutants), np.array(scores_rc), np.array(scores_rest), np.array(scores_wt)
        std_mutants.append(sem(scores_mutants[~np.isnan(scores_mutants)]))
        std_rc.append(sem(scores_rc[~np.isnan(scores_rc)]))
        std_rest.append(sem(scores_rest[~np.isnan(scores_rest)]))
        std_wt.append(sem(scores_wt[~np.isnan(scores_wt)]))
    t = np.arange(1, 1+15//window)
    value_mutants, value_rc, value_rest, value_wt = np.array(value_mutants), np.array(value_rc), np.array(value_rest), np.array(value_wt)
    std_mutants, std_rc, std_rest, std_wt = np.array(std_mutants), np.array(std_rc), np.array(std_rest), np.array(std_wt) 
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    if separate_wt:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_rc, lw = 2, c = "green", label = "RC members")
        ax.fill_between(t, value_rc - std_rc, value_rc + std_rc, color="green", alpha=0.2)
        ax.plot(t, value_rest, lw = 2, c = "blue", label = "Others")
        ax.fill_between(t, value_rest - std_rest, value_rest + std_rest, color="blue", alpha=0.2)
    else:
        ax.plot(t, value_mutants, lw = 2, c = "red", label = "Mutants")
        ax.fill_between(t, value_mutants - std_mutants, value_mutants + std_mutants, color="red", alpha=0.2)
        ax.plot(t, value_wt, lw = 2, c = "green", label = "WT")
        ax.fill_between(t, value_wt - std_wt, value_wt + std_wt, color="green", alpha=0.2)
        
    ax.set_title(measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window+1, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    if ax is None:
        ax.legend()

    
def mk_measures_dataframe(variable = "approaches", window = 3, mnn = 4, mutual = False, rm_weak_histo = True):
    """
    Formats the measures made on the graphs into a pandas dataframe and saves to a csv file
    """

    unweighted_features = ["indegree", "outdegree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]
    
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    
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
    features_df.to_csv(f"..\\data\\{save_name}", index = False)
    return features_df

def run_tsne_and_plot(df_path, dim = 2):
    """
    Reads the dataframe, performs t-SNE dimensionality reduction on selected features,
    and plots the results in 3D, coloring points based on the 'mutant' column.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 5:23]
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        # Perform t-SNE dimensionality reduction to 3 components
        tsne = TSNE(n_components=2, perplexity=90, max_iter=500, init = 'random')
        results = tsne.fit_transform(features)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df['mutant'], alpha=0.9)
        
    if dim == 3:
        # Perform t-SNE dimensionality reduction to 3 components
        tsne = TSNE(n_components=3, perplexity=30, max_iter=500, init = 'random')
        results = tsne.fit_transform(features)
        
        # Plot the t-SNE results in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                             c=df['mutant'], alpha=0.9)
    
    ax.set_title('t-SNE reduction', fontsize=15)
    ax.set_xlabel('TSNE1', fontsize=12)
    ax.set_ylabel('TSNE2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('TSNE3', fontsize=12)
    
    plt.show()
    
def SE_and_plot(df_path, dim = 2, nn = 10, color_var = "mutant"):
    """
    Reads the dataframe, performs spectral embedding dimensionality reduction on selected features,
    and plots the results in 3D, coloring points based on the 'mutant' column.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 5:23]
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        results = (manifold.SpectralEmbedding(
                n_components=2, n_neighbors=nn
            )
            .fit_transform(features_normalized)
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df[color_var], alpha=0.9)
        
    if dim == 3:
        results = (manifold.SpectralEmbedding(
                n_components=3, n_neighbors=nn
            )
            .fit_transform(features_normalized)
        )
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                             c=df[color_var], alpha=0.9)
    
    ax.set_title(f'Spectral embedding reduction\ncoloring {color_var}', fontsize=15)
    ax.set_xlabel('SE1', fontsize=12)
    ax.set_ylabel('SE2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('SE3', fontsize=12)
    
    plt.show()
    
def MDS_and_plot(df_path, dim = 2, color_var = "mutant"):
    """
    Reads the dataframe, performs MDS dimensionality reduction on selected features,
    and plots the results in 3D, coloring points based on the 'mutant' column.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 5:23]
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        mds = manifold.MDS(2, max_iter=100, n_init=1)
        results= mds.fit_transform(features_normalized)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df[color_var], alpha=0.9)
        
    if dim == 3:
        mds = manifold.MDS(3, max_iter=100, n_init=1)
        results= mds.fit_transform(features_normalized)
             
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                             c=df[color_var], alpha=0.9)
    
    ax.set_title(f'MDS reduction\ncoloring {color_var}', fontsize=15)
    ax.set_xlabel('MDS1', fontsize=12)
    ax.set_ylabel('MDS2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('MDS3', fontsize=12)
    
    plt.show()
    
def isomap_and_plot(df_path, dim = 2, nn = 10, color_var = "mutant", subset = None):
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 5:23]
    if subset is not None:
        features = df.loc[:, subset]
    

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        results = (
            manifold.Isomap(n_neighbors=nn, n_components=2)
            .fit_transform(features_normalized)
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df[color_var], alpha=0.9)
        
    if dim == 3:
        results = (
            manifold.Isomap(n_neighbors=nn, n_components=3)
            .fit_transform(features_normalized)
        )
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                             s = 100, c=df[color_var], alpha=0.9)
    
    ax.set_title(f'Isomap reduction\ncoloring {color_var}', fontsize=15)
    ax.set_xlabel('iso 1', fontsize=12)
    ax.set_ylabel('iso 2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('iso 3', fontsize=12)
    
    plt.show()
    
def LLE_and_plot(df_path, dim = 2, nn = 10, color_var = "mutant", subset = None):
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 14:23]
    if subset is not None:
        features = df.loc[:, subset]
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        results = (
            manifold.LocallyLinearEmbedding(n_neighbors=nn, method = "standard", n_components=2)
            .fit_transform(features_normalized)
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df[color_var], alpha=0.9)
        
    if dim == 3:
        results = (
            manifold.LocallyLinearEmbedding(n_neighbors=nn, method = "standard", n_components=3)
            .fit_transform(features_normalized)
        )
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                             s = 100, c=df[color_var], alpha=0.9)
    
    ax.set_title(f'LLE reduction\ncoloring {color_var}', fontsize=15)
    ax.set_xlabel('LLE  1', fontsize=12)
    ax.set_ylabel('LLE  2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('LLE  3', fontsize=12)
    
    plt.show()
    
def train_mlp_classifier(df_path, pred_var = "mutant", subset=None):
    """
    Trains a simple feed-forward neural network using sklearn's MLPClassifier.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
        subset (list): Optional list of feature columns to use.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    # Select features (columns 6 to 23) and target ('mutant')
    X = df.iloc[:, 5:23]
    if subset is not None:
        X = df.loc[:, subset]
    y = df[pred_var]
    
    accuracies, mlps = [], []

    for i in tqdm(range(10)):
        rus = RandomUnderSampler()
        X_balanced, y_balanced = rus.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train a feed-forward neural network
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        mlps.append(mlp)
        y_pred = mlp.predict(X_test)
    
        accuracies.append(accuracy_score(y_test, y_pred))
        
    print(f"aveaged accuracy = {np.mean(accuracies)}")
    
    return mlps[np.argmax(accuracies)], X_test, y_test, X.columns.to_list()

def evaluate_feature_importance(mlp, X_test, y_test, feature_names):
    result = permutation_importance(mlp, X_test, y_test, n_repeats=10, random_state=42)
    importance = result.importances_mean
    std = result.importances_std
    sorted_idx = np.argsort(importance)[::-1]
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], xerr = std[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # df = mk_measures_dataframe("approaches", 3, 5, False, False)
    # run_tsne_and_plot("..\\data\\graph_features_approaches_d3.csv", 3)
    # SE_and_plot("..\\data\\graph_features_approaches_d3_nn4.csv", 3, 30, "RC")
    # MDS_and_plot("..\\data\\graph_features_approaches_d3_nn4.csv", 3, "mutant")
    # isomap_and_plot("..\\data\\graph_features_approaches_d3.csv", 3, 50, "mutant")
    # LLE_and_plot("..\\data\\graph_features_approaches_d3.csv", 2, 100, "mutant")
    # subset =  ["transitivity", "summed bibcoupling", "summed outjaccard", "summed insubcomponent", "summed outsubcomponent", "outdegree"]
    # isomap_and_plot("..\\data\\graph_features_approaches_d3_nn4_all_muts.csv", 3, 50, "RC")#,subset)

    mlp, x_test, y_test, feature_names = train_mlp_classifier("..\\data\\graph_features_approaches_d3_nn4.csv", "mutant")#, subset)
    evaluate_feature_importance(mlp, x_test, y_test, feature_names)

    