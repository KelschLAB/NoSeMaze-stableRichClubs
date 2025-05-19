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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, normalize
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from tqdm import tqdm 
from sklearn.manifold import TSNE
from sklearn import manifold
sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels
import mpl_toolkits.mplot3d  # noqa: F401
from scipy.stats import ttest_ind
from scipy import stats



datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def rescale(arr):
    if len(arr) > 0:
        normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    else:
        return 
    return normalized_arr

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
         
    elif measure == "out persistance" or measure == "in persistance":
        if window == 1:
            datapath2 = "..\\data\\both_cohorts_1day\\"+labels[graph_idx]+"\\"+variable+"_resD1_"+str(day+1)+".csv"
        elif window == 3:
            datapath2 = "..\\data\\both_cohorts_3days\\"+labels[graph_idx]+"\\"+variable+"_resD3_"+str(day+1)+".csv"
        elif window == 7:
            datapath2 = "..\\data\\averaged\\"+labels[graph_idx]+"\\"+variable+"_resD7_"+str(day+1)+".csv"
        else:
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
     #   cond2 = np.abs(data2 - data) + data2 <= 1.5*data2
      #  persistance = np.sum(np.logical_or(cond1, cond2), axis = axis)
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
    
def SE_and_plot(df_path, dim = 2, nn = 10, subset = None, show_RC = True, avg = False):
    """
    Reads the dataframe, performs spectral embedding dimensionality reduction on selected features,
    and plots the results in 3D, coloring points based on the 'mutant' column.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    if avg:
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: 'std' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
        df = df.dropna()
        
    # Select features from columns 6 to 23 (Python indexing starts at 0, so columns 5 to 22)
    features = df.iloc[:, 5:23]
    if subset is not None:
        features = df.loc[:, subset]
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if dim == 2:
        results = (manifold.SpectralEmbedding(
                n_components=2, n_neighbors=nn
            )
            .fit_transform(features_normalized)
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        if not show_RC:
            ax.scatter(results[~df["mutant"].values, 0], results[~df["mutant"].values, 1],
                                 c="k", alpha=0.9, s = 100, label = "control")
        if show_RC:
            other_idx = np.logical_and(~df["mutant"], ~df["RC"]) 
            RC_idx = np.logical_and(~df["mutant"], df["RC"])
            ax.scatter(results[other_idx, 0], results[other_idx, 1],
                                 c="k", alpha=0.9, s = 100, label = "non-members")
            ax.scatter(results[RC_idx, 0], results[RC_idx, 1],
                                 c="green", alpha=0.9, s = 100, label = "RC")
            
        ax.scatter(results[df["mutant"].values, 0], results[df["mutant"].values, 1],
                             c="red", alpha=0.9, s = 100, label = "mutant")
        

        
    if dim == 3:
        results = (manifold.SpectralEmbedding(
                n_components=3, n_neighbors=nn
            )
            .fit_transform(features_normalized)
        )
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(results[~df["mutant"].values, 0], results[~df["mutant"].values, 1], results[~df["mutant"].values, 2],
                             c="k", alpha=0.9, s = 100, label = "control")
        ax.scatter(results[df["mutant"].values, 0], results[df["mutant"].values, 1], results[df["mutant"].values, 2],
                             c="red", alpha=0.9, s = 100, label = "mutant")
    
    ax.set_title(f'Spectral embedding reduction', fontsize=15)
    ax.set_xlabel('SE1', fontsize=12)
    ax.set_ylabel('SE2', fontsize=12)
    if dim == 3:
        ax.set_zlabel('SE3', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def manual_plot(df_path, dim = 2, variables = None, show_RC = True, avg = True):
    """
    Reads the dataframe, performs spectral embedding dimensionality reduction on selected features,
    and plots the results in 3D, coloring points based on the 'mutant' column.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    
    if avg:
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: 'std' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
        df = df.dropna()
        
    
    results = df[variables].values
    
    pca = PCA(n_components=dim)
    results = pca.fit_transform(results)
    
    # scaler = MaxAbsScaler()
    # results = scaler.fit_transform(results)
    
    if results.shape[1] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        if not show_RC:
            ax.scatter(results[~df["mutant"].values, 0], results[~df["mutant"].values, 1],
                                 c="k", alpha=0.9, s = 100, label = "control")
        if show_RC:
            other_idx = np.logical_and(~df["mutant"], ~df["RC"]) 
            RC_idx = np.logical_and(~df["mutant"], df["RC"])
            ax.scatter(results[other_idx, 0], results[other_idx, 1],
                                 c="k", alpha=0.9, s = 100, label = "non-members")
            ax.scatter(results[RC_idx, 0], results[RC_idx, 1],
                                 c="green", alpha=0.9, s = 100, label = "RC")
            
        ax.scatter(results[df["mutant"].values, 0], results[df["mutant"].values, 1],
                             c="red", alpha=0.9, s = 100, label = "mutant")
        
    if results.shape[1] == 3:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        if not show_RC:
            ax.scatter(results[~df["mutant"].values, 0], results[~df["mutant"].values, 1], results[~df["mutant"].values, 2],
                                 c="k", alpha=0.9, s = 100, label = "control")
        if show_RC:
            other_idx = np.logical_and(~df["mutant"], ~df["RC"]) 
            RC_idx = np.logical_and(~df["mutant"], df["RC"])
            ax.scatter(results[other_idx, 0], results[other_idx, 1], results[other_idx, 2],
                                 c="k", alpha=0.9, s = 100, label = "non-members")
            ax.scatter(results[RC_idx, 0], results[RC_idx, 1], results[RC_idx, 2],
                                 c="green", alpha=0.9, s = 100, label = "RC")
            
        ax.scatter(results[df["mutant"].values, 0], results[df["mutant"].values, 1], results[df["mutant"].values, 2],
                             c="red", alpha=0.9, s = 100, label = "mutant")
    
    ax.set_xlabel(f"Transformed {variables[0]}", fontsize=12)
    ax.set_ylabel(f"Transformed {variables[1]}", fontsize=12)
    if len(variables) == 3:
        ax.set_zlabel(f"Transformed {variables[2]}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


def statistic(x, y):
    x = np.concatenate(x)
    y = np.concatenate(y)
    # return np.mean(x) - np.mean(y)
    return np.median(x) - np.median(y)


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
    grouped_data2 = [list(d) for d in grouped_data2]
    
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

    if p < 0.05:
        x1 = 1
        x2 = 2
        bottom, top = ax.get_ylim()
        y_range = top - bottom
        bar_height = (y_range * 0.05 * 1) + top
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
        text_height = bar_height + (y_range * 0.01)
        ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
    
def plot_std(df_path, var, std = False, show_RC = False, sep_RC = False, ax = None):
    # Read the dataframe
    df = pd.read_csv(df_path)
    # df = df.dropna()

    # isolating the variable of interest in order not to drop rows with valid data with .dropna()
    df = df.loc[:, ["mutant", "Timestamp_base1", "Group_ID", "RC", "Mouse_RFID", var]]
    df = df.dropna()
    
    if var == "instrength" or var == "outstrength" or var == "normed outstrength" or var == "normed outstrength":
        df = df[df["Timestamp_base1"] != 0] 
        #df = df[df["Timestamp_base1"] != 1] 
        # df = df[df["Timestamp_base1"] != 2] 


    if std:
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: 'std' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
        # df = df.dropna()
        # if var == "instrength" or var == "outstrength":
        #     df = df.loc[df[var] < 170, :]

    data = [
        df.loc[df["mutant"], ['Mouse_RFID', var]],  
        df.loc[np.logical_and(~df["mutant"], ~df["RC"]), ['Mouse_RFID', var]]  
    ]
    
    if show_RC:
        data = [
            df.loc[df["mutant"], ['Mouse_RFID', var]],  
            df.loc[~df["mutant"], ['Mouse_RFID', var]] 
        ]
        
    if sep_RC:
        data = [
            df.loc[df["RC"], ['Mouse_RFID', var]], 
            df.loc[~df["RC"], ['Mouse_RFID', var]]  
        ]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        
    size = 100
    alpha = 0.5
    numeric_data = [data[i].drop("Mouse_RFID", axis = 1, inplace = False).values for i in range(2)]
    numeric_data = [np.concatenate(d) for d in numeric_data]
    if not sep_RC:
        bp = ax.boxplot(numeric_data, labels=["Mutants", "Non-members WT"], showfliers = False)
    elif sep_RC:
        bp = ax.boxplot(numeric_data, labels=["RC", "Non-members"], showfliers = False)

    if not sep_RC:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[df["mutant"],var].values))], 
                    df.loc[df["mutant"],var], alpha = alpha, s = size, color = "red"); 
        
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(~df["mutant"], ~df["RC"]), var].values))], 
                    df.loc[np.logical_and(~df["mutant"], ~df["RC"]), var], alpha = alpha, s = size, color = "gray", label = "Non-member"); 
        
        if show_RC:
            ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(df["mutant"], df["RC"]), var].values))], 
                        df.loc[np.logical_and(df["mutant"], df["RC"]), var], alpha = alpha, s = size, color = "green", label = "RC member"); 
            
            ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(~df["mutant"], df["RC"]), var].values))], 
                        df.loc[np.logical_and(~df["mutant"], df["RC"]), var], alpha = alpha, s = size, color = "green", label = "RC member"); 
    elif sep_RC:
        ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[df["RC"],var].values))], 
                    df.loc[df["RC"],var], alpha = alpha, s = size, color = "green"); 
        
        ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[~df["RC"], var].values))], 
                    df.loc[~df["RC"], var], alpha = alpha, s = size, color = "gray", label = "Non-member"); 
        
    
    
    add_significance(data, var, ax, bp)

    ax.set_ylabel("σ (dV(t)/dt)")
    # ax.set_yscale("log")
    ax.set_title(f"{var}")
   
    
def train_mlp_classifier(df_path, pred_var = "mutant", subset=None, avg = False):
    """
    Trains a simple feed-forward neural network using sklearn's MLPClassifier.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
        subset (list): Optional list of feature columns to use.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    df = df.dropna()
    # df = df.fillna(-1)
    
    if avg:
        df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
          'mutant': 'first',  # Keep the first value (or use another aggregation function)
          'RC': 'first',      # Keep the first value (or use another aggregation function)
          **{col: 'std' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
          })
        # df = df.dropna()
    
    
    # Select features (columns 6 to 23) and target ('mutant')
    X = df.iloc[:, 5:23]
    if subset is not None:
        X = df.loc[:, subset]
    y = df[pred_var]
    
    accuracies, mlps = [], []
    
    # rus = RandomUnderSampler()
    # X_balanced, y_balanced = rus.fit_resample(X, y)

    
    for i in tqdm(range(20)):
        # rus = RandomUnderSampler()
        # X_balanced, y_balanced = rus.fit_resample(X, y)
        X_i, X_gt, y_i, y_gt = train_test_split(X, y, stratify = y, test_size=0.05)
        rus = RandomUnderSampler()
        X_gt, y_gt = rus.fit_resample(X_gt, y_gt)
        
        # smote = SMOTE()
        X_train, y_train = rus.fit_resample(X_i, y_i)
        
        # X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.01)
        
        # Normalize the features
        scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_gt)
        X_test = X_gt
        
        # Train a feed-forward neural network
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), solver = "lbfgs", activation = "tanh", max_iter=3000)
        mlp.fit(X_train, y_train)
        mlps.append(mlp)
        y_pred = mlp.predict(X_test)
    
        accuracies.append(accuracy_score(y_gt, y_pred))
        
    print(f"aveaged accuracy = {np.mean(accuracies)}")
    
    return #mlps[np.argmax(accuracies)], X_test, y_gt, features.to_list()

def evaluate_feature_importance(mlp, X_test, y_test, feature_names):
    result = permutation_importance(mlp, X_test, y_test, n_repeats=10, random_state=42)
    importance = result.importances_mean
    std = result.importances_std
    sorted_idx = np.argsort(importance)[::-1]  # Sort in descending order of importance

    # Ensure alignment of indices
    y_positions = range(len(sorted_idx))  # Y-axis positions for the bars

    # Plot the horizontal bar chart
    plt.barh(y_positions, importance[sorted_idx], xerr=std[sorted_idx], align='center')
    plt.yticks(y_positions, [feature_names[i] for i in sorted_idx])  # Set feature names as y-ticks
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # df = mk_measures_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)
    # df = mk_derivatives_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)
    # df = mk_derivatives_dataframe("approaches", 1, 5, mutual = False, rm_weak_histo = True)
    # run_tsne_and_plot("..\\data\\graph_features_approaches_d3.csv", 3)
    # SE_and_plot("..\\data\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", 2, 30,  "mutant", None, True)
    # MDS_and_plot("..\\data\\graph_features_approaches_d3_nn4.csv", 3, "mutant")
    # LLE_and_plot("..\\data\\graph_features_approaches_d3.csv", 2, 100, "mutant")


    
    unweighted_features = ["indegree", "outdegree", "degree", "summed cocitation", "summed bibcoupling", 
                          "summed common neighbors", "summed injaccard", "summed outjaccard", "summed jaccard"]
    
    weighted_features = ["eigenvector_centrality", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]
    
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, ax in enumerate(axs.flatten()):
    #     plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn3_rm_weak.csv", unweighted_features[idx], True, False, False, ax)
    #     # plot_std("..\\data\\processed_metrics\\graph_features_approaches_d1_nn5_rm_weak.csv", unweighted_features[idx], True, False, False, ax)
    # # plt.suptitle("graph_features_approaches_d1_nn5_rm_weak")
    # plt.suptitle("metrics_derivative_approaches_d1_nn3_rm_weak")

    # SE_and_plot("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", 2, 7, ["indegree", "summed jaccard", "summed common neighbors", "transitivity"], True, True)
    # SE_and_plot("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", 2, 6, unweighted_features, True, True)

    # manual_plot("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", 2, ["degree", "summed jaccard", "instrength"], True, True)


    fig, axs = plt.subplots(1, 2, figsize = (10, 10))
    plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn3_rm_weak.csv", "summed injaccard", std =True, show_RC=False, sep_RC =False, ax =axs[0])#,subset)
    plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn3_rm_weak.csv", "summed injaccard", True, True, False, axs[1])#,subset)
        
    # plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "instrength", True, False, False, axs[0, 0])#,subset)
    # plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "outstrength", True, False, False, axs[0, 1])#,subset)
    # plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "instrength", True, True, False, axs[1, 0])#,subset)
    # plot_std("..\\data\\processed_metrics\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "outstrength", True, True, False, axs[1, 1])#,subset)
    plt.suptitle("nn cut = 3, permutation test on the mean")


    # print("############################### prediction on metrics ts #######################################")
    # train_mlp_classifier("..\\data\\processed_metrics\graph_features_approaches_d1_nn5_rm_weak.csv", "mutant", unweighted_features, True)
    # print("############################### prediction on derivative ts #######################################")
    # train_mlp_classifier("..\\data\\processed_metrics\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "mutant", unweighted_features, True)

    # mlp, x_test, y_test, feature_names = train_mlp_classifier("..\\data\\graph_features_approaches_d3_nn5_rm_weak.csv", "mutant", subset, False)
    # evaluate_feature_importance(mlp, x_test, y_test, feature_names)

    