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
         
    elif measure == "out persistance":
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
        # For computing persistance
        if mnn is not None:
            data = np.where(data > 0.01, 1, 0) # replacing remaining connections by 1
            data2 = np.where(data2 > 0.01, 1, 0)
        else:
            data= np.where(data > 0.01, data, 0) # replacing remaining connections by 1
            data2 = np.where(data2 > 0.01, data2, 0)
            
        axis = 1 
        persistance = np.sum((2*np.abs(data2 - data)/(data+data2)) <= 0.5, axis = axis)
        scores_mutants = [persistance[i] for i in mutants]
        scores_rc = [persistance[i] for i in rc]
        scores_rest = [persistance[i] for i in others]  
        scores_wt = [persistance[i] for i in wt]  
        scores_all = [persistance[i] for i in range(len(RFIDs))]
        
    else:
        raise Exception("Unknown or misspelled input measurement.") 

    return scores_mutants, scores_rc, scores_rest, scores_wt, scores_all, {"Mouse_RFID": RFIDs, "mutant": is_mutant, "RC": is_RC, "Group_ID": int(labels[graph_idx][1:])}
    
def mk_measures_dataframe(variable = "approaches", window = 3, mnn = 4, mutual = False, rm_weak_histo = True, avg = False):
    """
    Formats the measures made on the graphs into a pandas dataframe and saves to a csv file
    """

    unweighted_features = ["indegree", "outdegree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard", "out persistance"]
    
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

def mk_derivatives_dataframe(variable = "approaches", window = 3, mnn = 4, mutual = False, rm_weak_histo = True, avg = False):
    """
    Formats the derivative of the measures made on the graphs as a function of time into a pandas dataframe and saves to a csv file
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
                res_t1 = measures(j, d+1, window, feature, variable, None, True, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                res_t2 = measures(j, d+2, window, feature, variable, None, True, rm_weak_histo) # weighted edge measures do not need graph cut : mnn = None
                if res_t1 is None or res_t2 is None:
                    continue
                scores_t1, metadata = np.array(res_t1[4]), res_t2[5]
                scores_t2, _ = np.array(res_t2[4]), res_t1[5]
                scores = scores_t2 - scores_t1
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
    features_df.to_csv(f"..\\data\\{save_name}", index = False)
    return features_df

def run_tsne_and_plot(df_path, dim = 2, pred_var = "mutant", subset = None, avg = True):
    """
    Reads the dataframe, performs t-SNE dimensionality reduction on selected features,
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
        # Perform t-SNE dimensionality reduction to 3 components
        tsne = TSNE(n_components=2, perplexity=30, n_iter = 500)
        results = tsne.fit_transform(features)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(results[:, 0], results[:, 1],
                             c=df['mutant'], alpha=0.9)
        
    if dim == 3:
        # Perform t-SNE dimensionality reduction to 3 components
        tsne = TSNE(n_components=3, perplexity=30, n_iter = 500)
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
    
def SE_and_plot(df_path, dim = 2, nn = 10, color_var = "mutant", subset = None, avg = False):
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
    features_normalized = features# scaler.fit_transform(features)
    
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
    

    

    
def isomap_and_plot(df_path, dim = 2, nn = 10, color_var = "mutant", subset = None, avg = False):
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
    features_normalized = features#scaler.fit_transform(features)
    
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

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def add_significance(data, ax, bp):
    """Computes and adds p-value significance to input ax and boxplot"""
    # Initialise a list of combinations of groups that are significantly different
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for combination in combinations:
        data1 = data[combination[0] - 1]
        data2 = data[combination[1] - 1]
        # Significance
        # U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        # U, p = stats.ttest_ind(data1, data2,equal_var=False)
        result = stats.permutation_test((data1, data2), statistic)
        U = result.statistic  # This gives the test statistic
        p = result.pvalue      # This gives the p-value
        print(p)
        if p < 0.05:
            significant_combinations.append([combination, p])
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        # Get the y-axis limits
        bottom, top = ax.get_ylim()
        y_range = top - bottom
        bar_height = (y_range * 0.05 * level) + top
        bar_tips = bar_height - (y_range * 0.02)
        ax.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
        )
        # Significance level
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (y_range * 0.01)
        ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
    
def plot_std(df_path, var, avg = False, ax = None):
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
        
    data = [
        df.loc[df["mutant"], var].values,  # Mutants
        df.loc[np.logical_and(~df["mutant"], ~df["RC"]), var].values  # Non-mutants
    ]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        
    size = 100
    alpha = 0.5
    bp = ax.boxplot(data, labels=["Mutants", "Non-members WT"], showfliers = False)
    
    t_stat, p_value = ttest_ind(data[0], data[1], equal_var=False)


    # # standard deviation of mutants across time
    ax.scatter([1 + np.random.normal()*0.05 for i in range(len(df.loc[df["mutant"],var].values))], 
                df.loc[df["mutant"],var], alpha = alpha, s = size, color = "red"); 
    
    ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(~df["mutant"], ~df["RC"]), var].values))], 
                df.loc[np.logical_and(~df["mutant"], ~df["RC"]), var], alpha = alpha, s = size, color = "gray", label = "Non-member"); 
    
    # ax.scatter([2 + np.random.normal()*0.05 for i in range(len(df.loc[np.logical_and(~df["mutant"], df["RC"]), var].values))], 
    #             df.loc[np.logical_and(~df["mutant"], df["RC"]), var], alpha = alpha, s = size, color = "green", label = "RC member"); 
    
    add_significance(data, ax, bp)

    # ax.set_ylabel("σ (V(t)/dt)")
    ax.set_ylabel("σ (V(t))")

    ax.set_title(f"{var}")
   
    
def train_mlp_classifier(df_path, window = 3, pred_var = "mutant", subset=None, avg = False):
    """
    Trains a simple feed-forward neural network using sklearn's MLPClassifier.
    
    Args:
        df_path (str): Path to the CSV file containing the dataframe.
        subset (list): Optional list of feature columns to use.
    """
    # Read the dataframe
    df = pd.read_csv(df_path)
    # df = df.dropna()
    df = df.fillna(-1)
    
    # if avg:
    #     df = df.groupby(['Mouse_RFID', 'Group_ID'], as_index=False).agg({
    #       'mutant': 'first',  # Keep the first value (or use another aggregation function)
    #       'RC': 'first',      # Keep the first value (or use another aggregation function)
    #       **{col: 'mean' for col in df.select_dtypes(include='number').columns if col not in ['Group_ID']}
    #       })
        
     # Reshape the data: Group by Mouse_RFID and Group_ID
    if subset is not None:
        features = subset
    else:
        features = df.columns[5:23]  # Default feature columns (adjust as needed)
    grouped = df.groupby(['Mouse_RFID', 'Group_ID'])
    X = []
    y = []
    for (rfid, group_id), group_data in grouped:
        tmp_X = []
        for timepoint in range(15//window):
            features_t = group_data.loc[group_data["Timestamp_base"+str(window)] == timepoint, features]
            if len(features_t) > 0: # if the data for this day exists, add all features value
                tmp_X.append(list(features_t.values.flatten()))
            else: # if no data was available for this day, add non informative value to keep size consistent
                continue
              #  tmp_X.extend([-1 for i in range(len(features))])
              
        X.append(np.std(np.array(tmp_X), axis = 0))
        y.append(group_data[pred_var].iloc[0])  # Use the first value of the target (assumes it's consistent)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    

    
    # Select features (columns 6 to 23) and target ('mutant')
    # X = df.iloc[:, 5:23]
    # if subset is not None:
    #     X = df.loc[:, subset]
    # y = df[pred_var]
    
    accuracies, mlps = [], []
    
    # rus = RandomUnderSampler()
    # X_balanced, y_balanced = rus.fit_resample(X, y)
    X, X_gt, y, y_gt= train_test_split(X, y, test_size=0.2)
    print(f"Portion of mutants in GT : {np.sum(y_gt)/len(y_gt)}")
    rus = RandomUnderSampler()
    X_gt, y_gt = rus.fit_resample(X_gt, y_gt)
    
    for i in tqdm(range(20)):
        # rus = RandomUnderSampler()
        # X_balanced, y_balanced = rus.fit_resample(X, y)
        
        
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X, y)
        
        # X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.01)
        
        # Normalize the features
        scaler = StandardScaler()
       # X_train = scaler.fit_transform(X_train)
       # X_test = scaler.transform(X_gt)
        X_test = X_gt
        
        # Train a feed-forward neural network
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 50, 25, 25), activation = "relu", max_iter=1000, random_state=42)
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
    # df = mk_measures_dataframe("approaches", 1, 3, False, True)
    # df = mk_derivatives_dataframe("approaches", 1, 3, False, True)
    # run_tsne_and_plot("..\\data\\graph_features_approaches_d3.csv", 3)
    # SE_and_plot("..\\data\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", 2, 30,  "mutant", None, True)
    # MDS_and_plot("..\\data\\graph_features_approaches_d3_nn4.csv", 3, "mutant")
    # LLE_and_plot("..\\data\\graph_features_approaches_d3.csv", 2, 100, "mutant")
    subset = ["summed outsubcomponent"]#, "transitivity", "outdegree"] #"outdegree", "outcloseness", "summed outsubcomponent"
    
    unweighted_features = ["indegree", "outdegree", "summed cocitation", "summed bibcoupling", 
                           "summed injaccard", "summed outjaccard"]
    
    weighted_features = ["eigenvector_centrality", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]
    
    fig, axs = plt.subplots(2, 3, figsize = (16, 13))
    for idx, ax in enumerate(axs.flatten()):
        # plot_std("..\\data\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", weighted_features[idx], True, ax)
        plot_std("..\\data\\graph_features_approaches_d1_nn5_rm_weak.csv", unweighted_features[idx], True, ax)
    plt.suptitle("graph_features_approaches_d1_nn5_rm_weak")


    # plot_std("..\\data\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "indegree", True)#,subset)
    # plot_std("..\\data\\metrics_derivative_approaches_d1_nn5_rm_weak.csv", "summed injaccard", True)#,subset)


    # print("############################### prediction on metrics ts #######################################")
    # train_mlp_classifier("..\\data\\graph_features_approaches_d1_nn3_rm_weak.csv", 1, "mutant", None, True)
    # print("############################### prediction on derivative ts #######################################")
    # train_mlp_classifier("..\\data\\metrics_derivative_approaches_d1_nn3_rm_weak.csv",1, "mutant", None, True)

    # mlp, x_test, y_test, feature_names = train_mlp_classifier("..\\data\\graph_features_approaches_d3_nn5_rm_weak.csv", "mutant", subset, False)
    # evaluate_feature_importance(mlp, x_test, y_test, feature_names)

    