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
from tqdm import tqdm

sys.path.append('..\\src\\')
from read_graph import read_graph, read_labels
from graph_metrics import measures

datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def format_plot(ax, bp, xticklabels = ["RC", "Mutants", "Others"]):
    """ Sets the x-axis the RC/mutants/Others, changes the color of the bars in the boxplot."""
    ax.set_xticklabels(xticklabels, fontsize = 20)
    colors = sns.color_palette('pastel')
    for patch, color in zip(bp['boxes'], colors): # set colors
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color='k')
    
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
        bar_height = (y_range * 0.07 * level) + top
        bar_tips = bar_height - (y_range * 0.02)
        plt.plot(
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
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
        

labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]

def plot_measure_timeseries(measure = "hub", window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    plots the timeseries of the specified measurment for all animals, averaging over each animal in the different categories (mut, RC, wt),
    for each day. Meaning, each colored datapoint for a given day represents the average value for all WT, mutants and RC (given the corresponding color).
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
        
def plot_derivative_timeseries(measure = "hub", window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    plots the derivative of timeseries of the specified measurment for all animals, averaging over each animal in the different categories (mut, RC, wt),
    for each day. Meaning, each colored datapoint for a given day represents the average derivative value for all WT, 
    mutants and RC (given the corresponding color).
    """
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_t2 = measures(j, d+2, window, measure, variable, mnn, mutual)
            if scores_t1 is None or scores_t2 is None:
                continue
            scores_mutants.extend(np.array(scores_t2[0]) - np.array(scores_t1[0]))
            scores_rc.extend(np.array(scores_t2[1]) - np.array(scores_t1[1]))
            scores_rest.extend(np.array(scores_t2[2]) - np.array(scores_t1[2]))
            scores_wt.extend(np.array(scores_t2[3]) - np.array(scores_t1[3]))
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
        add_legend = True
        fig, ax = plt.subplots(1, 1)
    else:
        add_legend = False

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
        
    ax.set_title("Time derivative of "+ measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    if add_legend:
        ax.legend()
        
def plot_measure_all(measure, window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    "plots the derivative of the timeseries of the specified measurment for all animals, without any aggregation"
    value_mutants, value_rest, value_rc = [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, _ = [], [], [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_mutants.extend(np.array(scores_t1[0]))
            scores_rest.extend(np.array(scores_t1[2]))
            scores_rc.extend(np.array(scores_t1[1]))

        value_rc.append(scores_rc) 
        value_rest.append(scores_rest) 
        value_mutants.append(scores_mutants) 
        
    t = np.arange(1, 1+15//window)
    value_mutants= np.array(value_mutants)
    value_rest= np.array(value_rest)
    value_rc= np.array(value_rc)

    if ax is None:
        add_legend = True
        fig, ax = plt.subplots(1, 3, sharey=True) if separate_wt else plt.subplots(1, 2, sharey=True)
    else:
        add_legend = False

    if separate_wt:
        ax[0].plot(t, value_mutants[:, :], lw = 3, c = "red", label = "Mutant", alpha = 0.5)
        ax[1].plot(t, value_rest[:, :], lw = 3, c = "k", label = "Non-member WT", alpha = 0.5)
        ax[2].plot(t, value_rc[:, :], lw = 3, c = "green", label = "sRC", alpha = 0.5)
        plt.ylabel("S(t)")
    else:
        ax[0].plot(t, value_mutants[:, :], lw = 3, c = "red", label = "Mutant", alpha = 0.5)
        ax[1].plot(t, value_rest[:, :], lw = 3, c = "k", label = "Non-member WT", alpha = 0.5)

    ax.set_title(measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
        
def plot_derivative_all(measure, window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    "plots the derivative of the timeseries of the specified measurment for all animals, without any aggregation"
    value_mutants, value_rest, value_rc = [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, _ = [], [], [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual)
            scores_t2 = measures(j, d+2, window, measure, variable, mnn, mutual)
            scores_mutants.extend(np.array(scores_t2[0]) - np.array(scores_t1[0]))
            scores_rest.extend(np.array(scores_t2[2]) - np.array(scores_t1[2]))
            scores_rc.extend(np.array(scores_t2[1]) - np.array(scores_t1[1]))

        value_rc.append(scores_rc) 
        value_rest.append(scores_rest) 
        value_mutants.append(scores_mutants) 
        
    t = np.arange(1, 1+15//window)
    value_mutants= np.array(value_mutants)
    value_rest= np.array(value_rest)
    value_rc= np.array(value_rc)


    if ax is None:
        add_legend = True
        fig, ax = plt.subplots(1, 3, sharey=True) if separate_wt else plt.subplots(1, 2, sharey=True)
    else:
        add_legend = False

    if separate_wt:
        ax[0].plot(t, value_mutants[:, :], lw = 3, c = "red", label = "Mutant", alpha = 0.2)
        ax[1].plot(t, value_rest[:, :], lw = 3, c = "k", label = "Non-member WT", alpha = 0.2)
        ax[2].plot(t, value_rc[:, :], lw = 3, c = "green", label = "sRC", alpha = 0.2)
    else:
        ax[0].plot(t, value_mutants[:, :], lw = 3, c = "red", label = "Mutant", alpha = 0.2)
        ax[1].plot(t, value_rest[:, :], lw = 3, c = "k", label = "Non-member WT", alpha = 0.2)
    plt.ylabel("dS(t)/dt")
       # ax.legend()

    ax.set_title("Time derivative of "+ measure, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels

def plot_timeseries_example(measure, derivative = True, window = 3, variable = "approaches", idx = [0, 0, 0],
                            mnn = None, mutual = True, threshold = 0.0, ax = None):
    "plots the derivative of the timeseries of the specified measurment for all animals, without any aggregation"
    value_mutants, value_rest, value_rc = [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    t_window = range(15//window) if measure != "outstrength" else range(1, 15//window)
    for d in t_window:
        scores_mutants, scores_rc, scores_rest, _ = [], [], [], []
        group_mutant, group_rest = [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual, True, threshold)
            if derivative:
                scores_t2 = measures(j, d+2, window, measure, variable, mnn, mutual, True, threshold)
                scores_mutants.extend(np.array(scores_t2[0]) - np.array(scores_t1[0]))
                scores_rest.extend(np.array(scores_t2[2]) - np.array(scores_t1[2]))
                scores_rc.extend(np.array(scores_t2[1]) - np.array(scores_t1[1]))
            else:
                scores_mutants.extend(np.array(scores_t1[0]))
                scores_rest.extend(np.array(scores_t1[2]))
                scores_rc.extend(np.array(scores_t1[1]))
            group_mutant.extend([scores_t1[5]["Group_ID"] for m in range(len(scores_t1[0]))])
            group_rest.extend([scores_t1[5]["Group_ID"] for m in range(len(scores_t1[2]))])

        value_rc.append(scores_rc) 
        value_rest.append(scores_rest) 
        value_mutants.append(scores_mutants) 
        
    t = np.arange(1, 1+15//window)
    value_mutants= np.array(value_mutants)
    value_rest= np.array(value_rest)
    value_rc= np.array(value_rc)
    separate_wt = True if len(idx) == 3 else False
    assert len(idx) == 2 or len(idx) == 3, "Please provide 2 indices for WT vs mutants or 3 indices for RC vs WT vs mutant."

    if ax is None:
        add_legend = True
        fig, ax = plt.subplots(1, 1)
    else:
        add_legend = False

    if separate_wt:
        ax.plot(t, value_mutants[:, idx[0]], lw = 3, c = "red", label = "Mutant", alpha = 0.8)
        ax.plot(t, value_rest[:, idx[1]], lw = 3, c = "k", label = "Non-member WT", alpha = 0.8)
        ax.plot(t, value_rc[:, idx[2]], lw = 3, c = "green", label = "sRC", alpha = 0.8)
    else:
        ax.plot(t, value_mutants[:, idx[0]], lw = 3, c = "red", label = "Mutant", alpha = 0.8)
        print(f"Mutant from group {group_mutant[idx[0]]}")
        print(np.nanstd(value_mutants[:, idx[0]]))
        ax.plot(t, value_rest[:, idx[1]], lw = 3, c = "k", label = "Non-member WT", alpha = 0.8)
        print(f"WT from group {group_rest[idx[1]]}")
        print(np.nanstd(value_rest[:, idx[1]]))
        if not derivative:
            ax.axhline(np.nanmean(value_mutants[:, idx[0]]), ls = "--", color = "red", alpha = 0.8)
            ax.axhline(np.nanmean(value_rest[:, idx[1]]), ls = "--", color = "k", alpha = 0.8)

    
    ylabel = "dS(t)/dt" if derivative else "μ[S(t)]"
    plt.ylabel(ylabel)
    ax.legend()
    title = "Time derivative of "+ measure if derivative else measure
    ax.set_title(title, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    
  
def plot_persistance(out = True, window = 3, variable = "approaches", separate_wt = False, mnn = None, mutual = True, ax = None):
    """
    Computes the average persistance of the animal over all graphs defined in the different cohort.
    The persistance rewards active edges that are preserved, meaning that an absence of edge does NOT 
    influence the value of the persistance. The persistance of a node is the sum of preserved existing edges
    over two time points. Again, an absence of edge does NOT influence this value.
    """
    
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    value_mutants, value_rc, value_rest, value_wt = [], [], [], []
    std_mutants, std_rc, std_rest, std_wt = [], [], [], []
    for d in tqdm(range(1, 15//window)):
        scores_mutants, scores_rc, scores_rest, scores_wt = [], [], [], []
        for j in range(len(labels)):
            if window == 1:
                datapath_t1 = "..\\data\\both_cohorts_1day\\"+labels[j]+"\\"+variable+"_resD1_"+str(d)+".csv"
                datapath_t2 = "..\\data\\both_cohorts_1day\\"+labels[j]+"\\"+variable+"_resD1_"+str(d+1)+".csv"
            elif window == 3:
                datapath_t1 = "..\\data\\both_cohorts_3days\\"+labels[j]+"\\"+variable+"_resD3_"+str(d)+".csv"
                datapath_t2 = "..\\data\\both_cohorts_3days\\"+labels[j]+"\\"+variable+"_resD3_"+str(d+1)+".csv"
            elif window == 7:
                datapath_t1 = "..\\data\\averaged\\"+labels[j]+"\\"+variable+"_resD7_"+str(d)+".csv"
                datapath_t2 = "..\\data\\averaged\\"+labels[j]+"\\"+variable+"_resD7_"+str(d+1)+".csv"
            else:
                print("Incorrect time window")
                return
            try:
                data_t1 = read_graph([datapath_t1], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
                data_t2 = read_graph([datapath_t2], percentage_threshold = 0, mnn = mnn, mutual = mutual)[0]
                arr_t1 = np.loadtxt(datapath_t1, delimiter=",", dtype=str)
                arr_t2 = np.loadtxt(datapath_t2, delimiter=",", dtype=str)
                RFIDs = arr_t1[0, 1:].astype(str)
            except Exception as e:
                continue    
            if variable == "interactions":
                mode = 'undirected'
                data_t1 = (data_t1 + np.transpose(data_t1))/2 # ensure symmetry
                data_t2 = (data_t2 + np.transpose(data_t2))/2 # ensure symmetry
            elif variable == "approaches":
                mode = 'directed'
            else:
                raise NameError("Incorrect input argument for 'variable'.")
    
            if mnn is not None:
                data_t1 = np.where(data_t1 > 0.01, 1, 0) # replacing remaining connections by 1
                data_t2 = np.where(data_t2 > 0.01, 1, 0)
            else:
                data_t1 = np.where(data_t1 > 0.01, data_t1, 0) # replacing remaining connections by 1
                data_t2 = np.where(data_t2 > 0.01, data_t2, 0)
                
            axis = 1 if out else 0
            # persistance = np.sum((2*np.abs(data_t2 - data_t1)/(data_t1+data_t2)) <= 0.5, axis = axis)
            cond = np.abs(data_t2 - data_t1) <= 0.5*data_t1
            persistance = np.sum(cond, axis = axis)
 
            curr_metadata_df = metadata_df.loc[metadata_df["Group_ID"] == int(labels[j][1:]), :]
            # figuring out index of true mutants in current group
            is_mutant = []#[True if curr_metadata_df.loc[metadata_df["Mouse_RFID"] == rfid, "mutant"].values else False for rfid in RFIDs]
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
                    print("Mouse not found, treat as WT")
                    is_mutant.append(False)

            is_RC = [True if curr_metadata_df.loc[curr_metadata_df["Mouse_RFID"] == rfid, "RC"].values else False for rfid in RFIDs] # list of boolean stating which mice are RC
            graph_length = data_t1.shape[0]
            mutants = np.where(is_mutant)[0] if len(np.where(is_mutant)) != 0 else [] # indices of mutants in this group
            rc = np.where(is_RC)[0] if len(np.where(is_mutant)) != 0 else [] # indices of RC in this group
            others = np.arange(graph_length)[np.logical_and(~np.isin(np.arange(graph_length), rc), ~np.isin(np.arange(graph_length), mutants))]
            wt = np.arange(graph_length)[~np.isin(np.arange(graph_length), mutants)]
            
            scores_mutants = [persistance[i] for i in mutants]
            scores_rc = [persistance[i] for i in rc]
            scores_rest = [persistance[i] for i in others]
            scores_wt = [persistance[i] for i in wt]
            
        value_mutants.append(np.nanmean(scores_mutants)) 
        value_rc.append(np.nanmean(scores_rc))
        value_rest.append(np.nanmean(scores_rest))
        value_wt.append(np.nanmean(scores_wt))
        scores_mutants, scores_rc, scores_rest, scores_wt = np.array(scores_mutants), np.array(scores_rc), np.array(scores_rest), np.array(scores_wt)
        std_mutants.append(sem(scores_mutants[~np.isnan(scores_mutants)]))
        std_rc.append(sem(scores_rc[~np.isnan(scores_rc)]))
        std_rest.append(sem(scores_rest[~np.isnan(scores_rest)]))
        std_wt.append(sem(scores_wt[~np.isnan(scores_wt)]))
    
    t = np.arange(1, 15//window)
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
    
    name = "Persistance out" if out else "Persistance in"
    ax.set_title(name, fontsize = 15)
    ax.set_xlabel("Day", fontsize = 13)
    ax.set_ylabel("Value", fontsize = 13)
    ax.set_xticks(range(1, 15//window, 1)) 
    ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust major tick labels
    ax.legend()
    plt.show()
    
    
        
def plot_raster(measure, derivative = True, window = 1, normalize = False, variable = "approaches", separate_wt = False,
                mnn = None, mutual = True, threshold = 0.0, ax = None):
    "plots the derivative of the timeseries of the specified measurment for all animals, without any aggregation"
    value_mutants, value_rest, value_rc = [], [], []
    for d in range(15//window):
        scores_mutants, scores_rc, scores_rest, _ = [], [], [], []
        id_mutants, id_rc, id_rest = [], [], []
        for j in range(len(labels)):
            scores_t1 = measures(j, d+1, window, measure, variable, mnn, mutual, True, threshold)
            if derivative:
                scores_t2 = measures(j, d+2, window, measure, variable, mnn, mutual, True, threshold)
                v_mut = np.abs(np.array(scores_t2[0]) - np.array(scores_t1[0]))
                v_rest = np.abs(np.array(scores_t2[2]) - np.array(scores_t1[2]))
                v_rc = np.abs(np.array(scores_t2[1]) - np.array(scores_t1[1]))
            else:
                v_mut = np.array(np.array(scores_t1[0]))
                v_rest = np.array(np.array(scores_t1[2]))
                v_rc = np.array(np.array(scores_t1[1]))
            scores_mutants.extend(v_mut)
            scores_rest.extend(v_rest)
            scores_rc.extend(v_rc)
            id_mutants.extend([j]*len(v_mut))
            id_rc.extend([j]*len(v_rc))
            id_rest.extend([j]*len(v_rest))

        value_rc.append(scores_rc) 
        value_rest.append(scores_rest) 
        value_mutants.append(scores_mutants) 
        
    # v_max = max([np.nanmax(scores_rc), np.nanmax(scores_rest), np.nanmax(scores_mutants)]) if normalize else 1       

        
    min_t = 1 if "strength" in measure else 0 
    t = np.arange(min_t + 1, 1+15//window)
    value_mutants= np.array(value_mutants)
    value_rest= np.array(value_rest)
    value_rc= np.array(value_rc)
    id_mutants = np.array(id_mutants)
    id_rc = np.array(id_rc)
    id_rest = np.array(id_rest)

    if normalize:
        for idx in np.unique(id_mutants):
            max_mut = max(np.nanstd(value_mutants[:, id_mutants == idx], axis = 0))
            max_rc = max(np.nanstd(value_rc[:, id_rc == idx], axis = 0)) if value_rc[:, id_rc == idx].size != 0 else -np.inf
            max_rest = max(np.nanstd(value_rest[:, id_rest == idx], axis = 0))
            v_max = max([max_mut, max_rc, max_rest])
            value_mutants[:, id_mutants == idx] = value_mutants[:, id_mutants == idx]/v_max
            value_rc[:, id_rc == idx] = value_rc[:, id_rc == idx]/v_max
            value_rest[:, id_rest == idx] = value_rest[:, id_rest == idx]/v_max
    
    total_len = value_mutants.shape[1]+value_rest.shape[1]+value_rc.shape[1] if separate_wt else value_mutants.shape[1]+value_rest.shape[1]
    im = np.zeros((total_len, len(t)))
    counter_y = 0
    for idx in range(value_mutants.shape[1]):
        im[counter_y, :] = np.abs(value_mutants[min_t:, idx])
        counter_y += 1
    for idx in range(value_rest.shape[1]):
        im[counter_y, :] = np.abs(value_rest[min_t:, idx])
        counter_y += 1
    if separate_wt:
        for idx in range(value_rc.shape[1]):
            im[counter_y, :] = np.abs(value_rc[min_t:, idx])
            counter_y += 1
            
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(im**2, cmap = "Reds", aspect='auto')
    # ax.colorbar()
    title = f"{measure} fluctuations (renormalized)" if normalize else f"{measure} fluctuations"
    ax.set_title(title)
    if not separate_wt:
        ax.axhline(value_mutants.shape[1], ls = "--", lw = 3, label = "mutants")
    elif separate_wt:
        ax.axhline(value_mutants.shape[1], ls = "--", lw = 3, label = "mutants")
        ax.axhline(value_mutants.shape[1]+value_rest.shape[1], ls = "--", lw = 3, label = "RC")

    ax.legend()
    # plt.show()
            
if __name__ == "__main__":

    unweighted_features = ["indegree", "outdegree", "transitivity", "summed cocitation", "summed bibcoupling", 
        "summed insubcomponent", "summed outsubcomponent", "summed injaccard", "summed outjaccard"]
    
    weighted_features = ["authority", "hub", "eigenvector_centrality", "constraint", "pagerank", "incloseness", "outcloseness",
        "instrength", "outstrength"]

    mnn = 3
    mutual = False
    var = "interactions"
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, ax in enumerate(axs.flatten()):
    #     try:
    #         plot_measure_timeseries(unweighted_features[idx], 1, 'approaches', False, mnn = mnn, mutual = mutual, ax = ax)
    #         # plot_derivative_timeseries(unweighted_features[idx], 1, 'approaches', False, mnn = mnn, mutual = mutual, ax = ax)
    #     except:
    #         continue
    # # plt.tight_layout()
    # title = f"mnn = {mnn}" if mutual else f"nn = {mnn}"
    # fig.suptitle(title)
    # plot_derivative_timeseries("instrength", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    # for i in range(10):
    # for i in range(53, 62):
    # plot_derivative_example("outdegree", 1, "approaches", [10, 56], None, mutual)
    
    # plot_derivative_example("summed injaccard", 1, "approaches", [16, 66], 3, mutual)
    # plot_derivative_example("outstrength", 1, "approaches", [16, 67], None, mutual)
    # plot_timeseries_example(measure = "out persistence", derivative = False, window = 1, 
    #                         variable = "approaches", idx = [10, 57], mnn = None, mutual = mutual)

    # plot_timeseries_example(measure = "summed injaccard", derivative = True, window = 1, 
    #                        variable = "approaches", idx = [10, 57], mnn = None, mutual = mutual)
    
    # plot_timeseries_example(measure = "instrength", derivative = True, window = 1, 
    #                        variable = "approaches", idx = [30, 92], mnn = None, mutual = mutual)

    # plot_derivative_example("outdegree", 1, "approaches", [16, 67], None, mutual)




    # plot_measure_timeseries("degree", 1, var, False, mnn = mnn, mutual = mutual)
    # plot_derivative_timeseries("outdegree", 1, var, False, mnn = None, mutual = mutual)
    
    # plot_derivative_all("summed injaccard", 1, 'approaches', False, mnn = mnn, mutual = mutual)
    # plot_raster("summed injaccard")
    # plot_raster("outstrength")
    # fig, axs = plt.subplots(3, 3, figsize = (16, 13))
    # for idx, var in enumerate(weighted_features):
    #     plot_raster(var, normalize=True, ax = axs.flatten()[idx], mnn = None, mutual = False)
    # plot_raster("outdegree", normalize=True, mnn = 5, mutual = False)
    # plot_raster("summed injaccard", normalize=True, mnn = 5, mutual = False)
    # plot_derivative_example("outdegree", 1, "approaches", [1, 9], None, False, 1)
    # plot_raster("outdegree", normalize=False, mnn = None, mutual = False, separate_wt=False, threshold = 1)
    # plot_raster("summed injaccard", normalize=False, mnn = 5, mutual = False, separate_wt=True)

    # plot_raster("out persis", normalize=True)

    # plot_raster("instrength")




    