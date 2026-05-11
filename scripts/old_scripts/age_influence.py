## script for doing sanity check of the mice distribution in the two cohorts, i.e. checking that the age disparity or mutant distribution does not affect too much

import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import pandas as pd
import os
import sys
import seaborn as sns
from scipy.stats import ttest_ind, ttest_1samp
from scipy import stats
import seaborn as sns
from tqdm import tqdm
sys.path.append('..\\src\\')
from read_graph import read_graph
from utils import add_significance as bp_sig
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


datapath = "..\\data\\chasing\\single\\"
datapath = "..\\data\\averaged\\"
#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Arial"

def statistic(x, y):
    x = np.concatenate(x)
    y = np.concatenate(y)
    return np.mean(x) - np.mean(y)
    # return np.median(x) - np.median(y)

def add_significance(data, rfids, ax, bp):
    """Computes and adds p-value significance to input ax and boxplot"""
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    rfids1 = rfids[0]
    rfids2 = rfids[1]
    data1, data2 = data[0], data[1]

    # Group data by RFID
    grouped_data1 = [data1[np.where(rfids1 == rfid)].tolist() for rfid in np.unique(rfids1)]
    grouped_data2 = [data2[np.where(rfids2 == rfid)].tolist() for rfid in np.unique(rfids2)]
    
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

def activity_1st_x_2nd(var = "interactions"):
    #compares the activity of the 1st and the 2nd cohorts
    labels = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
    activity_first = []
    activity_second = []
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\both_cohorts_7days\\"+g+f"\\{var}_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\"+g+f"\\{var}_resD7_2.csv"], percentage_threshold = 0)[0]
        for mouse in range(data.shape[0]):
            if idx < 9:
                activity_first.append(np.sum(np.sum(data[mouse, :])))
            else:
                activity_second.append(np.sum(np.sum(data[mouse, :])))
                
    fig, ax = plt.subplots(1, 1)
    data = [activity_first, activity_second]
    plt.scatter([1]*len(activity_first)+np.random.rand(len(activity_first))*0.4 - 0.2, activity_first)
    plt.scatter([2]*len(activity_second)+np.random.rand(len(activity_second))*0.4 - 0.2, activity_second)
    bp = ax.boxplot(data, widths=0.6, showfliers = False, zorder=1)
    add_significance(data, ax, bp)
    plt.ylabel("Outgoing approaches")
    plt.xlabel("Cohort")
    plt.title(var)
    
def age_distribution():
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    plt.hist(metadata_df["age"], rwidth = 0.9, facecolor = "gray", color = None, bins = 20)
    plt.xlabel("Age in weeks")
    plt.ylabel("Count")
    
def activity_x_age(var = "interactions", show_mut = False):
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17"]
    age = []
    activity = []
    colors = []
    names = []
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_2.csv"], percentage_threshold = 0)[0]
        rfids = np.loadtxt("..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv", delimiter=",", dtype=str)[1:, 0].tolist()
        for idx, mouse in enumerate(rfids):
            if mouse in metadata_df.loc[metadata_df["Group_ID"] == int(g), "Mouse_RFID"].values:
                names.append(mouse)
                ismut = metadata_df.loc[(metadata_df["Mouse_RFID"] == mouse) & (metadata_df["Group_ID"] == int(g)), "mutant"].values[0] 
                ismut = bool(ismut) & bool(metadata_df.loc[(metadata_df["Mouse_RFID"] == mouse) & (metadata_df["Group_ID"] == int(g)), "histology"].values[0] != "weak")
                activity.append(np.sum(data[idx, :]))
                age.append(metadata_df.loc[(metadata_df["Mouse_RFID"] == mouse) & (metadata_df["Group_ID"] == int(g)), "age"].values[0])
                if ismut:
                    colors.append("red")
                else:
                    colors.append("gray")

    age, activity, names = np.array(age), np.array(activity), np.array(names)
    fig, ax = plt.subplots(1, 1)
    if show_mut:
        plt.scatter(age, activity, s = 60, c = colors, alpha = 0.5)
    else:
        plt.scatter(age, activity, s = 60, c = "k", alpha = 0.5)

    young = age < 30
    ax.plot([15, 25], [np.median(activity[young])]*2, c = "k", label = f"Median {var}")
    ax.plot([50, 95], [np.median(activity[~young])]*2, c = "k")
    ax.set_xlabel("Age in weeks", fontsize = 15)
    ax.set_ylabel(var, fontsize = 15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    
    # fig, ax2 = plt.subplots(1, 1)
    # data = [activity[young], activity[~young]]
    # bp = plt.boxplot(data)
    # add_significance(data, [names[young], names[~young]], ax2, bp)
    # ax2.set_title("Significance test for differences of activity between cohorts")
    
def activity_RC_x_noRC(var = "interactions"):
    # compares the overall activity of groups which do have a sRC and groups which do not.
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17"]
    all_ac = []
    ac_RC = []
    ac_no_RC = []
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_2.csv"], percentage_threshold = 0)[0]
        rfids = np.loadtxt("..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv", delimiter=",", dtype=str)[1:, 0].tolist()
        if g not in ["13", "14", "16"]:
            ac_RC.append(np.sum(np.sum(data[:, :])))
        else:
            ac_no_RC.append(np.sum(np.sum(data[:, :])))
        all_ac.append(np.sum(np.sum(data[:, :])))
    fig, ax = plt.subplots(1, 1, figsize = (4, 6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter([1]*len(ac_RC)+np.random.rand(len(ac_RC))*0.4 - 0.2, ac_RC, c = "k")
    plt.scatter([2]*len(ac_no_RC)+np.random.rand(len(ac_no_RC))*0.4 - 0.2, ac_no_RC, c = "gray")
    # data = [ac_RC, ac_no_RC]
    # bp = plt.boxplot(data)
    # def median(x, y, axis):
    #     return np.median(x, axis=axis) - np.median(y, axis=axis)
    # bp_sig(data, ax, bp, median)
    plt.xticks([1, 2], labels = ["sRC", "no sRC"], fontsize = 15)
    plt.ylabel(var, fontsize = 15)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    
def mutNum_x_noRC():
    all_ac = [1, 1, 2, 1, 2, 1, 2, 1, 1, 4, 2, 4, 2, 4, 4, 2]
    ac_RC = [1, 1, 2, 1, 2, 1, 2, 1, 1, 4, 2, 4, 2]
    ac_no_RC = [2, 4, 4]

    fig, ax = plt.subplots(1, 1, figsize = (4, 6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter([1]*len(ac_RC)+np.random.rand(len(ac_RC))*0.4 - 0.2, ac_RC, c = "k", s = 150)
    plt.scatter([2]*len(ac_no_RC)+np.random.rand(len(ac_no_RC))*0.4 - 0.2, ac_no_RC, c = "gray", s = 150)
    data = [ac_RC, ac_no_RC]
    # bp = plt.boxplot(data, showfliers=False)
    # def median(x, y, axis):
    #     return np.median(x, axis=axis) - np.median(y, axis=axis)
    # bp_sig(data, ax, bp, median)
    ax.set_ylim([0, 5])

    plt.xticks([1, 2], labels = ["sRC", "no sRC"], fontsize = 15)
    plt.ylabel("Mutants in group", fontsize = 15)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.yticks([1,2,3,4])
    plt.tight_layout()
    
def mutNum_x_activity(var = "interactions"):
    # mutnum  = [1, 1, 2, 1, 2, 1, 2, 1, 1, 4, 2, 4, 2, 4, 4, 2]
    mutnum  = [1, 1, 2, 1, 2, 1, 2, 1, 1, 4, 2, 4, 2]

    all_ac = []
    ac_RC = []
    ac_no_RC = []
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17"]
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_2.csv"], percentage_threshold = 0)[0]
        rfids = np.loadtxt("..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv", delimiter=",", dtype=str)[1:, 0].tolist()
        if g not in ["13", "14", "16"]:
            ac_RC.append(np.sum(np.sum(data[:, :])))
        else:
            ac_no_RC.append(np.sum(np.sum(data[:, :])))
        all_ac.append(np.sum(np.sum(data[:, :])))
        
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter(mutnum, ac_RC, s = 100, c = "k", alpha = 0.5, label = "Pearson coeff = "+str(np.round(np.corrcoef(mutnum, ac_RC)[0,1], 2)))
    coef = np.polyfit(mutnum, ac_RC,1)
    poly1d_fn = np.poly1d(coef) 
    plt.plot(mutnum, poly1d_fn(mutnum), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker

    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xticks([0,1,2,3,4,5])
    plt.xlabel("Number of mutants", fontsize = 15)
    plt.ylabel(f"{var}",fontsize = 15)
    plt.tight_layout()
    
def norm(arr):
    return (arr - np.mean(arr))/np.std(arr)
    
def GLM_model(var = "interactions"):
    all_ac = []
    ac_RC = []
    ac_no_RC = []
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17"]
    for idx, g in enumerate(labels):
        data = read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv"], percentage_threshold = 0)[0] + read_graph(["..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_2.csv"], percentage_threshold = 0)[0]
        rfids = np.loadtxt("..\\data\\both_cohorts_7days\\G"+g+f"\\{var}_resD7_1.csv", delimiter=",", dtype=str)[1:, 0].tolist()
        if g not in ["13", "14", "16"]:
            ac_RC.append(np.sum(np.sum(data[:, :])))
        else:
            ac_no_RC.append(np.sum(np.sum(data[:, :])))
        all_ac.append(np.sum(np.sum(data[:, :])))
        
    metadata_path = "..\\data\\meta_data.csv"
    metadata_df = pd.read_csv(metadata_path)
    avg_age_per_group = metadata_df.groupby('Group_ID')['age'].mean().reset_index()
    scaler = StandardScaler()

    data = pd.DataFrame({
        f'{var}': norm(np.array(all_ac)),
        'mutNum': norm(np.array([1, 1, 2, 1, 2, 1, 2, 1, 1, 4, 2, 4, 2, 4, 4, 2])),
        'avgAge': norm(np.array(avg_age_per_group["age"].values[[0,1,2,3,4,5, 6,7,9,10,11,12,13,14,15,16]])), 
        'sRC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]  
    })
    
    X = data[[f'{var}', 'mutNum', "avgAge"]]  # Predictors
    y = data['sRC']                  # Binary target
    
    # # Add intercept (required for statsmodels)
    # X = sm.add_constant(X)
    
    # # Binomial GLM with logit link (logistic regression)
    # logit_model = sm.GLM(y, X, family=sm.families.Binomial())
    # results = logit_model.fit()
    # print(results.summary())
    
    # Add L2 penalty (ridge regression)
    logit_reg = sm.Logit(y, sm.add_constant(X))
    results = logit_reg.fit_regularized(alpha=2.0)  # alpha = regularization strength
    print(results.summary())
    
    
if __name__ == "__main__":
    activity_x_age("approaches")
    # activity_RC_x_noRC("approaches")
    # activity_RC_x_noRC("interactions")
    # mutNum_x_noRC()
    # mutNum_x_activity("approaches")
    # GLM_model()