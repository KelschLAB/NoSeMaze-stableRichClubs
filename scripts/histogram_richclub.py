import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
from weighted_rc import weighted_rich_club
import pandas as pd

plt.rcParams.update({"text.usetex": True})

# indices of rc and mutants in excel sheet for cohort 1
rc_index_in_excel1 = np.array([7, 12, 14, 15, 25, 26, 31, 34, 35, 45, 53, 59, 67, 69, 70, 76, 78, 95, 98]) - 2 # RC
mutants_index_in_excel1 = np.array([6, 9, 13, 21, 27, 28, 33, 41, 51, 54, 58, 65, 66, 73, 75, 92, 94]) - 2 # Mutants
other_index_in_excel1 = np.arange(99)[~np.isin(np.arange(99), np.concatenate((rc_index_in_excel1, mutants_index_in_excel1)))] # Others
# indices of rc and mutants in excel sheet for cohort 1
rc_index_in_excel2 = np.array([9, 11, 12, 14, 50, 51]) - 2
mutants_index_in_excel2 = np.array([3, 5, 6, 7, 17, 19, 22, 28, 29, 34, 40, 43, 46]) - 2
other_index_in_excel2 = np.arange(109)[~np.isin(np.arange(109), np.concatenate((rc_index_in_excel2, mutants_index_in_excel2)))] # Others

def rich_club_piechart():
    fig, ax = plt.subplots(1, 2)
    plt.style.use('seaborn')
    sizes = [8/9, 1/9]
    labels = ["Rich club", "No stable club"]
    ax[0].pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    sizes =     sizes = [4/7, 3/7]
    labels = ["Single club", "No stable club"]
    ax[1].pie(sizes, labels=labels, autopct='%1.f\%%',
        wedgeprops={"linewidth" : 2.0, "edgecolor": "white"},
        textprops={'size': 'xx-large'})
    plt.show()

def mutants_RC_hist():
    fig = plt.Figure(figsize = (4,8))
    mutants_in_rc = [1, 0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,1,0]
    # counts, bins = plt.hist(mutants_in_rc, bins = 2, label = ["0", "1"], rwidth = 0.7)
    plt.bar([0, 0.3], [16, 2], width=0.25)
    # tick_values, tick_labels = plt.xticks()
    plt.xticks([0, 0.3], ["0", "1"])
    plt.xlabel("Number of mutants in the rich-club", fontsize = 25)
    plt.ylabel("Count", fontsize = 25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)

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
    nn -= 1
    mnn_arr = np.zeros_like(arr)  
    neighbors_i = np.argsort(-arr, 1) #computing the nearest neighbors for local nn estimation
    neighbors_j = np.argsort(-arr, 0)
    for i in range(arr.shape[1]):
        for j in range(i, arr.shape[1]):
            if any(np.isin(neighbors_i[i, 0:nn], j)) and any(np.isin(neighbors_j[0:nn, j], i)): #local nearest distance estimation
                mnn_arr[i, j] += arr[i, j]
                mnn_arr[j, i] += arr[j, i]
    return mnn_arr

def rc_coefficient_histogram(nn = 4, k = 3):
    coefficients = []
    path = "..\\data\\averaged\\"
    for G in os.listdir(path):
        path_to_file = path+G+"\\interactions_resD7_1.csv"
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        data = mnn_cut(data, nn)
        graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
        try:
            plt.show()
            nx_graph = graph.to_networkx()
            # coefficients.append(nx.rich_club_coefficient(nx_graph, normalized=True)[3])
            coefficients.append(weighted_rich_club(data, k))
        except:
            pass
    for G in os.listdir(path):
        path_to_file = path+G+"\\interactions_resD7_2.csv"
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        data = mnn_cut(data, nn)
        try:
            graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
            plt.show()
            nx_graph = graph.to_networkx()
            # coefficients.append(nx.rich_club_coefficient(nx_graph, normalized=True)[3])
            coefficients.append(weighted_rich_club(data, k))
        except:
            pass
    # path = "..\\data\\validation cohort\\7days\\"
    # for G in os.listdir(path):
    #     path_to_file = path+G+"\\interactions_resD7_1.csv"
    #     arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
    #     data = arr[1:, 1:].astype(float)
    #     data = mnn_cut(data, nn)
    #     try:
    #         graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    #         plt.show()
    #         nx_graph = graph.to_networkx()
    #         coefficients.append(nx.rich_club_coefficient(nx_graph, normalized=True)[3])
    #     except:
    #         pass
    # for G in os.listdir(path):
    #     path_to_file = path+G+"\\interactions_resD7_2.csv"
    #     arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
    #     data = arr[1:, 1:].astype(float)
    #     data = mnn_cut(data, nn)
    #     graph = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    #     try:
    #         plt.show()
    #         nx_graph = graph.to_networkx()
    #         coefficients.append(nx.rich_club_coefficient(nx_graph, normalized=True)[3])
    #     except:
    #         pass
    plt.hist(coefficients, bins=7)


def total_chasings_cohort():
    fig, ax = plt.subplots(1, 1)
    cohort_num = [1, 2, 4, 6, 7, 8, 10]
    control_cohort_num = [11, 12, 14, 15, 17]
    total_chasings = np.array([4, 24, 17, 24, 29, 30, 10])
    total_control_chasings = np.array([ 31, 54, 42, 38, 62])
    for i in range(len(cohort_num)):
        ax.bar(i, total_chasings[i], 0.5, color = "blue", alpha = 0.4) 
    for i in range(len(control_cohort_num)):
        ax.bar(i+7, total_control_chasings[i], 0.5, color = "red", alpha = 0.4) 
    plt.xlabel("Cohort index")
    plt.ylabel("Number of chasings (10\% threshold)", fontsize = 14); 
    total_idx = []
    total_idx.extend(cohort_num); total_idx.extend(control_cohort_num)
    plt.xticks(range(12), total_idx)
    ax.set_title("Total chasings in $1^{st}$ vs validation cohort", fontsize = 15)
    ax.bar(np.nan, np.nan, color = "blue", alpha = 0.4, label = r"$1^{st}$ cohort") 
    ax.bar(np.nan, np.nan, color = "red", alpha = 0.4, label = r"Validation cohort")
    ax.legend()

def tuberank_vs_rc():
    """Plots the tube rank of rich clulb members for both cohorts"""
    arr = np.array([2,4, 2,3,8, 10,2,1, 6,1,5,3,1,5,2,1,2,6,4,3,1,2,1,2,4,8]) # tube rank
    tuberank_rc1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel1, 9].astype(float)
    tuberank_rc2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                        sheet_name = 0).to_numpy()[:, 1:][rc_index_in_excel2, 9].astype(float)
    arr = np.concatenate((tuberank_rc1, tuberank_rc2))
    # arr = tuberank_rc1
    plt.figure()
    plt.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.95, color = "gray") 
    plt.xlabel("Tube rank")
    plt.ylabel("Number of observations"); 
    ticklabels = [i for i in range(1, 11)]
    plt.xticks([i + 0.5 for i in range(1, 11)], ticklabels)
    plt.title("Tube rank of rich-club members")
    
def chasingrank_vs_rc():
    """Plots the chasing rank of rich clulb members for both cohorts"""
    arr = np.array([2,3, 1,5,10, 1,2,4, 2, 4, 1, 2, 1,4,6,5,6,1,2,2,3,1,2,2,3,6]) # chasing rank
    plt.hist(arr, bins = [i for i in range(1, 11)], align = 'mid', rwidth = 0.95, color = "gray") 
    plt.xlabel("Chasing rank") 
    plt.xlabel("Tube rank")
    plt.ylabel("Number of observations"); 
    ticklabels = [i for i in range(1, 11)]
    plt.xticks([i + 0.5 for i in range(1, 11)], ticklabels)
    plt.title("Chasing rank of the rich-club members")
    
def tuberank_vs_nonrc():
    """Plots the tube rank of nonrich clulb members for both cohorts"""
    tuberank_others1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel1, 9].astype(float)
    tuberank_others2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel2, 9].astype(float)
    arr = np.concatenate((tuberank_others1, tuberank_others2))
    plt.figure()
    plt.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.95, color = "gray") 
    plt.xlabel("Tube rank", fontsize = 15)
    plt.ylabel("Number of observations", fontsize = 15); 
    ticklabels = [i for i in range(1, 12)]
    plt.xticks([i + 0.5 for i in range(1, 12)], ticklabels)
    plt.title("Tube rank of non rich-club members")
    
def chasingrank_vs_nonrc():
    """Plots the chasing rank of nonrich club members for both cohorts"""
    chasingrank_others1 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\reduced_data.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel1, 11].astype(float)
    chasingrank_others2 = pd.read_excel(r"C:\Users\Agarwal Lab\Corentin\Python\NoSeMaze\data\meta-data_validation.xlsx", 
                                       sheet_name = 0).to_numpy()[:, 1:][other_index_in_excel2, 11].astype(float)
    arr = np.concatenate((chasingrank_others1, chasingrank_others2))
    plt.figure()
    plt.hist(arr, bins = [i for i in range(1, 12)], align = 'mid', rwidth = 0.95, color = "gray") 
    plt.xlabel("Chasing rank", fontsize = 15) 
    plt.ylabel("Number of observations", fontsize = 15); 
    ticklabels = [i for i in range(1, 12)]
    plt.xticks([i + 0.5 for i in range(1, 12)], ticklabels)
    plt.title("Chasing rank of non rich-club members")
    
def chasings_vs_rc():
    tot10 = np.array([4, 24, 17, 24, 29, 30, 10]) # total chasing from RC for thres 10%
    tot30 = np.array([1, 5, 5, 10, 12, 18, 1]) # total chasing from RC for thres 30%
    tot50 = np.array([1, 2, 3, 6, 6, 4, 1]) # chasing from RC for thres 50%
    arr10 = np.array([2, 9, 8, 16, 14, 15, 7]) # chasing from RC for thres 10%
    arr30 = np.array([1, 4, 4, 9, 7, 10, 1]) # chasing from RC for thres 30%
    arr50 = np.array([1, 2, 3, 5, 3, 2 ,1]) # chasing from RC for thres 50%
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    labels = ['G1', 'G2', 'G4', 'G6', 'G7', 'G8', 'G10']
    x = np.arange(len(labels))*1
    width = 0.35
    fig, ax = plt.subplots()
    
    # colors = ["#ffeda0", "#feb24c", "#f03b20"] # reds
    colors = ["#edf8b1", "#7fcdbb", "#2c7fb8"] # veridis
    # colors = ['#f0f0f0','#bdbdbd','#636363'] # grays
    # colors = ['#e0ecf4','#9ebcda','#8856a7'] # purples
    # colors = ['#a1dab4','#41b6c4','#225ea8'] # veridis dark
    
    rects1 = ax.bar(x - width, 100*arr10/tot10, 2*width/3, label='10\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/3, 100*arr30/tot30, 2*width/3, label='30\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x + width/3, 100*arr50/tot50, 2*width/3, label='50\%', align = 'edge', color = colors[2]) 

    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total chasings fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members chasings')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    ax.hlines(25, x[0] - width, x[-1] + width, linestyle = "--", color = "k", alpha = 0.5)  # 25 because there are 2 to 3 members of stable rich club
    ax.annotate('Random chance', xy=(4.5, 25), xytext=(4 + width/10, 70), ha = 'left',      # 25 because there are 2 to 3 members of stable rich club
            arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 5), textcoords='data', xycoords='data')
    plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC.png", dpi = 150)
    plt.show()
    
def chasings_vs_rc_validation():
    tot10 = np.array([31, 54, np.nan, 42, 38, np.nan,  62]) # total chasing from RC for thres 10%
    tot30 = np.array([17, 15, np.nan, 19, 19, np.nan,  22]) # total chasing from RC for thres 30%
    tot50 = np.array([7, 8, np.nan, 8, 11, np.nan,  5]) # chasing from RC for thres 50%
    tot70 = np.array([3, 4, np.nan, 3, 3, np.nan,  1]) # chasing from RC for thres 70%
    arr10 = np.array([27, 31, np.nan, 31, 21, np.nan,  46]) # chasing from RC for thres 10%
    arr30 = np.array([17, 14, np.nan, 17, 11, np.nan,  15]) # chasing from RC for thres 30%
    arr50 = np.array([7, 8, np.nan, 8, 6, np.nan,  3]) # chasing from RC for thres 50%
    arr70 = np.array([3, 4, np.nan, 3, 2, np.nan,  1]) # chasing from RC for thres 70%

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    labels = ['G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17']
    x = np.arange(len(labels))*1
    width = 0.5
    fig, ax = plt.subplots()
    
    # colors = ["#ffeda0", "#feb24c", "#f03b20"] # reds
    colors = ["#edf8b1", "#7fcdbb", "#2c7fb8", "#225ea8"] # veridis
    # colors = ['#f0f0f0','#bdbdbd','#636363'] # grays
    # colors = ['#e0ecf4','#9ebcda','#8856a7'] # purples
    # colors = ['#a1dab4','#41b6c4','#225ea8'] # veridis dark
    
    rects1 = ax.bar(x - width/2, 100*arr10/tot10, width/4, label='10\%', align = 'edge', color = colors[0]) 
    rects2 = ax.bar(x - width/4, 100*arr30/tot30, width/4, label='30\%', align = 'edge', color = colors[1])
    rects3 = ax.bar(x, 100*arr50/tot50, width/4, label='50\%', align = 'edge', color = colors[2]) 
    rects4 = ax.bar(x + width/4, 100*arr70/tot70, width/4, label='70\%', align = 'edge', color = colors[3]) 
    ax.annotate('No rich-club', xy=(2, 30), xytext=(2, 25), rotation = 90)      # 25 because there are 2 to 3 members of stable rich club
    ax.annotate('No rich-club', xy=(5, 30), xytext=(5, 25), rotation = 90)      # 25 because there are 2 to 3 members of stable rich club


    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel('Total chasings fraction (\%)')
    ax.set_xlabel('Cohort')
    ax.set_title('Rich-club members chasings, validation cohort')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")
    # ax.hlines(40, x[0] - width, x[-1] + width, linestyle = "--", color = "k", alpha = 0.5)  # 25 because there are 2 to 3 members of stable rich club
    # ax.annotate('Random chance', xy=(5,45), xytext=(4.2 + width/10, 70), ha = 'left',      # 25 because there are 2 to 3 members of stable rich club
    #         arrowprops=dict(facecolor='black', shrink=0.05, width = 0.5, headwidth = 5), textcoords='data', xycoords='data')
    plt.savefig("C:\\Users\\Agarwal Lab\\Corentin\\Python\\clusterGUI\\plots\\chasings_vs_RC_validation.png", dpi = 150)
    plt.show()
    
if __name__ == "__main__":
    # rich_club_piechart()
    # mutants_RC_hist()
    # rc_coefficient_histogram(4, 3)
    # chasings_vs_rc()
    # tuberank_vs_rc()
    # chasingrank_vs_nonrc()
    # tuberank_vs_nonrc()
# chasings_vs_rc_validation()
# total_chasings_cohort()

    