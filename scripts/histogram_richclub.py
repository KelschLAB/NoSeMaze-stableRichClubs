import matplotlib.pyplot as plt
import numpy as np

def tuberank_vs_rc():
    arr = np.array([2, 3, 5, 1, 2, 4, 1, 7, 4, 2, 2, 1, 1, 6, 4, 6, 5, 1, 2, 3]) # chasing rank
    arr = np.array([2, 4, 2, 3, 8, 2, 10, 1, 8, 2, 6, 5, 3, 1, 2, 5, 1, 2, 6, 2, 4]) # tube rank
    plt.hist(arr, bins = [i for i in range(1, 11)], align = 'mid', rwidth = 0.95, color = "gray") 
    # plt.xlabel("Chasing rank") 
    plt.xlabel("Tube rank")
    plt.ylabel("Number of observations"); 
    ticklabels = [i for i in range(1, 11)]
    plt.xticks([i + 0.5 for i in range(1, 11)], ticklabels)
    # plt.title("Chasing rank of the rich-club members")
    plt.title("Tube rank of rich-club members")
    
def chasings_vs_rc():
    tot10 = np.array([4, 24, 17, 24, 29, 30, 10]) # total chasing from RC for thres 10%
    tot30 = np.array([1, 5, 5, 10, 12, 18, 1]) # total chasing from RC for thres 30%
    tot50 = np.array([1, 2, 3, 6, 6, 4, 1]) # chasing from RC for thres 50%
    arr10 = np.array([2, 9, 8, 16, 14, 15, 7]) # chasing from RC for thres 10%
    arr30 = np.array([1, 4, 4, 9, 7, 10, 1]) # chasing from RC for thres 30%
    arr50 = np.array([1, 2, 3, 5, 3, 2 ,1]) # chasing from RC for thres 50%
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    labels = ['G1', 'G2', 'G4', 'G6', 'G7', 'G8', 'G10']
    x = np.arange(len(labels))
    width = 0.20
    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x - width/1.5, arr10/tot10, width, label='10%')
    rects3 = ax.bar(x + width/1.5, arr50/tot50, width, label='50%')
    rects2 = ax.bar(x, arr30/tot30, width, label='30%')

    
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Threshold:")

    
    # def autolabel(rects):
    #    for rect in rects:
    #       height = rect.get_height()
    #       ax.annotate('{}'.format(height),
    #          xy=(rect.get_x() + rect.get_width() / 2, height),
    #          xytext=(0, 3), # 3 points vertical offset
    #          textcoords="offset points",
    #          ha='center', va='bottom')
    
    # autolabel(rects1)
    # autolabel(rects2)
    
    plt.show()
    
chasings_vs_rc()

    