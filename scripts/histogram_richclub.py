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
    
chasings_vs_rc()

    