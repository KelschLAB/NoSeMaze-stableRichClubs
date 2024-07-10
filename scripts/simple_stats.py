import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import pandas as pd
import seaborn as sns


path = "C:\\Users\\Corentin offline\\Documents\\GitHub\\clusterGUI\\data\\reduced_data.xlsx"
df = pd.read_excel(path)
sns.catplot(data=df, x = "mutant",y = "rank_by_tube", kind="violin", color=".9", inner=None)
sns.swarmplot(data=df, x = "mutant",y = "rank_by_tube", size=5)
plt.title("Tube rank")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "rank_by_chasing", kind="violin", color=".9", inner=None)
sns.swarmplot(data=df, x = "mutant",y = "rank_by_chasing", size=5)
plt.title("Chasing rank")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "time_in_arena_average", kind="violin", color=".9", inner=None)
sns.swarmplot(data=df, x = "mutant",y = "time_in_arena_average", size=5)
plt.title("time_in_arena_average")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed", kind="violin", color=".9", inner=None)
sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed", size=5)
plt.title("cs_plus_detection_speed")
plt.tight_layout()
