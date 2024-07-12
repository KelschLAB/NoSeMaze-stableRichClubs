import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
import os
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


path = "..\\data\\reduced_data.xlsx"
df = pd.read_excel(path)
sns.catplot(data=df, x = "mutant", y = "rank_by_tube", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "rank_by_tube", size=5, hue = "mutant")
plt.title("Tube rank")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "rank_by_chasing", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "rank_by_chasing", size=5, hue = "mutant")
plt.title("Chasing rank")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "time_in_arena_average", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "time_in_arena_average", size=5, hue = "mutant")
oxtype = df.loc[:, "mutant"].to_numpy()
times = df.loc[:, "time_in_arena_average"].to_numpy()
dist_wt = times[~oxtype]
dist_mu = times[oxtype]
t, p = ttest_ind(dist_wt[~np.isnan(dist_wt)], dist_mu[~np.isnan(dist_mu)])
plt.title("time_in_arena_average\n p-value = 0.28")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed", size=5, hue = "mutant")
plt.title("cs_plus_detection_speed")
plt.tight_layout()

sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", size=5, hue = "mutant")
plt.title("cs_plus_detection_speed_crossreversal_shaping")
plt.tight_layout()


sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_crossreversal_shaping", size=5, hue = "mutant")
plt.title("cs_plus_detection_speed_crossreversal_shaping")
plt.tight_layout()

sns.catplot(data=df, x = "mutant",y = "cs_plus_detection_speed_intraphase_shaping", color=".9", kind="box", width = 0.33)
sns.swarmplot(data=df, x = "mutant",y = "cs_plus_detection_speed_intraphase_shaping", size=5, hue = "mutant")
plt.title("cs_plus_detection_speed_intraphase_shaping")
plt.tight_layout()
