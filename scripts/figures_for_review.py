import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

metadatapath = "..\\data\\meta_data.csv"
metadata = pd.read_csv(metadatapath)
valid_groups = [1,2,3,4,5,6,7,8,10,11,12,15,17] # Groups for which a sRC was observed

### Pie/bar chart for showing RC sizes 
sizes = [np.sum(metadata['mutant'] & (metadata['Group_ID'].isin(valid_groups))), np.sum(metadata['mutant'] &  metadata['RC'] & (metadata['Group_ID'].isin(valid_groups))), 
         np.sum(~metadata['mutant'] &  metadata['RC'] & (metadata['Group_ID'].isin(valid_groups))), np.sum(~metadata['mutant'] & ~metadata['RC'] & (metadata['Group_ID'].isin(valid_groups)))]
# V1
labels = ['non-sRC OXTR ' + str(sizes[0]),  'sRC OXTR '+ str(sizes[1]), 'sRC WT '+ str(sizes[2]), 'non-sRC WT '+ str(sizes[3])] 
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, hatch=['', '', '', ''],
       colors=['red', 'darkred', 'dimgray', 'lightgray'])
plt.show()

#V2
s = 0.3
r = 1.1
fig, ax = plt.subplots()
sizes = [np.sum(~metadata['mutant'] &  metadata['RC'] & (metadata['Group_ID'].isin(valid_groups))), np.sum(metadata['mutant'] &  metadata['RC'] & (metadata['Group_ID'].isin(valid_groups))),
         np.sum(metadata['mutant'] & (metadata['Group_ID'].isin(valid_groups))), np.sum(~metadata['mutant'] & ~metadata['RC'] & (metadata['Group_ID'].isin(valid_groups)))]

ax.pie([sizes[0]+sizes[1], sizes[2]+sizes[3]], radius=r, colors=['blue', 'lightblue'], 
       wedgeprops=dict(width=s, edgecolor='w'))

ax.pie(sizes, radius=r-s, colors=["dimgray", "darkred", "red", "lightgray"], 
       wedgeprops=dict(width=s, edgecolor='w'))
plt.show()

#V3
species = ("WT", "OXTR")
weight_counts = {"sRC": np.array([27, 2]), "non-sRC": np.array([69, 34])}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(2)

weight_count = weight_counts["sRC"]
p = ax.bar(species, weight_count, width, label="sRC", bottom=bottom, color = "dimgray")
bottom += weight_count
weight_count = weight_counts["non-sRC"]
p = ax.bar(species, weight_count, width, label="non-sRC", bottom=bottom, color = "lightgray")

ax.legend(loc="upper right")
plt.show()