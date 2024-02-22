import matplotlib.pyplot as plt
import numpy as np

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