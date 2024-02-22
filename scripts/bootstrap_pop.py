import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

number_of_hits = []
shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

for i in tqdm(range(10000)):
    hits = 0
    for k in range(10):
        for j in range(5):    
            np.random.shuffle(shuffled_arr)
            # if shuffled_arr[0] == 1 or shuffled_arr[0] == 2 or shuffled_arr[0] == 3: #for curated  cohort
            if shuffled_arr[0] == 1 or shuffled_arr[0] == 2 or shuffled_arr[1] == 1 or shuffled_arr[1] == 2\
                or shuffled_arr[2] == 1 or shuffled_arr[2] == 2 or shuffled_arr[3] == 1 or shuffled_arr[3] == 2: #for control cohort
                hits += 1
    number_of_hits.append((100*hits)/50)
    
h = plt.hist(number_of_hits, bins = 25, density = True)
plt.title("Random chance of mutant in rich-club\n Control cohort")
plt.xlabel("Observed percentage", fontsize=15)
plt.ylabel("Probability of observation", fontsize=15)
