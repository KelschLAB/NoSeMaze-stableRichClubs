import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def mutant_in_cohort():
    number_of_hits = []
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(10000)):
        hits = 0
        for k in range(10):
            for j in range(5):    
                np.random.shuffle(shuffled_arr)
                if shuffled_arr[0] == 1 or shuffled_arr[0] == 2 or shuffled_arr[0] == 3: #for curated  cohort
                    hits += 1
        number_of_hits.append((100*hits)/50)
        
    h = plt.hist(number_of_hits, bins = 25, density = True)
    plt.title("Random chance of mutant in rich-club\n Control cohort")
    plt.xlabel("Observed percentage", fontsize=15)
    plt.ylabel("Probability of observation", fontsize=15)


def mutants_in_validation():
    number_of_hits = []
    mutants = np.array([1, 2, 3, 4])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(10000)):
        hits = 0
        # expID 11
        np.random.shuffle(shuffled_arr)
        hits += np.sum(np.isin(mutants, shuffled_arr[:5]))
        # expID 12
        np.random.shuffle(shuffled_arr)
        hits += np.sum(np.isin(mutants, shuffled_arr[:4]))
        # expID 14
        np.random.shuffle(shuffled_arr)
        hits += np.sum(np.isin(mutants, shuffled_arr[:4]))
        # expID 15
        np.random.shuffle(shuffled_arr)
        hits += np.sum(np.isin(mutants, shuffled_arr[:3]))
        # expID 17
        np.random.shuffle(shuffled_arr)
        hits += np.sum(np.isin(mutants, shuffled_arr[:6]))
            
        number_of_hits.append((100*hits)/(22))
        
    h = plt.hist(number_of_hits, bins = 30, density = True)
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.legend()
    plt.title("Random chance of mutant in rich-club\n Control cohort")
    plt.xlabel("Percentage of mutants in RC", fontsize=15)
    plt.ylabel("Probability of observation", fontsize=15)


mutants_in_validation()