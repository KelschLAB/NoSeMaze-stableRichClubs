import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def mutant_in_cohort():
    number_of_hits = []
    mutants = np.array([1, 2])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(10000)):
        hits = 0
        for j in range(8):
            # expID 1
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 2
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 3
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 4
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 5
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 6
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 7
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 8
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 10
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))

        number_of_hits.append((100*hits)/(9*8))
        
    h = plt.hist(number_of_hits, bins = 15, density = True, label = "Expected by random chance")
    plt.title("Random chance of mutant in rich-club\n 1st cohort")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(11.1, color = 'red', label = "Experimentally observed")
    plt.legend()


def mutants_in_validation():
    number_of_hits = []
    mutants = np.array([1, 2, 3])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(4000)):
        hits = 0
        for j in range(8):
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 14
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 17
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            
        number_of_hits.append((100*hits)/(5*8))
        
    plt.hist(number_of_hits, density = True, label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(20, color = 'red', label = "Experimentally observed")
    plt.title("Random chance of mutant in rich-club\n 2nd cohort")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()


mutant_in_cohort()
# mutants_in_validation()