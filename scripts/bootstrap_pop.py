import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def mutant_in_cohort():
    number_of_hits = []
    mutants = np.array([1, 2])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(5000)):
        hits = 0
        for j in range(1):
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

        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance")
    plt.title("Random chance of mutant in rich-club\n 1st cohort")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(1, color = 'red', label = "Experimentally observed")
    plt.legend()


def mutants_in_both():
    number_of_hits = []
    mutants = np.array([1, 2, 3])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(5000)):
        hits = 0
        for j in range(1):
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
            # # expID 18
            # np.random.shuffle(shuffled_arr)
            # hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            
        number_of_hits.append(hits)
        
    plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(2, color = 'red', label = "Experimentally observed")
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()


def mutants_in_validation():
    number_of_hits = []
    mutants = np.array([1, 2, 3])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(1000)):
        hits = 0
        for j in range(1):
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 16
            # np.random.shuffle(shuffled_arr)
            # hits += np.any(np.isin(mutants, shuffled_arr[:3]))
            # expID 17
            np.random.shuffle(shuffled_arr)
            hits += np.any(np.isin(mutants, shuffled_arr[:2]))

            
        number_of_hits.append(hits)
        
    plt.hist(number_of_hits, bins = np.arange(7), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(0, color = 'red', label = "Experimentally observed")
    plt.title("Random chance of mutant in rich-club\n 2nd cohort")
    plt.xlabel("Groups with at least 1 mutants in RC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    
    
def littermates_in_club():
    number_of_hits = []
    shuffled_arr1 = np.array([41, 31, 2, 3, 3, 5, 6, 7, 7, 8])
    # shuffled_arr2 = np.array([25, 28, 40, 41, 42, 2, 5, 6, 8])
    shuffled_arr3 = np.array([25, 29, 29, 29, 32, 32, 2, 3, 4])
    shuffled_arr4 = np.array([25, 31, 2, 3, 5, 5, 6, 7, 7, 8])
    shuffled_arr5 = np.array([25, 28, 29, 29, 30, 42, 2, 3, 6])
    shuffled_arr6 = np.array([25, 29, 41, 4, 5, 5, 6, 7, 8, 8])
    shuffled_arr7 = np.array([25, 25, 29, 29, 32, 32, 2, 3, 5])
    shuffled_arr8 = np.array([25, 28, 29, 40, 41, 2, 5, 6, 8]) 
    shuffled_arr10 = np.array([28, 29, 40, 41, 42, 2, 5, 5, 6])
    shuffled_arr11 = np.array([13, 14, 14, 15, 15, 15, 17, 17, 18, 22])
    shuffled_arr12 = np.array([11, 12, 13, 14, 15, 20, 21, 23, 23, 23])
    shuffled_arr15 = np.array([13, 14, 14, 15, 15, 15, 16, 17, 17, 22])
    shuffled_arr17 = np.array([10, 13, 19, 20, 21, 21, 22, 23, 23, 23])


    for i in tqdm(range(1000)):
        hits = 0
        # expID 1
        np.random.shuffle(shuffled_arr1)
        if shuffled_arr1[0] == shuffled_arr1[1]:
            hits += 1
        # expID 2
        # np.random.shuffle(shuffled_arr2)
        # if shuffled_arr2[0] == shuffled_arr2[1] or shuffled_arr2[1] == shuffled_arr2[2] or shuffled_arr2[0] == shuffled_arr2[2]:
        #     hits += 1
            
        np.random.shuffle(shuffled_arr3)
        if shuffled_arr3[0] == shuffled_arr3[1] or shuffled_arr3[1] == shuffled_arr3[2] or shuffled_arr3[0] == shuffled_arr3[2]:
            hits += 1
        # expID 4
        np.random.shuffle(shuffled_arr4)
        if shuffled_arr4[0] == shuffled_arr4[1]:
            hits += 1
        # expID 5
        np.random.shuffle(shuffled_arr5)
        if shuffled_arr5[0] == shuffled_arr5[1]:
            hits += 1
        # expID 6
        np.random.shuffle(shuffled_arr6)
        if shuffled_arr6[0] == shuffled_arr6[1]:
            hits += 1
        # expID 7
        np.random.shuffle(shuffled_arr7)
        if shuffled_arr3[0] == shuffled_arr7[1] or shuffled_arr7[1] == shuffled_arr7[2] or shuffled_arr7[0] == shuffled_arr7[2]:
            hits += 1
        # expID 8
        np.random.shuffle(shuffled_arr8)
        if shuffled_arr8[0] == shuffled_arr8[1]:
            hits += 1
        # expID 10
        np.random.shuffle(shuffled_arr10)
        if shuffled_arr10[0] == shuffled_arr10[1]:
            hits += 1
            
        np.random.shuffle(shuffled_arr11)
        if shuffled_arr11[0] == shuffled_arr11[1]:
            hits += 1

        np.random.shuffle(shuffled_arr12)
        if shuffled_arr12[0] == shuffled_arr12[1]:
            hits += 1
            
        np.random.shuffle(shuffled_arr15)
        if shuffled_arr15[0] == shuffled_arr15[1] or shuffled_arr15[1] == shuffled_arr15[2] or shuffled_arr15[0] == shuffled_arr15[2]:
            hits += 1
        
        np.random.shuffle(shuffled_arr17)
        if shuffled_arr17[0] == shuffled_arr17[1]:
                hits += 1

        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    plt.title("Random chance of litter mates in rich club")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 98), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(3, color = 'red', label = "Experimentally observed")
    plt.legend()

littermates_in_club()
# mutant_in_cohort()
# for i in range(1):
    # plt.figure()
    # mutant_in_cohort()
    # mutants_in_validation()
    # mutants_in_both()