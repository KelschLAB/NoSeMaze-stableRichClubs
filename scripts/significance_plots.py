import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def chasing_towards(rc_size = 3, total_chasings = 100):
    """
    Computes the distribution of observed chasings towards sRC members, if chasings were
    distributed by random chance.
    """
    percentage_list = []
    for i in tqdm(range(10000)):
        shuffled_arr = np.array([1,2,3,4,5,6,7,8,9,10])
        hits = 0
        for chasing in range(total_chasings): # assuming 500 chasing events per cohort
            np.random.shuffle(shuffled_arr)
            if shuffled_arr[0] in [1,2,3]:
                hits += 1
        percentage_list.append(100*(hits/total_chasings))
        
    fig = plt.figure()
    h = plt.hist(percentage_list, bins = np.arange(0, 100, 2), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(percentage_list, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(percentage_list, 97.5), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(44, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.sum(h[0][22:]), 5)), (10, 0.05), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.xlabel("Percentage of chasing towards other sRC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    print(np.sum(h[0][22:]))
    plt.show()

def mutants_in_rc_mnn2_k2():
    number_of_hits = []
    mutants = np.array([1, 2])
    mutants_G11 = np.array([1,2,3,4])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    for i in tqdm(range(1000)):
        hits = 0
        for j in range(1):
            # expID 1
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 2
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 3
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 4
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 5
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 6
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 7
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:3]))
            # expID 8
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:1]))
            # expID 10
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:3]))
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants_G11, shuffled_arr[:1]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:3]))
            # # expID 18
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(mutants, shuffled_arr[:2]))
            
        number_of_hits.append(hits)
    
    fig = plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    # plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(2, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.cumsum(h[0]), 4)[2]), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    plt.show()

def mutants_in_rc_mnn3_k3(weak = False): #mnn = 3, rc = 3
    number_of_hits = []
    single_mutants = np.array([1])
    double_mutants = np.array([1, 2])
    triple_mutants = np.array([1,2,3])
    quadruple_mutants = np.array([1,2,3,4])
    quintuple_mutants = np.array([1,2,3,4,5])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if weak:
        full_histo_mutants = [double_mutants, double_mutants,double_mutants,double_mutants,double_mutants,double_mutants,double_mutants, double_mutants, double_mutants, 
                              quadruple_mutants, quintuple_mutants, triple_mutants, triple_mutants, triple_mutants]
    else:
        full_histo_mutants = [single_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, single_mutants,
                              quadruple_mutants, double_mutants, quadruple_mutants, double_mutants]

    for i in tqdm(range(1000)):
        hits = 0
        for j in range(1):
            # expID 1
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[0], shuffled_arr[:2]))
            # expID 2
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[1], shuffled_arr[:3]))
            # expID 3
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[2], shuffled_arr[:3]))
            # expID 4
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[3], shuffled_arr[:2]))
            # expID 5
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[4], shuffled_arr[:2]))
            # expID 6
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[5], shuffled_arr[:2]))
            # expID 7
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[6], shuffled_arr[:3]))
            # expID 8
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[7], shuffled_arr[:2]))
            # expID 10
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[8], shuffled_arr[:3]))
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[9], shuffled_arr[:2]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[10], shuffled_arr[:2]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[11], shuffled_arr[:3]))
            # # expID 17
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[12], shuffled_arr[:2]))
            
        number_of_hits.append(hits)
    
    fig = plt.figure()
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    if weak:
        observed = 2
    else:
        observed = 2
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    # plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.cumsum(h[0]), 4)[observed]), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    plt.savefig("C:\\Users\\wolfgang.kelsch\\Downloads\\mutants_in_RC_mnn3_rc3.svg")
    plt.show()
    
def mutants_in_rc_mnn4_k4(weak = False): #actually 4 mnn and rc 4
    number_of_hits = []
    single_mutants = np.array([1])
    double_mutants = np.array([1, 2])
    triple_mutants = np.array([1,2,3])
    quadruple_mutants = np.array([1,2,3,4])
    quintuple_mutants = np.array([1,2,3,4,5])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if weak:
        full_histo_mutants = [double_mutants, double_mutants,double_mutants,double_mutants,double_mutants,double_mutants,double_mutants, double_mutants, double_mutants, 
                              quadruple_mutants, quintuple_mutants, triple_mutants, quadruple_mutants, triple_mutants]
    else:
        full_histo_mutants = [single_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, single_mutants,
                              quadruple_mutants, double_mutants, quadruple_mutants, triple_mutants, double_mutants]

    for i in tqdm(range(1000)):
        hits = 0
        for j in range(1):
            # expID 1
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[0], shuffled_arr[:2]))
            # expID 2
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[1], shuffled_arr[:3]))
            # expID 3
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[2], shuffled_arr[:4]))
            # expID 4
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[3], shuffled_arr[:2]))
            # expID 5
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[4], shuffled_arr[:3]))
            # expID 6
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[5], shuffled_arr[:4]))
            # expID 7
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[6], shuffled_arr[:3]))
            # expID 8
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[7], shuffled_arr[:3]))
            # expID 10
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[8], shuffled_arr[:3]))
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[9], shuffled_arr[:3]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[10], shuffled_arr[:3]))
            # expID 14
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[11], shuffled_arr[:1]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[12], shuffled_arr[:2]))
            # # expID 17
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[13], shuffled_arr[:2]))
            
        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(17), density = True, align = 'left', label = "Expected by random chance")
    if weak:
        observed = 5
    else:
        observed = 3 
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    # plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.cumsum(h[0]), 4)[observed]), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    print(np.round(np.cumsum(h[0]), 4)[observed])
    plt.legend()
    plt.show()
    
def mutants_in_both_mnn5_k5(weak = False):
    number_of_hits = []
    single_mutants = np.array([1])
    double_mutants = np.array([1, 2])
    triple_mutants = np.array([1,2,3])
    quadruple_mutants = np.array([1,2,3,4])
    quintuple_mutants = np.array([1,2,3,4,5])
    shuffled_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if weak:
        full_histo_mutants = [double_mutants, double_mutants,double_mutants,double_mutants,double_mutants,double_mutants,double_mutants, double_mutants, double_mutants, 
                          quadruple_mutants, quintuple_mutants, triple_mutants, quadruple_mutants, triple_mutants]
    else:
        full_histo_mutants = [single_mutants, double_mutants, single_mutants, single_mutants, double_mutants, single_mutants, double_mutants, single_mutants, single_mutants,
                              quadruple_mutants, double_mutants, double_mutants, triple_mutants, triple_mutants]
    for i in tqdm(range(1000)):
        hits = 0
        for j in range(1):
            # expID 1
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[0], shuffled_arr[:3]))
            # expID 2
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[1], shuffled_arr[:5]))
            # expID 3
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[2], shuffled_arr[:5]))
            # expID 4
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[3], shuffled_arr[:4]))
            # expID 5
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[4], shuffled_arr[:4]))
            # expID 6
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[5], shuffled_arr[:4]))
            # expID 7
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[6], shuffled_arr[:2]))
            # expID 8
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[7], shuffled_arr[:4]))
            # expID 10
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[8], shuffled_arr[:4]))
            # expID 11
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[9], shuffled_arr[:3]))
            # expID 12
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[10], shuffled_arr[:4]))
            # expID 14
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[11], shuffled_arr[:2]))
            # expID 15
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[12], shuffled_arr[:2]))
            # # expID 17
            np.random.shuffle(shuffled_arr)
            hits += np.sum(np.isin(full_histo_mutants[13], shuffled_arr[:2]))
            
        number_of_hits.append(hits)
    
    if weak:
        observed = 7
    else:
        observed = 6
    h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance")
    plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k', label = "95% CI")
    # plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(observed, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(np.cumsum(h[0]), 4)[observed]), (1.5, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.title("Random chance of mutant in rich-club\n both cohorts")
    plt.xlabel("Number of mutants in sRC", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.legend()
    plt.show()
    
"""
Littermates indices, that will be shuffled to compute statistical significance under the various combination of parameters chosen for analysis
"""
shuffled_arr1 = np.array([41, 31, 2, 3, 3, 5, 6, 7, 7, 8])
shuffled_arr2 = np.array([25, 28, 40, 41, 42, 2, 5, 6, 8])
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

def littermates_in_rc_mnn2_k2():
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 2 and k = 2.
    """
    number_of_hits = []
    for i in tqdm(range(1000)):
        hits = 0
        # expID 1
        np.random.shuffle(shuffled_arr1)
        if shuffled_arr1[0] == shuffled_arr1[1]:
            hits += 1
        #expID 2
        np.random.shuffle(shuffled_arr2)
        if shuffled_arr2[0] == shuffled_arr2[1]:
            hits += 1
        #expID 3
        np.random.shuffle(shuffled_arr3)
        if shuffled_arr3[0] == shuffled_arr3[1]:
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
        if shuffled_arr7[0] == shuffled_arr7[1] or shuffled_arr7[1] == shuffled_arr7[2] or shuffled_arr7[0] == shuffled_arr7[2]:
            hits += 1
        # expID 10
        np.random.shuffle(shuffled_arr10)
        if shuffled_arr10[0] == shuffled_arr10[1] or shuffled_arr10[1] == shuffled_arr10[2] or shuffled_arr10[0] == shuffled_arr10[2]:
            hits += 1
        # expID 12   
        np.random.shuffle(shuffled_arr12)
        if shuffled_arr12[0] == shuffled_arr12[1]:
            hits += 1
        # expID 15    
        np.random.shuffle(shuffled_arr15)
        if shuffled_arr15[0] == shuffled_arr15[1] or shuffled_arr15[1] == shuffled_arr15[2] or shuffled_arr15[0] == shuffled_arr15[2]:
            hits += 1
        # expID 17
        if shuffled_arr17[0] == shuffled_arr17[1]:
                hits += 1

        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    # plt.axvline(np.percentile(number_of_hits, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(2, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[2], 4)), (3, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()

def littermates_in_rc_mnn3_k3():
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 3 and k = 3.
    """
    number_of_hits = []
    for i in tqdm(range(500)):
        hits = 0
        # expID 1
        np.random.shuffle(shuffled_arr1)
        if shuffled_arr1[0] == shuffled_arr1[1]:
            hits += 1
        #expID 2
        np.random.shuffle(shuffled_arr2)
        if shuffled_arr2[0] == shuffled_arr2[1] or shuffled_arr2[1] == shuffled_arr2[2] or shuffled_arr2[0] == shuffled_arr2[2]:
            hits += 1
        #expID 3
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
        if shuffled_arr7[0] == shuffled_arr7[1] or shuffled_arr7[1] == shuffled_arr7[2] or shuffled_arr7[0] == shuffled_arr7[2]:
            hits += 1
        # expID 8
        np.random.shuffle(shuffled_arr8)
        if shuffled_arr8[0] == shuffled_arr8[1]:
            hits += 1
        # expID 10
        np.random.shuffle(shuffled_arr10)
        if shuffled_arr10[0] == shuffled_arr10[1] or shuffled_arr10[1] == shuffled_arr10[2] or shuffled_arr10[0] == shuffled_arr10[2]:
            hits += 1
        # expID 11
        np.random.shuffle(shuffled_arr11)
        if shuffled_arr11[0] == shuffled_arr11[1]:
            hits += 1
        # expID 12   
        np.random.shuffle(shuffled_arr12)
        if shuffled_arr12[0] == shuffled_arr12[1]:
            hits += 1
        # expID 15
        np.random.shuffle(shuffled_arr15)
        if shuffled_arr15[0] == shuffled_arr15[1] or shuffled_arr15[1] == shuffled_arr15[2] or shuffled_arr15[0] == shuffled_arr15[2]:
            hits += 1
        # expID 17
        if shuffled_arr17[0] == shuffled_arr17[1]:
                hits += 1

        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    # plt.axvline(np.percentile(number_of_hits, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(3, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[3], 4)), (3, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()
    
def littermates_in_rc_mnn4_k4():
    """
    Computes the chance of finding pairs of littermates in the RC by random chance for mnn = 4 and k = 4.
    """
    number_of_hits = []
    for i in tqdm(range(1000)):
        hits = 0
        # expID 1
        np.random.shuffle(shuffled_arr1)
        if shuffled_arr1[0] == shuffled_arr1[1]:
            hits += 1
        #expID 2
        np.random.shuffle(shuffled_arr2)
        if shuffled_arr2[0] == shuffled_arr2[1] or shuffled_arr2[1] == shuffled_arr2[2] or shuffled_arr2[0] == shuffled_arr2[2]:
            hits += 1
        #expID 3
        np.random.shuffle(shuffled_arr3)
        if shuffled_arr3[0] == shuffled_arr3[1] or shuffled_arr3[0] == shuffled_arr3[2] or shuffled_arr3[0] == shuffled_arr3[3] or shuffled_arr3[1] == shuffled_arr3[2] or shuffled_arr3[1] == shuffled_arr3[3] or shuffled_arr3[2] == shuffled_arr3[3]:
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
        if shuffled_arr6[0] == shuffled_arr6[1] or shuffled_arr6[0] == shuffled_arr6[2] or shuffled_arr6[0] == shuffled_arr6[3] or shuffled_arr6[1] == shuffled_arr6[2] or shuffled_arr6[1] == shuffled_arr6[3] or shuffled_arr6[2] == shuffled_arr6[3]:
            hits += 1
        # expID 7
        np.random.shuffle(shuffled_arr7)
        if shuffled_arr7[0] == shuffled_arr7[1]:
            hits += 1
        # expID 8
        np.random.shuffle(shuffled_arr8)
        if shuffled_arr8[0] == shuffled_arr8[1] or shuffled_arr8[0] == shuffled_arr8[2] or shuffled_arr8[1] == shuffled_arr8[2]:
            hits += 1
        # expID 10
        np.random.shuffle(shuffled_arr10)
        if shuffled_arr10[0] == shuffled_arr10[1] or shuffled_arr10[1] == shuffled_arr10[2] or shuffled_arr10[0] == shuffled_arr10[2]:
            hits += 1
        # expID 11
        np.random.shuffle(shuffled_arr11)
        if shuffled_arr11[0] == shuffled_arr11[1] or shuffled_arr11[0] == shuffled_arr11[2] or shuffled_arr11[1] == shuffled_arr11[2]:
            hits += 1
        # expID 12   
        np.random.shuffle(shuffled_arr12)
        if shuffled_arr12[0] == shuffled_arr12[1] or shuffled_arr12[0] == shuffled_arr12[2] or shuffled_arr12[1] == shuffled_arr12[2]:
            hits += 1
        # expID 15
        np.random.shuffle(shuffled_arr15)
        if shuffled_arr15[0] == shuffled_arr15[1]:
            hits += 1
        # expID 17
        if shuffled_arr17[0] == shuffled_arr17[1]:
                hits += 1

        number_of_hits.append(hits)
        
    h = plt.hist(number_of_hits, bins = np.arange(10), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.95, color = 'gray')
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of litter mates in rich club\n both cohorts")
    plt.xlabel("Groups with litter mates in rich club", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    # plt.axvline(np.percentile(number_of_hits, 2.5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(4, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[4], 4)), (3, 0.2), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.legend()
    plt.show()
    
def reshuffled_rc_mnn2_k2(cumulative = False):
    """
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data_k2.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(1000)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
              
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(14), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(14), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    # plt.axvline(np.percentile(number_of_hits, 5), ls = "--", color = 'k')
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.axvline(7, color = 'red', label = "Experimentally observed")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[7], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.xlim([0, 14])
    plt.legend(loc='upper left')
  #  plt.tight_layout()
    plt.show()        
    
def reshuffled_rc_mnn3_k3(cumulative = False):
    """
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """ 
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data_k3.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(1000)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
            
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(15), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(15), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[7], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.axvline(7, color = 'red', label = "Experimentally observed")
    plt.xlim([0, 15])
    plt.legend(loc='upper left')
  #  plt.tight_layout()
    plt.show()    
    
def reshuffled_rc_mnn4_k4(cumulative = False):
    """ 
    Computes the chance of finding an RC members again in the RC after reshuffling.
    To do this, the function iterates through each individual RC member, and how often they got reshulffed (j for loop), and 
    looks at the chance it would be again in the RC via random chance.
    """
    number_of_hits = []
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_to_metadata= "..\\data\\meta_data_k4.csv"
    df = pd.read_csv(path_to_metadata)
    RCs = df.loc[df["RC"], "Mouse_RFID"].unique()
    
    for i in tqdm(range(1000)):
        hits = 0
        
        for RFID in RCs:
            reshuffled_times = np.sum(df["Mouse_RFID"] == RFID)
            if reshuffled_times == 1: # if the mouse was not reshuffled, ignore and move to next one
                continue
            else:
                corresp_group_ID = df.loc[df["Mouse_RFID"] == RFID, "Group_ID"].values # getting the group IDs where this RC mouse was reshuffled
                # getting corresponding size of the club. Since we cannot order where the true 'first' time of the RC member was, the shuffling of RC_sizes ensures no bias.
                RC_sizes = np.random.permutation([np.sum(df.loc[df["Group_ID"] == ID, "RC"]) for ID in corresp_group_ID]) 
                for j in range(reshuffled_times - 1):
                    shuffled_arr = np.random.permutation(arr)
                    hits += 1 in shuffled_arr[:RC_sizes[j]]
        number_of_hits.append(hits)
    
    if cumulative:
        h = plt.hist(number_of_hits, bins = np.arange(16), density = True, cumulative = - 1, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    else:
        h = plt.hist(number_of_hits, bins = np.arange(16), density = True, align = 'left', label = "Expected by random chance", rwidth = 0.97)
    reversed_cumsum = np.cumsum(h[0][::-1])[::-1]
    plt.title("Random chance of being shuffled back in rich-club\n both cohorts")
    plt.xlabel("Number of 'conserved' RC members", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.axvline(np.percentile(number_of_hits, 95), ls = "--", color = 'k', label = "95% CI")
    plt.annotate("p-value = "+str(np.round(reversed_cumsum[8], 3)), (8, 0.1), bbox=dict(facecolor='white', edgecolor='none', pad=1.0), ha='center')
    plt.axvline(8, color = 'red', label = "Experimentally observed")
    plt.xlim([0, 16])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()    

chasing_towards()
# mutants_in_rc_mnn2_k2()
# mutants_in_rc_mnn3_k3(True)
# mutants_in_rc_mnn4_k4(False)
# mutants_in_both_mnn5_k5(True)

# littermates_in_rc_mnn2_k2()    
# littermates_in_rc_mnn3_k3()
# littermates_in_rc_mnn4_k4()

# reshuffled_rc_mnn2_k2()
# reshuffled_rc_mnn3_k3()
# reshuffled_rc_mnn4_k4()


