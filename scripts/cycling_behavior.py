import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import tkinter as tk
from tkinter import filedialog
from scipy.stats import binom, spearmanr
from itertools import product
from termcolor import colored
import warnings


path = "\\\\zisvfs12\\Home\\wolfgang.kelsch\\Documents\\GitHub\\data_julia"
path = "W:\\group_entwbio\\data\\Jonathan\\NoSeMaze_Experiment\\data\\raw\\cohort01\\"
path = "C:\\Users\\Corentin offline\\Downloads\\cohorts\\"

def count_cycles(inter_trial = 10, verbose = False, path = None, **kwargs):
    #load an read data
    if path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir=path, title = "Select a file for p_value estimation")
    else:
        file_path = path
    df = pd.read_csv(file_path, sep=";")
    RFIDs, timeouts, timestamps = df["a_animal_id"].to_numpy(), df["k_timeout"].to_numpy(), df["b_timestamp"].to_numpy()
    
    # filtering out 'default' and NaNs
    if verbose:
        print("\nDefault values observed "+ str(len(np.where(RFIDs == "default")[0]))+" times\n")
    where_not_default = np.where(RFIDs != "default")
    RFIDs, timeouts, timestamps = RFIDs[where_not_default], timeouts[where_not_default], timestamps[where_not_default]
    where_not_nan = np.where(timestamps != np.nan)
    RFIDs, timeouts, timestamps, times = RFIDs[where_not_nan], timeouts[where_not_nan], timestamps[where_not_nan], []

    # convert data timestamps to absolute time in seconds
    for t, time in enumerate(timestamps):
        try:
            year = int(time[0:4])
            month = int(time[5:7])          
            day = int(time[8:10])
            hour = int(time[11:13])
            minutes = int(time[14:16])
            seconds = int(time[17:19])
        except Exception as e:
            print(e)
            continue
        date = datetime(year, month, day, hour, minutes, seconds)
        times.append(date.timestamp())
        
    # identify repeating motifs and which mice are involved in them
    repetitions = []
    repeting_mice = []      
    for i in range(len(RFIDs)-4):
        # checking for ABABC sequences
        if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] != RFIDs[i+4]): 
            if (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                repetitions.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]))
                repeting_mice.append(RFIDs[i])
                repeting_mice.append(RFIDs[i+1])
                    
        # checking for ABABAC sequences          
        if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] != RFIDs[i+5]): 
            if (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                repetitions.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]))
                repeting_mice.append(RFIDs[i])
                repeting_mice.append(RFIDs[i+1])

        # checking for ABABABC sequences          
        elif (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] == RFIDs[i+5] and RFIDs[i] != RFIDs[i+6]): 
            if (times[i+5] - times[i+5] < inter_trial) and (times[i+5] - times[i+4] < inter_trial) and (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                repetitions.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]+", "+RFIDs[i+5]))
                repeting_mice.append(RFIDs[i])
                repeting_mice.append(RFIDs[i+1])
                    
        # checking for ABABABAC sequences
        if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] == RFIDs[i+5] and RFIDs[i] == RFIDs[i+6] and RFIDs[i+1] != RFIDs[i+7]): 
            if (times[i+6] - times[i+5] < inter_trial) and (times[i+5] - times[i+4] < inter_trial) and (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                repetitions.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]+", "+RFIDs[i+5]+", "+RFIDs[i+6]))
                repeting_mice.append(RFIDs[i])
                repeting_mice.append(RFIDs[i+1])
              
    # display results                           
    if verbose:
        min_rep = kwargs["min_rep"] if "min_rep" in kwargs else 1
        for name, number in Counter(repeting_mice).most_common():
            print(name+ ", "+str(number)+" times in cycle")
        print("________________________________")
        for name, number in Counter(repetitions).most_common():
            if number >= min_rep:
                if len(name.split(",")) == 4: 
                    name += "            "+"            "+"            "
                elif len(name.split(",")) == 5: 
                    name += "            "+"            "
                elif len(name.split(",")) == 6: 
                    name += "            "
                print(f"{name}  occured {number} times")
                
    return repeting_mice

def count_cycles_surrogate(motif_length, repetitions, inter_trial = 10, iterations = 100, path = None):
    """
    Returns the probability that motifs of length 'motif_length' repeat at least 'repetitions' times by random chance 
    in a randomized data of same length as the input, and respecting the condition that members of a motif should not 
    differ more than 'inter_trial' seconds between two timesteps. This is achieved by randomizing 'iteration' times and
    seeing how often this was true.
    """
    hits = 0 # variable for storing the probability

    # extracting timestamps to get distribution of ITI
    if path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir=path, title = "Select a file for p_value estimation")
    else:
        file_path = path
        
    # convert data timestamps to absolute time in s, and extract timeseries of ITI
    df = pd.read_csv(file_path, sep=";")
    RFIDs, timestamps = df["a_animal_id"].to_numpy(), df["b_timestamp"].to_numpy()
    names, times = [], []
    for t, time in enumerate(timestamps): # extract date information
        try:
            year = int(time[0:4])
            month = int(time[5:7])          
            day = int(time[8:10])
            hour = int(time[11:13])
            minutes = int(time[14:16])
            seconds = int(time[17:19])
        except Exception as e:
            print(e)
            continue
        date = datetime(year, month, day, hour, minutes, seconds) # get time in seconds
        times.append(date.timestamp())
        names.append(RFIDs[t])
    names, times = np.array(names), np.array(times)
    delta_t = np.array([times[i+1] - times[i] for i in range(len(times)-1) if names[i] != "default"]) #timeseries of ITI for all mice (default exclulded)
    
    # create a surrogate timeseries of timestamps mimicking the statistics of original file
    np.random.shuffle(delta_t)
    times = np.cumsum(np.concatenate((delta_t, np.random.choice(delta_t, len(RFIDs) - len(delta_t) + 1))))
    data_len = len(times)

    # iterate n times and see how often motifs of length 'motif_length' are repeated at least 'repetitions' times.
    for idx in tqdm(range(iterations)):
        reps = []
        RFIDs = np.random.randint(0, 10, data_len).astype(str)
        # check for motifs of length 'motif_length' 
        for i in range(len(RFIDs)-4):
            # checking for ABABC sequences
            if motif_length == 4:
                if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] != RFIDs[i+4]): 
                    if (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                        reps.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]))
                        
            # checking for ABABAC sequences       
            elif motif_length == 5:
                if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] != RFIDs[i+5]): 
                    if (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                        reps.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]))

            # checking for ABABABC sequences
            elif motif_length == 6:     
                if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] == RFIDs[i+5] and RFIDs[i] != RFIDs[i+6]): 
                    if (times[i+5] - times[i+4] < inter_trial) and (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                        reps.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]+", "+RFIDs[i+5]))
                        
            elif motif_length == 7: 
                if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] == RFIDs[i+5] and RFIDs[i] == RFIDs[i+6] and RFIDs[i+1] != RFIDs[i+7]): 
                    if (times[i+6] - times[i+5] < inter_trial) and (times[i+5] - times[i+4] < inter_trial) and (times[i+4] - times[i+3] < inter_trial) and (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
                        reps.append((RFIDs[i]+", "+RFIDs[i+1]+", "+RFIDs[i+2]+", "+RFIDs[i+3]+", "+RFIDs[i+4]+", "+RFIDs[i+5]))
                        
        # check if motifs of length 'motif_length' repeated at least 'repetitions' times
        numbers = np.array([Counter(reps).most_common()[i][1] for i in range(len(Counter(reps).most_common()))])
        if any(numbers >= repetitions):
            hits += 1
                    
    return hits/iterations # return probability

def format_results(save_file, threshold = 0.75):
    """
    A function that looks, for each cohort, for repeting motifs in the corresponding lickport data,
    and identifies the mice that initiate the motifs the most often. Saves the results in 'save_file' 
    and creates a new column if it does not exist or update the existing one otherwise.
    Input:
        save_file: the file in which the data results will be saved.
        threshold: quantile value for deciding which mice are engaging most often in cycling behavior.
    """
    df = pd.read_csv(save_file, sep=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Drop unnamed columns if they exist
    df["cooperative"] = [False for i in range(len(df["Mouse_RFID"]))] # initialize column to false for all mice
    df["cycles_number"] = [None for i in range(len(df["Mouse_RFID"]))] # initialize column to false for all mice

    # cohort_names = df["cohort"].unique()
    cohort_names = ["cohort01", "cohort02", "cohort03", "cohort04", "cohort05", 
                    "cohort06", "cohort07", "cohort08", "cohort09", "cohort10", "cohort11", "cohort12", "cohort14",
                    "cohort16", "cohort17","cohort18", "cohort19", "cohort20", "cohort21"]
    for cohort in cohort_names:
        # extract the data for this particular cohort
        cohort_path = "C:\\Users\\Corentin offline\\Downloads\\cohorts\\"+cohort
        data_files = os.listdir(cohort_path)
        sorted_files = sorted(data_files, key=lambda f: os.path.getsize(os.path.join(cohort_path, f)), reverse=True)
        data_path = os.path.join(cohort_path, sorted_files[0]) # analyse the largest data file, as folder contains multiple files
        # Get the list of mice that were in cycles
        mice_in_cycles = Counter(count_cycles(inter_trial = 10, verbose = False, path = data_path)).most_common() 
        # get how often the mice were involved in cycles
        mice_names = np.array([mouse[0] for mouse in mice_in_cycles])
        repetitions = np.array([mouse[1] for mouse in mice_in_cycles])
        # record number of cycles per mouse
        for i, mouse_name in enumerate(mice_names):
            try:
                df.loc[(df["Mouse_RFID"] == mouse_name) & (df["cohort"] == cohort), "cycles_number"] = repetitions[i]
            except:
                warnings.warn("RFID tag discrepency. Skipping this one...")
        # Get list of cooperative mice, i.e. in the top quantile of involvment in cycles 
        qtl = np.quantile(repetitions, threshold)
        cooperative_mice = [mouse[0] for mouse in mice_in_cycles if mouse[1] > qtl] 
        for mouse_name in cooperative_mice:
            try:
                df.loc[(df["Mouse_RFID"] == mouse_name) & (df["cohort"] == cohort), "cooperative"] = True
            except:
                warnings.warn("RFID tag discrepency. Skipping this one...")
                
    df.to_csv(save_file, index = False) # index = True would create an index column, but there is already one in original data
    
def cooperation_correlation(path_to_metadata, variable: str, plot = False):
    """
    Computes the correlation coefficient between the number of cycles performed by mice 
    and the variable specified in input.
    """
    df = pd.read_csv(path_to_metadata, sep=",")
    try:
        where_nan = (df[variable].isnull()) | (df["cycles_number"].isnull())
    except:
        raise Exception("Incorrect 'variable' input argument. Please make sure it is a column of the input data.")
    x, y = df[variable].loc[~where_nan], df.loc[~where_nan, "cycles_number"]
    res, p = spearmanr(x, y)
    if np.abs(res) > 0.4 and p < 0.05:
        color = 'green'  
    elif np.abs(res) > 0.2:
        color = 'white'
    else:
        color = 'red'
    print(colored(f"Correlation with {variable}: {round(res, 3)}, p-value: {round(p, 3)}", color))
    if plot:
        plt.figure()
        plt.scatter(x, y, c = 'k')
        plt.ylabel("Number of cycles")
        plt.xlabel(variable)
        plt.show()
    
# def time_spent_cycling_distribution(inter_trial = 10, iterations = 100, path = None):
#     # extracting timestamps to get distribution of ITI
#     if path is None:
#         root = tk.Tk()
#         root.withdraw()
#         file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir=path, title = "Select a file for p_value estimation")
#     else:
#         file_path = path
        
#     df = pd.read_csv(file_path, sep=";")
#     RFIDs, timestamps = df["a_animal_id"].to_numpy(), df["b_timestamp"].to_numpy()
#     names, times = [], []
#     for t, time in enumerate(timestamps):
#         try:
#             year = int(time[0:4])
#             month = int(time[5:7])          
#             day = int(time[8:10])
#             hour = int(time[11:13])
#             minutes = int(time[14:16])
#             seconds = int(time[17:19])
#         except Exception as e:
#             print(e)
#             continue
#         date = datetime(year, month, day, hour, minutes, seconds)
#         times.append(date.timestamp())
#         names.append(RFIDs[t])
#     names, times = np.array(names), np.array(times)
#     delta_t = np.array([times[i+1] - times[i] for i in range(len(times)-1) if names[i] != "default"])
#     # create a surrogate timeseries of timestamps mimicking the statistics of original file
#     np.random.shuffle(delta_t)
#     times = np.cumsum(np.concatenate((delta_t, np.random.choice(delta_t, len(RFIDs) - len(delta_t) + 1))))
#     data_len = len(times)

#     cycling_times = [] # container for all the time spent cycling values, to obtaion the distribution from
#     for idx in tqdm(range(iterations)):
#         RFIDs = np.random.randint(0, 10, data_len).astype(str)
#         time_in_cycles = {name: 0 for name in np.arange(10).astype(str)} # time surrogate mice spent cycling in this iteration
#         for i in range(len(RFIDs)-10):
#             # checking for ABABC sequences
#             if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] != RFIDs[i+4]): 
#                 if (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
#                     time_in_cycles[RFIDs[i]] += times[i+3] - times[i]
#                     time_in_cycles[RFIDs[i+1]] += times[i+3] - times[i]

#             # checking for ABABAC sequences       
#             if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] != RFIDs[i+5]): 
#                 if (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
#                     time_in_cycles[RFIDs[i]] += times[i+4] - times[i]
#                     time_in_cycles[RFIDs[i+1]] += times[i+4] - times[i]

#             # checking for ABABABC sequences
#             if (RFIDs[i] == RFIDs[i+2]) and (RFIDs[i+1] == RFIDs[i+3]) and (RFIDs[i] != RFIDs[i+1] and RFIDs[i] == RFIDs[i+4] and RFIDs[i+1] == RFIDs[i+5] and RFIDs[i] != RFIDs[i+6]): 
#                 if (times[i+3] - times[i+2] < inter_trial) and (times[i+2] - times[i+1] < inter_trial) and (times[i+1] - times[i] < inter_trial): # checking that time between trials is less than inter_trial seconds
#                     time_in_cycles[RFIDs[i]] += times[i+5] - times[i]
#                     time_in_cycles[RFIDs[i+1]] += times[i+5] - times[i]
                            
#         for time in time_in_cycles.values():
#             cycling_times.append(time)
    
#     plt.hist(cycling_times)
#     return cycling_times


# format_results("C:\\Users\\Corentin offline\\Downloads\\LickportData(2).csv")
# df = pd.read_csv("C:\\Users\\Corentin offline\\Downloads\\LickportData(2).csv", sep=",")
# print("-----------------------------------------------------------------------------------")
# for var in list(df.columns.values):
#     cooperation_correlation("C:\\Users\\Corentin offline\\Downloads\\LickportData(2).csv", var)
# print("-----------------------------------------------------------------------------------")

count_cycles(12, True)#, path "C:\\Users\\Corentin offline\\Downloads\\cohorts\\cohort05\\trials_2020-10-23T145225")
# cooperation_correlation("C:\\Users\\Corentin offline\\Downloads\\LickportData(2).csv", "cuberoot_cs_minus_modulation_averaged", True)
# cooperation_correlation("C:\\Users\\Corentin offline\\Downloads\\LickportData(2).csv", "pause_duration_at_US_rev1", True)


# mice = count_cycles(10, True, min_rep = 2)
# print(count_cycles_surrogate(7, 2, 12, 300, "C:\\Users\\Corentin offline\\Downloads\\cohorts\\cohort05\\lickport\\trials_2020-10-23T145225.csv"))
# ts = time_spent_cycling_distribution(10, 100)

# for i in range(15):
#     print(i)
#     print(count_cycles(i))