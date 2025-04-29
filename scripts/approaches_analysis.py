import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

path_to_data = "W:\\group_entwbio\\data\\Luise\\NoSeMaze2023\\DLC_output_windows\\GJ1\\D1\\G1D1_trajectoriesInfos_pre120frames.csv"
# path_to_data = "W:\\group_entwbio\\data\\Luise\\NoSeMaze2023\\DLC_output_windows\\GJ1\\meta_data\\"

def load_data(path_to_data):
    if os.path.isfile(path_to_data):
        files = [path_to_data]
    elif os.path.isdir(path_to_data):
        files = os.listdir(path_to_data)
        files = [os.path.join(path_to_data, file) for file in files]
    # Storage for descriptors
    approach_start_x, approach_start_y = [], []
    approach_end_x, approach_end_y = [], []
    approachee_start_x, approachee_start_y = [], []
    approacher_distance = []
    
    for path_to_file in tqdm(files, "Reading files ... "):
        # read data and filter out NaNs
        df = pd.read_csv(path_to_file)
        df = df.dropna()
        # Figure out who is approacher/approachee
        approacher = ["A" if df.iloc[i]["total_length_a"] > df.iloc[i]["total_length_b"] else "B" for i in range(len(df))]
        df["approacher"] = approacher
    
        for i in range(len(df)):
            if df.iloc[i]["approacher"] == "A":
                approach_start_x.append(df.iloc[i]["xstart_a"])
                approach_start_y.append(df.iloc[i]["ystart_a"])
                approach_end_x.append(df.iloc[i]["xend_a"])
                approach_end_y.append(df.iloc[i]["yend_a"])
                approachee_start_x.append(df.iloc[i]["xstart_b"])
                approachee_start_y.append(df.iloc[i]["ystart_b"])
                approacher_distance.append(df.iloc[i]["total_length_a"])
    
            elif df.iloc[i]["approacher"] == "B":
                approach_start_x.append(df.iloc[i]["xstart_b"])
                approach_start_y.append(df.iloc[i]["ystart_b"])
                approach_end_x.append(df.iloc[i]["xend_b"])
                approach_end_y.append(df.iloc[i]["yend_b"])
                approachee_start_x.append(df.iloc[i]["xstart_a"])
                approachee_start_y.append(df.iloc[i]["ystart_a"])
                approacher_distance.append(df.iloc[i]["total_length_b"])

    return approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_distance

# define plotting functions
def plot_2d_distribution(path_to_data, approach_type = "start_point"):
    """
    Function to show the distribution of approaches as described by "approach_type", as a scatter plot
    overlayed with a fit of the distribution via a 2d gaussian distribution.
    """
    approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, _ = load_data(path_to_data)
    
    plt.figure(figsize=(5, 5))
    grid_points = 50
    c, s, alpha = "k", 20, 0.3
    if approach_type == "start_point":
        x, y = approach_start_x, approach_start_y
        plt.scatter(x, y, c=c, s=s, alpha=alpha, edgecolors=None)
        plt.title("Approach start point")
        
    elif approach_type == "end_point":
        x, y = approach_end_x, approach_end_y
        plt.scatter(x, y, c=c, s=s, alpha=alpha)
        plt.title("Approach end point")

    elif approach_type == "approachee_start":
        x, y = approachee_start_x, approachee_start_y
        plt.scatter(x, y, c=c, s=s, alpha=alpha, edgecolors=None)
        plt.title("Approachee start point")
        
    # # Overlay 2D Gaussian kernel density estimate
    # xy = np.vstack([x, y])
    # kde = gaussian_kde(xy, 0.4)
    # xmin, xmax = min(x), max(x)
    # ymin, ymax = min(y), max(y)
    # X, Y = np.meshgrid(np.linspace(xmin, xmax, grid_points), np.linspace(ymin, ymax, grid_points))
    # Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    # plt.imshow(Z, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='RdPu', aspect='auto', alpha=0.7, norm=mcolors.Normalize())
    # plt.colorbar(label="Density")
    # plt.show()
    
    # Compute 2D histogram
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    H, xedges, yedges = np.histogram2d(x, y, bins=grid_points, range=[[xmin, xmax], [ymin, ymax]])
    H[np.where(H == 0)] = 1e-15
    H = np.log(H)
    
    # Smooth the histogram with a Gaussian filter
    sigma = 3  # Adjust sigma for better smoothing
    H_smooth = gaussian_filter(H, sigma=sigma)
    
    # Normalize the heatmap
    H_smooth = H_smooth / np.max(H_smooth)
    
    # Plot the heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(H_smooth.T, extent=extent, origin='lower', cmap='afmhot', aspect='auto', alpha=0.7)
    plt.xticks([])
    plt.yticks([])
  #  plt.colorbar(label="Normalized Density")
    plt.show()

    
def plot_speed(path_to_data):
    _, _, _, _, _, _, distances = load_data(path_to_data)
    speed = np.array(distances)/4
    plt.figure(figsize=(5, 5))
    plt.boxplot(speed)
    plt.ylabel("Speed in cm/s")
    plt.show()

plot_2d_distribution(path_to_data, "start_point")
# plot_2d_distribution(path_to_data, "end_point")
# plot_speed(path_to_data)