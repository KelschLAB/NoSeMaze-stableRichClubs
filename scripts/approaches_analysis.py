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
import matplotlib.pyplot as plt
import matplotlib as mpl

# path_to_data = "W:\\group_entwbio\\data\\Luise\\NoSeMaze2023\\DLC_output_windows\\GJ1\\D1\\G1D1_trajectoriesInfos_pre120frames.csv"
path_to_data = "C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\approach_meta_data\\GJ1\\"

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
    approacher_distance, approachee_distance = [], []
    
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
                approachee_distance.append(df.iloc[i]["total_length_b"])

            elif df.iloc[i]["approacher"] == "B":
                approach_start_x.append(df.iloc[i]["xstart_b"])
                approach_start_y.append(df.iloc[i]["ystart_b"])
                approach_end_x.append(df.iloc[i]["xend_b"])
                approach_end_y.append(df.iloc[i]["yend_b"])
                approachee_start_x.append(df.iloc[i]["xstart_a"])
                approachee_start_y.append(df.iloc[i]["ystart_a"])
                approacher_distance.append(df.iloc[i]["total_length_b"])
                approachee_distance.append(df.iloc[i]["total_length_a"])

    return approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_distance, approachee_distance 

# define plotting functions
def plot_2d_distribution(path_to_data, approach_type = "start_point", cutoff = 0.1, colmap = "bwr"):
    """
    Function to show the distribution of approaches as described by "approach_type", as a scatter plot
    overlayed with a fit of the distribution via a 2d gaussian distribution.
    the cutoff argument serves to rescale the colormap
    """
    approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_d, approachee_d = load_data(path_to_data)
    
    approacher_d, approachee_d = np.array(approacher_d), np.array(approachee_d)
    true_approaches = np.where(approacher_d > 1.5*approachee_d)
    
    scaling_factor = 0.1 # 1 pixel = 0.1 cm (From Luise's thesis)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    c, s, alpha = "k", 12, 0.05
    if approach_type == "start_point":
        x, y = np.array(approach_start_x)*scaling_factor, np.array(approach_start_y)*scaling_factor
        plt.scatter(x[true_approaches], y[true_approaches], c=c, s=s, alpha=alpha, edgecolors=None)
        plt.title("Approach start point")
        
    elif approach_type == "end_point":
        x, y = np.array(approach_end_x)*scaling_factor, np.array(approach_end_y)*scaling_factor
        plt.scatter(x[true_approaches], y[true_approaches], c=c, s=s, alpha=alpha)
        plt.title("Approach end point")

    elif approach_type == "approachee_start":
        x, y = np.array(approachee_start_x)*scaling_factor, np.array(approachee_start_y)*scaling_factor
        plt.scatter(x, y, c=c, s=s, alpha=alpha, edgecolors=None)
        plt.title("Approachee start point")
        
    elif approach_type == "interactions":   
        x, y = np.array(approach_end_x)*scaling_factor, np.array(approach_end_y)*scaling_factor
        plt.scatter(x, y, c=c, s=s, alpha=alpha)
        plt.title("Interactions")
        
    else:
        raise("unknwon input type")
    
    # Compute 2D histogram
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    H, xedges, yedges = np.histogram2d(x, y, bins=[int(np.round(xmax)), int(np.round(ymax))], range=[[xmin, xmax], [ymin, ymax]])
    H[np.where(H == 0)] = 1e-15
    H = np.log(H)
    max_val = np.max(H) # to recover correct scaling of colorbar after plotting (imshow normalizes colormap from 0 to 1)

    
    # Smooth the histogram with a Gaussian filter
    sigma = 0.42 # Adjust sigma for better smoothing
    H_smooth = gaussian_filter(H, sigma=sigma)
    
    # Normalize the heatmap
    H_smooth = H_smooth / np.max(H_smooth)
    H_smooth[H_smooth < cutoff] = 0
    H_smooth[H_smooth > 1 - cutoff] = 1
    
    # Plot the heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(H_smooth.T, extent=extent, origin='lower', cmap=colmap, aspect='auto')
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    # cbar = plt.colorbar(label="Density")
    norm = mpl.colors.Normalize(vmin=cutoff, vmax=1-cutoff)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colmap), ax = ax, label = "Density (events / cm²)", shrink=0.7)
    ticks = np.array([float(t.get_text().replace('−','-')) for t in cbar.ax.get_yticklabels()])
    cbar.set_ticks([ticks[i] for i in range(1, len(ticks), 2)])
    ticks_val = [int(np.round(np.exp(t*max_val))) for t in ticks]
    cbar.set_ticklabels([ticks_val[i] for i in range(1, len(ticks_val), 2)])
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()
    
# define plotting functions
def plot_approach_quiver(path_to_data, approach_type = "start_point"):
    """
    Function to show the average approach directionality
    """
    xs, ys, xe, ye, _, _, _, _ = load_data(path_to_data) # xstart, ystart, xend, yend
    
    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    grid_points = 50
    c, s, alpha = "k", 20, 0.1
    
    xs, ys = np.array(xs), np.array(ys)
    xe, ye = np.array(xe), np.array(ye)
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x = np.linspace(xmin, xmax, grid_points)
    y = np.linspace(ymin, ymax, grid_points)
    xspace, yspace = np.abs(x[1] - x[0]), np.abs(y[1] - y[0])
    xv, yv = np.meshgrid(x, y)
    density, U, V = np.zeros((grid_points, grid_points)), np.zeros((grid_points, grid_points)), np.zeros((grid_points, grid_points))
    
    for i, xcoord in enumerate(x):
        for j, ycoord in enumerate(y):
            xind = np.logical_and(xs > xcoord - xspace, xs < xcoord + xspace)
            yind = np.logical_and(ys > ycoord - yspace, ys < ycoord + yspace)
            valid_ind = np.where(np.logical_and(xind, yind))
            density[i, j] = np.sum(valid_ind)
            if len(valid_ind[0]) > 0:  # Check if valid_ind is not empty
                 U[j, i] = np.mean(xe[valid_ind] - xs[valid_ind])
                 V[j, i] = np.mean(ye[valid_ind] - ys[valid_ind])
            else:
                U[j, i] = 0  # Default value for empty cells
                V[j, i] = 0  # Default value for empty cells
            
    density[np.where(density == 0)] = 1e-15
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(density))
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Reds"), ax = ax, label = "density of approaches")
    density = np.sqrt(density)
    # density = density - np.min(density)
    # normalized_density = density / np.max(density)  # Normalize to [0, 1]
    # normalized_density[np.where(normalized_density == 0)] = 0.00001
    # cmap = matplotlib.cm.get_cmap('Greys')
    # plt.quiver(xv, yv, U.T, V.T, [d for d in normalized_density], cmap = "bone_r", linewidth = 20)#, alpha = normalized_density)
    ax.imshow(density, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='Reds', aspect='auto')

    ax.quiver(xv, yv, U.T/10, V.T/10)#, alpha = normalized_density)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(r"Mean approach direction per 50 px²")
    plt.show()

    
def hist_travelled_dist(path_to_data):
    _, _, _, _, _, _, distances, _ = load_data(path_to_data)
    scaling_factor = 0.1 # cm per pixel (Luise's thesis)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.hist(np.array(distances)*scaling_factor, 10, color = "gray", rwidth=0.9, edgecolor = "k", alpha = 0.7)
    plt.xlabel("Approach length in cm", fontsize = 15)
    plt.yscale("log")
    plt.ylabel("Count", fontsize = 15)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_traj(path_to_data, index):
    files = os.listdir(path_to_data)
    file = os.path.join(path_to_data, files[index])
    df = pd.read_csv(file)
    partner_x = df["x_partner_smooth"]
    partner_y = df["y_partner_smooth"]
    focus_x = df["x_focal_smooth"]
    focus_y = df["y_focal_smooth"]
    plt.plot(partner_x, partner_y, c = "gray", label = "Approchee")
    plt.plot(focus_x, focus_y, c = "k", label = "Approacher")
    plt.xlim([0, 680])
    plt.ylim([0, 480])
    plt.legend()
    plt.show()
    
# plot_2d_distribution(path_to_data, "start_point", 0, "hot_r")
# plot_2d_distribution(path_to_data, "interactions", 0, "bwr")
# hist_travelled_dist(path_to_data)
plot_traj("C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\approach_meta_data\\trajectories\\", 24)