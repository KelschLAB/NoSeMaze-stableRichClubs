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
path_to_data = "..\\data\\trajectories\\"
groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G10", "G11", "G12", "G13", "G14", "G15", "G16", "G17"]
setup_name = ["AM2", "AM1", "AM2", "AM2", "AM2", "AM1", "AM2", "AM1", "AM2", "AM1", "AM2", "AM3", "AM4", "AM1", "AM3", "AM4"]
path_to_groups = [path_to_data+g for g in groups]

def flip(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    span = max_val - min_val
    midpoint = min_val + span / 2
    mirrored = 2 * midpoint - arr
    return mirrored


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
        df["approacher"] = np.where(df["total_length_a"] > df["total_length_b"], "A", "B")
        
        # Filter rows by approacher/approachee
        a_mask = df["approacher"] == "A"
        b_mask = ~a_mask
        
        # Process A approachers
        approach_start_x.extend(df.loc[a_mask, "xstart_a"].tolist())
        approach_start_y.extend(df.loc[a_mask, "ystart_a"].tolist())
        approach_end_x.extend(df.loc[a_mask, "xend_a"].tolist())
        approach_end_y.extend(df.loc[a_mask, "yend_a"].tolist())
        approachee_start_x.extend(df.loc[a_mask, "xstart_b"].tolist())
        approachee_start_y.extend(df.loc[a_mask, "ystart_b"].tolist())
        approacher_distance.extend(df.loc[a_mask, "total_length_a"].tolist())
        approachee_distance.extend(df.loc[a_mask, "total_length_b"].tolist())
        
        # Process B approachers
        approach_start_x.extend(df.loc[b_mask, "xstart_b"].tolist())
        approach_start_y.extend(df.loc[b_mask, "ystart_b"].tolist())
        approach_end_x.extend(df.loc[b_mask, "xend_b"].tolist())
        approach_end_y.extend(df.loc[b_mask, "yend_b"].tolist())
        approachee_start_x.extend(df.loc[b_mask, "xstart_a"].tolist())
        approachee_start_y.extend(df.loc[b_mask, "ystart_a"].tolist())
        approacher_distance.extend(df.loc[b_mask, "total_length_b"].tolist())
        approachee_distance.extend(df.loc[b_mask, "total_length_a"].tolist())

    return approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_distance, approachee_distance 

# define plotting functions
def plot_2d_distribution(group_idx = "all", approach_type = "start_point", cutoff = 0.1, colmap = "bwr", ax = None):
    """
    Function to show the distribution of approaches as described by "approach_type", as a scatter plot
    overlayed with a fit of the distribution via a 2d gaussian distribution.
    the cutoff argument serves to rescale the colormap
    """
    paths = [path_to_data+g for g in groups]
    if group_idx == "all":
        plot_scatter = False
        approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_d, approachee_d = [], [], [], [], [], [], [], []
        for idx, p in enumerate(paths):
            data = load_data(p)
            print(len(data[0]))
            if setup_name[idx] == "AM2" or setup_name[idx] == "AM4":
                approach_start_x.append(np.array(data[0]))
                approach_start_y.append(np.array(data[1]))
                approach_end_x.append(np.array(data[2]))
                approach_end_y.append(np.array(data[3]))
                approachee_start_x.append(np.array(data[4]))
                approachee_start_y.append(np.array(data[5]))
                approacher_d.append(np.array(data[6]))
                approachee_d.append(np.array(data[7]))
            elif setup_name[idx] == "AM1":
                approach_start_x.append(np.array(data[0]))
                approach_start_y.append(flip(np.array(data[1])))
                approach_end_x.append(np.array(data[2]))
                approach_end_y.append(flip(np.array(data[3])))
                approachee_start_x.append(np.array(data[4]))
                approachee_start_y.append(flip(np.array(data[5])))
                approacher_d.append(np.array(data[6]))
                approachee_d.append(np.array(data[7]))
            elif setup_name[idx] == "AM3":
                approach_start_x.append(flip(np.array(data[0])))
                approach_start_y.append(np.array(data[1]))
                approach_end_x.append(flip(np.array(data[2])))
                approach_end_y.append(np.array(data[3]))
                approachee_start_x.append(flip(np.array(data[4])))
                approachee_start_y.append(np.array(data[5]))
                approacher_d.append(np.array(data[6]))
                approachee_d.append(np.array(data[7]))
            else:
                continue
                
        approach_start_x = np.concatenate(approach_start_x)
        approach_start_y = np.concatenate(approach_start_y)
        approach_end_x = np.concatenate(approach_end_x)
        approach_end_y = np.concatenate(approach_end_y)
        approachee_start_x = np.concatenate(approachee_start_x)
        approachee_start_y = np.concatenate(approachee_start_y)
        approacher_d = np.concatenate(approacher_d)
        approachee_d = np.concatenate(approachee_d)
        
    else:
        plot_scatter = True
        approach_start_x, approach_start_y, approach_end_x, approach_end_y, approachee_start_x, approachee_start_y, approacher_d, approachee_d = load_data(paths[group_idx])
        if setup_name[group_idx] == "AM1":
            approach_start_y = flip(np.array(approach_start_y))
            approach_end_y = flip(np.array(approach_end_y))
            approachee_start_y = flip(np.array(approachee_start_y))
        if setup_name[group_idx] == "AM3":
            approach_start_x = flip(np.array(approach_start_x))
            approach_end_x = flip(np.array(approach_end_x))
            approachee_start_x = flip(np.array(approachee_start_x))
    
    approacher_d, approachee_d = np.array(approacher_d), np.array(approachee_d)
    true_approaches = np.where(approacher_d > 1.5*approachee_d)
    
    scaling_factor = 0.1 # 1 pixel = 0.1 cm (From Luise's thesis)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(approach_type)
    else:
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    c, s, alpha = "k", 5, 0.1
    if approach_type == "start_point":
        x, y = np.array(approach_start_x)*scaling_factor, np.array(approach_start_y)*scaling_factor
        if plot_scatter:
            ax.scatter(x[true_approaches], y[true_approaches], c=c, s=s, alpha=alpha, edgecolors=None)
        
    elif approach_type == "end_point":
        x, y = np.array(approach_end_x)*scaling_factor, np.array(approach_end_y)*scaling_factor
        if plot_scatter:
            ax.scatter(x[true_approaches], y[true_approaches], c=c, s=s, alpha=alpha)

    elif approach_type == "approachee_start":
        x, y = np.array(approachee_start_x)*scaling_factor, np.array(approachee_start_y)*scaling_factor
        if plot_scatter:
            ax.scatter(x, y, c=c, s=s, alpha=alpha, edgecolors=None)
        
    elif approach_type == "interactions":   
        x, y = np.array(approach_end_x)*scaling_factor, np.array(approach_end_y)*scaling_factor
        if plot_scatter:
            ax.scatter(x, y, c=c, s=s, alpha=alpha, edgecolor='none')
            ax.set_aspect('equal')
           # plt.savefig(f"C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\plots\\interactions_heatmap\\{path_to_data[-2:]}.tif", dpi = 300)

    else:
        raise("unknwon input type")
    
    if not plot_scatter:
        # Compute 2D histogram
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        H, xedges, yedges = np.histogram2d(x, y, bins=[int(np.round(xmax)), int(np.round(ymax))], range=[[xmin, xmax], [ymin, ymax]])
        sigma = 1.25 # Adjust sigma for better smoothing
        H_smooth = gaussian_filter(H, sigma=sigma)  # Smooth the histogram with a Gaussian filter
    
        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(H_smooth.T, extent=extent, origin='lower', cmap=colmap, aspect = "equal")
        cbar = plt.colorbar(label = "Density (events / cm²)", shrink = 0.6)
        # norm = mpl.colors.Normalize(vmin=cutoff, vmax=1-cutoff)
        # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colmap), ax = ax, label = "Density (events / cm²)", shrink=0.7)
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig("C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\plots\\interactions_heatmap\\all_groups.svg", dpi = 600)
    

    
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
    fig, ax = plt.subplots(figsize=(5, 5))
    scale_factor = 0.1 # converting px to cm
    
    files = os.listdir(path_to_data)
    file = os.path.join(path_to_data, files[index])
    df = pd.read_csv(file)
    partner_x = np.array(df["x_partner_smooth"])
    partner_y = np.array(df["y_partner_smooth"])
    focus_x = np.array(df["x_focal_smooth"])
    focus_y = np.array(df["y_focal_smooth"])
    partner_x, partner_y = scale_factor*partner_x[~np.isnan(partner_x)], scale_factor*partner_y[~np.isnan(partner_y)]
    focus_x, focus_y = scale_factor*focus_x[~np.isnan(focus_x)], scale_factor*focus_y[~np.isnan(focus_y)]
    
    c_partner = "k" if df["partner_length"][0] > df["focal_length"][0] else "gray"
    c_focus = "k" if df["partner_length"][0] < df["focal_length"][0] else "gray"

    plt.plot(partner_x, partner_y, c = c_partner , label = "Approchee")
    plt.plot(focus_x, focus_y, c = c_focus, label = "Approacher")
    dx = partner_x[-1] - partner_x[-2]
    dy = partner_y[-1] - partner_y[-2]
    plt.arrow(partner_x[-1], partner_y[-1], dx, dy, head_width=scale_factor*15, head_length=scale_factor*15, fc=c_partner , ec=c_partner )
    dx = focus_x[-1] - focus_x[-2]
    dy = focus_y[-1] - focus_y[-2]
    plt.arrow(focus_x[-1], focus_y[-1], dx, dy, head_width=scale_factor*15, head_length=scale_factor*15, fc=c_focus, ec=c_focus)
    plt.xlim([0, scale_factor*500])
    plt.ylim([0, scale_factor*500])
    ax.set_xlabel("x (cm)", fontsize = 18)
    ax.set_ylabel("y (cm)", fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()

# fig, ax = plt.subplots(1, 1, figsize = (3,  2.5))

plot_2d_distribution("all", "interactions", 0, "turbo")
fig, axs = plt.subplots(4, 4, figsize = (10, 10))
for i in range(16):
    plot_2d_distribution(i, "interactions", 0, "turbo", ax = axs.flatten()[i])
    axs.flatten()[i].set_title(groups[i])
# hist_travelled_dist(path_to_data)
# plot_traj("C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\approach_meta_data\\trajectories\\", 35)