import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from copy import deepcopy
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from igraph.drawing.colors import ClusterColoringPalette
import random
from warnings import warn


path = "..\\data\\"
file = "G2\\interactions_resD7_1.csv"

def isSymmetric(mat):
    transmat = np.array(mat).transpose()
    if np.array_equal(mat, transmat):
        return True
    return False

def rescale(arr, max_val = 5):
    normalized_arr = (arr - np.min(arr))/np.max(arr)
    return normalized_arr*max_val

def inverse(arr):
    """
    Computes the 'inverse' of a graph matrix by taking 1/x where x is an entry
    only when x != 0 and represents the edge values (or distances between vertices).
    """
    inv_arr = deepcopy(arr)
    inv_arr[inv_arr == 0] = 10e-10#np.Inf
    inv_arr = 1/inv_arr
    return inv_arr

def read_graph(path_to_file):
    arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
    data = arr[1:, 1:].astype(float)
    return data
    

def display_graph(path_to_file, ax, **kwargs):
    """
    This function displays the graph to analyze and colors the vertices/edges according
    to the given input parameters.

    Parameters
    ----------
    path_to_file : string
        path specifying where the file containing the matrix representing the graph 
        is stored. Should be a .csv file.
    ax : matplotlib.axis
        the axis to plot the graph onto.
    **kwargs : strings
        layout : specifies which layout to use for displaying the graph. see igraph documentation for 
            a detailed list of all layout. should be given as a string as stated in the igraph doc.
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.

    """
    random.seed(1) #to make sure all graphs are always displayed with same coordinates when changing metrics
    
    if "layout" in kwargs:
        layout_style = kwargs["layout"]
    else:
        layout_style = "fr"
            
    if len(path_to_file) > 1:
        warn("More than one input graph path has been provided. \n Multilayer plotting is not yet supported. Only first one will be displayed.")
        data = read_graph(path_to_file[0])
    else:
        data = read_graph(path_to_file[0])

    # arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
    # data = arr[1:, 1:].astype(float)
    # rounded_data = data.round(2)
    # inv_data = inverse(data)
    
    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
        # inv_g = ig.Graph.Weighted_Adjacency(inv_data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        # inv_g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    cmap1 = LinearSegmentedColormap.from_list("vertex_cmap", ["blue", "red"])
    cmap1 = cm.Reds
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            node_color = "blue"
            node_size = 15
        elif kwargs["node_metric"] == "betweenness":
            edge_betweenness = g.betweenness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_betweenness = ig.rescale(edge_betweenness)
            node_size = [(1+e)*15 for e in edge_betweenness]
            node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            edge_strength = ig.rescale(edge_strength)
            node_size = [(1+e)*15 for e in edge_strength]
            node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            edge_closeness = g.closeness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_closeness = ig.rescale(edge_closeness)
            node_size = [(1+e)*15 for e in edge_closeness]
            node_color = [cmap1(b) for b in edge_closeness]
        elif kwargs["node_metric"] == "hub score":
            edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
            edge_hub = ig.rescale(edge_hub)
            node_size = [(1+e)*15 for e in edge_hub]
            node_color = [cmap1(b) for b in edge_hub]
        elif kwargs["node_metric"] == "authority score":
            edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
            edge_authority = ig.rescale(edge_authority)
            node_size = [(1+e)*15 for e in edge_authority]
            node_color = [cmap1(b) for b in edge_authority]
        elif kwargs["node_metric"] == "eigenvector centrality":
            edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
            edge_hub = ig.rescale(edge_evc)
            node_size = [(1+e)*15 for e in edge_evc]
            node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
            edge_pagerank = ig.rescale(edge_pagerank)
            node_size = [(1+e)*15 for e in edge_pagerank]
            node_color = [cmap1(b) for b in edge_pagerank]
    else:
        node_color = "red"
        node_size = 15
        
    if "idx" in kwargs:
        if len(kwargs["idx"]) == 0:
            marker_frame_color = node_color
        else:
            cmap = get_cmap('Spectral')
            palette = ClusterColoringPalette(kwargs["cluster_num"])
            marker_frame_color = [palette[i] for i in kwargs["idx"]]#cmap(kwargs["idx"])

    layout = g.layout(layout_style)
    visual_style = {}
    visual_style["vertex_size"] = node_size
    visual_style["vertex_color"] = node_color
    visual_style["edge_arrow_width"] = 5
    visual_style["edge_width"] = rescale(np.array([w['weight'] for w in g.es]))
    visual_style["layout"] = layout
    visual_style["vertex_frame_color"] = marker_frame_color
    visual_style["edge_curved"] = 0
    visual_style["vertex_frame_width"] = 3
    # g.vs["label"] = [v.index for v in g.vs()]
    # visual_style["vertex_label_size"] = 20
    # visual_style["vertex_label_dist"] = 0.5


    # visual_style["vertex_font"] = "Times"
    ig.plot(g, target=ax, **visual_style)
    

# f, a = plt.subplots(1,1)
# display_graph(path+file, a, node_metric = "betweenness")