import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from copy import deepcopy
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from igraph.drawing.colors import ClusterColoringPalette
import random
import os
from copy import copy
from warnings import warn
from tkinter import messagebox as mb
from clustering_algorithm import *
from multilayer_plot import *
import pandas as pd

def isSymmetric(mat):
    transmat = np.array(mat).transpose()
    if np.array_equal(mat, transmat):
        return True
    return False

def rescale(arr, max_val = 5):
    normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    return normalized_arr*max_val

def inverse(arr, remove_feedback_loops = True):
    """
    Computes the 'inverse' of a graph matrix by taking 1/x where x is an entry
    only when x != 0 and represents the edge values (or distances between vertices).
    """
    inv_arr = deepcopy(arr)
    inv_arr[inv_arr == 0] = 10e-10 #np.Inf
    inv_arr = 1/inv_arr
    if remove_feedback_loops: # affinity for each element with itself is set to 0 in this case.
        idx = np.diag_indices(inv_arr.shape[0], 2)
        inv_arr[idx] = 0 
    return inv_arr

def nn_cut(arr, nn = 2):
    """
    Cuts the edges of the graph in the input array by keeping only the edges that are nearest neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.  The cut is NOT direction specific,
    meaning that two nodes are nearest neighbors if they are nearest neighbors in the outgoing direction 
    OR in the ingoing direction.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
    assert nn >= 1, "nn should be a positive integer bigger than 1." # nn = 0 means self ineractions with is trivially true (unless stated otherwise)
    nn_arr = np.zeros_like(arr) 
    neighbors_i = np.argsort(-arr, 1) #computing the nearest neighbors for local nn estimation
    neighbors_j = np.argsort(-arr, 0)
    for i in range(nn_arr.shape[0]):
        # nearest neighbours keep their value
        nn_arr[i, neighbors_i[i, :nn]] = arr[i, neighbors_i[i, :nn]]
        nn_arr[neighbors_j[:nn, i], i] = arr[neighbors_j[:nn, i], i] 
        # other elements are put to 0 (0.01 for visualization).
        nn_arr[i, neighbors_i[i, nn:]] = 0.01
        nn_arr[neighbors_j[nn:, i], i] = 0.01  

    return nn_arr

def mnn_cut(arr, nn = 2):
    """
    Cuts the edges of the graph in the input array by keeping only the edges that are mutual nearest neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
    assert nn >= 1, "nn should be a positive integer bigger than 1." # nn = 0 means self ineractions with is trivially true (unless stated otherwise)
    mnn_arr = np.zeros_like(arr) + 0.01 #0.01 for visualization, to force igraph to keep layout
    neighbors_i = np.argsort(-arr, 1) #computing the nearest neighbors for local nn estimation
    neighbors_j = np.argsort(-arr, 0)
    for i in range(arr.shape[1]):
        for j in range(i, arr.shape[1]):
            if any(np.isin(neighbors_i[i, 0:nn], j)) and any(np.isin(neighbors_j[0:nn, j], i)): #local nearest distance estimation
                mnn_arr[i, j] += arr[i, j]
                mnn_arr[j, i] += arr[j, i]
    return mnn_arr

def read_labels(path_to_file):
    """
    Reads a list of labels for the nodes to plot in the graph

    Parameters
    ----------
    path_to_file : TYPE string or list of string containing the path to the file(s)

    Returns
    -------
    TYPE
        list of nodes labels
    """

    if type(path_to_file) == str:
        arr = pd.read_csv(path_to_file)
        labels = arr.iloc[:, 0]
        return [l for l in labels]
    
    elif type(path_to_file) == list:
        arr = pd.read_csv(path_to_file[0])
        labels = arr.iloc[:, 0]
        return [l for l in labels]
        
def read_graph(path_to_file, percentage_threshold = 0.01, mnn = None, return_ig = False,
               avg_graph = False, affinity = True, rm_fb_loops = True, mutual = True):
    """
    Reads a file containing the weights defning the adjacency matrix. 

    Parameters
    ----------
    path_to_file : TYPE string or list of string containing the path to the file(s)
    percentage_threshold : parameter specifying the edge value threshold under which edges are not displayed.
        should be given as a percentage of the maximum edge value. Default is 0.0
    mnn : number of nearest neighbours for graph cut. Default is None
    return_ig : TYPE, optional
        Whether or not to return the read graphs as an ig.Graph. if false, numpy arrays are returned.
        The default is False.
    affinity: TYPE Bool. Whethere or not the data stored in the graph represents affinity between nodes or distance.
        If true, the graph is read as is, otherwise the values in the input matrice(s) are inverted.

    Returns
    -------
    TYPE
        list of graphs as np array or ig.Graph
    """

    random.seed(1) #making sure layout of plots stays the same when changing metrics

    if not avg_graph:
        #accounting for potential multiple layers
        data = []
        for i in range(len(path_to_file)):
            arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
            layer_data = arr[1:, 1:].astype(float)
            if not affinity: # if graph edges represent distances and not affinities, need to invert values
               layer_data = inverse(layer_data, rm_fb_loops)

            threshold = np.max(layer_data) * (percentage_threshold / 100.0)
            layer_data = np.where(layer_data < threshold, 0.01, layer_data)  #0.01 for visualization, to force igraph to keep layout
            if mnn is not None:
                if mutual:
                    layer_data = mnn_cut(layer_data, mnn) # mutual nearest neighbours
                else:
                    layer_data = nn_cut(layer_data, mnn)  # nearest neighbours

            data.append(layer_data)
        if return_ig: # return_ig specifies if the output should be returned as an ig.Graph.
            layers = [ig.Graph.Weighted_Adjacency(d, mode='directed') for d in data]
            return layers
        return data
    
    elif avg_graph:
        # averaged graph
        arr = np.loadtxt(path_to_file[0], delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)/len(path_to_file)
        for i in range(1, len(path_to_file)):
            arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
            data += arr[1:, 1:].astype(float)/len(path_to_file) 
        if not affinity:
           data = inverse(data, rm_fb_loops)
           
        threshold = np.max(data) * (percentage_threshold / 100.0)
        data = np.where(data < threshold, 0.01, data)
        if mnn is not None:
            if mutual:
                data = mnn_cut(data, mnn)
            else:
                data = nn_cut(data, mnn)
        if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
            avg_layer = ig.Graph.Weighted_Adjacency(data, mode='directed') 
            return [avg_layer]
        return [data]
        
def randomize_graph(path_to_file, percentage_threshold = 0.01, mnn = None, return_ig = False,
                    avg_graph = False, affinity = True, mutual = True):
    """
    Reads a file containing the weights defning the adjacency matrix, then randomizes the graph 
    for bootstrap purposes.

    see 'read_graph' for input params.
    """

    random.seed(1) #making sure layout of plots stays the same when changing metrics

    if len(path_to_file) == 1:
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        np.random.shuffle(data) # randomizing graph while keeping weights constant.
        if not affinity: # If input data represents distance, need to invert values
            data = inverse(data)
        threshold = np.max(data) * (percentage_threshold / 100.0)
        data = np.where(data < threshold, 0.01, data)
        
        if mnn is not None:
            if mutual:
                data = mnn_cut(data, mnn)
            else:
                data = nn_cut(data, mnn)

        if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
            graph = [ig.Graph.Weighted_Adjacency(data, mode='directed')]
            return graph
        return [data]
    
    elif len(path_to_file) > 1:
        arr = np.loadtxt(path_to_file[0], delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)/len(path_to_file)
        for i in range(1, len(path_to_file)):
            arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
            data += arr[1:, 1:].astype(float)/len(path_to_file) 
        np.random.shuffle(data) # randomizing graph while keeping weights constant.
        if not affinity: # If input data represents distance, need to invert values
            data = inverse(data)
        threshold = np.max(data) * (percentage_threshold / 100.0)
        data = np.where(data < threshold, 0.01, layer_data)  #0.01 for visualization, to force igraph to keep layout
        if mnn is not None:
            if mutual:
                data = mnn_cut(data, mnn)
            else:
                data = nn_cut(data, mnn)

        if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
            avg_layer = ig.Graph.Weighted_Adjacency(data, mode='directed')
            return [avg_layer]
        return [data]
        
def rich_club_weights(graph, k, min_val = 0.05):
    """
    Computes the rich-club for input graph: each node that has a degree < k is not part of it.

    Parameters
    ----------
    graph : the graph to compute onto
    k: the degree for rich-club computation
    min_val: minimum weight value (defaults to 0.05)

    Returns
    -------
    A list of nodes weights where each member of the rich-club is 1 and others are 'min_val'
    """

    weights = []
    for i in range(len(graph.vs())):
        incident_edges = np.array([e["weight"] for e in graph.vs[i].incident()])
        if np.sum(incident_edges > 0.01) >= k:
            weights.append(1)
        else:
            weights.append(min_val)
    return weights


def k_core_weights(data, k, min_val = 0.05):
    """
    Computes the k_core(s) for input graph: nodes that have at least k connexdion among themselves are
    part of k-core.

    Parameters
    ----------
    graph : the graph to compute onto
    k: the degree for rich-club computation
    min_val: minimum weight value (defaults to 0.05)

    Returns
    -------
    A list of nodes weights where each member of the rich-club is 1 and others are 'min_val'
    """
    
    if type(data) != np.ndarray:
        data = np.array(data.get_adjacency(attribute="weight").data)
    
    weights = []
    # First figure out rich-club within data matrix
    binary_data = copy(data)
    binary_data[data > min_val] = 1
    binary_data[data <= min_val] = 0
    core_matrix = binary_data
    core_matrix[np.sum(binary_data, 1) < k, :] = 0
    core_matrix[:, np.sum(binary_data, 0) < k] = 0
    # Then, figure out if members of rich-club are connected enough
    for i in range(core_matrix.shape[0]):
        if np.sum(core_matrix[i, :]) >= k:
            weights.append(1)
        else:
            weights.append(min_val)
    return weights

def rich_club_size(graph, k, min_val = 0.05):
    """
    Returns the size of the rich_club of input graph  for degree = k.
    """
    sizes = rich_club_weights(graph, k)
    core_size = 0
    for s in sizes:
        if s > 0.1:
            core_size += 1
    return core_size


def k_core_size(graph, k, min_val = 0.05):
    """
    Returns the size of the k core of input graph for degree = k.
    """
    sizes =  k_core_weights(graph, k)
    core_size = 0
    for s in sizes:
        if s > 0.1:
            core_size += 1
    return core_size



def rich_club_p_value(path_to_file, k, percentage_threshold, mnn, affinity = True, bootstrap_iter = 250, mutual = True):
    """
    Computes an estimation of the k core size in the random case to give a p-value for
    the computed rich-club in the input graph. Does so by randomizing the network (keep weights same) 
    this is done 'bootstrap_iter' times and allows to the distribution of the random rich_club values.

    Parameters
    ----------
    path_to_file 
    k : Int. degree value for the rich-club computation
    bootstrap_iter : Int. number of iterations for the bootstrap estimation.

    Returns
    -------
    p_value

    """
    if len(path_to_file) > 1:
        mb.showwarning(title = "Warning", message = "The measurment of the rich-club p-value is given for the averaged graph, as no multilayer single confidence interval can be given. ")
    input_layer = randomize_graph(path_to_file, percentage_threshold=percentage_threshold,\
                                       mnn=mnn,return_ig= True, avg_graph = True, affinity = affinity, mutual = mutual)
    actual_observation = rich_club_size(input_layer, k)
    random_observations = np.zeros(bootstrap_iter)
    for i in range(bootstrap_iter):
        random_layer = randomize_graph(path_to_file, percentage_threshold=percentage_threshold,\
                                       mnn=mnn,return_ig= True, avg_graph = True, affinity = affinity, mutual = mutual)
        random_observations[i] = rich_club_size(random_layer, k)
    p_value = np.sum(random_observations == actual_observation)/bootstrap_iter
    return p_value

def community_clustering(path_to_file):
    """
    Clusters input graph into communities, follow the optimal community algorithm

    Parameters
    ----------
    path_to_file : list of path to graph files

    Returns 
    -------
    idx: list of indexes for the nodes.
    """
    if len(path_to_file) > 1:
        mb.showwarning(title = "Warning", message = "Community will be computed on the averaged graph, as the algorithm cannot deal with multilyer information.")
        data = read_graph(path_to_file, avg_graph=True, affinity = affinity)
    else:
        data = read_graph(path_to_file, affinity = affinity)
    
    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    communities = g.community_optimal_modularity(weights = [1/(e['weight']) for e in g.es()])

    total_length = data.shape[0]
    idx = [0 for i in range(total_length)]
    for i in range(len(communities)):
        for j in communities[i]:
            idx[j] = i
    return idx
    
def display_graph(path_to_file, ax, percentage_threshold = 0.0, mnn = None, avg_graph = False,
                  affinity = True, rm_fb_loops = True, mutual = True, **kwargs):
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
    threshold : parameter specifying the edge value threshold under which edges are not displayed.
    mnn : number of nearest neighbours for graph cut
    avg_graph: Bool. In case of multilayers, whether or not to plot the averaged graph instead.
    affinity: Whether the edges represent an affinity measurement (if True) or a distance (if False) (i.e. high value = high similarity or 
        high value = high difference). Defaults to True.
    rm_fb_loops: Bool. Whether or not to remove the feedback loops (edges of nodes with themselves) in graph display.
    **kwargs : strings
        layout : specifies which layout to use for displaying the graph. see igraph documentation for 
            a detailed list of all layout. should be given as a string as stated in the igraph doc.
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.
    """

    random.seed(1) #making sure layout of plots stays the same when changing metrics
    
    if "layout" in kwargs:
        layout_style = kwargs["layout"]
    else:
        layout_style = "fr"
        
    node_labels = read_labels(path_to_file) if "node_labels" not in kwargs else kwargs["node_labels"]
        
    if len(path_to_file) > 1 and not avg_graph:
        layer_labels = kwargs["layer_labels"] if "layer_labels" in kwargs else None
        display_graph_3d(path_to_file, ax = ax, percentage_threshold = percentage_threshold, mnn = mnn, affinity = affinity, \
                         rm_fb_loops = rm_fb_loops, mutual = mutual, layout = layout_style, node_metric = kwargs["node_metric"], idx = kwargs["idx"], \
                         cluster_num = kwargs["cluster_num"], layer_labels = layer_labels, node_labels = node_labels, deg = kwargs["deg"], 
                         node_size = kwargs["node_size"], edge_width = kwargs["edge_width"], scale_edge_width = kwargs["scale_edge_width"],
                         between_layer_edges = kwargs["between_layer_edges"])
        return
    else:
        data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, mutual = mutual, \
                          avg_graph = avg_graph, affinity = affinity, rm_fb_loops = rm_fb_loops)[0]

    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    # default values
    node_color = "gray"
    default_node_size = kwargs["node_size"] if "node_size" in kwargs else 15
    default_edge_width = kwargs["edge_width"] if "edge_width" in kwargs else 5

    # cmap1 = cm.Reds
    cmap1 = cm.Grays
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            node_size = default_node_size
        if kwargs["node_metric"] == "betweenness":
            edge_betweenness = g.betweenness(weights = [1/e['weight'] for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_betweenness = ig.rescale(edge_betweenness)
            node_size = [(1+e)*default_node_size for e in edge_betweenness]
            node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            edge_strength = ig.rescale(edge_strength)
            node_size = [(1+e)*default_node_size for e in edge_strength]
            node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            edge_closeness = g.closeness(weights = [1/e['weight'] for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_closeness = ig.rescale(edge_closeness)
            node_size = [(1+e)*default_node_size for e in edge_closeness]
            node_color = [cmap1(b) for b in edge_closeness]
        elif kwargs["node_metric"] == "hub score":
            edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
            edge_hub = ig.rescale(edge_hub)
            node_size = [(1+e)*default_node_size for e in edge_hub]
            node_color = [cmap1(b) for b in edge_hub]
        elif kwargs["node_metric"] == "authority score":
            edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
            edge_authority = ig.rescale(edge_authority)
            node_size = [(1+e)*default_node_size for e in edge_authority]
            node_color = [cmap1(b) for b in edge_authority]
        elif kwargs["node_metric"] == "eigenvector centrality":
            random.seed(1)
            edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
            edge_evc = ig.rescale(edge_evc)
            node_size = [(1+e)*default_node_size for e in edge_evc]
            node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
            edge_pagerank = ig.rescale(edge_pagerank)
            node_size = [(1+e)*default_node_size for e in edge_pagerank]
            node_color = [cmap1(b) for b in edge_pagerank]
        elif kwargs["node_metric"] == "rich-club":
            k_degree = kwargs["deg"]
            node_size = rich_club_weights(g, k_degree, 0.3)
            # node_color = [cmap1(0.5) if b == 1 else cmap1(0.2) for b in node_size]
            node_color = ["gray" for b in node_size]
            node_size = [n*default_node_size for n in node_size]
        elif kwargs["node_metric"] == "k-core":
            k_degree = kwargs["deg"]
            node_size = k_core_weights(data, k_degree, 0.5)
            node_color = [cmap1(0.99) if b == 1 else cmap1(0.2) for b in node_size]
            node_size = [n*default_node_size for n in node_size]
        
    if "idx" in kwargs:
        if len(kwargs["idx"]) == 0:
            marker_frame_color = node_color
        else:
            cmap = get_cmap('Spectral')
            values, _ = np.unique(kwargs["idx"], return_counts=True)
            palette = ClusterColoringPalette(len(values))
            marker_frame_color = [palette[i] for i in kwargs["idx"]]#cmap(kwargs["idx"])
    else:
        marker_frame_color = node_color
        
    layout = g.layout(layout_style)
    visual_style = {}
    visual_style["vertex_size"] = node_size
    visual_style["vertex_color"] = node_color
    visual_style["vertex_frame_color"] = marker_frame_color
    # visual_style["vertex_frame_color"] = [(0.5, 0.5, 0.5, 0)] * g.vcount()
    edge_cmap = get_cmap('Greys')
    visual_style["edge_arrow_width"] = rescale(np.array([w['weight'] for w in g.es]), default_edge_width)*(default_edge_width)
    
    if "scale_edge_width" in kwargs and type(kwargs["scale_edge_width"]) == bool:
        if kwargs["scale_edge_width"]: #if true, adapt edge_thickness to edge value, else all edges are shown with same width.
            display_edge_width = rescale(np.array([w['weight'] for w in g.es]), default_edge_width)
            edge_color = [edge_cmap(edge) for edge in rescale(np.array([w['weight'] for w in g.es])) - 0.01]
        else:
            display_edge_width = np.array([0.99 if w['weight'] > 0.01 else 0 for w in g.es])*default_edge_width
            edge_color = [edge_cmap(0.99) for edge in g.es]
    else:
        display_edge_width = rescale(np.array([w['weight'] for w in g.es]), default_edge_width)
        edge_color = [edge_cmap(edge) for edge in rescale(np.array([w['weight'] for w in g.es])) - 0.01]

    visual_style["edge_width"] = display_edge_width
    visual_style["edge_color"] = edge_color
    
    visual_style["layout"] = layout

    if isSymmetric(data):
        visual_style["edge_curved"] = 0.0
    else:
        visual_style["edge_curved"] = 0.2
    visual_style["vertex_frame_width"] = 3
    visual_style["vertex_label"] = node_labels
    # g.vs["label"] =  node_labels#[v.index for v in g.vs()]
    g.vs["name"] = node_labels
    # visual_style["vertex_label_size"] = 20
    # visual_style["vertex_label_dist"] = 0.5

    visual_style["vertex_font"] = "Times"
    ig.plot(g, target=ax, **visual_style)
    
def display_graph_3d(path_to_file, ax, percentage_threshold = 0.0, mnn = None, affinity = True, rm_fb_loops = True, mutual = True, **kwargs):
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
    threshold : parameter specifying the edge value threshold under which edges are not displayed.
    mnn : number of nearest neighbours for graph cut
    affinity: Whether the edges represent an affinity measurement (if True) or a distance (if False) (i.e. high value = high similarity or 
        high value = high difference). Defaults to True.
    rm_fb_loops: Bool. Whether or not to remove feedback loops from graph display
    **kwargs : strings
        layout : specifies which layout to use for displaying the graph. see igraph documentation for 
            a detailed list of all layout. should be given as a string as stated in the igraph doc.
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.

    """
    random.seed(1) #making sure layout of plots stays the same when changing metrics

    layers_layout = read_graph(path_to_file, percentage_threshold = 0, mnn = None, return_ig=True, affinity = affinity, rm_fb_loops = rm_fb_loops, mutual = mutual) #here to make sure layout stays consistent upon graph cut
    layers = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, return_ig=True, affinity = affinity, rm_fb_loops = rm_fb_loops, mutual = mutual) 
    layers_data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, return_ig=False, affinity = affinity, rm_fb_loops = rm_fb_loops, mutual = mutual)

    default_node_size = kwargs["node_size"] if "node_size" in kwargs else 15
    default_edge_width = kwargs["edge_width"] if "edge_width" in kwargs else 5
    node_color = "red"
    node_size = []
    for g in layers:
        size = np.array([default_node_size for v in range(g.vcount())])
        node_size.append(size)
        
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            node_size = []
            for g in layers:
                size = np.array([default_node_size for v in range(g.vcount())])
                node_size.append(size)
                
        elif kwargs["node_metric"] == "betweenness":
            node_size = []
            for g in layers:
                edge_betweenness = g.betweenness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_betweenness = ig.rescale(edge_betweenness)
                node_size.append(np.array(edge_betweenness)*default_node_size+0.07)
        elif kwargs["node_metric"] == "strength":
            node_size = []
            for g in layers:
                edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
                edge_strength = ig.rescale(edge_strength)
                node_size.append(np.array(edge_strength)*default_node_size+0.07)
        elif kwargs["node_metric"] == "closeness":
            node_size = []
            for g in layers:
                edge_closeness = g.closeness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_closeness = ig.rescale(edge_closeness)
                node_size.append(np.array(edge_closeness)*default_node_size+0.07)
        elif kwargs["node_metric"] == "hub score":
            node_size = []
            for g in layers:
                edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
                edge_hub = ig.rescale(edge_hub)
                node_size.append(np.array(edge_hub)*default_node_size+0.07)
        elif kwargs["node_metric"] == "authority score":
            node_size = []
            for g in layers:
                edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
                edge_authority = ig.rescale(edge_authority)
                node_size.append(np.array(edge_authority)*default_node_size+0.07)
        elif kwargs["node_metric"] == "eigenvector centrality":
            node_size = []
            for g in layers:
                edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
                edge_evc = ig.rescale(edge_evc)
                node_size.append(np.array(edge_evc)*default_node_size+0.07)
        elif kwargs["node_metric"] == "page rank":
            node_size = []
            for g in layers:
                edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
                edge_pagerank = ig.rescale(edge_pagerank)
                node_size.append(np.array(edge_pagerank)*default_node_size+0.07)
        elif kwargs["node_metric"] == "rich-club":
            node_size = []
            for g in layers:
                k_degree = kwargs["deg"]
                size = rich_club_weights(g, k_degree, 0.01)
                node_size.append(np.array([n*default_node_size for n in size]))
        elif kwargs["node_metric"] == "k-core":
            node_size = []
            for d in layers_data:
                k_degree = kwargs["deg"]
                size = k_core_weights(d, k_degree, 0.01)
                node_size.append(np.array([n*default_node_size for n in size]))
        
    if "idx" in kwargs:
        if len(kwargs["idx"]) == 0:
            marker_frame_color = None
        else:
            color_num = len(np.unique(kwargs["idx"]))
            cmap = get_cmap('Spectral')
            palette = ClusterColoringPalette(color_num)
            marker_frame_color = [palette[i] for i in kwargs["idx"]]#cmap(kwargs["idx"])
    else:
        marker_frame_color = None

    if "layout" in kwargs: 
        if kwargs["layout"] == "circle": 
            layout=nx.circular_layout
        elif kwargs["layout"] == "large" or kwargs["layout"] == "fr":
            layout=nx.spring_layout
        elif kwargs["layout"] == "kk": 
            layout=nx.kamada_kawai_layout
        elif kwargs["layout"] ==  "random": 
            layout=nx.random_layout
        elif kwargs["layout"] ==  "drl": 
            layout=nx.spectral_layout
        elif kwargs["layout"] == "tree":
            layout = nx.planar_layout
        else:
            layout=nx.spring_layout
    else:
        layout=nx.spring_layout
        
    if "node_labels" in kwargs:
        node_labels = kwargs["node_labels"]
    else:
        node_labels = None
        
    if "layer_labels" in kwargs:
        layer_labels = kwargs["layer_labels"]
    else:
        layer_labels = None
    
    scale_edge_width = kwargs["scale_edge_width"] if "scale_edge_width" in kwargs and type(kwargs["scale_edge_width"]) == bool else True   
    LayeredNetworkGraph(layers_layout, layers, layers_data, ax=ax, layout=layout, 
                        node_labels = node_labels, nodes_width=node_size, node_edge_colors=marker_frame_color, 
                        layer_labels=layer_labels, default_edge_width=default_edge_width,
                        scale_edge_width = scale_edge_width, between_layer_edges = kwargs["between_layer_edges"])
    ax.set_axis_off()

    
def display_stats(path_to_file, ax, percentage_threshold = 0.0, mnn = None, affinity = True, avg_graph = False,
                  mutual = True, node_metric = "none", stacked = True, **kwargs):
    """
    This function displays a histogram representation of the metrics of the graph to analyze.

    Parameters
    ----------
    path_to_file : string
        path specifying where the file containing the matrix representing the graph 
        is stored. Should be a .csv file.
    ax : matplotlib.axis
        the axis to plot the graph onto.
    mnn: whether or not to do a graph cut based on mutual nearest neighbors. If an 'int' is provided,
        this represents the number of nn to take into account.
    affinity: bool. Whether or not the graph type is affinity or distance.
    **kwargs : strings
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.

    """    
            
    if len(path_to_file) != 1 and not avg_graph:
        display_stats_multilayer(path_to_file, ax, percentage_threshold, mnn, affinity, mutual,
                                 node_metric = node_metric, deg = kwargs["deg"], stacked = stacked)
        return
    else:
        data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, affinity = affinity, 
                          avg_graph = avg_graph, mutual = mutual)[0]

    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    if node_metric == "none":
        edge_strength = [e['weight'] for e in g.es()]
        ax.hist(edge_strength)

    elif node_metric == "betweenness":
        edge_betweenness = g.betweenness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
        edge_betweenness = ig.rescale(edge_betweenness)
        ax.hist(edge_betweenness)
    elif node_metric == "strength":
        edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
        edge_strength = ig.rescale(edge_strength)
        ax.hist(edge_strength)
    elif node_metric == "closeness":
        edge_closeness = g.closeness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
        edge_closeness = ig.rescale(edge_closeness)
        ax.hist(edge_closeness)
    elif node_metric == "hub score":
        edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
        edge_hub = ig.rescale(edge_hub)
        ax.hist(edge_hub)
    elif node_metric == "authority score":
        edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
        edge_authority = ig.rescale(edge_authority)
        ax.hist(edge_authority)
    elif node_metric == "eigenvector centrality":
        edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
        edge_evc = ig.rescale(edge_evc)
        ax.hist(edge_evc)
    elif node_metric == "page rank":
        edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
        edge_pagerank = ig.rescale(edge_pagerank)
        ax.hist(edge_pagerank)
    elif node_metric == "rich-club":
        k_degree = kwargs["deg"]
        for i in range(len(path_to_file)):
            g = read_graph(path_to_file[i], percentage_threshold, mnn, return_ig=True)[0]
            core_size = rich_club_size(g, k_degree)
            p_value = rich_club_p_value(path_to_file[i], k_degree, percentage_threshold, mnn)
            ax.text(0.1, 0.1+0.1*i, os.path.basename(path_to_file[i])+": rich-club size = "+str(core_size)+" p_value = "+str(p_value),\
                    transform=ax.transAxes, fontsize=15)
        ax.axis("off")
    else: 
        edge_strength = [e['weight'] for e in g.es()]
        ax.hist(edge_strength)
        ax.set_xlabel("Edge values")
        ax.set_title("No 'node metric' was selected, showing edge values.")
        return
        
    ax.set_ylabel("Count")
    if node_metric != "none":
        ax.set_xlabel(node_metric +" values")
    else:
        ax.set_xlabel("Edge values")
        ax.set_title("No 'node metric' was selected, showing edge values.")
            
def display_stats_multilayer(path_to_file, ax, percentage_threshold = 0.0, mnn = None, affinity = True, mutual = True,
                             node_metric = "none",stacked = True, **kwargs):
    """
    This function displays a histogram representation of the metrics of the graphs to analyze.

    Parameters
    ----------
    path_to_file : string
        path specifying where the file containing the matrix representing the graph 
        is stored. Should be a .csv file.
    ax : matplotlib.axis
        the axis to plot the graph onto.
    mnn: whether or not to do a graph cut based on mutual nearest neighbors. If an 'int' is provided,
        this represents the number of nn to take into account.
    affinity: bool. Whether or not the graph type is affinity or distance.
    **kwargs : strings
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.
    """    

    data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, affinity = affinity, mutual = mutual)
    graph_data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn,
                            affinity = affinity, mutual = mutual, return_ig = True)

    if node_metric == "none":
        node_size = []
        for g in graph_data:
            edge_strength = [e['weight'] for e in g.es()]
            node_size.append(np.array(edge_strength))
            
    elif node_metric == "betweenness":
        node_size = []
        for g in graph_data:
            edge_betweenness = g.betweenness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            node_size.append(np.array(edge_betweenness))
    elif node_metric == "strength":
        node_size = []
        for g in graph_data:
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            node_size.append(np.array(edge_strength))
    elif node_metric == "closeness":
        node_size = []
        for g in graph_data:
            edge_closeness = g.closeness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            node_size.append(np.array(edge_closeness))
    elif node_metric == "hub score":
        node_size = []
        for g in graph_data:
            edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
            node_size.append(np.array(edge_hub))
    elif node_metric == "authority score":
        node_size = []
        for g in graph_data:
            edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
            node_size.append(np.array(edge_authority))
    elif node_metric == "eigenvector centrality":
        node_size = []
        for g in graph_data:
            edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
            node_size.append(np.array(edge_evc))
    elif node_metric == "page rank":
        node_size = []
        for g in graph_data:
            edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
            node_size.append(np.array(edge_pagerank))

    elif node_metric == "rich-club":
        node_size = []
        k_degree = kwargs["deg"]
        for g in graph_data:
            core_size = rich_club_size(g, k_degree)
            node_size.append([core_size])
    else: 
        node_size = []
        for g in graph_data:
            edge_strength = [e['weight'] for e in g.es()]
            node_size.append(np.array(edge_strength))
        ax.hist(node_size, histtype='bar', stacked=stacked, rwidth = 0.8)
        ax.set_ylabel("Count")
        labels_legend = [os.path.basename(path).split(".")[0] for path in path_to_file]
        ax.legend(labels=labels_legend)
        ax.set_xlabel("Edge values")
        ax.set_title("No 'node metric' was selected, showing edge values.")
        return
    
    ax.hist(node_size, histtype='bar', stacked=stacked, rwidth = 0.8)
    ax.set_ylabel("Count")
    labels_legend = [os.path.basename(path).split(".")[0] for path in path_to_file]
    ax.legend(labels=labels_legend)
    if node_metric != "none":
        ax.set_xlabel(node_metric+" values")
    else:
        ax.set_xlabel("Edge values")
        ax.set_title("No 'node metric' was selected, showing edge values.")

        
        
if __name__ == '__main__':

    path = "..\\data\\social network matrices 3days\\G5\\"
    file1 = "approaches_resD3_1.csv"
    file2 = "approaches_resD3_2.csv"
    file3 = "approaches_resD3_3.csv"
    file4 = "interactions_resD3_2.csv"

    data = read_graph([path+file1], mnn = 3, return_ig=False)[0]
    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
    
    # c = community_clustering(path+file)
    # f = plt.Figure()
    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(111, projection='3d')
    display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "k-core", mutual = True, idx = [], cluster_num = None)
    # plt.show()
    # c = display_stats([path+file1, path+file2, path+file3], ax = ax, mnn = 4, node_metric = "rich-club", deg = 2)
    # g = read_graph(path+file, return_ig=True)
    # display_graph([path+"\\interactions_resD7_1.csv", path+"\\interactions_resD7_1.csv"], a, node_metric = "closeness", deg =3, cluster_num = 2, idx = [1, 1, 1, 0,0,1,1,1,1,1])
    
# clusterer = graphClusterer(D, True, "fully connected")
# cluster_num = 2
# clusterer.k_elbow_curve(a)
# nn = 4
# _, idx, _, _ = clusterer.clustering(cluster_num, isAffinity = True)
# clusterer.sigma_grid_search(a, 30, 2)
