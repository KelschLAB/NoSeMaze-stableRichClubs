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

def inverse(arr):
    """
    Computes the 'inverse' of a graph matrix by taking 1/x where x is an entry
    only when x != 0 and represents the edge values (or distances between vertices).
    """
    inv_arr = deepcopy(arr)
    inv_arr[inv_arr == 0] = 10e-10 #np.Inf
    inv_arr = 1/inv_arr
    return inv_arr

def nn_cut(arr, nn = 2):
    """
    Cuts the edges of the graph in the input array by keeping only the edges that are mutual neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
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
    Cuts the edges of the graph in the input array by keeping only the edges that are mutual neighbors.
    Nodes that are further away the 'nn' neighbours have their edges cut.

        Parameter:
                arr: array representing the graph
                nn: number of nearest neighbors to keep
        return:
                array representing the graph with cut edges
    """
    assert nn > 1, "nn should be bigger than 1."
    nn -= 1
    # arr[arr == 0] = -1*np.inf
    # if not(isSymmetric(arr)):
    #     warn("Input graph is directed, will compute nearest neighbours instead.")
    #     mnn_arr = nn_cut(arr, nn)
    #     return mnn_arr
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
        
def read_graph(path_to_file, percentage_threshold = 0.01, mnn = None, return_ig = False, avg_graph = False):
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

    Returns
    -------
    TYPE
        list of graphs as np array or ig.Graph
    """

    random.seed(1) #making sure layout of plots stays the same when changing metrics

    if type(path_to_file) == str:
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        threshold = np.max(data) * (percentage_threshold / 100.0)
        data = np.where(data < threshold, 0.01, data)
        if mnn is not None:
            data = mnn_cut(data, mnn)
        if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
            graph = [ig.Graph.Weighted_Adjacency(data, mode='directed')]
            return graph
        return [data]
    
    if type(path_to_file) == list:
        if not avg_graph:
            ## separated layers
            data = []
            for i in range(len(path_to_file)):
                arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
                layer_data = arr[1:, 1:].astype(float)
                threshold = np.max(layer_data) * (percentage_threshold / 100.0)
                layer_data = np.where(layer_data < threshold, 0.01, layer_data)  #0.01 for visualization, to force igraph to keep layout
                if mnn is not None:
                    layer_data = mnn_cut(layer_data, mnn)
                data.append(layer_data)
                # data.append(rescale(arr[1:, 1:].astype(float), 10)) #normalizing input data
            if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
                layers = [ig.Graph.Weighted_Adjacency(d, mode='directed') for d in data]
                return layers
            return data
        
        elif avg_graph:
            # averaged graph
            arr = np.loadtxt(path_to_file[0], delimiter=",", dtype=str)
            data = rescale(arr[1:, 1:].astype(float), 1)/len(path_to_file) #normalizing input data # average graph
            for i in range(1, len(path_to_file)):
                arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
                data += rescale(arr[1:, 1:].astype(float), 1)/len(path_to_file) #normalizing input data
                # data.append(rescale(arr[1:, 1:].astype(float), 1)) #normalizing input data
            if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
                layers = ig.Graph.Weighted_Adjacency(data, mode='directed') 
                return data
            return data
        
def randomize_graph(path_to_file, percentage_threshold = 0.01, mnn = None, return_ig = False):
    """
    Reads a file containing the weights defning the adjacency matrix, then randomizes the graph 
    for bootstrap purposes.

    see 'read_graph' for input params.
    """

    random.seed(1) #making sure layout of plots stays the same when changing metrics

    if type(path_to_file) == str:
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        np.random.shuffle(data) # randomizing graph while keeping weights constant.
        threshold = np.max(data) * (percentage_threshold / 100.0)
        data = np.where(data < threshold, 0.01, data)
        if mnn is not None:
            data = mnn_cut(data, mnn)
        if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
            graph = [ig.Graph.Weighted_Adjacency(data, mode='directed')]
            return graph
        return [data]
    
    if type(path_to_file) == list:
        if not avg_graph:
            ## separated layers
            data = []
            for i in range(len(path_to_file)):
                arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
                layer_data = arr[1:, 1:].astype(float)
                np.random.shuffle(layer_data) # randomizing graph while keeping weights constant.
                threshold = np.max(layer_data) * (percentage_threshold / 100.0)
                layer_data = np.where(layer_data < threshold, 0.01, layer_data)  #0.01 for visualization, to force igraph to keep layout
                if mnn is not None:
                    layer_data = mnn_cut(layer_data, mnn)
                data.append(layer_data)
                # data.append(rescale(arr[1:, 1:].astype(float), 10)) #normalizing input data
            if return_ig: # return ig specifies if the read graphs should be returned as an ig.Graph.
                layers = [ig.Graph.Weighted_Adjacency(d, mode='directed') for d in data]
                return layers
            return data
        
        
def k_core_weights(graph, k, min_val = 0.05):
    """
    Computes the k-core for input graph: each node that has a degree < k is not part of it.

    Parameters
    ----------
    graph : the graph to compute onto
    k: the degree for k-core computation
    min_val: minimum weight value (defaults to 0.05)

    Returns
    -------
    A list of nodes weights where each member of the k-core is 1 and others are 'min_val'
    """

    weights = []
    for i in range(len(graph.vs())):
        incident_edges = np.array([e["weight"] for e in graph.vs[i].incident()])
        if np.sum(incident_edges > 0.01) >= k:
            weights.append(1)
        else:
            weights.append(min_val)
    return weights

def k_core_size(graph, k, min_val = 0.05):
    """
    Returns the size of the k_core of input graph  for degree = k.
    """
    sizes = k_core_weights(graph, k)
    core_size = 0
    for s in sizes:
        if s > 0.1:
            core_size += 1
    return core_size

def k_core_p_value(path_to_file, k, percentage_threshold, mnn, bootstrap_iter = 250):
    """
    Computes an estimation of the k core size in the random case to give a p-value for
    the computed k-core in the input graph. Does so by randomizing the network (keep weights same) 
    this is done 'bootstrap_iter' times and allows to the distribution of the random k_core values.

    Parameters
    ----------
    path_to_file 
    k : Int. degree value for the k-core computation
    bootstrap_iter : Int. number of iterations for the bootstrap estimation.

    Returns
    -------
    p_value

    """
    input_layer = randomize_graph(path_to_file, percentage_threshold=percentage_threshold,\
                                       mnn=mnn,return_ig= True)[0]
    actual_observation = k_core_size(input_layer, k)
    random_observations = np.zeros(bootstrap_iter)
    for i in range(bootstrap_iter):
        random_layer = randomize_graph(path_to_file, percentage_threshold=percentage_threshold,\
                                       mnn=mnn,return_ig= True)[0]
        random_observations[i] = k_core_size(random_layer, k)
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
    if type(path_to_file) == list:
        warn("Computing communities over averaged graph.")
        data = read_graph(path_to_file, avg_graph=True)
    else:
        data = read_graph(path_to_file)
    
    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
        # inv_g = ig.Graph.Weighted_Adjacency(inv_data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        # inv_g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    communities = g.community_optimal_modularity(weights = [1/(e['weight']) for e in g.es()])
    # communities = g.community_walktrap(weights = [1/(e['weight']) for e in g.es()], steps = 40).as_clustering()

    total_length = data.shape[0]
    idx = [0 for i in range(total_length)]
    for i in range(len(communities)):
        print(communities[i])
        for j in communities[i]:
            idx[j] = i
    return idx
    
def display_graph(path_to_file, ax, percentage_threshold = 0.0, mnn = None, **kwargs):
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
        
    warn("Metric computation only supports affinity type graph at the moment. Correct that.")    
    node_labels = read_labels(path_to_file)
        
    if type(path_to_file) == list and len(path_to_file) > 1:
        layer_labels = kwargs["layer_labels"] if "layer_labels" in kwargs else None
        warn("Multilayer integration for statistics still needs to be implemented.")
        display_graph_3d(path_to_file, ax = ax, percentage_threshold = percentage_threshold, mnn = mnn, layout = layout_style, \
                         node_metric = kwargs["node_metric"], idx = kwargs["idx"], cluster_num = kwargs["cluster_num"], layer_labels = layer_labels, \
                             node_labels = node_labels, deg = kwargs["deg"])
        return
    else:
        data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn)[0]
    
    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')

    # default values
    node_color = "blue"
    node_size = 15

    cmap1 = cm.Reds
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            node_color = "blue"
            node_size = 15
        elif kwargs["node_metric"] == "betweenness":
            edge_betweenness = g.betweenness(weights = [1/e['weight'] for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_betweenness = ig.rescale(edge_betweenness)
            node_size = [(1+e)*15 for e in edge_betweenness]
            node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            edge_strength = ig.rescale(edge_strength)
            node_size = [(1+e)*15 for e in edge_strength]
            node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            edge_closeness = g.closeness(weights = [1/e['weight'] for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
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
            random.seed(1)
            edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
            edge_evc = ig.rescale(edge_evc)
            node_size = [(1+e)*15 for e in edge_evc]
            node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
            edge_pagerank = ig.rescale(edge_pagerank)
            node_size = [(1+e)*15 for e in edge_pagerank]
            node_color = [cmap1(b) for b in edge_pagerank]
        elif kwargs["node_metric"] == "k-core":
            k_degree = kwargs["deg"]
            node_size = k_core_weights(g, k_degree, 0.2)
            node_color = [cmap1(0.99) if b == 1 else cmap1(0.2) for b in node_size]
            node_size = [n*20 for n in node_size]
        
    if "idx" in kwargs:
        if len(kwargs["idx"]) == 0:
            marker_frame_color = node_color
        else:
            cmap = get_cmap('Spectral')
            palette = ClusterColoringPalette(kwargs["cluster_num"])
            marker_frame_color = [palette[i] for i in kwargs["idx"]]#cmap(kwargs["idx"])
    else:
        marker_frame_color = node_color
        
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
    visual_style["vertex_label"] = node_labels
    # g.vs["label"] =  node_labels#[v.index for v in g.vs()]
    g.vs["name"] = node_labels
    # visual_style["vertex_label_size"] = 20
    # visual_style["vertex_label_dist"] = 0.5

    # visual_style["vertex_font"] = "Times"
    ig.plot(g, target=ax, **visual_style)
    
def display_graph_3d(path_to_file, ax, percentage_threshold = 0.0, mnn = None, **kwargs):
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
    layers_layout = read_graph(path_to_file, return_ig=True) #here to make sure layout stays consistent upon graph cut
    layers = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn, return_ig=True)

    node_size = 15 #default value
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            # node_color = "blue"
            node_size = 15
    
        elif kwargs["node_metric"] == "betweenness":
            node_size = []
            for g in layers:
                edge_betweenness = g.betweenness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_betweenness = ig.rescale(edge_betweenness)
                node_size.append(np.array(edge_betweenness)+0.07)
            # node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            node_size = []
            for g in layers:
                edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
                edge_strength = ig.rescale(edge_strength)
                node_size.append(np.array(edge_strength)+0.07)
            # node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            node_size = []
            for g in layers:
                edge_closeness = g.closeness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_closeness = ig.rescale(edge_closeness)
                node_size.append(np.array(edge_closeness)+0.07)
            # node_color = [cmap1(b) for b in edge_closeness]
        elif kwargs["node_metric"] == "hub score":
            node_size = []
            for g in layers:
                edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
                edge_hub = ig.rescale(edge_hub)
                node_size.append(np.array(edge_hub)+0.07)
            # node_color = [cmap1(b) for b in edge_hub]
        elif kwargs["node_metric"] == "authority score":
            node_size = []
            for g in layers:
                edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
                edge_authority = ig.rescale(edge_authority)
                node_size.append(np.array(edge_authority)+0.07)
            # node_color = [cmap1(b) for b in edge_authority]
        elif kwargs["node_metric"] == "eigenvector centrality":
            node_size = []
            for g in layers:
                edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
                edge_evc = ig.rescale(edge_evc)
                node_size.append(np.array(edge_evc)+0.07)
            # node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            node_size = []
            for g in layers:
                edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
                edge_pagerank = ig.rescale(edge_pagerank)
                node_size.append(np.array(edge_pagerank)+0.07)
            # node_color = [cmap1(b) for b in edge_pagerank]
        elif kwargs["node_metric"] == "k-core":
            node_size = []
            for g in layers:
                k_degree = kwargs["deg"]
                size = k_core_weights(g, k_degree, 0.01)
                node_size.append(np.array([n*1 for n in size]))
    else:
        node_color = "red"
        node_size = 15
        
    if "idx" in kwargs:
        if len(kwargs["idx"]) == 0:
            marker_frame_color = None
        else:
            cmap = get_cmap('Spectral')
            palette = ClusterColoringPalette(kwargs["cluster_num"])
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
            
    LayeredNetworkGraph(layers_layout, layers, ax=ax, layout=layout, node_labels = node_labels, nodes_width=node_size, node_edge_colors=marker_frame_color, layer_labels=layer_labels)
    ax.set_axis_off()

    
def display_stats(path_to_file, ax, percentage_threshold = 0.0, mnn = None, **kwargs):
    """
    This function displays a histogram representation of the metrics of the graph to analyze.

    Parameters
    ----------
    path_to_file : string
        path specifying where the file containing the matrix representing the graph 
        is stored. Should be a .csv file.
    ax : matplotlib.axis
        the axis to plot the graph onto.
    **kwargs : strings
        node_metric : specifies which metric to use in order to color and size the vertices of the graph.
            allowed values: ["strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]

    Returns
    -------
    None.

    """    
            
    if type(path_to_file) == list:
        warn("More than one input graph path has been provided. \n Multilayer plotting is not yet supported. Only first one will be displayed.")
        data = read_graph(path_to_file[0], percentage_threshold = percentage_threshold, mnn = mnn)[0]
    else:
        data = read_graph(path_to_file, percentage_threshold = percentage_threshold, mnn = mnn)[0]

    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            ax.text(0.4, 0.5, 'Please select a metric', transform=ax.transAxes, fontsize=20)
            ax.axis("off")
        elif kwargs["node_metric"] == "betweenness":
            edge_betweenness = g.betweenness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_betweenness = ig.rescale(edge_betweenness)
            ax.hist(edge_betweenness)
        elif kwargs["node_metric"] == "strength":
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            edge_strength = ig.rescale(edge_strength)
            ax.hist(edge_strength)
        elif kwargs["node_metric"] == "closeness":
            edge_closeness = g.closeness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_closeness = ig.rescale(edge_closeness)
            ax.hist(edge_closeness)
        elif kwargs["node_metric"] == "hub score":
            edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
            edge_hub = ig.rescale(edge_hub)
            ax.hist(edge_hub)
        elif kwargs["node_metric"] == "authority score":
            edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
            edge_authority = ig.rescale(edge_authority)
            ax.hist(edge_authority)
        elif kwargs["node_metric"] == "eigenvector centrality":
            edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
            edge_evc = ig.rescale(edge_evc)
            ax.hist(edge_evc)
        elif kwargs["node_metric"] == "page rank":
            edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
            edge_pagerank = ig.rescale(edge_pagerank)
            ax.hist(edge_pagerank)
        elif kwargs["node_metric"] == "k-core":
            k_degree = kwargs["deg"]
            for i in range(len(path_to_file)):
                g = read_graph(path_to_file[i], percentage_threshold, mnn, return_ig=True)[0]
                core_size = k_core_size(g, k_degree)
                p_value = k_core_p_value(path_to_file[i], k_degree, percentage_threshold, mnn)
                ax.text(0.1, 0.1+0.1*i, os.path.basename(path_to_file[i])+": k-core size = "+str(core_size)+" p_value = "+str(p_value),\
                        transform=ax.transAxes, fontsize=15)
            ax.axis("off")

        
if __name__ == '__main__':

    path = "..\\data\\G1\\"
    file = "interactions_resD7_1.csv"
    # c = community_clustering(path+file)
    print(read_labels(path+file))
    f = plt.Figure()
    a = f.add_subplot(111)#, projection='3d')
    display_graph(path+file, a)
    # c = display_stats([path+file], ax = a, mnn = 3, node_metric = "k-core", deg = 2)
    # g = read_graph(path+file, return_ig=True)

    # display_graph([path+"\\interactions_resD7_1.csv", path+"\\interactions_resD7_1.csv"], a, node_metric = "closeness", cluster_num = 2, idx = [1, 1, 1, 0,0,1,1,1,1,1])
    
# clusterer = graphClusterer(D, True, "fully connected")
# cluster_num = 2
# clusterer.k_elbow_curve(a)
# nn = 4
# _, idx, _, _ = clusterer.clustering(cluster_num, isAffinity = True)
# clusterer.sigma_grid_search(a, 30, 2)
