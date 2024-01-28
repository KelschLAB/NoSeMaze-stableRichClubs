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

path = "C:\\Users\\Corentin offline\\Documents\\Python Scripts\\micemaze\\data\\G2\\"
file = "approach_prop_resD7_1.csv"

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

def read_graph(path_to_file, return_ig = False, avg_graph = False):
    """
    Reads a file containing the weights defning the adjacency matrix. 

    Parameters
    ----------
    path_to_file : TYPE string or list of string containing the path to the file(s)
    return_ig : TYPE, optional
        Whether or not to return the read graphs as an ig.Graph. if false, numpy arrays are returned.
        The default is False.

    Returns
    -------
    TYPE
        list of graphs as np array or ig.Graph
    """
    if type(path_to_file) == str:
        arr = np.loadtxt(path_to_file, delimiter=",", dtype=str)
        data = arr[1:, 1:].astype(float)
        return data
    
    if type(path_to_file) == list:
        if not avg_graph:
            ## separated layers
            data = []
            for i in range(len(path_to_file)):
                arr = np.loadtxt(path_to_file[i], delimiter=",", dtype=str)
                data.append(arr[1:, 1:].astype(float)) #normalizing input data
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
        
    warn("Metric computation only supports affinity type graph at the moment. Correct that.")    
        
    if type(path_to_file) == list and len(path_to_file) > 1:
        warn("Multilayer integration for statistics still needs to be implemented.")
        display_graph_3d(path_to_file, ax = ax, layout = layout_style, \
                         node_metric = kwargs["node_metric"], idx = kwargs["idx"], cluster_num = kwargs["cluster_num"])
        return
    else:
        data = read_graph(path_to_file)[0]
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
        
    # cmap1 = LinearSegmentedColormap.from_list("vertex_cmap", ["blue", "red"])
    cmap1 = cm.Reds
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            node_color = "blue"
            node_size = 15
        elif kwargs["node_metric"] == "betweenness":
            edge_betweenness = g.betweenness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_betweenness = ig.rescale(edge_betweenness)
            node_size = [(1+e)*15 for e in edge_betweenness]
            node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
            edge_strength = ig.rescale(edge_strength)
            node_size = [(1+e)*15 for e in edge_strength]
            node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            edge_closeness = g.closeness(weights = [1/(e['weight']) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
            edge_closeness = ig.rescale(edge_closeness)
            node_size = [(1+e)*15 for e in edge_closeness]
            node_color = [cmap1(b) for b in edge_closeness]
        elif kwargs["node_metric"] == "hub score":
            edge_hub = g.hub_score(weights = [1/(e['weight']) for e in g.es()])
            edge_hub = ig.rescale(edge_hub)
            node_size = [(1+e)*15 for e in edge_hub]
            node_color = [cmap1(b) for b in edge_hub]
        elif kwargs["node_metric"] == "authority score":
            edge_authority = g.authority_score(weights = [1/(e['weight']) for e in g.es()])
            edge_authority = ig.rescale(edge_authority)
            node_size = [(1+e)*15 for e in edge_authority]
            node_color = [cmap1(b) for b in edge_authority]
        elif kwargs["node_metric"] == "eigenvector centrality":
            edge_evc = g.eigenvector_centrality(weights = [1/(e['weight']) for e in g.es()])
            edge_evc = ig.rescale(edge_evc)
            node_size = [(1+e)*15 for e in edge_evc]
            node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            edge_pagerank = g.personalized_pagerank(weights = [1/(e['weight']) for e in g.es()])
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
    # g.vs["label"] = [v.index for v in g.vs()]
    # visual_style["vertex_label_size"] = 20
    # visual_style["vertex_label_dist"] = 0.5

    # visual_style["vertex_font"] = "Times"
    ig.plot(g, target=ax, **visual_style)
    
def display_graph_3d(path_to_file, ax, **kwargs):
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
    layers = read_graph(path_to_file, True)

    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            # node_color = "blue"
            node_size = 15
    
        elif kwargs["node_metric"] == "betweenness":
            node_size = []
            for g in layers:
                edge_betweenness = g.betweenness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_betweenness = ig.rescale(edge_betweenness)
                node_size.append(np.array(edge_betweenness))
            # node_color = [cmap1(b) for b in edge_betweenness]
        elif kwargs["node_metric"] == "strength":
            node_size = []
            for g in layers:
                edge_strength = g.strength(weights = [e['weight'] for e in g.es()])
                edge_strength = ig.rescale(edge_strength)
                node_size.append(np.array(edge_strength))
            # node_color = [cmap1(b) for b in edge_strength]
        elif kwargs["node_metric"] == "closeness":
            node_size = []
            for g in layers:
                edge_closeness = g.closeness(weights = [1/(e['weight']**2) for e in g.es()]) #taking the inverse of edge values as we want high score to represent low distances
                edge_closeness = ig.rescale(edge_closeness)
                node_size.append(np.array(edge_closeness))
            # node_color = [cmap1(b) for b in edge_closeness]
        elif kwargs["node_metric"] == "hub score":
            node_size = []
            for g in layers:
                edge_hub = g.hub_score(weights = [e['weight'] for e in g.es()])
                edge_hub = ig.rescale(edge_hub)
                node_size.append(np.array(edge_hub))
            # node_color = [cmap1(b) for b in edge_hub]
        elif kwargs["node_metric"] == "authority score":
            node_size = []
            for g in layers:
                edge_authority = g.authority_score(weights = [e['weight'] for e in g.es()])
                edge_authority = ig.rescale(edge_authority)
                node_size.append(np.array(edge_authority))
            # node_color = [cmap1(b) for b in edge_authority]
        elif kwargs["node_metric"] == "eigenvector centrality":
            node_size = []
            for g in layers:
                edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
                edge_evc = ig.rescale(edge_evc)
                node_size.append(np.array(edge_evc))
            # node_color = [cmap1(b) for b in edge_evc]
        elif kwargs["node_metric"] == "page rank":
            node_size = []
            for g in layers:
                edge_pagerank = g.personalized_pagerank(weights = [e['weight'] for e in g.es()])
                edge_pagerank = ig.rescale(edge_pagerank)
                node_size.append(np.array(edge_pagerank))
            # node_color = [cmap1(b) for b in edge_pagerank]
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

    LayeredNetworkGraph(layers, ax=ax, layout=layout, nodes_width=node_size, node_edge_colors=marker_frame_color)
    ax.set_axis_off()

    
def display_stats(path_to_file, ax, **kwargs):
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
        data = read_graph(path_to_file[0])
    else:
        data = read_graph(path_to_file)

    if isSymmetric(data):
        g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
    else:
        g = ig.Graph.Weighted_Adjacency(data, mode='directed')
        
    if "node_metric" in kwargs:
        if kwargs["node_metric"] == "none":
            ax.text(0.5, 0.5, 'Please select a metric', transform=ax.transAxes)
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

        
if __name__ == '__main__':
    c = community_clustering(path+file)

    # f = plt.Figure()
    # a = f.add_subplot(111, projection='3d')
    # c = display_graph(path+file, a, node_metric = "betweenness")
# D = [read_graph(path+file)]

    # display_graph([path+"\\interactions_resD7_1.csv", path+"\\interactions_resD7_1.csv"], a, node_metric = "closeness", cluster_num = 2, idx = [1, 1, 1, 0,0,1,1,1,1,1])
    
# clusterer = graphClusterer(D, True, "fully connected")
# cluster_num = 2
# clusterer.k_elbow_curve(a)
# nn = 4
# _, idx, _, _ = clusterer.clustering(cluster_num, isAffinity = True)
# clusterer.sigma_grid_search(a, 30, 2)
