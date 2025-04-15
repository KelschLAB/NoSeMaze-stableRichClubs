import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.cm import ScalarMappable, get_cmap


def rescale(arr, max_val = 5):
    normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    return normalized_arr*max_val

def eigenvec_explanation():
    # creating example graph
    A = np.array([[0, 1, 0, 0, 0, 1],
                  [1, 0, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0]])
    g = ig.Graph.Weighted_Adjacency(A, mode = "undirected")
    # computing eigevec centralites
    edge_evc = g.eigenvector_centrality(weights = [e['weight'] for e in g.es()])
    # edge_evc = ig.rescale(edge_evc)
    cmap1 = cm.GnBu
    node_size = [(1+e)*25 for e in edge_evc]
    node_color = [cmap1(b) for b in edge_evc]
    # plotting
    fig, ax = plt.subplots(figsize = (4, 4))
    node_names = np.round(edge_evc, 2)
    g.vs['label'] = node_names  
    visual_style = {}
    visual_style["vertex_size"] = node_size
    visual_style["vertex_color"] = node_color
    edge_cmap = get_cmap('Greys')
    # visual_style["edge_color"] = [edge_cmap(edge) for edge in rescale(np.array([w['weight'] for w in g.es])) - 0.01]
    # visual_style["edge_arrow_width"] = rescale(np.array([w['weight'] for w in g.es]))*5
    # visual_style["edge_width"] = rescale(np.array([w['weight'] for w in g.es]))
    # visual_style["layout"] = layout
    visual_style["vertex_frame_color"] = node_color
    ig.plot(g, target=ax, **visual_style)
    # adjusting the plot
    margin = 0.1 # How much margin to add around figure to ensure igraph does not crop it
    l, r = ax.get_xlim()
    d, t = ax.get_ylim()
    ax.set_xlim(l - margin, r + margin)
    ax.set_ylim(d - margin, t + margin)
    plt.title("Eigenvector centrality")

# https://cgi.cse.unsw.edu.au/~cs2521/21T3/ass/ass2/resources/Weighted-PageRank-Algorithm.pdf
# def unnormalized_pagerank(graph, tol=1e-6, max_iter=100):
    # transition_matrix = np.array(graph.get_adjacency(attribute="weight").data, dtype=float)
    # n = transition_matrix.shape[0]
    # # transition_matrix = adj_matrix / adj_matrix.sum(axis=0, where=adj_matrix.sum(axis=0) != 0, keepdims=True)
    # pagerank = np.ones(n) / n
    # teleport = np.ones(n) / n
    # for iteration in range(max_iter):
    #     new_pagerank = np.dot(transition_matrix, pagerank) 
    #     if np.linalg.norm(new_pagerank - pagerank, ord=1) < tol:
    #         print(f"Converged after {iteration + 1} iterations.")
    #         break
    #     pagerank = new_pagerank
    # return pagerank
    
def pagerank_explanation():
    # creating example graph
    A = np.array([[0, 3, 0, 0],
                  [0.5, 0, 1, 3],
                  [0, 0, 0, 3],
                  [0, 0, 0.01, 0]])
    
    A = np.array([[0, 0, 0, 1, 0],
                  [2, 0, 0, 1, 0],
                  [2, 0, 0, 1, 0],
                  [10, 1, 1, 0, 1],
                  [2, 0, 0, 1, 0]])
    
    g = ig.Graph.Weighted_Adjacency(A, mode = "directed")
    # computing eigevec centralites
    pr = g.pagerank(weights = [e['weight'] for e in g.es()], damping = 0.999)
    # edge_evc = ig.rescale(edge_evc)
    cmap1 = cm.PuBu
    node_size = [(1+e)*40 for e in pr]
    node_color = [cmap1(b) for b in pr]
    # plotting
    fig, ax = plt.subplots(figsize = (4, 4))
    # node_names = [str(i)+" :"+str(n) for i,n in enumerate(np.round(pr, 2))]
    node_names = np.round(pr, 2)

    g.vs['label'] = node_names  
    visual_style = {}
    visual_style["vertex_size"] = node_size
    visual_style["vertex_color"] = node_color
    edge_cmap = get_cmap('Greys')
    visual_style["edge_color"] = [edge_cmap(edge) for edge in np.array([w['weight'] for w in g.es]) + 0.25]
    visual_style["edge_arrow_width"] = 9# rescale(np.array([w['weight'] for w in g.es])) + 0.2
    visual_style["edge_width"] = 0.5#rescale(np.array([w['weight'] for w in g.es]))+0.25
    # visual_style["layout"] = layout
    visual_style["vertex_frame_color"] = node_color
    ig.plot(g, target=ax, layout = "reingold_tilford", **visual_style)
    # adjusting the plot
    margin = 0.1 # How much margin to add around figure to ensure igraph does not crop it
    l, r = ax.get_xlim()
    d, t = ax.get_ylim()
    ax.set_xlim(l - margin, r + margin)
    ax.set_ylim(d - margin, t + margin)
    plt.title("Pagerank")

    
if __name__ == "__main__":
    # eigenvec_explanation()
    pagerank_explanation()