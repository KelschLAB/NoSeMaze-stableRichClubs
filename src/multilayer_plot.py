"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib import cm
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Text3D
from read_graph import *


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


class LayeredNetworkGraph(object):

    def __init__(self, graphs_layout, graphs, graphs_data, node_labels=None, layout=nx.spring_layout, nodes_width = None,
                 default_edge_width = 5, ax=None, node_edge_colors = None, layer_labels = None, scale_edge_width = True, between_layer_edges = True):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs_layout :  List of graphs (ig, converted to nx), without any graph cut params. 
            This is to ensure the consistency of the layout under changes of graph cut.
        graphs : list of ig graphs, that are converted to networkx.Graph objects List of graphs, 
            one for each layer.
        graphs_data : list of numpy arrays containing the exact weights of the graphs.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.
        """
        
        self.cmap_edges = cm.Greys
        self.graphs_layout = [g.to_networkx() for g in graphs_layout] # for layout, should be read without graph-cut (mnn or threshold) in order to stay constant.
        self.graphs = [g.to_networkx() for g in graphs]
        self.data = [data for data in graphs_data]
        self.edge_width = []
        self.alphas = []
        self.node_edge_colors = node_edge_colors
        self.layer_labels = layer_labels
        self.between_layer_edges = between_layer_edges
        self.symmetry = [] # to store whether or not graphs are directed
        for g in self.graphs:
            weights = nx.get_edge_attributes(g, "weight").values()
            if scale_edge_width:
                g_edge_width = self.rescale(np.array([w for w in weights]), default_edge_width)
            else:
                g_edge_width = np.array([1 if w > 0.01 else 0 for w in weights])*default_edge_width
            self.edge_width.extend(g_edge_width)
            self.alphas.extend(self.rescale(np.array([w for w in weights]), 0.5)+0.5)
        for graph_index, d in enumerate(self.data):
            self.symmetry.extend([self.isSymmetric(d) for i in range(len(self.graphs[graph_index].edges))])
                
        self.edge_colors = []
        for w in self.edge_width:
            self.edge_colors.append(self.cmap_edges((w - 0.001)/default_edge_width)) #subtracting 0.001 to ensure colormap is not called with 1 (which would be cycle back to 0).
        
        self.nodes_width = nodes_width
        self.total_layers = len(graphs)

        self.node_labels = node_labels
        self.layout = layout

        if ax:
            self.ax = ax
        else:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, projection='3d')

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()
        
    def isSymmetric(self, mat):
        transmat = np.array(mat).transpose()
        if np.array_equal(mat, transmat):
            return True
        return False
        
    def rescale(self, arr, max_val = 5):
        if len(np.unique(arr)) == 1:
            normalized_arr = (arr/arr[0])*max_val
        else:
            normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
        return normalized_arr*max_val

    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])

    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])

    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])

    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefore, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.

        composition = self.graphs_layout[0]
        for h in self.graphs_layout[1:]:
            composition = nx.compose(composition, h)

        try: #not all layouts are random, and therefore some dont accept 'seed' argument
            pos = self.layout(composition, seed = 1, *args, **kwargs)
        except:
            pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs_layout):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, arrow, *args, **kwargs):
        # segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        # line_collection = Line3DCollection(segments, *args, **kwargs)
        # self.ax.add_collection3d(line_collection)
        counter = 0
        for source, target in edges:
            if self.symmetry[counter] or not arrow:
                style = "-"
            else:
                style = '-|>'
            # facecolor=self.edge_colors,  colors=self.edge_colors, alpha=1, linestyle='-', zorder=2, linewidths=self.edge_width
            if "facecolor" in kwargs:
                fc = kwargs["facecolor"][counter]
            else:
                fc = None
            if "colors" in kwargs:
                c = kwargs["colors"][counter]
            else:
                c = 'k'
            if "linestyle" in kwargs:
                ls = kwargs["linestyle"]
            if "alpha" in kwargs:
                a = kwargs["alpha"]
            if "linewidths" in kwargs:
                lw = kwargs["linewidths"][counter]
            else:
                lw = 1
            if lw != 0:
                x, y, z = self.node_positions[source]
                u, v, w = self.node_positions[target]
                # self.ax.plot([x, u], [y, v], [z, w])
                arrow_prop_dict = dict(mutation_scale=20, arrowstyle=style, shrinkA=10, shrinkB=5, alpha = self.alphas[counter], linestyle = ls, color = c, lw = lw)
                a = Arrow3D([x, u], [y, v], [z, w], **arrow_prop_dict)
                self.ax.add_artist(a)
            counter += 1

    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)

    def draw_plane(self, z, *args, layer_label = None, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)
        if layer_label != None:
            self.ax.text(-1, -1, z, os.path.basename(layer_label))

    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if z == 0:
                self.ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)

    def draw(self):
        self.draw_edges(self.edges_within_layers, arrow = True, alpha=0.7, linestyle='-', zorder=2
                        , linewidths=self.edge_width, facecolor=self.edge_colors,  colors=self.edge_colors)
        if self.between_layer_edges:
            self.draw_edges(self.edges_between_layers, arrow = False, color='k', alpha=0.2, linestyle='--', zorder=2, lw = 1)
        cmap_list = [cm.Reds, cm.Blues, cm.Greens, cm.Oranges, cm.Purples]*self.total_layers
        for z in range(self.total_layers):
            if self.layer_labels != None:
                self.draw_plane(z, layer_label = self.layer_labels[z], alpha=0.05, zorder=1)
            else:
                self.draw_plane(z, alpha=0.05, zorder=1)
            if self.nodes_width is not None and type(self.nodes_width) == list:
                colors = [cmap_list[z](width) for width in self.rescale(self.nodes_width[z], 1)]
                self.draw_nodes([node for node in self.nodes if node[1]==z], \
                                s=self.nodes_width[z]*50, zorder=3, \
                                edgecolors = self.node_edge_colors, linewidths=2, c = colors, depthshade=False)
            else:
                self.draw_nodes([node for node in self.nodes if node[1]==z], edgecolors = self.node_edge_colors, linewidths=2.5, s=300, zorder=3, depthshade=False)

        if self.node_labels != None:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)

if __name__ == '__main__':

    path = "..\\data\\social network matrices 3days\\G1\\"
    # files = [path+"chasing_resD1_1.csv", path+"chasing_resD1_3.csv"]
    files = [path+"interactions_resD3_1.csv", path+"interactions_resD3_2.csv"]

  
    layers_layout = read_graph(files, 0, None, True)
    layers = read_graph(files, 10, None, True)

    # node_labels = {nn : str(nn) for nn in range(4*n)}

    # initialise figure and plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2,2,1), zoom=1.4)
    LayeredNetworkGraph(layers_layout, layers, read_graph(files), ax=ax, layout=nx.circular_layout)
    ax.set_axis_off()
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Greys), ax=ax, label="Edge weight")
    plt.show()