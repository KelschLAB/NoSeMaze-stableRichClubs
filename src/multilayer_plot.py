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
from mpl_toolkits.mplot3d.art3d import Text3D

from read_graph import *


class LayeredNetworkGraph(object):

    def __init__(self, graphs_layout, graphs, node_labels=None, layout=nx.spring_layout, nodes_width = None, ax=None, node_edge_colors = None, layer_labels = None):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """
        
        self.cmap_edges = cm.viridis
        self.graphs = [g.to_networkx() for g in graphs_layout]
        self.edge_width = []
        self.node_edge_colors = node_edge_colors
        self.layer_labels = layer_labels
        scale_factor = 5
        for g in graphs:
            self.edge_width.extend(self.rescale(np.array([w['weight'] for w in g.es]), scale_factor))
        self.edge_colors = []
        for w in self.edge_width:
            self.edge_colors.append(self.cmap_edges((w - 0.001)/scale_factor)) #subtracting 0.001 to ensure colormap is not called with 1 (which would be cycle back to 0).
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
        
    def rescale(self, arr, max_val = 5):
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

        composition = self.graphs[0]
        for h in self.graphs[1:]:
            composition = nx.compose(composition, h)

        try: #not all layouts are random, and therefore some dont accept 'seed' argument
            pos = self.layout(composition, seed = 1, *args, **kwargs)
        except:
            pos = self.layout(composition, *args, **kwargs)

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            self.node_positions.update({(node, z) : (*pos[node], z) for node in g.nodes()})

    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z, *args, **kwargs)

    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)

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
            if node in node_labels:
                ax.text(*self.node_positions[(node, z)], node_labels[node], *args, **kwargs)

    def draw(self):
        self.draw_edges(self.edges_within_layers,  facecolor=self.edge_colors,  colors=self.edge_colors, alpha=1, linestyle='-', zorder=2, linewidths=self.edge_width)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.2, linestyle='--', zorder=2, lw = 1)
        cmap_list = [cm.Reds, cm.Blues, cm.Greens, cm.Oranges, cm.Purples]*self.total_layers
        for z in range(self.total_layers):
            if self.layer_labels != None:
                self.draw_plane(z, layer_label = self.layer_labels[z], alpha=0.05, zorder=1)
            else:
                self.draw_plane(z, alpha=0.05, zorder=1)
            if self.nodes_width is not None and type(self.nodes_width) == list:
                colors = [cmap_list[z](width) for width in self.nodes_width[z]]
                self.draw_nodes([node for node in self.nodes if node[1]==z], \
                                s=self.nodes_width[z]*500, zorder=3, \
                                edgecolors = self.node_edge_colors, linewidths=2, c = colors, depthshade=False)
            else:
                self.draw_nodes([node for node in self.nodes if node[1]==z], s=300, zorder=3, depthshade=False)

        if self.node_labels:
            self.draw_node_labels(self.node_labels,
                                  horizontalalignment='center',
                                  verticalalignment='center',
                                  zorder=100)

if __name__ == '__main__':

    path = "C:\\Users\\Corentin offline\\Documents\\Python Scripts\\micemaze\\data\\G2\\"
    files = [path+"approach_prop_resD7_1.csv", path+"HWI_t_resD7_1.csv"]
    
    layers = read_graph(files, 0, None, True)
    # node_labels = {nn : str(nn) for nn in range(4*n)}

    # initialise figure and plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2,2,1), zoom=1.4)
    LayeredNetworkGraph(layers, ax=ax, layout=nx.circular_layout)
    ax.set_axis_off()
    plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.viridis), ax=ax, label="Edge weight")
    plt.show()