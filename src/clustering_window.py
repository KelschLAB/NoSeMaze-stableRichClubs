import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
import igraph as ig
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
import os
from read_graph import *
from clustering_algorithm import * 
from multilayer_plot import *

def rescale(arr, max_val = 5):
    normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    return normalized_arr*max_val

class SpectralClustWindow(tk.Toplevel):
    def __init__(self, root = None, app = None, path_to_file = None):

        super().__init__(master = root)
        self.root = root
        self.geometry("1000x600")
        self.title('Clustering analysis window')
        self.app = app
        
        #variables stored after interacting with the buttons
        # self.dirpath = dirpath
        self.path_to_file = path_to_file
        self.clusterer = None
        self.idx = None
        self.nn = 7
        self.cluster_num = 2
        
        ## frames ##
        # navigation frame
        menu_frame = tk.Frame(self, bg = "gray", height= 100)
        menu_frame.pack(side="top", fill="x")
        # hyperparams
        self.selection_btn_frame = tk.Frame(menu_frame, bg = "gray", height=50)
        self.selection_btn_frame.pack(side="top", fill="x")
        self.hyperparam_btn_frame = tk.Frame(menu_frame, bg = "blue", height=50)
        self.hyperparam_btn_frame.pack(side="top", fill="x")
        # content frame
        self.content_frame = tk.Frame(self, bg = "white") #green
        self.content_frame.pack(side="top")
        # clustering  buttons frames
        self.clustering_btn_frame = tk.Frame(self, bg = "gray", height=50)
        self.clustering_btn_frame.pack(side="bottom", fill="both")
        
        # selecting the graph combination for clustering
        tk.Label(self.selection_btn_frame, text="Laplacian combination").pack(side = "left")
        self.laplacian_selector=ttk.Combobox(self.selection_btn_frame, state = "readonly")
        self.laplacian_selector.pack(side="left", fill="x")
        self.laplacian_selector.set("fully connected")
        self.laplacian_selector["values"] = ["fully connected", "nearest neighbours", "epsilon neighbourhood"]
        self.laplacian_selector.bind('<<ComboboxSelected>>',  self.laplacian_changed)

        self.hyperparams_buttons_fc() # setting up hyperparams button on start
        self.clustering_buttons_fc() # setting up clustering buttons on start
        
        D = read_graph(self.path_to_file)
        self.clusterer = graphClusterer(D, self.app.edge_type == "affinity", self.laplacian_selector.get())
         
    ## dynamic buttons displaying ##    
    def laplacian_changed(self, event): # if laplacian changed due to different graph connectivity, adapt buttons
        self.clusterer.connectivity_type = self.laplacian_selector.get()
        if self.laplacian_selector.get() == "fully connected":
            self.hyperparams_buttons_fc()
            self.clustering_buttons_fc()
        elif self.laplacian_selector.get() == "nearest neighbours":
            self.hyperparams_buttons_nn()
            self.clustering_buttons_nn()
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            self.hyperparams_buttons_epsilon()
            self.clustering_buttons_epsilon()
    
    def clustering_buttons_fc(self):
        for fm in self.clustering_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
            # selecting the number of clusters
        tk.Label(self.selection_btn_frame, text="Number of clusters").pack(side = "left")
        self.cluster_number_field = tk.Entry(self.selection_btn_frame, width = 5)
        self.cluster_number_field["justify"] = "center"
        self.cluster_number_field.pack(side = "left")
        self.cluster_number_field.insert(-1, 2)
        if not(self.app.edge_type == "affinity"):
            # selecting the number of nn
            tk.Label(self.clustering_btn_frame, text="Sigma").pack(side = "left", padx = 5)
            self.nn_field = tk.Entry(self.clustering_btn_frame, width = 5)
            self.nn_field["justify"] = "center"
            self.nn_field.pack(side = "left", padx = 0)
            self.nn_field.insert(-1, 7)
         
        connectivity_button=tk.Button(self.selection_btn_frame)
        connectivity_button["justify"] = "center"
        connectivity_button["text"] = "Show connectivity"
        connectivity_button.pack(side="left", fill="x", padx = 5)
        connectivity_button["command"] = self.plot_graph_connectivity    
            
        cluster_button=tk.Button(self.clustering_btn_frame)
        cluster_button["justify"] = "center"
        cluster_button["text"] = "Cluster!"
        cluster_button.pack(side="top", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_graphs
            
    def clustering_buttons_nn(self):
        for fm in self.clustering_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
            # selecting the number of clusters
        tk.Label(self.clustering_btn_frame, text="Number of clusters").pack(side = "left")
        self.cluster_number_field = tk.Entry(self.clustering_btn_frame, width = 5)
        self.cluster_number_field["justify"] = "center"
        self.cluster_number_field.pack(side = "left")
        self.cluster_number_field.insert(-1, 2)
        if not(self.app.edge_type == "affinity"):
            # selecting the number of nn
            tk.Label(self.clustering_btn_frame, text="Sigma").pack(side = "left", padx = 5)
            self.nn_field = tk.Entry(self.clustering_btn_frame, width = 5)
            self.nn_field["justify"] = "center"
            self.nn_field.pack(side = "left", padx = 0)
            self.nn_field.insert(-1, 7)
        # selecting the number of nn
        tk.Label(self.clustering_btn_frame, text="Nearest neighbours").pack(side = "left", padx = 5)
        self.mnn_field = tk.Entry(self.clustering_btn_frame, width = 5)
        self.mnn_field["justify"] = "center"
        self.mnn_field.pack(side = "left", padx = 0)
        self.mnn_field.insert(-1, 7)
        
        connectivity_button=tk.Button(self.clustering_btn_frame)
        connectivity_button["justify"] = "center"
        connectivity_button["text"] = "Show connectivity"
        connectivity_button.pack(side="left", fill="x", padx = 5)
        connectivity_button["command"] = self.plot_graph_connectivity
        
        cluster_button=tk.Button(self.clustering_btn_frame)
        cluster_button["justify"] = "center"
        cluster_button["text"] = "Cluster!"
        cluster_button.pack(side="left", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_graphs
            
    def clustering_buttons_epsilon(self):
        for fm in self.clustering_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()   
            # selecting the number of clusters
        tk.Label(self.clustering_btn_frame, text="Number of clusters").pack(side = "left")
        self.cluster_number_field = tk.Entry(self.clustering_btn_frame, width = 5)
        self.cluster_number_field["justify"] = "center"
        self.cluster_number_field.pack(side = "left")
        self.cluster_number_field.insert(-1, 2)
        if not(self.app.edge_type == "affinity"):
            # selecting the number of nn
            tk.Label(self.clustering_btn_frame, text="Sigma").pack(side = "left", padx = 5)
            self.nn_field = tk.Entry(self.clustering_btn_frame, width = 5)
            self.nn_field["justify"] = "center"
            self.nn_field.pack(side = "left", padx = 0)
            self.nn_field.insert(-1, 7)
        tk.Label(self.clustering_btn_frame, text="Epsilon").pack(side = "left", padx = 5)
        self.epsilon_field = tk.Entry(self.clustering_btn_frame, width = 5)
        self.epsilon_field["justify"] = "center"
        self.epsilon_field.pack(side = "left", padx = 0)
        self.epsilon_field.insert(-1, 7)
        
        connectivity_button=tk.Button(self.clustering_btn_frame)
        connectivity_button["justify"] = "center"
        connectivity_button["text"] = "Show connectivity"
        connectivity_button.pack(side="left", fill="x", padx = 5)
        connectivity_button["command"] = self.plot_graph_connectivity
    
        cluster_button=tk.Button(self.clustering_btn_frame)
        cluster_button["justify"] = "center"
        cluster_button["text"] = "Cluster!"
        cluster_button.pack(side="left", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_graphs

    def hyperparams_buttons_fc(self): #display following buttons when 'fc' option is selected
        for fm in self.hyperparam_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
        # Cluster num button
        tk.Label(self.hyperparam_btn_frame, text="Hyperparameter exploration").pack(side = "left")
        cluster_num_button= tk.Button(self.hyperparam_btn_frame)
        cluster_num_button["justify"] = "center"
        cluster_num_button["text"] = "Cluster number"
        cluster_num_button.pack(side="left", fill="x", padx = 5)
        cluster_num_button["command"] = self.plot_clusNum_stats
        # nn button
        if not(self.app.edge_type == "affinity"): # if input graph is distance, needs to be converted to affinity with nn parameter
            nn_button= tk.Button(self.hyperparam_btn_frame)
            nn_button["justify"] = "center"
            nn_button["text"] = "Sigma"
            nn_button.pack(side="left", fill="x", padx = 5)
            nn_button["command"] = self.plot_nn_curve
            
    def hyperparams_buttons_nn(self): #display following buttons when 'nn' option is selected
        for fm in self.hyperparam_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
        tk.Label(self.hyperparam_btn_frame, text="Hyperparameter exploration").pack(side = "left")
        cluster_num_button= tk.Button(self.hyperparam_btn_frame)
        cluster_num_button["justify"] = "center"
        cluster_num_button["text"] = "Cluster number"
        cluster_num_button.pack(side="left", fill="x", padx = 5)
        cluster_num_button["command"] = self.plot_clusNum_stats
        if not(self.app.edge_type == "affinity"): # if input graph is distance, needs to be converted to affinity with nn parameter
            nn_button= tk.Button(self.hyperparam_btn_frame)
            nn_button["justify"] = "center"
            nn_button["text"] = "Sigma"
            nn_button.pack(side="left", fill="x", padx = 5)
            nn_button["command"] = self.plot_nn_curve
        mnn_button= tk.Button(self.hyperparam_btn_frame)
        mnn_button["justify"] = "center"
        mnn_button["text"] = "Nearest neighbours"
        mnn_button.pack(side="left", fill="x", padx = 5)
        mnn_button["command"] = self.plot_mnn_curve
            
    def hyperparams_buttons_epsilon(self): #display following buttons when 'epsilon' option is selected
        for fm in self.hyperparam_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
        tk.Label(self.hyperparam_btn_frame, text="Hyperparameter exploration").pack(side = "left")
        cluster_num_button= tk.Button(self.hyperparam_btn_frame)
        cluster_num_button["justify"] = "center"
        cluster_num_button["text"] = "Cluster number"
        cluster_num_button.pack(side="left", fill="x", padx = 5)
        cluster_num_button["command"] = self.plot_clusNum_stats
        if not(self.app.edge_type == "affinity"): # if input graph is distance, needs to be converted to affinity with nn parameter
            self.nn_button= tk.Button(self.hyperparam_btn_frame)
            nn_button["justify"] = "center"
            nn_button["text"] = "Sigma"
            nn_button.pack(side="left", fill="x", padx = 5)
            nn_button["command"] = self.plot_nn_curve
        epsilon_button= tk.Button(self.hyperparam_btn_frame)
        epsilon_button["justify"] = "center"
        epsilon_button["text"] = "epsilon"
        epsilon_button.pack(side="left", fill="x", padx = 5)
        epsilon_button["command"] = self.plot_epsilon_curve
    
    def cluster_graphs(self):
        cluster_num = int(self.cluster_number_field.get())
        nn = int(self.nn_field.get())
        if self.laplacian_selector.get() == "fully connected":
            connectivity_param = None
        elif self.laplacian_selector.get() == "nearest neighbours":
            connectivity_param = int(self.mnn_field.get())
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            connectivity_param = int(self.epsilon_field.get())
        _, self.app.idx, _, _ = self.clusterer.clustering(cluster_num, self.app.edge_type == "affinity", nn, connectivity_param)
        self.app.cluster_num = cluster_num
        self.app.plot_in_frame(layout_style = self.app.layout_style, node_metric = self.app.node_metric, \
                               percentage_threshold=self.app.percentage_threshold, mnn = self.app.mnn_number, deg = self.app.degree)
        
    def plot_clusNum_stats(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        # progressbar = ttk.Progressbar(self.content_frame, length= 100)
        # progressbar.pack(side="left")
        self.clusterer.k_elbow_curve(a)#, progress_bar=progressbar)
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        
    def plot_nn_curve(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        self.clusterer.nn_grid_search(a, 30, int(self.cluster_number_field.get()))
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        
    def plot_mnn_curve(self):   
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        if not(self.app.edge_type == "affinity"):
            self.clusterer.mnn_grid_search(a, 10, int(self.cluster_number_field.get()), int(self.nn_field.get()))
        else:
            self.clusterer.mnn_grid_search(a, 10, int(self.cluster_number_field.get()), None)
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
    
    def plot_epsilon_curve(self):   
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        if not(self.app.edge_type == "affinity"):
            self.clusterer.epsilon_grid_search(a, 30, int(self.cluster_number_field.get()), int(self.nn_field.get()))
        else:
            self.clusterer.epsilon_grid_search(a, 30, int(self.cluster_number_field.get()), None)
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        
    def plot_graph_connectivity(self):   
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(750*px,450*px))
        
        cluster_num = int(self.cluster_number_field.get())
        nn = int(self.mnn_field.get())

        if self.laplacian_selector.get() == "fully connected":
            connectivity_param = None
        elif self.laplacian_selector.get() == "nearest neighbours":
            connectivity_param = int(self.mnn_field.get())
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            connectivity_param = int(self.epsilon_field.get())
        S, _, _, _ = self.clusterer.clustering(cluster_num, self.app.edge_type == "affinity", connectivity_param, self.laplacian_selector.get())
        S_layout, _, _, _ = self.clusterer.clustering(cluster_num, self.app.edge_type == "affinity", None, None)

        if len(S) > 1: #3d plot
            a = f.add_subplot(111, projection='3d')
            a.set_box_aspect((2,2,1), zoom=1.5)
            layers_layout = [ig.Graph.Weighted_Adjacency(data, mode='directed') for data in S_layout]
            layers = [ig.Graph.Weighted_Adjacency(data, mode='directed') for data in S]
            graphs_data = [data for data in S]
            LayeredNetworkGraph(layers_layout, layers, graphs_data, ax=a, layout = nx.spring_layout)
            # LayeredNetworkGraph(layers_layout, layers, read_graph(files), ax=ax, layout=nx.circular_layout)
            a.set_axis_off()
        else: #2d plot
            a = f.add_subplot(111)
            if isSymmetric(S[0]):
                g = ig.Graph.Weighted_Adjacency(S[0], mode='undirected')
            else:
                g = ig.Graph.Weighted_Adjacency(S[0], mode='directed')
        
            # layout = g.layout(layout_style)
            visual_style = {}
            visual_style["edge_width"] = rescale(np.array([w['weight'] for w in g.es]))

            ig.plot(g, target=a, **visual_style)
        
        
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top")
         
    # def graph_selected(self, event):
    #     self.path_to_file = self.dirpath + "/" +  self.graph_selector.get()
    #     self.clusterer = graphClusterer([read_graph(self.path_to_file)])
        # self.plot_hyperparams()
    
    # def plot_hyperparams(self):
    #     for fm in self.content_frame.winfo_children():
    #         fm.destroy()
    #         self.master.update()
    #     px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    #     f = Figure(figsize=(800*px,300*px), dpi = 100)
    #     a = f.add_subplot(111)
    #     # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
    #     display_graph(self.path_to_file, a)
    #     canvas = FigureCanvasTkAgg(f, master=self.content_frame)
    #     NavigationToolbar2Tk(canvas, self.content_frame)
    #     canvas.draw()
    #     canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
     
        