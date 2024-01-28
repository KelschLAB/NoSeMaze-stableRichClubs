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

def rescale(arr, max_val = 5):
    normalized_arr = (arr - np.min(arr))/(np.max(arr)-np.min(arr))
    return normalized_arr*max_val

class NewWindow(tk.Toplevel):
    def __init__(self, root = None, app = None, path_to_file = None):

        super().__init__(master = root)
        self.root = root
        self.title("New Window")
        self.geometry("1000x600")
        label = tk.Label(self, text ="This is a new Window")
        label.pack()
        self.title('Clustering analysis window')
        self.app = app
        
        #variables stored after interacting with the buttons
        # self.dirpath = dirpath
        self.path_to_file = path_to_file
        self.clusterer = None
        self.isAffinity = tk.BooleanVar()
        self.isAffinity.set(True)
        self.idx = None
        self.nn = 7
        self.cluster_num = 2
        
        ## frames ##
        # navigation frame
        menu_frame = tk.Frame(self, bg = "gray", height= 100)
        menu_frame.pack(side="top", fill="x")
        # hyperparams
        selection_btn_frame = tk.Frame(menu_frame, bg = "gray", height=50)
        selection_btn_frame.pack(side="top", fill="x")
        self.hyperparam_btn_frame = tk.Frame(menu_frame, bg = "blue", height=50)
        self.hyperparam_btn_frame.pack(side="top", fill="x")
        # content frame
        self.content_frame = tk.Frame(self, bg = "green")
        self.content_frame.pack(side="top")
        # clustering  buttons frames
        self.clustering_btn_frame = tk.Frame(self, bg = "red", height=50)
        self.clustering_btn_frame.pack(side="bottom", fill="both")
        
        ## buttons ##
        # selecting the graph for clustering
        # self.graph_selector=tk.Menubutton(selection_btn_frame, text = "Select graph file(s)")
        # # filling the values of the graph selection menu for clustering
        # menu = tk.Menu(self.graph_selector, tearoff=False)
        # self.active_path_list = [] # storing selected paths here
        # self.path_variable_list = []
        # self.path_label_list = []
        # path_list = os.listdir(self.dirpath)
        # for i in range(0, len(path_list)):
        #     globals()['var'+str(i)] = tk.StringVar() # Creating variables to store paths dynamically
        # for i in range(0, len(path_list)): # Adding values to the actual Menubutton
        #     self.path_variable_list.append(globals()['var'+str(i)])
        #     self.path_label_list.append(path_list[i])
        #     menu.add_checkbutton(label = self.path_label_list[i], variable = self.path_variable_list[i], command=self.get_checked)
        # self.graph_selector.configure(menu=menu)
        # self.graph_selector.pack(side="left", fill="x", padx = 5)
                
        # ## static buttons functions ##  
        # def get_checked(self):
        #     lst = []
        #     for i, item in enumerate(self.path_variable_list):
        #         if item.get() == "1":
        #             lst.append(self.path_label_list[i])
        #     self.active_path_list = lst
        #     self.path_to_file = [self.dirpath + "/" + self.active_path_list[i] for i in range(len(self.active_path_list))]
        
        # graph type
        tk.Label(selection_btn_frame, text="Type of graph").pack(side = "left")
        self.affinity_button = tk.Radiobutton(selection_btn_frame, text="Affinity", variable=self.isAffinity, value=True)
        self.affinity_button.pack(side="left")
        self.distance_button = tk.Radiobutton(selection_btn_frame,text="Distance", variable=self.isAffinity, value = False)
        self.distance_button.pack(side="left")
        
        # selecting the graph combination for clustering
        tk.Label(selection_btn_frame, text="Laplacian combination").pack(side = "left")
        self.laplacian_selector=ttk.Combobox(selection_btn_frame, state = "readonly")
        self.laplacian_selector.pack(side="left", fill="x")
        self.laplacian_selector.set("fully connected")
        self.laplacian_selector["values"] = ["fully connected", "nearest neighbours", "epsilon neighbourhood"]
        self.laplacian_selector.bind('<<ComboboxSelected>>',  self.laplacian_changed)

        self.hyperparams_buttons_fc() # setting up hyperparams button on start
        self.clustering_buttons_fc() # setting up clustering buttons on start
        self.affinity_button["command"] = self.similarity_changed #commands need to be added after laplacian selector is created
        self.distance_button["command"] = self.similarity_changed
        
        D = read_graph(self.path_to_file)
        self.clusterer = graphClusterer(D, self.isAffinity.get(), self.laplacian_selector.get())
         
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
            
    def similarity_changed(self): # if similarity (aff. or diff.) radiobutton was changed, adapt buttons
        self.clusterer.isAffinity = self.isAffinity.get()
        if self.laplacian_selector.get() == "fully connected":
            self.hyperparams_buttons_fc()
            self.clustering_buttons_fc()
        elif self.laplacian_selector.get() == "nearest neighbours":
            self.hyperparams_buttons_nn()
            self.clustering_buttons_nn()
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            self.hyperparams_buttons_epsilon()
            self.clustering_buttons_epsilon()
            self.clusterer.connectivity_type
    
    def clustering_buttons_fc(self):
        for fm in self.clustering_btn_frame.winfo_children():
            fm.destroy()
            self.root.update()
            # selecting the number of clusters
        tk.Label(self.clustering_btn_frame, text="Number of clusters").pack(side = "left")
        self.cluster_number_field = tk.Entry(self.clustering_btn_frame, width = 5)
        self.cluster_number_field["justify"] = "center"
        self.cluster_number_field.pack(side = "left")
        self.cluster_number_field.insert(-1, 2)
        if not(self.isAffinity.get()):
            # selecting the number of nn
            tk.Label(self.clustering_btn_frame, text="Sigma").pack(side = "left", padx = 5)
            self.nn_field = tk.Entry(self.clustering_btn_frame, width = 5)
            self.nn_field["justify"] = "center"
            self.nn_field.pack(side = "left", padx = 0)
            self.nn_field.insert(-1, 7)
         
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
        if not(self.isAffinity.get()):
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
        if not(self.isAffinity.get()):
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
        if not(self.isAffinity.get()): # if input graph is distance, needs to be converted to affinity with nn parameter
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
        if not(self.isAffinity.get()): # if input graph is distance, needs to be converted to affinity with nn parameter
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
        if not(self.isAffinity.get()): # if input graph is distance, needs to be converted to affinity with nn parameter
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
        if not(self.isAffinity.get()):
            nn = int(self.nn_field.get())
        else:
            nn = None
        if self.laplacian_selector.get() == "fully connected":
            connectivity_param = None
        elif self.laplacian_selector.get() == "nearest neighbours":
            connectivity_param = int(self.mnn_field.get())
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            connectivity_param = int(self.epsilon_field.get())
        _, self.app.idx, _, _ = self.clusterer.clustering(cluster_num, self.isAffinity.get(), nn, connectivity_param)
        self.app.cluster_num = cluster_num
        
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
        if not(self.isAffinity.get()):
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
        if not(self.isAffinity.get()):
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
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        cluster_num = int(self.cluster_number_field.get())
        if not(self.isAffinity.get()):
            nn = int(self.nn_field.get())
        else:
            nn = None
        if self.laplacian_selector.get() == "fully connected":
            connectivity_param = None
        elif self.laplacian_selector.get() == "nearest neighbours":
            connectivity_param = int(self.mnn_field.get())
        elif self.laplacian_selector.get() == "epsilon neighbourhood":
            connectivity_param = int(self.epsilon_field.get())
        S, _, _, _ = self.clusterer.clustering(cluster_num, self.isAffinity.get(), nn, connectivity_param)
        
        if isSymmetric(S[0]):
            g = ig.Graph.Weighted_Adjacency(S[0], mode='undirected')
        else:
            g = ig.Graph.Weighted_Adjacency(S[0], mode='directed')
        print("Displaying first graph from stack")
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
     
        