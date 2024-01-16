import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog
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

class NewWindow(tk.Toplevel):
    def __init__(self, root = None, app = None, dirpath = None):

        super().__init__(master = root)
        self.title("New Window")
        self.geometry("1000x600")
        label = tk.Label(self, text ="This is a new Window")
        label.pack()
        self.title('Clustering analysis window')
        self.app = app
        
        #variables stored after interacting with the buttons
        self.dirpath = dirpath
        self.clusterer = None
        self.idx = None
        self.nn = 7
        self.cluster_num = 2
        
        # navigation frame
        menu_frame = tk.Frame(self, bg = "gray", height= 100)
        menu_frame.pack(side="top", fill="x")
        selection_btn_frame = tk.Frame(menu_frame, bg = "gray", height=50)
        selection_btn_frame.pack(side="top", fill="x")
        hyperparam_btn_frame = tk.Frame(menu_frame, bg = "blue", height=50)
        hyperparam_btn_frame.pack(side="top", fill="x")

        # content frame
        self.content_frame = tk.Frame(self, bg = "green")
        self.content_frame.pack(side="top")
        clustering_btn_frame = tk.Frame(self, bg = "red", height=50)
        clustering_btn_frame.pack(side="bottom", fill="both")
        
        ## clustering buttons frame
        # selecting the graph for clustering
        self.graph_selector=tk.Menubutton(selection_btn_frame, text = "Select graph file(s)")
        # filling the values of the graph selection menu for clustering
        menu = tk.Menu(self.graph_selector, tearoff=False)
        self.active_path_list = [] # storing selected paths here
        self.path_variable_list = []
        self.path_label_list = []
        path_list = os.listdir(self.dirpath)
        for i in range(0, len(path_list)):
            globals()['var'+str(i)] = tk.StringVar() # Creating variables to store paths dynamically
        for i in range(0, len(path_list)): # Adding values to the actual Menubutton
            self.path_variable_list.append(globals()['var'+str(i)])
            self.path_label_list.append(path_list[i])
            menu.add_checkbutton(label = self.path_label_list[i], variable = self.path_variable_list[i], command=self.get_checked)
            
        self.graph_selector.configure(menu=menu)
        self.graph_selector.pack(side="left", fill="x", padx = 5)
  
        # self.graph_selector.set("Graph file")
        # self.graph_selector.bind('<<ComboboxSelected>>', self.get_checked)
        
        # selecting the graph combination for clustering
        tk.Label(clustering_btn_frame, text="Laplacian combination").pack(side = "left")
        self.laplacian_selector=ttk.Combobox(clustering_btn_frame, state = "readonly")
        self.laplacian_selector.pack(side="left", fill="x")
        self.laplacian_selector.set("least-square based")
        self.laplacian_selector["values"] = ["least-square based", "eigevalue based"]
        
        # selecting the number of clusters
        tk.Label(clustering_btn_frame, text="Number of clusters").pack(side = "left")
        self.cluster_number_field = tk.Entry(clustering_btn_frame, width = 5)
        self.cluster_number_field["justify"] = "center"
        self.cluster_number_field.pack(side = "left")
        self.cluster_number_field.insert(-1, 2)
        
        # selecting the number of nn
        tk.Label(clustering_btn_frame, text="Nearest neighbours").pack(side = "left", padx = 5)
        self.nn_field = tk.Entry(clustering_btn_frame, width = 5)
        self.nn_field["justify"] = "center"
        self.nn_field.pack(side = "left", padx = 0)
        self.nn_field.insert(-1, 7)
        
        cluster_button=tk.Button(clustering_btn_frame)
        cluster_button["justify"] = "center"
        cluster_button["text"] = "Cluster!"
        cluster_button.pack(side="left", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_graphs

        ## hyperparameters frame
        tk.Label(hyperparam_btn_frame, text="Hyperparameter exploration").pack(side = "left")

        cluster_num_button= tk.Button(hyperparam_btn_frame)
        cluster_num_button["justify"] = "center"
        cluster_num_button["text"] = "Cluster number"
        cluster_num_button.pack(side="left", fill="x", padx = 5)
        cluster_num_button["command"] = self.plot_clusNum_stats
  
        sigma_button= tk.Button(hyperparam_btn_frame)
        sigma_button["justify"] = "center"
        sigma_button["text"] = "Sigma"
        sigma_button.pack(side="left", fill="x", padx = 5)
        sigma_button["command"] = self.plot_sigma_curve
        
    def get_checked(self):
        lst = []
        for i, item in enumerate(self.path_variable_list):
            if item.get() == "1":
                lst.append(self.path_label_list[i])
        self.active_path_list = lst
        self.path_to_file = [self.dirpath + "/" + self.active_path_list[i] for i in range(len(self.active_path_list))]

    def cluster_graphs(self):
        D = read_graph(self.path_to_file)
        self.clusterer = graphClusterer(D)
        cluster_num = int(self.cluster_number_field.get())
        nn = int(self.nn_field.get())
        _, self.app.idx, _, _ = self.clusterer.clustering(cluster_num, isAffinity = False, nn = nn)
        self.app.cluster_num = cluster_num
        
    def plot_clusNum_stats(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
        self.clusterer.k_elbow_curve(a)
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        
    def plot_sigma_curve(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
        self.clusterer.sigma_grid_search(a)
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
     
        