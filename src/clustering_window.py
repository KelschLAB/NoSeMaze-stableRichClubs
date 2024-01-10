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

class NewWindow(tk.Toplevel):
    def __init__(self, master = None, dirpath = None):
        super().__init__(master = master)
        self.title("New Window")
        self.geometry("1000x600")
        label = tk.Label(self, text ="This is a new Window")
        label.pack()
        self.title('Clustering analysis window')
        
        self.master = master
        
        #variables stored after interacting with the buttons
        self.dirpath = dirpath
        
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
        self.graph_selector=ttk.Combobox(selection_btn_frame , state = "readonly")
        self.graph_selector.pack(side="left", fill="x", padx = 5)
        self.graph_selector.set("Graph file")
        self.graph_selector.bind('<<ComboboxSelected>>', self.graph_selected)
        self.graph_selector["values"] = os.listdir(self.dirpath)
        
        # selecting the graph for clustering
        self.laplacian_selector=ttk.Combobox(selection_btn_frame , state = "readonly")
        self.laplacian_selector.pack(side="left", fill="x", padx = 5)
        self.laplacian_selector.set("Laplacian combination")
        self.laplacian_selector["values"] = ["least-square based", "eigevalue based"]
        
        # selecting the number of clusters
        tk.Label(selection_btn_frame, text="Number of clusters").pack(side = "left")
        cluster_number_field = tk.Entry(selection_btn_frame, width = 5)
        cluster_number_field["justify"] = "center"
        cluster_number_field.pack(side = "left")
        cluster_number_field.insert(-1, 2)
        
        cluster_button=tk.Button(selection_btn_frame)
        cluster_button["justify"] = "center"
        cluster_button["text"] = "Cluster!"
        cluster_button.pack(side="left", fill="x", padx = 5)
        
        ## hyperparameters frame
        tk.Label(hyperparam_btn_frame, text="Hyperparameter exploration").pack(side = "left")

        cluster_num_button= tk.Button(hyperparam_btn_frame)
        cluster_num_button["justify"] = "center"
        cluster_num_button["text"] = "Cluster number"
        cluster_num_button.pack(side="left", fill="x", padx = 5)
  
        sigma_button= tk.Button(hyperparam_btn_frame)
        sigma_button["justify"] = "center"
        sigma_button["text"] = "Sigma"
        sigma_button.pack(side="left", fill="x", padx = 5)

    def graph_selected(self, event):
        self.path_to_file = self.dirpath + "/" +  self.graph_selector.get()
        self.plot_hyperparams()
    
    def plot_hyperparams(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            self.master.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,300*px), dpi = 100)
        a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
        display_graph(self.path_to_file, a)
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
     
        