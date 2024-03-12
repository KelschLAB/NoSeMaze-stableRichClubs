import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from pprint import pprint

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
from clustering_window import *

#To-do: - include statement that graph display is from average when several subgraphs are selected.
#       - implement average measures (average of graph != average of graph measures.)
#       - fix display bug (metrics are only colored properly for first layers, not next ones.)
#       - make code work for affinity/distance graph. right now, mnn only works for affinity, as well as all display metrics.
#           that inckude making the metric dependant on it, and the mnn as well as threshold function.
#       - fix k-core with new visualization (0.01 instead of 0)
#       - arrow heads in graph cut should also disapear
#       - mnn in clustering should be fixed (the code was not right)
#       - include a reset button?
#       - remove references to avg graph, or figure why that would be useful.
#       - directed graph in 3D for multilayers!!! 

class App:
    def __init__(self, root):
        #setting window size
        width=1000
        height=600
        
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        root.title("Multilayer graph analysis")
        
        #variables stored after interacting with the buttons
        self.dirpath = None
        self.path_to_file = None
        self.layout_style = "fr"
        self.node_metric = "none"
        self.percentage_threshold = 0.0
        self.mnn_number = None
        self.degree = 0
        self.idx = []
        self.cluster_num = 0
        self.display_type = "plot"
    
        # Frames
        menu_frame = tk.Frame(root, bg = "gray", height= 100)
        menu_frame.pack(side="top", fill="x")
        btn_frame = tk.Frame(menu_frame, bg = "gray", height=50) #main buttons frame
        btn_frame.pack(side="top", fill="x")
        result_display_frame = tk.Frame(menu_frame, bg = "blue", height=50) #result display frame
        result_display_frame.pack(side="top", fill="x")
        self.content_frame = tk.Frame(root, bg = "white")         # content frame
        self.content_frame.pack()#fill=tk.BOTH, expand=True)
        
        # Dir selection button
        path_button=tk.Button(btn_frame)
        path_button["justify"] = "center"
        path_button["text"] = "files path"
        path_button.pack(side="left", fill="x", padx = 5)
        path_button["command"] = self.path_button_command
        
        # file(s) selection Menubutton
        self.graph_selector=tk.Menubutton(btn_frame, text = "Select graph file(s)")
        self.graph_selector.pack(side="left", fill="x", padx = 5)
        self.path_variable_list = [] # storing the menu options here
        self.path_label_list = []
        self.active_path_list = [] # storing selected paths here
        # self.graph_selector.set("Graph file")
        self.graph_selector.bind('<<ComboboxSelected>>', self.get_checked)

        # layout selection
        layout_list = ["circle", "drl", "fr", "kk", "large", "random", "tree"]
        self.plot_selector=ttk.Combobox(btn_frame, values = layout_list, state = "readonly")
        self.plot_selector.pack(side="left", fill="x", padx = 5)
        self.plot_selector.set("Graph layout")
        self.plot_selector.bind('<<ComboboxSelected>>', self.plot_changed)

        # metric selection for nodes
        metric_values = ["none", "strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score", "k-core"]
        self.node_metric_selector=ttk.Combobox(btn_frame, values = metric_values, state = "readonly")
        self.node_metric_selector.pack(side="left", fill="x", padx = 5)
        self.node_metric_selector.set("Node metric")
        self.node_metric_selector.bind('<<ComboboxSelected>>', self.node_changed)

        # Graph-cut type selection
        graphcut_values = ["none", "threshold", "mutual nearest neighbors"]
        self.graphcut_selector=ttk.Combobox(btn_frame, values = graphcut_values, state = "readonly")
        self.graphcut_selector.pack(side="left", fill="x", padx = 5)
        self.graphcut_selector.set("Graph-cut type")
        self.graphcut_selector.bind('<<ComboboxSelected>>', self.graphcut_param_window)
        
        # button to open clustering window
        cluster_button=tk.Button(btn_frame)
        cluster_button["text"] = "cluster nodes"
        cluster_button.pack(side="left", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_button_command
        
        # Starting instructions label
        self.label = tk.Label(root, font = 'Helvetica 12 bold', text ="1. Select the directory where your files are stored with the  'files path'  button. \n 2. Then, select the graph file with the  'graph file'  drop-down menu to start the analysis.")
        self.label.pack(side = 'top', pady = 50)
        
        # result display widgets
        tk.Label(result_display_frame, text="Display").pack(side = "left")
        plot_btn = tk.Button(result_display_frame, text='plot')
        plot_btn.pack(side = "left", padx = 5)
        plot_btn["command"] = self.plot_clicked
        stats_btn = tk.Button(result_display_frame, text='statistics')
        stats_btn.pack(side = "left")
        stats_btn["command"] = self.stats_clicked

    def graphcut_param_window(self, event):
        if self.graphcut_selector.get() == "none":
            self.percentage_threshold = 0.0
            self.mnn_number = None
            if self.display_type == "plot":
                self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
            else:
                self.stats_in_frame()
            return
        self.new_window = tk.Toplevel(root)
        self.new_window.title("Enter Parameter Value")
        tk.Label(self.new_window, text="Enter " + self.graphcut_selector.get()).grid(row=0,column=0)
        self.graphcut_entry = tk.Entry(self.new_window)
        self.graphcut_entry.grid(row=1,column=0)
        if self.graphcut_selector.get() == "threshold":
            tk.Label(self.new_window, text="%").grid(row=1,column=1)
        elif self.graphcut_selector.get() == "Mutual nearest neighbors":
            tk.Label(self.new_window, text="Neighbours").pack()
        tk.Button(self.new_window, text="Cut!", command=self.graph_cut_changed).grid(row=2,column=0)

    def graph_cut_changed(self):
        if self.graphcut_selector.get() == "threshold":
            self.mnn_number = None
            self.percentage_threshold = float(self.graphcut_entry.get())
        elif self.graphcut_selector.get() == "mutual nearest neighbors":
            self.percentage_threshold = 0.0
            self.mnn_number = int(self.graphcut_entry.get())
        self.node_metric = self.node_metric_selector.get()
        self.new_window.destroy()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        else:
            self.stats_in_frame(node_metric = self.node_metric)
        
    def plot_clicked(self):
        self.display_type = "plot"
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        
    def stats_clicked(self):
        self.display_type = "stats"
        self.stats_in_frame()
        
    def get_checked(self):
        """ Updates list of paths when graph layer selector is clicked """
        lst = []
        for i, item in enumerate(self.path_variable_list):
            if item.get() == "1":
                lst.append(self.path_label_list[i])
        self.active_path_list = lst
        self.path_to_file = [self.dirpath + "/" + self.active_path_list[i] for i in range(len(self.active_path_list))]   
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)

    def path_button_command(self):
        """ Selects the directory path where graph layers are contained, and updates the list of selectable graph layer """
        self.dirpath = filedialog.askdirectory()
        # dirpath = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.csv"), ("All files", "*.txt*")])
        # self.graph_selector["values"] = os.listdir(self.dirpath)
        menu = tk.Menu(self.graph_selector, tearoff=False)
        self.path_variable_list = []
        self.path_label_list = []
        # main list holding menu values
        path_list = os.listdir(self.dirpath)
                
        # Creating variables to store paths dynamically
        for i in range(0, len(path_list)):
            globals()['var'+str(i)] = tk.StringVar(self.graph_selector)
        # Finally adding values to the actual Menubutton
        for i in range(0, len(path_list)):
            self.path_variable_list.append(globals()['var'+str(i)])
            self.path_label_list.append(path_list[i])
            menu.add_checkbutton(label = self.path_label_list[i], variable = self.path_variable_list[i], command=self.get_checked)
        self.graph_selector.configure(menu=menu)

    def plot_changed(self, event):
        self.layout_style = self.plot_selector.get()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)

    def k_core_window(self):
        self.new_window = tk.Toplevel(root)
        self.new_window.title("Degree value for k-core")
        tk.Label(self.new_window, text="Enter degree").grid(row=0,column=0)
        self.k_core_entry = tk.Entry(self.new_window)
        self.k_core_entry.grid(row=1,column=0)
        tk.Button(self.new_window, text="Compute k-core!", command=self.k_core_changed).grid(row=2,column=0)
        
    def k_core_changed(self):
        self.degree = int(self.k_core_entry.get())
        self.new_window.destroy()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)
        else:
            self.stats_in_frame()
        
    def node_changed(self, event):
        self.node_metric = self.node_metric_selector.get()
        if self.node_metric == "k-core":
            self.k_core_window()
            
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        else:
            self.stats_in_frame()
        
    def plot_in_frame(self, layout_style = "fr", node_metric = "none", percentage_threshold=0.0, mnn = None, deg = 0):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            root.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(950*px,500*px))
        if len(self.path_to_file) > 1:
            a = f.add_subplot(111, projection='3d')
            a.set_box_aspect((2,2,1), zoom=1.5)
        else:
            a = f.add_subplot(111)
        display_graph(self.path_to_file, a, percentage_threshold = self.percentage_threshold, mnn = self.mnn_number,\
                      layout = layout_style, node_metric = self.node_metric, idx = self.idx, \
                          cluster_num = self.cluster_num, layer_labels=self.path_to_file, deg = self.degree)
        f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Reds), ax=a, label="Relative metric value", shrink = 0.3, location = 'left')
        f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.viridis), ax=a, label="Relative edge value", shrink = 0.3, location = 'right', pad = 0.1)
        f.subplots_adjust(left=0, bottom=0, right=0.948, top=1, wspace=0, hspace=0)

        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")
        
    def stats_in_frame(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            root.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,400*px), dpi = 100)
        # f = Figure()
        a = f.add_subplot(111)
        display_stats(self.path_to_file, a, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, node_metric = self.node_metric, deg = self.degree)
    
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")

    def cluster_button_command(self):
        self.clustertype_wdw = tk.Toplevel(root)
        self.clustertype_wdw.geometry("250x250")
        self.clustertype_wdw.title("Clustering type")
        unsupervised_button = tk.Button(self.clustertype_wdw, text="Unsupervised")
        unsupervised_button.pack(side="left")
        unsupervised_button["command"] = self.unsupervised_button
        supervised_button = tk.Button(self.clustertype_wdw, text="Supervised")
        supervised_button.pack(side="left")
        supervised_button["command"] = self.supervised_button
        
    def supervised_button(self):
        NewWindow(root, self, self.path_to_file)
        self.clustertype_wdw.destroy()
        
    def unsupervised_button(self):
        self.idx = community_clustering(self.path_to_file)
        self.cluster_num = max(self.idx)+1
        self.clustertype_wdw.destroy()
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
