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
from clustering_window import *

class App:
    def __init__(self, root):
        #setting title
        root.title("undefined")
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
    
        # navigation frame
        menu_frame = tk.Frame(root, bg = "gray", height= 100)
        menu_frame.pack(side="top", fill="x")
        btn_frame = tk.Frame(menu_frame, bg = "gray", height=50)
        btn_frame.pack(pady=10, fill="x")
        # stats_btn = tk.Button(btn_frame, text='statistics')
        
        # content frame
        self.content_frame = tk.Frame(root, bg = "white")
        self.content_frame.pack()#fill=tk.BOTH, expand=True)

        path_button=tk.Button(btn_frame)
        path_button["justify"] = "center"
        path_button["text"] = "files path"
        path_button.pack(side="left", fill="x", padx = 5)
        path_button["command"] = self.path_button_command
        
        self.graph_selector=ttk.Combobox(btn_frame, state = "readonly")
        self.graph_selector.pack(side="left", fill="x", padx = 5)
        self.graph_selector.set("Graph file")
        self.graph_selector.bind('<<ComboboxSelected>>', self.graph_changed)

        layout_list = ["circle", "drl", "fr", "kk", "large", "random", "tree"]
        self.plot_selector=ttk.Combobox(btn_frame, values = layout_list, state = "readonly")
        self.plot_selector.pack(side="left", fill="x", padx = 5)
        self.plot_selector.set("Graph layout")
        self.plot_selector.bind('<<ComboboxSelected>>', self.plot_changed)

        metric_values = ["none", "strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]
        self.node_metric_selector=ttk.Combobox(btn_frame, values = metric_values, state = "readonly")
        self.node_metric_selector.pack(side="left", fill="x", padx = 5)
        self.node_metric_selector.set("Node metric")
        self.node_metric_selector.bind('<<ComboboxSelected>>', self.node_changed)

        edge_metric_selector=ttk.Combobox(btn_frame, state = "readonly")
        edge_metric_selector.pack(side="left", fill="x", padx = 5)
        edge_metric_selector.set("Edge metric")

        cluster_button=tk.Button(btn_frame)
        cluster_button["text"] = "cluster nodes"
        cluster_button.pack(side="left", fill="x", padx = 5)
        cluster_button["command"] = self.cluster_button_command
        
        self.label = tk.Label(root, font = 'Helvetica 12 bold', text ="1. Select the directory where your files are stored with the  'files path'  button. \n 2. Then, select the graph file with the  'graph file'  drop-down menu to start the analysis.")
        self.label.pack(side = 'top', pady = 50)

    def path_button_command(self):
        self.dirpath = filedialog.askdirectory()
        # dirpath = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.csv"), ("All files", "*.txt*")])
        self.graph_selector["values"] = os.listdir(self.dirpath)
        
    def graph_changed(self, event):
        self.path_to_file = self.dirpath + "/" +  self.graph_selector.get()
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)
    
    def plot_changed(self, event):
        self.layout_style = self.plot_selector.get()
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)
        
    def node_changed(self, event):
        self.node_metric = self.node_metric_selector.get()
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)
        
    def plot_in_frame(self, layout_style = "fr", node_metric = "none"):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            root.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,400*px), dpi = 100)
        a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
        display_graph(self.path_to_file, a, layout = layout_style, node_metric = node_metric)
        
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")

    def cluster_button_command(self):
        NewWindow(root, self.dirpath)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
