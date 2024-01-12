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
        self.idx = []
        self.cluster_num = 0
    
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
        metric_values = ["none", "strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score"]
        self.node_metric_selector=ttk.Combobox(btn_frame, values = metric_values, state = "readonly")
        self.node_metric_selector.pack(side="left", fill="x", padx = 5)
        self.node_metric_selector.set("Node metric")
        self.node_metric_selector.bind('<<ComboboxSelected>>', self.node_changed)

        # metric selection for edges
        edge_metric_selector=ttk.Combobox(btn_frame, state = "readonly")
        edge_metric_selector.pack(side="left", fill="x", padx = 5)
        edge_metric_selector.set("Edge metric")
        
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
        stats_btn = tk.Button(result_display_frame, text='statistics')
        stats_btn.pack(side = "left")
        
    def get_checked(self):
        lst = []
        for i, item in enumerate(self.path_variable_list):
            if item.get() == "1":
                lst.append(self.path_label_list[i])
        self.active_path_list = lst
        
        self.path_to_file = [self.dirpath + "/" + self.active_path_list[i] for i in range(len(self.active_path_list))]
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)

    def path_button_command(self):
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
            globals()['var'+str(i)] = tk.StringVar()
        # Finally adding values to the actual Menubutton
        for i in range(0, len(path_list)):
            self.path_variable_list.append(globals()['var'+str(i)])
            self.path_label_list.append(path_list[i])
            menu.add_checkbutton(label = self.path_label_list[i], variable = self.path_variable_list[i], command=self.get_checked)
        self.graph_selector.configure(menu=menu)

        
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
        display_graph(self.path_to_file, a, layout = layout_style, node_metric = node_metric, idx = self.idx, cluster_num = self.cluster_num)
        
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")

    def cluster_button_command(self):
        NewWindow(root, self, self.dirpath)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
