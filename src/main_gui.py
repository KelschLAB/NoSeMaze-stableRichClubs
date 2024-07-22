import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from pprint import pprint
import webbrowser
from tkinter import filedialog
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from functools import partial
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
import os
from read_graph import *
from clustering_window import *

#To-do: 
#       - figure out why -1 in nn for mnn
#       - Check that feedback loops rm works, and that display of distance graphs also works
#       - mnn in clustering should be fixed (the displayed cut graph isnt cut properly)
#       - write tests

###### For future version: 
#       - Include compatibility with other formats (right now, only compatible with csv format).
#       - include minimal spanning tree, and jaccard metric (Edge metrics for visual graph analytics: a comparative study) in later version of project.
#       - add local clustering coefficient metric
#       - Include measure of topology type? i.e. homophily/clustering/degree/spatial 
#           (see Spatially embedded recurrent neural networks reveal widespread links between structural and functional neuroscience, 2023)

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
        self.mutual = True
        self.degree = 0
        self.idx = []
        self.cluster_num = 0
        self.display_type = "plot"
        self.edge_type = "affinity"
        self.view_type = "3D"
        self.histo_type = "stacked"
        self.remove_loops = True # for the app to know if feedback loops should be plotted or not 
        self.edge_type_var = tk.IntVar(value = 1) # variable for changing affinity/distance in settings window
        self.view_var = tk.IntVar(value = 1)      # variable for changing view from 3D to average in settings window
        self.loops_var = tk.IntVar(value = 1)     # variable for the removal of feedback loops in graph display
    
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
        path_button=tk.Menubutton(btn_frame, text = "File")
        path_button.menu = tk.Menu(path_button, tearoff=False)   
        path_button["menu"]= path_button.menu  
        path_button.menu.add_command(label="Open",command = self.path_button_command)
        path_button.menu.add_command(label="Settings", command=self.settings_window)
        path_button.menu.add_command(label="Reset",command = self.reset)
        link = "https://stackoverflow.com/questions/71458060/how-to-open-a-link-with-an-specific-button-tkinter" #link to docs
        path_button.menu.add_command(label="Help", command =lambda: webbrowser.open(link))
        path_button["justify"] = "center"
        path_button.pack(side="left", fill="x", padx = 5)
        
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
        metric_values = ["none", "strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score", "rich-club", "k-core"]
        self.node_metric_selector=ttk.Combobox(btn_frame, values = metric_values, state = "readonly")
        self.node_metric_selector.pack(side="left", fill="x", padx = 5)
        self.node_metric_selector.set("Node metric")
        self.node_metric_selector.bind('<<ComboboxSelected>>', self.node_changed)

        # Graph-cut type selection
        graphcut_values = ["none", "threshold", "mutual nearest neighbors", "nearest neighbors"]
        self.graphcut_selector=ttk.Combobox(btn_frame, values = graphcut_values, state = "readonly")
        self.graphcut_selector.pack(side="left", fill="x", padx = 5)
        self.graphcut_selector.set("Graph-cut type")
        self.graphcut_selector.bind('<<ComboboxSelected>>', self.graphcut_param_window)
        
        # button to open clustering window
        cluster_button = tk.Button(btn_frame)
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
        
    # functions for 'file' menu 
    def path_button_command(self):
        """ Selects the directory path where graph layers are contained, and updates the list of selectable graph layer """
        self.dirpath = filedialog.askdirectory(title="Select the directory which contains the graph file(s)")
        menu = tk.Menu(self.graph_selector, tearoff=False)
        self.path_variable_list = []
        self.path_label_list = []
        # main list holding menu values
        path_list = [p for p in os.listdir(self.dirpath) if p.endswith(".csv")]
                
        # Creating variables to store paths dynamically
        for i in range(0, len(path_list)):
            globals()['var'+str(i)] = tk.StringVar(self.graph_selector)
        # Finally adding values to the actual Menubutton
        for i in range(0, len(path_list)):
            self.path_variable_list.append(globals()['var'+str(i)])
            self.path_label_list.append(path_list[i])
            menu.add_checkbutton(label = self.path_label_list[i], variable = self.path_variable_list[i], command=self.get_checked)
        self.graph_selector.configure(menu=menu)

    def reset(self):
        """
        Resets every user-selected options aside from the path/graph selected and the type of graph edges.
        """
        def reset_clicked(self, win):
            self.plot_selector.set("Graph layout")
            self.node_metric_selector.set("Node metric")
            self.graphcut_selector.set("Graph-cut type")
            self.node_metric = None
            self.idx = []
            self.percentage_threshold = 0.0
            self.mnn_number = None
            self.mutual = True
            self.plot_in_frame()
            win.destroy()
            
        popup = tk.Toplevel(root)
        popup.wm_title("Reset plot parameters?")
        label = ttk.Label(popup, text=" Graph layout, color labels and cut threshold will be reset.\nPath and edge type (affinity/distance) will not be affected.")
        label.pack(side="top")
        B1 = ttk.Button(popup, text="Ok", command = partial(reset_clicked, self, popup))
        B1.pack(side="left", padx = 50)
        B2 = ttk.Button(popup, text="No", command = popup.destroy)
        B2.pack(side="left")
        
    def settings_window(self):
        """
        Opens a window showing the current settings (edge type, multilayer view and stats type),
        to allow user to change them.
        """
        settings_popup = tk.Toplevel(root)
        settings_popup.wm_title("Settings")
        if self.edge_type == "distance":
            self.edge_type_var.set(2)
            
        edge_label = tk.Label(settings_popup, text="Edge type")
        edge_label.grid(row = 1, column = 1)
        dist_button = tk.Radiobutton(settings_popup, text="Distance", variable = self.edge_type_var, value = 2, command = self.switch_edge_type)
        dist_button.grid(row = 1, column = 2)    
        aff_button = tk.Radiobutton(settings_popup, text="Affinity", variable = self.edge_type_var, value = 1, command = self.switch_edge_type)
        aff_button.grid(row = 1, column = 3)

        multilayer_label = tk.Label(settings_popup, text="Multilayer display")
        multilayer_label.grid(row = 2, column = 1)
        avg_button = tk.Radiobutton(settings_popup, text="Average", variable = self.view_var, value = 2, command = self.switch_view_type)
        avg_button.grid(row = 2, column = 2)
        multilayer_button = tk.Radiobutton(settings_popup, text="3D", variable = self.view_var, value = 1, command = self.switch_view_type)
        multilayer_button.grid(row = 2, column = 3)
        
        multilayer_label = tk.Label(settings_popup, text="Histogram type")
        multilayer_label.grid(row = 3, column = 1)
        sbs_button = tk.Radiobutton(settings_popup, text="Side-by-side", variable = self.view_var, value = 1, command = self.switch_histo_type)
        sbs_button.grid(row = 3, column = 2)
        stacked_button = tk.Radiobutton(settings_popup, text="Stacked", variable = self.view_var, value = 2, command = self.switch_histo_type)
        stacked_button.grid(row = 3, column = 3)
        
        fbloop_label = tk.Label(settings_popup, text="Remove feedback loops in plots: ")
        fbloop_label.grid(row = 4, column = 1)
        fb_loop_button = tk.Checkbutton(settings_popup, text="", variable = self.loops_var, onvalue = 1, offvalue = 0, command = self.loops_button_clicked)
        if self.loops_var.get() == 1:
            fb_loop_button.select()
        fb_loop_button.grid(row = 4, column = 2)

    def switch_edge_type(self):
        if self.edge_type == "affinity":
            self.edge_type = "distance"
        elif self.edge_type == "distance":
            self.edge_type = "affinity"
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)
        
    def switch_view_type(self):
        if self.view_type == "3D":
            self.view_type = "avg"
        elif self.view_type == "avg":
            self.view_type = "3D"
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)
         
    def switch_histo_type(self):
        if self.histo_type == "stacked":
            self.histo_type = "side-by-side"
        elif self.histo_type == "side-by-side":
            self.histo_type = "stacked"
        self.stats_in_frame()
    
    def loops_button_clicked(self):
        if self.remove_loops:
            self.remove_loops = False
        elif not self.remove_loops:
            self.remove_loops = True
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)
            

    # central function for plotting the graph(s)
    def plot_in_frame(self, layout_style = "fr", node_metric = "none", percentage_threshold=0.0, mnn = None, deg = 0):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            root.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(950*px,500*px))
        if len(self.path_to_file) > 1 and self.view_type == "3D":
            a = f.add_subplot(111, projection='3d')
            a.set_box_aspect((2,2,1), zoom=1.5)
        else:
            a = f.add_subplot(111)
            
        display_graph(self.path_to_file, a, percentage_threshold = self.percentage_threshold, mnn = self.mnn_number, mutual = self.mutual, \
                      avg_graph = self.view_type == "avg", affinity = self.edge_type == "affinity",  rm_fb_loops = self.remove_loops, \
                      layout = layout_style, node_metric = self.node_metric, \
                      idx = self.idx, cluster_num = self.cluster_num, layer_labels=self.path_to_file, deg = self.degree)
        
        f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Greys), ax=a, label="Normalized edge value", shrink = 0.3, location = 'right', pad = 0.1)
        if node_metric != "none":
            f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Reds), ax=a, label="Normalized metric value", shrink = 0.3, location = 'left')
            # f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.viridis), ax=a, label="Relative edge value", shrink = 0.3, location = 'right', pad = 0.1)
        else:
            cb = f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Reds), ax=a, label="Normalized metric value", shrink = 0.3, location = 'left')
            cb.remove()
        
        f.subplots_adjust(left=0, bottom=0, right=0.948, top=1, wspace=0, hspace=0)

        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")
        
    # central function for displaying the statistics of the graph(s)
    def stats_in_frame(self):
        for fm in self.content_frame.winfo_children():
            fm.destroy()
            root.update()
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        f = Figure(figsize=(800*px,400*px), dpi = 100)
        a = f.add_subplot(111)
        display_stats(self.path_to_file, a, percentage_threshold=self.percentage_threshold, 
                      affinity = self.edge_type == "affinity", mnn = self.mnn_number, mutual = self.mutual, 
                      node_metric = self.node_metric, avg_graph = self.view_type == "avg",
                      stacked = self.histo_type == "stacked", deg = self.degree)
    
        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.config(text="")

    # function for graph selection and display
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

    def graphcut_param_window(self, event):
        """
        Prompt for selecting the parameter for graph cut, i.e. removal of edges.
        
        If 'threshold' is selected, a percentage has to be given as an input. This 
            percentage is scaled w.r.t to the strongest edge in the graph. Any edge with 
            a value below the input threshold (in % of strongest edge) will be removed.
        If 'mutual nearest neighbors' is selected, only the edges between nodes that are
            mutual nearest neighbors are preserved. The input value specifies the neighboring
            'degree', e.g. 1 means only 1st neighbors are preserved, 2 means 
            up to the 2nd nearest neighbors, 3 means up to the 3rd nearest neighbor etc.
        """
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
            self.mutual = True
        elif self.graphcut_selector.get() == "nearest neighbors":
            self.percentage_threshold = 0.0
            self.mnn_number = int(self.graphcut_entry.get())
            self.mutual = False
        self.node_metric = self.node_metric_selector.get()
        self.new_window.destroy()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        else:
            self.stats_in_frame()
        
    def plot_clicked(self):
        self.display_type = "plot"
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        
    def stats_clicked(self):
        self.display_type = "stats"
        self.stats_in_frame()
        
    def plot_changed(self, event):
        self.layout_style = self.plot_selector.get()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)

    def rich_club_window(self):
        self.new_window = tk.Toplevel(root)
        self.new_window.title("Degree value for rich-club")
        tk.Label(self.new_window, text="Enter degree").grid(row=0,column=0)
        self.rich_club_entry = tk.Entry(self.new_window)
        self.rich_club_entry.grid(row=1,column=0)
        tk.Button(self.new_window, text="Compute rich-club!", command=self.rich_club_changed).grid(row=2,column=0)
        
    def k_core_window(self):
        self.new_window = tk.Toplevel(root)
        self.new_window.title("Degree value for k-core")
        tk.Label(self.new_window, text="Enter degree").grid(row=0,column=0)
        self.rich_club_entry = tk.Entry(self.new_window)
        self.rich_club_entry.grid(row=1,column=0)
        tk.Button(self.new_window, text="Compute k-core!", command=self.rich_club_changed).grid(row=2,column=0)
        
    def rich_club_changed(self):
        self.degree = int(self.rich_club_entry.get())
        self.new_window.destroy()
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric, percentage_threshold=self.percentage_threshold, mnn = self.mnn_number, deg = self.degree)
        else:
            self.stats_in_frame()
        
    def node_changed(self, event):
        self.node_metric = self.node_metric_selector.get()
        if self.node_metric == "rich-club":
            self.rich_club_window()
            
        if self.node_metric == "k-core":
            self.k_core_window()
            
        if self.display_type == "plot":
            self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric,\
                               percentage_threshold=self.percentage_threshold, mnn = self.mnn_number)
        else:
            self.stats_in_frame()

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
        SpectralClustWindow(root, self, self.path_to_file)
        self.clustertype_wdw.destroy()
        
    def unsupervised_button(self):
        self.idx = community_clustering(self.path_to_file)
        print(self.idx)
        self.cluster_num = max(self.idx)+1
        self.clustertype_wdw.destroy()
        self.plot_in_frame(layout_style = self.layout_style, node_metric = self.node_metric)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
