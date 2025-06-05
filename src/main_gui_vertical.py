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
from settings_window import settingsWindow

#To-do: 
#       - what does it mean to be neighbors in terms of chasing?? (directed interaction). Rich club coloring does not seem to work
#           properly in these cases. It only colors nodes that have outgoing edges.
#       - setting: stacked histogram and side by side are inverted.
#       - fix overlapping small white link between nodes that make visualization hard
#       - remove colorbar from 3D view
#       - figure out why -1 in nn for mnn
#       - Check that feedback loops rm works, and that display of distance graphs also works
#       - mnn in clustering should be fixed (the displayed cut graph isnt cut properly)
#       - statistical test of random chance for rich club in statistics tabs
#       - add to settings possibility to change colormaps for layers, base node size, base colors and colormap for edges.
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
        width=1200
        height=700
        
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=True, height=True)
        root.title("Multilayer graph analysis")

        # variables that can change after interacting with the buttons
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
        self.edge_thickness_var = tk.StringVar(value = "5") # variable for changing edge type in settings window
        self.node_thickness_var = tk.StringVar(value = "15") # variable for changing edge type in settings window
        self.scale_edge_width = True # variable for scaling the thickness of edge to their value
        self.between_layer_edges = True

        self.color1 = "#E4F8FF"
        self.color2 = "#FFF7E3"
        self. color3 = "#FFE8E3"
        self.color_display_btn = "#FFD7CF"
    
        # Frames
        menu_frame = tk.Frame(root, bg = self.color1, highlightbackground="gray", highlightthickness=1)
        menu_frame.place(relx=0, rely=0, relwidth=0.22, relheight=0.2)
        
        btn_frame = tk.Frame(root, bg = self.color2, highlightbackground="gray", highlightthickness=1) #main buttons frame
        btn_frame.place(relx=0, rely=0.2, relwidth=0.22, relheight=0.65)
        
        result_display_frame = tk.Frame(root, bg = self.color3, highlightbackground="gray", highlightthickness=1) #result display frame
        result_display_frame.place(relx=0, rely=0.78, relwidth=0.22, relheight=0.22)

        self.content_frame = tk.Frame(root) # content frame, for plotting and stats
        self.content_frame.place(relx=0.22, rely=0.1, relwidth=0.78, relheight=0.8)

        # File menu/settings button
        load_label = tk.Label(menu_frame, text = "Data loading / Settings", font = 'Helvetica 12 bold', bg = self.color1)
        load_label.place(relx = 0.1, rely = 0.1, relwidth=0.8, relheight=0.35)
        load_button = tk.Button(menu_frame, text = "Open")
        load_button.place(relx=0.05, rely = 0.5, relwidth=0.43, relheight=0.35)
        load_button["command"] = self.load_button_command
        settings_button = tk.Menubutton(menu_frame, text = "Settings")
        settings_button.menu = tk.Menu(settings_button, tearoff=False)   
        settings_button["menu"]= settings_button.menu  
        settings_button.menu.add_command(label="Settings", command=self.settings_window)
        settings_button.menu.add_command(label="Reset",command = self.reset)
        link = "https://stackoverflow.com/questions/71458060/how-to-open-a-link-with-an-specific-button-tkinter" #link to docs
        settings_button.menu.add_command(label="Help", command =lambda: webbrowser.open(link))
    #    settings_button["justify"] = "center"
        settings_button.place(relx=0.53,rely = 0.5, relwidth=0.43, relheight=0.35)
        
        # Graph file(s) selection Menubutton
        analysis_label = tk.Label(btn_frame, text = "Data analysis", font = 'Helvetica 12 bold', bg = self.color2)
        analysis_label.place(relx = 0.1, rely = 0.05, relwidth=0.8)
        origin, distance_between = 0.15, 0.15
        padx, pady, font = 0, 30, '5'
        graph_selector_label = tk.Label(btn_frame, text = "Sub-graph:", bg = self.color2)
        graph_selector_label.place(relx= 0.07, rely = origin, relwidth=0.25, relheight=0.08)
        self.graph_selector=ttk.Menubutton(btn_frame, text = "Select graph file(s)")
        self.graph_selector.place(relx= 0.34, rely = origin, relwidth=0.57, relheight=0.08)

        # self.graph_selector.grid(row = 0, column = 1, padx = padx, pady = pady)
        # self.graph_selector.pack(side="left", fill="x", padx = 5)
        self.path_variable_list = [] # storing the menu options here
        self.path_label_list = []
        self.active_path_list = [] # storing selected paths here
        # self.graph_selector.set("Graph file")
        self.graph_selector.bind('<<ComboboxSelected>>', self.get_checked)

        # layout selection
        layout_label = tk.Label(btn_frame, text = "Layout: ", bg = self.color2)
        layout_label.place(relx=0.12, rely = origin+distance_between, relheight=0.06)
        layout_list = ["circle", "drl", "fr", "kk", "large", "random", "tree"]
        self.plot_selector=ttk.Combobox(btn_frame, values = layout_list, state = "readonly")
        # self.plot_selector.pack(side="left", fill="x", padx = 5)
        # self.plot_selector.grid(row = 1, column = 1, pady = pady)
        self.plot_selector.place(relx=0.35, rely = origin+distance_between, relheight=0.06)

        self.plot_selector.set("Graph layout")
        self.plot_selector.bind('<<ComboboxSelected>>', self.plot_changed)

        # metric selection for nodes
        metric_label = tk.Label(btn_frame, text = "Metric: ", bg = self.color2)
        metric_label.place(relx=0.12, rely = origin+2*distance_between, relheight=0.06)
        metric_values = ["none", "strength", "betweenness", "closeness", "eigenvector centrality", "page rank", "hub score", "authority score", "rich-club", "k-core"]
        self.node_metric_selector=ttk.Combobox(btn_frame, values = metric_values, state = "readonly")
        self.node_metric_selector.place(relx=0.35, rely = origin+2*distance_between, relheight=0.06)
        self.node_metric_selector.set("Node metric")
        self.node_metric_selector.bind('<<ComboboxSelected>>', self.node_changed)

        # Graph-cut type selection
        graphcut_label = tk.Label(btn_frame, text = "Graph cut: ", bg = self.color2)
        graphcut_label.place(relx=0.05, rely = origin+3*distance_between, relheight=0.06)
        graphcut_values = ["none", "threshold", "mutual nearest neighbors", "nearest neighbors"]
        self.graphcut_selector=ttk.Combobox(btn_frame, values = graphcut_values, state = "readonly")
        self.graphcut_selector.place(relx=0.35, rely = origin+3*distance_between, relheight=0.06)
        self.graphcut_selector.set("Graph-cut type")
        self.graphcut_selector.bind('<<ComboboxSelected>>', self.graphcut_param_window)
        
        # button to open clustering window
        cluster_button = tk.Button(btn_frame)
        cluster_button["text"] = "cluster nodes"
        cluster_button.place(relx=0.2, rely = origin+4*distance_between, relwidth= 0.6, relheight=0.1)
        cluster_button["command"] = self.cluster_button_command
        
        # Starting instructions label
        self.label = tk.Label(self.content_frame, font = 'Helvetica 13 bold', 
                              text ="1. Select the directory/folder where your files are stored with the  'Open'  button. \n 2. Then, select the graph file(s) with the 'sub-graph' drop-down menu to start plotting.\n 3. You can switch the result display with the 'plot' and 'statistics' buttons")
        self.label.place(relx=0.1, rely=0.2, relwidth=0.8, relheight=0.4)
        
        # result display widgets
        tk.Label(result_display_frame, text="Display type", font = 'Helvetica 12 bold', bg =  self.color3).place(relx = 0.1, rely = 0.1, relwidth=0.8, relheight=0.2)
        self.plot_btn = tk.Button(result_display_frame, text='Plot')
        self.plot_btn.place(relx = 0.1, rely = 0.4, relwidth=0.38, relheight=0.2)
        self.plot_btn["command"] = self.plot_clicked
        self.stats_btn = tk.Button(result_display_frame, text='Statistics')
        self.stats_btn.place(relx = 0.5, rely = 0.4, relwidth=0.39, relheight=0.2)
        self.stats_btn["command"] = self.stats_clicked
        self.plot_btn.config(bg="#d1d1d1")
        self.stats_btn.config(bg="#f0f0f0")
        
    # functions for 'file' menu 
    def load_button_command(self):
        """ Selects the directory/folder path where graph layers are contained, and updates the list of selectable graph layer """
        self.dirpath = filedialog.askdirectory(title="Select the directory/folder which contains the graph file(s)")
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
        settings_menu = settingsWindow(root, self) # creates a settings window
         
    # central function for plotting the graph(s)
    def plot_in_frame(self):
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
                      layout = self.layout_style, node_metric = self.node_metric, \
                      idx = self.idx, cluster_num = self.cluster_num, layer_labels=self.path_to_file, deg = self.degree,
                      edge_width = int(self.edge_thickness_var.get()), node_size = int(self.node_thickness_var.get()), 
                      scale_edge_width = self.scale_edge_width, between_layer_edges = self.between_layer_edges)
            
        if self.scale_edge_width:
            f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Greys), ax=a, label="Normalized edge value", shrink = 0.3, location = 'right', pad = 0.1)
        if self.node_metric != "none":
            f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Reds), ax=a, label="Normalized metric value", shrink = 0.3, location = 'left')
            # f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.viridis), ax=a, label="Relative edge value", shrink = 0.3, location = 'right', pad = 0.1)
        else: # to keep layout consistent across changes of settings
            cb = f.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cm.Reds), ax=a, label="Normalized metric value", shrink = 0.3, location = 'left')
            cb.remove()
        
        f.subplots_adjust(left=0, bottom=0, right=0.948, top=1, wspace=0, hspace=0)

        canvas = FigureCanvasTkAgg(f, master=self.content_frame)
        NavigationToolbar2Tk(canvas, self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()#fill=tk.BOTH, expand=True, side="top") 
        self.label.destroy()
        
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
       # self.label.config(text="")

    # function for graph selection and display
    def get_checked(self):
        """ Updates list of paths when graph layer selector is clicked """
        lst = []
        for i, item in enumerate(self.path_variable_list):
            if item.get() == "1":
                lst.append(self.path_label_list[i])
        self.active_path_list = lst
        self.path_to_file = [self.dirpath + "/" + self.active_path_list[i] for i in range(len(self.active_path_list))]   
        self.plot_in_frame()

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
                self.plot_in_frame()
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
            self.plot_in_frame()
        else:
            self.stats_in_frame()
        
    def plot_clicked(self):
        self.display_type = "plot"
        self.plot_in_frame()
        self.plot_btn.config(bg="#d1d1d1")
        self.stats_btn.config(bg="#f0f0f0")

    def stats_clicked(self):
        self.display_type = "stats"
        self.stats_in_frame()
        self.plot_btn.config(bg="#f0f0f0")
        self.stats_btn.config(bg="#d1d1d1")
        
    def plot_changed(self, event):
        self.layout_style = self.plot_selector.get()
        if self.display_type == "plot":
            self.plot_in_frame()

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
            self.plot_in_frame()
        else:
            self.stats_in_frame()
        
    def node_changed(self, event):
        self.node_metric = self.node_metric_selector.get()
        if self.node_metric == "rich-club":
            self.rich_club_window()
            
        if self.node_metric == "k-core":
            self.k_core_window()
            
        if self.display_type == "plot":
            self.plot_in_frame()
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
        self.plot_in_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
