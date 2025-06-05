import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter import filedialog

class settingsWindow(tk.Toplevel):
    """
    Opens a window showing the current settings (edge type, multilayer view and stats type),
        to allow user to change them.
    """    
    def __init__(self, root, app):
        super().__init__(root)
        self.root = root
        self.app = app
        self.title("Settings")
        self.geometry("400x300")
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if self.app.edge_type == "distance":
            self.app.edge_type_var.set(2)
            
        edge_label = tk.Label(self, text="Edge type")
        edge_label.grid(row = 1, column = 1)
        dist_button = tk.Radiobutton(self, text="Distance", variable = self.app.edge_type_var, value = 2, command = self.switch_edge_type)
        dist_button.grid(row = 1, column = 2)    
        aff_button = tk.Radiobutton(self, text="Affinity", variable = self.app.edge_type_var, value = 1, command = self.switch_edge_type)
        aff_button.grid(row = 1, column = 3)

        multilayer_label = tk.Label(self, text="Multilayer display")
        multilayer_label.grid(row = 2, column = 1)
        avg_button = tk.Radiobutton(self, text="Average", variable = self.app.view_var, value = 2, command = self.switch_view_type)
        avg_button.grid(row = 2, column = 2)
        multilayer_button = tk.Radiobutton(self, text="3D", variable = self.app.view_var, value = 1, command = self.switch_view_type)
        multilayer_button.grid(row = 2, column = 3)
        
        multilayer_label = tk.Label(self, text="Histogram type")
        multilayer_label.grid(row = 3, column = 1)
        sbs_button = tk.Radiobutton(self, text="Side-by-side", variable = self.app.view_var, value = 1, command = self.switch_histo_type)
        sbs_button.grid(row = 3, column = 2)
        stacked_button = tk.Radiobutton(self, text="Stacked", variable = self.app.view_var, value = 2, command = self.switch_histo_type)
        stacked_button.grid(row = 3, column = 3)
        
        fbloop_label = tk.Label(self, text="Remove feedback loops in plots: ")
        fbloop_label.grid(row = 4, column = 1)
        fb_loop_button = tk.Checkbutton(self, text="", variable = self.app.loops_var, onvalue = 1, offvalue = 0, command = self.loops_button_clicked)
        if self.app.loops_var.get() == 1:
            fb_loop_button.select()
        fb_loop_button.grid(row = 4, column = 2)
        
        scale_edge_label = tk.Label(self, text="Scale edge width: ")
        scale_edge_label.grid(row = 5, column = 1)
        scale_edge_button = tk.Checkbutton(self, text="", command = self.scale_edge_clicked)
        if self.app.scale_edge_width:
            scale_edge_button.select()
        scale_edge_button.grid(row = 5, column = 2)
        
        tk.Label(self, text="Edge thickness:").grid(row=6, column=1)
        self.edge_thickness_var = tk.StringVar() 
        self.edge_thickness_entry = tk.Entry(self, textvariable=self.edge_thickness_var)
        self.edge_thickness_entry.insert(0, self.app.edge_thickness_var.get())
        self.edge_thickness_entry.bind('<Return>', lambda event: self.on_enter_pressed(event))
        self.edge_thickness_entry.grid(row=6, column=2)
        
        tk.Label(self, text="Node thickness:").grid(row=7, column=1)
        self.node_thickness_var = tk.StringVar() 
        self.node_thickness_entry = tk.Entry(self, textvariable=self.node_thickness_var)
        self.node_thickness_entry.insert(0, self.app.node_thickness_var.get())
        self.node_thickness_entry.bind('<Return>', lambda event: self.on_enter_pressed(event))
        self.node_thickness_entry.grid(row=7, column=2)
        
        between_layer_label = tk.Label(self, text="Draw edges between layers: ")
        between_layer_label.grid(row = 8, column = 1)
        between_layer_button = tk.Checkbutton(self, text="", command = self.between_layer_clicked)
        if self.app.between_layer_edges:
            between_layer_button.select()
        between_layer_button.grid(row = 8, column = 2)

    def on_enter_pressed(self, event):
        edge_thickness_value = self.edge_thickness_entry.get()  # Note: using direct widget reference
        node_thickness_value = self.node_thickness_entry.get()  # Note: using direct widget reference
        self.app.edge_thickness_var.set(edge_thickness_value)
        self.app.node_thickness_var.set(node_thickness_value)
        self.app.plot_in_frame()
        
    def switch_edge_type(self):
        if self.app.edge_type == "affinity":
            self.app.edge_type = "distance"
        elif self.app.edge_type == "distance":
            self.app.edge_type = "affinity"
        self.app.plot_in_frame()
        
    def switch_view_type(self):
        if self.app.view_type == "3D":
            self.app.view_type = "avg"
        elif self.app.view_type == "avg":
            self.app.view_type = "3D"
        self.app.plot_in_frame()
                               
    def switch_histo_type(self):
        if self.app.histo_type == "stacked":
            self.app.histo_type = "side-by-side"
        elif self.app.histo_type == "side-by-side":
            self.app.histo_type = "stacked"
        self.app.stats_in_frame()
    
    def loops_button_clicked(self):
        if self.app.remove_loops:
            self.app.remove_loops = False
        elif not self.app.remove_loops:
            self.app.remove_loops = True
        self.app.plot_in_frame()
        
    def scale_edge_clicked(self):
        if self.app.scale_edge_width:
            self.app.scale_edge_width = False
        elif not self.app.scale_edge_width:
            self.scale_edge_width = True
        self.app.plot_in_frame()
        
    def between_layer_clicked(self):
        if self.app.between_layer_edges:
            self.app.between_layer_edges = False
        elif not self.app.between_layer_edges:
            self.app.between_layer_edges = True
        self.app.plot_in_frame()
        
    def on_close(self):
        self.destroy()


