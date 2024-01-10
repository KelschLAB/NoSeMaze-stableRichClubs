import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
from read_graph import *

path_to_file = None

root = tk.Tk()
root.geometry('500x500')
root.title("Multilayer graph analysis")


##### switching functions 
def switch(page):
    for fm in main_frame.winfo_children():
        fm.destroy()
        root.update()
    page()
    
def load_menu():
    load_frame = tk.Frame(main_frame, bg = "grey")
    load_frame.pack(pady=0)
    
    def open_file_dialog():
        global path_to_file
        path_to_file = filedialog.askopenfilename(title="Select a File", filetypes=[("Text files", "*.csv"), ("All files", "*.txt*")])
        # if file_path:
        # selected_file_label.config(text=f"Selected File: {file_path}")
        # process_file(file_path, path_to_file)
    
    # def process_file(file_path, storage_variable):
    #     # Implement your file processing logic here
    #     # For demonstration, let's just display the contents of the selected file
    #     # try:
    #     #     with open(file_path, 'r') as file:
    #     #         file_contents = file.read()
    #     #         file_text.delete('1.0', tk.END)
    #     #         file_text.insert(tk.END, file_contents)
    #     # except Exception as e:
    #     selected_file_label.config(text=file_path)
    #     storage_variable = file_path
    
    open_button = tk.Button(main_frame, text="Open File", command=open_file_dialog)
    open_button.pack(side = "top")
   
    selected_file_label = tk.Label(main_frame, text="Selected File:")
    selected_file_label.pack(side = "top")
   
    file_text = tk.Text(main_frame, wrap=tk.WORD, height=10, width=10)
    file_text.pack(side = "top")

def stats_menu():
    ##### Stats menu button
    stats_btn_frame = tk.Frame(main_frame, bg = "grey")
    stats_btn_frame.pack(pady=0)
    stats_btn_frame.configure(width=400, height=100)
    stats_btn = tk.Button(stats_btn_frame, text='statistics')
    stats_btn.place(x=0, y=0, width= 100, height = 100)
    
    def plot_in_frame():
        print(path_to_file)
        f = Figure(figsize=(3,3), dpi = 100)
        a = f.add_subplot(111)
        # a.plot([1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10])
        display_graph(path_to_file, a)
        
        canvas = FigureCanvasTkAgg(f, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    plot_btn = tk.Button(stats_btn_frame, text='plot',command=plot_in_frame)
    plot_btn.place(x=100, y=0, width= 100, height = 100)
    


def clustering_menu():
    ##### Clustering menu button
    cluster_btn_frame = tk.Frame(main_frame, bg = "grey")
    cluster_btn_frame.pack(pady=0)
    cluster_btn_frame.configure(width=400, height=100)
    param_btn = tk.Button(cluster_btn_frame, text='hyperparameters \n exploration')
    param_btn.place(x=0, y=0, width= 100, height = 100)
    cluster_btn = tk.Button(cluster_btn_frame, text='Run clustering')
    cluster_btn.place(x=150, y=0, width= 150, height = 100)
    vis_btn = tk.Button(cluster_btn_frame, text='Results vizualisation')
    vis_btn.place(x=250, y=0, width= 150, height = 100)


###### Home buttons and page
home_frame = tk.Frame(root, bg = "grey")
home_frame.pack(side="left")
home_frame.configure(width=100, height=500)
main_frame = tk.Frame(root, bg = "blue")
main_frame.pack(fill=tk.BOTH, expand=True)
load_btn = tk.Button(home_frame, text='Home', command=lambda: switch(load_menu))
load_btn.place(x=0, y=0, width= 100, height = 100)
stats_btn = tk.Button(home_frame, text='statistics', command=lambda: switch(stats_menu))
stats_btn.place(x=0, y=200, width= 100, height = 100)
cluster_btn = tk.Button(home_frame, text='clustering', command=lambda: switch(clustering_menu))
cluster_btn.place(x=0, y=400, width= 100, height = 100)



load_menu()
root.mainloop()