from cx_Freeze import setup, Executable

setup(
name="Graph_visualizer",
version="1.0",
description="GUI for analysis and visualization of multilayered graphs",
executables=[Executable("main_gui_vertical.py")]
)