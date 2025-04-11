import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

def plot_genotype_in_rc():
    """
    Plot The flow of mice genotypes going in the RC, i.e.
    ox- and ox+ mice.
    """
    # Define the first step
    source = [0, 1, 0, 1]  # Starting nodes
    target = [2, 2, 3, 3]  # Ending nodes
    value = [29, 2, 72, 30]  # Flow values

    # Define node labels and colors
    labels = ["WT", "Mutant", "RC", "Non-RC"]
    colors = ["green", "red", "blue", "gray"]

    # Create the Sankey diagram
    link = dict(source=source, target=target, value=value)
    node = dict(label=labels, color=colors, pad=15, thickness=20)
    data = go.Sankey(link=link, node=node, arrangement="snap")  # Use "snap" to control node order

    fig = go.Figure(data)
    fig.show()

def plot_rc_shfulle_flow():
    """
    Plot The flow of mice genotypes going in the RC, i.e.
    ox- and ox+ mice, after reshuffling, and shows how it flows back.
    """
    # Define the first step
    source = [0, 0, 1, 1]  # Starting nodes
    target = [2, 3, 2, 3]  # Ending nodes
    value = [8, 2, 2, 8]   # Flow values

    # Add the second step
    source += [2, 2, 3, 3]    # Starting nodes for the second step
    target += [4, 5, 4, 5]    # Ending nodes for the second step
    value += [5, 3, 6, 4]     # Flow values for the second step

    # Create the Sankey diagram
    link = dict(source=source, target=target, value=value)
    data = go.Sankey(link=link)

    fig = go.Figure(data)
    fig.show()

plot_genotype_in_rc()