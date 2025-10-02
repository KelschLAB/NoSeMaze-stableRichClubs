# Overview 
This repository contains the code for the article ["Stable clique membership in mouse societies requires oxytocin-enabled social sensory states"](https://www.biorxiv.org/content/10.1101/2025.08.26.672298v1). 
The data generated in this study are still under active use, so we provide here a limited dataset as a working example. The data are fully available upon request. 
The scripts used to produce the plots & results are accessible in the ```script``` folder of this repository.

## Scripts 
The main findings of this article can be reproduced by running the functions present in the ```if __name__ == "__main__":``` sections of the following scripts:
- To plot the time evolution of a group's social network as well as the detected rich-club members (Fig. 4), run the ```rc_in_graph_plots.py``` script.
- To compare experimental observations (number of OXTRΔAON in stable rich-clubs (sRC), littermates in sRC or reshuffled mice in sRC Fig. 5 & 6) to random chance, run the ```significance_plots.py``` script.
- To compute the normalized edge fluctuations (NEF, Fig. 6) of the different members of a social network, run the ```temporal_graph_metrics.py``` script.

To conduct the analysis of 

## Data format
The social network data read by the scripts provided in this repository should be saved as ```.csv```files, where the first row and column is used for indexing (animal RFID tags). 
Data is stored in matrix format meaning the number in row i and column j indicates the number of interaction between animal i and animal j. If the variable stored has a direction (for example approaches), 
then the enty [i, j] indicates the number of times animal i initiated the interaction towards animal j. Such a matrix might therefore not be symmetric. 
In case you are unsure, have a look at the example data provided in this repository.

## Versions requirements & Installation
