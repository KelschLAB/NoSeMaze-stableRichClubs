import unittest
import sys
import matplotlib.pyplot as plt
sys.path.append("..\\src")
from read_graph import *

path = "..\\data\\social network matrices 3days\\G5\\"
file1 = "approaches_resD3_1.csv"
file2 = "approaches_resD3_2.csv"
file3 = "approaches_resD3_3.csv"
file4 = "interactions_resD3_2.csv"

data = read_graph([path+file1], mnn = 3, return_ig=False)[0]
if isSymmetric(data):
    g = ig.Graph.Weighted_Adjacency(data, mode='undirected')
else:
    g = ig.Graph.Weighted_Adjacency(data, mode='directed')

class TestSum(unittest.TestCase):
    
    ## Test different keyword arguments in single layer case
    def test_default(self):
        """Tests if the displaying of graph with 'k-core' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "none", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....") 
            
    def test_betweenness(self):
        """Tests if the displaying of graph with 'betweenness' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "betweenness", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....") 

    def test_kcore(self):
        """Tests if the displaying of graph with 'k-core' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "k-core", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_rc(self):
        """Tests if the displaying of graph with 'rich-club' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "rich-club", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
        
    def test_pagerank(self):
        """Tests if the displaying of graph with 'page rank' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "page rank", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_hubscore(self):
        """Tests if the displaying of graph with 'hub score' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "hub score", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_evcentrality(self):
        """Tests if the displaying of graph with 'eigenvector centrality' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "eigenvector centrality", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")          
            
    def test_authority(self):
        """Tests if the displaying of graph with 'authority score' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "authority score", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")      
        
    def test_closeness(self):
        """Tests if the displaying of graph with 'closeness' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "closeness", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")  
            
    def test_strength(self):
        """Tests if the displaying of graph with 'strength' keyword argument runs without error."""
        try:
            fig, ax = plt.subplots(1, 1)
            display_graph([path+file4], ax, mnn = 4, deg = 2, node_metric = "strength", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")  
            
    ## Test different keyword arguments in multi-layer case
    def test_default_3d(self):
        """Tests if the displaying of graph with 'k-core' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "none", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....") 
            
    def test_betweenness_3d(self):
        """Tests if the displaying of graph with 'betweenness' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "betweenness", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....") 

    def test_kcore_3d(self):
        """Tests if the displaying of graph with 'k-core' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "k-core", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_rc_3d(self):
        """Tests if the displaying of graph with 'rich-club' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "rich-club", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
        
    def test_pagerank_3d(self):
        """Tests if the displaying of graph with 'page rank' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "page rank", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_hubscore_3d(self):
        """Tests if the displaying of graph with 'hub score' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "hub score", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")
            
    def test_evcentrality_3d(self):
        """Tests if the displaying of graph with 'eigenvector centrality' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "eigenvector centrality", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")          
            
    def test_authority_3d(self):
        """Tests if the displaying of graph with 'authority score' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "authority score", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")      
        
    def test_closeness_3d(self):
        """Tests if the displaying of graph with 'closeness' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "closeness", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")  
            
    def test_strength_3d(self):
        """Tests if the displaying of graph with 'strength' keyword argument runs without error."""
        try:
            fig = plt.Figure()
            ax = fig.add_subplot(111, projection='3d')
            display_graph([path+file4, path+file3], ax, mnn = 4, deg = 2, node_metric = "strength", mutual = True, idx = [], cluster_num = None)
            plt.close(fig)
        except:
            self.fail("....")  
            
 


if __name__ == '__main__':
    unittest.main()