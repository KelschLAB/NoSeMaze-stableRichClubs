import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.cluster as clust
from scipy.linalg import inv, sqrtm, eigh
from collections import Counter
from sklearn.manifold import TSNE
from warnings import warn
from copy import deepcopy
from matplotlib.lines import Line2D
import torch
from multilayer_sc import *

class graphClusterer:
    """A class to perform clustering on single or multiple graphs.

    Parameters:
        input_graphs (list): List containing the matrices of the graphs to cluster.

    Attributes:
        D (ndarray): The matrix of pairwise distance between input series.
            This is computed with DTW (see [1]).

        S (ndarray): The matrix of pairwise similarities between input series.
            This is computed using a gaussian kernel using a local nn (see [4]).

        idx (list): List of indexes storing the result of the clustering. 
            Each index corresponds to a series in the input, and indicates which cluster it belongs to.

        V (ndarray): The lowest eigenvectors of the laplacian matrix. This is useful for dimensionality reduction and
            visualization of the different input signals.

    References:
        [2]: Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17, 395-416.
        [3]: Tran, L. (2012). Application of three graph Laplacian based semi-supervised learning methods to protein function prediction problem. arXiv preprint arXiv:1211.4289.
        [4]: Zelnik-Manor, L., & Perona, P. (2004). Self-tuning spectral clustering. Advances in neural information processing systems, 17.
    """

    def __init__(self, input_graphs, isAffinity, connectivity_type):
        self.D = input_graphs 
        self.connectivity_type = connectivity_type
        self.idx = [] # initializing as empty, fill after clustering
        self.V = [] # for storing eigenvectors of the laplacian
        self.isAffinity = isAffinity

    def mutual_nn_affinity(self, S, nn = 2):
        """
        Makes a sparse affinity matrix by keeping only the nn mutual nearest neighbours in the connections. 
        Other connexions are set to 0.
        """
        mnn_S = []
        nn -= 1 # to account for python indexing
        for graph in range(len(S)):
            curr_S = np.zeros_like(S[graph]) 
            neighbors_i = np.argsort(-S[graph], 1) #computing the nearest neighbors for local nn estimation
            neighbors_j = np.argsort(-S[graph], 0)
            for i in range(S[graph].shape[1]):
                for j in range(i, S[graph].shape[1]):
                    if any(np.isin(neighbors_i[i, 0:nn], j)) and any(np.isin(neighbors_j[0:nn, j], i)): #local nearest distance estimation
                        curr_S[i, j] += S[graph][i, j]
                        curr_S[j, i] += S[graph][j, i]
            mnn_S.append(curr_S)
            
        return mnn_S
        
    def epsilon_affinity(self, S, epsilon = 2):
        """
        Makes a sparse affinity matrix by keeping only the neighbours within epsilon of each other. 
        Other connexions are set to 0.
        """
        eps_S = []
        for graph in range(len(S)):
            curr_S = np.zeros_like(S[graph]) 
            for i in range(S[graph].shape[1]):
                for j in range(i, S[graph].shape[1]):
                    if S[graph][i, j] >= epsilon: #local nearest distance estimation
                        curr_S[i, j] = S[graph][i, j]
                        curr_S[j, i] = S[graph][j, i]
            eps_S.append(curr_S)
        return eps_S

    def compute_S(self, nn = None):
        """Computes the affinity matrix S from the distance matrix D.
       
        The conversion from element [i, j] of distance matrix to affinity is done using a gaussian kernel, whose nn is obtained
        via a the product of distances from the 'nn' nearest neighbours of i and j respectively (see ref [4]).
        
        Parameter:
            nn (int): number of nearest neighbours to use in the kernel. 
       
        Returns:
            S (ndarray): The similarity matrix representing similarities among the input series.
        """
        
        if self.isAffinity:
            return self.D
        
        if nn is None:
            nn = int(np.ceil(0.1*self.D.shape[1])) #value inspired by litterature. See ref [4].
    
        D = self.D
        
        S = []
        for graph in range(len(D)):
            curr_S = np.zeros_like(self.D[graph]) #Affinity matrix
            # pbar = tqdm(desc = "Transforming to affinity matrix", total = self.D.shape[1]*self.D.shape[1]) #progress bar
            neighbors_i = np.sort(D[graph], 1) #computing the nearest neighbors for local nn estimation
            neighbors_j = np.sort(D[graph], 0)
            for i in range(self.D[graph].shape[1]):
                for j in range(i, self.D[graph].shape[1]):
                    d = neighbors_i[i, nn]*neighbors_j[nn, j] #local nearest distance estimation
                    curr_S[i, j] = np.exp(-(D[graph][i, j]**2)/d)
                    curr_S[j, i] = np.exp(-(D[graph][j, i]**2)/d)
                    # pbar.update()
            assert not np.isnan(S).any(), "NaN in affinity matrix, consider setting 'nn' a bit higher."
            S.append(curr_S)
        return S

    def clustering(self, cluster_num, isAffinity, nn, connectivity_param = None):
        """Performs spectral clustering on input series to classify them into meaningfull clusters.

        Parameters:
            cluster_num (int): Number of clusters to look for.
            
            isAffinity (bool): whether or not the values in the graph are representing affinities. If False,
                the function 'compute_S' is called before clustering.
            
            nn (int): Number of nearest neighbours to estimate a local nn for the gaussian kernel (see ref [4]).
                if not provided, defaults to the 10-percentile of the distances, meaning the number of neighbour in the lowest 10% 
                of sorted distances in the D matrix.

        Returns:
            clustered_series (list(list)): Stores the series of each detected cluster, grouped by cluster index. Each list of 'clustered_series' 
                correspond to a cluster and contains all series that are detected to belong to this cluster.

            S (ndarray): the distance matrix.

            idx (list): the cluster indexes outputed by the spectral clustering.

            V (ndarray): the lowest eigenvectors of the Laplacian matrix after diagonalization. This can be used to provide
                a lower dimensionality representation of the input timeseries.

            lambdas (array_like): the eigenvalues of the Laplacian matrix. Plotting them can inform about the number of cluster to
                look for, if similarities among different clusters are fairly low.
        """
        S = self.compute_S(nn)
        if self.connectivity_type == "nearest neighbours" and connectivity_param != None:
            S = self.mutual_nn_affinity(S, nn)
        elif self.connectivity_type == "epsilon neighbourhood" and connectivity_param != None:
            S = self.epsilon_affinity(S, nn)
        
        lambdas, V, idx = scml(S, cluster_num, torch.device("cpu"))

        self.idx = idx
        self.V = V

        return S, idx, V, lambdas

    # def merge_clusters(self, clusters_to_merge):
    #     """Merges two clusters together by putting all their elements under the same index in the 'idx' property.
    #     This operation is done in place.

    #     Parameters:
    #         clusters_to_merge (array_like): an array containing the indexes of the clusters to merge together.
    #     """
    #     new_clusters = deepcopy(self.clustered_series)
    #     merged_cluster = new_clusters[clusters_to_merge[0]]
    #     for i in range(1, len(clusters_to_merge)):
    #         tmp = new_clusters[clusters_to_merge[i]]
    #         for elt in tmp:
    #             merged_cluster.append(elt)

    #         new_clusters[i] = []; #removing corresponding cluster
    #         idx_filter = self.idx == clusters_to_merge[i]
    #         self.idx[idx_filter] = clusters_to_merge[0]; #replacing entries of 'idx' equal to idx2 by idx1.
        
    #     new_clusters = list(filter(lambda c : len(c) != 0, new_clusters))
    #     new_clusters[clusters_to_merge[0]] = merged_cluster; #replacing first cluster from 'clusters_to_merge' by merged cluster.
    #     self.clustered_series = new_clusters
    #     new_idx = np.unique(self.idx)
    #     for i in range(len(new_idx)): #re-ordering the idx in memory so that there are no gaps
    #         idx_filter = self.idx == new_idx[i]
    #         self.idx[idx_filter] = i

    #     print("Merged clusters. To recover previous clusters, run 'clustering' function anew")

    # def split_clusters(self, clust_idx, split_num):
    #     """
    #     This function splits a cluster into several subclusters. This is done in place.
    #     Parameters:
    #         clust_idx (int): the index of the cluster to split. Cluster indexing starts from 0.

    #         split_num (int): how many parts should the cluster be splitted into.
    #     """
    #     assert type(clust_idx) == int, "'clust_idx' expected to be integer."
    #     S_split = np.transpose(self.S[self.idx == clust_idx])[self.idx == clust_idx]
    #     new_idx = clust.spectral_clustering(S_split, n_clusters = split_num)
    #     new_clusters = [self.clustered_series[i] for i in range(clust_idx)] # initializing with all element that have idx < clust_idx
    #     old_cluster = self.clustered_series[clust_idx]
    #     for i in range(split_num): # appending splitted elements
    #         filter_k = new_idx == i
    #         new_clusters.append([old_cluster[k] for k in range(len(filter_k)) if filter_k[k]])
    #     for i in range(clust_idx+1, len(self.clustered_series)): # appending remaining elements from clusters with idx > clust_idx
    #         new_clusters.append(self.clustered_series[i])

    #     self.idx[self.idx > clust_idx] += split_num - 1
    #     self.idx[self.idx == clust_idx] += new_idx + clust_idx - 1#replacing indexes of cluster to split by new indexes.
    #     self.clustered_series = new_clusters
        
    #     print("Splitted cluster %d in %d parts. To recover old clusters, run 'clustering' function anew" %(clust_idx, split_num))

    def dunn_index(self):
        """
            Computes the dunn index that characterizes the quality of a clustering. In the case where isAffinity is True, the lower the better. 
            Otherwise, the higher value of the Dunn index, the better.
        """
        cluster_num = np.max(self.idx)+1
        if self.isAffinity:
            dunn_idx = 0
            for layer in range(len(self.D)):
                intra_cluster_a = np.zeros(cluster_num)
                inter_cluster_a = np.zeros(cluster_num)
                for i in range(cluster_num):
                    num_elts = np.sum(self.idx == i)
                    inside_values = self.D[layer][self.idx == i, :][:, self.idx == i].flatten()
                    try: #in case where cluster contains only on element, this will return an error
                        delta_intra = np.min(inside_values[inside_values != 0])
                    except:
                        delta_intra = np.min(inside_values)
                    if i > 0:
                        delta_inter = np.max(self.D[layer][self.idx == i, :][:, self.idx != i].flatten())
                    else:
                        delta_inter = 0
                    intra_cluster_a[i] = delta_intra
                    inter_cluster_a[i] = delta_inter
                dunn_idx_layer = np.max(inter_cluster_a)/np.min(intra_cluster_a)
                dunn_idx += dunn_idx_layer/len(self.D)
            return dunn_idx
        else:
            dunn_idx = 0
            for layer in range(len(self.D)):
                intra_cluster_d = np.zeros(cluster_num)
                inter_cluster_d = np.zeros(cluster_num)
                for i in range(cluster_num):
                    num_elts = np.sum(self.idx == i)
                    delta_intra = np.max(self.D[layer][self.idx == i, :][:, self.idx == i].flatten())
                    if i > 0:
                        delta_inter = np.min(self.D[layer][self.idx == i, :][:, self.idx != i].flatten())
                    else:
                        delta_inter = np.inf
                    intra_cluster_d[i] = delta_intra
                    inter_cluster_d[i] = delta_inter
                dunn_idx_layer = np.min(inter_cluster_d)/np.max(intra_cluster_d)
                dunn_idx += dunn_idx_layer/len(self.D)
            return dunn_idx
    
    def k_elbow_curve(self, ax, iterations = 20, nn= None, **kwargs):
        """Plots the elbow curve to determine what number of clusters to look for. 

        This curve corresponds to the average intra-cluster distance, which will decrease as a function of the number of cluster.
        However, past the optimal number of cluster there will be no improvement. Therefore the number of clusters to look for
        should be at the elbow point. This can take a while, since it iterates through the data for each possible value of k.

        Parameters:
            iterations (int): up to which value of cluster_num should the algorithm iterate to. defaults to 20.

            nn (int): Number of nearest neighbours to use for the local nn in the gaussian kernel used to compute the similarity matrix.
        
        Optional:
            prograss_bar: if provided, will update a progress bar after each iteration
        """
    
        if iterations >= self.D[0].shape[1]:
            iterations = self.D[0].shape[1]-1
            warn("Too few elements in input timeseries. \n Restricting range to avoid errors.")
        
        mean_intracluster_dist = np.zeros(iterations)
        dunn_indices = np.zeros(iterations)
        for k in tqdm(range(1, iterations)):
            _, idx, _ ,_  = self.clustering(k, False, 5) #No minimal outlier filtering to do not alter data.
            dunn_indices[k] = self.dunn_index()
            for graphs in range(len(self.D)):
                mean_intra = 0
                for i in range(k):
                    for j in range(len(idx)):
                        for l in range(j+1, len(idx)):
                            if idx[j] == i and idx[l] == i:
                                d = self.D[graphs][j, l]
                                mean_intra += d/np.sum(idx == i)
                mean_intracluster_dist[k] += mean_intra/len(self.D)
                if "progress_bar" in kwargs:
                    kwargs["progress_bar"].step((k/iterations)*99.9)
        
        mean_intracluster_dist = (mean_intracluster_dist - np.min(mean_intracluster_dist))/np.max(mean_intracluster_dist)
        dunn_indices = dunn_indices[dunn_indices != np.inf]
        dunn_indices = (dunn_indices - np.min(dunn_indices))/np.max(dunn_indices)
        if self.isAffinity:
            k_curve = ax.plot(np.arange(iterations)+1, mean_intracluster_dist, label = "Intracluster affinity (higher better)")
            ax.plot(np.arange(len(dunn_indices))+1, dunn_indices, label = "Dunn index (lower better)")
        else:
            k_curve = ax.plot(np.arange(iterations)+1, mean_intracluster_dist, label = "Intracluster distance (lower better)")
            ax.plot(np.arange(len(dunn_indices))+1, dunn_indices, label = "Dunn index (higher better)")
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel("Normalized values")
        ax.legend()
        return k_curve
                
    def nn_grid_search(self, ax, num_pts = 15, clust_num = 10, connectivity_param = None):
        """Does a grid search, looking for an optimal value of the number of nearest neighbours
        for the local nn to be used in the clustering (see ref [4]). 
        
        Similar to the k-elbow curve,
        the optimal value of 'nn' should be the one which minimizes the intracluster distance. In practice,
        htis curve can fluctuate a lot, and taking the default value for nn is usually sufficient.
        The search goes between 5 and the 15%-tile of the distances of each element in the distance matrix.

        Parameters:
            num_pts (int): How many points to use for the grid search.

            clust_num (int): How many cluster should be looked for. defaults to 10.
        """

        if clust_num >= self.D[0].shape[1]:
            clust_num = self.D[0].shape[1]
            warn("Too few elements in input graph. \n Restricting range to avoid errors.")
        
        mean_intracluster_dist = np.zeros(num_pts)
        nn_range = np.ceil(np.linspace(2, 3*np.ceil(0.05*self.D[0].shape[1]), num_pts)).astype(int)
        for p, nn in enumerate(tqdm(nn_range)):
            _, idx, _, _ = self.clustering(clust_num, self.isAffinity, nn, connectivity_param) #no percentile filter as low nn values break distance matrix.
            for graphs in range(len(self.D)):
                mean = 0
                for i in range(clust_num):
                    for j in range(len(idx)):
                        for l in range(j+1, len(idx)):
                            if idx[j] == i and idx[l] != i:
                                d = self.D[graphs][j, l]
                                mean += d
                mean_intracluster_dist[p] += mean/(clust_num*len(self.D))

        k_curve = ax.plot(nn_range, mean_intracluster_dist)
        ax.set_xlabel('nn value')
        if self.isAffinity:
            ax.set_ylabel('Mean intercluster affinity')
        else:
            ax.set_ylabel('Mean intercluster distance')
        ax.axvline(np.ceil(0.05*self.D[0].shape[1]), 0, 1, ls=":", label = "Default value")
        return 
    
    def mnn_grid_search(self, ax, num_pts = 15, clust_num = 10, nn = 7):
        """Does a grid search, looking for an optimal value of the number of nearest neighbours
        for the local nn to be used in the clustering (see ref [4]). 
        
        Similar to the k-elbow curve,
        the optimal value of 'nn' should be the one which minimizes the intracluster distance. In practice,
        htis curve can fluctuate a lot, and taking the default value for nn is usually sufficient.
        The search goes between 5 and the 15%-tile of the distances of each element in the distance matrix.

        Parameters:
            num_pts (int): How many points to use for the grid search.

            clust_num (int): How many cluster should be looked for. defaults to 10.
        """

        if clust_num >= self.D[0].shape[1]:
            clust_num = self.D[0].shape[1]
            warn("Too few elements in input graph. \n Restricting range to avoid errors.")
        
        mean_intracluster_dist = np.zeros(num_pts)
        mnn_range = np.ceil(np.linspace(2, 4*np.ceil(0.1*self.D[0].shape[1]), num_pts)).astype(int)
        for p, mnn in enumerate(tqdm(mnn_range)):
            _, idx, _, _ = self.clustering(clust_num, False, nn, mnn) #no percentile filter as low nn values break distance matrix.
            for graphs in range(len(self.D)):
                mean = 0
                for i in range(clust_num):
                    for j in range(len(idx)):
                        for l in range(j+1, len(idx)):
                            if idx[j] == i and idx[l] != i:
                                d = self.D[graphs][j, l]
                                mean += d
                mean_intracluster_dist[p] += mean/(clust_num*len(self.D))

        k_curve = ax.plot(mnn_range, mean_intracluster_dist)
        ax.set_xlabel('Nearest neighbour number')
        if self.isAffinity:
            ax.set_ylabel('Mean intercluster affinity')
        else:
            ax.set_ylabel('Mean intercluster distance')
        return 
    
    def epsilon_grid_search(self, ax, num_pts = 10, clust_num = 10, nn = 7):
        """Does a grid search, looking for an optimal value of the number of nearest neighbours
        for the local nn to be used in the clustering (see ref [4]). 
        
        Similar to the k-elbow curve,
        the optimal value of 'nn' should be the one which minimizes the intracluster distance. In practice,
        htis curve can fluctuate a lot, and taking the default value for nn is usually sufficient.
        The search goes between 5 and the 15%-tile of the distances of each element in the distance matrix.

        Parameters:
            num_pts (int): How many points to use for the grid search.

            clust_num (int): How many cluster should be looked for. defaults to 10.
        """

        if clust_num >= self.D[0].shape[1]:
            clust_num = self.D[0].shape[1]
            warn("Too few elements in input graph. \n Restricting range to avoid errors.")
        
        all_val = [] #find
        for i in range(len(self.D)):
            all_val.extend(self.D[i].flatten())
        max_val = np.max(all_val)

        mean_intracluster_dist = np.zeros(num_pts)
        epsilon_range = np.linspace(0, max_val, num_pts)
        for p, epsilon in enumerate(tqdm(epsilon_range)):
            _, idx, _, _ = self.clustering(clust_num, self.isAffinity, nn, epsilon) #no percentile filter as low nn values break distance matrix.
            for graphs in range(len(self.D)):
                mean = 0
                for i in range(clust_num):
                    for j in range(len(idx)):
                        for l in range(j+1, len(idx)):
                            if idx[j] == i and idx[l] != i:
                                d = self.D[graphs][j, l]
                                mean += d
                mean_intracluster_dist[p] += mean/(clust_num*len(self.D))

        k_curve = ax.plot(epsilon_range, mean_intracluster_dist)
        ax.set_xlabel('Epsilon value')
        if self.isAffinity:
            ax.set_ylabel('Mean intercluster affinity')
        else:
            ax.set_ylabel('Mean intercluster distance')
        return 

    def plot_eigenvalues(self, eigvalue_num, percentile_filter = 5, min_elts = 2, nn = None):
        """Helper function to visualize the shape of the detected clusters. 
        
        In theory, the number of eigenvalues close to 0 should indicate the number 
        of clustes to use, and so the point at which a large gap to the next eigenvalue indicates the number of clusters to look for.
        This explanation is however based on perturbation theory (see ref [2]) and will only work well in cases where 
        the clusters are "well behaved" i.e. when there are not too many outliers, the densities are uniform and the clusters well separated.

        Parameters:
            eigvalue_num (int): how many eigenvalues to display. 

            percentile_filter (int): whether or not to remove outlier elements that have a distance much larger than other elements. 
                'outlier_filter' specifies a percentile of the highest distances to exclude. 
                For instance, if 'outlier_filter' is 2, the highest 2% of distances will be remove from analysis.

            min_elts (int): the minimum number of elements allowed in a cluster. If a cluster contains less elements than 'min_elts', 
                these elements are removed from the input, and the analysis starts again.

            nn (int): the number of nearest neighbours to use in the kernel for computing the affinity matrix.
        """
        _,_,_,lambdas = self.clustering(eigvalue_num, nn)
        ev_plot = plt.plot(lambdas, lw = 1, c = 'black', marker = "^", markersize = 10)
        plt.title("Spectral gaps graph")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.show()
        return ev_plot





