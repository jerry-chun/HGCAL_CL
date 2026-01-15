import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from tqdm import tqdm

def Agglomerative(all_predictions, DataLoader, threshold = 0.7, metric = 'cosine', linkage = 'average'):
    all_cluster_labels = []             

    for i, (pred, data) in enumerate(zip(all_predictions, loader)):

        xyz = data.x[:, :3]

        k = 20
        connectivity = kneighbors_graph(
            xyz,
            n_neighbors=k,
            mode="connectivity",
            include_self=True,
            n_jobs=-1,
        )

        agglomerative = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=24,
            linkage="ward",         
            metric="euclidean",
            connectivity=connectivity,
            compute_distances=True,
        )

        cluster_labels = agglomerative.fit_predict(pred)
        all_cluster_labels.append(cluster_labels)   
    return all_cluster_labels
