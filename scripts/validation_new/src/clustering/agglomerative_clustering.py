import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from tqdm import tqdm

def Agglomerative(all_predictions,  threshold = 0.7, metric = 'cosine', linkage = 'average'):
    all_cluster_labels = []             

    for i, pred in enumerate(all_predictions):


        agglomerative = AgglomerativeClustering(
            n_clusters=None,                 
            distance_threshold=threshold,
            linkage=linkage,
            metric=metric,
            compute_distances=True
        )
        cluster_labels = agglomerative.fit_predict(pred) 

        all_cluster_labels.append(cluster_labels)

    #all_cluster_labels = np.array(all_cluster_labels)
    return all_cluster_labels
