# src/clustering/clusterer.py

import time
import torch
from .oc_clustering import oc_cluster_single_event
from .Density_Clustering import Density_Clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

import numpy as np
import torch.nn.functional as F



def _cluster_agglomerative(config, model, data_loader, device):
    reconstruction_labels = []
    start_time = time.time()
    dt = config["distance_threshold"]
    met = config["metric"]
    link = config["linkage"]
    task = config["task"]

    for i, data in enumerate(data_loader):
        print(i)
        data = data.to(device)

        out = model(data.x, data.x_batch)
        
        preds = out[0] if task == "contrastive" else out[1]
        # raw embeddings for agglomerative — paper uses delta_agg=9.5 on unnormalized space
        if task != "contrastive":
            preds = F.normalize(preds, p=2, dim=1)

        agglomerative = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dt,
            linkage=link,
            metric=met,
        )

        preds_np = preds.detach().cpu().numpy()
        cluster_labels = agglomerative.fit_predict(preds_np)
        
        reconstruction_labels.append(cluster_labels)

        if i > config["max_events"] - 1:
            break

    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / len(reconstruction_labels)
    print(
        f"Agglomerative Clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return reconstruction_labels


def _cluster_density(config, model, data_loader, device):
    all_reco_ids = []
    start_time = time.time()

    beta_thr = config["oc_beta_thr"]
    oc_td = config["oc_td"]
    task = config["task"]

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print(i)
            data = data.to(device)

            if task == "oc":
                beta, cluster_coords, batch = model(data.x, data.x_batch)
                
                cluster_labels = oc_cluster_single_event(
                    cluster_coords,
                    beta,
                    beta_thr= beta_thr,   
                    td = oc_td,  
                )
            else:
                out = model(data.x, data.x_batch)
                preds = F.normalize(out[0], p=2, dim=1)
                
                cluster_labels = Density_Clustering(
                    preds,
                    beta_thr=beta_thr,
                    td=oc_td,
                )


            all_reco_ids.append(cluster_labels.cpu().numpy())
            if i > config["max_events"] - 1:
                break

    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / max(1, len(all_reco_ids))
    print(
        f"[OC] Inference + Density clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return all_reco_ids


def clusterer(config, model, data_loader, device):
    clustering = config.get("cluster")

    if clustering == "agglomerative":
        return _cluster_agglomerative(config, model, data_loader, device)
    elif clustering == "density":
        return _cluster_density(config, model, data_loader, device)
    else:
        raise ValueError(f"Unknown task '{task}' in clusterer.")
