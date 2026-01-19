# src/clustering/clusterer.py

import time
import torch

from .oc_clustering import oc_cluster_single_event
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def _cluster_contrastive(config, model, data_loader, device):
    reconstruction_labels = []
    start_time = time.time()

    for i, data in enumerate(data_loader):
        data = data.to(device)

        out = model(data.x, data.x_batch)
        preds = out[0]
        #xyz = data.x[:, :3].detach().cpu().numpy()
        """
        k = 8
        connectivity = kneighbors_graph(
            xyz,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
            n_jobs=-1,
        )
        """
        agglomerative = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=37,
            linkage="ward",         
            metric="euclidean",
            # connectivity=connectivity,
            # compute_distances=True,
        )

        preds_np = preds.detach().cpu().numpy()
        cluster_labels = agglomerative.fit_predict(preds_np)

        reconstruction_labels.append(cluster_labels)






    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / len(reconstruction_labels)
    print(
        f"[contrastive] Clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return reconstruction_labels


def _cluster_oc(config, model, data_loader, device):
    all_reco_ids = []
    start_time = time.time()

    beta_thr = config["oc_beta_thr"]
    min_sep = config["oc_min_center_separation"]
    use_cut = config["oc_use_distance_cut"]
    assign_r = config["oc_assignment_radius"]

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)

            beta, cluster_coords, batch = model(data.x, data.x_batch)

            num_events = int(batch.max().item() + 1)

            for evt in range(num_events):
                evt_mask = (batch == evt)

                coords_evt = cluster_coords[evt_mask]
                beta_evt = beta[evt_mask]
                
                print('beta:', beta_evt)
                

                cluster_ids_evt = oc_cluster_single_event(
                    coords_evt,
                    beta_evt,
                    beta_thr= 0.1,   
                    td = 0.3,  
                )

                all_reco_ids.append(cluster_ids_evt.cpu().numpy())

            if i > config["max_events"] - 1:
                break

    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / max(1, len(all_reco_ids))
    print(
        f"[OC] Inference + OC clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return all_reco_ids


def clusterer(config, model, data_loader, device):
    task = config.get("task", "contrastive")

    if task == "contrastive":
        return _cluster_contrastive(config, model, data_loader, device)
    elif task == "oc":
        return _cluster_oc(config, model, data_loader, device)
    else:
        raise ValueError(f"Unknown task '{task}' in clusterer.")
