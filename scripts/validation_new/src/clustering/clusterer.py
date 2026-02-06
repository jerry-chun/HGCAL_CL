# src/clustering/clusterer.py

import time
import torch

from .oc_clustering import oc_cluster_single_event
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

import numpy as np
import torch.nn.functional as F

import torch

import torch

@torch.no_grad()
def oc_like_cluster_from_embeddings(
    emb: torch.Tensor,                         
    *,
    k_density: int = 16,                       # kNN for density
    tau: float = 0.2,                          # beta calc
    beta_thr: float = 0.1,                     # from paper
    td: float = 0.8,                           # from paper
    assign_remaining_to_nearest: bool = True,
) -> torch.Tensor:
    """
    OC-like clustering from embeddings only.

    1) Build a density proxy using kNN distance:
         d_k(i) = distance to k-th nearest neighbour of point i
         beta_i = exp(-d_k(i)/tau)
    2) Candidate centers: beta > beta_thr, sorted by beta desc
    3) keep a center if it's >= td from all kept centers
    4) Assign: in center order, claim all unassigned points within td
    5) assign any leftovers to nearest center
    """
    device = emb.device
    N = emb.shape[0]
    cluster_ids = torch.full((N,), -1, dtype=torch.long, device=device)

    # density proxy via full pairwise distances (O(N^2))
    d = torch.cdist(emb, emb) 

    k = min(k_density + 1, N)
    vals, _ = torch.topk(d, k=k, dim=1, largest=False) 
    dk = vals[:, -1]  

    beta = torch.exp(-dk / tau)  

    # candidate centers
    cand_idx = torch.nonzero(beta > beta_thr, as_tuple=False).view(-1)
    if cand_idx.numel() == 0:
        centers = torch.argmax(beta).view(1)
    else:
        cand_idx = cand_idx[torch.argsort(beta[cand_idx], descending=True)]

        # greedy NMS for well-separated centers
        centers_list = []
        centers_x = None 

        for idx in cand_idx:
            x = emb[idx].view(1, -1)
            if centers_x is None:
                centers_list.append(idx)
                centers_x = x
                continue

            if torch.all(torch.cdist(x, centers_x).squeeze(0) >= td):
                centers_list.append(idx)
                centers_x = torch.cat([centers_x, x], dim=0)

        centers = torch.stack(centers_list) if centers_list else torch.argmax(beta).view(1)
        centers_x = emb[centers]

    K = centers_x.shape[0]
    if K == 0:
        return cluster_ids

    # sort centers by beta desc 
    order = torch.argsort(beta[centers], descending=True)
    centers = centers[order]
    centers_x = centers_x[order]

    # 4) radius assignment
    unassigned = cluster_ids == -1
    for k_i in range(K):
        if not unassigned.any():
            break

        idx_u = torch.nonzero(unassigned, as_tuple=False).view(-1)
        d_u = torch.norm(emb[idx_u] - centers_x[k_i], dim=-1)
        in_ball = d_u <= td

        if in_ball.any():
            cluster_ids[idx_u[in_ball]] = k_i
            unassigned = cluster_ids == -1

    # nearest-center completion
    if assign_remaining_to_nearest and (cluster_ids == -1).any():
        idx_u = torch.nonzero(cluster_ids == -1, as_tuple=False).view(-1)
        d_all = torch.cdist(emb[idx_u], centers_x)  # (U, K)
        cluster_ids[idx_u] = torch.argmin(d_all, dim=1)

    return cluster_ids



def _cluster_contrastive(config, model, data_loader, device):
    reconstruction_labels = []
    start_time = time.time()
    dt = config["distance_threshold"]
    met = config["metric"]
    link = config["linkage"]

    for i, data in enumerate(data_loader):
        data = data.to(device)

        out = model(data.x, data.x_batch)
        preds = out[0]
        #preds = F.normalize(preds, p=2, dim=1)

        
        cluster_labels = oc_like_cluster_from_embeddings(
            preds,
            k_density=36,     
            tau=0.2,         

            beta_thr=0.3,    
            td=1.1,          

            assign_remaining_to_nearest=True,
        ).numpy()
        #cluster_labels = clusterer.fit_predict(preds_np)  # noise = -1
        
        reconstruction_labels.append(cluster_labels)

        if i > config["max_events"] - 1:
            break

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
    oc_td = config["oc_td"]

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
                    beta_thr= beta_thr,   
                    td = oc_td,  
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
