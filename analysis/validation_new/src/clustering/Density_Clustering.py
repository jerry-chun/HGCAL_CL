# src/clustering/Density_clusterer.py

import time
import torch
from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def Density_Clustering(
    emb: torch.Tensor,                         
    *,
    k_density: int = 24,                       # kNN for density
    tau: float = 0.1,                          # beta calc
    beta_thr: float = 0.1,                     
    td: float = 0.8,                           
    assign_remaining_to_nearest: bool = True,
) -> torch.Tensor:
    """
    Density_Clustering from embeddings only.

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

    # density proxy via full pairwise distances 
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

