import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv
from tqdm import tqdm
import numpy as np
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def object_condensation_loss(
    beta: torch.Tensor,
    cluster_coords: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    q_min: float = 0.1,
    eps: float = 1e-4,  
    s_att: float = 1.0,
    s_rep: float = 1.0,
    s_coward: float = 1.0,
) -> torch.Tensor:
    """
    OC loss for a single event, adapted to our dataset:
      - every hit belongs to a truth shower (no noise)
      - object-balanced attractive term
      - full repulsion 
      - coward term only (no noise penalty)
    """
    device = beta.device
    beta = beta.view(-1)
    N = beta.numel()

    assert cluster_coords.shape[0] == N
    assert group_ids.shape[0] == N

    # numeric safety 
    beta = beta.clamp(min=eps, max=1.0 - eps)

    unique_obj_ids = torch.unique(group_ids)
    K = unique_obj_ids.numel()
    if K == 0:
        return torch.zeros((), device=device)

    # (N, K) membership mask
    attractive_mask = (group_ids[:, None] == unique_obj_ids[None, :])

    # charge
    q = torch.arctanh(beta).pow(2) + q_min  

    # pick condensation point per object 
    alpha_idx = torch.argmax(q[:, None] * attractive_mask, dim=0)  

    # coords and charges of condensation points
    x_k = cluster_coords[alpha_idx]        
    q_k = q[alpha_idx][None, :]           
    # distances: 
    dist = torch.cdist(cluster_coords, x_k)

    # Attractive (object-balanced)
    v_att_jk = q[:, None] * q_k * attractive_mask * dist.pow(2)  

    # average per object then across objects 
    obj_counts = attractive_mask.sum(dim=0).clamp_min(1.0)
    v_att = torch.mean(torch.sum(v_att_jk, dim=0) / obj_counts)

    # Repulsive (hinge with radius 1.0 in latent space)
    rep_mask = ~attractive_mask
    rep_counts = rep_mask.sum(dim=0).clamp_min(1.0)

    v_rep_jk = q[:, None] * q_k * rep_mask * F.relu(1.0 - dist)   
    v_rep = torch.mean(torch.sum(v_rep_jk, dim=0) / rep_counts)

    # Coward: condensation points should have beta -> 1
    l_coward = torch.mean(1.0 - beta[alpha_idx])
    return s_att * v_att + s_rep * v_rep + s_coward * l_coward


# Train / Val loops 
def train_oc(train_loader, model, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        assoc_tensor = data.assoc  
        batch = data.x_batch      
        with autocast():
            beta, cluster_coords, _ = model(data.x, batch)

            counts = torch.bincount(batch)
            counts = counts[counts > 0]

            beta_splits = torch.split(beta, counts.tolist())
            coord_splits = torch.split(cluster_coords, counts.tolist())
            group_splits = torch.split(assoc_tensor, counts.tolist())

            loss = torch.stack([
                object_condensation_loss(
                    beta=b.float(),
                    cluster_coords=c.float(),
                    group_ids=g,
                    q_min=0.1,
                    eps=1e-3,
                    s_att=50.0,
                    s_rep=50.0,
                    s_coward=1,
                )
                for b, c, g in zip(beta_splits, coord_splits, group_splits)
            ]).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def test_oc(test_loader, model, device):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)

        assoc_tensor = data.assoc
        batch = data.x_batch

        with autocast():
            beta, cluster_coords, _ = model(data.x, batch)

            counts = torch.bincount(batch)
            counts = counts[counts > 0]

            beta_splits = torch.split(beta, counts.tolist())
            coord_splits = torch.split(cluster_coords, counts.tolist())
            group_splits = torch.split(assoc_tensor, counts.tolist())

            loss = torch.stack([
                object_condensation_loss(
                    beta=b.float(),
                    cluster_coords=c.float(),
                    group_ids=g,
                    q_min=0.1,
                    eps=1e-3,
                    s_att=50,
                    s_rep=50.0,
                    s_coward=0.5,
                )
                for b, c, g in zip(beta_splits, coord_splits, group_splits)
            ]).mean()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(test_loader))

