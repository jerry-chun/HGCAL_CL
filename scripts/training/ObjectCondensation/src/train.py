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
    noise_label: int = -1,
    q_min: float = 0.1,
    s_att: float = 1.0,
    s_rep: float = 1.0,
    s_coward: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    device = beta.device
    beta = beta.view(-1)
    N = beta.numel()

    assert cluster_coords.shape[0] == N
    assert group_ids.shape[0] == N

    eps = 1e-1
    beta = beta.clamp(min=eps, max=1.0 - eps)

    weights = torch.ones_like(beta)

    not_noise = group_ids > noise_label

    if not not_noise.any():
        return s_noise * beta.mean()

    unique_obj_ids = torch.unique(group_ids[not_noise])
    if unique_obj_ids.numel() == 0:
        l_noise = beta[~not_noise].mean() if (~not_noise).any() else torch.tensor(0.0, device=device)
        return s_noise * l_noise

    attractive_mask = group_ids[:, None] == unique_obj_ids[None, :]

    q = torch.arctanh(beta).pow(2) + q_min

    alpha_idx = torch.argmax(q[:, None] * attractive_mask, dim=0)
    x_k = cluster_coords[alpha_idx]
    q_k = q[alpha_idx][None, :]

    dist = torch.cdist(cluster_coords, x_k)

    v_att_jk = weights[:, None] * q[:, None] * q_k * attractive_mask * dist.pow(2)
    v_att = torch.mean(
        torch.sum(v_att_jk, dim=0) /
        (torch.sum(attractive_mask, dim=0) + eps)
    )

    rep_mask = ~attractive_mask
    v_rep_jk = weights[:, None] * q[:, None] * q_k * rep_mask * F.relu(1.0 - dist)
    v_rep = torch.mean(
        torch.sum(v_rep_jk, dim=0) /
        (torch.sum(rep_mask, dim=0) + eps)
    )

    l_coward = torch.mean(1.0 - beta[alpha_idx])

    l_noise = torch.mean(beta[~not_noise]) if (~not_noise).any() else torch.tensor(0.0, device=device)

    return (
        s_att * v_att +
        s_rep * v_rep +
        s_coward * l_coward +
        s_noise * l_noise
    )


def train_oc(train_loader, model, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        assoc_tensor = data.assoc  
        with autocast():
            beta, cluster_coords, _ = model(data.x, data.x_batch)

            counts = torch.bincount(data.x_batch)
            counts = counts[counts > 0]

            beta_splits = torch.split(beta, counts.tolist())                    
            coord_splits = torch.split(cluster_coords, counts.tolist())         
            group_splits = torch.split(assoc_tensor, counts.tolist())           

            loss = torch.stack([
                object_condensation_loss(
                    beta=b,
                    cluster_coords=c,
                    group_ids=g,
                    noise_label=-1,   
                    q_min=0.1,
                    s_att=1.0,
                    s_rep=1.0,
                    s_coward=1.0,
                    s_noise=1.0,
                )
                for b, c, g in zip(beta_splits, coord_splits, group_splits)
            ]).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())

    return total_loss / len(train_loader)


@torch.no_grad()
def test_oc(test_loader, model, device):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)

        assoc_tensor = data.assoc

        with autocast():
            beta, cluster_coords, _ = model(data.x, data.x_batch)

            counts = torch.bincount(data.x_batch)
            counts = counts[counts > 0]

            beta_splits = torch.split(beta, counts.tolist())
            coord_splits = torch.split(cluster_coords, counts.tolist())
            group_splits = torch.split(assoc_tensor, counts.tolist())

            loss = torch.stack([
                object_condensation_loss(
                    beta=b,
                    cluster_coords=c,
                    group_ids=g,
                    noise_label=-1,
                    q_min=0.1,
                    s_att=1.0,
                    s_rep=1.0,
                    s_coward=1.0,
                    s_noise=1.0,
                )
                for b, c, g in zip(beta_splits, coord_splits, group_splits)
            ]).mean()

        total_loss += float(loss.detach())

    return total_loss / len(test_loader)

