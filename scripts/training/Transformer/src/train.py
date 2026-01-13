import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import math

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import knn_graph
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import math


def contrastive_loss(embeddings, group_ids, temperature=0.1):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining, using group_ids only.

    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[j]),
          where j is a randomly chosen index (≠ i) such that group_ids[j] == group_ids[i].
      - Random negative similarity: rand_neg_sim = sim(embeddings[i], embeddings[k]),
          where k is a randomly chosen index such that group_ids[k] != group_ids[i].
                                       group_ids[k] != group_ids[i] }
    The loss per anchor is:
         loss_i = - log( exp(pos_sim_orig / temperature) / 
                         ( exp(pos_sim_orig / temperature) + exp(blended_neg / temperature) ) )

    Anchors that lack any valid positives or negatives contribute 0.

    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.

    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()  # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    
    # --- Positives ---
    # same group (excluding self)
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask.fill_diagonal_(False)  
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)

    # Select a random positive candidate
    rand_vals_pos = torch.rand_like(sim_matrix)
    rand_vals_pos = rand_vals_pos * pos_mask.float() - (1 - pos_mask.float())
    rand_pos_indices = torch.argmax(rand_vals_pos, dim=1)
    rand_pos_sim = sim_matrix[idx, rand_pos_indices]
    pos_sim_orig = torch.where(no_valid_pos, sim_matrix[idx, idx], rand_pos_sim)

    # --- Negatives ---
    # Build a mask for negatives
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    valid_neg_counts = neg_mask.sum(dim=1)
    no_valid_neg = (valid_neg_counts == 0)

    # Random negative similarity
    rand_vals_neg = torch.rand_like(sim_matrix)
    rand_vals_neg = rand_vals_neg * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals_neg, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]

    # Loss
    numerator = torch.exp(pos_sim_orig / temperature)
    denominator = numerator + torch.exp(rand_neg_sim / temperature)
    loss = -torch.log(numerator / denominator)
    loss = loss.masked_fill(no_valid_neg, 0.0)

    return loss.mean()

#################################
# Training and Testing Functions
#################################


def train_new(train_loader, model, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Make assoc_tensor (best: have dataset return it as a tensor already)
        assoc_tensor = data.assoc

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            # GPU counts (no cpu/numpy)
            counts = torch.bincount(data.x_batch)
            counts = counts[counts > 0]

            splits = torch.split(embeddings, counts.tolist())
            group_splits = torch.split(assoc_tensor, counts.tolist())

            loss = torch.stack([
                contrastive_loss(e, g, temperature=0.1)
                for e, g in zip(splits, group_splits)
            ]).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())

    return total_loss / len(train_loader)

@torch.no_grad()
def test_new(test_loader, model, device):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)

        # Make assoc_tensor a Long tensor on GPU
        assoc_tensor = data.assoc
        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            # Count nodes per event on GPU (no cpu/numpy)
            counts = torch.bincount(data.x_batch)
            counts = counts[counts > 0]

            # Split per-event
            splits = torch.split(embeddings, counts.tolist())
            group_splits = torch.split(assoc_tensor, counts.tolist())

            # Per-event contrastive loss, averaged over events
            loss = torch.stack([
                contrastive_loss(e, g, temperature=0.1)
                for e, g in zip(splits, group_splits)
            ]).mean()

        total_loss += float(loss.detach())

    return total_loss / len(test_loader)



