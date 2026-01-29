import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm


def _group_mean(values: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    sums = values.new_zeros((num_groups,))
    cnts = values.new_zeros((num_groups,))
    sums.index_add_(0, group_ids, values)
    cnts.index_add_(0, group_ids, torch.ones_like(values))
    return sums / cnts.clamp_min(1.0)


def supcon_loss_shower_equal(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Full supervised contrastive loss within ONE event (all positives, all negatives),
    but with SHOWER-EQUAL weighting:

      1) Compute per-anchor SupCon loss L_i (mean over positives for that anchor)
      2) Average L_i within each shower (group)
      3) Average over showers that have at least one valid anchor (>=2 hits)

    This ensures each shower contributes equally, not proportional to number of hits.


    """
    device = embeddings.device
    N = embeddings.size(0)
    if N <= 1:
        return embeddings.new_tensor(0.0)

    # Remap group ids to [0..G-1] for cheap bincount/scatter
    _, labels = torch.unique(group_ids, sorted=False, return_inverse=True)
    G = int(labels.max().item()) + 1

    # Compute logits in fp32 for stability (important under autocast)
    z = F.normalize(embeddings.float(), p=2, dim=1)          
    logits = (z @ z.t()) / float(temperature)                

    # Masks
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~not_self, float("-inf"))     # exclude self from denom

    pos_mask = (labels[:, None] == labels[None, :]) & not_self  
    valid_anchor = pos_mask.any(dim=1)                          

    if not bool(valid_anchor.any()):
        return embeddings.new_tensor(0.0)

    # log denom: log sum_{a != i} exp(logits_{i,a})
    log_denom = torch.logsumexp(logits, dim=1)          

    # log-prob matrix
    log_prob = logits - log_denom[:, None]          

    # mean log-prob over positives per anchor
    pos_counts = pos_mask.sum(dim=1).clamp_min(1)           
    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_counts)

    loss_i = -mean_log_prob_pos                             

    # Shower-equal reduction
    loss_valid = loss_i[valid_anchor]                          
    labels_valid = labels[valid_anchor]                        

    # Per-shower mean of anchor losses
    per_shower = _group_mean(loss_valid, labels_valid, G)      

    shower_has_valid = torch.zeros((G,), dtype=torch.bool, device=device)
    shower_has_valid.index_fill_(0, torch.unique(labels_valid), True)

    loss = per_shower[shower_has_valid].mean()

    # Return in original dtype
    return loss.to(embeddings.dtype)


def _split_by_batch(x: torch.Tensor, batch: torch.Tensor):
    """
    Split x into a tuple of tensors per event
    Uses bincount on GPU (fast) and torch.split.
    """
    counts = torch.bincount(batch)
    counts = counts[counts > 0]
    return torch.split(x, counts.tolist())

import torch
import torch.nn.functional as F


def supcon_loss_node_equal(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Full supervised contrastive loss (SupCon) within ONE event,
    with NODE-EQUAL weighting (each anchor contributes equally).

    embeddings: (N, D)
    group_ids:  (N,) shower id per hit
    temperature: tau

    Returns scalar loss (0 if no anchor has positives).
    """
    device = embeddings.device
    N = embeddings.size(0)
    if N <= 1:
        return embeddings.new_tensor(0.0)

    # Normalize + cosine similarity (fp32 for stability)
    z = F.normalize(embeddings.float(), p=2, dim=1)
    logits = (z @ z.t()) / float(temperature)

    # Masks
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~not_self, float("-inf"))

    labels = group_ids
    pos_mask = (labels[:, None] == labels[None, :]) & not_self
    valid_anchor = pos_mask.any(dim=1)

    if not bool(valid_anchor.any()):
        return embeddings.new_tensor(0.0)

    # Denominator
    log_denom = torch.logsumexp(logits, dim=1)

    # Log-probabilities
    log_prob = logits - log_denom[:, None]

    # Mean over positives per anchor
    pos_counts = pos_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (
        log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_counts
    )

    loss_i = -mean_log_prob_pos

    # NODE-EQUAL reduction
    loss = loss_i[valid_anchor].mean()
    return loss.to(embeddings.dtype)

def sampled_contrastive_loss(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.25,
    *,
    k_pos: int = 16,
    k_neg: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Partially supervised contrastive loss within ONE event.

    For each anchor:
      - sample up to k_pos positives from same group
      - sample up to k_neg negatives from other groups
      - denominator includes only sampled positives + sampled negatives

    NODE-EQUAL reduction.

    embeddings: (N, D)
    group_ids:  (N,)
    temperature: tau
    """
    device = embeddings.device
    N = embeddings.size(0)
    if N <= 1:
        return embeddings.new_tensor(0.0)

    z = F.normalize(embeddings.float(), p=2, dim=1)
    logits = (z @ z.t()) / float(temperature)

    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~not_self, float("-inf"))

    labels = group_ids
    loss_vals = []

    for i in range(N):
        # positives
        pos_idx = torch.nonzero(
            (labels == labels[i]) & (torch.arange(N, device=device) != i),
            as_tuple=False,
        ).flatten()

        if pos_idx.numel() == 0:
            continue

        if pos_idx.numel() > k_pos:
            perm = torch.randperm(pos_idx.numel(), device=device, generator=generator)
            pos_idx = pos_idx[perm[:k_pos]]

        # negatives
        neg_idx = torch.nonzero(labels != labels[i], as_tuple=False).flatten()
        if neg_idx.numel() == 0:
            continue

        if neg_idx.numel() > k_neg:
            perm = torch.randperm(neg_idx.numel(), device=device, generator=generator)
            neg_idx = neg_idx[perm[:k_neg]]

        denom_idx = torch.cat([pos_idx, neg_idx], dim=0)

        # compute loss for anchor i
        log_denom = torch.logsumexp(logits[i, denom_idx], dim=0)
        mean_pos = logits[i, pos_idx].mean()
        loss_vals.append(-mean_pos + log_denom)

    if len(loss_vals) == 0:
        return embeddings.new_tensor(0.0)

    return torch.stack(loss_vals).mean().to(embeddings.dtype)


def train_new(
    train_loader,
    model,
    optimizer,
    device,
    scaler,
    *,
    temperature: float = 0.1,
):
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        assoc = data.assoc  

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            emb_events = _split_by_batch(embeddings, data.x_batch)
            gid_events = _split_by_batch(assoc, data.x_batch)

            # Mean over events
            loss = torch.stack([
                sampled_contrastive_loss(e, g, temperature=temperature)
                for e, g in zip(emb_events, gid_events)
            ]).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def test_new(
    test_loader,
    model,
    device,
    *,
    temperature: float = 0.1,
):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)
        assoc = data.assoc

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            emb_events = _split_by_batch(embeddings, data.x_batch)
            gid_events = _split_by_batch(assoc, data.x_batch)

            loss = torch.stack([
                sampled_contrastive_loss(e, g, temperature=temperature)
                for e, g in zip(emb_events, gid_events)
            ]).mean()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(test_loader))
