import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

def _split_by_batch(x: torch.Tensor, batch: torch.Tensor):
    """
    Split x into a tuple of tensors per event
    Uses bincount on GPU (fast) and torch.split.
    """
    counts = torch.bincount(batch)
    counts = counts[counts > 0]
    return torch.split(x, counts.tolist())

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
    return loss

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
                supcon_loss_node_equal(e, g, temperature=temperature)
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
                supcon_loss_node_equal(e, g, temperature=temperature)
                for e, g in zip(emb_events, gid_events)
            ]).mean()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(test_loader))
