import torch
import torch.nn.functional as F


def contrastive_multi_neg_per_group(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
    num_pos: int = 5,
    num_neg: int = 5,
) -> torch.Tensor:
    """
    A simple per-event contrastive loss:
    - pulls hits from the same particle (group) together
    - pushes hits from other particles apart

    For each hit (anchor):
      - randomly sample positives from the same group
      - randomly sample negatives from other groups
      - contrastive loss: positives vs negatives

    Works for events with multiple showers, even if group sizes differ.
    """
    device = embeddings.device
    N = embeddings.shape[0]

    # Normalize for cosine similarity
    z = F.normalize(embeddings, p=2, dim=1)

    unique_groups, group_idx_per_node = group_ids.unique(return_inverse=True)
    G = unique_groups.numel()
    all_indices = torch.arange(N, device=device)

    pos_idx = torch.full((N, num_pos), -1, dtype=torch.long, device=device)
    neg_idx = torch.full((N, num_neg), -1, dtype=torch.long, device=device)

    for g in range(G):
        members = (group_idx_per_node == g).nonzero(as_tuple=False).view(-1)
        m = members.numel()
        if m < 2:
            continue

        # Positives: same group, avoid self
        shifts = torch.randint(1, m, (m, num_pos), device=device)
        pos_choices = members[(torch.arange(m, device=device).unsqueeze(1) + shifts) % m]
        pos_idx[members] = pos_choices

        # Negatives: different groups
        neg_candidates = all_indices[group_idx_per_node != g]
        if neg_candidates.numel() > 0:
            idx = torch.randint(0, neg_candidates.numel(), (m, num_neg), device=device)
            neg_idx[members] = neg_candidates[idx]

    valid = (pos_idx[:, 0] >= 0) & (neg_idx[:, 0] >= 0)
    if not valid.any():
        return torch.tensor(0.0, device=device)

    a = z[valid]
    p = z[pos_idx[valid]]
    n = z[neg_idx[valid]]

    sim_pos = (a.unsqueeze(1) * p).sum(dim=-1)
    sim_neg = (a.unsqueeze(1) * n).sum(dim=-1)

    pos_logits = sim_pos / temperature
    neg_logits = sim_neg / temperature

    log_pos = torch.logsumexp(pos_logits, dim=1)
    log_all = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)

    loss = -(log_pos - log_all).mean()
    return loss
