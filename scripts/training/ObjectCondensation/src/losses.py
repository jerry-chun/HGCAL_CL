# losses.py

import torch
import torch.nn.functional as F


def object_condensation_loss(
    beta: torch.Tensor,
    cluster_coords: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    q_min: float = 0.1,
    noise_label: int = -1,
    weights: torch.Tensor | None = None,
    s_att: float = 1.0,
    s_rep: float = 1.0,
    s_coward: float = 1.0,
    s_noise: float = 1.0,
) -> torch.Tensor:
    """
    Object Condensation loss.

    Encourages hits belonging to the same object to cluster together,
    while pushing different objects apart. Non-noise hits are driven to
    condensation points, noise hits are suppressed, and condensation
    points themselves are encouraged to commit (beta → 1).
    """
    device = beta.device
    beta = beta.view(-1)
    N = beta.numel()

    # Basic sanity checks
    assert cluster_coords.shape[0] == N
    assert group_ids.shape[0] == N

    # Keep beta away from numerical extremes
    eps = 1e-1
    beta = beta.clamp(min=eps, max=1.0 - eps)

    if weights is None:
        weights = torch.ones_like(beta)

    # Identify true hits vs noise
    not_noise = group_ids > noise_label

    # If we don’t have any real objects, only noise penalty applies
    if not not_noise.any():
        return s_noise * beta.mean()

    unique_oids = torch.unique(group_ids[not_noise])
    if unique_oids.numel() == 0:
        l_noise = beta[~not_noise].mean() if (~not_noise).any() else torch.tensor(0.0, device=device)
        return s_noise * l_noise

    attractive_mask = (group_ids[:, None] == unique_oids[None, :])

    # Charge term: condensation likelihood → “importance”
    q = torch.arctanh(beta) ** 2 + q_min
    if torch.isnan(q).any() or torch.isinf(q).any():
        raise RuntimeError("Invalid values in charge computation")

    # Pick condensation points as the highest-charge hit per object
    alphas = torch.argmax(q[:, None] * attractive_mask, dim=0)
    x_k = cluster_coords[alphas]          # (n_objs, D)
    q_k = q[alphas][None, :]              # (1, n_objs)

    # Distances from each hit to each condensation point
    dist = torch.cdist(cluster_coords, x_k)

    # Attractive component — pull same-object hits inward
    v_att_j_k = weights[:, None] * q[:, None] * q_k * attractive_mask * dist.pow(2)
    v_att = torch.mean(torch.sum(v_att_j_k, dim=0) / (torch.sum(attractive_mask, dim=0) + eps))

    # Repulsive component — keep other-object hits away
    rep_mask = ~attractive_mask
    v_rep_j_k = weights[:, None] * q[:, None] * q_k * rep_mask * F.relu(1.0 - dist)
    v_rep = torch.mean(torch.sum(v_rep_j_k, dim=0) / (torch.sum(rep_mask, dim=0) + eps))

    # “Coward” term — condensation points should have β close to 1
    l_coward = torch.mean(1.0 - beta[alphas])

    # Noise suppression
    l_noise = torch.mean(beta[~not_noise]) if (~not_noise).any() else torch.tensor(0.0, device=device)

    return (
        s_att * v_att +
        s_rep * v_rep +
        s_coward * l_coward +
        s_noise * l_noise
    )


