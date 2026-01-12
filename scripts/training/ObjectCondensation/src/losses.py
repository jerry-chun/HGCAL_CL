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
    Object Condensation loss:
    - Pulls hits of the same object together
    - Pushes different objects apart
    - Encourages condensation points to have high β
    - Penalizes noise β
    """
    device = beta.device
    beta = beta.view(-1)
    N = beta.numel()

    assert cluster_coords.shape[0] == N
    assert group_ids.shape[0] == N

    eps = 1e-1
    beta = beta.clamp(min=eps, max=1.0 - eps)

    if weights is None:
        weights = torch.ones_like(beta)

    not_noise = group_ids > noise_label

    # If everything is noise, only penalize high β on noise
    if not not_noise.any():
        return s_noise * beta.mean()

    unique_obj_ids = torch.unique(group_ids[not_noise])
    if unique_obj_ids.numel() == 0:
        l_noise = beta[~not_noise].mean() if (~not_noise).any() else torch.tensor(0.0, device=device)
        return s_noise * l_noise

    # Mask: (hit i belongs to object k)
    attractive_mask = (group_ids[:, None] == unique_obj_ids[None, :])

    # Charge measure
    q = torch.arctanh(beta).pow(2) + q_min

    # Pick condensation points = highest-charge hit per object
    alpha_idx = torch.argmax(q[:, None] * attractive_mask, dim=0)
    x_k = cluster_coords[alpha_idx]     
    q_k = q[alpha_idx][None, :]          

    # Distances to condensation points
    dist = torch.cdist(cluster_coords, x_k)

    # Attractive: same-object hits -> shrink towards condensation point
    v_att_j_k = weights[:, None] * q[:, None] * q_k * attractive_mask * dist.pow(2)
    v_att = torch.mean(
        torch.sum(v_att_j_k, dim=0) /
        (torch.sum(attractive_mask, dim=0) + eps)
    )

    # Repulsive: different-object hits → keep separated
    rep_mask = ~attractive_mask
    v_rep_j_k = weights[:, None] * q[:, None] * q_k * rep_mask * F.relu(1.0 - dist)
    v_rep = torch.mean(
        torch.sum(v_rep_j_k, dim=0) /
        (torch.sum(rep_mask, dim=0) + eps)
    )

    # Condensation point should have β ≈ 1
    l_coward = torch.mean(1.0 - beta[alpha_idx])

    # Noise 
    l_noise = torch.mean(beta[~not_noise]) if (~not_noise).any() else torch.tensor(0.0, device=device)

    return (
        s_att * v_att +
        s_rep * v_rep +
        s_coward * l_coward +
        s_noise * l_noise
    )
