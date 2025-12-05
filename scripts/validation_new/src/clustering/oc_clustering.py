# src/oc/oc_clustering.py

import torch


def oc_cluster_single_event(
    cluster_coords: torch.Tensor,
    beta: torch.Tensor,
    *,
    beta_thr: float = 0.1,
    min_center_separation: float = 0.5,
    use_distance_cut: bool = False,
    assignment_radius: float = 1.0,
) -> torch.Tensor:
    """
    Turn OC outputs (x, beta) for ONE event into integer cluster IDs.

    Returns:
        cluster_ids: (N,) tensor of ints
                     -1 for noise/unassigned,
                     0..K-1 for clusters.
    """
    device = cluster_coords.device
    N = cluster_coords.shape[0]
    cluster_ids = torch.full((N,), -1, dtype=torch.long, device=device)


    # 1. global β max is always a centre
    sorted_idx = torch.argsort(beta, descending=True)
    centers = [sorted_idx[0]]
    x_centers = [cluster_coords[sorted_idx[0]]]

    # 2. add more centres with β >= beta_thr and separated enough
    for idx in sorted_idx[1:]:
        if beta[idx] < beta_thr:
            break
        x_new = cluster_coords[idx]
        too_close = False
        for xc in x_centers:
            if torch.norm(x_new - xc) < min_center_separation:
                too_close = True
                break
        if not too_close:
            centers.append(idx)
            x_centers.append(x_new)

    center_indices = torch.tensor(centers, device=device, dtype=torch.long)
    K = center_indices.numel()
    centers_x = cluster_coords[center_indices]   # (K, D)

    # 3. assign hits
    diff = cluster_coords.unsqueeze(1) - centers_x.unsqueeze(0)  # (N, K, D)
    dists = torch.norm(diff, dim=-1)                             # (N, K)

    min_dists, nearest_center = torch.min(dists, dim=1)

    if use_distance_cut:
        assigned_mask = min_dists <= assignment_radius
        cluster_ids[assigned_mask] = nearest_center[assigned_mask]
    else:
        cluster_ids = nearest_center

    return cluster_ids
