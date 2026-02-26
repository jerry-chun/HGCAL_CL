import torch


@torch.no_grad()
def oc_cluster_single_event(
    cluster_coords: torch.Tensor,
    beta: torch.Tensor,
    *,
    beta_thr: float = 0.1,   
    td: float = 0.8,         
    assign_remaining_to_nearest: bool = True,
) -> torch.Tensor:
    """

    Steps:
      1) Candidate centers: hits with beta > beta_thr
      2) Sort candidates by beta descending
      3) Greedy non-max suppression: accept a candidate as a center if it's >= td away from all accepted centers
      4) Radius assignment (in descending-beta center order):
         assign all currently-unassigned hits within distance td to that center
    Returns:
        cluster_ids: (N,) long tensor
            -1 for unassigned (only if assign_remaining_to_nearest=False or no centers found)
             0..K-1 for clusters
    """
    device = cluster_coords.device
    beta = beta.view(-1)
    N = beta.numel()
    assert cluster_coords.shape[0] == N

    cluster_ids = torch.full((N,), -1, dtype=torch.long, device=device)

    # Candidate centers by beta
    cand_mask = beta > beta_thr
    cand_idx = torch.nonzero(cand_mask, as_tuple=False).view(-1)

    if cand_idx.numel() == 0:
        center_idx = torch.argmax(beta).view(1)
        centers = center_idx
    else:
        cand_beta = beta[cand_idx]
        order = torch.argsort(cand_beta, descending=True)
        cand_idx = cand_idx[order]

        #Greedy NMS to pick centers (>= td apart)
        centers_list = []
        centers_x = None  

        for idx in cand_idx:
            x = cluster_coords[idx].view(1, -1)  

            if centers_x is None:
                centers_list.append(idx)
                centers_x = x
                continue

            # distance to existing centers
            d = torch.cdist(x, centers_x).squeeze(0)  
            if torch.all(d >= td):
                centers_list.append(idx)
                centers_x = torch.cat([centers_x, x], dim=0)

        centers = torch.stack(centers_list) if len(centers_list) else torch.argmax(beta).view(1)
        centers_x = cluster_coords[centers]  

    K = centers_x.shape[0]
    if K == 0:
        return cluster_ids  

    # 3) Radius assignment in descending-beta center order
    center_order = torch.argsort(beta[centers], descending=True)
    centers = centers[center_order]
    centers_x = centers_x[center_order]

    unassigned = cluster_ids == -1

    for k in range(K):
        if not unassigned.any():
            break

        idx_u = torch.nonzero(unassigned, as_tuple=False).view(-1)
        d = torch.norm(cluster_coords[idx_u] - centers_x[k], dim=-1)  
        in_ball = d <= td

        if in_ball.any():
            assigned_idx = idx_u[in_ball]
            cluster_ids[assigned_idx] = k
            unassigned = cluster_ids == -1

    # As our data has no noise this is set to True
    if assign_remaining_to_nearest and (cluster_ids == -1).any():
        idx_u = torch.nonzero(cluster_ids == -1, as_tuple=False).view(-1)
        d_all = torch.cdist(cluster_coords[idx_u], centers_x)  
        nearest = torch.argmin(d_all, dim=1)
        cluster_ids[idx_u] = nearest

    return cluster_ids
