import torch
import torch.nn.functional as F

def contrastive_multi_neg_per_group(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    For each anchor i:
      - pick 1 positive j (same group, j != i)
      - for every other group, pick 1 random negative
      - loss: -log( exp(s_pos/T) / (exp(s_pos/T) + sum_g exp(s_neg_g/T)) )
    """
    device = embeddings.device
    N, D = embeddings.shape

    # L2-normalize to use cosine similarity
    z = F.normalize(embeddings, p=2, dim=1)  

    unique_groups, group_idx_per_node = group_ids.unique(return_inverse=True)
    G = unique_groups.numel()

    #positives: one per anchor
    pos_idx = torch.full((N,), -1, dtype=torch.long, device=device)

    for g in range(G):
        idxs = (group_idx_per_node == g).nonzero(as_tuple=False).view(-1)
        m = idxs.numel()
        perm = idxs[torch.randperm(m, device=device)]
        # avoid self pairs
        if (perm == idxs).all():
            perm = perm.roll(1)
        pos_idx[idxs] = perm

    has_pos = pos_idx >= 0

    #negatives: 1 random from every group, then drop own group
    neg_idx_full = torch.empty(N, G, dtype=torch.long, device=device)

    for g in range(G):
        members = (group_idx_per_node == g).nonzero(as_tuple=False).view(-1)
        rand_choices = members[torch.randint(0, members.numel(), (N,), device=device)]
        neg_idx_full[:, g] = rand_choices

    all_cols = torch.arange(G, device=device)
    other_cols_per_group = torch.stack(
        [torch.cat([all_cols[:g], all_cols[g+1:]]) for g in range(G)],
        dim=0
    )  

    cols_for_each_anchor = other_cols_per_group[group_idx_per_node]  
    neg_idx = torch.gather(neg_idx_full, 1, cols_for_each_anchor)   

    #similarities and loss
    z_anchor = z[has_pos]             
    z_pos    = z[pos_idx[has_pos]]     
    z_neg    = z[neg_idx[has_pos]]    

    pos_logits = (z_anchor * z_pos).sum(dim=-1) / temperature         
    neg_logits = (z_anchor.unsqueeze(1) * z_neg).sum(dim=-1) / temperature  

    # positive / (positive + all negatives)
    num = torch.exp(pos_logits)
    den = num + torch.exp(neg_logits).sum(dim=1)

    loss = -torch.log(num / den)
    return loss.mean()

