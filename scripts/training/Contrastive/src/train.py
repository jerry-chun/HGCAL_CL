import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch_geometric.nn.pool import knn_graph

def _group_sums(x: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    x: (N, D) 
    group_ids: (N,) in [0, num_groups-1]
    returns: (num_groups, D) or (num_groups,)
    """

    out = x.new_zeros((num_groups, x.size(1)))
    out.index_add_(0, group_ids, x)
    return out

def _group_counts(group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    return torch.bincount(group_ids, minlength=num_groups).clamp_min(1)

def _sample_k_from_mask(mask_2d: torch.Tensor, k: int) -> torch.Tensor:
    """
    Uniform-ish sampling without replacement from a boolean mask per row, via random scores + topk.
    mask_2d: (N, M) bool
    returns indices: (N, k) long, each row chooses k True positions (assumes at least k Trues per row).
    """
    r = torch.rand(mask_2d.shape, device=mask_2d.device)
    r = r.masked_fill(~mask_2d, float("-inf"))
    _, idx = torch.topk(r, k=k, dim=1)
    return idx  

def contrastive_loss_event_v3(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    coords: torch.Tensor,
    temperature: float = 0.1,
    k_geo: int = 32,
    k_neg: int = 8,
    lambda_compact: float = 0.2,
) -> torch.Tensor:
    device = embeddings.device
    N = embeddings.size(0)

    z = F.normalize(embeddings, p=2, dim=1)  

   
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    g = inv
    G = uniq.numel()

    # 1) Positive sampling (random within group)
    sim_full = z @ z.t() 
    idx = torch.arange(N, device=device)

    same = (g[:, None] == g[None, :])
    same.fill_diagonal_(False)

    rpos = torch.rand_like(sim_full)
    rpos = rpos.masked_fill(~same, float("-inf"))
    pos_j = torch.argmax(rpos, dim=1)
    s_pos = sim_full[idx, pos_j]  

    # 2) Two-tier negatives:
    #    (a) LOCAL semi-hard from geo-kNN
    #    (b) GLOBAL random fallback if not enough local negatives
    edge_index = knn_graph(coords, k=k_geo, loop=False)
    src = edge_index[0]
    dst = edge_index[1]

    s_edge = (z[src] * z[dst]).sum(dim=1)

    order = torch.argsort(src)
    src = src[order]
    dst = dst[order]
    s_edge = s_edge[order]

    dst_mat = dst.view(N, k_geo)    
    s_mat = s_edge.view(N, k_geo)   

    # local negative mask
    local_neg_mask = (g[dst_mat] != g[:, None])  
    local_neg_counts = local_neg_mask.sum(dim=1) 

    s_local = s_mat.masked_fill(~local_neg_mask, float("-inf"))
    s_local_topk, _ = torch.topk(s_local, k=k_neg, dim=1)  

    # (b) GLOBAL fallback negatives (random) for rows with insufficient locals 
    global_neg_mask = (g[:, None] != g[None, :])
    global_neg_mask.fill_diagonal_(False)

    global_neg_idx = _sample_k_from_mask(global_neg_mask, k=k_neg)  

    s_global = sim_full[idx[:, None], global_neg_idx]  

    use_local = (local_neg_counts >= k_neg)  
    s_neg_topk = torch.where(use_local[:, None], s_local_topk, s_global) 

    # 3) InfoNCE with K negatives (per-anchor)
    pos_logit = s_pos / temperature          
    neg_logits = s_neg_topk / temperature    

    denom = torch.logsumexp(torch.cat([pos_logit[:, None], neg_logits], dim=1), dim=1)
    loss_nce = -(pos_logit - denom)        

    # 4) Centroid compactness (per-anchor)
    sum_c = _group_sums(z, g, G)              
    mu = F.normalize(sum_c, p=2, dim=1)      
    cos_to_mu = (z * mu[g]).sum(dim=1)       
    loss_comp = 1.0 - cos_to_mu             

    loss_i = loss_nce + lambda_compact * loss_comp  
    
    print('1',loss_nce)
    print('2', lambda_compact * loss_comp)

    # 5) Equal per-particle weighting
    sum_loss_c = _group_sums(loss_i, g, G)         
    cnt_c = _group_counts(g, G).to(sum_loss_c)     
    mean_loss_c = sum_loss_c / cnt_c                
    loss_event = mean_loss_c.mean()

    return loss_event


def contrastive_loss_batch_v3(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    batch_ids: torch.Tensor,
    coords: torch.Tensor,
    temperature: float = 0.1,
    k_geo: int = 32,
    k_neg: int = 8,
    lambda_compact: float = 0.2,
) -> torch.Tensor:
    counts = torch.bincount(batch_ids)
    counts = counts[counts > 0]

    emb_splits = torch.split(embeddings, counts.tolist())
    gid_splits = torch.split(group_ids, counts.tolist())
    coord_splits = torch.split(coords, counts.tolist())

    losses = []
    for e, g, c in zip(emb_splits, gid_splits, coord_splits):
        losses.append(
            contrastive_loss_event_v3(
                embeddings=e,
                group_ids=g,
                coords=c,
                temperature=temperature,
                k_geo=k_geo,
                k_neg=k_neg,
                lambda_compact=lambda_compact,
            )
        )
    return torch.stack(losses).mean()


def train_new(
    train_loader,
    model,
    optimizer,
    device,
    scaler,
    temperature: float = 0.1,
    k_geo: int = 32,
    k_neg: int = 8,
    lambda_compact: float = 0.2,
    coord_cols=(0, 1, 2),  
):
    model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        coords = data.x[:, list(coord_cols)].contiguous()

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)
            loss = contrastive_loss_batch_v3(
                embeddings=embeddings,
                group_ids=data.assoc,
                batch_ids=data.x_batch,
                coords=coords,
                temperature=temperature,
                k_geo=k_geo,
                k_neg=k_neg,
                lambda_compact=lambda_compact,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())

    return total_loss / len(train_loader)


@torch.no_grad()
def test_new(
    test_loader,
    model,
    device,
    temperature: float = 0.1,
    k_geo: int = 32,
    k_neg: int = 8,
    lambda_compact: float = 0.2,
    coord_cols=(0, 1, 2),  
):
    model.eval()
    total_loss = 0.0

    for data in test_loader:
        data = data.to(device, non_blocking=True)
        coords = data.x[:, list(coord_cols)].contiguous()

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)
            loss = contrastive_loss_batch_v3(
                embeddings=embeddings,
                group_ids=data.assoc,
                batch_ids=data.x_batch,
                coords=coords,
                temperature=temperature,
                k_geo=k_geo,
                k_neg=k_neg,
                lambda_compact=lambda_compact,
            )

        total_loss += float(loss.detach())

    return total_loss / len(test_loader)
