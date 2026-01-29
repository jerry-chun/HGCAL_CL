import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv


import torch

@torch.no_grad()
def knn_edge_index_matmul(
    x: torch.Tensor,
    k: int,
    *,
    loop: bool = False,
    q_block: int = 1024,     # query block size
    r_block: int = 4096,     # reference block size
) -> torch.Tensor:
    """
    Exact kNN for a single event WITHOUT allocating NxN.
    Uses blockwise matmul distance + running topK merge.

    Returns edge_index (2, N*k).
    """
    assert x.dim() == 2
    N, D = x.shape
    if k >= N:
        raise ValueError(f"k must be < N (k={k}, N={N})")

    device = x.device
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

    # Precompute norms once
    x2_all = (xf * xf).sum(dim=1)  # (N,)

    # Running best (for each query point): store k smallest dist2 and indices
    best_dist = torch.full((N, k), float("inf"), device=device, dtype=xf.dtype)
    best_idx  = torch.full((N, k), -1, device=device, dtype=torch.long)

    arangeN = torch.arange(N, device=device, dtype=torch.long)

    for q0 in range(0, N, q_block):
        q1 = min(q0 + q_block, N)
        q = xf[q0:q1]                    # (Q, D)
        q2 = x2_all[q0:q1]               # (Q,)

        # Local running best for this query block (keeps memory smaller)
        bd = torch.full((q1 - q0, k), float("inf"), device=device, dtype=xf.dtype)
        bi = torch.full((q1 - q0, k), -1, device=device, dtype=torch.long)

        for r0 in range(0, N, r_block):
            r1 = min(r0 + r_block, N)
            r = xf[r0:r1]                # (R, D)
            r2 = x2_all[r0:r1]           # (R,)

            # dist2 = ||q||^2 + ||r||^2 - 2 q r^T  -> (Q, R)
            # Using fp32 matmul path via xf float above
            gram = q @ r.t()
            dist2 = q2[:, None] + r2[None, :] - 2.0 * gram
            dist2.clamp_(min=0.0)

            if not loop:
                # mask self distances when the diagonal lies in this block
                # global indices for queries and refs:
                q_idx = arangeN[q0:q1]
                r_idx = arangeN[r0:r1]
                # positions where q_idx == r_idx
                # for each q, self is at column (q_idx - r0) if in [r0,r1)
                mask = (q_idx >= r0) & (q_idx < r1)
                if mask.any():
                    cols = (q_idx[mask] - r0).long()
                    dist2[mask, cols] = float("inf")

            # Get topk within this ref block
            vals, idx_local = torch.topk(dist2, k=k, largest=False, sorted=False)  # (Q,k)
            idx_global = idx_local + r0                                           # (Q,k)

            # Merge with running best for this q-block: concat then topk
            merged_vals = torch.cat([bd, vals], dim=1)            # (Q, 2k)
            merged_idx  = torch.cat([bi, idx_global], dim=1)      # (Q, 2k)
            new_vals, new_pos = torch.topk(merged_vals, k=k, largest=False, sorted=False)
            bd = new_vals
            bi = torch.gather(merged_idx, dim=1, index=new_pos)

        best_dist[q0:q1] = bd
        best_idx[q0:q1] = bi

    # Build COO edge_index: i -> best_idx[i, j]
    row = arangeN.view(N, 1).expand(N, k)
    edge_index = torch.stack([row.reshape(-1), best_idx.reshape(-1)], dim=0)
    return edge_index

class Net(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.3,
        contrastive_dim=8,
        k=20,
        loop: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.loop = loop

        # Same encoder as your DynamicEdgeConv net
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        def build_mlp():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Static EdgeConv layers
        self.edgeconv_layers = nn.ModuleList([
            EdgeConv(nn=build_mlp(), aggr="max")
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim),
        )

    @torch.no_grad()
    def build_graph(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Build exact kNN graph ONCE from encoded features (recommended),
        using matmul+topk (exact, fast for N~1k).
        """
        return knn_edge_index_matmul(x_enc, k=self.k, loop=self.loop)

    def forward(self, x: torch.Tensor, batch=None, edge_index: torch.Tensor | None = None):
        """
        Single-event forward: batch ignored (kept for API compatibility).

        x: (N, 5)
        edge_index: optional precomputed (2, E). If not provided, built once.
        """
        x_enc = self.lc_encode(x)

        if edge_index is None:
            edge_index = self.build_graph(x_enc)

        for conv in self.edgeconv_layers:
            x_enc = conv(x_enc, edge_index)
            x_enc = F.elu(x_enc)
            x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        out = self.output(x_enc)
        return out, batch

