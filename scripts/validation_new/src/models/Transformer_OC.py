# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import knn_graph


class Net_OC(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.1,
        k=20,
        num_heads=4,
        edge_hidden_dim=32,
        edge_out_dim=16,
        cluster_dim=2,      
        prop_dim=0,        
    ):
        """
        Graph network with Transformer-style attention and Object Condensation heads.

        Encoder + attention stack is identical to your previous setup.
        The old 'output' MLP is now a shared trunk feeding:
          - cluster_head: x_i in latent clustering space (R^{cluster_dim})
          - beta_head: scalar β_i in (0, 1)
          - (optional) prop_head: per-hit properties p_i (for L_p)

        Args mirror your previous model except for:
            cluster_dim: dimension of OC clustering space.
            prop_dim:    number of property outputs per node (e.g. 1 for energy).
        """
        super(Net_OC, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.num_heads = num_heads
        self.edge_out_dim = edge_out_dim
        self.cluster_dim = cluster_dim
        self.prop_dim = prop_dim

        # Indices in x: [x, y, z, E, layer, ...]
        self.idx_x = 0
        self.idx_y = 1
        self.idx_z = 2
        self.idx_E = 3
        self.idx_layer = 4

        # Encoder: 5 LC features -> hidden_dim
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        raw_edge_dim = 6
        self.edge_mlp = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_hidden_dim),
            nn.ELU(),
            nn.Linear(edge_hidden_dim, edge_out_dim),
            nn.ELU(),
        )

        self.att_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=0.0,
                edge_dim=edge_out_dim,
                aggr="add",
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Shared MLP (same structure as your old 'output' minus last Linear)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

        # OC heads
        self.cluster_head = nn.Linear(32, cluster_dim)
        self.beta_head = nn.Linear(32, 1)

        self.prop_head = None
        if prop_dim > 0:
            self.prop_head = nn.Linear(32, prop_dim)

    def build_graph(self, x, batch):
        """
        Build static k-NN graph and edge attributes from physical features.

        Args:
            x (Tensor): Raw node features (N, 7).
            batch (Tensor): Batch indices for each node (N,).

        Returns:
            edge_index (LongTensor): (2, num_edges)
            edge_attr  (Tensor): (num_edges, edge_out_dim)
        """
        pos = x[:, [self.idx_x, self.idx_y, self.idx_z]]  # (N, 3)

        edge_index = knn_graph(
            x=pos,
            k=self.k,
            batch=batch,
            loop=False,
        )

        row, col = edge_index

        pos_i = pos[row]
        pos_j = pos[col]
        dpos = pos_j - pos_i
        dist = dpos.norm(dim=-1, keepdim=True) + 1e-6

        layer_i = x[row, self.idx_layer].unsqueeze(-1)
        layer_j = x[col, self.idx_layer].unsqueeze(-1)
        d_layer = layer_j - layer_i

        E_i = x[row, self.idx_E].unsqueeze(-1)
        E_j = x[col, self.idx_E].unsqueeze(-1)
        eps = 1e-6
        E_ratio = torch.log((E_j + eps) / (E_i + eps))

        # [dx, dy, dz, dist, d_layer, log(E_j/E_i)]
        edge_raw = torch.cat([dpos, dist, d_layer, E_ratio], dim=-1)
        edge_attr = self.edge_mlp(edge_raw)

        return edge_index, edge_attr

    def forward(self, x, batch=None):
        """
        Args:
            x (Tensor): Input node features of shape (N, 7).
            batch (Tensor): Batch indices for nodes (N,).

        Returns:
            cluster_coords (N, cluster_dim),
            beta           (N,),
            prop_pred      (N, prop_dim) or None,
            batch
        """
        h = self.lc_encode(x)

        edge_index, edge_attr = self.build_graph(x, batch)

        for att, norm in zip(self.att_layers, self.norms):
            h_norm = norm(h)
            h_att = att(h_norm, edge_index, edge_attr)
            h = h + F.elu(h_att)

        feat = self.output_mlp(h)

        cluster_coords = self.cluster_head(feat)
        beta = torch.sigmoid(self.beta_head(feat)).squeeze(-1)

        prop_pred = None
        if self.prop_head is not None:
            prop_pred = self.prop_head(feat)

        return cluster_coords, beta, prop_pred, batch
