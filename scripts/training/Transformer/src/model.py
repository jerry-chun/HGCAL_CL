import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.nn.pool import knn_graph


class Net(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.1,
        contrastive_dim=8,
        k=20,
        num_heads=4,
        edge_hidden_dim=32,
        edge_out_dim=16,
    ):
        """
        Graph network using Transformer-style attention on a static k-NN graph.

        Input node features: [x, y, z, E, layer]
          - x, y in [-100, 100] cm
          - z in [~322, ~520] cm
          - E in [0.001, ~0.6]
          - layer in [1, 50]
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.num_heads = num_heads
        self.edge_out_dim = edge_out_dim

        # indices in input feature vector
        self.idx_x = 0
        self.idx_y = 1
        self.idx_z = 2
        self.idx_E = 3
        self.idx_layer = 4

        # Normalisation hyperparameters (tuned to your detector)
        # After centering per event, EM showers ~10–20 cm, hadronic ~40 cm
        self.xy_scale = 10.0         # typical lateral scale [cm]

        # Front face and total depth in z
        self.z_front = 322.0         # [cm]
        self.z_scale = 200.0         # total depth → z' ~ [0, 1]

        # Layer IDs
        self.layer_min = 1.0
        self.layer_max = 50.0

        # Energy eps (data min is 1e-3)
        self.E_eps = 1e-5

        # Node encoder on normalised features
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Edge MLP on [dx, dy, dz, dist, d_layer, log(Ej/Ei)]
        raw_edge_dim = 6
        self.edge_mlp = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_hidden_dim),
            nn.ELU(),
            nn.Linear(edge_hidden_dim, edge_out_dim),
            nn.ELU(),
        )

        # Transformer-style conv layers + LayerNorm
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

        # Output head to contrastive embedding
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim),
        )

    # Event-wise normalisation of node features
    def normalize_eventwise(self, x, batch):
        """
        x:     (N, 5) = [x, y, z, E, layer] in raw units
        batch: (N,)   graph/event indices

        Returns:
            x_norm: (N, 5) = [x', y', z', E', layer']
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        pos = x[:, 0:3]      # x, y, z
        E   = x[:, 3:4]      # energy
        L   = x[:, 4:5]      # layer id

        # Number of nodes per event (G graphs)
        ones = torch.ones_like(L)
        n_nodes = global_add_pool(ones, batch)  # (G, 1)

        # 1) Center x,y per event and scale 
        pos_xy = pos[:, 0:2]                              # (N, 2)
        pos_xy_sum = global_add_pool(pos_xy, batch)       # (G, 2)
        pos_xy_mean = pos_xy_sum / (n_nodes + 1e-6)       # (G, 2)
        pos_xy_centered = pos_xy - pos_xy_mean[batch]     # (N, 2)
        pos_xy_norm = pos_xy_centered / self.xy_scale     # (N, 2)

        # 2) z: subtract front face and scale to [0, ~1]
        z = (pos[:, 2:3] - self.z_front) / self.z_scale   # (N, 1)

        pos_norm = torch.cat([pos_xy_norm, z], dim=-1)    # (N, 3)

        # 3) Energy: log(E) + per-event z-score
        E_log = torch.log(E + self.E_eps)                 # (N, 1)
        E_sum = global_add_pool(E_log, batch)             # (G, 1)
        E2_sum = global_add_pool(E_log * E_log, batch)    # (G, 1)
        E_mean = E_sum / (n_nodes + 1e-6)                 # (G, 1)
        E_var = E2_sum / (n_nodes + 1e-6) - E_mean * E_mean
        E_std = torch.sqrt(torch.clamp(E_var, min=1e-6))  # (G, 1)

        E_norm = (E_log - E_mean[batch]) / E_std[batch]   # (N, 1)

        # 4) Layer: global normalisation to [-1, 1]
        L_norm01 = (L - self.layer_min) / (self.layer_max - self.layer_min)
        L_norm = L_norm01 * 2.0 - 1.0                     # (N, 1)

        x_norm = torch.cat([pos_norm, E_norm, L_norm], dim=-1)
        return x_norm

    # Graph construction + edge attributes
    def build_graph(self, x_raw, x_norm, batch):
        """
        x_raw:  (N, 5) raw features [x,y,z,E,layer]
        x_norm: (N, 5) normalised features
        batch:  (N,)
        Returns:
            edge_index: (2, E)
            edge_attr:  (E, edge_out_dim)
        """
        if batch is None:
            batch = x_raw.new_zeros(x_raw.size(0), dtype=torch.long)

        # Use normalised positions for kNN and geometric edge features
        pos = x_norm[:, 0:3]   # (N, 3)

        edge_index = knn_graph(
            x=pos,
            k=self.k,
            batch=batch,
            loop=False,
        )

        row, col = edge_index

        # Geometry in normalised space
        pos_i = pos[row]
        pos_j = pos[col]
        dpos = pos_j - pos_i                           # (E, 3)
        dist = dpos.norm(dim=-1, keepdim=True) + 1e-6  # (E, 1)

        # Layer difference in normalised layer space [-1,1]
        L_norm = x_norm[:, 4:5]                        # (N, 1)
        L_i = L_norm[row]
        L_j = L_norm[col]
        d_layer = L_j - L_i                            # (E, 1)

        # Energy ratio using raw energies (scale-invariant)
        E_i_raw = x_raw[row, self.idx_E].unsqueeze(-1)
        E_j_raw = x_raw[col, self.idx_E].unsqueeze(-1)
        E_ratio = torch.log((E_j_raw + self.E_eps) / (E_i_raw + self.E_eps))  # (E, 1)

        # Concatenate into edge features: [dx, dy, dz, dist, d_layer, log(Ej/Ei)]
        edge_raw = torch.cat([dpos, dist, d_layer, E_ratio], dim=-1)
        edge_attr = self.edge_mlp(edge_raw)
        return edge_index, edge_attr

    # Forward
    def forward(self, x, batch=None):
        """
        x:     (N, 5) = [x, y, z, E, layer]
        batch: (N,) graph indices

        Returns:
            embeddings: (N, contrastive_dim)
            batch:      (N,)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x_raw = x
        x_norm = self.normalize_eventwise(x_raw, batch)

        # Node encoding on normalised features
        h = self.lc_encode(x_norm)

        # Build graph + edge attributes
        edge_index, edge_attr = self.build_graph(x_raw, x_norm, batch)

        # TransformerConv blocks with residual + LayerNorm
        for att, norm in zip(self.att_layers, self.norms):
            h_norm = norm(h)
            h_att = att(h_norm, edge_index, edge_attr)
            h = h + F.elu(h_att)

        out = self.output(h)
        return out, batch
