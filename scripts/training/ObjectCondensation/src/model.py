import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv


class Net(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.3,
        k=20,
        coord_dim=3,   # NEW: embedding/cluster coordinate dimension for OC
    ):
        """
        DynamicEdgeConv network adapted for Object Condensation.

        Outputs:
          - beta: (N,) in (0,1)
          - cluster_coords: (N, coord_dim)
          - batch: unchanged
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.coord_dim = coord_dim

        # Input encoder (your code currently uses 5 input features)
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # Build the MLP (edge function) for each DynamicEdgeConv layer.
        def build_mlp():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Create a stack of DynamicEdgeConv layers.
        self.edgeconv_layers = nn.ModuleList([
            DynamicEdgeConv(
                nn=build_mlp(),
                k=self.k,
                aggr="max",
            )
            for _ in range(num_layers)
        ])

        # Keep the old name `self.output`, but make it a trunk to 32 dims (same as before)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

        # NEW: two tiny heads for Object Condensation
        self.beta_head = nn.Linear(32, 1)            # -> (N, 1)
        self.coord_head = nn.Linear(32, coord_dim)   # -> (N, coord_dim)

    def forward(self, x, batch=None):
        """
        Args:
          x: (N, 5) node features
          batch: (N,) batch vector (optional)

        Returns:
          beta: (N,)
          cluster_coords: (N, coord_dim)
          batch: (N,) (unchanged)
        """
        # Encode input features to hidden_dim
        x_enc = self.lc_encode(x)  # (N, hidden_dim)

        # Pass through DynamicEdgeConv layers
        for conv in self.edgeconv_layers:
            x_enc = conv(x_enc, batch)  # edges computed dynamically
            x_enc = F.elu(x_enc)
            x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        # Shared trunk (kept as `self.output` to minimize diffs)
        h = self.output(x_enc)  # (N, 32)

        # OC outputs
        beta = torch.sigmoid(self.beta_head(h)).squeeze(-1)  # (N,)
        cluster_coords = self.coord_head(h)                  # (N, coord_dim)

        return beta, cluster_coords, batch

