import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv

class Net_Contrastive(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.3,
        contrastive_dim=8,
        k=20 
    ):
        """
        Initializes a graph network that uses DynamicEdgeConv layers.
        """
        super(Net_Contrastive, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k

        # Input encoder 
        self.lc_encode = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Build the MLP (edge function) for each DynamicEdgeConv layer.
        def build_mlp():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Create a stack of DynamicEdgeConv layers.
        self.edgeconv_layers = nn.ModuleList([
            DynamicEdgeConv(
                nn=build_mlp(),
                k=self.k,
                aggr='max'  # or 'mean', 'sum', etc.
            )
            for _ in range(num_layers)
        ])

        # Output layer.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, batch=None):
        """
        Forward pass of the DynamicEdgeConv-based graph network.
        """
        # Encode input features to hidden_dim.
        x_enc = self.lc_encode(x)  

        # Pass through each DynamicEdgeConv layer.
        for conv in self.edgeconv_layers:
            x_enc = conv(x_enc, batch)   
            x_enc = F.elu(x_enc)
            x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        # Final output transformation.
        out = self.output(x_enc)
        return out, batch
