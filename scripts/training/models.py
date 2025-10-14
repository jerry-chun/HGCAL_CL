# models.py
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class EdgeConvNet(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, contrastive_dim=8, k=20):
        super().__init__()
        self.k = k
        self.lc_encode = nn.Sequential(
            nn.Linear(6, 32), nn.ELU(),
            nn.Linear(32, hidden_dim), nn.ELU()
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            )
            self.convs.append(DynamicEdgeConv(mlp, k=k, aggr="max"))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, contrastive_dim),
        )

    def forward(self, x, batch):
        x = self.lc_encode(x)
        for conv in self.convs:
            x = x + conv(x, batch)  # residual
        return self.output(x), batch

# Minimal “zoo” 
MODEL_ZOO = {
    "edgeconv": EdgeConvNet,
}
