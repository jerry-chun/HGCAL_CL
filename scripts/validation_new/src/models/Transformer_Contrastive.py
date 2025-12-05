import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import knn_graph


class Net_Contrastive(nn.Module):
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

        Args:
            hidden_dim (int): Node hidden feature dimension.
            num_layers (int): Number of attention layers.
            dropout (float): Dropout rate in the output head.
            contrastive_dim (int): Output dimension of the final embedding.
            k (int): Number of neighbors in k-NN for the static graph.
            num_heads (int): Number of attention heads per layer.
            edge_hidden_dim (int): Hidden dim in edge attribute MLP.
            edge_out_dim (int): Output dim of edge attribute MLP (edge_dim for attention).
        """
        super(Net_Contrastive, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k
        self.num_heads = num_heads
        self.edge_out_dim = edge_out_dim

        self.idx_x = 0
        self.idx_y = 1
        self.idx_z = 2
        self.idx_E = 3
        self.idx_layer = 4

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

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim),
        )

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
        pos = x[:, [self.idx_x, self.idx_y, self.idx_z]]  

        # Static k-NN graph in geometry space.
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

        # Layer difference
        layer_i = x[row, self.idx_layer].unsqueeze(-1) 
        layer_j = x[col, self.idx_layer].unsqueeze(-1)  
        d_layer = layer_j - layer_i  

        # Energy ratio (log(E_j / E_i))
        E_i = x[row, self.idx_E].unsqueeze(-1)
        E_j = x[col, self.idx_E].unsqueeze(-1)
        eps = 1e-6
        E_ratio = torch.log((E_j + eps) / (E_i + eps))  

        #[dx, dy, dz, dist, d_layer, log(E_j/E_i)]
        edge_raw = torch.cat([dpos, dist, d_layer, E_ratio], dim=-1)

        edge_attr = self.edge_mlp(edge_raw)  

        return edge_index, edge_attr

    def forward(self, x, batch=None):
        """
        Forward pass of the attention-based graph network.

        Args:
            x (Tensor): Input node features of shape (N, 7).
            batch (Tensor, optional): Batch indices for nodes (N,).

        Returns:
            tuple: (embeddings, batch)
        """

        h = self.lc_encode(x)  

        edge_index, edge_attr = self.build_graph(x, batch)

        for att, norm in zip(self.att_layers, self.norms):
            h_norm = norm(h)
            h_att = att(h_norm, edge_index, edge_attr)  
            h = h + F.elu(h_att)  

        out = self.output(h)  
        return out, batch

