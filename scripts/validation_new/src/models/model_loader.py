# src/models/model_loader.py

import torch

from .Transformer_Contrastive import Net_Contrastive
from .Transformer_OC import Net_OC


def model_loader(config, model_path, device):

    task = config.get("task", "contrastive")
    model_name = config.get("model_name", "Transformer")

    if task == "contrastive":
        if model_name == "Transformer":
            model = Net_Contrastive(
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                contrastive_dim=config["contrastive_dim"],
                k=config["k"],
                num_heads=config["num_heads"],
                edge_hidden_dim=config["edge_hidden_dim"],
                edge_out_dim=config["edge_out_dim"],
            ).to(device)
        else:
            raise ValueError(f"Contrastive: Model {model_name} not recognized.")

    elif task == "oc":
        if model_name == "Transformer":
            model = Net_OC(
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                k=config["k"],
                num_heads=config["num_heads"],
                edge_hidden_dim=config["edge_hidden_dim"],
                edge_out_dim=config["edge_out_dim"],
                cluster_dim=2,  
                prop_dim=0,     
            ).to(device)
        else:
            raise ValueError(f"OC: Model {model_name} not recognized.")
    else:
        raise ValueError(f"Unknown task '{task}' (expected 'contrastive' or 'oc').")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
