# src/models/model_loader.py

import torch

from .model_Contrastive import Net_Contrastive
from .model_OC import Net_OC


def model_loader(config, device):

    task = config.get("task", "contrastive")
    path = config["path"]

    if task == "contrastive":
        model = Net_Contrastive(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            contrastive_dim=config["contrastive_dim"],
            k=config["k"],
        ).to(device)

    elif task == "oc":
        model = Net_OC(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            k=config["k"],
            coord_dim=config["coord_dim"],  
        ).to(device)
    else:
        raise ValueError(f"Unknown task '{task}' (expected 'contrastive' or 'oc').")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
