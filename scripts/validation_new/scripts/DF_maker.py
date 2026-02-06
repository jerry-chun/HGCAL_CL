#!/usr/bin/env python3
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import torch
import argparse
from pathlib import Path
from src.data_loader.Data_Loader import Data_Loader
from src.models.model_loader import model_loader
from src.clustering.clusterer import clusterer
from src.metrics.build_dataframe import build_dataframe


def main():
    ap = argparse.ArgumentParser(
        description="Produce Pandas DataFrame with Information from ROOT files"
    )
    ap.add_argument("-i", "--input", required=True, help="Input ROOT file")
    ap.add_argument(
        "-max_events", "--max_events",
        type=int, default=1000,
        help="Maximum number of events to process"
    )
    ap.add_argument("-o", "--out", required=True, help="Output CSV path")
    #task / model type
    ap.add_argument("-task",
        type=str,
        choices=["contrastive", "oc"],
        default="oc",
        help="Type of model / reconstruction to use"
    )
    ap.add_argument(
        "-distance_threshold", type=float, default=0.3,
        help="Distance threshold for clustering (contrastive task)"
    )
   
    #OC-specific clustering
    ap.add_argument(
        "-oc_td", type=float, default=0.8,
        help="td variable of OC Method"
    )
    ap.add_argument("-final_dim", type = int, default = 16)
    ap.add_argument("-model_path", required = True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader
    loader = Data_Loader(
        root=args.input,
        split="test",
        max_events=args.max_events,
        batch_size=1,
        shuffle=False,
        follow_batch=["x"],
    )

    # Model config & load
    model_config = {
        "task": args.task,
        "hidden_dim": 64,
        "num_layers": 3,
        "dropout": 0.01,
        "k": 24,

        "contrastive_dim": args.final_dim,
        "coord_dim": args.final_dim,
        "path" : args.model_path
    }

    model = model_loader(
        config=model_config,
        device=device,
    )
    print(f"Model loaded successfully (task={args.task}).")

    # Clustering / OC config
    cluster_config = {
        "task": args.task,
        # contrastive path:
        "distance_threshold": args.distance_threshold,
        "metric": "cosine",
        "linkage": "average",
        "max_events": args.max_events,
        # OC path:
        "oc_beta_thr": 0.1,
        "oc_td" : args.oc_td
    }

    reconstruction_labels = clusterer(
        config=cluster_config,
        model=model,
        data_loader=loader,
        device=device,
    )
    print("Clustering / OC reconstruction completed successfully.")

    df = build_dataframe(reconstruction_labels, loader)
    print("DataFrame constructed successfully.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
