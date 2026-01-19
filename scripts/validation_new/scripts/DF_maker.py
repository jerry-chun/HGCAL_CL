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
    ap.add_argument("--out", required=True, help="Output CSV path")
    #task / model type
    ap.add_argument(
        "--task",
        type=str,
        choices=["contrastive", "oc"],
        default="contrastive",
        help="Type of model / reconstruction to use"
    )

    #shared model hyperparameters
    ap.add_argument(
        "-clustering_algorithm", type=str,
        default="agglomerative",
        help="Clustering algorithm to use (contrastive task)"
    )
    ap.add_argument(
        "-distance_threshold", type=float, default=0.3,
        help="Distance threshold for clustering (contrastive task)"
    )
    ap.add_argument(
        "-metric", type=str, default="cosine",
        help="Distance metric for clustering (contrastive task)"
    )
    ap.add_argument(
        "-linkage", type=str, default="average",
        help="Linkage method for clustering (contrastive task)"
    )
    ap.add_argument(
        "-cluster_events", type=int, default=1000,
        help="Maximum number of events to cluster (contrastive task)"
    )

    #OC-specific clustering
    ap.add_argument(
        "--oc_beta_thr", type=float, default=0.1,
        help="Minimum beta for extra OC centers"
    )
    ap.add_argument(
        "--oc_min_center_separation", type=float, default=0.5,
        help="Minimum separation of OC centers in latent space"
    )
    ap.add_argument(
        "--oc_use_distance_cut", action="store_true",
        help="If set, only assign hits within a radius to OC centers"
    )
    ap.add_argument(
        "--oc_assignment_radius", type=float, default=1.0,
        help="Assignment radius in OC latent space (if use_distance_cut)"
    )

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
    print(len(loader))

    # Model config & load
    model_config = {
        "task": "contrastive",
        "hidden_dim": 64,
        "num_layers": 3,
        "dropout": 0.01,
        "k": 24,

        "contrastive_dim": 16,
        "coord_dim": 3,
        "path" : "/vols/cms/mm1221/geant4sim/scripts/training/Contrastive/runs/EM_2_10/best_model.pt"
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
        "algorithm": args.clustering_algorithm,
        "distance_threshold": args.distance_threshold,
        "metric": args.metric,
        "linkage": args.linkage,
        "max_events": args.cluster_events,
        # OC path:
        "oc_beta_thr": args.oc_beta_thr,
        "oc_min_center_separation": args.oc_min_center_separation,
        "oc_use_distance_cut": args.oc_use_distance_cut,
        "oc_assignment_radius": args.oc_assignment_radius,
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
