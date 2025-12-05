#!/usr/bin/env python3
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import torch
import argparse

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

    #task / model type
    ap.add_argument(
        "--task",
        type=str,
        choices=["contrastive", "oc"],
        default="contrastive",
        help="Type of model / reconstruction to use"
    )

    #shared model hyperparameters
    ap.add_argument("-model_name", type=str, default="Transformer",
                    help="Model name to use")
    ap.add_argument("-hidden_dim", type=int, default=64,
                    help="Hidden dimension size")
    ap.add_argument("-num_layers", type=int, default=5,
                    help="Number of layers in the model")
    ap.add_argument("-dropout", type=float, default=0.0038,
                    help="Dropout rate")
    ap.add_argument("-k", type=int, default=32,
                    help="Number of neighbors for k-NN")
    ap.add_argument("-num_heads", type=int, default=8,
                    help="Number of attention heads")
    ap.add_argument("-edge_hidden_dim", type=int, default=16,
                    help="Edge hidden dimension size")
    ap.add_argument("-edge_out_dim", type=int, default=16,
                    help="Edge output dimension size")
    ap.add_argument(
        "-model_path", "--model_path",
        required=True,
        help="Path to the pre-trained model"
    )

    # contrastive-specific hyperparameters 
    ap.add_argument(
        "-contrastive_dim", type=int, default=64,
        help="Contrastive embedding dimension (contrastive task only)"
    )

    #clustering for contrastive 
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
        "--oc_beta_thr", type=float, default=0.2,
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
        "task": args.task,
        "model_name": args.model_name,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "k": args.k,
        "num_heads": args.num_heads,
        "edge_hidden_dim": args.edge_hidden_dim,
        "edge_out_dim": args.edge_out_dim,
        "contrastive_dim": args.contrastive_dim
    }

    model = model_loader(
        config=model_config,
        model_path=args.model_path,
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

    df.to_csv("output_dataframe.csv", index=False)
    print("DataFrame saved to output_dataframe.csv")


if __name__ == "__main__":
    main()
