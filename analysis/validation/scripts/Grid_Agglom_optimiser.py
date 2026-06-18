"""
Coarse-to-fine grid search over agglomerative distance_threshold.

Goal: find thresholds where:
  - ratio ≈ 1
  - purity ≈ 1
  - efficiency ≈ 1

We:
  1) Do a coarse scan over [min_threshold, max_threshold] with n_coarse points.
  2) Compute a simple "error" from (ratio, purity, efficiency) w.r.t. 1.
  3) Pick the best coarse threshold.
  4) Do a fine scan in a smaller window around that best coarse threshold.
  5) Save ALL results (coarse + fine) to CSV for later Pareto analysis.
"""

import argparse
import os
import sys
from typing import Dict, Any

import numpy as np
import torch
import pandas as pd

from src.data_loader.Data_Loader import Data_Loader
from src.models.model_loader import model_loader
from src.clustering.clusterer import clusterer
from src.metrics.build_dataframe import build_dataframe
from src.metrics.calculations import calc_purity, calc_efficiency, calc_response, calc_ratio


def parse_args():
    ap = argparse.ArgumentParser(description="Coarse-to-fine threshold optimization.")
    # Data / model
    ap.add_argument("-i", "--input", required=True, help="Input ROOT file.")
    ap.add_argument("--max_events", type=int, default=100, help="Max events to load.")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    ap.add_argument("--model_name", type=str, default="Transformer")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0038)
    ap.add_argument("--contrastive_dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=32, help="k-NN neighbors in model.")
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--edge_hidden_dim", type=int, default=16)
    ap.add_argument("--edge_out_dim", type=int, default=16)
    ap.add_argument("--model_path", required=True, help="Path to pretrained model.")

    # Clustering config
    ap.add_argument("--clustering_algorithm", type=str, default="agglomerative")
    ap.add_argument("--metric", type=str, default="cosine")
    ap.add_argument("--linkage", type=str, default="average")
    ap.add_argument("--cluster_events", type=int, default=100)

    # Metrics config
    ap.add_argument("--purity_threshold", type=float, default=0.2)
    ap.add_argument("--efficiency_threshold", type=float, default=0.7)

    # Grid search config
    ap.add_argument("--min_threshold", type=float, default=0.1)
    ap.add_argument("--max_threshold", type=float, default=0.4)
    ap.add_argument("--n_coarse", type=int, default=5)
    ap.add_argument("--n_fine", type=int, default=8)
    ap.add_argument("--fine_window", type=float, default=0.1)

    # Output
    ap.add_argument("-o", "--output", type=str, default="threshold_grid_search.csv")

    return ap.parse_args()


def evaluate_threshold(distance_threshold: float, data_loader, model, base_cluster_config: Dict[str, Any],
                       purity_threshold: float, efficiency_threshold: float, device) -> Dict[str, Any]:
    cluster_config = dict(base_cluster_config)
    cluster_config["distance_threshold"] = float(distance_threshold)

    model.eval()
    with torch.no_grad():
        reconstruction_labels = clusterer(config=cluster_config, model=model, data_loader=data_loader, device = device)

    df = build_dataframe(reconstruction_labels, data_loader)

    purity = calc_purity(df, threshold=purity_threshold)
    efficiency = calc_efficiency(df, threshold=efficiency_threshold)
    response_mean, response_std = calc_response(df)
    ratio = calc_ratio(df)

    err_p = abs(1.0 - purity)
    err_e = abs(1.0 - efficiency)
    err_r = abs(1.0 - ratio)
    err_combined = float(np.sqrt(err_p**2 + err_e**2 + err_r**2))

    if np.isnan(response_std):
        response_std = 0.0

    return {
        "distance_threshold": float(distance_threshold),
        "purity": purity,
        "efficiency": efficiency,
        "response_mean": response_mean,
        "response_std": response_std,
        "ratio": ratio,
        "err_purity": err_p,
        "err_efficiency": err_e,
        "err_ratio": err_r,
        "err_combined": err_combined
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = Data_Loader(root=args.input, split="test", max_events=args.max_events,
                         batch_size=args.batch_size, shuffle=False, follow_batch=["x"])

    model_config = {
        "model_name": args.model_name, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "dropout": args.dropout,
        "contrastive_dim": args.contrastive_dim, "k": args.k,
        "num_heads": args.num_heads, "edge_hidden_dim": args.edge_hidden_dim,
        "edge_out_dim": args.edge_out_dim
    }

    model = model_loader(config=model_config, model_path=args.model_path, device=device)
    print("Model loaded.")

    base_cluster_config = {
        "algorithm": args.clustering_algorithm,
        "metric": args.metric,
        "linkage": args.linkage,
        "max_events": args.cluster_events,
    }

    records = []

    print("\n=== Coarse scan ===")
    coarse_vals = np.linspace(args.min_threshold, args.max_threshold, args.n_coarse)
    for t in coarse_vals:
        print(f"\nCoarse threshold t={t:.4f}")
        res = evaluate_threshold(t, loader, model, base_cluster_config,
                                 args.purity_threshold, args.efficiency_threshold, device = device)
        res["stage"] = "coarse"
        records.append(res)
        print(f"  {res['purity']:.3f} purity, {res['efficiency']:.3f} eff, {res['ratio']:.3f} ratio, err={res['err_combined']:.4f}")

    df_coarse = pd.DataFrame.from_records(records)
    best_t = df_coarse.loc[df_coarse["err_combined"].idxmin()]["distance_threshold"]
    print(f"\nBest coarse threshold: {best_t:.4f}")

    fine_min = max(args.min_threshold, best_t - args.fine_window / 2)
    fine_max = min(args.max_threshold, best_t + args.fine_window / 2)

    print("\n=== Fine scan ===")
    print(f"Fine scan range: [{fine_min:.4f}, {fine_max:.4f}]")
    fine_vals = np.linspace(fine_min, fine_max, args.n_fine)

    for t in fine_vals:
        print(f"\nFine threshold t={t:.4f}")
        res = evaluate_threshold(t, loader, model, base_cluster_config,
                                 args.purity_threshold, args.efficiency_threshold, device = device)
        res["stage"] = "fine"
        records.append(res)
        print(f"  {res['purity']:.3f} purity, {res['efficiency']:.3f} eff, {res['ratio']:.3f} ratio, err={res['err_combined']:.4f}")

    df = pd.DataFrame.from_records(records).sort_values(["distance_threshold", "stage"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")

    best = df.loc[df["err_combined"].idxmin()]
    print("\nBest overall:")
    print(best)


if __name__ == "__main__":
    main()

