# src/clustering/clusterer.py

import time
import torch

from .agglomerative_clustering import Agglomerative
from .oc_clustering import oc_cluster_single_event


def _cluster_contrastive(config, model, data_loader, device):
    all_predictions = []
    start_time = time.time()

    for i, data in enumerate(data_loader):
        data = data.to(device)

        preds = model(data.x, data.x_batch)

        all_predictions.append(preds[0].detach().cpu().numpy())

        if i > config["max_events"] - 1:
            break

    if config["algorithm"] == "agglomerative":
        reconstruction_labels = Agglomerative(
            all_predictions,
            threshold=config["distance_threshold"],
            metric=config["metric"],
            linkage=config["linkage"],
        )
    else:
        raise ValueError(
            f"Clustering algorithm {config['algorithm']} not recognized for contrastive task."
        )

    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / len(reconstruction_labels)
    print(
        f"[contrastive] Clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return reconstruction_labels


def _cluster_oc(config, model, data_loader, device):
    all_reco_ids = []
    start_time = time.time()

    beta_thr = config["oc_beta_thr"]
    min_sep = config["oc_min_center_separation"]
    use_cut = config["oc_use_distance_cut"]
    assign_r = config["oc_assignment_radius"]

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)

            cluster_coords, beta, prop_pred, batch = model(data.x, data.x_batch)

            # batch may contain multiple events (if batch_size>1)
            num_events = int(batch.max().item() + 1)

            for evt in range(num_events):
                evt_mask = (batch == evt)

                coords_evt = cluster_coords[evt_mask]
                beta_evt = beta[evt_mask]

                cluster_ids_evt = oc_cluster_single_event(
                    coords_evt,
                    beta_evt,
                    beta_thr=beta_thr,
                    min_center_separation=min_sep,
                    use_distance_cut=use_cut,
                    assignment_radius=assign_r,
                )

                all_reco_ids.append(cluster_ids_evt.cpu().numpy())

            if i > config["max_events"] - 1:
                break

    end_time = time.time()
    time_diff = end_time - start_time
    inf_time = time_diff / max(1, len(all_reco_ids))
    print(
        f"[OC] Inference + OC clustering completed in {time_diff:.2f} s. "
        f"Average time per event: {inf_time:.4f} s."
    )
    return all_reco_ids


def clusterer(config, model, data_loader, device):
    task = config.get("task", "contrastive")

    if task == "contrastive":
        return _cluster_contrastive(config, model, data_loader, device)
    elif task == "oc":
        return _cluster_oc(config, model, data_loader, device)
    else:
        raise ValueError(f"Unknown task '{task}' in clusterer.")
