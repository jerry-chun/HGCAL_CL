#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader.Data_Loader import Data_Loader
from src.models.model_loader import model_loader


# Energy-core filtering (per event)

@torch.no_grad()
def energy_core_mask_per_event(
    y: torch.Tensor,
    E: torch.Tensor,
    *,
    keep_frac: float = 0.95,         
    min_hits_per_group: int = 2,     
) -> torch.Tensor:
    """
    Build boolean mask selecting per-shower energy-core hits:
      For each truth label:
        sort hits by descending energy
        keep smallest prefix whose cumulative energy reaches keep_frac * total_energy(label)

    Returns:
      mask (N,) on CPU
    """
    y = y.detach().cpu().long().view(-1)
    E = E.detach().cpu().float().view(-1).clamp_min(0.0)
    N = y.numel()
    if E.numel() != N:
        raise ValueError(f"E length {E.numel()} != N {N}")

    mask = torch.zeros(N, dtype=torch.bool)

    for lab in torch.unique(y):
        idx = (y == lab).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue

        Ei = E[idx]
        tot = float(Ei.sum().item())

        if tot <= 0.0:
            mask[idx] = True
            continue

        order = torch.argsort(Ei, descending=True)
        idx_sorted = idx[order]
        Ei_sorted = Ei[order]

        csum = torch.cumsum(Ei_sorted, dim=0) / tot
        keep = csum <= keep_frac

        # make sure we keep at least min_hits_per_group if possible
        if keep.sum().item() < min_hits_per_group and idx_sorted.numel() >= min_hits_per_group:
            keep[:min_hits_per_group] = True
        elif keep.sum().item() == 0 and idx_sorted.numel() > 0:
            keep[0] = True

        mask[idx_sorted[keep]] = True

    return mask



def sample_pairs_from_labels(y: torch.Tensor, n_pos: int, n_neg: int, rng) -> tuple[torch.Tensor, ...]:
    """
    Sample positive (same-label) and negative (different-label) index pairs.
    """
    y_np = y.detach().cpu().numpy()
    labels = np.unique(y_np)
    by_label = {lab: np.where(y_np == lab)[0] for lab in labels}
    pos_labels = [lab for lab in labels if by_label[lab].size >= 2]

    # positives
    if len(pos_labels) == 0 or n_pos <= 0:
        i_pos = j_pos = np.empty((0,), dtype=np.int64)
    else:
        i_pos = np.empty((n_pos,), dtype=np.int64)
        j_pos = np.empty((n_pos,), dtype=np.int64)
        for t in range(n_pos):
            lab = rng.choice(pos_labels)
            idx = by_label[lab]
            a, b = rng.choice(idx, size=2, replace=False)
            i_pos[t], j_pos[t] = a, b

    # negatives
    if labels.size < 2 or n_neg <= 0:
        i_neg = j_neg = np.empty((0,), dtype=np.int64)
    else:
        i_neg = np.empty((n_neg,), dtype=np.int64)
        j_neg = np.empty((n_neg,), dtype=np.int64)
        for t in range(n_neg):
            la, lb = rng.choice(labels, size=2, replace=False)
            a = rng.choice(by_label[la])
            b = rng.choice(by_label[lb])
            i_neg[t], j_neg[t] = a, b

    return (
        torch.from_numpy(i_pos).long(),
        torch.from_numpy(j_pos).long(),
        torch.from_numpy(i_neg).long(),
        torch.from_numpy(j_neg).long(),
    )


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 P(equal)."""
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")
    diff = pos_scores[:, None] - neg_scores[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


@torch.no_grad()
def recall_at_k_cosine(z: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Recall@k with full cosine similarity matrix."""
    N = z.size(0)
    if N <= 1:
        return float("nan")

    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)

    kk = min(k, N - 1)
    nn_idx = torch.topk(sim, k=kk, dim=1, largest=True).indices
    hit = (y[nn_idx] == y[:, None]).any(dim=1)
    return float(hit.float().mean().item())


@torch.no_grad()
def knn_contamination(z: torch.Tensor, y: torch.Tensor, K: int) -> float:
    """
    kNN contamination@K (group-balanced):
      fraction of top-K neighbors that are different-label, averaged per label.
    """
    N = z.size(0)
    if N <= 1:
        return float("nan")

    K_eff = min(K, N - 1)
    if K_eff <= 0:
        return float("nan")

    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)

    nn_idx = torch.topk(sim, k=K_eff, dim=1, largest=True).indices
    same = (y[nn_idx] == y[:, None]).float()
    contam_per_node = 1.0 - same.mean(dim=1)

    vals = []
    for lab in torch.unique(y):
        mask = (y == lab)
        if mask.any():
            vals.append(contam_per_node[mask].mean())
    if len(vals) == 0:
        return float("nan")
    return float(torch.stack(vals).mean().item())


def safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


@torch.no_grad()
def event_metrics(
    emb: torch.Tensor,
    y: torch.Tensor,
    E: torch.Tensor,
    *,
    core_keep_frac: float,
    n_pos_pairs: int,
    n_neg_pairs: int,
    rng: np.random.Generator,
) -> dict:
    """
    Compute per-event metrics on energy-core hits (per shower keep_frac of energy).
    Returns both median and tail quantile geometry summaries.
    """
    emb = emb.detach().float().cpu()
    y = y.detach().long().view(-1).cpu()
    E = E.detach().float().view(-1).cpu()

    N_all = emb.size(0)
    if N_all != y.numel() or N_all != E.numel():
        raise ValueError(f"emb N={N_all}, labels N={y.numel()}, energies N={E.numel()}")

    n_groups_all = int(torch.unique(y).numel())

    # --- core selection ---
    core_mask = energy_core_mask_per_event(y, E, keep_frac=core_keep_frac)
    emb_c = emb[core_mask]
    y_c = y[core_mask]

    N = emb_c.size(0)
    n_groups = int(torch.unique(y_c).numel()) if N > 0 else 0

    if N <= 1 or n_groups == 0:
        return dict(
            n_nodes_all=int(N_all),
            n_groups_all=int(n_groups_all),
            n_nodes=int(N),
            n_groups=int(n_groups),
            r_at_1=float("nan"),
            r_at_5=float("nan"),
            r_at_10=float("nan"),
            auc=float("nan"),
            intra_compactness=float("nan"),
            inter_separation=float("nan"),
            intra_q99=float("nan"),
            inter_q01=float("nan"),
            knn_contam_at_10=float("nan"),
            knn_contam_at_100=float("nan"),
        )

    # L2-normalize for cosine geometry
    z = F.normalize(emb_c, p=2, dim=1)

    # Recall@K on core hits
    r1 = recall_at_k_cosine(z, y_c, k=1)
    r5 = recall_at_k_cosine(z, y_c, k=5)
    r10 = recall_at_k_cosine(z, y_c, k=10)

    # Sample pairs on core hits
    i_pos, j_pos, i_neg, j_neg = sample_pairs_from_labels(y_c, n_pos_pairs, n_neg_pairs, rng)

    if i_pos.numel() > 0:
        s_pos = (z[i_pos] * z[j_pos]).sum(dim=1).numpy()
        d_pos = (1.0 - s_pos)
    else:
        s_pos = np.array([], dtype=np.float32)
        d_pos = np.array([], dtype=np.float32)

    if i_neg.numel() > 0:
        s_neg = (z[i_neg] * z[j_neg]).sum(dim=1).numpy()
        d_neg = (1.0 - s_neg)
    else:
        s_neg = np.array([], dtype=np.float32)
        d_neg = np.array([], dtype=np.float32)

    auc = auc_from_scores(s_pos, s_neg)

    intra_med = float(np.median(d_pos)) if d_pos.size > 0 else float("nan")
    inter_med = float(np.median(d_neg)) if d_neg.size > 0 else float("nan")

    # Tail quantiles 
    intra_q99 = safe_quantile(d_pos, 0.99)
    inter_q01 = safe_quantile(d_neg, 0.01)

    # kNN contamination 
    knn_cont_10 = knn_contamination(z, y_c, K=10)
    knn_cont_100 = knn_contamination(z, y_c, K=100)

    return dict(
        n_nodes_all=int(N_all),
        n_groups_all=int(n_groups_all),
        n_nodes=int(N),
        n_groups=int(n_groups),
        r_at_1=float(r1),
        r_at_5=float(r5),
        r_at_10=float(r10),
        auc=float(auc),
        intra_compactness=float(intra_med),
        inter_separation=float(inter_med),
        intra_q99=float(intra_q99),
        inter_q01=float(inter_q01),
        knn_contam_at_10=float(knn_cont_10),
        knn_contam_at_100=float(knn_cont_100),
    )


# Main

def parse_args():
    p = argparse.ArgumentParser(description="Build per-event embedding metrics dataframe (energy-core filtered).")
    p.add_argument("-i", "--input_file", required=True, help="Input ROOT file path")
    p.add_argument("-o", "--outcsv", required=True, help="Output CSV path")
    p.add_argument(
        "-task", "--task", required=True,
        choices=["contrastive", "oc", "object_condensation"],
        help="Which task/model head to use"
    )
    p.add_argument("-model_path", "--model_path", required=True, help="Path to .pt model weights")
    p.add_argument("-final_dim", "--final_dim", type=int, default=16, help="Embedding dimension (e.g. 16)")

    # data loader knobs
    p.add_argument("--split", default="test")
    p.add_argument("--max_events", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--shuffle", action="store_true", default=False)

    # metrics knobs
    p.add_argument("--n_pos_pairs", type=int, default=4000)
    p.add_argument("--n_neg_pairs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=123)

    # energy-core knob
    p.add_argument(
        "--core_keep_frac",
        type=float,
        default=0.95,
        help="Per-shower energy fraction to keep (0.80 keeps hits that sum to 80%% of that shower's energy).",
    )

    # model hyperparams if loader needs them
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--k", type=int, default=24)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = Data_Loader(
        root=args.input_file,
        split=args.split,
        max_events=args.max_events,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        follow_batch=["x"],
    )

    model_config = {
        "task": "contrastive" if args.task == "contrastive" else "oc",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "k": args.k,
        "contrastive_dim": args.final_dim,
        "coord_dim": args.final_dim,
        "path": args.model_path,
    }

    model = model_loader(config=model_config, device=device)
    model.eval()

    all_predictions = []
    true_labels = []
    all_energies = []

    for i, data in enumerate(loader):
        data = data.to(device)

        out = model(data.x, data.x_batch)
        preds = out[0] if args.task == "contrastive" else out[1]

        all_predictions.append(preds.detach())
        all_energies.append(data.x[:, 3].detach())  
        true_labels.append(data.assoc.detach())

    rng = np.random.default_rng(args.seed)
    rows = []
    for event_id, (preds, y, E) in enumerate(zip(all_predictions, true_labels, all_energies)):
        m = event_metrics(
            preds,
            y,
            E,
            core_keep_frac=args.core_keep_frac,
            n_pos_pairs=args.n_pos_pairs,
            n_neg_pairs=args.n_neg_pairs,
            rng=rng,
        )
        m["event_id"] = int(event_id)
        rows.append(m)

    df = pd.DataFrame(rows)
    out_path = Path(args.outcsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(df)} events to {out_path}")


if __name__ == "__main__":
    main()