# train.py
import argparse
from pathlib import Path
import json
import random, numpy as np, torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import CCV1
from models import MODEL_ZOO
from losses import LOSS_ZOO

def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
@torch.no_grad()
def offset_edges_fast(x_pe, x_ne, x_batch, x_ptr):
    # x_ptr[x_batch] returns the start index of each node's graph
    base = x_ptr[x_batch]              # shape [N]
    return x_pe + base, x_ne + base

def train_epoch(loader, model, optimizer, device, loss_fn, loss_kwargs):
    model.train(); total = 0.0
    for data in tqdm(loader, desc="train"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.x_batch)
        x_pe_global, x_ne_global = offset_edges_fast(data.x_pe, data.x_ne, data.x_batch, data.x_ptr)

        z = out[0] if isinstance(out, (tuple, list)) else out
        loss = loss_fn(z, x_pe_global, x_ne_global, **loss_kwargs)
        loss.backward(); optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(loader, model, device, loss_fn, loss_kwargs):
    model.eval(); total = 0.0
    for data in tqdm(loader, desc="val"):
        data = data.to(device)
        out = model(data.x, data.x_batch)
        z = out[0] if isinstance(out, (tuple, list)) else out
        x_pe_global, x_ne_global = offset_edges_fast(data.x_pe, data.x_ne, data.x_batch, data.x_ptr)
        loss = loss_fn(z, x_pe_global, x_ne_global, **loss_kwargs)
        total += float(loss.item())
    return total / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    # basics
    ap.add_argument("--run-dir", default="runs/edgeconv_contrastive_pairs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda", choices=["auto","cpu","cuda"])
    # data
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=int(1e8))
    ap.add_argument("--step-size", type=int, default=1000)
    ap.add_argument("--train-path", type=str, default = "/vols/cms/mm1221/Independent/Data/photons_2/train/")
    ap.add_argument("--val-path", type=str, default = "/vols/cms/mm1221/Independent/Data/photons_2/val/")
    # model / loss
    ap.add_argument("--model", default="edgeconv", choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--contrastive-dim", type=int, default=16)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--loss", default="contrastive_pairs", choices=list(LOSS_ZOO.keys()))
    ap.add_argument("--temperature", type=float, default=0.01)
    # optim
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-size-sched", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else args.device)

    # data
    train_set = CCV1(root=args.train_path, split="train", step_size=args.step_size, max_events=args.max_events)
    val_set   = CCV1(root=args.val_path, split="val",   step_size=args.step_size, max_events=args.max_events)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=["x"])
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, follow_batch=["x"])

    # model
    ModelCls = MODEL_ZOO[args.model]
    model = ModelCls(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        k=args.k,
    ).to(device)

    # loss
    loss_fn = LOSS_ZOO[args.loss]
    loss_kwargs = dict(temperature=args.temperature)

    # optim
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size_sched, gamma=args.gamma)

    # run dir + save config
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "args.json", "w") as f: json.dump(vars(args), f, indent=2)

    best_val, bad = float("inf"), 0
    hist = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        # resample pos/neg pairs each epoch (deterministic)
        train_set.set_epoch(epoch); val_set.set_epoch(epoch)

        print(f"Epoch {epoch}/{args.epochs}")
        tr = train_epoch(train_loader, model, opt, device, loss_fn, loss_kwargs)
        va = eval_epoch(val_loader, model, device, loss_fn, loss_kwargs)
        sched.step()

        hist["epoch"].append(epoch); hist["train_loss"].append(tr); hist["val_loss"].append(va)

        # checkpoints
        torch.save(model.state_dict(), run_dir / "last.pt")
        if va < best_val:
            best_val, bad = va, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

        print(f"epoch={epoch} train={tr:.6f} val={va:.6f}")

    # save history
    try:
        import pandas as pd
        import pandas as pd
        import pandas as pd
        df = pd.DataFrame(hist)
        df.to_csv(run_dir / "loss.csv", index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
