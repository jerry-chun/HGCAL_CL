#!/usr/bin/env python3
import argparse, os
import numpy as np
import awkward as ak
import uproot
from HGCALEventClusterer_2D import HGCALEventClusterer_2D

_orig_vstack = np.__dict__.get('_orig_vstack', np.vstack)
def _patched_vstack(arrays, dtype=None):
    stacked = _orig_vstack(arrays)
    return stacked.astype(dtype) if dtype is not None else stacked
np.vstack = _patched_vstack

def main():
    ap = argparse.ArgumentParser(description="Run CLUE2D per-event and write LC friend tree (minimal fields).")
    ap.add_argument("-i", "--input", required=True, help="Input ROOT file")
    ap.add_argument("-o", "--output", help="Output ROOT file (default: <input>_LC.root)")
    ap.add_argument("-t", "--tree", default="HGCALTBout;1", help="Input TTree name")
    ap.add_argument("--dc", type=float, default=0.013)
    ap.add_argument("--rhoc", type=float, default=0.05)
    ap.add_argument("--dm", type=float, default=0.026)
    ap.add_argument("--pPB", type=int, default=10)
    ap.add_argument("--backend", default="cpu serial")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.output:
        base = os.path.basename(args.input)
        rootless = base[:-5] if base.endswith(".root") else base
        args.output = os.path.join(os.path.dirname(args.input), f"{rootless}_LC.root")

    fin = uproot.open(args.input)
    tree = fin[args.tree]
    n_events = tree.num_entries
    
    lc_layer, lc_cluster_id, lc_n_hits = [], [], []
    lc_energy, lc_x, lc_y, lc_z = [], [], [], []
    lc_shower_id, lc_purity = [], []

    clusterer = HGCALEventClusterer_2D(
        dc=args.dc, rhoc=args.rhoc, dm=args.dm, pPB=args.pPB, backend=args.backend
    )

    for i in range(int(n_events)):
        arr = tree.arrays(
            ["hit_x", "hit_y", "hit_z", "hit_Edep", "hit_layer", "hit_showerid"],
            entry_start=i, entry_stop=i+1
        )
        print(i)
        x = np.asarray(arr["hit_x"][0], dtype=np.float32)
        y = np.asarray(arr["hit_y"][0], dtype=np.float32)
        z = np.asarray(arr["hit_z"][0], dtype=np.float32)
        e = np.asarray(arr["hit_Edep"][0], dtype=np.float32)
        L = np.asarray(arr["hit_layer"][0], dtype=np.int32)
        s = np.asarray(arr["hit_showerid"][0], dtype=np.int32)

        clusterer.read_event(x, y, z, e, L, sids=s)
        clusterer.cluster_event(verbose=args.verbose)
        df = clusterer.get_cluster_summary(as_dataframe=True)

        lc_layer.append(df["layer"].to_numpy(np.int32).tolist())
        lc_cluster_id.append(df["cluster_id"].to_numpy(np.int32).tolist())
        lc_n_hits.append(df["n_hits"].to_numpy(np.int32).tolist())
        lc_energy.append(df["energy"].to_numpy(np.float32).tolist())
        lc_x.append(df["x"].to_numpy(np.float32).tolist())
        lc_y.append(df["y"].to_numpy(np.float32).tolist())
        lc_z.append(df["z"].to_numpy(np.float32).tolist())
        lc_shower_id.append(df["shower_id"].to_numpy(np.int32).tolist())
        lc_purity.append(df["purity"].to_numpy(np.float32).tolist())
        if args.verbose and ((i+1) % 100 == 0 or (i+1) == n_events):
            print(f"Processed {i+1}/{n_events} events", flush=True)

    friend = {
        "lc_layer":      ak.Array(lc_layer),
        "lc_cluster_id": ak.Array(lc_cluster_id),
        "lc_n_hits":     ak.Array(lc_n_hits),
        "lc_energy":     ak.Array(lc_energy),
        "lc_x":          ak.Array(lc_x),
        "lc_y":          ak.Array(lc_y),
        "lc_z":          ak.Array(lc_z),
        "lc_shower_id":  ak.Array(lc_shower_id),
        "lc_purity":    ak.Array(lc_purity)
    }

    with uproot.recreate(args.output, compression=uproot.ZSTD(1)) as fout:
        fout["LC"] = friend
    print(f"Wrote LC friend to: {args.output}")

if __name__ == "__main__":
    main()

