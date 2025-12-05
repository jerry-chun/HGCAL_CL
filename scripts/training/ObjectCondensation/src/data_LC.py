#!/usr/bin/env python3
# data.py

import os
import os.path as osp
import glob

import numpy as np
import awkward as ak
import uproot
import h5py  # kept if you use it elsewhere

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


def _sample_pos_neg_numpy(groups, rng):
    """
    Utility for contrastive sampling (unused by default, but kept here
    in case you want to re-enable x_pe/x_ne).
    """
    N = len(groups)
    x_pe = np.arange(N, dtype=np.int64)
    x_ne = np.arange(N, dtype=np.int64)

    uniq = np.unique(groups)
    idxs_all = np.arange(N)
    by_g = {g: np.flatnonzero(groups == g) for g in uniq}

    # Positive pairs: same group, different index
    for _, idxs in by_g.items():
        m = len(idxs)
        if m > 1:
            r = rng.integers(0, m - 1, size=m)
            pos = np.arange(m)
            r = r + (r >= pos)
            x_pe[idxs] = idxs[r]

    # Negative pairs: different groups
    for _, idxs in by_g.items():
        comp = np.setdiff1d(idxs_all, idxs, assume_unique=True)
        if len(comp) > 0:
            partners = rng.choice(comp, size=len(idxs), replace=True)
            x_ne[idxs] = partners

    return x_pe, x_ne


class CCV1(Dataset):
    """
    Contrastive Clustering V1 dataset.

    Expects ROOT files in `root/raw/` with a TTree (default name "LC")
    containing per-event jagged arrays with branches:

      - hit_x
      - hit_y
      - hit_z
      - hit_Edep
      - hit_layer
      - hit_showerid

    Each event i becomes a torch_geometric Data object with:

      - x: [N_i, 5]  (x, y, z, Edep, layer)
      - assoc: [N_i] (shower id per hit)
    """

    url = "/dummy/"

    def __init__(
        self,
        root,
        split="train",
        step_size=10,
        max_events=1e8,
        transform=None,
    ):
        super().__init__(root, transform)
        self.split = split
        self.step_size = step_size
        self.max_events = max_events
        self._epoch = 0

        # Load everything into memory once
        self.fill_data(max_events)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    # ------------------------------------------------------------------
    # PyG Dataset required properties
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self):
        # All ROOT files in raw_dir
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, "*.root")))
        # Or, if you want a single file:
        # raw_files = [osp.join(self.raw_dir, "step3_NTUPLE.root")]
        return raw_files

    @property
    def processed_file_names(self):
        # We don’t use the processed/ mechanism; everything is in memory
        return []

    def download(self):
        raise RuntimeError(
            "Dataset not found. Please download it from {} and move all "
            "*.root files to {}".format(self.url, self.raw_dir)
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def fill_data(self, max_events):
        counter = 0

        print("### Loading data")
        for fi, path in enumerate(tqdm(self.raw_paths)):
            # NOTE: change "LC" to your actual TTree name if different.
            # e.g. f"{path}:HGCALTBout;1"
            for array in uproot.iterate(
                f"{path}:{'LC'}",
                [
                    "lc_x",
                    "lc_y",
                    "lc_z",
                    "lc_energy",
                    "lc_layer",
                    "lc_shower_id",
                    "lc_n_hits",
                ],
                step_size=self.step_size,
            ):
                tmp_hits_x = array["lc_x"]
                tmp_hits_y = array["lc_y"]
                tmp_hits_z = array["lc_z"]
                tmp_hits_e = array["lc_energy"]
                tmp_hits_layer = array["lc_layer"]
                tmp_hits_showerid = array["lc_shower_id"]
                tmp_lc_n_hits = array["lc_n_hits"]
                
                mask = tmp_lc_n_hits > 1
                tmp_hits_x = tmp_hits_x[mask]
                tmp_hits_y = tmp_hits_y[mask]
                tmp_hits_z = tmp_hits_z[mask]
                tmp_hits_e = tmp_hits_e[mask]
                tmp_hits_layer = tmp_hits_layer[mask]
                tmp_hits_showerid = tmp_hits_showerid[mask]
                tmp_lc_n_hits = tmp_lc_n_hits[mask]
                
                
                # Make sure step_size is never larger than the chunk
                self.step_size = min(self.step_size, len(tmp_hits_x))

                if counter == 0:
                    # First chunk: initialise
                    self.hits_x = tmp_hits_x
                    self.hits_y = tmp_hits_y
                    self.hits_z = tmp_hits_z
                    self.hits_e = tmp_hits_e
                    self.hits_layer = tmp_hits_layer
                    self.hits_showerid = tmp_hits_showerid
                    self.lc_n_hits = tmp_lc_n_hits
                else:
                    # Concatenate subsequent chunks
                    self.hits_x = ak.concatenate((self.hits_x, tmp_hits_x))
                    self.hits_y = ak.concatenate((self.hits_y, tmp_hits_y))
                    self.hits_z = ak.concatenate((self.hits_z, tmp_hits_z))
                    self.hits_e = ak.concatenate((self.hits_e, tmp_hits_e))
                    self.hits_layer = ak.concatenate(
                        (self.hits_layer, tmp_hits_layer)
                    )
                    self.hits_showerid = ak.concatenate(
                        (self.hits_showerid, tmp_hits_showerid)
                    )
                    self.lc_n_hits = ak.concatenate(
                        (self.lc_n_hits, tmp_lc_n_hits)
                    )

                counter += 1
                if len(self.hits_x) > max_events:
                    print(f"Reached {max_events} events!")
                    break

            if len(self.hits_x) > max_events:
                break

    # ------------------------------------------------------------------
    # PyG Dataset required methods
    # ------------------------------------------------------------------
    def len(self):
        # Number of events is length of the outer jagged array
        return len(self.hits_x)

    def get(self, idx):
        # Per-event jagged → dense numpy arrays
        hit_x = np.array(self.hits_x[idx])
        hit_y = np.array(self.hits_y[idx])
        hit_z = np.array(self.hits_z[idx])
        hit_e = np.array(self.hits_e[idx])
        hit_l = np.array(self.hits_layer[idx])
        gids = np.array(self.hits_showerid[idx])
        hit_noh = np.array(self.lc_n_hits[idx])

        # Features: x, y, z, Edep, layer
        feats = np.column_stack((hit_x, hit_y, hit_z, hit_e, hit_l, hit_noh)).astype(np.float32)
        x = torch.from_numpy(feats)

        # If you want contrastive pairs, uncomment:
        # rng = np.random.default_rng(seed=(self._epoch * 1_000_003 + idx))
        # x_pe, x_ne = _sample_pos_neg_numpy(gids, rng)

        data = Data(
            x=x,
            assoc=torch.from_numpy(gids),
            # x_pe=torch.from_numpy(x_pe),
            # x_ne=torch.from_numpy(x_ne),
        )

        return data
