#!/usr/bin/env python3
# data.py

import os.path as osp
import glob

import numpy as np
import awkward as ak
import uproot

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

class CCV1(Dataset):
    """
    Contrastive Clustering V1 dataset.

    Expects ROOT files in `root/raw/` with a TTree (default name "HGCALTBout")
    containing per-event jagged arrays with branches:

      - hit_x
      - hit_y
      - hit_z
      - hit_Edep
      - hit_layer
      - hit_showerid
      - hit_purity

    Each event i becomes a torch_geometric Data object with:

      - x:      [N_i, 5]  (x, y, z, Edep, layer)
      - assoc:  [N_i]     (shower id per hit)
      - purity: [N_i]
    """

    url = "/dummy/"

    def __init__(
        self,
        root,
        split="train",
        step_size=200,     # bumped up by default: fewer chunks = faster
        max_events=int(1e8),
        transform=None,
    ):
        super().__init__(root, transform)
        self.split = split
        self.step_size = int(step_size)
        self.max_events = int(max_events)
        self._epoch = 0

        # Load everything into memory once (but efficiently)
        self.fill_data(self.max_events)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def raw_file_names(self):
        # All ROOT files in raw_dir
        return sorted(glob.glob(osp.join(self.raw_dir, "*.root")))

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
    # Data loading (FAST: collect chunks in lists, concatenate once)
    # ------------------------------------------------------------------
    def fill_data(self, max_events: int):
        print("### Loading data")

        chunks_x, chunks_y, chunks_z = [], [], []
        chunks_e, chunks_layer, chunks_sid, chunks_purity = [], [], [], []

        total_events = 0
        stop = False

        for path in tqdm(self.raw_paths):
            if stop:
                break

            for array in uproot.iterate(
                f"{path}:HGCALTBout",  
                [
                    "hit_x",
                    "hit_y",
                    "hit_z",
                    "hit_Edep",
                    "hit_layer",
                    "hit_showerid",
                    "hit_purity",
                ],
                step_size=self.step_size,
                remind=False,  # less printing overhead
            ):
                chunks_x.append(array["hit_x"])
                chunks_y.append(array["hit_y"])
                chunks_z.append(array["hit_z"])
                chunks_e.append(array["hit_Edep"])
                chunks_layer.append(array["hit_layer"])
                chunks_sid.append(array["hit_showerid"])
                chunks_purity.append(array["hit_purity"])

                # Count how many EVENTS were added in this chunk (outer length)
                total_events += len(array["hit_x"])

                if total_events >= max_events:
                    stop = True
                    break

        if total_events == 0:
            raise RuntimeError(
                "No events found. Check your raw_dir, TTree name, or branch names."
            )

        # One concatenate per branch (fast)
        hits_x = ak.concatenate(chunks_x)
        hits_y = ak.concatenate(chunks_y)
        hits_z = ak.concatenate(chunks_z)
        hits_e = ak.concatenate(chunks_e)
        hits_layer = ak.concatenate(chunks_layer)
        hits_sid = ak.concatenate(chunks_sid)
        hits_purity = ak.concatenate(chunks_purity)
        print(f"### Loaded {len(self.hits_x)} events into memory")

    # ------------------------------------------------------------------
    # PyG Dataset required methods
    # ------------------------------------------------------------------
    def len(self):
        # Number of events is length of outer jagged array
        return len(self.hits_x)

    def get(self, idx):
        # Pull one event (jagged entry) -> numpy arrays
        hit_x = np.asarray(self.hits_x[idx])
        hit_y = np.asarray(self.hits_y[idx])
        hit_z = np.asarray(self.hits_z[idx])
        hit_e = np.asarray(self.hits_e[idx])
        hit_l = np.asarray(self.hits_layer[idx])
        gids = np.asarray(self.hits_showerid[idx])
        purity = np.asarray(self.hits_purity[idx])

        # Features: x, y, z, Edep, layer
        feats = np.column_stack((hit_x, hit_y, hit_z, hit_e, hit_l)).astype(np.float32)

        data = Data(
            x=torch.from_numpy(feats),
            assoc=torch.from_numpy(gids).long(),
            purity=torch.from_numpy(purity).float(),
        )
        return data
