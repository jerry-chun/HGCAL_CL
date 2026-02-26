import numpy as np
import os.path as osp
import uproot
import awkward as ak
import torch
from torch_geometric.data import Data, Dataset


class CCV1(Dataset):
    """
    One-file dataset for HGCAL layer clusters.

    Only keeps events with len(PrimaryEnergies) in keep_groups.
    """

    def __init__(self, root_file, transform=None, max_events=1e8, inp="train", keep_groups=(6, 7, 8,9)):
        self.root_file = osp.abspath(root_file)
        self.step_size = 10
        self.inp = inp
        self.max_events = int(max_events)
        self.keep_groups = tuple(int(k) for k in keep_groups)

        super().__init__(root=".", transform=transform)
        self.fill_data(self.max_events)

    def download(self):
        return

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def fill_data(self, max_events):
        if not osp.exists(self.root_file):
            raise FileNotFoundError(f"ROOT file not found: {self.root_file}")

        print(f"### Loading ROOT file: {self.root_file}")
        print(f"### Keeping only events with len(PrimaryEnergies) in {self.keep_groups}")

        # We'll build these up only from filtered events
        self.stsCP_vertices_x = None
        self.stsCP_vertices_y = None
        self.stsCP_vertices_z = None
        self.stsCP_vertices_energy = None
        self.stsCP_vertices_layer_id = None
        self.stsCP_vertices_indexes = None
        self.stsCP_vertices_purity = None
        self.PE = None

        loaded = 0  # number of kept events loaded so far

        for array in uproot.iterate(
            f"{self.root_file}:HGCALTBout",
            [
                "hit_x",
                "hit_y",
                "hit_z",
                "hit_Edep",
                "hit_layer",
                "hit_showerid",
                "hit_purity",
                "PrimaryEnergies",
            ],
            step_size=self.step_size,
        ):
            # ---- per-chunk mask (BEFORE concatenation) ----
            pe_counts = ak.num(array["PrimaryEnergies"], axis=1)          # (n_events,)
            keep_mask = np.isin(ak.to_numpy(pe_counts), self.keep_groups) # numpy bool mask

            # If nothing in this chunk matches, skip cheaply
            n_keep_chunk = int(ak.sum(keep_mask))
            if n_keep_chunk == 0:
                continue

            # Apply mask to all branches in this chunk
            x_f      = array["hit_x"][keep_mask]
            y_f      = array["hit_y"][keep_mask]
            z_f      = array["hit_z"][keep_mask]
            e_f      = array["hit_Edep"][keep_mask]
            layer_f  = array["hit_layer"][keep_mask]
            idx_f    = array["hit_showerid"][keep_mask]
            pur_f    = array["hit_purity"][keep_mask]
            pe_f     = array["PrimaryEnergies"][keep_mask]

            # If we only need some of them to reach max_events, slice here
            remaining = max_events - loaded
            if remaining <= 0:
                break
            if n_keep_chunk > remaining:
                sl = slice(0, remaining)
                x_f, y_f, z_f = x_f[sl], y_f[sl], z_f[sl]
                e_f, layer_f  = e_f[sl], layer_f[sl]
                idx_f, pur_f  = idx_f[sl], pur_f[sl]
                pe_f          = pe_f[sl]
                n_keep_chunk = remaining

            # ---- concatenate only filtered events ----
            if self.stsCP_vertices_x is None:
                self.stsCP_vertices_x = x_f
                self.stsCP_vertices_y = y_f
                self.stsCP_vertices_z = z_f
                self.stsCP_vertices_energy = e_f
                self.stsCP_vertices_layer_id = layer_f
                self.stsCP_vertices_indexes = idx_f
                self.stsCP_vertices_purity = pur_f
                self.PE = pe_f
            else:
                self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x, x_f))
                self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y, y_f))
                self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z, z_f))
                self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy, e_f))
                self.stsCP_vertices_layer_id = ak.concatenate((self.stsCP_vertices_layer_id, layer_f))
                self.stsCP_vertices_indexes = ak.concatenate((self.stsCP_vertices_indexes, idx_f))
                self.stsCP_vertices_purity = ak.concatenate((self.stsCP_vertices_purity, pur_f))
                self.PE = ak.concatenate((self.PE, pe_f))

            loaded += n_keep_chunk

        if self.stsCP_vertices_x is None:
            # No events matched
            self.stsCP_vertices_x = ak.Array([])
            self.stsCP_vertices_y = ak.Array([])
            self.stsCP_vertices_z = ak.Array([])
            self.stsCP_vertices_energy = ak.Array([])
            self.stsCP_vertices_layer_id = ak.Array([])
            self.stsCP_vertices_indexes = ak.Array([])
            self.stsCP_vertices_purity = ak.Array([])
            self.PE = ak.Array([])

        print(f"Loaded {len(self.stsCP_vertices_x)} events (post-filter)")

    def len(self):
        return len(self.stsCP_vertices_x)

    def get(self, idx):
        lc_x = np.asarray(self.stsCP_vertices_x[idx])[:, None]
        lc_y = np.asarray(self.stsCP_vertices_y[idx])[:, None]
        lc_z = np.asarray(self.stsCP_vertices_z[idx])[:, None]
        lc_e = np.asarray(self.stsCP_vertices_energy[idx])[:, None]
        lc_layer = np.asarray(self.stsCP_vertices_layer_id[idx])[:, None]

        x = torch.from_numpy(np.concatenate((lc_x, lc_y, lc_z, lc_e, lc_layer), axis=-1)).float()

        return Data(
            x=x,
            assoc=torch.from_numpy(np.asarray(self.stsCP_vertices_indexes[idx])).long(),
            purity=torch.from_numpy(np.asarray(self.stsCP_vertices_purity[idx])).float(),
            PrimaryEnergies=torch.from_numpy(np.asarray(self.PE[idx])).float(),
        )