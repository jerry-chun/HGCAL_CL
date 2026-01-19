import numpy as np
import glob
import os
import os.path as osp
import uproot
import awkward as ak
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class CCV1(Dataset):
    """
    One-file dataset for HGCAL layer clusters.
    """

    def __init__(self, root_file, transform=None, max_events=1e8, inp="train"):
        """
        Parameters
        ----------
        root_file : str
            Path to a single ROOT file
        """
        self.root_file = osp.abspath(root_file)
        self.step_size = 10
        self.inp = inp
        self.max_events = max_events

        # Dummy root to satisfy PyG Dataset
        super().__init__(root=".", transform=transform)

        self.fill_data(max_events)

    # ---- Disable PyG download logic completely ----
    def download(self):
        return

    # ---- Required by PyG but unused ----
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    # ---- Core loader ----
    def fill_data(self, max_events):
        if not osp.exists(self.root_file):
            raise FileNotFoundError(f"ROOT file not found: {self.root_file}")

        print(f"### Loading ROOT file: {self.root_file}")

        counter = 0

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
            if counter == 0:
                self.stsCP_vertices_x = array["hit_x"]
                self.stsCP_vertices_y = array["hit_y"]
                self.stsCP_vertices_z = array["hit_z"]
                self.stsCP_vertices_energy = array["hit_Edep"]
                self.stsCP_vertices_layer_id = array["hit_layer"]
                self.stsCP_vertices_indexes = array["hit_showerid"]
                self.stsCP_vertices_purity = array["hit_purity"]
                self.PE = array["PrimaryEnergies"]
            else:
                self.stsCP_vertices_x = ak.concatenate(
                    (self.stsCP_vertices_x, array["hit_x"])
                )
                self.stsCP_vertices_y = ak.concatenate(
                    (self.stsCP_vertices_y, array["hit_y"])
                )
                self.stsCP_vertices_z = ak.concatenate(
                    (self.stsCP_vertices_z, array["hit_z"])
                )
                self.stsCP_vertices_energy = ak.concatenate(
                    (self.stsCP_vertices_energy, array["hit_Edep"])
                )
                self.stsCP_vertices_layer_id = ak.concatenate(
                    (self.stsCP_vertices_layer_id, array["hit_layer"])
                )
                self.stsCP_vertices_indexes = ak.concatenate(
                    (self.stsCP_vertices_indexes, array["hit_showerid"])
                )
                self.stsCP_vertices_purity = ak.concatenate(
                    (self.stsCP_vertices_purity, array["hit_purity"])
                )
                self.PE = ak.concatenate((self.PE, array["PrimaryEnergies"]))

            counter += 1
            if len(self.stsCP_vertices_x) >= max_events:
                break

        print(f"Loaded {len(self.stsCP_vertices_x)} events")

    # ---- PyG interface ----
    def len(self):
        return len(self.stsCP_vertices_x)

    def get(self, idx):
        lc_x = np.asarray(self.stsCP_vertices_x[idx])[:, None]
        lc_y = np.asarray(self.stsCP_vertices_y[idx])[:, None]
        lc_z = np.asarray(self.stsCP_vertices_z[idx])[:, None]
        lc_e = np.asarray(self.stsCP_vertices_energy[idx])[:, None]
        lc_layer = np.asarray(self.stsCP_vertices_layer_id[idx])[:, None]

        x = torch.from_numpy(
            np.concatenate((lc_x, lc_y, lc_z, lc_e, lc_layer), axis=-1)
        ).float()

        return Data(
            x=x,
            assoc=torch.from_numpy(
                np.asarray(self.stsCP_vertices_indexes[idx])
            ).long(),
            purity=torch.from_numpy(
                np.asarray(self.stsCP_vertices_purity[idx])
            ).float(),
            PrimaryEnergies=torch.from_numpy(
                np.asarray(self.PE[idx])
            ).float(),
        )

