# data.py (only showing changed/important bits)
import numpy as np
import awkward as ak
import uproot
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

import os
import os.path as osp

import glob

import h5py
import uproot

def _sample_pos_neg_numpy(groups, rng):
    N = len(groups)
    x_pe = np.arange(N, dtype=np.int64)
    x_ne = np.arange(N, dtype=np.int64)
    uniq = np.unique(groups)
    idxs_all = np.arange(N)
    by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    for _, idxs in by_g.items():
        m = len(idxs)
        if m > 1:
            r = rng.integers(0, m - 1, size=m)
            pos = np.arange(m)
            r = r + (r >= pos)
            x_pe[idxs] = idxs[r]
    for _, idxs in by_g.items():
        comp = np.setdiff1d(idxs_all, idxs, assume_unique=True)
        if len(comp) > 0:
            partners = rng.choice(comp, size=len(idxs), replace=True)
            x_ne[idxs] = partners
    return x_pe, x_ne

class CCV1(Dataset):
    url = "/dummy/"
    def __init__(self, root, split="train", step_size=10, max_events=1e8, transform=None):
        super().__init__(root, transform)
        self.split = split
        self.step_size = step_size
        self.max_events = max_events
        self._epoch = 0
        self.fill_data(max_events)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def fill_data(self,max_events):
        counter = 0
        arrLens0 = []
        arrLens1 = []

        print("### Loading data")
        for fi,path in enumerate(tqdm(self.raw_paths)):



            for array in uproot.iterate(f"{path}:{'LC'}", ["lc_layer", "lc_x", "lc_y", "lc_z", "lc_energy", "lc_n_hits", "lc_shower_id"], step_size=self.step_size):
                
                tmp_stsCP_vertices_x = array['lc_x']
                tmp_stsCP_vertices_y = array['lc_y'] 
                tmp_stsCP_vertices_z = array['lc_z']
                tmp_stsCP_vertices_energy = array['lc_energy']
                tmp_stsCP_vertices_noh = array['lc_n_hits']
                tmp_stsCP_vertices_shower_id = array['lc_shower_id']
                tmp_stsCP_vertices_layer_id = array['lc_layer']
                self.step_size = min(self.step_size,len(tmp_stsCP_vertices_x))

                
                skim_mask_noh = tmp_stsCP_vertices_noh > 1.0
                tmp_stsCP_vertices_x = tmp_stsCP_vertices_x[skim_mask_noh]
                tmp_stsCP_vertices_y = tmp_stsCP_vertices_y[skim_mask_noh]
                tmp_stsCP_vertices_z = tmp_stsCP_vertices_z[skim_mask_noh]
                tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy[skim_mask_noh]
                tmp_stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id[skim_mask_noh]
                tmp_stsCP_vertices_noh = tmp_stsCP_vertices_noh[skim_mask_noh]
                tmp_stsCP_vertices_shower_id = tmp_stsCP_vertices_shower_id[skim_mask_noh]

                if counter == 0:
                    self.stsCP_vertices_x = tmp_stsCP_vertices_x
                    self.stsCP_vertices_y = tmp_stsCP_vertices_y
                    self.stsCP_vertices_z = tmp_stsCP_vertices_z
                    self.stsCP_vertices_energy = tmp_stsCP_vertices_energy
                    self.stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id
                    self.stsCP_vertices_noh = tmp_stsCP_vertices_noh
                    self.stsCP_vertices_shower_id = tmp_stsCP_vertices_shower_id
                else:
                    self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x, tmp_stsCP_vertices_x))
                    self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y, tmp_stsCP_vertices_y))
                    self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z, tmp_stsCP_vertices_z))
                    self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy, tmp_stsCP_vertices_energy))
                    self.stsCP_vertices_layer_id = ak.concatenate((self.stsCP_vertices_layer_id, tmp_stsCP_vertices_layer_id))
                    self.stsCP_vertices_noh = ak.concatenate((self.stsCP_vertices_noh, tmp_stsCP_vertices_noh))
                    self.stsCP_vertices_shower_id = ak.concatenate((self.stsCP_vertices_shower_id, tmp_stsCP_vertices_shower_id))
                
                counter += 1
                if len(self.stsCP_vertices_x) > max_events:
                    print(f"Reached {max_events}!")
                    break
            if len(self.stsCP_vertices_x) > max_events:
                break
                
    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return len(self.stsCP_vertices_x)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        
        #raw_files = [osp.join(self.raw_dir, 'step3_NTUPLE.root')]

        return raw_files

    @property
    def processed_file_names(self):
        return []

    def get(self, idx):
        lc_x = np.array(self.stsCP_vertices_x[idx])
        lc_y = np.array(self.stsCP_vertices_y[idx])
        lc_z = np.array(self.stsCP_vertices_z[idx])
        lc_e = np.array(self.stsCP_vertices_energy[idx])
        lc_l = np.array(self.stsCP_vertices_layer_id[idx])
        lc_n = np.array(self.stsCP_vertices_noh[idx])
        gids = np.array(self.stsCP_vertices_shower_id[idx])

        feats = np.column_stack((lc_x, lc_y, lc_z, lc_e, lc_l, lc_n)).astype(np.float32)
        x = torch.from_numpy(feats)

        rng = np.random.default_rng(seed=(self._epoch * 1_000_003 + idx))
        x_pe, x_ne = _sample_pos_neg_numpy(gids, rng)

        return Data(
            x=x,
            assoc=torch.from_numpy(gids),
            x_pe=torch.from_numpy(x_pe),
            x_ne=torch.from_numpy(x_ne),
        )
