import numpy as np
import subprocess
import tqdm
from tqdm import tqdm

import os
import os.path as osp

import glob

import h5py
import uproot

import torch
from torch import nn


from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import awkward as ak
import random

#singularity shell --bind /afs/cern.ch/user/p/pkakhand/public/CL/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif

#singularity shell --bind /eos/project/c/contrast/public/solar/  /afs/cern.ch/user/p/pkakhand/geometricdl.sif
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103cuda/x86_64-centos9-gcc11-opt/setup.sh


class CCV1(Dataset):
    r'''
        input: layer clusters

    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp = 'train'):
        super(CCV1, self).__init__(root, transform)
        self.step_size = 500
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self, max_events):
        chunks_x, chunks_y, chunks_z = [], [], []
        chunks_e, chunks_l, chunks_id, chunks_p = [], [], [], []
        total = 0

        print("### Loading data")
        for fi, path in enumerate(tqdm(self.raw_paths)):
            for array in uproot.iterate(f"{path}:HGCALTBout",
                                        ["hit_x", "hit_y", "hit_z", "hit_Edep",
                                         "hit_layer", "hit_showerid", "hit_purity"],
                                        step_size=self.step_size):
                chunks_x.append(array['hit_x'])
                chunks_y.append(array['hit_y'])
                chunks_z.append(array['hit_z'])
                chunks_e.append(array['hit_Edep'])
                chunks_l.append(array['hit_layer'])
                chunks_id.append(array['hit_showerid'])
                chunks_p.append(array['hit_purity'])
                total += len(array['hit_x'])
                if total >= max_events:
                    print(f"Reached {max_events}!")
                    break
            if total >= max_events:
                break

        self.stsCP_vertices_x              = ak.concatenate(chunks_x)
        self.stsCP_vertices_y              = ak.concatenate(chunks_y)
        self.stsCP_vertices_z              = ak.concatenate(chunks_z)
        self.stsCP_vertices_energy         = ak.concatenate(chunks_e)
        self.stsCP_vertices_layer_id       = ak.concatenate(chunks_l)
        self.stsCP_vertices_indexes        = ak.concatenate(chunks_id)
        self.stsCP_stsCP_vertices_purity   = ak.concatenate(chunks_p)
     
            
            
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
        edge_index = torch.empty((2,0), dtype=torch.long)
 
        lc_x = np.expand_dims(np.asarray(self.stsCP_vertices_x[idx]), axis = 1)
        lc_y = np.expand_dims(np.asarray(self.stsCP_vertices_y[idx]), axis = 1)
        lc_z = np.expand_dims(np.asarray(self.stsCP_vertices_z[idx]), axis = 1)
        lc_e = np.expand_dims(np.asarray(self.stsCP_vertices_energy[idx]), axis = 1)
        lc_layer_id = np.expand_dims(np.asarray(self.stsCP_vertices_layer_id[idx]), axis = 1)


        lc_indexes = np.asarray(self.stsCP_vertices_indexes[idx])
        purity = np.asarray(self.stsCP_stsCP_vertices_purity[idx])


        flat_lc_feats = np.concatenate((lc_x, lc_y, lc_z, lc_e, lc_layer_id),axis=-1)        

        x = torch.from_numpy(flat_lc_feats).float()


        
        return Data(
            x = x,
            assoc = torch.from_numpy(lc_indexes).long(),
            purity = torch.from_numpy(purity).float(),
            )

