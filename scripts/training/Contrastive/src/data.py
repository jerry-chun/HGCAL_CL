import numpy as np
import subprocess
import tqdm
from tqdm import tqdm
import pandas as pd

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

    def fill_data(self,max_events):
        counter = 0
        arrLens0 = []
        arrLens1 = []

        print("### Loading data")
        for fi,path in enumerate(tqdm(self.raw_paths)):
            
            for array in uproot.iterate(f"{path}:HGCALTBout", ["hit_x", "hit_y", "hit_z", "hit_Edep", "hit_layer", 
                                                               "hit_showerid", "hit_purity"], step_size=self.step_size):
            
                tmp_stsCP_vertices_x = array['hit_x']
                tmp_stsCP_vertices_y = array['hit_y']
                tmp_stsCP_vertices_z = array['hit_z']
                tmp_stsCP_vertices_energy = array['hit_Edep']
                tmp_stsCP_vertices_indexes = array['hit_showerid']
                tmp_stsCP_vertices_purity = array['hit_purity']
                tmp_stsCP_vertices_layer_id = array['hit_layer']

                
                # weighted energies (A LC appears in its caloparticle assignment array as the energy it contributes not full energy)
                #tmp_stsCP_vertices_energy = tmp_stsCP_vertices_energy * tmp_stsCP_vertices_multiplicity
                
                self.step_size = min(self.step_size,len(tmp_stsCP_vertices_x))

                if counter == 0:
                    self.stsCP_vertices_x = tmp_stsCP_vertices_x
                    self.stsCP_vertices_y = tmp_stsCP_vertices_y
                    self.stsCP_vertices_z = tmp_stsCP_vertices_z
                    self.stsCP_vertices_energy = tmp_stsCP_vertices_energy
                    self.stsCP_vertices_layer_id = tmp_stsCP_vertices_layer_id
                    self.stsCP_vertices_indexes = tmp_stsCP_vertices_indexes
                    self.stsCP_stsCP_vertices_purity = tmp_stsCP_vertices_purity
                else:
                    self.stsCP_vertices_x = ak.concatenate((self.stsCP_vertices_x,tmp_stsCP_vertices_x))
                    self.stsCP_vertices_y = ak.concatenate((self.stsCP_vertices_y,tmp_stsCP_vertices_y))
                    self.stsCP_vertices_z = ak.concatenate((self.stsCP_vertices_z,tmp_stsCP_vertices_z))
                    self.stsCP_vertices_energy = ak.concatenate((self.stsCP_vertices_energy,tmp_stsCP_vertices_energy))
                    self.stsCP_vertices_layer_id = ak.concatenate((self.stsCP_vertices_layer_id,tmp_stsCP_vertices_layer_id))
                    self.stsCP_vertices_indexes = ak.concatenate((self.stsCP_vertices_indexes,tmp_stsCP_vertices_indexes))
                    self.stsCP_stsCP_vertices_purity = ak.concatenate((self.stsCP_stsCP_vertices_purity, tmp_stsCP_vertices_purity))

                #print(len(self.stsCP_vertices_x))
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

