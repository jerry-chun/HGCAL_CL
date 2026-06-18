import pandas as pd 
import awkward as ak
import torch
import numpy as np
import tqdm
import itertools
from .reco_to_sim import reco_to_sim
def build_dataframe(reconstructed_label, loader):
    num_reco = len(reconstructed_label)
    rows = []
    for i, data in enumerate(loader):
        if i >= num_reco:
            break
        CP_ids = data.assoc
        PrimaryEnergies = data.PrimaryEnergies
        reco_ids = reconstructed_label[i]

        hit_energy = np.array(data.x[:,3])
        hit_purity = np.ones_like(hit_energy)
        unique_cp_ids = np.unique(CP_ids)
        unique_reco_ids = np.unique(reco_ids)

        for rid in unique_reco_ids:
            if rid == -1:
                continue
            rmask = np.array((reco_ids == rid))
            for cid in unique_cp_ids:
                cmask = np.array((CP_ids == cid))
                PE = PrimaryEnergies[cid]
                
                cp_energy = hit_energy[cmask].sum()
                reco_energy = hit_energy[rmask].sum()
                shared_energy = hit_energy[rmask & cmask].sum()
                RtS = reco_to_sim(hit_energy, rmask, cmask, hit_purity)
                rows.append({
                    'event_id': i,
                    'cp_id': cid,
                    'reco_id': rid,
                    'cp_energy': cp_energy,
                    'reco_energy': reco_energy,
                    'shared_energy': shared_energy,
                    'RtS': RtS,
                    'PrimaryEnergy' : PE.item()
                })
    df = pd.DataFrame(rows).sort_values(['event_id', 'cp_id', 'reco_id']).reset_index(drop=True)
    return df


    return None

