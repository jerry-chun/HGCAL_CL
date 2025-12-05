import pandas as pd 
import numpy as np


def calc_purity(df, threshold=0.2):
    total_reco = (
        df[['event_id', 'reco_id']]
        .drop_duplicates()
        .shape[0]
    )
    associated_reco = (
        df.loc[df['RtS'] < threshold, ['event_id', 'reco_id']]
        .drop_duplicates()
        .shape[0]
    )
    purity = associated_reco / total_reco if total_reco > 0 else 0.0
    return purity

def calc_efficiency(df, threshold=0.5):
    total_cp = (
        df[['event_id', 'cp_id']]
        .drop_duplicates()
        .shape[0]
    )
    associated_cp = (
        df.loc[(df['shared_energy']/df['cp_energy']) > threshold, ['event_id', 'cp_id']]
        .drop_duplicates()
        .shape[0]
    )
    efficiency = associated_cp / total_cp if total_cp > 0 else 0.0
    return efficiency

def calc_response(df):
    # 1) For each (event_id, cp_id) pick the row with max shared_energy
    idx_best = df.groupby(['event_id', 'cp_id'])['shared_energy'].idxmax()
    best_matches = df.loc[idx_best].copy()

    # 2) Compute response = reco_energy / cp_energy
    best_matches = best_matches[best_matches['cp_energy'] > 0]
    best_matches['response'] = (
        best_matches['reco_energy'] / best_matches['cp_energy']
    )

    # 3) Mean and std of response
    mean_resp = best_matches['response'].mean()
    std_resp  = best_matches['response'].std(ddof=1) 

    return mean_resp, std_resp

def calc_ratio(df):
    total_reco = (
        df[['event_id', 'reco_id']]
        .drop_duplicates()
        .shape[0]
    )
    
    total_cp = (
        df[['event_id', 'cp_id']]
        .drop_duplicates()
        .shape[0]
    )
    
    ratio = total_reco / total_cp if total_cp > 0 else 0.0
    return ratio
