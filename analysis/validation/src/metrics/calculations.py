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

def calc_merge_rate(df, threshold=0.2):
    # Total number of reconstructed objects
    total_reco = (
        df[['event_id', 'reco_id']]
        .drop_duplicates()
        .shape[0]
    )

    if total_reco == 0:
        return 0.0

    # Count how many CPs each reco object is associated with
    reco_cp_counts = (
        df.loc[df['RtS'] < threshold, ['event_id', 'reco_id', 'cp_id']]
        .drop_duplicates()
        .groupby(['event_id', 'reco_id'])['cp_id']
        .nunique()
    )

    # Reco objects associated with more than one CP
    merged_reco = (reco_cp_counts > 1).sum()

    merge_rate = merged_reco / total_reco
    return merge_rate

def calc_response(df):
    idx_best = df.groupby(['event_id', 'cp_id'])['shared_energy'].idxmax()
    best_matches = df.loc[idx_best].copy()

    best_matches = best_matches[best_matches['cp_energy'] > 0]
    best_matches['response'] = (
        best_matches['reco_energy'] / best_matches['cp_energy']
    )

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
