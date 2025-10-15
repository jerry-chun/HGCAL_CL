This folder takes the hit level data files and applies CLUE2D across all events and files.

It extends the CLUEstering package (https://github.com/cms-patatrack/CLUEstering) to apply across all detector layers within an event and then across all events in individual files.

A submit file based on HTCondor is provided to run across multiple events efficiently.

The main packages include:
    
### `HGCALEventClusterer_2D`

Per-layer 2D CLUE clustering for a single HGCAL event.  
Runs CLUE on each detector layer and computes energy-weighted cluster centroids, energies, and purities.

**Init**
```
HGCALEventClusterer_2D(
    dc=1.3,          # core distance
    rhoc=0.015,      # density threshold
    dm=None,         # max distance (defaults to 2*dc)
    pPB=10,          # points per bucket
    backend="cpu serial"
)
```
**Inputs**
```
read_event(
    x, y, z, e, layers, sids=None
)
```
All arrays must have the same length.
sids = optional shower IDs per hit (for purity and coloring).
**Main method**
```
cluster_event(verbose=True)
```
Runs CLUE clustering per layer and stores results in cluster_info, cluster_purities, etc.
**Outputs**

get_cluster_summary() → per-cluster table (layer, energy, centroid, purity, etc.)
return_avg_purity() → mean cluster purity
**Optional plots**
plot_event_3d(), plot_single_layer(), plot_hits_per_cluster_hist(), plot_cluster_purity_hist()
Deps: numpy, matplotlib, pandas, CLUEstering



### LC_writer.py

**Description**
Runs CLUE2D clustering on each event in a ROOT file and writes a Layer-Cluster (LC) friend tree.

**Usage**
python LC_writer.py -i <input.root> [-o <output.root>] [--dc 0.013 --rhoc 0.05 --dm 0.026 --pPB 10 --backend "cpu serial" --verbose]

**Arguments**
- -i, --input    : Input ROOT file
- -o, --output   : Output ROOT file (default: <input>_LC.root)
- -t, --tree     : Input tree name (default: HGCALTBout;1)
- --dc           : CLUE core distance (default: 0.013)
- --rhoc         : CLUE density threshold (default: 0.05)
- --dm           : CLUE max distance (default: 0.026)
- --pPB          : CLUE points per bucket (default: 10)
- --backend      : CLUE backend string (default: "cpu serial")
- --verbose      : Print per-event progress

**Inputs (from ROOT tree)**
- hit_x, hit_y, hit_z
- hit_Edep
- hit_layer
- hit_showerid

**Processing**
- Instantiates HGCALEventClusterer_2D with the provided CLUE parameters.
- For each event: runs per-layer clustering and collects per-cluster information:
  layer, cluster_id, n_hits, energy, x, y, z, shower_id, purity.

**Outputs**
- Writes a friend tree named "LC" with branches:
  lc_layer, lc_cluster_id, lc_n_hits, lc_energy, lc_x, lc_y, lc_z, lc_shower_id, lc_purity

Dependencies
- numpy, awkward, uproot, HGCALEventClusterer_2D




