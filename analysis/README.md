## analysis

- `training/`: trains the clustering models.
  - `contrastive/`: our main method: embeds hits with a contrastive loss, then clusters in embedding space.
  - `objectcondensation/`: object condensation (OC) approach, kept as a comparison point. Not the focus going forward, may not be worth maintaining long-term.
- `validation/`: runs trained models on test data, builds dataframes of reco-to-sim associations, and computes performance metrics.

See `validation/README.md` for details on the validation side.

### Improvements

- **Switch from ROOT to HDF5**: data is currently read directly from ROOT files via `uproot` on every run, which is slow. Preprocessing once into HDF5 would make loading much faster.
- **Batched data loading**: data is currently loaded fully into memory (via `awkward`/`uproot.iterate` then concatenated). Loading in batches instead would let this scale to much larger datasets.
- will make iterating much quicker



