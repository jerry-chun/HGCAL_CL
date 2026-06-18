## validation

Loads a trained model, runs it on test data, clusters the output, and compares the resulting clusters against the simulated truth (caloparticles).

- `src/data_loader/`: reads the raw ROOT files into memory.
- `src/models/`: model definitions/loaders for both contrastive and OC checkpoints.
- `src/clustering/`: turns model output (embeddings or OC features) into clusters.
- `src/metrics/`: builds the reco-to-sim dataframe and computes performance metrics.
- `scripts/`: entry points — `DF_maker.py` builds the dataframe, `metrics_calculate.py` computes metrics, `Grid_Agglom_optimiser.py` tunes clustering hyperparameters, `Embeddings_metrics.py` is secondary (see below).
- `notebooks/`: plotting of final results and embeddings.
- `run.sh` / `submit.sh` / `submit.sub`: batch submission.

### Metrics

The metrics that matter are:
- **Purity** — fraction of hits in a reco cluster that genuinely belong to the same caloparticle.
- **Efficiency** — fraction of a caloparticle's hits that are correctly captured in a reco cluster.
- **Number ratio** — ratio of the number of reco clusters to the number of true caloparticles.

These three give the clearest picture of clustering quality. Embedding-space metrics (e.g. in `Embeddings_metrics.py` / `Embeddings_plot.ipynb`) are secondary diagnostics, not the main thing to optimise for.
