# hgcal_event_clusterer.py
import numpy as np
import matplotlib.pyplot as plt
import CLUEstering as clue
import pandas as pd


class HGCALEventClusterer_2D:
    def __init__(self, dc=1.3, rhoc=0.015, dm=None, pPB=10, backend="cpu serial"):
        self.dc = float(dc)
        self.dm = float(2 * dc if dm is None else dm)
        self.rhoc = float(rhoc)
        self.pPB = int(pPB)
        self.backend = backend

        # per-event inputs
        self.x = self.y = self.z = self.e = None
        self.layers = None
        self.sids = None

        # results (per event)
        self.total_hits = 0
        self.total_layer_clusters = 0
        self.cluster_hit_counts = []
        self.cluster_info = []          # list of dicts (summary per cluster)
        self.cluster_purities = []      # list of floats in [0,1]

        # cached arrays for event-level plotting
        self._cx = self._cy = self._cz = self._cE = self._cSID = None

        # SID→color mapping for legends
        self._sid_to_idx = None
        self._sid_palette_size = 0

    # ---------- I/O ----------
    def read_event(self, x, y, z, e, layers, sids=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.e = np.asarray(e)
        self.layers = np.asarray(layers).astype(int)
        self.sids = None if sids is None else np.asarray(sids)

        n = self.x.size
        if not (n == self.y.size == self.z.size == self.e.size == self.layers.size and
                (self.sids is None or self.sids.size == n)):
            raise ValueError("x, y, z, e, layers (and sids if provided) must have the same length")

        # reset results
        self.total_hits = 0
        self.total_layer_clusters = 0
        self.cluster_hit_counts = []
        self.cluster_info = []
        self.cluster_purities = []
        self._cx = self._cy = self._cz = self._cE = self._cSID = None

        # build stable SID color map (if sids provided)
        if self.sids is not None:
            uniq_sorted = np.sort(np.unique(self.sids))
            self._sid_to_idx = {sid: i for i, sid in enumerate(uniq_sorted)}
            self._sid_palette_size = max(1, len(uniq_sorted))
        else:
            self._sid_to_idx = None
            self._sid_palette_size = 0

    # ---------- Helpers ----------
    @staticmethod
    def _energy_weighted_sid_and_purity(sid_array, weight_array):
        """
        Returns (winning_sid, purity) where:
          winning_sid = SID with the largest total energy in the cluster
          purity      = (energy of winning_sid) / (total energy in cluster)
        If sid_array is empty/None: returns (-1, 0.0)
        """
        if sid_array is None or len(sid_array) == 0:
            return -1, 0.0
        sid_vals = np.asarray(sid_array, dtype=int)
        w = np.asarray(weight_array, dtype=float)
        if sid_vals.size == 0 or w.size == 0 or w.sum() <= 0:
            return -1, 0.0
        uniq, inv = np.unique(sid_vals, return_inverse=True)
        energy_per_sid = np.bincount(inv, weights=w, minlength=uniq.size)
        winner_idx = int(np.argmax(energy_per_sid))
        winner_sid = int(uniq[winner_idx])
        purity = float(energy_per_sid[winner_idx] / (np.sum(w) + 1e-12))
        return winner_sid, purity

    def _sid_to_color_indices(self, sid_array):
        if sid_array is None or self._sid_to_idx is None:
            return np.zeros_like(sid_array if sid_array is not None else np.array([0]))
        return np.vectorize(lambda s: self._sid_to_idx.get(int(s), 0))(sid_array)

    def _add_sid_legend(self, ax, cmap):
        if self._sid_to_idx is None:
            return
        for sid, idx in self._sid_to_idx.items():
            ax.scatter([], [], c=[cmap(idx)], label=f"SID {sid}")
        ax.legend(loc="best", fontsize=8, markerscale=1.5)

    # ---------- Core ----------
    def cluster_event(self, verbose=True):
        if self.x is None:
            raise RuntimeError("Call read_event(...) first")

        cx, cy, cz, cE, cSID = [], [], [], [], []

        for L in np.unique(self.layers):
            mask = (self.layers == L)
            if not np.any(mask):
                continue

            xL = self.x[mask]; yL = self.y[mask]; zL = self.z[mask]
            eL = self.e[mask]; sL = None if self.sids is None else self.sids[mask]
            self.total_hits += xL.size

            data = {"x0": xL.tolist(), "x1": yL.tolist(), "weight": eL.tolist()}
            clust = clue.clusterer(self.dc, self.rhoc, self.dm, self.pPB)
            clust.read_data(data)
            clust.run_clue(self.backend)

            cluster_ids = np.asarray(clust.cluster_ids)
            unique_cids = np.unique(cluster_ids)
            unique_cids = unique_cids[unique_cids != -1]  # drop outliers

            self.total_layer_clusters += unique_cids.size
            if verbose:
                print(f"Layer {int(L):>3}: hits={xL.size:>5}, clusters={unique_cids.size:>3}")

            for cid in unique_cids:
                sel = (cluster_ids == cid)
                nhits = int(np.count_nonzero(sel))
                w = eL[sel]
                Etot = float(np.sum(w))
                if Etot == 0:
                    continue

                cx_i = float(np.sum(xL[sel] * w) / Etot)
                cy_i = float(np.sum(yL[sel] * w) / Etot)
                cz_i = float(np.mean(zL[sel]))

                # Energy-weighted SID and purity
                if sL is not None:
                    shower_id, purity = self._energy_weighted_sid_and_purity(sL[sel], w)
                else:
                    shower_id, purity = -1, 0.0

                self.cluster_hit_counts.append(nhits)
                self.cluster_purities.append(purity)
                self.cluster_info.append({
                    "layer": int(L),
                    "cluster_id": int(cid),
                    "n_hits": nhits,
                    "energy": Etot,
                    "x": cx_i,
                    "y": cy_i,
                    "z": cz_i,
                    "shower_id": int(shower_id),
                    "purity": purity,
                })

                cx.append(cx_i); cy.append(cy_i); cz.append(cz_i)
                cE.append(Etot); cSID.append(shower_id)

        self._cx = np.asarray(cx); self._cy = np.asarray(cy)
        self._cz = np.asarray(cz); self._cE = np.asarray(cE)
        self._cSID = np.asarray(cSID, dtype=int) if len(cSID) else np.asarray([], dtype=int)

        if verbose:
            print(f"\nTotal hits in event: {self.total_hits}")
            print(f"Total layer clusters in event: {self.total_layer_clusters}")

    # ---------- Accessors ----------
    def get_cluster_summary(self, as_dataframe=True):
        if len(self.cluster_info) == 0:
            print("No cluster info available; run cluster_event() first.")
            return None
        return pd.DataFrame(self.cluster_info) if as_dataframe else list(self.cluster_info)

    # ---------- Plots ----------
    def plot_event_3d(self, show_hits=True, show_clusters=True):
        if self.x is None:
            raise RuntimeError("No event loaded. Call read_event(...)")
        if self._cx is None:
            raise RuntimeError("No clustering results. Call cluster_event(...)")

        cmap = plt.get_cmap("tab20", max(1, self._sid_palette_size))

        if show_hits:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            if self.sids is not None:
                cidx_hits = self._sid_to_color_indices(self.sids)
                ax.scatter(self.x, self.y, self.z, c=cidx_hits, s=3, cmap=cmap)
            else:
                ax.scatter(self.x, self.y, self.z, c="blue", s=3, alpha=0.6)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.set_title("All hits (colored by shower_id)" if self.sids is not None else "All hits")
            self._add_sid_legend(ax, cmap)
            plt.tight_layout(); plt.show()

        if show_clusters:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sizes = 20 + 120 * (self._cE / (self._cE.max() + 1e-12)) if self._cE.size else 30
            if self._cSID is not None and self._cSID.size:
                cidx_clu = self._sid_to_color_indices(self._cSID)
                ax.scatter(self._cx, self._cy, self._cz, c=cidx_clu, s=sizes,
                           cmap=cmap, marker="o", edgecolors="k", linewidths=0.3)
            else:
                ax.scatter(self._cx, self._cy, self._cz, c="red", s=sizes,
                           marker="o", edgecolors="k", linewidths=0.3)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.set_title("Layer clusters (colored by shower_id, size ~ energy)")
            self._add_sid_legend(ax, cmap)
            plt.tight_layout(); plt.show()

    def plot_hits_per_cluster_hist(self, bins="auto"):
        if not self.cluster_hit_counts:
            print("No clusters found to histogram. Run cluster_event() first.")
            return
        vals = np.asarray(self.cluster_hit_counts)
        plt.figure(figsize=(7, 5))
        plt.hist(vals, bins=bins)
        plt.xlabel("Hits per layer-cluster"); plt.ylabel("Count of clusters")
        plt.title("Histogram: hits per layer-cluster (all layers)")
        plt.tight_layout(); plt.show()
        print(f"Clusters histogrammed: {vals.size}")
        print(f"Mean hits/cluster: {vals.mean():.2f} | Median: {np.median(vals):.0f} | Max: {vals.max()}")

    def plot_single_layer(self, layer, overlay=False):
        if self.x is None:
            raise RuntimeError("No event loaded. Call read_event(...)")

        layer = int(layer)
        mask = (self.layers == layer)
        if not np.any(mask):
            print(f"No hits found in layer {layer}")
            return

        xL, yL, eL = self.x[mask], self.y[mask], self.e[mask]
        sL = None if self.sids is None else self.sids[mask]

        data = {"x0": xL.tolist(), "x1": yL.tolist(), "weight": eL.tolist()}
        clust = clue.clusterer(self.dc, self.rhoc, self.dm, self.pPB)
        clust.read_data(data)
        clust.run_clue(self.backend)

        cluster_ids = np.asarray(clust.cluster_ids)
        unique_cids = np.unique(cluster_ids)
        unique_cids = unique_cids[unique_cids != -1]
        n_clusters = unique_cids.size
        print(f"Layer {layer} — hits: {xL.size}, clusters: {n_clusters}")

        cx, cy, cE, cSID = [], [], [], []
        for cid in unique_cids:
            sel = (cluster_ids == cid)
            w = eL[sel]
            Etot = float(np.sum(w))
            if Etot == 0:
                continue
            cx.append(float(np.sum(xL[sel] * w) / Etot))
            cy.append(float(np.sum(yL[sel] * w) / Etot))
            cE.append(Etot)
            sid, _ = self._energy_weighted_sid_and_purity(sL[sel], w) if sL is not None else (-1, 0.0)
            cSID.append(sid)

        cx, cy, cE, cSID = map(np.asarray, (cx, cy, cE, cSID))
        sizes = 20 + 120 * (cE / (cE.max() + 1e-12)) if cE.size else 30

        if self._sid_palette_size:
            cmap = plt.get_cmap("tab20", self._sid_palette_size)
        else:
            guess = 1 if sL is None else max(1, len(np.unique(sL)))
            cmap = plt.get_cmap("tab20", guess)

        cidx_hits = self._sid_to_color_indices(sL) if (sL is not None and self._sid_to_idx) else None
        cidx_clu  = self._sid_to_color_indices(cSID) if (cSID.size and self._sid_to_idx) else None

        if overlay:
            fig, ax = plt.subplots(figsize=(6, 6))
            if cidx_hits is not None:
                ax.scatter(xL, yL, c=cidx_hits, s=8, cmap=cmap, alpha=0.6, label="hits")
            else:
                ax.scatter(xL, yL, c="blue", s=8, alpha=0.6, label="hits")
            if cE.size:
                if cidx_clu is not None:
                    ax.scatter(cx, cy, c=cidx_clu, s=sizes, cmap=cmap,
                               marker="o", edgecolors="k", linewidths=0.3, label="clusters")
                else:
                    ax.scatter(cx, cy, c="red", s=sizes, marker="o",
                               edgecolors="k", linewidths=0.3, label="clusters")
            ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(f"Layer {layer} — hits + clusters (2D)")
            self._add_sid_legend(ax, cmap)
            plt.tight_layout(); plt.show()
        else:
            # hits
            fig, ax = plt.subplots(figsize=(6, 6))
            if cidx_hits is not None:
                ax.scatter(xL, yL, c=cidx_hits, s=8, cmap=cmap, alpha=0.6)
            else:
                ax.scatter(xL, yL, c="blue", s=8, alpha=0.6)
            ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(f"Layer {layer} — hits (2D)")
            self._add_sid_legend(ax, cmap)
            plt.tight_layout(); plt.show()

            # clusters
            fig, ax = plt.subplots(figsize=(6, 6))
            if cE.size:
                if cidx_clu is not None:
                    ax.scatter(cx, cy, c=cidx_clu, s=sizes, cmap=cmap,
                               marker="o", edgecolors="k", linewidths=0.3)
                else:
                    ax.scatter(cx, cy, c="red", s=sizes, marker="o",
                               edgecolors="k", linewidths=0.3)
            ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(f"Layer {layer} — clusters (2D)")
            self._add_sid_legend(ax, cmap)
            plt.tight_layout(); plt.show()

    # ---------- NEW: Purity histogram ----------
    def plot_cluster_purity_hist(self, bins=20):
        """
        Purity per cluster = (max energy contributed by any SID) / (total cluster energy).
        Shows histogram and prints the average purity for the event.
        """
        if not self.cluster_purities:
            print("No cluster purities available. Run cluster_event() first.")
            return

        pur = np.asarray(self.cluster_purities, dtype=float)
        plt.figure(figsize=(7, 5))
        plt.hist(pur, bins=bins, range=(0.0, 1.0))
        plt.xlabel("Layer-cluster purity")
        plt.ylabel("Number of clusters")
        plt.title("Histogram: layer-cluster purity (event)")
        plt.tight_layout()
        plt.show()

        print(f"Clusters included: {pur.size}")
        print(f"Average purity: {pur.mean():.3f} | Median: {np.median(pur):.3f} | Min/Max: {pur.min():.3f}/{pur.max():.3f}")
        return pur.mean()
    
    def return_avg_purity(self):
        pur = np.asarray(self.cluster_purities, dtype=float)
        return pur.mean()

    
    def get_shower_barycenter_separation(self, weight_by_energy=False, exclude_sid_values=(-1,)):
        """
        Compute the separation (Euclidean distance) between the barycenters of exactly two showers.

        Parameters
        ----------
        weight_by_energy : bool
            If True, compute energy-weighted barycenters.
            If False, compute simple arithmetic mean over hits.
        exclude_sid_values : tuple or list
            Shower IDs to exclude (default excludes -1).

        Returns
        -------
        float
            Euclidean distance between the two shower barycenters.
        """
        if self.x is None or self.sids is None:
            raise RuntimeError("Need hits and sids loaded. Call read_event(..., sids=...) first.")

        # Exclude unwanted SIDs (e.g. -1)
        sids_arr = np.asarray(self.sids)
        if exclude_sid_values is not None and len(exclude_sid_values) > 0:
            mask_valid = ~np.isin(sids_arr, np.asarray(exclude_sid_values, dtype=int))
        else:
            mask_valid = np.ones_like(sids_arr, dtype=bool)

        xv, yv, zv, ev, sv = self.x[mask_valid], self.y[mask_valid], self.z[mask_valid], self.e[mask_valid], sids_arr[mask_valid]

        uniq_sids = np.unique(sv.astype(int))
        if uniq_sids.size != 2:
            raise ValueError(f"Expected exactly 2 showers, found {uniq_sids.size}: {uniq_sids}")

        barycenters = []
        for sid in uniq_sids:
            m = (sv == sid)
            if weight_by_energy:
                wsum = np.sum(ev[m])
                if wsum <= 0:  # fallback to plain mean
                    xb, yb, zb = np.mean(xv[m]), np.mean(yv[m]), np.mean(zv[m])
                else:
                    xb = np.sum(xv[m] * ev[m]) / wsum
                    yb = np.sum(yv[m] * ev[m]) / wsum
                    zb = np.sum(zv[m] * ev[m]) / wsum
            else:
                xb, yb, zb = np.mean(xv[m]), np.mean(yv[m]), np.mean(zv[m])
            barycenters.append((xb, yb, zb))

        # Compute Euclidean distance between the two barycenters
        b0, b1 = barycenters
        separation = float(np.linalg.norm(np.array(b0) - np.array(b1)))
        return separation


