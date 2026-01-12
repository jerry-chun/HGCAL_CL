import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from src.losses import contrastive_multi_neg_per_group


def train_new(train_loader, model, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)
        purity_tensor = data.purity.to(device=data.x.device, dtype=torch.long)  # unused for now

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            batch = data.x_batch  # (N,)
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)  # (num_events,)

            losses = []
            event_node_counts = []  # counts only for events we actually use

            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count
                event_embeddings = embeddings[start_idx:end_idx]
                event_group_ids = assoc_tensor[start_idx:end_idx]
                event_purity = purity_tensor[start_idx:end_idx]

                # ---- SKIP single-shower events ----
                # if only one unique group id, nothing contrastive to learn
                num_groups = torch.unique(event_group_ids).numel()

                if num_groups <= 1 or num_groups >= 11:
                    start_idx = end_idx
                    continue

                # -----------------------------------

                loss_event = contrastive_multi_neg_per_group(
                    event_embeddings,
                    event_group_ids,
                )
                losses.append(loss_event)
                event_node_counts.append(count)

                start_idx = end_idx

            if len(losses) == 0:
                # all events in this batch were single-shower → skip batch
                continue

            # --- node-weighted reduction over *kept* events ---
            losses = torch.stack(losses)            # (num_kept_events,)
            weights = torch.tensor(
                event_node_counts, device=losses.device, dtype=torch.float32
            )
            weights = weights / weights.sum()
            loss = (losses * weights).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


@torch.no_grad()
def test_new(test_loader, model, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)
        purity_tensor = data.purity.to(device=data.x.device, dtype=torch.long)  # unused

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            batch = data.x_batch
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)

            losses = []
            event_node_counts = []

            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count
                event_embeddings = embeddings[start_idx:end_idx]
                event_group_ids = assoc_tensor[start_idx:end_idx]
                event_purity = purity_tensor[start_idx:end_idx]

                # ---- SKIP single-shower events ----
                num_groups = torch.unique(event_group_ids).numel()

                if num_groups <= 1 or num_groups >= 11:
                    start_idx = end_idx
                    continue
                # -----------------------------------

                loss_event = contrastive_multi_neg_per_group(
                    event_embeddings,
                    event_group_ids,
                )
                losses.append(loss_event)
                event_node_counts.append(count)

                start_idx = end_idx

            if len(losses) == 0:
                # no usable events in this batch
                continue

            # --- node-weighted reduction over *kept* events ---
            losses = torch.stack(losses)
            weights = torch.tensor(
                event_node_counts, device=losses.device, dtype=torch.float32
            )
            weights = weights / weights.sum()
            loss = (losses * weights).sum()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches
