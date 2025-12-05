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

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            batch = data.x_batch  # (N,)
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)

            losses = []
            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count
                event_embeddings = embeddings[start_idx:end_idx]
                event_group_ids = assoc_tensor[start_idx:end_idx]

                loss_event = contrastive_multi_neg_per_group(
                    event_embeddings,
                    event_group_ids,
                    temperature=0.1,
                )
                losses.append(loss_event)
                start_idx = end_idx

            loss = torch.stack(losses).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def test_new(test_loader, model, device):
    model.eval()
    total_loss = 0.0

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)

        with autocast():
            embeddings, _ = model(data.x, data.x_batch)

            batch = data.x_batch
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)

            losses = []
            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count
                event_embeddings = embeddings[start_idx:end_idx]
                event_group_ids = assoc_tensor[start_idx:end_idx]

                loss_event = contrastive_multi_neg_per_group(
                    event_embeddings,
                    event_group_ids,
                    temperature=0.1,
                )
                losses.append(loss_event)
                start_idx = end_idx

            loss = torch.stack(losses).mean()

        total_loss += loss.item()

    return total_loss / len(test_loader)
