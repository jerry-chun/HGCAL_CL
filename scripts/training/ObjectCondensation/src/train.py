# train.py

from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler

from src.losses import object_condensation_loss


def train_oc(
    train_loader,
    model,
    optimizer,
    device,
    scaler: GradScaler,
    *,
    q_min: float = 0.1,
    noise_label: int = -1,
    s_att: float = 1.0,
    s_rep: float = 1.0,
    s_coward: float = 1.0,
    s_noise: float = 1.0,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in tqdm(train_loader, desc="Training (OC)"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            cluster_coords, beta, prop_pred, _ = model(data.x, data.x_batch)

            batch = data.x_batch
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)

            losses = []
            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count

                event_cluster = cluster_coords[start_idx:end_idx]
                event_beta = beta[start_idx:end_idx]
                event_groups = assoc_tensor[start_idx:end_idx]

                # skip single
                num_groups = torch.unique(event_groups).numel()

                if num_groups <= 1 or num_groups >= 11:
                    start_idx = end_idx
                    continue

                loss_event = object_condensation_loss(
                    beta=event_beta,
                    cluster_coords=event_cluster,
                    group_ids=event_groups,
                    q_min=q_min,
                    noise_label=noise_label,
                    weights=None,
                    s_att=s_att,
                    s_rep=s_rep,
                    s_coward=s_coward,
                    s_noise=s_noise,
                )
                losses.append(loss_event)
                start_idx = end_idx

            if len(losses) == 0:
                # no usable events in this batch
                continue

            loss = torch.stack(losses).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


@torch.no_grad()
def test_oc(
    val_loader,
    model,
    device,
    *,
    q_min: float = 0.1,
    noise_label: int = -1,
    s_att: float = 1.0,
    s_rep: float = 1.0,
    s_coward: float = 1.0,
    s_noise: float = 1.0,
):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for data in tqdm(val_loader, desc="Validation (OC)"):
        data = data.to(device, non_blocking=True)
        assoc_tensor = data.assoc.to(device=data.x.device, dtype=torch.long)

        with autocast():
            cluster_coords, beta, prop_pred, _ = model(data.x, data.x_batch)

            batch = data.x_batch
            num_events = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_events)

            losses = []
            start_idx = 0
            for count in counts.tolist():
                end_idx = start_idx + count

                event_cluster = cluster_coords[start_idx:end_idx]
                event_beta = beta[start_idx:end_idx]
                event_groups = assoc_tensor[start_idx:end_idx]

                num_groups = torch.unique(event_groups).numel()

                if num_groups <= 1 or num_groups >= 11:
                    start_idx = end_idx
                    continue

                loss_event = object_condensation_loss(
                    beta=event_beta,
                    cluster_coords=event_cluster,
                    group_ids=event_groups,
                    q_min=q_min,
                    noise_label=noise_label,
                    weights=None,
                    s_att=s_att,
                    s_rep=s_rep,
                    s_coward=s_coward,
                    s_noise=s_noise,
                )
                losses.append(loss_event)
                start_idx = end_idx

            if len(losses) == 0:
                # no usable events in this batch
                continue

            loss = torch.stack(losses).mean()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches
