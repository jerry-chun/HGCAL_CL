# losses.py
import torch
import torch.nn.functional as F

def contrastive_pairs(z, pos_idx, neg_idx, temperature=0.05):
    z = F.normalize(z, dim=1)
    z_pos = z[pos_idx]
    z_neg = z[neg_idx]
    s_pos = (z * z_pos).sum(dim=1)
    s_neg = (z * z_neg).sum(dim=1)
    logits = torch.stack([s_pos, s_neg], dim=1) / temperature
    labels = torch.zeros(len(z), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels, reduction="mean")

LOSS_ZOO = {
    "contrastive_pairs": contrastive_pairs,
}
