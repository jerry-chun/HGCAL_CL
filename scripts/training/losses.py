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

def nt_xent_one_pos_one_neg(z, pos_idx, neg_idx, temperature=0.05):
    z = F.normalize(z, dim=1)

    z_pos = z[pos_idx]
    z_neg = z[neg_idx]

    s_pos = (z * z_pos).sum(dim=1) / temperature
    s_neg = (z * z_neg).sum(dim=1) / temperature

    exp_pos = torch.exp(s_pos)
    exp_neg = torch.exp(s_neg)
    loss = -torch.log(exp_pos / (exp_pos + exp_neg))
    return loss.mean()


LOSS_ZOO = {
    "contrastive_pairs": contrastive_pairs,
    "nt_xent_one_pos_one_neg": nt_xent_one_pos_one_neg
}
