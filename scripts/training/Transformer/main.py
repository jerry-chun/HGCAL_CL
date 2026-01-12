import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import CCV1
from src.train import train_new, test_new
from src.model import Net
import csv

from torch.cuda.amp import autocast, GradScaler


# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")

ipath = "/vols/cms/mm1221/Independent/Data/EM_2_20/train/"
vpath = "/vols/cms/mm1221/Independent/Data/EM_2_20/val/"

data_train = CCV1(root=ipath, split="train", step_size=1000, max_events=700000)
data_val   = CCV1(root=vpath,   split="val",   step_size=1000, max_events=200000)

model = Net(
    hidden_dim=64,
    num_layers=3,
    dropout=0.01,
    contrastive_dim=16,
    k=24,
    num_heads=4,
    edge_hidden_dim=16,
    edge_out_dim=16,
).to(device)

BS = 32

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

train_loader = DataLoader(
    data_train,
    batch_size=BS,
    shuffle=True,
    follow_batch=['x'],
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    data_val,
    batch_size=BS,
    shuffle=False,
    follow_batch=['x'],
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# New output directory name to reflect hits & cubesim
output_dir = '/vols/cms/mm1221/Independent/Transformer/runs/EM_2_10/'
os.makedirs(output_dir, exist_ok=True)

best_val_loss = float('inf')
patience = 300
no_improvement_epochs = 0

csv_path = os.path.join(output_dir, 'training_loss.csv')
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss'])

print("Starting full training with curriculum for hard negative mining...")
epochs = 100

scaler = GradScaler()
for epoch in range(epochs):


    print(f"Epoch {epoch+1}/{epochs}")

    train_loss = train_new(train_loader, model, optimizer, device, scaler = scaler)
    val_loss   = test_new(val_loader,  model, device)

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    else:
        no_improvement_epochs += 1

    state_dicts = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch+1}.pt'))

    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch + 1, float(train_loss), float(val_loss)])

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")
    print(f"Appended epoch {epoch+1} to {csv_path}")

    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")
