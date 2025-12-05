# main.py

import os
import csv
import torch
from torch_geometric.loader import DataLoader
from torch.cuda.amp import GradScaler

from src.data import CCV1          # same as before
from src.model import Net
from src.train import train_oc, test_oc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading data...")

ipath = "/vols/cms/mm1221/Independent/Data/Electron_2/train/"
vpath = "/vols/cms/mm1221/Independent/Data/Electron_2/val/"

data_train = CCV1(root=ipath, split="train", step_size=1000, max_events=700000)
data_val   = CCV1(root=vpath,   split="val",   step_size=1000, max_events=300000)

BS = 32

train_loader = DataLoader(
    data_train,
    batch_size=BS,
    shuffle=True,
    follow_batch=["x"],
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    data_val,
    batch_size=BS,
    shuffle=False,
    follow_batch=["x"],
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

model = Net(
    hidden_dim=64,
    num_layers=5,
    dropout=0.0038,
    k=32,
    num_heads=8,
    edge_hidden_dim=16,
    edge_out_dim=16,
    cluster_dim=2,   
    prop_dim=0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

output_dir = "/vols/cms/mm1221/Independent/ObjectCondensation/runs/Electron_FULL_hd64_nl5_k32_dp0038_nh8_ehd16_eod16/"
os.makedirs(output_dir, exist_ok=True)

best_val_loss = float("inf")
patience = 300
no_improvement_epochs = 0

csv_path = os.path.join(output_dir, "training_loss.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])

print("Starting Object Condensation training...")
epochs = 100
scaler = GradScaler()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    train_loss = train_oc(
        train_loader,
        model,
        optimizer,
        device,
        scaler,
        q_min=0.1,
        noise_label=-1,       
        s_att=1.0,
        s_rep=1.0,         
        s_coward=1.0, 
        s_noise=1.0,
    )
    val_loss = test_oc(
        val_loader,
        model,
        device,
        q_min=0.1,
        noise_label=-1,
        s_att=1.0,
        s_rep=1.0,
        s_coward=1.0,
        s_noise=1.0,
    )

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    else:
        no_improvement_epochs += 1

    state_dicts = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "lr": scheduler.state_dict(),
    }
    torch.save(state_dicts, os.path.join(output_dir, f"epoch-{epoch+1}.pt"))

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch + 1, float(train_loss), float(val_loss)])

    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}"
    )
    print(f"Appended epoch {epoch+1} to {csv_path}")

    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
print("Training complete. Final model saved.")
