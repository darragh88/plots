
"""
japan_tgcn_pyg_only.py
----------------------
Minimal, *pure* PyTorch Geometric example (no torch‑geometric‑temporal) for
forecasting a scalar target per Japanese power‑market region using a
Temporal GCN built from first principles.

HOW TO USE
==========
1. Prepare your data:
   • region_dfs : dict[str, pandas.DataFrame]
     Each DataFrame must
     - have a shared DateTimeIndex
     - contain *one* column named 'target'        (scalar you want to predict)
     - all other columns = time‑aligned features usable at prediction time.

   Example:
       region_dfs = {
           'Hokkaido': pd.read_csv('hokkaido.csv', index_col=0, parse_dates=True),
           'Tohoku'  : pd.read_csv('tohoku.csv',   index_col=0, parse_dates=True),
           ...
       }

2. Build a (N × N) adjacency matrix 'adjacency' giving tie‑line connections.
   adjacency[i,j] = 1 if regions i and j are connected, else 0.
   Capacities can be encoded as weights.

3. Run:
       python japan_tgcn_pyg_only.py

   The script trains for one epoch and prints running MAE every 100 snapshots.
   Adapt the hyper‑parameters, train/val split and epochs to your needs.

Dependencies
------------
* torch >= 2.0
* torch_geometric >= 2.4
* pandas, numpy

"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv


# ------------------------------------------------------------------
# 1. Custom temporal dataset (snapshot iterator)
# ------------------------------------------------------------------
class MyTemporalDataset(torch.utils.data.Dataset):
    """Stores a list of (X_t, y_t) snapshots for t=0…T-1."""

    def __init__(self, X_list, y_list, edge_index, edge_weight):
        assert len(X_list) == len(y_list)
        self.X_list = X_list      # list of torch.FloatTensor (num_nodes × num_feats)
        self.y_list = y_list      # list of torch.FloatTensor (num_nodes,)
        self.edge_index = edge_index      # (2 × E) LongTensor
        self.edge_weight = edge_weight    # (E,) FloatTensor

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.y_list[idx]


# ------------------------------------------------------------------
# 2. DIY Temporal GCN (= GCN + GRU memory)
# ------------------------------------------------------------------
class DIY_TGCN(nn.Module):
    """A minimal reproduction of TGCN using PyG GCNConv + GRUCell."""

    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        # Hidden state will be initialised lazily the first time forward() is called
        self.register_buffer("h_mem", torch.zeros(num_nodes, hidden_channels))

    def reset_memory(self):
        self.h_mem.zero_()

    def forward(self, x, edge_index, edge_weight=None):
        """x: (num_nodes × in_channels)"""
        gcn_out = self.gcn(x, edge_index, edge_weight)
        self.h_mem = self.gru(gcn_out, self.h_mem)
        return self.h_mem  # (num_nodes × hidden)


# ------------------------------------------------------------------
# 3. Imbalance forecasting model: DIY_TGCN + linear head
# ------------------------------------------------------------------
class ImbalanceNet(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden: int = 64):
        super().__init__()
        self.tgcn = DIY_TGCN(num_nodes=num_nodes,
                             in_channels=in_channels,
                             hidden_channels=hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.tgcn(x, edge_index, edge_weight)
        return self.out(h).squeeze(-1)   # (num_nodes,)


# ------------------------------------------------------------------
# 4. Utility: build dataset from dict of DataFrames + adjacency
# ------------------------------------------------------------------
def build_dataset(region_dfs: dict, adjacency: np.ndarray):
    """Converts per‑region DataFrames into MyTemporalDataset."""
    regions = sorted(region_dfs.keys())
    num_nodes = len(regions)

    # 4a) intersect DateTimeIndex
    common_idx = region_dfs[regions[0]].index
    for r in regions[1:]:
        common_idx = common_idx.intersection(region_dfs[r].index)
    common_idx = common_idx.sort_values()

    X_list, y_list = [], []
    for ts in common_idx:
        node_feats, node_tgt = [], []
        for r in regions:
            row = region_dfs[r].loc[ts]
            node_tgt.append(row["target"])
            node_feats.append(row.drop("target").values)
        X_list.append(torch.tensor(np.vstack(node_feats), dtype=torch.float32))
        y_list.append(torch.tensor(np.array(node_tgt), dtype=torch.float32))

    edge_index = torch.tensor(np.vstack(np.nonzero(adjacency)), dtype=torch.long)
    edge_weight = torch.tensor(adjacency[np.nonzero(adjacency)], dtype=torch.float32).view(-1)

    return MyTemporalDataset(X_list, y_list, edge_index, edge_weight)


# ------------------------------------------------------------------
# 5. Training loop (single epoch)
# ------------------------------------------------------------------
def train_one_epoch(model, loader, dataset, optimiser, loss_fn, device="cpu"):
    model.train()
    total_loss = 0.0
    for step, (x_t, y_t) in enumerate(loader):
        x_t = x_t.squeeze(0).to(device)   # dataset returns 3‑D because of DataLoader
        y_t = y_t.squeeze(0).to(device)

        optimiser.zero_grad()
        y_pred = model(x_t, dataset.edge_index.to(device), dataset.edge_weight.to(device))
        loss = loss_fn(y_pred, y_t)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"[{step:>5}] MAE = {loss.item():8.4f}")

    return total_loss / len(loader)


# ------------------------------------------------------------------
# 6. Main – dummy example with random data
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIY TGCN example (PyG‑only)")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # --------------------------
    # create dummy toy dataset
    # --------------------------
    num_nodes = 9
    num_feats = 4
    num_steps = 500

    regions = [f"Region_{i}" for i in range(num_nodes)]
    idx = pd.date_range("2025‑01‑01", periods=num_steps, freq="H")

    rng = np.random.default_rng(42)
    region_dfs = {}
    for r in regions:
        df = pd.DataFrame(rng.normal(size=(num_steps, num_feats + 1)),
                          index=idx,
                          columns=[f"feat_{j}" for j in range(num_feats)] + ["target"]).astype("float32")
        region_dfs[r] = df

    adjacency = np.ones((num_nodes, num_nodes), dtype="float32") - np.eye(num_nodes, dtype="float32")

    dataset = build_dataset(region_dfs, adjacency)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImbalanceNet(num_nodes=num_nodes, in_channels=num_feats, hidden=64).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    for epoch in range(args.epochs):
        mae = train_one_epoch(model, loader, dataset, optimiser, loss_fn, device)
        print(f"Epoch {epoch+1}: mean MAE = {mae:.4f}")

    print("Training complete. Replace the dummy data with your real market data and enjoy! ")
