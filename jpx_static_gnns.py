
# jpx_static_gnns.py  ── Static (non‑temporal) GNNs for Japanese price spreads
# ===========================================================================
# 1. region_dfs : dict[str, pd.DataFrame]
#       • Each DataFrame has identical DateTimeIndex and same columns
#       • Column "target" = numeric value to predict
# 2. adjacency  : np.ndarray  (N × N; zeros on diagonal; duplicate both
#    directions if capacities differ)
#
# Replace the two placeholders in section (0) before running:
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATv2Conv, NNConv
)

# ─────────────────────────────────────────────────────────────────────────────
# (0)  INSERT YOUR DATA HERE
# ---------------------------------------------------------------------------
# Example:
# from my_data_loader import region_dfs, adjacency
# region_dfs = {...}
# adjacency  = ...
region_dfs = None      # <-- replace
adjacency  = None      # <-- replace
# ─────────────────────────────────────────────────────────────────────────────

def make_edge_tensors(adj: np.ndarray):
    src, dst = np.nonzero(adj)
    edge_index  = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)
    return edge_index, edge_weight


def build_snapshots(region_dfs: dict[str, pd.DataFrame],
                    adj: np.ndarray):
    regions = sorted(region_dfs)
    ts_common = sorted(
        set.intersection(*(set(df.index) for df in region_dfs.values()))
    )
    edge_index, edge_weight = make_edge_tensors(adj)
    snaps = []
    for ts in ts_common:
        feats, tgts = [], []
        for r in regions:
            row = region_dfs[r].loc[ts]
            tgts.append(row['target'])
            feats.append(row.drop('target').to_numpy())
        x = torch.tensor(np.vstack(feats), dtype=torch.float32)
        y = torch.tensor(tgts,         dtype=torch.float32)
        snaps.append(
            Data(x=x,
                 edge_index=edge_index,
                 edge_weight=edge_weight,
                 y=y,
                 snap_time=torch.tensor([pd.Timestamp(ts).value]))
        )
    return snaps


def static_split(snaps, cutoff_date: str):
    cutoff = pd.Timestamp(cutoff_date).value
    train = [g for g in snaps if g.snap_time <= cutoff]
    test  = [g for g in snaps if g.snap_time  > cutoff]
    return train, test


class GCNReg(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(d_in, hidden)
        self.head  = nn.Linear(hidden, 1)
    def forward(self, data):
        h = torch.relu(self.conv1(data.x, data.edge_index, data.edge_weight))
        return self.head(h).squeeze(-1)


class SAGEReg(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(d_in, hidden)
        self.head  = nn.Linear(hidden, 1)
    def forward(self, data):
        h = torch.relu(self.conv1(data.x, data.edge_index))
        return self.head(h).squeeze(-1)


class GATReg(nn.Module):
    def __init__(self, d_in, hidden=64, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(d_in, hidden // heads, heads=heads)
        self.head  = nn.Linear(hidden, 1)
    def forward(self, data):
        h = torch.relu(self.conv1(data.x, data.edge_index))
        return self.head(h).squeeze(-1)


class NNConvReg(nn.Module):
    """Edge‑conditioned convo; edge_weight is capacity scalar."""
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, d_in * hidden)
        )
        self.conv1 = NNConv(d_in, hidden, nn=self.edge_mlp, aggr='mean')
        self.head  = nn.Linear(hidden, 1)
    def forward(self, data):
        ew = data.edge_weight.view(-1, 1)
        h = torch.relu(self.conv1(data.x, data.edge_index, ew))
        return self.head(h).squeeze(-1)


MODELS = {
    'gcn':    GCNReg,
    'sage':   SAGEReg,
    'gat':    GATReg,
    'nnconv': NNConvReg,
}

def train_epoch(model, loader, opt):
    model.train(); loss_fn = nn.L1Loss(); tot = 0
    for data in loader:
        opt.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward(); opt.step(); tot += loss.item()
    return tot / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); loss_fn = nn.L1Loss(); tot = 0
    for data in loader:
        tot += loss_fn(model(data), data.y).item()
    return tot / len(loader)


def main(region_dfs, adjacency, cutoff='2024-06-30',
         epochs=10, batch_size=1):
    snaps = build_snapshots(region_dfs, adjacency)
    train_ds, test_ds = static_split(snaps, cutoff)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    d_in = train_ds[0].x.shape[1]
    for tag, Net in MODELS.items():
        model = Net(d_in)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_epoch(model, train_dl, opt)
        mae = evaluate(model, test_dl)
        print(f"{tag.upper():7s} | test MAE = {mae:.4f}")


if __name__ == '__main__':
    if region_dfs is None or adjacency is None:
        raise RuntimeError(
            'Please load `region_dfs` and `adjacency` at the top of the file '
            'before executing.')
    main(region_dfs, adjacency)
