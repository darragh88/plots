"""
gnn_examples.py
===============

A reference module that collects three self‑contained PyTorch Geometric
examples demonstrating how to consume **edge features** for different tasks:

1. **Node‑level regression**  (GAT with `edge_attr`)  
2. **Node‑level classification** (GAT with `edge_attr`)  
3. **Edge‑level classification** (GraphSAGE with `edge_attr`)

All models return **raw logits** or **raw predictions** so they can be paired
with any loss/metric you like.  Training loops are *not* included—plug these
models into the loaders / early‑stopping scaffolding you already have.

Author: ChatGPT (compiled from your snippets #1, #2 & #3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


# --------------------------------------------------------------------------- #
#  Helper metrics                                                             #
# --------------------------------------------------------------------------- #

def mean_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error for node‑level *regression* tasks."""
    return torch.mean(torch.abs(preds - targets)).item()


def node_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Node‑level accuracy (multi‑class)."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def edge_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Edge‑level accuracy that works for binary or multi‑class labels."""
    if logits.size(1) == 1:                   # binary BCE + sigmoid style
        preds = (logits > 0).long().view(-1)
    else:                                     # multi‑class CE
        preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# --------------------------------------------------------------------------- #
#  1. Edge‑aware Node Regression (GAT)                                        #
# --------------------------------------------------------------------------- #

class GATReg(nn.Module):
    """2‑layer GAT for per‑node **regression**, with multi‑dim edge features."""

    def __init__(
        self,
        in_dim: int,              # node feature size
        hidden: int = 128,
        out_dim: int = 1,
        heads: int = 4,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_dim,
            hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, data):                 # ↳ returns shape (N, out_dim)
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, ei, edge_attr=ea))
        x = F.relu(self.conv2(x, ei, edge_attr=ea))
        return self.lin(x)                   # **raw** regression outputs


# --------------------------------------------------------------------------- #
#  2. Edge‑aware Node Classification (GAT)                                    #
# --------------------------------------------------------------------------- #

class GATNodeClassifier(nn.Module):
    """Same backbone as *GATReg* but returns logits over `num_classes`."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden: int = 128,
        heads: int = 4,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_dim,
            hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            hidden * heads,
            hidden,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):                # ↳ returns (N, C) logits
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, ei, edge_attr=ea))
        x = F.relu(self.conv2(x, ei, edge_attr=ea))
        return self.lin(x)                  # raw class logits


# --------------------------------------------------------------------------- #
#  3. Edge Classification (GraphSAGE + MLP predictor)                         #
# --------------------------------------------------------------------------- #

class EdgeMLPPredictor(nn.Module):
    """MLP that maps concatenated endpoint embeddings ➜ edge logits."""

    def __init__(self, emb_dim: int, hidden: int = 128, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (E, num_classes)."""
        h_src = h[edge_index[0]]            # (E, d)
        h_dst = h[edge_index[1]]
        return self.mlp(torch.cat([h_src, h_dst], dim=-1))


class SAGEEdgeClassifier(nn.Module):
    """2‑layer GraphSAGE encoder + MLP edge predictor (edge‑aware)."""

    def __init__(
        self,
        in_dim: int,              # node feature size
        edge_dim: int,            # edge feature size
        hidden: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, edge_dim=edge_dim)
        self.conv2 = SAGEConv(hidden, hidden, edge_dim=edge_dim)
        self.pred  = EdgeMLPPredictor(hidden, hidden, num_classes)

    def forward(self, data):                     # ↳ returns (E, C) logits
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, ei, edge_attr=ea))
        x = F.relu(self.conv2(x, ei, edge_attr=ea))
        return self.pred(x, ei)
