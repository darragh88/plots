"""
gnn_nan_debug_template.py
------------------------
Template training loop for PyTorch + PyTorch Geometric that:
  • Detects NaNs/Infs in inputs, outputs, loss, and gradients.
  • Clips gradients to avoid exploding weights.
  • Shows where the first NaN appears (raises RuntimeError).
Adapt as needed (replace `YourGNNModel`, `YourDataset`, `criterion`, etc.).

Author: ChatGPT
"""

import torch
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------
# 1. Replace these with your real model and dataset
# ---------------------------------------------------------------------
class YourGNNModel(torch.nn.Module):
    """Minimal 2‑layer GCN example; edit freely."""
    def __init__(self, in_dim: int = 5, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        from torch_geometric.nn import GCNConv, LayerNorm
        self.conv1 = GCNConv(in_dim, hidden)
        self.norm1 = LayerNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm2 = LayerNorm(hidden)
        self.head  = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        import torch.nn.functional as F
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = F.relu(self.norm2(self.conv2(x, edge_index)))
        return self.head(x)

# Dummy stub dataset --------------------------------------------------
class YourDataset(torch.utils.data.Dataset):
    """Creates random graphs with 9 nodes, 5 features, and 1 regression target."""
    def __init__(self, num_graphs: int = 100):
        super().__init__()
        from torch_geometric.data import Data
        import torch
        import torch_geometric.utils as pyg_utils

        self.data_list = []
        for _ in range(num_graphs):
            num_nodes = 9
            x = torch.randn(num_nodes, 5)
            # Simple fully‑connected graph (replace with real edges)
            edge_index = pyg_utils.dense_to_sparse(torch.ones(num_nodes, num_nodes))[0]
            y = torch.randn(num_nodes, 1)  # regression target
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ---------------------------------------------------------------------
# 2. Training utilities
# ---------------------------------------------------------------------
def has_nan(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def debug_train(model: torch.nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                device: torch.device = torch.device("cpu"),
                grad_clip: float = 1.0):
    """Runs one epoch with detailed NaN/Inf diagnostics."""
    torch.autograd.set_detect_anomaly(True)  # catches bad backward ops
    model.train()

    for step, data in enumerate(loader):
        data = data.to(device)

        # ---- Input checks ---------------------------------------------------
        for name, tens in [('x', data.x), ('y', data.y)]:
            if has_nan(tens):
                raise RuntimeError(f"NaN/Inf detected in {name} BEFORE forward, step {step}")

        # ---- Forward pass ---------------------------------------------------
        out = model(data.x, data.edge_index, data.batch)
        if has_nan(out):
            raise RuntimeError(f"NaN/Inf detected in MODEL OUTPUT at step {step}")

        loss = criterion(out, data.y)
        if has_nan(loss):
            raise RuntimeError(f"NaN/Inf detected in LOSS at step {step}")

        # ---- Backward pass --------------------------------------------------
        loss.backward()

        # ---- Gradient checks ------------------------------------------------
        for n, p in model.named_parameters():
            if p.grad is not None and has_nan(p.grad):
                raise RuntimeError(f"NaN/Inf detected in grad of {n} at step {step}")

        # ---- Optimiser step -------------------------------------------------
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step:03d} | loss = {loss.item():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset  = YourDataset(num_graphs=200)
    loader   = DataLoader(dataset, batch_size=16, shuffle=True)
    model    = YourGNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss(reduction='mean')

    print("Beginning debug training...")
    debug_train(model, loader, optimizer, criterion, device)
    print("Training finished without NaNs!")


if __name__ == "__main__":
    main()
