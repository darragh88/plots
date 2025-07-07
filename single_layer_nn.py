
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ─── Data Preparation ────────────────────────────────────────────────
# Create synthetic data: y = 3x + noise
torch.manual_seed(0)
X = torch.linspace(0, 1, 100).unsqueeze(1)  # shape (100, 1)
y = 3 * X + 0.3 * torch.randn_like(X)

# ─── Model Definition ────────────────────────────────────────────────
class SingleLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SingleLayerNN()

# ─── Training Setup ─────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 100
losses = []

# ─── Training Loop ──────────────────────────────────────────────────
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# ─── Prediction on New Data ─────────────────────────────────────────
model.eval()
new_X = torch.tensor([[1.5], [2.0], [2.5]])
predictions = model(new_X).detach().numpy()
print("Predictions on new data:", predictions.flatten())

# ─── Plot Loss Curve ────────────────────────────────────────────────
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()
