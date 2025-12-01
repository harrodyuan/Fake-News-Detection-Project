import time
import torch
import torch.nn as nn
import torch.optim as optim

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Mac GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create dummy large tensors
print("Creating dummy data...")
x = torch.randn(64, 1000).to(device)
y = torch.randint(0, 2, (64,)).to(device)

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting benchmark loop (1000 iterations)...")
start = time.time()
for _ in range(1000):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
end = time.time()

print(f"Time taken: {end - start:.4f} seconds")
print("Environment seems healthy if this finishes in < 5 seconds.")

