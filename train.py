import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Initialize Weights & Biases
# This creates a new "Run" in your cloud dashboard
wandb.init(
    project="fraud-detection-pipeline",
    config={
        "learning_rate": 0.01,
        "epochs": 50,
        "batch_size": 32,
        "architecture": "SimpleNN"
    }
)
config = wandb.config

# 2. Generate Synthetic Data
# We create 1000 samples simulating fraud detection (binary classification)
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert arrays to PyTorch Tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1) # Reshape for PyTorch

# 3. Define the Engine (A lightweight Neural Network)
class FraudClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.sigmoid(self.layer_2(x))
        return x

model = FraudClassifier()
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# 4. The Training Loop
print("Starting training...")
model.train() # Set model to training mode

for epoch in range(config.epochs):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate basic accuracy
    predicted_classes = predictions.round()
    acc = (predicted_classes.eq(y_train).sum() / float(y_train.shape[0])).item()

    # --- THE CRITICAL MLOps STEP ---
    # Log metrics to W&B for this epoch
    wandb.log({
        "epoch": epoch, 
        "loss": loss.item(), 
        "accuracy": acc
    })

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

# 5. Save the Artifact
# We must save the weights for Day 2 (FastAPI)
model_path = "model.pt"
torch.save(model.state_dict(), model_path)

# Tell W&B to save a copy of this specific model file to the cloud
wandb.save(model_path)

# End the W&B run
wandb.finish()
print("Training complete and model saved!")