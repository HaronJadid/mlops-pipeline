import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


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


X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1) 


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
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


print("Starting training...")
model.train() 

for epoch in range(config.epochs):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predicted_classes = predictions.round()
    acc = (predicted_classes.eq(y_train).sum() / float(y_train.shape[0])).item()


    wandb.log({
        "epoch": epoch, 
        "loss": loss.item(), 
        "accuracy": acc
    })

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")


model_path = "model.pt"
torch.save(model.state_dict(), model_path)


wandb.save(model_path)

wandb.finish()
print("Training complete and model saved!")