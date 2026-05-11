from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn


# FastAPI server instance
app = FastAPI(title="Fraud Detection API")

# Data validation schema using Pydantic
class FraudRequest(BaseModel):
    features: list[float]

# Neural Network Architecture 
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

# Load trained model weights
model = FraudClassifier()
model.load_state_dict(torch.load("model.pt", weights_only=True))

# Set model to evaluation mode
model.eval()

# Prediction endpoint
@app.post("/predict")
def predict_fraud(request: FraudRequest):
    # Validate input features
    if len(request.features) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 features are required.")
    
    # Convert the input data into a PyTorch Tensor
    input_tensor = torch.FloatTensor([request.features])
    
    # Turn off gradient calculation for inference
    with torch.no_grad():
        prediction = model(input_tensor)
        # Get the probability
        probability = prediction.item()
        # Round it to 0 or 1 for the final class
        is_fraud = bool(round(probability)) 
        
    return {
        "fraud_probability": probability,
        "is_fraud": is_fraud
    }