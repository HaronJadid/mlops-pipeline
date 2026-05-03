from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

# 1. Initialize the FastAPI Window
app = FastAPI(title="Fraud Detection API")

# 2. Define the Menu (Data Validation)
# We tell the API: "You are only allowed to accept a list of floating-point numbers"
class FraudRequest(BaseModel):
    features: list[float]

# 3. Rebuild the Engine
# We must define the exact same architecture so PyTorch knows how to load the weights.
# (Note: In a larger project, you would put this class in a shared 'model.py' file)
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

# 4. Load the Chef (The trained weights)
model = FraudClassifier()
model.load_state_dict(torch.load("model.pt", weights_only=True))

# CRITICAL MLOps STEP: Put the model in evaluation mode
model.eval()

# 5. Open the Window to the public
# We use @app.post because the user is POSTing a heavy payload (their data) to us.
@app.post("/predict")
def predict_fraud(request: FraudRequest):
    # Check if they sent exactly 10 features, as our model requires
    if len(request.features) != 10:
        raise HTTPException(status_code=400, detail="Exactly 10 features are required.")
    
    # Convert the user's Python list into a PyTorch Tensor
    input_tensor = torch.FloatTensor([request.features])
    
    # CRITICAL MLOps STEP: Turn off gradient calculation for inference
    with torch.no_grad():
        prediction = model(input_tensor)
        # Get the raw probability (e.g., 0.85)
        probability = prediction.item()
        # Round it to 0 or 1 for the final class
        is_fraud = bool(round(probability)) 
        
    return {
        "fraud_probability": probability,
        "is_fraud": is_fraud
    }