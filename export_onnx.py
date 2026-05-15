import torch
import torch.nn as nn
import onnx 

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
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model, 
    dummy_input, 
    "fraud_model.onnx", 
    input_names=["input_features"], 
    output_names=["fraud_probability"],
    opset_version=15 
)

onnx_model = onnx.load("fraud_model.onnx")
onnx_model.ir_version = 9  
onnx.save(onnx_model, "fraud_model.onnx")

print("Success! Model exported and spoofed to IR Version 9")