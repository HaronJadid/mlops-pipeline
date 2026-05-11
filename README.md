# End-to-End MLOps Pipeline: Fraud Detection 🚀

An automated, containerized machine learning pipeline that trains a PyTorch model, tracks experiments via Weights & Biases, and serves live inferences through a REST API.

## 🏗️ Architecture & Tech Stack

This project demonstrates a production-ready MLOps lifecycle:
* **Engine (Training):** `PyTorch` (Multi-Layer Perceptron)
* **Observability (Tracking):** `Weights & Biases`
* **Serving (API):** `FastAPI` & `Pydantic` (Data Validation)
* **Containerization:** `Docker` & `Docker Compose`
* **CI/CD:** `GitHub Actions` (Automated build testing)

## 📊 Experiment Tracking
During training, metrics (loss, accuracy) are streamed in real-time to the cloud. 
👉 **[View the Weights & Biases Training Report here]([YOUR_WANDB_LINK_HERE])**

## ⚙️ Local Deployment (Quickstart)

Thanks to Docker Compose, you can spin up the entire infrastructure with a single command. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/HaronJadid/mlops-pipeline.git](https://github.com/HaronJadid/mlops-pipeline.git)
   cd mlops-pipeline
   
Start the application:

Bash
docker-compose up --build

The API will instantly be live at `http://127.0.0.1:8000`.

## 🧪 Testing the API

You can test the live endpoint using the auto-generated FastAPI UI at `http://127.0.0.1:8000/docs`, or by sending a POST request via cURL:

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [0.5, -1.2, 3.14, 0.0, 1.1, -0.5, 2.2, 0.9, -3.0, 4.2]
}'
Expected Response:

JSON
{
  "fraud_probability": 0.8148,
  "is_fraud": true
}

