End-to-end MLOps pipeline for customer churn prediction using DVC, MLflow, FastAPI and Docker.

# Customer Churn Prediction MLOps Pipeline

This project presents an end-to-end Machine Learning and MLOps pipeline for customer churn prediction.

It was designed to demonstrate how a machine learning model can be developed, tracked, versioned, deployed and containerized using a reproducible project structure.

The goal is not only to train a predictive model, but also to show the engineering practices required to move machine learning closer to a production environment.

## What this project demonstrates

Data versioning and reproducibility with DVC  
Experiment tracking and model management with MLflow  
Model training using scikit-learn  
REST API deployment with FastAPI  
Containerization with Docker  
Modular project organization  
Clear separation between training, inference and infrastructure  

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| scikit-learn | Model training and preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| MLflow | Experiment tracking and model artifact management |
| DVC | Data versioning and reproducibility |
| FastAPI | Model inference API |
| Uvicorn | ASGI server |
| Docker | Containerization and deployment |
| Joblib | Model serialization |
| Git and GitHub | Version control |

## Architecture

The project follows a simple MLOps workflow:

```text
Raw Data
   ↓
Data Versioning with DVC
   ↓
Model Training
   ↓
Experiment Tracking with MLflow
   ↓
Model Serialization
   ↓
FastAPI Inference Service
   ↓
Dockerized Deployment
```

## Main Features

### Data Versioning

The dataset is tracked with DVC to improve reproducibility and make data changes easier to manage.

```bash
dvc add data/raw/churn.csv
dvc push
```

### Model Training

The training pipeline handles preprocessing, model training, metric logging and artifact storage.

```bash
python src/train.py
mlflow ui
```

MLflow UI:

```text
http://localhost:5000
```

### Model Inference API

The trained model is served through a FastAPI application.

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check |
| POST | /predict | Returns churn prediction |

Run the API locally:

```bash
uvicorn api.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

### Docker Deployment

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 --name churn-api-v1 churn-api
```

## Getting Started

### Prerequisites

Python 3.8 or higher  
Git  
Docker, optional  
DVC, if using data versioning  

### Installation

Clone the repository:

```bash
git clone https://github.com/felipejosc/customer-churn-mlops.git
cd customer-churn-mlops
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run the API:

```bash
uvicorn api.main:app --reload
```

## Project Purpose

This portfolio project was built to demonstrate practical skills in MLOps, backend development and machine learning deployment.

It shows how a churn prediction model can be structured beyond a notebook, with focus on reproducibility, experiment tracking, API serving and containerized deployment.
