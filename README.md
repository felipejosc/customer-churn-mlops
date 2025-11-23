This repository presents a complete Machine Learning + MLOps pipeline for customer churn prediction, including:

Data versioning with DVC

Model training and experiment tracking using MLflow

Deployment of an inference API with FastAPI

Containerization with Docker

A modular and reproducible project structure

This is a portfolio project designed to demonstrate practical skills in MLOps and production-ready model architecture.

Technologies Used: 

  "Python": "Main programming language",
  "scikit-learn": "Model development and preprocessing",
  "FastAPI": "Model inference API",
  "Uvicorn": "ASGI server for FastAPI",
  "MLflow": "Experiment tracking and model registry",
  "DVC": "Data version control and reproducibility",
  "Pandas": "Data manipulation",
  "NumPy": "Numerical computation",
  "Docker": "Containerization and deployment",
  "Joblib": "Model serialization",
  "Git/GitHub": "Version control and collaboration"


# Customer Churn Prediction - MLOps Pipeline:

## Architecture

### 1. Data Versioning with DVC
Dataset versioning and reproducibility using DVC:
```bash
dvc add data/raw/churn.csv
dvc push
```

### 2. Model Training with MLflow
The `train.py` script handles:
- Data preprocessing
- Model training
- Metrics logging (accuracy, AUC)
- Model artifact storage

```bash
python train.py
mlflow ui
```
Access MLflow UI: http://localhost:5000

### 3. Model Deployment (FastAPI)
REST API with the following endpoints:
- `GET /` - Health check
- `POST /predict` - Churn prediction

```bash
uvicorn api.main:app --reload
```
API Documentation: http://localhost:8000/docs

### 4. Containerization with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 --name churn-api-v1 churn-api
```

##  Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional)
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/felipejosc/customer-churn-mlops.git
cd customer-churn-mlops
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python src/train.py
```

4. Run the API
```bash
uvicorn api.main:app --reload
```

### Docker Deployment

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Tech Stack

- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker
- **ML Libraries**: scikit-learn, pandas, numpy
