from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

# Carregar modelo
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "churn_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def root():
    return {"message": "Churn Prediction API está funcionando!", "endpoints": ["/predict (POST)", "/docs"]}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return {
        "probabilidade_churn": float(prob),
        "recomendacao": "Oferecer desconto/benefícios" if prob > 0.6 else "Cliente estável"
    }
