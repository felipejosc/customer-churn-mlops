import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/raw/churn.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Converter TotalCharges para numérico, tratando strings vazias como NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    X_train, X_test, y_train, y_test = load_data()

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=500))
        ]
    )

    mlflow.set_experiment("customer-churn-mlops")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, preds.round())

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))

        print(f"AUC: {auc:.4f}, ACC: {acc:.4f}")

if __name__ == "__main__":
    main()
