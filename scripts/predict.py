"""Executa uma predição usando o modelo champion no MLflow Model Registry."""

import mlflow.pyfunc
import pandas as pd

MODEL_URI = "models:/online_shoppers_intention@champion"
RECORD = {
    "administrative": 0,
    "administrative_duration": 0.0,
    "informational": 0,
    "informational_duration": 0.0,
    "productrelated": 10,
    "productrelated_duration": 300.0,
    "bouncerates": 0.02,
    "exitrates": 0.05,
    "pagevalues": 15.0,
    "specialday": 0.0,
    "month": "Nov",
    "operatingsystems": 2,
    "browser": 2,
    "region": 1,
    "traffictype": 2,
    "visitortype": "Returning_Visitor",
    "weekend": False,
}


def predict():
    """Carrega o champion e retorna a predição para o registro de exemplo."""
    model = mlflow.pyfunc.load_model(MODEL_URI)
    prediction = model.predict(pd.DataFrame([RECORD]))[0]
    print(prediction)
    return prediction


if __name__ == "__main__":
    predict()
