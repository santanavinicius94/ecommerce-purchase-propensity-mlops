import os

import mlflow
from ecommerce_purchase_propensity_mlops.preprocessing import (
    assemble_full_pipeline,
    create_preprocessing_pipeline,
    load_data,
    normalize_columns,
    split_data,
)
from ecommerce_purchase_propensity_mlops.training import (
    evaluate_model,
    save_artifact,
    train_model,
)
from sklearn.linear_model import LogisticRegression

# Configurações do treinamento
# ==========

basepath = os.path.dirname(os.path.abspath(__file__))

DATA_FILEPATH = basepath + "/../data/raw/online_shoppers_intention.csv"
MODEL_FILEPATH = basepath + "/../models/logistic_regression.joblib"

RANDOM_STATE = 7
MODEL_NAME = "logistic_regression"
MODEL = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")


TARGET_COL = "revenue"
NUM_FEATURES = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "productrelated",
    "productrelated_duration",
    "bouncerates",
    "exitrates",
    "pagevalues",
    "specialday",
]

CAT_FEATURES = [
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
    "weekend",
]

mlflow.set_experiment("online_shoppers_intention_experiment")

# Execução do pipeline completo
# ==========

df = load_data(DATA_FILEPATH)
df = normalize_columns(df)

X_train, X_test, y_train, y_test = split_data(df, TARGET_COL, stratify=True)

mlflow.sklearn.autolog()
preprocessor = create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)
pipeline = assemble_full_pipeline(preprocessor, MODEL)
trained_pipeline = train_model(pipeline, X_train, y_train)

evaluate_model(MODEL_NAME, trained_pipeline, X_test, y_test)
save_artifact(trained_pipeline, MODEL_FILEPATH)
