import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_split_data(
    filepath: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 7,
    stratify: bool = False,
):
    """Carrega um CSV e o divide em conjuntos de treino e teste.

    Args:
        filepath (str): Caminho do arquivo CSV.
        target_col (str): Nome da coluna target para predição.
        test_size (float, optional): Proporção do conjunto de dados para incluir no teste. Defaults to 0.2.
        random_state (int, optional): Seed para reprodutibilidade. Defaults to 7.
        stratify (bool, optional): Flag para realizar estratificação dos dados de acordo com 'target_col'. Defaults to False.

    Returns:
        tuple: Uma tupla contendo (X_train, X_test, y_train, y_test).
    """

    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    to_stratify = y if stratify else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=to_stratify
    )

def create_preprocessing_pipeline(
    num_features: list, cat_features: list
) -> ColumnTransformer:
    """Cria um ColumnTransformer do scikit-learn para preprocessar dados numéricos e categóricos.

    Aplica as etapas de feature engineering e scaling para as features numéricas, e one-hot encoding
    para features categóricas.

    Args:
        num_features (list): Lista de features numéricas para serem transformadas.
        cat_features (list): Lista de features categóricas para serem transformadas.

    Returns:
        ColumnTransformer: componente configurado de pré-processamento do pipeline
    """
    num_transformer = make_pipeline(StandardScaler())
    cat_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="infrequent_if_exist", drop="first")
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )

    return preprocessor


def assemble_full_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Realiza o encadeamento do pipeline de preprocessamento com o modelo de machine learning em um único pipeline.

    Args:
        preprocessor (ColumnTransformer): O pipeline de trasformação das features.
        model: Um modelo compatível com o scikit-learn (e.g.: LogiticRegression, RandomForestClassifier).

    Returns:
        Pipeline: O pipeline do scikit-learn pronto para treino e previsão.
    """
    pipeline = make_pipeline(preprocessor, model)
    return pipeline
