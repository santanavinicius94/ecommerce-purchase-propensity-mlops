import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline


def train_model(pipeline, X_train, y_train) -> Pipeline:
    """Treina um pipeline do scikit-learn usando os dados de treinamento fornecidos.

    Args:
        pipeline: Um pipeline do scikit-learn ou modelo para ser treinado.
        X_train: Conjunto de dados de treinamento.
        y_train: Rótulos verdadeiros do conjunto de dados de treinamento.

    Returns:
        Pipeline: A instância treinada do pipeline/modelo.
    """
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model_name: str, pipeline, X_test, y_test):
    """Avalia o pipeline treinado em dados de teste, calcula métricas de performance, registra o experimento e printa um resumo.

    Args:
        model_name (str): Identificador/nome do modelo sendo avaliado.
        pipeline: O pipeline do scikit-learn treinado e suportando as funções predict e predict_proba.
        X_test: Conjunto de dados de teste.
        y_test: Rótulos verdadeiros do conjunto de dados de teste.

    Returns:
        np.ndarray: Os rótulos previstos para o conjunto de testes.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    average_precision = average_precision_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    metrics_dict = {
        "average_precision": round(average_precision, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
    }

    print(f"=== {model_name} ===")
    for metric_name, metric_value in metrics_dict.items():
        print(f"{metric_name}: {metric_value}")

    return y_pred


def save_artifact(pipeline, filename: str):
    """Serializa e salva o pipeline em disco usando joblib.

    Args:
        pipeline: O pipeline treinado do modelo a ser salvo.
        filename (str): O caminho de destino e nome do arquivo a ser salvo.
    """
    joblib.dump(pipeline, filename)
    print(f"Artefato salvo com sucesso em {filename}")
