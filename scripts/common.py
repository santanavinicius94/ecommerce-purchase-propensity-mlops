"""Utilitários compartilhados pelas etapas do pipeline."""

import json
from pathlib import Path

import mlflow
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "params.yaml"


def load_config(config_path: str | Path) -> dict:
    """Carrega o arquivo de parâmetros YAML de forma segura."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def project_path(value: str) -> Path:
    """Resolve um caminho definido no params.yaml a partir da raiz do projeto."""
    return PROJECT_ROOT / value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_mlflow(config: dict) -> None:
    """Aplica o destino de tracking e o experimento configurados."""
    mlflow_config = config["mlflow"]
    mlflow.set_experiment(mlflow_config["experiment_name"])


def load_run_id(config: dict) -> str:
    """Obtém o identificador do run criado na etapa de treino."""
    run_path = project_path(config["paths"]["mlflow_run"])
    with run_path.open(encoding="utf-8") as stream:
        return json.load(stream)["run_id"]
