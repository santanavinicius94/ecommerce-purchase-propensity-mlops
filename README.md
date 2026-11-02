# E-commerce Purchase Propensity

Projeto de Machine Learning Engineering para estimar propensão de compra em sessões de e-commerce.

## Estado atual

- CSV versionado com DVC e validado conforme o [dicionário de dados](docs/data_dictionary.md).
- Pré-processamento com log1p, StandardScaler e OneHotEncoder, ajustado apenas no treino.
- Regressão logística com classes balanceadas e split estratificado (seed 7).
- MLflow Tracking para parâmetros, modelo e métricas de teste na mesma execução.
- Testes automatizados de entrada, pré-processamento, treinamento, persistência e tracking.
- Dockerfile multi-stage para validação e treinamento batch.

Model Registry e orquestração com `dvc.yaml` permanecem pendentes. O versionamento de arquivos `.dvc` já existe, mas não representa um pipeline de execução.

## Ambiente local

Requisitos: Git, uv e Python 3.12 (versão definida em `.python-version`). Execute na raiz:

```powershell
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked ruff check .
```

O uv instala o pacote e o grupo de desenvolvimento respeitando `uv.lock`. Os testes usam dados sintéticos e MLflow temporário; não precisam do CSV original nem de servidor externo.

## Recuperar e validar dados

O remote DVC atual é local (`../../dvc-storage`), portanto não estará disponível automaticamente em outra máquina. Configure um armazenamento que contenha os objetos versionados:

```powershell
uv run --locked dvc remote modify --local local_remote "C:/dados/dvc-storage"
uv run --locked dvc pull data/raw/online_shoppers_intention.csv.dvc
uv run --locked python scripts/validate_data.py
```

Substitua o caminho pelo armazenamento real. O validador não altera o CSV: imprime JSON com contagens, duplicatas e hash; retorna 1 para entrada inválida. Consulte [dados brutos](data/raw/README.md).

## Treinar e consultar experimentos

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
uv run --locked python -m scripts.train_model
uv run --locked mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abra http://127.0.0.1:5000. O treinamento valida a entrada, separa treino/teste, ajusta o pipeline e salva `models/logistic_regression.joblib`. As métricas `test_precision`, `test_recall`, `test_accuracy` e `test_f1` usam o conjunto de teste. O modelo registrado como artefato de tracking ainda não é uma versão no Model Registry.

Para personalizar caminhos:

```powershell
uv run --locked python -m scripts.train_model --data data/raw/online_shoppers_intention.csv --model artifacts/model.joblib --experiment online_shoppers_intention_experiment
```

O script também aceita execução direta (`python scripts/train_model.py`). Importá-lo não inicia treinamento. Diretórios de saída são criados automaticamente. Consulte `--help` para os argumentos.

## Docker

O [guia de Docker](docs/docker.md) apresenta o build, a validação e o treinamento com armazenamento persistente. A imagem usa Python 3.12, dependências do lock e usuário sem privilégios; a validação é o comando padrão.

## Organização e validação

| Caminho | Responsabilidade |
| --- | --- |
| `scripts/validate_data.py` | Contrato e perfil do CSV |
| `scripts/train_model.py` | Execução do treino e tracking |
| `src/ecommerce_purchase_propensity_mlops/preprocessing.py` | Transformações e split |
| `src/ecommerce_purchase_propensity_mlops/training.py` | Treino, avaliação e serialização |
| `tests/test_validate_data.py` | Entradas inválidas e preservação dos dados |
| `tests/test_pipeline.py` | Pipeline, inferência, serialização e integração MLflow |

A suíte verifica comportamento, não garante qualidade preditiva no mundo real. O tratamento de duplicatas antes do split, disponibilidade de `PageValues` no momento da previsão, baseline e métricas para desbalanceamento ainda exigem avaliação. Consulte o [roadmap](docs/roadmap.md). O fluxo atual é acadêmico/batch; serving e monitoramento não estão implementados.
