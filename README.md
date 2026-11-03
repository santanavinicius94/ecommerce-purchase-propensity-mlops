# E-commerce Purchase Propensity

Pipeline de Machine Learning para estimar a propensão de compra em sessões de e-commerce. O fluxo usa DVC para dados e execução reproduzível, MLflow para tracking e Docker para execução em container.

## Pré-requisitos

- Git, Python 3.12 e [uv](https://docs.astral.sh/uv/)
- Acesso ao storage DVC compartilhado pela equipe
- Docker (opcional, para execução em container)

## Instalação e reprodução local

Clone o repositório, instale as dependências fixadas no lock file e recupere o dataset:

```bash
git clone <URL_DO_REPOSITORIO>
cd <DIRETORIO_DO_REPOSITORIO>
uv sync --locked

# Crie o storage DVC local e informe seu caminho ao repositório.
mkdir -p ../dvc-storage
uv run dvc remote modify --local local_remote ../dvc-storage
```

Preencha o storage com os objetos DVC fornecidos pela equipe. Em seguida, recupere os dados e reproduza o pipeline:

```bash
uv run dvc pull
uv run dvc repro
```

O pipeline executa `prepare`, `train` e `evaluate`. Os resultados ficam em `models/`, `reports/generated/` e `mlruns/`.

## Executar `dvc repro`

Use este comando após alterar dados, parâmetros ou código rastreado pelo DVC:

```bash
uv run dvc status
uv run dvc repro
```

Para executar todas as etapas novamente, inclusive as que já estão atualizadas:

```bash
uv run dvc repro --force
```

## Testar modelos e parâmetros

Altere a seção `model` em `params.yaml` e execute `uv run dvc repro`. O DVC identifica a mudança e refaz as etapas necessárias.

Regressão Logística:

```yaml
model:
  name: logistic_regression
  type: logistic_regression
  random_state: 7
  class_weight: balanced
```

Random Forest:

```yaml
model:
  name: random_forest
  type: random_forest
  random_state: 7
  n_estimators: 200
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  class_weight: balanced
  n_jobs: -1
```

## Visualizar experimentos no MLflow

Inicialize o servidor:

```bash
uv run mlflow server --port 5000
```

Abra <http://127.0.0.1:5000>. Cada run registra parâmetros, métricas de validação cruzada e métricas do conjunto de teste.

## Criar e executar a imagem Docker

Construa a imagem:

```bash
docker build -t ecommerce-propensity:local .
```

Com os dados já recuperados pelo DVC no host, execute a imagem com o dataset montado:

```bash
docker run --rm \
  --mount type=bind,source="$(pwd)/data/raw",target=/app/data/raw,readonly \
  ecommerce-propensity:local
```

## Executar uma predição

`scripts/predict.py` carrega `models:/online_shoppers_intention@champion`. Antes de executá-lo, registre o modelo no MLflow com esse nome e atribua a ele o alias `champion` na interface do MLflow.

```bash
uv run python scripts/predict.py
```

O script imprime a predição de um registro de exemplo.

## Verificações

```bash
uv run python -m pytest -q
uv run ruff check .
```
