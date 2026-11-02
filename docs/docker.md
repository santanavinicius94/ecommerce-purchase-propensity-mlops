# Docker no projeto

## Imagem e multi-stage build

O Dockerfile empacota Python 3.12, dependências e código para tarefas batch. Um container executa uma tarefa e termina. A validação é o comando padrão; o treino substitui esse comando.

O estágio `builder` usa uv 0.12.5 e `uv sync --locked --no-dev` para instalar as dependências de `uv.lock`. Primeiro instala apenas dependências, permitindo reutilizar cache quando o código muda. Depois instala o pacote com `--no-editable`, sem depender dos fontes originais.

O estágio `runtime` copia a `.venv` e os scripts. Usa a mesma base Linux/Python, executa como UID 10001 e não inclui uv, pytest ou Ruff. As dependências principais, incluindo DVC e ipykernel, continuam instaladas. Multi-stage não remove dependências de execução.

A `.dockerignore` exclui dados, ambientes locais, `.git`, `.dvc` e segredos do contexto. O CSV entra por mount; dados e resultados não ficam incorporados à imagem.

## Build e validação no PowerShell

Requisito: Docker Desktop em execução com containers Linux. Execute na raiz; o primeiro build precisa de internet.

```powershell
docker version
docker build --progress=plain -t ecommerce-propensity:local .
docker run --rm ecommerce-propensity:local python scripts/validate_data.py --help
```

Recupere o CSV com DVC no host, conforme o README. O container não contém o checkout DVC e não executa `dvc pull` automaticamente.

```powershell
Test-Path ./data/raw/online_shoppers_intention.csv
$rawDataPath = (Resolve-Path ./data/raw).Path
docker run --rm --mount "type=bind,source=$rawDataPath,target=/app/data/raw,readonly" ecommerce-propensity:local
$LASTEXITCODE
```

Espera-se JSON e código 0 para dados válidos. Arquivo ausente ou inválido retorna 1. Em Linux/macOS, use `source=$(pwd)/data/raw` no mount. `--rm` remove o container, preservando a imagem e os arquivos em volumes.

## Treinamento com persistência

Monte uma pasta gravável separada para banco MLflow, artefatos e modelo:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
$outputPath = (Resolve-Path ./artifacts).Path
$rawDataPath = (Resolve-Path ./data/raw).Path
docker run --rm --mount "type=bind,source=$rawDataPath,target=/app/data/raw,readonly" --mount "type=bind,source=$outputPath,target=/outputs" --workdir /outputs --env MLFLOW_TRACKING_URI=sqlite:////outputs/mlflow.db ecommerce-propensity:local python /app/scripts/train_model.py --data /app/data/raw/online_shoppers_intention.csv --model /outputs/models/logistic_regression.joblib
```

O diretório de trabalho `/outputs` mantém os artefatos locais do MLflow no armazenamento persistente. O caminho SQLite absoluto usa quatro barras. Em Linux, a pasta montada deve permitir escrita pelo UID 10001. Arquivos mantidos apenas no container desaparecem com `--rm`.

Para visualizar o tracking no mesmo ambiente de caminhos:

```powershell
docker run --rm --mount "type=bind,source=$outputPath,target=/outputs" --workdir /outputs -p 127.0.0.1:5000:5000 ecommerce-propensity:local mlflow ui --backend-store-uri sqlite:////outputs/mlflow.db --host 0.0.0.0 --port 5000
```

Abra http://127.0.0.1:5000. Para um servidor externo, configure `MLFLOW_TRACKING_URI` e o acesso aos artefatos; `localhost` dentro do container se refere ao próprio container. Model Registry e pipeline DVC ainda não estão implementados.

## Verificação

```powershell
docker run --rm ecommerce-propensity:local id -u
docker run --rm ecommerce-propensity:local python -c "import pandas, sklearn, mlflow, dvc; import ecommerce_purchase_propensity_mlops.training; print('Imports OK')"
docker run --rm ecommerce-propensity:local python scripts/train_model.py --help
uv run --locked python -m pytest -q
```

Espera-se UID 10001, imports sem erro e ajuda da CLI sem executar treino. A suíte completa roda no ambiente de desenvolvimento; pytest não é instalado na imagem de execução. O smoke test de treino exige o CSV e o volume gravável descritos acima.

## Limitações

O lock fixa dependências Python; a tag da imagem base pode receber atualizações. Reprodutibilidade estrita exige fixar imagens por digest e controlar ferramentas de build. O ambiente ainda inclui dependências de notebook; separá-las requer uma revisão específica das dependências. Esta é uma execução batch acadêmica, sem serving, monitoramento ou automação de build em CI.

Referências oficiais: [multi-stage builds](https://docs.docker.com/build/building/multi-stage/), [uv com Docker](https://docs.astral.sh/uv/guides/integration/docker/) e [autologging do MLflow](https://mlflow.org/docs/latest/ml/tracking/autolog/).
