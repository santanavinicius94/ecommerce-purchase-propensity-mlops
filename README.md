# E-commerce Purchase Propensity

Projeto de engenharia de machine learning para estimar a propensão de compra em sessões de e-commerce. 

## Estágio atual

Organização, dados brutos versionados, documentação e validação de entrada.

## Execução local

Requisito: Python 3.12 ou superior, conforme `pyproject.toml`. O validador e seus testes utilizam somente a biblioteca padrão; as dependências de ML estão declaradas em `pyproject.toml` e `uv.lock`.

```bash
git clone https://github.com/santanavinicius94/ecommerce-purchase-propensity-mlops.git
cd ecommerce-purchase-propensity-mlops
python -m venv .venv
```

Ative o ambiente no PowerShell com `.venv\Scripts\Activate.ps1` ou no Linux/macOS com `source .venv/bin/activate`. Na raiz, execute:

```bash
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

O validador imprime um resumo JSON, sem modificar os dados. Sucesso retorna código zero; entrada inválida retorna código diferente de zero e uma mensagem de erro.

O CSV é rastreado pelo DVC. Em um clone novo, recupere os dados antes da validação com `uv run dvc pull`, após configurar uv e o acesso ao remote com a dupla.

## Execução com Docker

O [guia de Docker](docs/docker.md) explica o Dockerfile multi-stage, os pré-requisitos,
os comandos de build e execução no PowerShell e a integração com DVC/MLflow.
A imagem usa Python 3.12 e `uv.lock`; executa a validação por padrão, recebendo
os dados por uma pasta montada em modo somente leitura.

## Organização

```text
data/raw/online_shoppers_intention.csv  # Dados originais
docs/data_dictionary.md                # Contrato dos campos
docs/roadmap.md                        # Próximas etapas
scripts/validate_data.py               # Validação e perfil inicial
tests/test_validate_data.py            # Testes do contrato de entrada
```
