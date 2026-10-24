# E-commerce Purchase Propensity

Projeto de engenharia de machine learning para estimar a propensão de compra em sessões de e-commerce. O alvo é `Revenue`: `TRUE` indica compra e `FALSE`, ausência de compra.

## Estágio atual

Fundação do Tech Challenge da Fase 2: organização, dados brutos versionados, documentação e validação de entrada. Ainda não há modelo treinado nem métricas de desempenho. MLflow, DVC, Docker e Kubernetes serão introduzidos em etapas posteriores.

## Execução local

Requisito: Python 3.11 ou superior. Esta etapa utiliza somente a biblioteca padrão; não há dependências externas para instalar.

```bash
git clone https://github.com/santanavinicius94/ecommerce-purchase-propensity-mlops.git
cd ecommerce-purchase-propensity-mlops
git switch codex/initial-project-setup
python -m venv .venv
```

Ative o ambiente no PowerShell com `.venv\Scripts\Activate.ps1` ou no Linux/macOS com `source .venv/bin/activate`. Na raiz, execute:

```bash
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

O validador imprime um resumo JSON, sem modificar os dados. Sucesso retorna código zero; entrada inválida retorna código diferente de zero e uma mensagem de erro. Para outro arquivo: `python scripts/validate_data.py --data caminho/arquivo.csv`.

## Organização

```text
data/raw/online_shoppers_intention.csv  # Dados originais
docs/data_dictionary.md                # Contrato dos campos
docs/roadmap.md                        # Próximas etapas
scripts/validate_data.py               # Validação e perfil inicial
tests/test_validate_data.py            # Testes do contrato de entrada
CONTRIBUTING.md                        # Fluxo de branches
```

O CSV contém 12.330 sessões, 17 atributos e o alvo. Há 1.908 compras (15,47%), nenhum campo vazio e 125 linhas duplicadas além da primeira ocorrência. As duplicatas foram preservadas; seu tratamento deve ser decidido antes da divisão dos dados. Consulte a [proveniência](data/raw/README.md) e o [dicionário](docs/data_dictionary.md).

Propomos Average Precision como métrica principal futura, acompanhada de precisão, recall, F1 e matriz de confusão. A escolha será confirmada conforme o custo dos erros do negócio. Não há meta numérica estabelecida; acurácia isolada não será suficiente diante do desbalanceamento.

## Orientação do professor

> para o TC da Fase 2, não é necessário desenvolver uma API (como FastAPI ou Flask) para servir o modelo. O foco central desta etapa é a engenharia de machine learning e a orquestração da infraestrutura, e não o serving em tempo real. Vocês podem realizar a validação e a apresentação das predições de forma mais direta:
>
> - Scripts Python (Batch): Um arquivo estruturado (ex: predict.py ou evaluate.py) que carrega o artefato do modelo, executa a inferência nos dados de teste e exibe ou salva os resultados.
> - Jupyter Notebook: Um notebook limpo, documentado e com o código modularizado demonstrando a execução das predições.

## Desenvolvimento

Novas branches partem de `develop` e retornam por pull request. `main` recebe entregas revisadas a partir de `develop`. Consulte o [fluxo de contribuição](CONTRIBUTING.md) e o [roadmap](docs/roadmap.md).
