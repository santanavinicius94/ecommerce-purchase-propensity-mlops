# E-commerce Purchase Propensity

Projeto de engenharia de machine learning para estimar a propensão de compra em sessões de e-commerce. 

## Estágio atual

Organização, dados brutos versionados, documentação e validação de entrada.

## Execução local

Requisito: Python 3.11 ou superior. Esta etapa utiliza somente a biblioteca padrão; não há dependências externas para instalar.

```bash
git clone https://github.com/santanavinicius94/ecommerce-purchase-propensity-mlops.git
cd ecommerce-purchase-propensity-mlops
git switch feature/initial-project-setup
python -m venv .venv
```

Ative o ambiente no PowerShell com `.venv\Scripts\Activate.ps1` ou no Linux/macOS com `source .venv/bin/activate`. Na raiz, execute:

```bash
python scripts/validate_data.py
python -m unittest discover -s tests -v
```

O validador imprime um resumo JSON, sem modificar os dados. Sucesso retorna código zero; entrada inválida retorna código diferente de zero e uma mensagem de erro.

## Organização

```text
data/raw/online_shoppers_intention.csv  # Dados originais
docs/data_dictionary.md                # Contrato dos campos
docs/roadmap.md                        # Próximas etapas
scripts/validate_data.py               # Validação e perfil inicial
tests/test_validate_data.py            # Testes do contrato de entrada
```