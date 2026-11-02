# Roadmap

## 1. Fundação — entrega atual

- Estrutura mínima, documentação e dataset bruto.
- Validação por linha de comando e testes de entradas inválidas.
- Pré-processamento e regressão logística com tracking MLflow implementados.
- Testes de split, transformações, inferência, serialização e tracking isolado.
- Docker multi-stage com comandos de validação e treinamento documentados.

## 2. Exploração e avaliação

- Analisar distribuições, desbalanceamento e duplicatas.
- Definir o momento da previsão e investigar vazamento de informação.
- Decidir o tratamento de duplicatas antes do split, evitando registros idênticos entre treino e teste.
- Definir treino/validação/teste estratificados com semente fixa e registrar limitações temporais.
- Confirmar a métrica principal e reservar o conjunto de teste.

Checkpoint: análise reproduzível e decisões registradas, preservando os dados brutos.

## 3. Baseline e predições offline

- Adicionar dependências de análise e modelagem com versões registradas.
- Modularizar preparação, treino e avaliação; ajustar transformações somente no treino.
- Comparar baseline simples e classificador no mesmo protocolo.
- Avaliar Average Precision, precisão, recall, F1 e matriz de confusão; selecionar limiar na validação.
- Demonstrar predições com script batch ou notebook documentado.

Checkpoint: execução reproduzível dos dados às predições, sem API obrigatória.

## 4. Evolução de MLOps

- Migrar o remote local DVC para armazenamento compartilhado quando necessário.
- Implementar o pipeline de execução em `dvc.yaml` e o MLflow Model Registry.
- Automatizar testes e build Docker em CI.
- Avaliar orquestração adicional somente quando houver requisitos operacionais.

Checkpoint: reprodução em ambiente limpo e demonstração do versionamento. Cada incremento terá sua branch e revisão.
