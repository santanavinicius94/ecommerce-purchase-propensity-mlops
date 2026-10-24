# Roadmap

## 1. Fundação — entrega atual

- Estrutura mínima, documentação e dataset bruto.
- Validação por linha de comando e testes de entradas inválidas.

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

- Introduzir DVC e definir armazenamento remoto de dados e artefatos.
- Introduzir MLflow para experimentos e modelos.
- Containerizar com Docker e padronizar a execução.
- Preparar Kubernetes e automações do pipeline.

Checkpoint: reprodução em ambiente limpo e demonstração do versionamento. Cada incremento terá sua branch e revisão.
