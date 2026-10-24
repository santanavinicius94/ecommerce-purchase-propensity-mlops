# Dicionário de dados

Cada linha representa uma sessão. O contrato mantém nomes e ordem das 18 colunas. Códigos numéricos de categorias não indicam ordem de importância.

| Campo | Tipo esperado | Descrição |
| --- | --- | --- |
| Administrative | Inteiro não negativo | Páginas administrativas visitadas |
| Administrative_Duration | Real não negativo | Tempo nessas páginas |
| Informational | Inteiro não negativo | Páginas informativas visitadas |
| Informational_Duration | Real não negativo | Tempo nessas páginas |
| ProductRelated | Inteiro não negativo | Páginas de produtos visitadas |
| ProductRelated_Duration | Real não negativo | Tempo nessas páginas |
| BounceRates | Real entre 0 e 1 | Taxa de rejeição |
| ExitRates | Real entre 0 e 1 | Taxa de saída |
| PageValues | Real não negativo | Valor associado às páginas |
| SpecialDay | Real entre 0 e 1 | Proximidade de data especial |
| Month | Categoria | Mês da sessão |
| OperatingSystems | Inteiro positivo, categoria | Sistema operacional |
| Browser | Inteiro positivo, categoria | Navegador |
| Region | Inteiro positivo, categoria | Região |
| TrafficType | Inteiro positivo, categoria | Origem do tráfego |
| VisitorType | Categoria | Tipo de visitante |
| Weekend | Booleano textual | Fim de semana |
| Revenue | Booleano textual, alvo | Ocorrência de compra |

Booleanos: `TRUE` e `FALSE`. Meses observados: `Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec`; o validador também aceita `Jan` e `Apr`. Visitantes: `New_Visitor`, `Returning_Visitor` e `Other`.

São rejeitados campos vazios, quantidade incorreta de colunas, números não finitos, domínios inválidos e arquivos sem registros. Duplicatas são reportadas, sem remoção. Essa validação não substitui a análise exploratória.

Antes da modelagem, avaliar a disponibilidade das variáveis no momento da decisão, especialmente `PageValues` e agregados da sessão. Excluir informações posteriores à previsão. `Revenue` nunca deve entrar nas features.

Descrições resumidas a partir da [documentação da UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset); regras de validação são decisões iniciais do projeto.
