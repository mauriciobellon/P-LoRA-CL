# Listas

## Lista de Figuras

**Figura 1** - Diagrama da arquitetura híbrida proposta: estrutura modular PNN com adaptadores LoRA ortogonais
**Figura 2** - Representação esquemática das conexões laterais entre módulos de tarefas
**Figura 3** - Visualização dos subespaços ortogonais de adaptação (O-LoRA)
**Figura 4** - Fluxo de treinamento sequencial com aplicação de EWC e replay gerativo
**Figura 5** - Gráfico de acurácia ao longo das tarefas para diferentes métodos
**Figura 6** - Taxa de esquecimento por tarefa comparando baselines e proposta
**Figura 7** - Visualização de Backward Transfer (BWT) e Forward Transfer (FWT)
**Figura 8** - Comparação de métodos: Average Accuracy vs. número de tarefas
**Figura 9** - Crescimento paramétrico por tarefa (parâmetros adicionais vs. tarefas)
**Figura 10** - Gráficos de ablação mostrando contribuição individual de cada componente

## Lista de Tabelas

**Tabela 1** - Comparação qualitativa de métodos de aprendizado contínuo (PNN, LoRA, EWC, replay gerativo)
**Tabela 2** - Configurações experimentais: hiperparâmetros, modelos base e datasets utilizados
**Tabela 3** - Resultados por tarefa: acurácia, F1-score e taxa de esquecimento
**Tabela 4** - Métricas agregadas: Average Accuracy, BWT, FWT e Remembering
**Tabela 5** - Custos computacionais: parâmetros adicionais por tarefa, tempo de treino e pico de VRAM
**Tabela 6** - Resultados de ablação: efeito de cada componente (O-LoRA, EWC, replay, conexões laterais)
**Tabela 7** - Comparação com baselines: fine-tuning sequencial, LoRA único e joint training
**Tabela 8** - Configurações detalhadas de hiperparâmetros por componente

## Lista de Algoritmos

**Algoritmo 1** - Treinamento sequencial com O-LoRA e imposição de ortogonalidade
**Algoritmo 2** - Cálculo da matriz de informação de Fisher para EWC
**Algoritmo 3** - Geração de exemplos sintéticos para replay gerativo
**Algoritmo 4** - Protocolo de avaliação cumulativa após cada tarefa
**Algoritmo 5** - Integração completa do pipeline de aprendizado contínuo

## Lista de Abreviaturas e Siglas

- **ACC**: Average Accuracy / Acurácia Média
- **BWT**: Backward Transfer / Transferência Retrógrada
- **CL**: Continual Learning / Aprendizado Contínuo
- **EWC**: Elastic Weight Consolidation / Consolidação Elástica de Pesos
- **FWT**: Forward Transfer / Transferência Proativa
- **LoRA**: Low-Rank Adaptation / Adaptação de Baixo Ranque
- **NLP**: Natural Language Processing / Processamento de Linguagem Natural
- **O-LoRA**: Orthogonal LoRA / LoRA Ortogonal
- **PLN**: Processamento de Linguagem Natural
- **PNN**: Progressive Neural Networks / Redes Neurais Progressivas
- **REM**: Remembering / Retenção
