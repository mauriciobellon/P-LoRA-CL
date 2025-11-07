# P-LoRA-CL - Status do Projeto

## Concluídos

1. **Modelo Base** (`src/plora_cl/models/base_model.py`)
   - Carrega transformers via `AutoModel` (sem cabeça de classificação)
   - Gerencia cabeças de classificação separadas por tarefa
   - Modelo base congelado por padrão

2. **Gerenciador de Adaptadores LoRA** (`src/plora_cl/models/lora_adapters.py`)
   - Detecção automática de módulos alvo (DistilBERT, BERT, etc.)
   - Criação de `PeftModel` separado por tarefa
   - Ativação e gerenciamento de adaptadores
   - Sistema de freeze/unfreeze de adaptadores

3. **O-LoRA** (`src/plora_cl/models/orthogonal_lora.py`)
   - Cálculo de perda ortogonal entre adaptadores
   - Acesso a adaptadores de tarefas anteriores
   - Extração de pesos LoRA (matrizes A e B)

4. **EWC** (`src/plora_cl/training/ewc.py`)
   - Cálculo da matriz de informação de Fisher
   - Cálculo da perda EWC
   - Suporte a Online EWC

5. **Replay Gerativo** (`src/plora_cl/training/replay.py`)
   - Estrutura básica implementada
   - Geração de exemplos sintéticos é placeholder (precisa implementar)

6. **Trainer** (`src/plora_cl/training/trainer.py`)
   - Loop de treinamento completo
   - Suporte a EWC, O-LoRA, Replay
   - Avaliação contínua em tarefas anteriores
   - Logging detalhado com progresso em tempo real
   - Gradient accumulation e warmup

7. **Métricas** (`src/plora_cl/evaluation/metrics.py`)
   - ACC (Average Accuracy)
   - BWT (Backward Transfer)
   - FWT (Forward Transfer)
   - Forgetting
   - Matriz de performance R_{i,j}

8. **CLI** (`src/plora_cl/cli/train.py`)
   - Interface de linha de comando
   - Suporte a configuração via argumentos ou YAML
   - Validação de argumentos

9. **Datasets** (`src/plora_cl/data/datasets.py`)
   - Configurações para 5 tarefas (AG News, Yelp, Amazon, DBPedia, Yahoo)
   - Carregamento via HuggingFace Datasets
   - Processamento e tokenização

10. **Experiment Tracking** (`src/plora_cl/evaluation/tracker.py`)
    - Salvamento de configurações
    - Logging de métricas
    - Rastreamento de custos computacionais

## Pendências

### Alta Prioridade

1. **Implementar geração de exemplos no Replay**
   - Arquivo: `src/plora_cl/training/replay.py`
   - Método: `PseudoReplayGenerator.generate_samples()`
   - Usar modelo base em modo geração

### Média Prioridade

1. **Implementar Conexões Laterais**
   - Adicionar lógica no `forward` do modelo
   - Integrar no loop de treinamento
   - Testar ablação com/sem conexões

2. **Implementar Joint Training Baseline**
   - Arquivo: `src/plora_cl/training/baselines.py`
   - Misturar dados de todas as tarefas
   - Treinar simultaneamente

3. **Scripts de Visualização**
   - Gerar gráficos de desempenho
   - Tabelas de métricas
   - Matrizes de confusão