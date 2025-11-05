# Sistema de Checkpointing

O P-LoRA-CL implementa um sistema robusto de checkpointing que permite salvar e retomar o treinamento em qualquer ponto.

## Funcionalidades

### Salvamento Automático

O sistema salva automaticamente checkpoints durante o treinamento em dois momentos:

1. **Por número de steps**: A cada N steps de otimização (configurável via `checkpoint_every`)
2. **Fim de época**: Ao final de cada época de treinamento

### O que é Salvo

Cada checkpoint contém:

- **Estado do modelo**:
  - Pesos do modelo base
  - Todos os task heads criados até o momento
  - Estados de todos os adaptadores LoRA

- **Estado do treinamento**:
  - Estado do otimizador (AdamW)
  - Estado do scheduler (linear warmup)
  - Step global atual
  - Tarefa, época e batch atuais

- **Componentes de CL**:
  - Fisher Information Matrix (EWC)
  - Parâmetros ótimos salvos (EWC)
  - Templates de replay gerativo
  - Matriz de accuracy e F1 (métricas)

- **Metadados**:
  - Lista de tarefas já treinadas
  - Seed usado
  - Loss atual

## Uso

### Configuração Básica

```yaml
# config.yaml
checkpoint_every: 100          # Salvar a cada 100 steps
keep_last_n_checkpoints: 3     # Manter apenas os 3 últimos
```

### Via Linha de Comando

```bash
# Treinar com checkpointing
uv run python -m plora_cl.cli.train \
  --experiment-name my-experiment \
  --checkpoint-every 100 \
  --keep-last-n-checkpoints 3

# Retomar treinamento
uv run python -m plora_cl.cli.train \
  --experiment-name my-experiment \
  --resume
```

### Desabilitar Checkpointing

```bash
# Desabilitar checkpointing completamente
uv run python -m plora_cl.cli.train \
  --experiment-name my-experiment \
  --checkpoint-every 0
```

## Estrutura de Diretórios

```
experiments/
└── <experiment-name>/
    ├── checkpoints/
    │   ├── checkpoint_task0_epoch0_batch99.pt
    │   ├── checkpoint_task0_epoch0_batch199.pt
    │   └── checkpoint_task0_epoch1_batch99.pt
    ├── results/
    └── config.yaml
```

## Gerenciamento de Espaço

Para economizar espaço em disco, o sistema mantém apenas os N últimos checkpoints **por tarefa** (configurável via `keep_last_n_checkpoints`). Checkpoints mais antigos **da mesma tarefa** são automaticamente removidos, mas **checkpoints de tarefas completadas são preservados**.

Isso significa que se você tem 5 tarefas e `keep_last_n_checkpoints=3`:
- Durante o treinamento da tarefa 0: mantém os 3 últimos checkpoints da tarefa 0
- Ao completar a tarefa 0: preserva o checkpoint final da tarefa 0
- Durante o treinamento da tarefa 1: mantém os 3 últimos checkpoints da tarefa 1 + checkpoint final da tarefa 0
- E assim por diante...

### Tamanho Médio de Checkpoints

Estimativas para DistilBERT-base com LoRA (r=8):

- Modelo base: ~260 MB
- Por adapter LoRA: ~2 MB
- EWC (Fisher + params): ~260 MB
- Otimizador (AdamW): ~260 MB
- **Total por checkpoint**: ~800 MB (aumenta com mais tarefas)

Recomendações:
- Para experimentos longos: `keep_last_n_checkpoints: 2-3`
- Para debugging: `keep_last_n_checkpoints: 5-10`
- Para produção: considerar salvar checkpoints em storage externo

## Retomada de Treinamento

Ao usar `--resume`, o sistema:

1. Busca o checkpoint mais recente em `experiments/<experiment-name>/checkpoints/`
2. Carrega todos os estados salvos
3. Retoma o treinamento exatamente do ponto onde parou:
   - Mesma tarefa
   - Mesma época
   - Mesmo batch (pula batches já processados)
   - Mesmo estado do otimizador e scheduler

### Exemplo de Retomada

```bash
# Início do treinamento
uv run python -m plora_cl.cli.train --experiment-name baseline

# ... treinamento interrompido no meio ...

# Retomar
uv run python -m plora_cl.cli.train --experiment-name baseline --resume

# Output:
# Loading checkpoint from experiments/baseline/checkpoints/checkpoint_task0_epoch1_batch150.pt...
# Checkpoint loaded! Resuming from task 0, epoch 1, batch 151
```

## API Programática

Para uso avançado, os métodos de checkpoint podem ser chamados diretamente:

```python
from plora_cl.training.trainer import CLTrainer

trainer = CLTrainer(
    experiment_name="custom",
    checkpoint_every=50,
)

# Salvar checkpoint manualmente
trainer.save_checkpoint(
    task_idx=0,
    epoch=1,
    batch_idx=100,
    optimizer=optimizer,
    scheduler=scheduler,
    loss=0.5,
)

# Carregar checkpoint
latest = trainer.get_latest_checkpoint()
if latest:
    opt_state, sched_state = trainer.load_checkpoint(str(latest))
```

## Considerações

### Segurança

Os checkpoints usam `torch.load(..., weights_only=False)` para permitir o carregamento de objetos NumPy (matrizes de métricas). **Apenas carregue checkpoints de fontes confiáveis** que você mesmo criou.

### Determinismo

O sistema salva e restaura a seed usada, mas note que:
- DataLoaders podem não ser determinísticos após retomada
- CUDA pode introduzir não-determinismo mesmo com seed fixa

### Compatibilidade

Checkpoints são vinculados à configuração do experimento. Mudanças na arquitetura (ex: rank do LoRA, número de tarefas) podem tornar checkpoints incompatíveis.

### Performance

- Salvar checkpoints adiciona ~1-2 segundos de overhead por save
- O impacto no tempo total de treinamento é mínimo (<1% para checkpoint_every=100)
- Carregar checkpoints leva ~2-3 segundos

## Troubleshooting

### Checkpoint não encontrado

```
Error: No checkpoint found
```

Verifique se o diretório `experiments/<experiment-name>/checkpoints/` existe e contém arquivos `.pt`.

### Erro ao carregar checkpoint

```
Error: state_dict size mismatch
```

O checkpoint pode ser de uma configuração incompatível. Verifique se os parâmetros do modelo (lora_r, model_name, etc) são os mesmos.

### Espaço em disco insuficiente

Reduza `keep_last_n_checkpoints` ou aumente `checkpoint_every` para economizar espaço.

