# P-LoRA-CL: Progressive LoRA with Orthogonal Constraints for Continual Learning

Este projeto implementa uma arquitetura híbrida para aprendizado contínuo em PLN que combina:
- Modularização progressiva inspirada em PNN
- Adaptadores LoRA com restrições ortogonais (O-LoRA)
- Consolidação Elástica de Pesos (EWC)
- Replay gerativo parcimonioso

## Instalação

```bash
# Criar ambiente virtual
uv venv
source .venv/bin/activate

# Instalar dependências
uv pip install -e .

# Instalar com dependências de desenvolvimento
uv pip install -e ".[dev]"
```

## Uso Básico

### Treinamento

```bash
# Treinar com configuração padrão
uv run python -m plora_cl.cli.train --experiment-name baseline

# Treinar com arquivo de configuração
uv run python -m plora_cl.cli.train --config experiments/config.yaml.example

# Treinar sem EWC (ablação)
uv run python -m plora_cl.cli.train --experiment-name ablation-no-ewc --no-ewc
```

### Visualização

```bash
# Gerar gráficos e tabelas
python scripts/visualize.py --experiment-name baseline --output-dir plots
```

## Estrutura do Projeto

```
src/plora_cl/
├── data/           # Carregamento e pré-processamento de dados
├── models/         # Arquiteturas de modelo (base, LoRA, EWC)
├── training/       # Loops de treinamento e estratégias
├── evaluation/     # Métricas e tracking
└── cli/            # Interface de linha de comando

experiments/        # Configurações e resultados de experimentos
tests/              # Testes unitários
scripts/            # Scripts auxiliares (visualização)
```

## Configuração

Veja `experiments/config.yaml.example` para exemplo de configuração.

Principais parâmetros:
- `model_name`: Modelo base (padrão: distilbert-base-uncased)
- `lora_r`: Rank dos adaptadores LoRA (padrão: 8)
- `lambda_ortho`: Peso da regularização ortogonal (padrão: 0.1)
- `lambda_ewc`: Peso do EWC (padrão: 100.0)
- `replay_ratio`: Proporção de replay no batch (padrão: 0.2)

## Testes

```bash
# Executar testes
uv run pytest tests -q

# Com cobertura
uv run pytest --cov=plora_cl tests
```

## Diferenças em Relação ao Paper

Veja `docs/CODE_DIFFERENCES.md` para detalhes sobre diferenças entre implementação e paper.

## Entregáveis

Veja `docs/DELIVERABLES.md` para lista completa de gráficos e tabelas necessários para o paper.
