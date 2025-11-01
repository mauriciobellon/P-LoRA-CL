# Capítulo 3 - Metodologia

## Arquitetura proposta

### Modelo base (BERT-base/DistilBERT) e configuração

A arquitetura proposta utiliza modelos base de porte moderado para garantir viabilidade computacional com recursos limitados. Especificamente, empregamos BERT-base (110M parâmetros) ou DistilBERT (66M parâmetros) como modelos base pré-treinados (Devlin et al., 2019; Sanh et al., 2019). Esses modelos oferecem um bom equilíbrio entre capacidade de representação e eficiência computacional, sendo amplamente utilizados em benchmarks de aprendizado contínuo.

O modelo base é mantido majoritariamente congelado durante todo o processo de aprendizado contínuo, com apenas componentes específicos sendo parcialmente destravados para aplicação de EWC. A cabeça de classificação padrão (um classificador linear sobre a representação [CLS]) é mantida genérica e pode ser adaptada por tarefa através dos adaptadores LoRA (Hu et al., 2021). Esta configuração permite que o modelo compartilhe conhecimento linguístico fundamental enquanto especializa-se para tarefas específicas através de módulos leves.

### Estrutura modular inspirada em PNN

A arquitetura incorpora princípios de modularização progressiva inspirados em PNN (Rusu et al., 2016), mas de forma parametricamente eficiente. Em vez de adicionar colunas completas de rede para cada tarefa, adicionamos apenas conjuntos de adaptadores LoRA leves. Cada tarefa recebe seu próprio conjunto de adaptadores que são congelados após o treinamento dessa tarefa, criando isolamento estrutural similar ao das PNNs, mas com crescimento paramétrico muito menor.

A estrutura modular permite que adaptadores de tarefas anteriores sejam mantidos em memória e ativados durante a inferência conforme necessário. Essa abordagem mantém a propriedade de isolamento completa das PNNs (adaptadores anteriores não são atualizados durante treinamento de novas tarefas) enquanto reduz drasticamente o custo de armazenamento e computação.

### Adaptadores LoRA por tarefa com restrição ortogonal

Para cada nova tarefa T_k, inicializamos um novo conjunto de adaptadores LoRA que será treinado especificamente para essa tarefa (Hu et al., 2021). Os adaptadores são injetados nas projeções de atenção (Q, K, V, O) e/ou nas camadas feed-forward do modelo base, dependendo da configuração escolhida. Utilizamos ranks reduzidos (r = 4 a 8) para manter a eficiência paramétrica, resultando em overhead típico de 0,1% a 2% dos parâmetros do modelo base por tarefa.

Durante o treinamento dos adaptadores para T_k, impomos restrições ortogonais (O-LoRA) que garantem que os novos adaptadores ocupem subespaços distintos dos adaptadores de tarefas anteriores (T_1, ..., T_{k-1}). Isso é feito através de um termo de regularização na função de perda que penaliza projeções dos novos adaptadores nos subespaços gerados pelos adaptadores anteriores, minimizando interferência entre tarefas (inspirado em OWM/OGD; Zeng et al., 2019; Farajtabar et al., 2019).

### Conexões laterais opcionais para transferência

Para promover transferência positiva entre tarefas, implementamos conexões laterais opcionais inspiradas em PNN (Rusu et al., 2016). Essas conexões permitem que o módulo atual (adaptadores da tarefa corrente) consuma representações dos módulos anteriores (adaptadores de tarefas passadas) sem atualizá-los. As conexões podem ser implementadas através de concatenação de features, soma ponderada, ou mecanismos de atenção que aprendem a combinar informações de diferentes adaptadores.

As conexões laterais são configuráveis e podem ser habilitadas ou desabilitadas para análise de ablação, permitindo quantificar seu impacto na transferência forward e no desempenho geral. Quando habilitadas, elas adicionam um pequeno overhead computacional mas podem melhorar significativamente o desempenho inicial em novas tarefas através de aproveitamento de conhecimento prévio.

### Aplicação de EWC em componentes compartilhados

O EWC é aplicado seletivamente apenas aos componentes compartilhados que permanecem treináveis durante o aprendizado contínuo. Notadamente, aplicamos EWC aos embeddings do modelo base e, potencialmente, a algumas camadas iniciais que são parcialmente destravadas para permitir ajustes finos de conhecimento linguístico fundamental.

Após o treinamento em cada tarefa T_i, estimamos a matriz de informação de Fisher sobre os dados de T_i para identificar quais pesos são críticos para o desempenho nessa tarefa. Nas tarefas subsequentes, incorporamos um termo de penalização EWC na função de perda que desencoraja grandes mudanças nesses pesos críticos, preservando conhecimento fundamental enquanto permite ajustes necessários para novas tarefas.

### Integração de replay gerativo parcimonioso

O replay gerativo é implementado de forma parcimoniosa para minimizar custo computacional. Utilizamos um modelo gerador leve (que pode ser o próprio modelo base configurado para geração ou um modelo auxiliar) para produzir exemplos sintéticos das tarefas anteriores, seguindo a linha de Deep Generative Replay (Shin et al., 2017). Antes de cada época de treinamento na tarefa atual T_k, geramos um conjunto balanceado de exemplos sintéticos representando as tarefas T_1, ..., T_{k-1}.

Esses exemplos sintéticos são intercalados com os dados reais da tarefa atual durante o treinamento, compondo tipicamente 10-30% de cada batch. A geração é guiada por prompts estruturados que especificam a tarefa e a classe desejada, e os exemplos gerados são validados automaticamente para garantir qualidade mínima antes de serem incorporados ao treinamento.

## Protocolo experimental

### Sequência de tarefas (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers)

O protocolo experimental utiliza uma sequência de cinco tarefas de classificação amplamente utilizadas em benchmarks de aprendizado contínuo: AG News (classificação de notícias em 4 categorias), Yelp Polarity (análise de sentimento binária em avaliações), Amazon Reviews (análise de sentimento em avaliações de produtos), DBPedia (classificação de entidades em 14 categorias), e Yahoo Answers (classificação de perguntas em 10 categorias). Essas coleções são disponibilizadas e amplamente adotadas a partir do repositório de Zhang, Zhao & LeCun (2015), permitindo comparações padronizadas.

A ordem das tarefas foi escolhida para simular mudanças de domínio progressivas — partindo de notícias formais, passando por avaliações de consumidores, até dados enciclopédicos e perguntas de usuários. Essa diversidade de domínios testa a robustez do método a diferentes distribuições textuais e desafia o modelo a manter conhecimento geral enquanto especializa-se para domínios específicos.

### Preparação e pré-processamento dos dados

Cada dataset é preparado seguindo práticas padrão de pré-processamento de texto. Removemos metadados irrelevantes, normalizamos espaços em branco e caracteres especiais, e garantimos que os textos estejam em formato adequado para o tokenizador do modelo base. Para tarefas de classificação multiclasse, mantemos todas as classes originais para maximizar a diversidade do desafio.

Os datasets são divididos em conjuntos de treino, validação e teste seguindo proporções padrão (tipicamente 70/15/15 ou conforme disponibilidade dos dados originais). É importante notar que seguimos um regime exemplar-free: após o treinamento em uma tarefa, não há acesso aos dados brutos dessa tarefa, exceto pelos exemplos sintéticos gerados para replay.

### Tokenização e configurações de comprimento máximo

A tokenização segue o tokenizador do modelo base (WordPiece para BERT). Configuramos comprimentos máximos específicos para cada dataset dentro da faixa de 256 a 512 tokens, balanceando capacidade de capturar contexto completo com eficiência computacional. Valores específicos são documentados em tabela para garantir reprodutibilidade.

Tokens especiais ([CLS], [SEP]) são adicionados conforme necessário pela arquitetura do modelo. Para tarefas que requerem pares de sequências, utilizamos o formato apropriado de separação. A tokenização é realizada uma vez antes do treinamento e os resultados são armazenados para evitar reprocessamento.

### Balanceamento por amostragem estratificada

Para garantir que cada classe seja adequadamente representada durante o treinamento, aplicamos amostragem estratificada quando necessário. Isso é particularmente importante para datasets desbalanceados como DBPedia e Yahoo Answers, onde algumas classes podem ter muito mais exemplos que outras.

O balanceamento é aplicado tanto nos conjuntos de treino quanto nos exemplos sintéticos gerados para replay, garantindo que o modelo veja representação adequada de todas as classes durante o treinamento contínuo. Isso previne viés em direção a classes majoritárias e garante avaliação justa do desempenho em todas as categorias.

## Fluxo de treinamento

### Processo sequencial tarefa por tarefa

O treinamento segue um protocolo estritamente sequencial: cada tarefa é treinada completamente antes de iniciar a próxima. Não há acesso a dados futuros durante o treinamento de uma tarefa atual, simulando um cenário realista onde tarefas chegam uma de cada vez ao longo do tempo.

Para cada tarefa T_k na sequência:
1. Avaliamos o modelo em todas as tarefas anteriores (T_1, ..., T_{k-1}) para medir desempenho atual antes de iniciar T_k
2. Inicializamos novos adaptadores LoRA para T_k
3. Congelamos adaptadores de tarefas anteriores (T_1, ..., T_{k-1})
4. Treinamos os novos adaptadores sobre T_k com perda composta (incluindo termos de ortogonalidade e EWC)
5. Intercalamos exemplos sintéticos de tarefas anteriores durante o treinamento
6. Após convergência, estimamos matriz de Fisher para EWC nas tarefas futuras
7. Avaliamos o modelo em todas as tarefas vistas até então (T_1, ..., T_k)

### Inicialização e congelamento de adaptadores anteriores

Novos adaptadores LoRA são inicializados seguindo a prática padrão: matrizes A são inicializadas aleatoriamente (distribuição normal pequena) e matrizes B são inicializadas com zeros, garantindo que ΔW = 0 inicialmente e o modelo começa com o comportamento do modelo base para a nova tarefa.

Adaptadores de tarefas anteriores são completamente congelados — seus parâmetros não são atualizados durante o treinamento da tarefa atual. Isso garante isolamento estrutural e previne interferência destrutiva. Os adaptadores congelados permanecem em memória e podem ser ativados durante a inferência quando necessário para a tarefa correspondente.

### Cálculo da matriz de Fisher para EWC

Após o treinamento em cada tarefa T_i, estimamos a matriz de informação de Fisher sobre um subconjunto representativo dos dados de treino de T_i. A estimativa utiliza a aproximação diagonal (apenas elementos diagonais da matriz), reduzindo significativamente o custo computacional enquanto mantendo efetividade prática.

A matriz de Fisher é calculada avaliando o gradiente da função de perda em relação aos parâmetros sobre os dados da tarefa, e então computando F_j = E[(∂L/∂θ_j)²] para cada parâmetro θ_j. Os valores são armazenados junto com os valores dos parâmetros após treinamento (θ_j*) para uso nas tarefas subsequentes através do termo de penalização EWC.

### Intercalação de exemplos sintéticos para replay

Antes de cada época de treinamento na tarefa atual T_k, geramos um conjunto balanceado de exemplos sintéticos representando as tarefas anteriores T_1, ..., T_{k-1}. A geração é guiada por prompts estruturados que especificam a tarefa e a classe desejada, e os exemplos são gerados utilizando o modelo configurado para geração de texto.

Os exemplos sintéticos são intercalados com os dados reais da tarefa atual durante o treinamento, compondo tipicamente 10-30% de cada batch. Essa proporção é um hiperparâmetro que pode ser ajustado, mas valores muito altos podem reduzir a plasticidade para a tarefa atual, enquanto valores muito baixos podem não fornecer reforço suficiente para tarefas anteriores.

### Função de perda composta (perda da tarefa, ortogonalidade, EWC)

A função de perda total durante o treinamento da tarefa T_k é uma combinação ponderada de múltiplos termos:

L_total = L_task + λ_ortho L_ortho + λ_ewc L_ewc

onde:
- L_task é a perda padrão da tarefa (cross-entropy para classificação) sobre os dados reais e sintéticos de T_k
- L_ortho é o termo de ortogonalidade que penaliza projeções dos novos adaptadores nos subespaços dos adaptadores anteriores
- L_ewc é o termo de penalização EWC que protege pesos críticos identificados por Fisher
- λ_ortho e λ_ewc são hiperparâmetros que controlam a força de cada regularização

A perda sobre exemplos sintéticos de tarefas anteriores também contribui para L_task, reforçando conhecimentos passados enquanto aprendemos a nova tarefa.

### Hiperparâmetros (taxa de aprendizado, weight decay, ranks LoRA)

Seguimos diretrizes conservadoras para calibração de hiperparâmetros baseadas em práticas estabelecidas na literatura. Utilizamos AdamW como otimizador com taxa de aprendizado na faixa de 1e-4 a 3e-4 para adaptadores LoRA, e weight decay até 0,01. Ranks LoRA são configurados entre 4 e 8, com alpha conforme prática do PEFT (tipicamente alpha = rank ou 2*rank).

Para o EWC, o hiperparâmetro λ é calibrado através de busca em conjunto de validação, tipicamente variando entre 100 e 10000 dependendo da escala dos valores de Fisher. Para ortogonalidade, o hiperparâmetro λ_ortho é tipicamente configurado entre 0,1 e 1,0, balanceando isolamento com plasticidade.

Parâmetros de decodificação para replay gerativo (temperature, top-p) são ajustados por tarefa para garantir qualidade das gerações, e early stopping é aplicado com base na métrica de validação corrente para evitar overfitting.

## Ambiente computacional

### Framework e bibliotecas (PyTorch, HuggingFace Transformers, PEFT, Avalanche)

A implementação utiliza PyTorch como framework principal de deep learning, aproveitando sua flexibilidade para implementar componentes customizados. HuggingFace Transformers fornece modelos base pré-treinados e utilitários de tokenização, enquanto PEFT (Parameter-Efficient Fine-Tuning) oferece implementações otimizadas de LoRA e gerenciamento de adaptadores.

Avalanche (ContinualAI) é utilizado para o protocolo de aprendizado contínuo, gerenciamento de sequências de tarefas, e implementação de EWC. O framework também fornece utilitários para avaliação cumulativa e cálculo de métricas padronizadas de CL. Componentes customizados são desenvolvidos para integração de O-LoRA, replay gerativo, e conexões laterais.

### Configuração de hardware (GPU intermediária, precisão mista)

Os experimentos são projetados para execução em uma única GPU intermediária (por exemplo, NVIDIA T4 com 16GB VRAM), garantindo viabilidade para contextos acadêmicos com recursos limitados. Utilizamos precisão mista (mixed precision) através de torch.cuda.amp para reduzir uso de memória e acelerar treinamento, permitindo batch sizes maiores e reduzindo tempo de treinamento.

Gradiente checkpointing é aplicado quando necessário para modelos maiores, trocando computação por memória e permitindo processar sequências mais longas ou batches maiores dentro das limitações de VRAM disponível.

### Acúmulo de gradiente e otimização de memória

Para otimizar uso de memória e permitir batch sizes efetivos maiores, utilizamos acúmulo de gradiente (gradient accumulation). Isso permite simular batch sizes maiores sem aumentar proporcionalmente o uso de VRAM, dividindo o batch em múltiplos micro-batches e acumulando gradientes antes de atualizar parâmetros.

Outras otimizações de memória incluem: remoção de gradientes não utilizados através de torch.no_grad() quando apropriado, uso de tipos de dados eficientes (float16 onde possível), e carregamento eficiente de dados através de DataLoader com num_workers otimizado.

## Protocolo de avaliação

### Avaliação cumulativa após cada tarefa

Após o treinamento em cada tarefa T_k, avaliamos o modelo em todas as tarefas vistas até então (T_1, ..., T_k) sem re-treino. Essa avaliação cumulativa permite rastrear como o desempenho em tarefas anteriores evolui à medida que novas tarefas são aprendidas, quantificando diretamente o esquecimento e a transferência.

A avaliação é realizada sobre conjuntos de teste dedicados para cada tarefa, garantindo que não há contaminação de dados de treino. Os resultados são registrados em uma matriz de desempenho R onde R_{i,j} representa a acurácia na tarefa j após ter treinado até a tarefa i.

### Cenário task-aware (fornecimento do ID da tarefa)

Avaliamos em cenário task-aware, onde o ID da tarefa é fornecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma premissa comum em aprendizado contínuo e explicitada claramente no trabalho. Embora seja uma limitação em relação a cenários completamente task-agnostic, é uma suposição razoável para muitas aplicações práticas onde o contexto permite identificar a tarefa.

O cenário task-aware facilita a avaliação e permite focar nos mecanismos de defesa contra esquecimento sem a complexidade adicional de seleção automática de adaptadores, que pode ser explorada em trabalhos futuros.

### Métricas por tarefa (acurácia, F1)

Para cada tarefa individual, reportamos acurácia (fração de predições corretas) e F1-score (média harmônica de precisão e recall, útil para tarefas desbalanceadas). Essas métricas fornecem visão granular do desempenho em cada tarefa e permitem identificar tarefas particularmente desafiadoras ou vulneráveis ao esquecimento.

F1-score é especialmente importante para tarefas multiclasse desbalanceadas como DBPedia e Yahoo Answers, onde acurácia pode ser enganosa devido a distribuições de classe desiguais.

### Métricas agregadas (ACC, BWT, FWT, Forgetting)

Além das métricas por tarefa, reportamos métricas agregadas que sumarizam o desempenho geral do modelo:

- **Average Accuracy (ACC)**: Média das acurácias finais em todas as tarefas após treinar a sequência completa
- **Backward Transfer (BWT)**: Média das diferenças entre desempenho final e desempenho pico em cada tarefa, medindo esquecimento agregado
- **Forward Transfer (FWT)**: Média das acurácias em tarefas futuras antes de treiná-las, medindo aproveitamento de conhecimento prévio
- **Forgetting**: Taxa de esquecimento média calculada como diferença entre desempenho pico e desempenho final em cada tarefa

Essas métricas fornecem visão holística do desempenho do modelo ao longo da sequência de tarefas e permitem comparação quantitativa com outros métodos.

### Custos computacionais (tempo, VRAM, parâmetros adicionais)

Além do desempenho, quantificamos os custos computacionais de cada abordagem:

- **Tempo de treinamento**: Total e por tarefa, medido em horas ou minutos
- **Pico de VRAM**: Uso máximo de memória durante treinamento
- **Parâmetros adicionais**: Crescimento do número de parâmetros com o número de tarefas

Essas métricas são essenciais para avaliar viabilidade prática das abordagens e fazer trade-offs informados entre desempenho e eficiência.

### Múltiplas sementes e estatísticas (média ± desvio-padrão)

Todos os experimentos são executados com múltiplas sementes aleatórias (tipicamente 3) para garantir robustez dos resultados e permitir cálculo de estatísticas. Reportamos média e desvio-padrão de todas as métricas principais, permitindo avaliação da variabilidade e significância estatística das diferenças observadas.

A variação entre sementes permite identificar se ganhos observados são consistentes ou dependem de inicialização aleatória específica, aumentando confiança nos resultados reportados.

## Baselines e ablações

### Fine-tuning sequencial do modelo completo

Como baseline representativo de esquecimento catastrófico sem mitigação, treinamos o modelo completo (todos os parâmetros) sequencialmente em cada tarefa sem nenhuma proteção contra esquecimento. Esta abordagem serve como lower bound e demonstra a severidade do problema de esquecimento catastrófico no contexto estudado.

Esperamos observar degradação significativa do desempenho em tarefas anteriores à medida que novas tarefas são aprendidas, fornecendo linha de base para comparar a efetividade das técnicas propostas.

### LoRA único sequencial

Como baseline intermediário, treinamos um único conjunto de adaptadores LoRA (sem ortogonalidade) sequencialmente reutilizado para todas as tarefas. Esta abordagem demonstra a eficiência paramétrica do LoRA mas também mostra que LoRA puro não resolve esquecimento quando usado sequencialmente.

Comparação com este baseline permite quantificar o valor adicional da ortogonalidade (O-LoRA) em reduzir interferência entre tarefas.

### Joint training (upper bound)

Como upper bound teórico, treinamos o modelo simultaneamente em todas as tarefas com acesso completo a todos os dados (joint training). Esta abordagem não é viável em cenários de aprendizado contínuo real, mas fornece referência do desempenho máximo possível se não houvesse restrições de dados e tempo.

A diferença entre joint training e os métodos de aprendizado contínuo quantifica o custo de aprender sequencialmente versus simultaneamente, e permite avaliar quão próximos os métodos propostos estão do limite teórico.

### Ablações seletivas (sem O-LoRA, sem EWC, sem replay, sem conexões laterais)

Para isolar a contribuição individual de cada componente, realizamos ablações sistemáticas onde removemos seletivamente cada mecanismo:

- **Sem O-LoRA**: Usa LoRA padrão sem restrições ortogonais
- **Sem EWC**: Remove termo de penalização EWC da função de perda
- **Sem replay**: Remove intercalação de exemplos sintéticos
- **Sem conexões laterais**: Desabilita conexões laterais entre módulos

Essas ablações permitem quantificar o valor marginal de cada componente e identificar sinergias ou redundâncias entre diferentes mecanismos. A análise de ablação é essencial para validar que a integração oferece ganhos superiores à soma das partes individuais e para identificar quais componentes são mais críticos para o desempenho final.
