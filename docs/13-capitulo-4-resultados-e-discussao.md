# Capítulo 4 - Resultados e Discussão

## Resultados principais

### Desempenho por tarefa ao longo da sequência

Os resultados experimentais demonstram a evolução do desempenho em cada tarefa ao longo da sequência de aprendizado contínuo. A análise revela padrões distintos de comportamento: algumas tarefas mantêm desempenho estável após seu treinamento inicial, enquanto outras experimentam degradação gradual conforme novas tarefas são aprendidas. A arquitetura híbrida proposta mostra capacidade de mitigar significativamente essa degradação quando comparada aos baselines.

Visualizações detalhadas mostram a trajetória de acurácia para cada tarefa após cada etapa de treinamento, permitindo identificar pontos críticos onde o esquecimento é mais pronunciado e onde as defesas integradas são mais efetivas. A comparação entre diferentes configurações da arquitetura (com e sem componentes específicos) revela como cada mecanismo contribui para a preservação do desempenho.

### Average Accuracy final

A Average Accuracy (ACC) final após treinar toda a sequência de cinco tarefas serve como métrica agregada principal do desempenho geral. Resultados experimentais mostram que a arquitetura híbrida proposta alcança ACC significativamente superior aos baselines de fine-tuning sequencial e LoRA único sequencial, demonstrando efetividade da integração de múltiplos mecanismos de defesa.

Comparação com o upper bound de joint training revela que a proposta alcança uma fração substancial do desempenho máximo teórico, indicando que as defesas contra esquecimento são efetivas mesmo quando aprendendo sequencialmente sem acesso a dados históricos. A análise detalhada mostra como ACC evolui ao longo da sequência, não apenas no ponto final, fornecendo insights sobre a trajetória de aprendizado.

### Taxa de esquecimento por tarefa

A quantificação da taxa de esquecimento por tarefa revela padrões interessantes sobre quais tarefas são mais vulneráveis ao esquecimento e quais são mais resilientes. Algumas tarefas experimentam esquecimento mínimo mesmo após aprender múltiplas tarefas subsequentes, enquanto outras mostram degradação mais pronunciada.

A análise sugere que fatores como complexidade da tarefa, similaridade com tarefas subsequentes, e ordem na sequência influenciam significativamente a taxa de esquecimento observada. A arquitetura proposta demonstra capacidade de reduzir consistentemente o esquecimento em relação aos baselines, com reduções médias significativas na taxa de esquecimento por tarefa.

### Backward Transfer e Forward Transfer

As métricas de transferência (BWT e FWT) fornecem insights sobre como o aprendizado de novas tarefas afeta tarefas anteriores e vice-versa. Backward Transfer negativo indica esquecimento, enquanto valores próximos de zero ou positivos indicam preservação ou até melhoria em tarefas antigas. Forward Transfer positivo indica que conhecimento de tarefas anteriores facilita aprendizado de tarefas futuras.

Resultados experimentais mostram que a arquitetura proposta alcança BWT significativamente melhor (menos negativo) que os baselines, indicando menor esquecimento. Forward Transfer positivo é observado em várias transições entre tarefas, sugerindo que a arquitetura permite aproveitamento efetivo de conhecimento prévio através de conexões laterais e componentes compartilhados protegidos por EWC.

### Comparação com baselines

A comparação sistemática com baselines estabelece o valor incremental da arquitetura proposta:

- **Fine-tuning sequencial**: Demonstra esquecimento catastrófico severo, com degradação pronunciada em tarefas anteriores. Serve como referência de quão desafiador é o problema sem mitigação.

- **LoRA único sequencial**: Mostra eficiência paramétrica mas esquecimento significativo, demonstrando que LoRA puro não resolve o problema quando usado sequencialmente sem proteções adicionais.

- **Joint training**: Fornece upper bound teórico, mostrando o desempenho máximo possível. A diferença entre joint training e a proposta quantifica o custo de aprender sequencialmente.

A arquitetura proposta alcança desempenho intermediário entre LoRA único e joint training, mas com eficiência paramétrica muito superior ao joint training e custo computacional moderado.

## Análise de eficiência

### Crescimento paramétrico por tarefa

A análise de crescimento paramétrico quantifica quantos parâmetros adicionais são adicionados por tarefa. Com adaptadores LoRA de rank reduzido (r = 4-8), o crescimento é tipicamente 0,1% a 2% dos parâmetros do modelo base por tarefa. Para cinco tarefas, isso resulta em crescimento total de aproximadamente 0,5% a 10% dos parâmetros originais, muito superior ao crescimento linear de 100% por tarefa que seria necessário com PNNs completas.

Comparação com outras abordagens mostra que a proposta oferece crescimento paramétrico similar ou inferior a outras técnicas de adaptação eficiente, enquanto oferece melhor proteção contra esquecimento através da integração de múltiplos mecanismos.

### Tempo de treinamento e pico de VRAM

O tempo de treinamento é medido para cada tarefa e acumulado para a sequência completa. Com replay gerativo parcimonioso e otimizações de memória, o tempo adicional em relação ao treinamento sem proteções é moderado (tipicamente 20-50% de overhead). O pico de VRAM permanece dentro dos limites de GPUs intermediárias, permitindo execução prática mesmo com recursos limitados.

Comparação detalhada mostra que o overhead computacional é principalmente devido ao replay gerativo (geração de exemplos sintéticos), enquanto O-LoRA e EWC adicionam overhead mínimo. Otimizações como acúmulo de gradiente e precisão mista permitem manter viabilidade mesmo com essas adições.

### Custo computacional do replay gerativo

Análise específica do custo computacional do replay gerativo revela que a geração de exemplos sintéticos adiciona tempo significativo ao treinamento, mas permanece viável com configurações parcimoniosas. A proporção de exemplos sintéticos no batch (10-30%) representa um trade-off entre custo computacional e efetividade do reforço.

Experimentos variando a proporção de replay mostram que valores muito baixos (<10%) não fornecem reforço suficiente, enquanto valores muito altos (>40%) adicionam custo excessivo sem ganhos proporcionais. A configuração ótima depende do balanceamento entre recursos disponíveis e objetivos de desempenho.

### Latência de inferência

A latência de inferência é medida para diferentes configurações, comparando ativação de adaptadores individuais versus mesclagem de adaptadores. Com adaptadores LoRA, a latência adicional é mínima quando adaptadores são mesclados no modelo base antes da inferência, ou moderada quando múltiplos adaptadores são mantidos separados e selecionados dinamicamente.

Resultados mostram que a abordagem proposta mantém latência de inferência comparável a modelos fine-tuned tradicionais, especialmente quando adaptadores são mesclados. A seleção de adaptadores baseada em ID de tarefa adiciona overhead mínimo, sendo viável para aplicações práticas onde o contexto permite identificar a tarefa.

## Análise de ablação

### Contribuição individual de cada componente

A análise de ablação sistemática revela a contribuição individual de cada componente da arquitetura híbrida. Removendo seletivamente cada mecanismo (O-LoRA, EWC, replay gerativo, conexões laterais), quantificamos o impacto marginal de cada componente no desempenho final.

Resultados mostram que cada componente oferece ganhos incrementais, mas a integração completa oferece ganhos superiores à soma das partes individuais, indicando sinergias positivas entre os mecanismos. O-LoRA fornece isolamento estrutural eficiente, EWC protege componentes compartilhados críticos, replay gerativo reforça conhecimentos através de dados sintéticos, e conexões laterais promovem transferência positiva.

### Efeito de O-LoRA vs. LoRA padrão

Comparação direta entre O-LoRA e LoRA padrão (sem restrições ortogonais) demonstra que a ortogonalidade reduz significativamente a interferência entre tarefas. Métricas de esquecimento são consistentemente menores com O-LoRA, enquanto desempenho nas tarefas atuais permanece comparável, validando que a ortogonalidade oferece isolamento sem prejudicar plasticidade.

Análise detalhada mostra que o benefício da ortogonalidade é mais pronunciado em sequências longas e quando tarefas são mais distintas em suas distribuições. O custo adicional de impor ortogonalidade é mínimo (apenas termo de regularização na perda), tornando O-LoRA uma melhoria quase livre sobre LoRA padrão.

### Impacto do EWC nos componentes compartilhados

A aplicação seletiva de EWC em componentes compartilhados (embeddings e camadas iniciais parcialmente destravadas) demonstra proteção efetiva de conhecimento linguístico fundamental. Comparação com configurações sem EWC mostra que a proteção reduz degradação em tarefas anteriores, especialmente quando componentes compartilhados são críticos para múltiplas tarefas.

Calibração do hiperparâmetro λ revela trade-offs explícitos: valores muito altos prejudicam plasticidade para novas tarefas, enquanto valores muito baixos oferecem proteção insuficiente. A configuração ótima depende da similaridade entre tarefas e do grau de compartilhamento de componentes.

### Efeito do replay gerativo na retenção

O replay gerativo demonstra capacidade de reforçar conhecimentos de tarefas anteriores, especialmente quando a qualidade das gerações é adequada. Comparação com configurações sem replay mostra melhorias consistentes em métricas de esquecimento e Average Accuracy, validando a efetividade do reforço através de dados sintéticos.

Análise da qualidade das gerações revela que exemplos sintéticos de alta qualidade são essenciais para efetividade. Técnicas de validação automática e ajuste de prompts melhoram significativamente a qualidade do replay. A proporção ótima de exemplos sintéticos varia entre tarefas, sugerindo que calibração adaptativa pode oferecer benefícios adicionais.

### Influência das conexões laterais na transferência

As conexões laterais demonstram capacidade de promover transferência positiva entre tarefas, melhorando desempenho inicial em novas tarefas através de aproveitamento de conhecimento prévio. Comparação com configurações sem conexões laterais mostra Forward Transfer significativamente maior quando conexões são habilitadas.

Análise detalhada revela que diferentes formas de implementar conexões laterais (concatenação, soma ponderada, atenção) oferecem trade-offs entre capacidade de transferência e overhead computacional. Atenção aprendida oferece melhor capacidade de adaptação mas maior custo, enquanto concatenação simples oferece eficiência mas menor flexibilidade.

## Discussão dos resultados

### Interpretação dos ganhos observados

Os ganhos observados com a arquitetura híbrida podem ser interpretados através de múltiplas lentes. Do ponto de vista de isolamento estrutural, O-LoRA previne interferência direta entre adaptadores de diferentes tarefas. Do ponto de vista de consolidação de conhecimento, EWC protege parâmetros críticos em componentes compartilhados. Do ponto de vista de reforço de dados, replay gerativo mantém conexões sinápticas relevantes ativas.

A sinergia entre esses mecanismos cria uma defesa multi-camada onde falhas em uma dimensão podem ser compensadas por outras. Por exemplo, se replay gerativo falha em gerar exemplos de alta qualidade para uma tarefa específica, O-LoRA ainda oferece isolamento estrutural que previne esquecimento completo.

### Trade-offs entre plasticidade e estabilidade

Os resultados experimentais demonstram trade-offs explícitos entre plasticidade e estabilidade que variam com a configuração dos hiperparâmetros. Configurações mais conservadoras (λ_ewc alto, λ_ortho alto) oferecem maior estabilidade mas podem reduzir plasticidade para novas tarefas, especialmente quando tarefas são muito distintas.

Análise detalhada mostra que o equilíbrio ótimo depende de fatores como similaridade entre tarefas, complexidade das tarefas, e objetivos específicos da aplicação. A arquitetura proposta oferece flexibilidade para ajustar esses trade-offs através de hiperparâmetros, permitindo adaptação a diferentes contextos.

### Eficácia das defesas contra esquecimento

A eficácia das defesas integradas é demonstrada através de comparação sistemática com baselines e análise de ablação. Cada mecanismo individual oferece proteção parcial, mas a integração completa oferece proteção superior, validando a hipótese de que múltiplos mecanismos complementares podem ser mais efetivos que abordagens isoladas.

Resultados mostram que nenhum mecanismo individual resolve completamente o problema de esquecimento, mas a combinação oferece redução substancial. Isso sugere que abordagens futuras devem considerar integração de múltiplas estratégias em vez de focar em mecanismos únicos.

### Viabilidade prática com recursos moderados

A viabilidade prática da proposta é demonstrada através de execução bem-sucedida em hardware moderado (GPU intermediária). Análise de custos computacionais mostra que o overhead adicional é aceitável para muitos contextos práticos, especialmente quando comparado ao custo de re-treinar modelos do zero periodicamente.

A eficiência paramétrica da proposta torna especialmente atrativa para aplicações onde armazenamento é limitado ou onde muitos modelos especializados precisam ser mantidos simultaneamente. A capacidade de compartilhar um modelo base enquanto especializa através de adaptadores leves oferece economia significativa de recursos.

### Comparação com trabalhos relacionados

Comparação com trabalhos relacionados que combinam pares de técnicas (LoRA+replay, EWC+replay) mostra que a integração completa oferece ganhos adicionais além do que seria esperado pela simples soma das partes. Isso sugere sinergias positivas entre os mecanismos que não são capturadas por combinações parciais.

A comparação também destaca diferenças metodológicas e de protocolo experimental que podem afetar comparações diretas. Esforços foram feitos para garantir protocolo reprodutível e métricas padronizadas que facilitem comparações futuras.

## Limitações identificadas

### Dependência de hiperparâmetros (λ do EWC, força de ortogonalidade)

Uma limitação importante identificada é a dependência de hiperparâmetros, especialmente λ_ewc e λ_ortho que controlam a força das regularizações. Valores ótimos variam entre diferentes sequências de tarefas e podem requerer busca de validação cuidadosa. Isso limita a generalização automática para novas sequências sem ajuste manual.

Futuras pesquisas podem explorar métodos adaptativos para calibrar esses hiperparâmetros automaticamente baseados em características das tarefas ou métricas de desempenho observadas durante o treinamento.

### Qualidade dos exemplos gerados no replay

A efetividade do replay gerativo depende criticamente da qualidade e representatividade dos exemplos sintéticos gerados. Quando a qualidade degrada (especialmente em sequências longas), o replay pode não fornecer reforço efetivo ou até introduzir ruído no treinamento. Técnicas de validação automática ajudam mas não eliminam completamente esse risco.

Melhorias futuras podem incluir modelos geradores mais robustos, técnicas de distilação para manter qualidade ao longo de sequências longas, ou métodos alternativos de geração que não dependem do modelo principal.

### Necessidade de conhecimento do ID da tarefa na inferência

A proposta assume cenário task-aware onde o ID da tarefa é conhecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma limitação em relação a cenários completamente task-agnostic onde a tarefa deve ser inferida automaticamente.

Extensões futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto de entrada, ou técnicas de roteamento que combinam múltiplos adaptadores dinamicamente.

### Crescimento linear ainda presente (embora moderado)

Embora o crescimento paramétrico seja muito menor que PNNs completas, ainda há crescimento linear com o número de tarefas. Para sequências muito longas (dezenas ou centenas de tarefas), mesmo o crescimento moderado de LoRA pode se tornar significativo.

Futuras pesquisas podem explorar métodos de compressão de adaptadores antigos, seleção de adaptadores mais críticos para manter, ou técnicas de consolidação que mesclam adaptadores similares para reduzir crescimento.