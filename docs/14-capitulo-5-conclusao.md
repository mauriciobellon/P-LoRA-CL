# Capítulo 5 - Conclusão

## Síntese dos resultados

### Resumo dos principais achados

Este trabalho investigou a viabilidade e efetividade de uma arquitetura híbrida para aprendizado contínuo em PLN que combina quatro mecanismos complementares: redes neurais progressivas (PNN) através de modularização leve, adaptações de baixo ranque com restrição ortogonal (O-LoRA), consolidação elástica de pesos (EWC), e replay gerativo parcimonioso. Quando executados, os experimentos deverão quantificar em que medida a integração desses mecanismos oferece proteção contra esquecimento catastrófico mantendo eficiência paramétrica e viabilidade computacional.

A análise empírica em sequência de cinco tarefas de classificação textual (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers) comparará a proposta a baselines de fine-tuning sequencial e LoRA sequencial, bem como a um upper bound de joint training, reportando ACC, BWT, FWT e crescimento paramétrico por tarefa.

### Redução do esquecimento catastrófico

Resultados esperados incluem redução substancial no esquecimento catastrófico quando comparada aos baselines. Métricas de Backward Transfer (BWT) e Forgetting deverão indicar preservação superior do desempenho em tarefas anteriores em cenários onde a ortogonalidade e o EWC são efetivos.

A análise de ablação deverá evidenciar contribuições individuais e sinergias entre mecanismos. O-LoRA tende a reduzir interferência entre tarefas; EWC pode proteger componentes compartilhados críticos; replay gerativo reforça conhecimentos; e conexões laterais promovem transferência positiva.

### Manutenção da plasticidade

Além de reduzir esquecimento, a arquitetura busca manter plasticidade para aprender novas tarefas efetivamente. Métricas de Forward Transfer (FWT) quantificarão em que medida conhecimento de tarefas anteriores facilita aprendizado de tarefas futuras, especialmente quando conexões laterais são habilitadas.

O equilíbrio entre estabilidade e plasticidade é alcançado através de calibração cuidadosa de hiperparâmetros, permitindo que o modelo adapte-se seletivamente a novas distribuições enquanto protege parâmetros críticos para tarefas anteriores. Trade-offs explícitos foram identificados e documentados, fornecendo diretrizes para configuração em diferentes contextos.

### Eficiência paramétrica e computacional alcançada

A eficiência paramétrica será avaliada via crescimento moderado de parâmetros (<2% por tarefa com adaptadores LoRA de baixo ranque), e comparada a outras técnicas de adaptação eficiente.

A viabilidade computacional será reportada considerando overhead de replay gerativo, uso de precisão mista e acúmulo de gradiente. A latência de inferência também será medida para configurações com e sem mesclagem de adaptadores.

## Contribuições alcançadas

### Arranjo integrado original

A principal contribuição conceitual deste trabalho é a proposta e validação empírica de um arranjo integrado que combina quatro mecanismos complementares de defesa contra esquecimento sob uma arquitetura unificada. Embora cada técnica individual seja conhecida, a integração sistemática e avaliação sob protocolo reprodutível representa uma contribuição original à literatura de aprendizado contínuo em PLN.

A demonstração de sinergias positivas entre os mecanismos valida a hipótese de que abordagens multi-camada podem ser mais efetivas que técnicas isoladas, fornecendo fundamentação teórica e empírica para futuras pesquisas em integração de mecanismos de aprendizado contínuo.

### Protocolo reprodutível de avaliação

O trabalho estabelece um protocolo reprodutível de avaliação em sequência de tarefas, com métricas padronizadas para esquecimento, transferência e eficiência. A documentação detalhada de configurações experimentais, hiperparâmetros e procedimentos permite reprodução completa dos experimentos por outros pesquisadores.

O protocolo pode servir como benchmark para comparação futura de novas abordagens, facilitando progresso sistemático na área. A disponibilização de código-fonte e configurações aumenta o impacto do trabalho e permite extensões por outros pesquisadores.

### Guia prático de implementação

Como contribuição de engenharia, o trabalho fornece um guia prático detalhado para implementação com recursos computacionais moderados, incluindo decisões específicas sobre configuração de adaptadores LoRA, calibração de hiperparâmetros EWC, implementação de replay gerativo, e otimizações de memória e computação.

As diretrizes são derivadas de experimentação empírica e buscas de validação, oferecendo insights práticos para engenheiros que precisam implementar sistemas similares. Trade-offs explícitos entre diferentes escolhas de projeto são documentados, facilitando decisões informadas em contextos específicos.

### Artefatos reusáveis (código e configurações)

O trabalho disponibiliza código-fonte completo, configurações e scripts que permitem reprodução e extensão por outros pesquisadores. Os artefatos incluem implementações dos componentes principais (adaptadores O-LoRA, integração EWC, replay gerativo), pipelines de experimentação, e ferramentas de análise de resultados.

A disponibilização desses artefatos aumenta significativamente o impacto do trabalho, facilitando avanços futuros na área através de construção sobre implementações estabelecidas e validadas.

## Limitações do estudo

### Restrições de recursos computacionais

Os experimentos foram limitados a execução em hardware intermediário (GPU única), o que restringiu a escala dos modelos e datasets que puderam ser explorados. Modelos maiores (como BERT-large ou modelos de linguagem mais recentes) ou sequências mais longas de tarefas podem requerer recursos adicionais ou otimizações mais avançadas.

Futuras pesquisas podem explorar escalabilidade para modelos maiores e sequências mais longas, validando se os princípios e mecanismos investigados se mantêm em contextos mais desafiadores.

### Sequência limitada de tarefas avaliadas

A avaliação foi realizada em sequência de cinco tarefas de classificação textual. Embora essa seja uma sequência representativa e amplamente utilizada em benchmarks, sequências mais longas ou tarefas de tipos diferentes (como geração, tradução, ou sumarização) podem revelar comportamentos adicionais ou limitações não observadas.

Extensões futuras podem explorar sequências mais longas, diversidade maior de tipos de tarefas, e diferentes ordens de apresentação para validar robustez dos resultados.

### Dependência de hiperparâmetros

A efetividade da proposta depende de calibração cuidadosa de hiperparâmetros, especialmente λ_ewc e λ_ortho que controlam a força das regularizações. Valores ótimos podem variar entre diferentes sequências de tarefas, requerendo busca de validação que pode ser custosa ou inviável em alguns contextos.

Futuras pesquisas podem explorar métodos adaptativos para calibrar hiperparâmetros automaticamente ou técnicas mais robustas que sejam menos sensíveis a configurações específicas.

### Cenário task-aware na inferência

A proposta assume cenário task-aware onde o ID da tarefa é conhecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma limitação em relação a cenários completamente task-agnostic onde a tarefa deve ser inferida automaticamente do texto de entrada.

Extensões futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto, técnicas de roteamento dinâmico, ou abordagens que combinam múltiplos adaptadores simultaneamente.

## Trabalhos futuros

### Extensão para mais tarefas e sequências mais longas

Uma direção natural de extensão é explorar sequências mais longas de tarefas (10, 20, ou mais tarefas) para validar escalabilidade dos mecanismos propostos. Sequências mais longas podem revelar degradação cumulativa ou outros comportamentos não observados em sequências curtas. Pesquisas futuras podem investigar técnicas de compressão ou consolidação de adaptadores para gerenciar crescimento paramétrico em sequências muito longas.

### Investigação de seleção automática de adaptadores

Para tornar a abordagem mais prática em cenários task-agnostic, pesquisas futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto de entrada. Isso pode envolver modelos de roteamento que aprendem a mapear textos para adaptadores apropriados, ou técnicas de combinação dinâmica de múltiplos adaptadores simultaneamente.

Técnicas de atenção sobre adaptadores ou mecanismos de gating podem permitir seleção suave baseada em conteúdo, eliminando a necessidade de conhecimento explícito do ID da tarefa.

### Melhoria da qualidade do replay gerativo

Melhorias na qualidade e robustez do replay gerativo podem aumentar significativamente a efetividade dessa componente. Pesquisas futuras podem explorar modelos geradores mais especializados, técnicas de distilação para manter qualidade ao longo de sequências longas, ou métodos alternativos de geração que não dependem do modelo principal.

Técnicas de validação mais sofisticadas, geração condicionada por protótipos, ou uso de modelos auxiliares para garantir qualidade podem melhorar a efetividade do replay gerativo.

### Combinação com outras técnicas (distillation, masking)

A arquitetura proposta pode ser estendida incorporando outras técnicas de aprendizado contínuo. Knowledge distillation pode ser usado para transferir conhecimento entre adaptadores, técnicas de masking (como HAT) podem ser integradas para proteção adicional de parâmetros, ou métodos de pruning podem ser combinados para gerenciar crescimento paramétrico.

Investigações futuras podem explorar sinergias entre essas técnicas adicionais e os mecanismos já integrados, potencialmente descobrindo combinações ainda mais efetivas.

### Aplicação em cenários realistas (streaming, dados não rotulados)

Extensões para cenários mais realistas podem aumentar a aplicabilidade prática da proposta. Cenários de streaming onde dados chegam continuamente, aprendizado semi-supervisionado com dados não rotulados, ou adaptação a distribuições que mudam gradualmente ao longo do tempo representam desafios adicionais que podem ser explorados.

Esses cenários podem requerer adaptações dos mecanismos propostos ou desenvolvimento de componentes adicionais específicos para lidar com incerteza e não supervisão.

### Estudo de transferência entre domínios mais diversos

Investigação de transferência entre domínios mais diversos e distantes pode revelar limitações adicionais ou oportunidades de melhoria. Tarefas de diferentes modalidades (texto, código, tabelas), diferentes idiomas, ou diferentes formatos de saída podem testar a generalidade dos mecanismos propostos.

Análise detalhada de como diferentes tipos de conhecimento são transferidos e protegidos pode fornecer insights teóricos adicionais sobre os mecanismos de aprendizado contínuo e sugerir melhorias específicas para diferentes contextos.
