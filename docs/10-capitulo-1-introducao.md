# Capítulo 1 - Introdução

## Contextualização e motivação

### Desafio do aprendizado contínuo em PLN

O aprendizado contínuo em Processamento de Linguagem Natural (PLN) representa um desafio fundamental para sistemas de inteligência artificial que precisam operar em ambientes dinâmicos. Em aplicações reais, modelos de linguagem enfrentam sequências de tarefas heterogêneas: desde classificação de sentimento em avaliações de produtos até perguntas e respostas sobre documentação técnica, passando por análise de intenções em sistemas de atendimento. Cada nova tarefa ou domínio requer adaptação do modelo, mas a capacidade de incorporar novos conhecimentos sem degradar o desempenho em tarefas anteriormente aprendidas permanece um problema em aberto.

Diferentemente do fine-tuning tradicional, onde um modelo é ajustado uma vez sobre um conjunto fixo de dados, o aprendizado contínuo implica treinar modelos sequencialmente em múltiplas tarefas ou distribuições que chegam ao longo do tempo. Essa natureza sequencial é natural em muitos contextos: assistentes virtuais que precisam aprender novos comandos sem esquecer funcionalidades antigas, sistemas de monitoramento que devem se adaptar a novas fontes de dados, ou aplicações educacionais que incorporam progressivamente novos tópicos curriculares.

Em tais cenários, a literatura reporta que atualizações para novas tarefas frequentemente interferem com parâmetros críticos de tarefas anteriores, produzindo degradações substanciais na ausência de mecanismos de proteção (McCloskey & Cohen, 1989; French, 1999). Esse fenômeno motiva o desenvolvimento de abordagens de aprendizado contínuo que combinem isolamento estrutural, regularização informativa e replay, especialmente em arquiteturas Transformer amplamente utilizadas em PLN.

### Esquecimento catastrófico em sequências de tarefas

O principal obstáculo ao aprendizado contínuo eficaz é o fenômeno do esquecimento catastrófico, documentado desde os primórdios das redes neurais (McCloskey & Cohen, 1989; French, 1999). Quando um modelo neural é treinado sequencialmente em tarefas diferentes, os gradientes da tarefa atual tendem a sobrescrever parâmetros críticos para tarefas anteriores, levando a uma degradação abrupta do desempenho em tarefas antigas. Esse efeito é particularmente pronunciado em arquiteturas modernas de Transformers, onde milhões de parâmetros são compartilhados entre diferentes camadas e a interferência entre tarefas pode ocorrer em múltiplos níveis de representação.

Em PLN, o esquecimento catastrófico se manifesta de forma especialmente problemática devido à natureza contextual e ambígua da linguagem natural. Um modelo que aprendeu a classificar sentimento em avaliações de restaurantes pode perder completamente essa capacidade ao ser ajustado para responder perguntas sobre notícias, mesmo que ambas as tarefas compartilhem conhecimento linguístico fundamental. Esse comportamento contrasta com o aprendizado humano, onde novos conhecimentos são integrados de forma mais gradual e seletiva, preservando habilidades anteriores.

### Restrições práticas (privacidade, custo, políticas de uso)

Além dos desafios técnicos, o aprendizado contínuo em PLN enfrenta restrições práticas significativas que limitam as estratégias viáveis. Muitas aplicações não podem armazenar dados brutos de tarefas anteriores devido a políticas de privacidade (como GDPR), restrições de custo de armazenamento, ou limitações de uso de dados. Por exemplo, um sistema de análise de sentimento em mídias sociais pode precisar descartar dados antigos após processamento, impossibilitando estratégias de replay baseadas em buffer de exemplos reais.

O custo computacional também é uma preocupação central. Métodos que requerem manter múltiplas cópias do modelo ou processar grandes volumes de dados históricos podem se tornar proibitivos em contextos com recursos limitados, como dispositivos móveis ou ambientes de borda. Além disso, muitos cenários reais exigem que o modelo seja eficiente tanto no treinamento quanto na inferência, limitando a viabilidade de abordagens que introduzem overhead significativo.

### Necessidade de equilibrar plasticidade e estabilidade

O desafio central do aprendizado contínuo pode ser entendido como o dilema estabilidade-plasticidade (Grossberg, 1987): um modelo deve ser suficientemente plástico para incorporar novos conhecimentos, mas suficientemente estável para preservar conhecimentos anteriores. Maximizar apenas a plasticidade leva ao esquecimento catastrófico, enquanto maximizar apenas a estabilidade impede o aprendizado de novas tarefas.

A solução ideal requer um equilíbrio dinâmico entre esses dois objetivos, permitindo que o modelo adapte-se seletivamente a novas distribuições enquanto protege parâmetros críticos para tarefas anteriores. Esse equilíbrio é particularmente desafiador em PLN, onde diferentes tarefas podem compartilhar conhecimento linguístico fundamental (beneficiando-se de transferência positiva) ou podem conflitar em suas representações (requerendo isolamento). Abordagens complementares têm sido propostas: arquiteturas que isolam atualizações (Progressive Neural Networks; Rusu et al., 2016), regularização baseada em informação (Elastic Weight Consolidation; Kirkpatrick et al., 2017) e mecanismos de replay (incluindo replay gerativo; Shin et al., 2017), além de métodos de adaptação parametricamente eficientes como adapters e LoRA (Houlsby et al., 2019; Hu et al., 2021).

## Problema de pesquisa

### Como reduzir o esquecimento catastrófico em sequências de tarefas de PLN sem armazenar dados brutos, mantendo eficiência paramétrica e custo de treino moderado?

Esta questão central orienta o desenvolvimento deste trabalho. O problema reconhece três restrições fundamentais: (i) a impossibilidade de armazenar dados brutos de tarefas anteriores, excluindo estratégias de replay baseadas em buffer real; (ii) a necessidade de eficiência paramétrica, limitando o crescimento do modelo com o número de tarefas; e (iii) a viabilidade computacional, exigindo que o custo de treinamento permaneça moderado mesmo com sequências longas de tarefas.

A combinação dessas restrições torna o problema particularmente desafiador. Métodos arquiteturais puros (como PNNs) oferecem isolamento completo, mas ao custo de crescimento linear de parâmetros. Métodos de regularização (como EWC) não requerem dados antigos, mas podem prejudicar a plasticidade em sequências longas. Métodos de replay gerativo evitam armazenamento de dados brutos, mas introduzem custo computacional adicional. A hipótese central deste trabalho é que uma combinação integrada dessas abordagens pode oferecer sinergias que mitigam suas limitações individuais, alcançando um equilíbrio superior entre estabilidade, plasticidade e eficiência.

## Objetivos gerais e específicos

### Objetivo geral

Investigar a viabilidade e efetividade de uma arquitetura híbrida para aprendizado contínuo em PLN que combine PNN, O-LoRA, EWC e replay gerativo, avaliando seu impacto sobre o equilíbrio entre plasticidade e estabilidade, a eficiência em parâmetros e o custo computacional.

Este objetivo geral busca consolidar evidências empíricas sobre a integração de quatro mecanismos complementares de defesa contra esquecimento, cada um atuando em diferentes dimensões do problema. A arquitetura proposta visa combinar os benefícios de cada abordagem enquanto mitiga suas limitações individuais através de integração cuidadosa.

### Objetivo específico 1

Projetar e implementar um protótipo com modelos base moderados e adaptadores LoRA específicos por tarefa, impondo ortogonalidade entre subespaços de atualização para minimizar interferência entre tarefas.

A implementação prática requer decisões de engenharia sobre quais camadas adaptar, como configurar os ranks dos adaptadores LoRA, e como impor eficientemente restrições ortogonais durante o treinamento. O objetivo é demonstrar viabilidade técnica com recursos computacionais moderados, utilizando modelos como BERT-base ou DistilBERT.

### Objetivo específico 2

Aplicar EWC a componentes compartilhados parcialmente destravados para preservar conhecimento crítico mantido em embeddings e outras camadas fundamentais do modelo base.

O EWC será aplicado seletivamente apenas aos componentes que permanecem treináveis durante o aprendizado contínuo, evitando overhead desnecessário em camadas completamente congeladas. A estimação da matriz de Fisher e a calibração do hiperparâmetro λ são aspectos críticos desta implementação.

### Objetivo específico 3

Incorporar replay gerativo parcimonioso para reforço de tarefas passadas sem retenção de dados originais, utilizando geração sintética guiada por prompts estruturados.

O replay gerativo será implementado de forma parcimoniosa para minimizar custo computacional, intercalando pequenos batches de exemplos sintéticos durante o treinamento. A qualidade e representatividade das gerações são fatores críticos para o sucesso desta estratégia.

### Objetivo específico 4

Definir um protocolo experimental reprodutível com sequência de tarefas heterogêneas de PLN, reavaliação cumulativa após cada etapa, e métricas padronizadas para avaliar esquecimento, transferência e eficiência.

O protocolo experimental deve permitir comparação justa com trabalhos relacionados e facilitar reprodução por outros pesquisadores. A sequência de tarefas (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers) foi escolhida para representar diferentes domínios textuais e desafiar o modelo com distribuições variadas.

### Objetivo específico 5

Realizar ablações sistemáticas que isolem a contribuição individual de cada componente (O-LoRA, EWC, replay gerativo, conexões laterais) para entender sinergias e identificar pontos de otimização.

As ablações permitem quantificar o valor marginal de cada componente e identificar redundâncias ou complementaridades. Essa análise é essencial para validar que a integração oferece ganhos superiores à soma das partes individuais.

## Justificativa

### Relevância prática do aprendizado contínuo em PLN

O aprendizado contínuo é essencial para viabilizar sistemas de PLN adaptativos em produção. Aplicações como assistentes virtuais, sistemas de recomendação baseados em linguagem natural, e ferramentas de análise de texto precisam incorporar continuamente novos conhecimentos sem perder funcionalidades anteriores. A incapacidade de fazer isso efetivamente limita a escalabilidade e longevidade desses sistemas, tornando necessário re-treinar modelos periodicamente do zero, o que é custoso e pode não ser viável em todos os contextos.

Além disso, a natureza evolutiva da linguagem natural — com novas gírias, tópicos emergentes e mudanças de estilo — torna o aprendizado contínuo particularmente relevante para sistemas que operam em contextos dinâmicos como mídias sociais, notícias ou conteúdo gerado por usuários.

### Lacuna na integração sistemática das técnicas propostas

Embora existam evidências empíricas sobre a eficácia de técnicas individuais (PNN, LoRA, EWC, replay gerativo) e sobre combinações de pares (LoRA+replay, EWC+replay), não há na literatura uma avaliação sistemática e integrada de um arranjo que combine todas essas quatro estratégias sob um mesmo protocolo experimental. Essa lacuna impede que pesquisadores e engenheiros tomem decisões informadas sobre quais combinações de técnicas são mais eficazes para seus contextos específicos.

A falta de protocolos padronizados também dificulta comparações justas entre diferentes abordagens e limita a reprodutibilidade dos resultados. Este trabalho busca preencher essa lacuna fornecendo uma avaliação abrangente e reprodutível de uma arquitetura híbrida integrada.

### Viabilidade para contexto acadêmico com recursos moderados

Um aspecto importante da proposta é sua viabilidade para contexto acadêmico com recursos computacionais moderados. Ao utilizar modelos base de porte médio (BERT-base/DistilBERT), adaptadores LoRA de baixo ranque, e replay gerativo parcimonioso, os experimentos podem ser executados em uma única GPU intermediária (como NVIDIA T4), tornando a pesquisa acessível para grupos sem acesso a infraestrutura de supercomputação.

Essa viabilidade não compromete a relevância científica do trabalho, pois os princípios e mecanismos investigados são transferíveis para modelos maiores e mais complexos. Pelo contrário, demonstra que soluções eficientes podem ser desenvolvidas mesmo com recursos limitados, aumentando a democratização do acesso à pesquisa em aprendizado contínuo.

## Principais contribuições

### Contribuição conceitual

Este trabalho propõe um arranjo integrado e original que articula quatro mecanismos complementares de defesa contra esquecimento: modularização progressiva (PNN), parametrização eficiente com isolamento ortogonal (O-LoRA), consolidação de pesos críticos (EWC), e reforço sintético de memórias (replay gerativo). Embora cada técnica individual seja conhecida, a integração sistemática sob uma arquitetura unificada representa uma contribuição conceitual nova, explorando sinergias entre abordagens que atuam em diferentes dimensões do problema.

A proposta teórica explora como diferentes mecanismos podem se complementar: onde PNN/O-LoRA fornecem isolamento estrutural, EWC protege componentes compartilhados, e replay gerativo reforça conhecimentos através de dados sintéticos. Essa integração multi-camada oferece redundância defensiva, onde falhas em um mecanismo podem ser compensadas por outros.

### Contribuição metodológica

O trabalho estabelece um protocolo reprodutível de avaliação em sequência de tarefas, com métricas padronizadas para esquecimento (Forgetting, BWT), transferência (FWT), eficiência (crescimento paramétrico, custo computacional), e procedimentos sistemáticos de ablação. Este protocolo facilita comparações futuras com novas variantes e pode servir como benchmark para a comunidade de aprendizado contínuo em PLN.

Além disso, o trabalho demonstra como diferentes métricas podem ser combinadas para fornecer uma visão holística do desempenho de sistemas de aprendizado contínuo, indo além de métricas agregadas simples para análises mais granulares de comportamento ao longo da sequência de tarefas.

### Contribuição de engenharia

O trabalho fornece um guia prático para implementação com recursos computacionais moderados, incluindo decisões específicas sobre quais camadas adaptar, como configurar ranks LoRA, como calibrar hiperparâmetros de EWC, e como dosar a taxa de replay gerativo. Essas diretrizes são derivadas de experimentação empírica e buscas de validação, oferecendo insights práticos para engenheiros que precisam implementar sistemas similares.

O guia também discute trade-offs explícitos entre diferentes escolhas de projeto e fornece heurísticas para seleção de hiperparâmetros baseadas em características das tarefas e recursos disponíveis.

### Contribuição de artefatos

O trabalho disponibiliza código-fonte, configurações e scripts reusáveis que permitem reprodução e extensão por outros pesquisadores. Os artefatos incluem implementações dos componentes principais (adaptadores O-LoRA, integração EWC, replay gerativo), pipelines de experimentação, e ferramentas de análise de resultados. A disponibilização desses artefatos aumenta o impacto do trabalho e facilita avanços futuros na área.

## Estrutura do trabalho

O restante deste trabalho está organizado da seguinte forma. O Capítulo 2 apresenta a fundamentação teórica sobre aprendizado contínuo em PLN, detalhando cada uma das técnicas base utilizadas (PNN, LoRA/O-LoRA, EWC, replay gerativo) e as métricas de avaliação padronizadas. O capítulo também revisa trabalhos correlatos que combinam pares de técnicas ou propõem abordagens alternativas, identificando lacunas na literatura.

O Capítulo 3 descreve detalhadamente a metodologia proposta, incluindo a arquitetura híbrida integrada, o protocolo experimental com sequência de tarefas, o fluxo de treinamento completo, as configurações do ambiente computacional, e os procedimentos de avaliação e ablação. O capítulo fornece informações suficientes para reprodução dos experimentos.

O Capítulo 4 apresenta e discute os resultados experimentais, incluindo análise do desempenho principal, eficiência computacional e paramétrica, resultados de ablação detalhados, e comparação com baselines representativos. O capítulo também identifica limitações e explora interpretações dos resultados observados.

Finalmente, o Capítulo 5 sintetiza os principais achados, discute as contribuições alcançadas, reconhece limitações do estudo, e propõe direções para trabalhos futuros que possam estender e melhorar a abordagem proposta.
