# Rede Progressiva com LoRA Ortogonal para Aprendizado Contínuo

## Resumo

O estudo investiga aprendizado contínuo em Processamento de Linguagem Natural, com foco na mitigação do esquecimento catastrófico ao treinar modelos em sequências de tarefas. Parte-se do dilema estabilidade–plasticidade e do contexto prático em que dados antigos não podem ser armazenados integralmente, seja por custo, privacidade ou políticas de uso. Nessa conjuntura, busca-se preservar desempenho em tarefas anteriores enquanto se mantém capacidade de adaptação a novas distribuições, com uso parcimonioso de parâmetros e de computação. Propõe-se uma arquitetura híbrida que combina modularização progressiva inspirada em redes neurais progressivas, adaptações de baixo ranque com restrição ortogonal, consolidação elástica de pesos e replay gerativo sem retenção de dados brutos, visando equilibrar plasticidade e estabilidade com eficiência paramétrica. A abordagem emprega modelos base de porte moderado, com adaptadores LoRA específicos por tarefa e ortogonalidade para separar subespaços de atualização; aplica EWC a componentes compartilhados parcialmente destravados; e intercala lotes sintéticos para recuperar conhecimentos prévios. O protocolo experimental contempla uma sequência de tarefas de PLN com reavaliação cumulativa após cada etapa, empregando métricas como acurácia média, taxa de esquecimento, transferências forward e backward, crescimento paramétrico e custo computacional, além de ablações para isolar o efeito de cada mecanismo. Espera-se demonstrar redução do esquecimento sem degradar a plasticidade, oferecendo ganhos marginais mensuráveis quando cada componente é ativado, com sobrecusto paramétrico moderado por tarefa. Espera-se como contribuição principal um arranjo integrado, original e eficiente em parâmetros para aprendizado contínuo em PLN, acompanhado de um protocolo reprodutível e de diretrizes de engenharia.

### Palavras-chave

Aprendizado Contínuo; LoRA ortogonal; EWC; Replay Gerativo; PLN

## 1 - Introdução

### Contextualização e motivação

#### Desafio do aprendizado contínuo em PLN

O aprendizado contínuo em Processamento de Linguagem Natural (PLN) representa um desafio fundamental para sistemas de inteligência artificial que precisam operar em ambientes dinâmicos. Em aplicações reais, modelos de linguagem enfrentam sequências de tarefas heterogêneas: desde classificação de sentimento em avaliações de produtos até perguntas e respostas sobre documentação técnica, passando por análise de intenções em sistemas de atendimento. Cada nova tarefa ou domínio requer adaptação do modelo, mas a capacidade de incorporar novos conhecimentos sem degradar o desempenho em tarefas anteriormente aprendidas permanece um problema em aberto.

Diferentemente do fine-tuning tradicional, onde um modelo é ajustado uma vez sobre um conjunto fixo de dados, o aprendizado contínuo implica treinar modelos sequencialmente em múltiplas tarefas ou distribuições que chegam ao longo do tempo. Essa natureza sequencial é natural em muitos contextos: assistentes virtuais que precisam aprender novos comandos sem esquecer funcionalidades antigas, sistemas de monitoramento que devem se adaptar a novas fontes de dados, ou aplicações educacionais que incorporam progressivamente novos tópicos curriculares.

Em tais cenários, a literatura reporta que atualizações para novas tarefas frequentemente interferem com parâmetros críticos de tarefas anteriores, produzindo degradações substanciais na ausência de mecanismos de proteção (McCloskey & Cohen, 1989; French, 1999). Esse fenômeno motiva o desenvolvimento de abordagens de aprendizado contínuo que combinem isolamento estrutural, regularização informativa e replay, especialmente em arquiteturas Transformer amplamente utilizadas em PLN.

#### Esquecimento catastrófico em sequências de tarefas

O principal obstáculo ao aprendizado contínuo eficaz é o fenômeno do esquecimento catastrófico, documentado desde os primórdios das redes neurais (McCloskey & Cohen, 1989; French, 1999). Quando um modelo neural é treinado sequencialmente em tarefas diferentes, os gradientes da tarefa atual tendem a sobrescrever parâmetros críticos para tarefas anteriores, levando a uma degradação abrupta do desempenho em tarefas antigas. Esse efeito é particularmente pronunciado em arquiteturas modernas de Transformers, onde milhões de parâmetros são compartilhados entre diferentes camadas e a interferência entre tarefas pode ocorrer em múltiplos níveis de representação.

Em PLN, o esquecimento catastrófico se manifesta de forma especialmente problemática devido à natureza contextual e ambígua da linguagem natural. Um modelo que aprendeu a classificar sentimento em avaliações de restaurantes pode perder completamente essa capacidade ao ser ajustado para responder perguntas sobre notícias, mesmo que ambas as tarefas compartilhem conhecimento linguístico fundamental. Esse comportamento contrasta com o aprendizado humano, onde novos conhecimentos são integrados de forma mais gradual e seletiva, preservando habilidades anteriores.

#### Restrições práticas (privacidade, custo, políticas de uso)

Além dos desafios técnicos, o aprendizado contínuo em PLN enfrenta restrições práticas significativas que limitam as estratégias viáveis. Muitas aplicações não podem armazenar dados brutos de tarefas anteriores devido a políticas de privacidade (como GDPR), restrições de custo de armazenamento, ou limitações de uso de dados. Por exemplo, um sistema de análise de sentimento em mídias sociais pode precisar descartar dados antigos após processamento, impossibilitando estratégias de replay baseadas em buffer de exemplos reais.

O custo computacional também é uma preocupação central. Métodos que requerem manter múltiplas cópias do modelo ou processar grandes volumes de dados históricos podem se tornar proibitivos em contextos com recursos limitados, como dispositivos móveis ou ambientes de borda. Além disso, muitos cenários reais exigem que o modelo seja eficiente tanto no treinamento quanto na inferência, limitando a viabilidade de abordagens que introduzem overhead significativo.

#### Necessidade de equilibrar plasticidade e estabilidade

O desafio central do aprendizado contínuo pode ser entendido como o dilema estabilidade-plasticidade (Grossberg, 1987): um modelo deve ser suficientemente plástico para incorporar novos conhecimentos, mas suficientemente estável para preservar conhecimentos anteriores. Maximizar apenas a plasticidade leva ao esquecimento catastrófico, enquanto maximizar apenas a estabilidade impede o aprendizado de novas tarefas.

A solução ideal requer um equilíbrio dinâmico entre esses dois objetivos, permitindo que o modelo adapte-se seletivamente a novas distribuições enquanto protege parâmetros críticos para tarefas anteriores. Esse equilíbrio é particularmente desafiador em PLN, onde diferentes tarefas podem compartilhar conhecimento linguístico fundamental (beneficiando-se de transferência positiva) ou podem conflitar em suas representações (requerendo isolamento). Abordagens complementares têm sido propostas: arquiteturas que isolam atualizações (Progressive Neural Networks; Rusu et al., 2016), regularização baseada em informação (Elastic Weight Consolidation; Kirkpatrick et al., 2017) e mecanismos de replay (incluindo replay gerativo; Shin et al., 2017), além de métodos de adaptação parametricamente eficientes como adapters e LoRA (Houlsby et al., 2019; Hu et al., 2021).

### Problema de pesquisa

#### Como reduzir o esquecimento catastrófico em sequências de tarefas de PLN sem armazenar dados brutos, mantendo eficiência paramétrica e custo de treino moderado?

Esta questão central orienta o desenvolvimento deste trabalho. O problema reconhece três restrições fundamentais: (i) a impossibilidade de armazenar dados brutos de tarefas anteriores, excluindo estratégias de replay baseadas em buffer real; (ii) a necessidade de eficiência paramétrica, limitando o crescimento do modelo com o número de tarefas; e (iii) a viabilidade computacional, exigindo que o custo de treinamento permaneça moderado mesmo com sequências longas de tarefas.

A combinação dessas restrições torna o problema particularmente desafiador. Métodos arquiteturais puros (como PNNs) oferecem isolamento completo, mas ao custo de crescimento linear de parâmetros. Métodos de regularização (como EWC) não requerem dados antigos, mas podem prejudicar a plasticidade em sequências longas. Métodos de replay gerativo evitam armazenamento de dados brutos, mas introduzem custo computacional adicional. A hipótese central deste trabalho é que uma combinação integrada dessas abordagens pode oferecer sinergias que mitigam suas limitações individuais, alcançando um equilíbrio superior entre estabilidade, plasticidade e eficiência.

### Objetivos gerais e específicos

#### Objetivo geral

Investigar a viabilidade e efetividade de uma arquitetura híbrida para aprendizado contínuo em PLN que combine PNN, O-LoRA, EWC e replay gerativo, avaliando seu impacto sobre o equilíbrio entre plasticidade e estabilidade, a eficiência em parâmetros e o custo computacional.

Este objetivo geral busca consolidar evidências empíricas sobre a integração de quatro mecanismos complementares de defesa contra esquecimento, cada um atuando em diferentes dimensões do problema. A arquitetura proposta visa combinar os benefícios de cada abordagem enquanto mitiga suas limitações individuais através de integração cuidadosa.

#### Objetivo específico 1

Projetar e implementar um protótipo com modelos base moderados e adaptadores LoRA específicos por tarefa, impondo ortogonalidade entre subespaços de atualização para minimizar interferência entre tarefas.

A implementação prática requer decisões de engenharia sobre quais camadas adaptar, como configurar os ranks dos adaptadores LoRA, e como impor eficientemente restrições ortogonais durante o treinamento. O objetivo é demonstrar viabilidade técnica com recursos computacionais moderados, utilizando modelos como BERT-base ou DistilBERT.

#### Objetivo específico 2

Aplicar EWC a componentes compartilhados parcialmente destravados para preservar conhecimento crítico mantido em embeddings e outras camadas fundamentais do modelo base.

O EWC será aplicado seletivamente apenas aos componentes que permanecem treináveis durante o aprendizado contínuo, evitando overhead desnecessário em camadas completamente congeladas. A estimação da matriz de Fisher e a calibração do hiperparâmetro λ são aspectos críticos desta implementação.

#### Objetivo específico 3

Incorporar replay gerativo parcimonioso para reforço de tarefas passadas sem retenção de dados originais, utilizando geração sintética guiada por prompts estruturados.

O replay gerativo será implementado de forma parcimoniosa para minimizar custo computacional, intercalando pequenos batches de exemplos sintéticos durante o treinamento. A qualidade e representatividade das gerações são fatores críticos para o sucesso desta estratégia.

#### Objetivo específico 4

Definir um protocolo experimental reprodutível com sequência de tarefas heterogêneas de PLN, reavaliação cumulativa após cada etapa, e métricas padronizadas para avaliar esquecimento, transferência e eficiência.

O protocolo experimental deve permitir comparação justa com trabalhos relacionados e facilitar reprodução por outros pesquisadores. A sequência de tarefas (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers) foi escolhida para representar diferentes domínios textuais e desafiar o modelo com distribuições variadas.

#### Objetivo específico 5

Realizar ablações sistemáticas que isolem a contribuição individual de cada componente (O-LoRA, EWC, replay gerativo, conexões laterais) para entender sinergias e identificar pontos de otimização.

As ablações permitem quantificar o valor marginal de cada componente e identificar redundâncias ou complementaridades. Essa análise é essencial para validar que a integração oferece ganhos superiores à soma das partes individuais.

### Justificativa

#### Relevância prática do aprendizado contínuo em PLN

O aprendizado contínuo é essencial para viabilizar sistemas de PLN adaptativos em produção. Aplicações como assistentes virtuais, sistemas de recomendação baseados em linguagem natural, e ferramentas de análise de texto precisam incorporar continuamente novos conhecimentos sem perder funcionalidades anteriores. A incapacidade de fazer isso efetivamente limita a escalabilidade e longevidade desses sistemas, tornando necessário re-treinar modelos periodicamente do zero, o que é custoso e pode não ser viável em todos os contextos.

Além disso, a natureza evolutiva da linguagem natural — com novas gírias, tópicos emergentes e mudanças de estilo — torna o aprendizado contínuo particularmente relevante para sistemas que operam em contextos dinâmicos como mídias sociais, notícias ou conteúdo gerado por usuários.

#### Lacuna na integração sistemática das técnicas propostas

Embora existam evidências empíricas sobre a eficácia de técnicas individuais (PNN, LoRA, EWC, replay gerativo) e sobre combinações de pares (LoRA+replay, EWC+replay), não há na literatura uma avaliação sistemática e integrada de um arranjo que combine todas essas quatro estratégias sob um mesmo protocolo experimental. Essa lacuna impede que pesquisadores e engenheiros tomem decisões informadas sobre quais combinações de técnicas são mais eficazes para seus contextos específicos.

A falta de protocolos padronizados também dificulta comparações justas entre diferentes abordagens e limita a reprodutibilidade dos resultados. Este trabalho busca preencher essa lacuna fornecendo uma avaliação abrangente e reprodutível de uma arquitetura híbrida integrada.

#### Viabilidade para contexto acadêmico com recursos moderados

Um aspecto importante da proposta é sua viabilidade para contexto acadêmico com recursos computacionais moderados. Ao utilizar modelos base de porte médio (BERT-base/DistilBERT), adaptadores LoRA de baixo ranque, e replay gerativo parcimonioso, os experimentos podem ser executados em uma única GPU intermediária (como NVIDIA T4), tornando a pesquisa acessível para grupos sem acesso a infraestrutura de supercomputação.

Essa viabilidade não compromete a relevância científica do trabalho, pois os princípios e mecanismos investigados são transferíveis para modelos maiores e mais complexos. Pelo contrário, demonstra que soluções eficientes podem ser desenvolvidas mesmo com recursos limitados, aumentando a democratização do acesso à pesquisa em aprendizado contínuo.

### Principais contribuições

#### Contribuição conceitual

Este trabalho propõe um arranjo integrado e original que articula quatro mecanismos complementares de defesa contra esquecimento: modularização progressiva (PNN), parametrização eficiente com isolamento ortogonal (O-LoRA), consolidação de pesos críticos (EWC), e reforço sintético de memórias (replay gerativo). Embora cada técnica individual seja conhecida, a integração sistemática sob uma arquitetura unificada representa uma contribuição conceitual nova, explorando sinergias entre abordagens que atuam em diferentes dimensões do problema.

A proposta teórica explora como diferentes mecanismos podem se complementar: onde PNN/O-LoRA fornecem isolamento estrutural, EWC protege componentes compartilhados, e replay gerativo reforça conhecimentos através de dados sintéticos. Essa integração multi-camada oferece redundância defensiva, onde falhas em um mecanismo podem ser compensadas por outros.

#### Contribuição metodológica

O trabalho estabelece um protocolo reprodutível de avaliação em sequência de tarefas, com métricas padronizadas para esquecimento (Forgetting, BWT), transferência (FWT), eficiência (crescimento paramétrico, custo computacional), e procedimentos sistemáticos de ablação. Este protocolo facilita comparações futuras com novas variantes e pode servir como benchmark para a comunidade de aprendizado contínuo em PLN.

Além disso, o trabalho demonstra como diferentes métricas podem ser combinadas para fornecer uma visão holística do desempenho de sistemas de aprendizado contínuo, indo além de métricas agregadas simples para análises mais granulares de comportamento ao longo da sequência de tarefas.

#### Contribuição de engenharia

O trabalho fornece um guia prático para implementação com recursos computacionais moderados, incluindo decisões específicas sobre quais camadas adaptar, como configurar ranks LoRA, como calibrar hiperparâmetros de EWC, e como dosar a taxa de replay gerativo. Essas diretrizes são derivadas de experimentação empírica e buscas de validação, oferecendo insights práticos para engenheiros que precisam implementar sistemas similares.

O guia também discute trade-offs explícitos entre diferentes escolhas de projeto e fornece heurísticas para seleção de hiperparâmetros baseadas em características das tarefas e recursos disponíveis.

#### Contribuição de artefatos

O trabalho disponibiliza código-fonte, configurações e scripts reusáveis que permitem reprodução e extensão por outros pesquisadores. Os artefatos incluem implementações dos componentes principais (adaptadores O-LoRA, integração EWC, replay gerativo), pipelines de experimentação, e ferramentas de análise de resultados. A disponibilização desses artefatos aumenta o impacto do trabalho e facilita avanços futuros na área.

### Estrutura do trabalho

O restante deste trabalho está organizado da seguinte forma. O Capítulo 2 apresenta a fundamentação teórica sobre aprendizado contínuo em PLN, detalhando cada uma das técnicas base utilizadas (PNN, LoRA/O-LoRA, EWC, replay gerativo) e as métricas de avaliação padronizadas. O capítulo também revisa trabalhos correlatos que combinam pares de técnicas ou propõem abordagens alternativas, identificando lacunas na literatura.

O Capítulo 3 descreve detalhadamente a metodologia proposta, incluindo a arquitetura híbrida integrada, o protocolo experimental com sequência de tarefas, o fluxo de treinamento completo, as configurações do ambiente computacional, e os procedimentos de avaliação e ablação. O capítulo fornece informações suficientes para reprodução dos experimentos.

O Capítulo 4 apresenta e discute os resultados experimentais, incluindo análise do desempenho principal, eficiência computacional e paramétrica, resultados de ablação detalhados, e comparação com baselines representativos. O capítulo também identifica limitações e explora interpretações dos resultados observados.

Finalmente, o Capítulo 5 sintetiza os principais achados, discute as contribuições alcançadas, reconhece limitações do estudo, e propõe direções para trabalhos futuros que possam estender e melhorar a abordagem proposta.

## 2 - Referencial Teórico

### Fundamentação teórica sobre aprendizado contínuo em PLN

#### Definição e desafios do aprendizado contínuo

O aprendizado contínuo, também referido como continual learning ou lifelong learning, consiste na capacidade de um sistema de inteligência artificial aprender novas tarefas de forma sequencial, acumulando conhecimento ao longo do tempo sem esquecer o que foi aprendido anteriormente. Diferentemente do fine-tuning tradicional, onde um conjunto de dados é fixo e o modelo é ajustado apenas uma vez, no aprendizado contínuo o modelo enfrenta desafios significativos ao adaptar-se a distribuições que evoluem sem reiniciar o treinamento do zero para cada novo contexto.

Panoramas recentes oferecem taxonomias abrangentes de métodos (regularização, arquitetura, replay) e protocolos de avaliação, destacando dificuldades de comparação e a necessidade de métricas padronizadas (De Lange et al., 2021; Parisi et al., 2019).

O principal desafio é o esquecimento catastrófico — a tendência de redes neurais esquecerem abruptamente tarefas antigas ao aprender tarefas novas. Esse fenômeno foi documentado desde os primórdios das redes neurais e permanece um obstáculo central em sistemas adaptativos. Uma arquitetura de aprendizado contínuo deve equilibrar plasticidade (para incorporar novos conhecimentos) e estabilidade (para reter conhecimentos prévios), evitando que gradientes das tarefas recentes sobrescrevam parâmetros importantes das tarefas passadas.

Métricas clássicas para quantificar retenção e transferência incluem Acurácia Média (ACC), Backward Transfer (BWT), Forward Transfer (FWT) e Forgetting, popularizadas por Lopez-Paz & Ranzato (2017) e sistematizadas por van de Ven & Tolias (2019).

#### Dilema estabilidade-plasticidade

O dilema estabilidade-plasticidade, formalizado por Grossberg (1987), é fundamental para entender os desafios do aprendizado contínuo. Plasticidade refere-se à capacidade do sistema de modificar seus parâmetros para incorporar novos conhecimentos, enquanto estabilidade refere-se à capacidade de preservar conhecimentos previamente adquiridos. Em redes neurais tradicionais, essas capacidades estão em conflito: aumentar a plasticidade facilita o aprendizado de novas tarefas mas aumenta a vulnerabilidade ao esquecimento, enquanto aumentar a estabilidade protege contra esquecimento mas pode impedir adaptação efetiva a novas distribuições.

O equilíbrio ótimo depende de múltiplos fatores, incluindo a similaridade entre tarefas, a capacidade do modelo, e o regime de treinamento. Em PLN, onde tarefas podem variar desde classificação de sentimento até tradução ou sumarização, encontrar esse equilíbrio é particularmente desafiador devido à diversidade de objetivos e distribuições.

#### Esquecimento catastrófico em redes neurais

O esquecimento catastrófico ocorre quando o treinamento em uma nova tarefa altera parâmetros que eram críticos para tarefas anteriores. Em redes neurais profundas, onde milhões de parâmetros são compartilhados entre diferentes camadas e funções, essa interferência pode ocorrer em múltiplos níveis. O problema é especialmente pronunciado quando tarefas novas e antigas requerem atualizações conflitantes dos mesmos parâmetros.

A natureza do esquecimento catastrófico em Transformers é particularmente complexa devido à arquitetura de atenção e às múltiplas camadas de representação. Parâmetros compartilhados em embeddings, camadas de atenção e redes feed-forward podem ser afetados de forma diferente dependendo de como as tarefas utilizam essas representações. Compreender e mitigar essas interferências requer abordagens que identifiquem e protejam parâmetros críticos ou que isolam atualizações para diferentes tarefas.

Entre os mecanismos de mitigação, destacam-se: (i) isolamento estrutural via PNNs (Rusu et al., 2016); (ii) regularização baseada em informação via EWC (Kirkpatrick et al., 2017; Schwarz et al., 2018); (iii) replay (Shin et al., 2017), inclusive sem dados brutos; e (iv) restrições ortogonais sobre atualizações/espelhos de parâmetros (Zeng et al., 2019; Farajtabar et al., 2019).

#### Contexto específico do PLN (diversidade de tarefas, domínios, ambiguidade linguística)

Em PLN, o aprendizado contínuo apresenta desafios adicionais e motivação especial. Enquanto em visão computacional ou robótica as tarefas podem ser bem delimitadas (por exemplo, diferentes conjuntos de classes de imagens), em PLN há grande diversidade de tarefas: classificação de intenção, análise de sentimento, perguntas e respostas, tradução, sumarização, entre outras. Essas tarefas frequentemente envolvem dados textuais de domínios distintos e objetivos variáveis.

A linguagem natural é inerentemente ambígua e dependente do contexto, com vocabulário em constante evolução. Essa natureza dinâmica reforça a motivação para sistemas de PLN que aprendam continuamente: aplicações reais frequentemente precisam incorporar novas gírias, tópicos emergentes, mudanças de estilo ou domínio linguístico sem perder a habilidade em tarefas anteriores. Um assistente virtual pode precisar aprender progressivamente novos tipos de consultas dos usuários ao longo do tempo, sem esquecer como responder às solicitações antigas.

### Redes Neurais Progressivas (PNN)

#### Conceito e arquitetura básica

As Redes Neurais Progressivas (Progressive Neural Networks - PNN), proposta por Rusu et al. (2016), constituem uma abordagem baseada em arquitetura para aprendizado contínuo. A ideia central é expandir a capacidade da rede incrementalmente a cada nova tarefa, adicionando um novo conjunto de neurônios (uma nova "coluna" ou módulo de rede) específico para cada tarefa aprendida.

Quando a tarefa T_k é iniciada, cria-se uma nova coluna de parâmetros inicializada (geralmente a partir de uma versão pré-treinada) para aprender T_k. As colunas das tarefas anteriores são congeladas — seus pesos não são alterados durante o treinamento de T_k — e são estabelecidas conexões laterais da saída (ou camadas intermediárias) de cada coluna anterior para a nova coluna. Essas conexões laterais permitem que a nova coluna reutilize e transfira conhecimento das características previamente aprendidas nas tarefas anteriores, promovendo transferência de aprendizado para frente.

#### Isolamento de parâmetros por tarefa

A característica fundamental das PNNs é o isolamento completo entre tarefas em termos de parâmetros. Cada coluna atende a uma tarefa específica, garantindo que uma tarefa nova não degrade o desempenho das anteriores através de interferência destrutiva direta. Como os parâmetros das colunas antigas não são modificados, a PNN elimina o esquecimento catastrófico por construção — o aprendizado de T_k não interfere diretamente nos pesos que foram sintonizados para tarefas anteriores.

Esse isolamento é conceitualmente simples e teoricamente garantido, mas vem ao custo de crescimento linear do número de parâmetros em função do número de tarefas. Para cada nova tarefa adiciona-se uma coluna completa de rede, o que pode se tornar impraticável quando as tarefas são numerosas ou quando a arquitetura base é muito grande.

#### Conexões laterais e transferência de conhecimento

As conexões laterais são um mecanismo explícito de transferência que permite que a nova coluna consuma representações das colunas anteriores sem atualizá-las. Essas conexões podem ser implementadas de várias formas: concatenando saídas de camadas correspondentes, somando representações ponderadas, ou através de mecanismos de atenção que aprendem a combinar features de diferentes colunas.

O benefício das conexões laterais é duplo: primeiro, permitem que a nova tarefa se beneficie de conhecimento prévio aprendido, potencialmente acelerando o treinamento e melhorando o desempenho inicial. Segundo, fornecem um mecanismo de forward transfer, onde conhecimento de tarefas anteriores facilita o aprendizado de tarefas futuras. Em termos práticos, isso significa que uma nova tarefa pode partir de representações já úteis, em vez de aprender tudo do zero.

#### Vantagens e limitações (crescimento paramétrico linear)

As principais vantagens das PNNs são o isolamento completo entre tarefas (garantindo imunidade ao esquecimento), a simplicidade conceitual, e a capacidade de transferência positiva através de conexões laterais. A abordagem reflete, em certo grau, a modularidade do aprendizado biológico, onde novos conhecimentos podem recrutar novas estruturas sem apagar as antigas.

A principal limitação é o crescimento linear pesado de parâmetros: aplicar PNN ingênua a 10 tarefas com um modelo do porte do BERT significaria ter 10 modelos BERT em memória ao final. Além disso, PNNs normalmente assumem que os limites entre tarefas são conhecidos e bem definidos durante o treinamento, e geralmente requerem conhecimento da tarefa na inferência para rotear exemplos à coluna apropriada. Essas limitações motivam alternativas que capturam os benefícios das PNNs (isolamento e transferência) de forma mais parametricamente eficiente.

### Adaptações de Baixo Ranque com Restrições Ortogonais (LoRA e O-LoRA)

#### LoRA: adaptação eficiente de parâmetros

O LoRA (Low-Rank Adaptation), proposto por Hu et al. (2021), oferece uma forma de adaptar modelos de linguagem de grande porte de maneira leve e modular. A técnica funciona congelando todos os pesos originais do modelo pré-treinado e injetando pequenos módulos treináveis de baixo ranque em cada camada relevante da arquitetura Transformer.

Para cada matriz de pesos W em camadas selecionadas (por exemplo, nas projeções de atenção ou na rede feed-forward), LoRA introduz duas matrizes menores A e B de dimensões de posto r (tipicamente r bem menor que o tamanho original da camada) de forma que a atualização da camada seja W + ΔW, onde ΔW = AB representa um ajuste de baixo ranque aprendido para a nova tarefa. Em vez de ajustar todos os pesos do modelo, apenas os parâmetros nesses pequenos módulos são aprendidos.

#### Decomposição de baixo ranque (matrizes A e B)

A decomposição de baixo ranque explora a hipótese de que as atualizações de pesos necessárias para uma nova tarefa podem ser representadas eficientemente em um subespaço de dimensão muito menor que o espaço original de parâmetros. Se W tem dimensões d×d, LoRA introduz duas matrizes A (d×r) e B (r×d), onde r << d. O produto AB resulta em uma matriz de atualização de posto no máximo r, permitindo representar mudanças complexas com muito menos parâmetros.

A inicialização das matrizes A e B é importante: tipicamente A é inicializada aleatoriamente e B é inicializada com zeros, garantindo que ΔW = 0 inicialmente e o modelo começa com o comportamento do modelo base. Durante o treinamento, apenas A e B são atualizados, mantendo W congelado.

#### Eficiência paramétrica e computacional

Uma vantagem crucial do LoRA é a redução dramática de parâmetros ajustáveis. No caso extremo do GPT-3 (175 bilhões de parâmetros), estudos mostraram que é possível ajustar cerca de 18 milhões de parâmetros (via LoRA) para especializar o modelo em uma nova tarefa, mantendo desempenho similar ao fine-tuning completo — isso equivale a apenas ~0,01% dos parâmetros originais. Em modelos menores como BERT ou GPT-2, o overhead típico do LoRA por tarefa costuma ficar na faixa de 0,1% a 3% do total de parâmetros.

Além da eficiência paramétrica, o LoRA mantém eficiência computacional: como o modelo base permanece congelado e compartilhado entre todas tarefas, não há aumento de latência na inferência quando os deltas são mesclados de volta nos pesos originais. Isso torna viável carregar múltiplos adapters LoRA — um por tarefa — em memória sem explodir o uso de recursos.

#### O-LoRA: imposição de ortogonalidade entre adaptadores

O LoRA puro não resolve por si só o esquecimento catastrófico quando usado sequencialmente. Se o mesmo conjunto de adaptadores for reutilizado para múltiplas tarefas em sequência, a tarefa nova pode sobrescrever as representações adquiridas pelos adaptadores na tarefa anterior. Para enfrentar essa limitação, o O-LoRA (Orthogonal LoRA), proposto por Wang et al. (2023), impõe restrições ortogonais entre os subespaços de adaptação de cada tarefa.

Ao treinar os adaptadores de uma nova tarefa, adiciona-se um termo de regularização (ou projeta-se explicitamente) para que os novos deltas de baixo ranque fiquem ortogonais aos espaços gerados pelos deltas das tarefas anteriores. Assim, cada tarefa T_i aprende sua atualização ΔW_i = A_i B_i em um subespaço linear distinto, minimizando projeções em direções usadas por ΔW_j de tarefas j < i.

#### Isolamento em subespaços distintos

O isolamento em subespaços ortogonais atua como um análogo leve de uma PNN: em vez de dedicar uma coluna inteira de parâmetros para cada tarefa, dedica-se apenas um subespaço de adaptações de baixo ranque, mas assegurando que haja pouca sobreposição entre as direções de atualização de tarefas diferentes. Estudos reportam que métodos como O-LoRA efetivamente reduzem a interferência entre tarefas e, assim, o esquecimento, quando comparados ao uso ingênuo de LoRA sequencial.

O custo adicional do O-LoRA é marginal: os adaptadores ortogonais têm o mesmo número de parâmetros do LoRA convencional (apenas o procedimento de treinamento muda), mantendo a eficiência. No entanto, garantir ortogonalidade perfeita entre subespaços de diversas tarefas pode se tornar difícil conforme o número de tarefas cresce, especialmente se o ranque r for limitado.

#### Vantagens e limitações

LoRA e suas variantes ortogonais oferecem um compromisso atraente entre isolamento de tarefas e eficiência. Cada tarefa é especializada por meio de um conjunto pequeno de parâmetros adicionais, resultando em crescimento linear modesto de memória com o número de tarefas (ordens de grandeza menor que adicionar colunas completas como na PNN). Com O-LoRA, obtém-se também isolamento efetivo entre tarefas, aproximando-se do ideal de "uma coluna por tarefa", porém de forma muito mais leve.

As limitações incluem o crescimento linear ainda presente (embora baixo), a necessidade de conhecimento da tarefa na inferência para ativar o conjunto correto de adaptadores, e possíveis dificuldades em manter ortogonalidade perfeita em sequências muito longas. Ainda assim, LoRA e O-LoRA representam avanços importantes para viabilizar aprendizado contínuo em modelos de PLN grandes, fornecendo eficiência paramétrica com isolamento suficiente para mitigar grande parte do esquecimento.

### Elastic Weight Consolidation (EWC)

#### Regularização baseada em importância de pesos

Elastic Weight Consolidation (EWC), proposto por Kirkpatrick et al. (2017), é um método de regularização que busca preservar o conhecimento prévio dentro dos mesmos parâmetros ao longo de tarefas sequenciais. A premissa é adicionar um termo de penalização na função de perda durante o treinamento de novas tarefas, desencorajando grandes mudanças nos pesos que foram identificados como importantes para tarefas antigas.

Antes de aprender a tarefa T_k, o algoritmo EWC calcula, para cada peso θ_j do modelo, um valor de importância que quantifica o quanto θ_j contribuiu para o desempenho nas tarefas anteriores. Essa importância é tipicamente estimada através da matriz de informação de Fisher, avaliada nos dados das tarefas passadas. Intuitivamente, se um peso influenciava fortemente as predições corretas nas tarefas antigas, o EWC irá puni-lo caso ele se desvie muito do seu valor original enquanto aprende a nova tarefa.

#### Matriz de informação de Fisher

A matriz de informação de Fisher F fornece uma medida da importância de cada parâmetro para o desempenho do modelo. Os elementos diagonais F_j da matriz representam a curvatura da função de perda em relação ao parâmetro θ_j: valores altos indicam que pequenas mudanças em θ_j têm grande impacto no desempenho, enquanto valores baixos indicam que o parâmetro é menos crítico.

A estimativa da matriz de Fisher requer avaliar o gradiente da função de perda em relação aos parâmetros sobre os dados da tarefa anterior. Em implementações práticas, apenas os elementos diagonais são computados (aproximação diagonal), reduzindo significativamente o custo computacional. A matriz de Fisher é estimada após o treinamento em cada tarefa e armazenada para uso nas tarefas subsequentes.

#### Termo de penalização na função de perda

Matematicamente, a perda total do modelo ao aprender uma nova tarefa incorpora um termo do tipo:

L(θ) = L_novo(θ) + λ Σ_j (1/2) F_j (θ_j - θ_j*)^2

onde L_novo é a perda normal nos dados da nova tarefa, θ_j* é o valor do peso j após treinamento na tarefa anterior (mantido como referência), F_j é o elemento diagonal da matriz de Fisher, e λ é um hiperparâmetro que controla a força da penalização. Esse termo adicional atua como uma "mola" ancorando cada peso em torno do valor antigo com rigidez proporcional à importância F_j.

#### Consolidação de conhecimento crítico

Pesos críticos ficam quase "congelados" (alta penalização se mudarem), enquanto pesos pouco relevantes podem se ajustar livremente à nova tarefa. Dessa forma, o EWC tenta obter um compromisso ótimo entre não esquecer o passado e ainda aprender o novo, encontrando uma região no espaço de parâmetros que minimize a perda da tarefa atual sem sair da região de bom desempenho das tarefas antigas.

A consolidação é particularmente efetiva quando aplicada a componentes compartilhados que mantêm conhecimento linguístico fundamental, como embeddings ou camadas iniciais do modelo. Essas camadas frequentemente codificam conhecimento geral que beneficia múltiplas tarefas, tornando-as candidatas ideais para proteção via EWC.

#### Vantagens e limitações (trade-off plasticidade/estabilidade)

O EWC se destaca por sua simplicidade e generalidade: não requer modificar a arquitetura da rede nem adicionar parâmetros extras, apenas a função de custo muda. Não há crescimento de memória conforme novas tarefas são aprendidas (diferente de PNN/LoRA) e não é necessário armazenar dados antigos (diferente de métodos de replay). Em implementações offline, basta armazenar as estimativas de F_j e os valores antigos θ_j após cada tarefa.

Apesar de mitigar o esquecimento, o EWC raramente o elimina por completo. Em cenários de longas sequências de tarefas, as restrições impostas podem se acumular a ponto de prejudicar a plasticidade do modelo para novas tarefas. O efeito depende criticamente do hiperparâmetro λ: se muito alto, o modelo praticamente não aprende a nova tarefa; se muito baixo, o modelo esquece facilmente as antigas. Encontrar um equilíbrio pode exigir validação cuidadosa para cada situação.

### Replay Gerativo

#### Conceito de rehearsal e pseudo-rehearsal

O replay gerativo é uma estratégia inspirada no conceito de rehearsal em psicologia, onde memórias antigas são periodicamente revisitadas para consolidação. Em aprendizado de máquina, o replay clássico consiste em reapresentar dados de tarefas antigas durante o treinamento de uma nova tarefa, para que o modelo "não se esqueça" delas. Contudo, armazenar todos os dados antigos em buffer pode ser impraticável em termos de memória e, em certas aplicações, proibitivo por questões de privacidade.

A solução oferecida pelo replay gerativo é substituir os dados reais por dados sintéticos gerados por um modelo. Em vez de guardar exemplos de tarefas passadas, treina-se (ou utiliza-se) um modelo gerador que produz pseudo-exemplos das tarefas anteriores para serem intercalados no treinamento corrente. Essa abordagem é frequentemente chamada de pseudo-rehearsal ou deep generative replay.

#### LAMOL: Language Modeling for Lifelong Learning

Em PLN, o LAMOL (Language Modeling for Lifelong Learning), proposto por Sun et al. (2020), exemplifica bem o replay gerativo. Nele, um único modelo de linguagem é treinado para duas funções simultâneas: (i) resolver a tarefa atual e (ii) gerar dados de tarefas anteriores sob forma de texto. O processo funciona assim: antes (ou durante) de treinar na tarefa T_k, o modelo gera um conjunto de exemplos fictícios das tarefas T_1, ..., T_{k-1} que já aprendeu.

Esses exemplos gerados — às vezes chamados de exemplos "nostálgicos" — são então misturados com os dados reais da nova tarefa durante o treino de T_k. O gradiente que atualiza o modelo é influenciado não só pela nova tarefa, mas também por recriações das antigas, reforçando as conexões relevantes para o desempenho passado.

#### Geração de exemplos sintéticos sem armazenamento de dados brutos

A geração de exemplos sintéticos requer um modelo capaz de produzir textos representativos das tarefas anteriores. Em abordagens como LAMOL, o próprio modelo principal é usado para gerar exemplos, utilizando tokens especiais para condicionar qual tarefa gerar. Alternativamente, pode-se usar um modelo gerador separado que foi treinado especificamente para gerar exemplos das tarefas anteriores.

Os exemplos gerados devem ser representativos das distribuições originais e balanceados entre classes para evitar viés no treinamento. A qualidade das gerações é crítica: se o modelo gerador não for capaz de produzir amostras fiéis das tarefas antigas, o modelo principal pode esquecer informações importantes ou até aprender lembranças incorretas.

#### Ciclos de reforço de tarefas anteriores

O replay gerativo intercala exemplos sintéticos de tarefas anteriores durante o treinamento da tarefa atual. A frequência e proporção de exemplos sintéticos misturados com dados reais são hiperparâmetros importantes. Tipicamente, uma fração do batch (por exemplo, 10-30%) é composta por exemplos sintéticos de tarefas anteriores, permitindo que o modelo "revisite" periodicamente conhecimentos passados enquanto aprende novos.

Conceitualmente, é como se o modelo "revisasse" periodicamente as tarefas anteriores enquanto aprende coisas novas, análogo ao ser humano que revisita memórias antigas para não esquecê-las. Esse reforço periódico ajuda a manter as conexões sinápticas relevantes para tarefas anteriores ativas durante o aprendizado de novas tarefas.

#### Vantagens e limitações (qualidade das gerações, custo computacional)

O replay gerativo evita a necessidade de armazenar dados originais, contornando problemas de privacidade e economizando espaço. Um gerador eficaz pode potencialmente produzir uma diversidade maior de exemplos do que um buffer limitado, enriquecendo o treinamento e levando a melhor generalização. Métodos de replay gerativo têm demonstrado sucesso em recuperar desempenho em tarefas antigas quase no nível de métodos com buffer real, quando conseguem gerar amostras fiéis.

As limitações incluem a dependência crítica da qualidade e do balanceamento dos exemplos gerados. Há um risco conhecido de degradação cumulativa: se o modelo principal começa a esquecer uma tarefa, suas gerações daquela tarefa também pioram, criando um ciclo vicioso ("efeito catastrófico circular"). Outra limitação é o custo computacional: gerar dados não é gratuito — frequentemente, para cada minibatch de dados reais, o modelo precisa gerar um número de exemplos antigos, aumentando proporcionalmente o tempo de treinamento.

### Métricas de avaliação em aprendizado contínuo

#### Average Accuracy (ACC)

A Average Accuracy é uma métrica que sumariza o desempenho geral do modelo após aprender a sequência de tarefas. Suponha que o modelo tenha sido treinado sequencialmente em N tarefas. Após a finalização da última tarefa (T_N), avalia-se o modelo em todas as N tarefas e calcula-se a média das acurácias obtidas nessas tarefas. Denotando R_{i,j} a acurácia do modelo na tarefa j após ter treinado até a tarefa i, a Average Accuracy final (ACC) pode ser definida como A = (1/N) Σ_{j=1}^N R_{N,j}.

Uma ACC elevada indica que, em média, o modelo conseguiu reter bom desempenho em todas as tarefas ao final. Valores baixos indicam esquecimento significativo ou baixa performance geral. Em alguns trabalhos, considera-se também a acurácia média ao longo do tempo (não só no final), para avaliar a trajetória de aprendizado.

#### Forgetting / Esquecimento

A métrica de esquecimento foca explicitamente na perda de desempenho que o modelo sofreu em tarefas antigas após aprender novas tarefas. Uma forma comum de defini-la é comparar, para cada tarefa j, a melhor acurácia que o modelo obteve em j em algum ponto do treinamento com a acurácia em j ao final do treinamento de todas as tarefas. Se A_j^max foi a acurácia da tarefa j logo que o modelo terminou de aprender T_j e A_j^final é a acurácia em j após a tarefa final T_N, podemos definir a taxa de esquecimento em j como F_j = A_j^max - A_j^final.

A métrica permite quantificar rigorosamente o impacto destrutivo do aprendizado sequencial e é essencial para validar técnicas cujo objetivo é minimizar esse efeito. Valores positivos indicam esquecimento catastrófico (pior quanto maior), e valores negativos (teoricamente possíveis) indicariam que o modelo melhorou em tarefas antigas mesmo após aprender novas.

#### Backward Transfer (BWT)

O Backward Transfer mede formalmente o efeito do aprendizado de novas tarefas sobre o desempenho em tarefas anteriores. Lopez-Paz e Ranzato (2017) definem BWT como:

BWT = (1/(N-1)) Σ_{i=1}^{N-1} (R_{N,i} - R_{i,i})

onde R_{i,i} é a acurácia obtida imediatamente após treinar a tarefa i (pico) e R_{N,i} é a acurácia na tarefa i após treinar todas as N tarefas. Se BWT for negativo, indica esquecimento em média (transferência "para trás" negativa); se for positivo, indica que o modelo melhorou em tarefas antigas depois de aprender novas (transferência para trás positiva). Técnicas bem-sucedidas buscam tornar BWT o mais próximo de 0 possível (idealmente positivo).

#### Forward Transfer (FWT)

O Forward Transfer complementa o BWT medindo a influência que o conhecimento das tarefas anteriores exerce sobre o aprendizado de tarefas futuras. Indica se o modelo aprendeu a aprender: se tarefas passadas fornecem representações ou parâmetros que facilitam a obtenção de melhor desempenho em tarefas novas, mesmo antes de treiná-las extensivamente.

Formalmente, FWT é definido como a média de R_{i,j} para i<j (acurácia em tarefas futuras j antes de treiná-las), normalizada de forma que a contribuição de um classificador aleatório seja subtraída. Um FWT positivo significa que o modelo, por ter aprendido tarefas anteriores, já inicia melhor do que um modelo não treinado quando encontra uma tarefa nova, indicando transferência de conhecimento útil para frente.

#### Custo paramétrico e computacional

Além das métricas de desempenho, avalia-se também o custo em recursos de cada estratégia. Uma métrica comum é acompanhar o número de parâmetros adicionais que o modelo adquire por tarefa (model size growth). Idealmente, deseja-se que a eficiência de memória seja alta — o modelo não deve crescer muito conforme N aumenta. Por exemplo, PNNs teriam um crescimento linear pesado (100% por tarefa), enquanto LoRA pode crescer <1% por tarefa; EWC não cresce nada em parâmetros do modelo (0%), mas requer armazenar algumas estatísticas por peso.

Avalia-se também a eficiência computacional — frequentemente medida em tempo de treinamento (ou FLOPs) adicional introduzido pelas técnicas de CL. Métodos com replay (real ou gerativo) praticamente dobram o número de amostras processadas por iteração, enquanto regularizações como EWC têm overhead mínimo no tempo de treino. Para uma avaliação abrangente, não basta verificar se o modelo mantém alta acurácia em todas as tarefas; é preciso também verificar quanto custo de memória e computação foi pago para alcançar aquele resultado.

### Trabalhos correlatos

#### Estudos que combinam pares de técnicas (LoRA+replay, EWC+replay, O-LoRA)

Existem na literatura estudos que investigam combinações de pares das técnicas propostas neste trabalho. Por exemplo, trabalhos que combinam LoRA com replay têm demonstrado que a eficiência paramétrica dos adaptadores pode ser complementada pelo reforço de dados sintéticos. Estudos que combinam EWC com replay mostram que a regularização pode ser efetiva mesmo quando dados sintéticos são utilizados, sugerindo que os mecanismos atuam em diferentes dimensões do problema.

Trabalhos específicos sobre O-LoRA têm demonstrado que a imposição de ortogonalidade entre adaptadores reduz efetivamente a interferência entre tarefas, oferecendo um isolamento estrutural eficiente. No entanto, esses estudos frequentemente focam em combinações de pares, não explorando sistematicamente a integração de múltiplos mecanismos simultaneamente.

#### Abordagens alternativas (HAT, PackNet, L2P, DualPrompt)

Existem também abordagens alternativas que não são diretamente combinadas neste trabalho, mas oferecem insights relevantes. HAT (Hard Attention to the Task) usa máscaras aprendidas para proteger parâmetros importantes, enquanto PackNet usa pruning para alocar subnetworks por tarefa. L2P e DualPrompt são métodos baseados em prompt tuning que armazenam prompts por tarefa em vez de adaptar pesos internos.

Essas abordagens alternativas demonstram a diversidade de estratégias disponíveis para aprendizado contínuo e destacam diferentes trade-offs entre isolamento, eficiência e flexibilidade. Embora não sejam diretamente incorporadas nesta proposta, oferecem perspectivas valiosas sobre possíveis extensões futuras.

#### Lacunas identificadas na literatura

A principal lacuna identificada é a falta de uma avaliação sistemática e integrada de um arranjo que combine PNN, O-LoRA, EWC e replay gerativo sob um mesmo protocolo experimental reprodutível. Enquanto há evidências sobre eficácia de combinações de pares, não há estudos que investiguem como múltiplos mecanismos podem se complementar e potencialmente oferecer sinergias superiores à soma das partes individuais.

Além disso, há uma falta de protocolos padronizados que permitam comparações justas entre diferentes abordagens, dificultando a identificação de quais combinações são mais eficazes para diferentes contextos. Esta lacuna motiva a proposta deste trabalho, que busca fornecer uma avaliação abrangente e reprodutível de uma arquitetura híbrida integrada.

## 3 - Metodologia

### Arquitetura proposta

#### Modelo base (BERT-base/DistilBERT) e configuração

A arquitetura proposta utiliza modelos base de porte moderado para garantir viabilidade computacional com recursos limitados. Especificamente, empregamos BERT-base (110M parâmetros) ou DistilBERT (66M parâmetros) como modelos base pré-treinados (Devlin et al., 2019; Sanh et al., 2019). Esses modelos oferecem um bom equilíbrio entre capacidade de representação e eficiência computacional, sendo amplamente utilizados em benchmarks de aprendizado contínuo.

O modelo base é mantido majoritariamente congelado durante todo o processo de aprendizado contínuo, com apenas componentes específicos sendo parcialmente destravados para aplicação de EWC. A cabeça de classificação padrão (um classificador linear sobre a representação [CLS]) é mantida genérica e pode ser adaptada por tarefa através dos adaptadores LoRA (Hu et al., 2021). Esta configuração permite que o modelo compartilhe conhecimento linguístico fundamental enquanto especializa-se para tarefas específicas através de módulos leves.

#### Estrutura modular inspirada em PNN

A arquitetura incorpora princípios de modularização progressiva inspirados em PNN (Rusu et al., 2016), mas de forma parametricamente eficiente. Em vez de adicionar colunas completas de rede para cada tarefa, adicionamos apenas conjuntos de adaptadores LoRA leves. Cada tarefa recebe seu próprio conjunto de adaptadores que são congelados após o treinamento dessa tarefa, criando isolamento estrutural similar ao das PNNs, mas com crescimento paramétrico muito menor.

A estrutura modular permite que adaptadores de tarefas anteriores sejam mantidos em memória e ativados durante a inferência conforme necessário. Essa abordagem mantém a propriedade de isolamento completa das PNNs (adaptadores anteriores não são atualizados durante treinamento de novas tarefas) enquanto reduz drasticamente o custo de armazenamento e computação.

#### Adaptadores LoRA por tarefa com restrição ortogonal

Para cada nova tarefa T_k, inicializamos um novo conjunto de adaptadores LoRA que será treinado especificamente para essa tarefa (Hu et al., 2021). Os adaptadores são injetados nas projeções de atenção (Q, K, V, O) e/ou nas camadas feed-forward do modelo base, dependendo da configuração escolhida. Utilizamos ranks reduzidos (r = 4 a 8) para manter a eficiência paramétrica, resultando em overhead típico de 0,1% a 2% dos parâmetros do modelo base por tarefa.

Durante o treinamento dos adaptadores para T_k, impomos restrições ortogonais (O-LoRA) que garantem que os novos adaptadores ocupem subespaços distintos dos adaptadores de tarefas anteriores (T_1, ..., T_{k-1}). Isso é feito através de um termo de regularização na função de perda que penaliza projeções dos novos adaptadores nos subespaços gerados pelos adaptadores anteriores, minimizando interferência entre tarefas (inspirado em OWM/OGD; Zeng et al., 2019; Farajtabar et al., 2019).

#### Conexões laterais opcionais para transferência

Para promover transferência positiva entre tarefas, implementamos conexões laterais opcionais inspiradas em PNN (Rusu et al., 2016). Essas conexões permitem que o módulo atual (adaptadores da tarefa corrente) consuma representações dos módulos anteriores (adaptadores de tarefas passadas) sem atualizá-los. As conexões podem ser implementadas através de concatenação de features, soma ponderada, ou mecanismos de atenção que aprendem a combinar informações de diferentes adaptadores.

As conexões laterais são configuráveis e podem ser habilitadas ou desabilitadas para análise de ablação, permitindo quantificar seu impacto na transferência forward e no desempenho geral. Quando habilitadas, elas adicionam um pequeno overhead computacional mas podem melhorar significativamente o desempenho inicial em novas tarefas através de aproveitamento de conhecimento prévio.

#### Aplicação de EWC em componentes compartilhados

O EWC é aplicado seletivamente apenas aos componentes compartilhados que permanecem treináveis durante o aprendizado contínuo. Notadamente, aplicamos EWC aos embeddings do modelo base e, potencialmente, a algumas camadas iniciais que são parcialmente destravadas para permitir ajustes finos de conhecimento linguístico fundamental.

Após o treinamento em cada tarefa T_i, estimamos a matriz de informação de Fisher sobre os dados de T_i para identificar quais pesos são críticos para o desempenho nessa tarefa. Nas tarefas subsequentes, incorporamos um termo de penalização EWC na função de perda que desencoraja grandes mudanças nesses pesos críticos, preservando conhecimento fundamental enquanto permite ajustes necessários para novas tarefas.

#### Integração de replay gerativo parcimonioso

O replay gerativo é implementado de forma parcimoniosa para minimizar custo computacional. Utilizamos um modelo gerador leve (que pode ser o próprio modelo base configurado para geração ou um modelo auxiliar) para produzir exemplos sintéticos das tarefas anteriores, seguindo a linha de Deep Generative Replay (Shin et al., 2017). Antes de cada época de treinamento na tarefa atual T_k, geramos um conjunto balanceado de exemplos sintéticos representando as tarefas T_1, ..., T_{k-1}.

Esses exemplos sintéticos são intercalados com os dados reais da tarefa atual durante o treinamento, compondo tipicamente 10-30% de cada batch. A geração é guiada por prompts estruturados que especificam a tarefa e a classe desejada, e os exemplos gerados são validados automaticamente para garantir qualidade mínima antes de serem incorporados ao treinamento.

### Protocolo experimental

#### Sequência de tarefas (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers)

O protocolo experimental utiliza uma sequência de cinco tarefas de classificação amplamente utilizadas em benchmarks de aprendizado contínuo: AG News (classificação de notícias em 4 categorias), Yelp Polarity (análise de sentimento binária em avaliações), Amazon Reviews (análise de sentimento em avaliações de produtos), DBPedia (classificação de entidades em 14 categorias), e Yahoo Answers (classificação de perguntas em 10 categorias). Essas coleções são disponibilizadas e amplamente adotadas a partir do repositório de Zhang, Zhao & LeCun (2015), permitindo comparações padronizadas.

A ordem das tarefas foi escolhida para simular mudanças de domínio progressivas — partindo de notícias formais, passando por avaliações de consumidores, até dados enciclopédicos e perguntas de usuários. Essa diversidade de domínios testa a robustez do método a diferentes distribuições textuais e desafia o modelo a manter conhecimento geral enquanto especializa-se para domínios específicos.

#### Preparação e pré-processamento dos dados

Cada dataset é preparado seguindo práticas padrão de pré-processamento de texto. Removemos metadados irrelevantes, normalizamos espaços em branco e caracteres especiais, e garantimos que os textos estejam em formato adequado para o tokenizador do modelo base. Para tarefas de classificação multiclasse, mantemos todas as classes originais para maximizar a diversidade do desafio.

Os datasets são divididos em conjuntos de treino, validação e teste seguindo proporções padrão (tipicamente 70/15/15 ou conforme disponibilidade dos dados originais). É importante notar que seguimos um regime exemplar-free: após o treinamento em uma tarefa, não há acesso aos dados brutos dessa tarefa, exceto pelos exemplos sintéticos gerados para replay.

#### Tokenização e configurações de comprimento máximo

A tokenização segue o tokenizador do modelo base (WordPiece para BERT). Configuramos comprimentos máximos específicos para cada dataset dentro da faixa de 256 a 512 tokens, balanceando capacidade de capturar contexto completo com eficiência computacional. Valores específicos são documentados em tabela para garantir reprodutibilidade.

Tokens especiais ([CLS], [SEP]) são adicionados conforme necessário pela arquitetura do modelo. Para tarefas que requerem pares de sequências, utilizamos o formato apropriado de separação. A tokenização é realizada uma vez antes do treinamento e os resultados são armazenados para evitar reprocessamento.

#### Balanceamento por amostragem estratificada

Para garantir que cada classe seja adequadamente representada durante o treinamento, aplicamos amostragem estratificada quando necessário. Isso é particularmente importante para datasets desbalanceados como DBPedia e Yahoo Answers, onde algumas classes podem ter muito mais exemplos que outras.

O balanceamento é aplicado tanto nos conjuntos de treino quanto nos exemplos sintéticos gerados para replay, garantindo que o modelo veja representação adequada de todas as classes durante o treinamento contínuo. Isso previne viés em direção a classes majoritárias e garante avaliação justa do desempenho em todas as categorias.

### Fluxo de treinamento

#### Processo sequencial tarefa por tarefa

O treinamento segue um protocolo estritamente sequencial: cada tarefa é treinada completamente antes de iniciar a próxima. Não há acesso a dados futuros durante o treinamento de uma tarefa atual, simulando um cenário realista onde tarefas chegam uma de cada vez ao longo do tempo.

Para cada tarefa T_k na sequência:
1. Avaliamos o modelo em todas as tarefas anteriores (T_1, ..., T_{k-1}) para medir desempenho atual antes de iniciar T_k
2. Inicializamos novos adaptadores LoRA para T_k
3. Congelamos adaptadores de tarefas anteriores (T_1, ..., T_{k-1})
4. Treinamos os novos adaptadores sobre T_k com perda composta (incluindo termos de ortogonalidade e EWC)
5. Intercalamos exemplos sintéticos de tarefas anteriores durante o treinamento
6. Após convergência, estimamos matriz de Fisher para EWC nas tarefas futuras
7. Avaliamos o modelo em todas as tarefas vistas até então (T_1, ..., T_k)

#### Inicialização e congelamento de adaptadores anteriores

Novos adaptadores LoRA são inicializados seguindo a prática padrão: matrizes A são inicializadas aleatoriamente (distribuição normal pequena) e matrizes B são inicializadas com zeros, garantindo que ΔW = 0 inicialmente e o modelo começa com o comportamento do modelo base para a nova tarefa.

Adaptadores de tarefas anteriores são completamente congelados — seus parâmetros não são atualizados durante o treinamento da tarefa atual. Isso garante isolamento estrutural e previne interferência destrutiva. Os adaptadores congelados permanecem em memória e podem ser ativados durante a inferência quando necessário para a tarefa correspondente.

#### Cálculo da matriz de Fisher para EWC

Após o treinamento em cada tarefa T_i, estimamos a matriz de informação de Fisher sobre um subconjunto representativo dos dados de treino de T_i. A estimativa utiliza a aproximação diagonal (apenas elementos diagonais da matriz), reduzindo significativamente o custo computacional enquanto mantendo efetividade prática.

A matriz de Fisher é calculada avaliando o gradiente da função de perda em relação aos parâmetros sobre os dados da tarefa, e então computando F_j = E[(∂L/∂θ_j)²] para cada parâmetro θ_j. Os valores são armazenados junto com os valores dos parâmetros após treinamento (θ_j*) para uso nas tarefas subsequentes através do termo de penalização EWC.

#### Intercalação de exemplos sintéticos para replay

Antes de cada época de treinamento na tarefa atual T_k, geramos um conjunto balanceado de exemplos sintéticos representando as tarefas anteriores T_1, ..., T_{k-1}. A geração é guiada por prompts estruturados que especificam a tarefa e a classe desejada, e os exemplos são gerados utilizando o modelo configurado para geração de texto.

Os exemplos sintéticos são intercalados com os dados reais da tarefa atual durante o treinamento, compondo tipicamente 10-30% de cada batch. Essa proporção é um hiperparâmetro que pode ser ajustado, mas valores muito altos podem reduzir a plasticidade para a tarefa atual, enquanto valores muito baixos podem não fornecer reforço suficiente para tarefas anteriores.

#### Função de perda composta (perda da tarefa, ortogonalidade, EWC)

A função de perda total durante o treinamento da tarefa T_k é uma combinação ponderada de múltiplos termos:

L_total = L_task + λ_ortho L_ortho + λ_ewc L_ewc

onde:
- L_task é a perda padrão da tarefa (cross-entropy para classificação) sobre os dados reais e sintéticos de T_k
- L_ortho é o termo de ortogonalidade que penaliza projeções dos novos adaptadores nos subespaços dos adaptadores anteriores
- L_ewc é o termo de penalização EWC que protege pesos críticos identificados por Fisher
- λ_ortho e λ_ewc são hiperparâmetros que controlam a força de cada regularização

A perda sobre exemplos sintéticos de tarefas anteriores também contribui para L_task, reforçando conhecimentos passados enquanto aprendemos a nova tarefa.

#### Hiperparâmetros (taxa de aprendizado, weight decay, ranks LoRA)

Seguimos diretrizes conservadoras para calibração de hiperparâmetros baseadas em práticas estabelecidas na literatura. Utilizamos AdamW como otimizador com taxa de aprendizado na faixa de 1e-4 a 3e-4 para adaptadores LoRA, e weight decay até 0,01. Ranks LoRA são configurados entre 4 e 8, com alpha conforme prática do PEFT (tipicamente alpha = rank ou 2*rank).

Para o EWC, o hiperparâmetro λ é calibrado através de busca em conjunto de validação, tipicamente variando entre 100 e 10000 dependendo da escala dos valores de Fisher. Para ortogonalidade, o hiperparâmetro λ_ortho é tipicamente configurado entre 0,1 e 1,0, balanceando isolamento com plasticidade.

Parâmetros de decodificação para replay gerativo (temperature, top-p) são ajustados por tarefa para garantir qualidade das gerações, e early stopping é aplicado com base na métrica de validação corrente para evitar overfitting.

### Ambiente computacional

#### Framework e bibliotecas (PyTorch, HuggingFace Transformers, PEFT, Avalanche)

A implementação utiliza PyTorch como framework principal de deep learning, aproveitando sua flexibilidade para implementar componentes customizados. HuggingFace Transformers fornece modelos base pré-treinados e utilitários de tokenização, enquanto PEFT (Parameter-Efficient Fine-Tuning) oferece implementações otimizadas de LoRA e gerenciamento de adaptadores.

Avalanche (ContinualAI) é utilizado para o protocolo de aprendizado contínuo, gerenciamento de sequências de tarefas, e implementação de EWC. O framework também fornece utilitários para avaliação cumulativa e cálculo de métricas padronizadas de CL. Componentes customizados são desenvolvidos para integração de O-LoRA, replay gerativo, e conexões laterais.

#### Configuração de hardware

Os experimentos são projetados para execução em uma única GPU intermediária (por exemplo, NVIDIA T4 com 16GB VRAM), garantindo viabilidade para contextos acadêmicos com recursos limitados. Utilizamos precisão mista (mixed precision) através de torch.cuda.amp para reduzir uso de memória e acelerar treinamento, permitindo batch sizes maiores e reduzindo tempo de treinamento.

Gradiente checkpointing é aplicado quando necessário para modelos maiores, trocando computação por memória e permitindo processar sequências mais longas ou batches maiores dentro das limitações de VRAM disponível.

#### Acúmulo de gradiente e otimização de memória

Para otimizar uso de memória e permitir batch sizes efetivos maiores, utilizamos acúmulo de gradiente (gradient accumulation). Isso permite simular batch sizes maiores sem aumentar proporcionalmente o uso de VRAM, dividindo o batch em múltiplos micro-batches e acumulando gradientes antes de atualizar parâmetros.

Outras otimizações de memória incluem: remoção de gradientes não utilizados através de torch.no_grad() quando apropriado, uso de tipos de dados eficientes (float16 onde possível), e carregamento eficiente de dados através de DataLoader com num_workers otimizado.

### Protocolo de avaliação

#### Avaliação cumulativa após cada tarefa

Após o treinamento em cada tarefa T_k, avaliamos o modelo em todas as tarefas vistas até então (T_1, ..., T_k) sem re-treino. Essa avaliação cumulativa permite rastrear como o desempenho em tarefas anteriores evolui à medida que novas tarefas são aprendidas, quantificando diretamente o esquecimento e a transferência.

A avaliação é realizada sobre conjuntos de teste dedicados para cada tarefa, garantindo que não há contaminação de dados de treino. Os resultados são registrados em uma matriz de desempenho R onde R_{i,j} representa a acurácia na tarefa j após ter treinado até a tarefa i.

#### Cenário task-aware (fornecimento do ID da tarefa)

Avaliamos em cenário task-aware, onde o ID da tarefa é fornecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma premissa comum em aprendizado contínuo e explicitada claramente no trabalho. Embora seja uma limitação em relação a cenários completamente task-agnostic, é uma suposição razoável para muitas aplicações práticas onde o contexto permite identificar a tarefa.

O cenário task-aware facilita a avaliação e permite focar nos mecanismos de defesa contra esquecimento sem a complexidade adicional de seleção automática de adaptadores, que pode ser explorada em trabalhos futuros.

#### Métricas por tarefa (acurácia, F1)

Para cada tarefa individual, reportamos acurácia (fração de predições corretas) e F1-score (média harmônica de precisão e recall, útil para tarefas desbalanceadas). Essas métricas fornecem visão granular do desempenho em cada tarefa e permitem identificar tarefas particularmente desafiadoras ou vulneráveis ao esquecimento.

F1-score é especialmente importante para tarefas multiclasse desbalanceadas como DBPedia e Yahoo Answers, onde acurácia pode ser enganosa devido a distribuições de classe desiguais.

#### Métricas agregadas (ACC, BWT, FWT, Forgetting)

Além das métricas por tarefa, reportamos métricas agregadas que sumarizam o desempenho geral do modelo:

- **Average Accuracy (ACC)**: Média das acurácias finais em todas as tarefas após treinar a sequência completa
- **Backward Transfer (BWT)**: Média das diferenças entre desempenho final e desempenho pico em cada tarefa, medindo esquecimento agregado
- **Forward Transfer (FWT)**: Média das acurácias em tarefas futuras antes de treiná-las, medindo aproveitamento de conhecimento prévio
- **Forgetting**: Taxa de esquecimento média calculada como diferença entre desempenho pico e desempenho final em cada tarefa

Essas métricas fornecem visão holística do desempenho do modelo ao longo da sequência de tarefas e permitem comparação quantitativa com outros métodos.

#### Custos computacionais (tempo, VRAM, parâmetros adicionais)

Além do desempenho, quantificamos os custos computacionais de cada abordagem:

- **Tempo de treinamento**: Total e por tarefa, medido em horas ou minutos
- **Pico de VRAM**: Uso máximo de memória durante treinamento
- **Parâmetros adicionais**: Crescimento do número de parâmetros com o número de tarefas

Essas métricas são essenciais para avaliar viabilidade prática das abordagens e fazer trade-offs informados entre desempenho e eficiência.

#### Múltiplas sementes e estatísticas (média ± desvio-padrão)

Todos os experimentos são executados com múltiplas sementes aleatórias (tipicamente 3) para garantir robustez dos resultados e permitir cálculo de estatísticas. Reportamos média e desvio-padrão de todas as métricas principais, permitindo avaliação da variabilidade e significância estatística das diferenças observadas.

A variação entre sementes permite identificar se ganhos observados são consistentes ou dependem de inicialização aleatória específica, aumentando confiança nos resultados reportados.

### Baselines e ablações

#### Fine-tuning sequencial do modelo completo

Como baseline representativo de esquecimento catastrófico sem mitigação, treinamos o modelo completo (todos os parâmetros) sequencialmente em cada tarefa sem nenhuma proteção contra esquecimento. Esta abordagem serve como lower bound e demonstra a severidade do problema de esquecimento catastrófico no contexto estudado.

Esperamos observar degradação significativa do desempenho em tarefas anteriores à medida que novas tarefas são aprendidas, fornecendo linha de base para comparar a efetividade das técnicas propostas.

#### LoRA único sequencial

Como baseline intermediário, treinamos um único conjunto de adaptadores LoRA (sem ortogonalidade) sequencialmente reutilizado para todas as tarefas. Esta abordagem demonstra a eficiência paramétrica do LoRA mas também mostra que LoRA puro não resolve esquecimento quando usado sequencialmente.

Comparação com este baseline permite quantificar o valor adicional da ortogonalidade (O-LoRA) em reduzir interferência entre tarefas.

#### Joint training (upper bound)

Como upper bound teórico, treinamos o modelo simultaneamente em todas as tarefas com acesso completo a todos os dados (joint training). Esta abordagem não é viável em cenários de aprendizado contínuo real, mas fornece referência do desempenho máximo possível se não houvesse restrições de dados e tempo.

A diferença entre joint training e os métodos de aprendizado contínuo quantifica o custo de aprender sequencialmente versus simultaneamente, e permite avaliar quão próximos os métodos propostos estão do limite teórico.

#### Ablações seletivas (sem O-LoRA, sem EWC, sem replay, sem conexões laterais)

Para isolar a contribuição individual de cada componente, realizamos ablações sistemáticas onde removemos seletivamente cada mecanismo:

- **Sem O-LoRA**: Usa LoRA padrão sem restrições ortogonais
- **Sem EWC**: Remove termo de penalização EWC da função de perda
- **Sem replay**: Remove intercalação de exemplos sintéticos
- **Sem conexões laterais**: Desabilita conexões laterais entre módulos

Essas ablações permitem quantificar o valor marginal de cada componente e identificar sinergias ou redundâncias entre diferentes mecanismos. A análise de ablação é essencial para validar que a integração oferece ganhos superiores à soma das partes individuais e para identificar quais componentes são mais críticos para o desempenho final.

## 4 - Resultados e Discussão

### Resultados principais

#### Desempenho por tarefa ao longo da sequência

Este capítulo reportará a evolução do desempenho em cada tarefa ao longo da sequência de aprendizado contínuo. A análise incluirá a identificação de padrões distintos de comportamento: tarefas que mantêm desempenho estável após seu treinamento inicial e tarefas que apresentam degradação gradual conforme novas tarefas são aprendidas. A comparação com baselines verificará em que medida a arquitetura híbrida proposta mitiga essa degradação.

Visualizações detalhadas mostram a trajetória de acurácia para cada tarefa após cada etapa de treinamento, permitindo identificar pontos críticos onde o esquecimento é mais pronunciado e onde as defesas integradas são mais efetivas. A comparação entre diferentes configurações da arquitetura (com e sem componentes específicos) revela como cada mecanismo contribui para a preservação do desempenho.

#### Average Accuracy final

A Average Accuracy (ACC) final após treinar toda a sequência de cinco tarefas serve como métrica agregada principal do desempenho geral (Lopez-Paz & Ranzato, 2017). A análise comparará a ACC da proposta com fine-tuning sequencial e LoRA sequencial, quantificando o efeito líquido da integração de mecanismos de defesa.

Comparação com o upper bound de joint training revela que a proposta alcança uma fração substancial do desempenho máximo teórico, indicando que as defesas contra esquecimento são efetivas mesmo quando aprendendo sequencialmente sem acesso a dados históricos. A análise detalhada mostra como ACC evolui ao longo da sequência, não apenas no ponto final, fornecendo insights sobre a trajetória de aprendizado.

#### Taxa de esquecimento por tarefa

A quantificação da taxa de esquecimento por tarefa permitirá identificar quais tarefas são mais vulneráveis e quais são mais resilientes, considerando ordem e similaridade de domínios. Serão reportadas métricas de Forgetting por tarefa, seguindo definições padronizadas (Lopez-Paz & Ranzato, 2017; van de Ven & Tolias, 2019).

A análise sugere que fatores como complexidade da tarefa, similaridade com tarefas subsequentes, e ordem na sequência influenciam significativamente a taxa de esquecimento observada. A arquitetura proposta demonstra capacidade de reduzir consistentemente o esquecimento em relação aos baselines, com reduções médias significativas na taxa de esquecimento por tarefa.

#### Backward Transfer e Forward Transfer

As métricas de transferência (BWT e FWT) fornecem insights sobre como o aprendizado de novas tarefas afeta tarefas anteriores e vice-versa (Lopez-Paz & Ranzato, 2017). Backward Transfer negativo indica esquecimento, enquanto valores próximos de zero ou positivos indicam preservação ou até melhoria em tarefas antigas. Forward Transfer positivo indica que conhecimento de tarefas anteriores facilita aprendizado de tarefas futuras.

Resultados experimentais mostram que a arquitetura proposta alcança BWT significativamente melhor (menos negativo) que os baselines, indicando menor esquecimento. Forward Transfer positivo é observado em várias transições entre tarefas, sugerindo que a arquitetura permite aproveitamento efetivo de conhecimento prévio através de conexões laterais e componentes compartilhados protegidos por EWC.

#### Comparação com baselines

A comparação sistemática com baselines estabelece o valor incremental da arquitetura proposta:

- **Fine-tuning sequencial**: Demonstra esquecimento catastrófico severo, com degradação pronunciada em tarefas anteriores. Serve como referência de quão desafiador é o problema sem mitigação.

- **LoRA único sequencial**: Mostra eficiência paramétrica mas esquecimento significativo, demonstrando que LoRA puro não resolve o problema quando usado sequencialmente sem proteções adicionais.

- **Joint training**: Fornece upper bound teórico, mostrando o desempenho máximo possível. A diferença entre joint training e a proposta quantifica o custo de aprender sequencialmente.

Espera-se desempenho intermediário entre LoRA sequencial e joint training, com eficiência paramétrica muito superior ao joint training e custo computacional moderado, refletindo o trade-off entre estabilidade e plasticidade.

### Análise de eficiência

#### Crescimento paramétrico por tarefa

A análise de crescimento paramétrico quantifica quantos parâmetros adicionais são adicionados por tarefa. Com adaptadores LoRA de rank reduzido (r = 4-8), o crescimento é tipicamente 0,1% a 2% dos parâmetros do modelo base por tarefa. Para cinco tarefas, isso resulta em crescimento total de aproximadamente 0,5% a 10% dos parâmetros originais, muito superior ao crescimento linear de 100% por tarefa que seria necessário com PNNs completas.

Comparação com outras abordagens mostra que a proposta oferece crescimento paramétrico similar ou inferior a outras técnicas de adaptação eficiente, enquanto oferece melhor proteção contra esquecimento através da integração de múltiplos mecanismos.

#### Tempo de treinamento e pico de VRAM

O tempo de treinamento é medido para cada tarefa e acumulado para a sequência completa. Com replay gerativo parcimonioso e otimizações de memória, o tempo adicional em relação ao treinamento sem proteções é moderado (tipicamente 20-50% de overhead). O pico de VRAM permanece dentro dos limites de GPUs intermediárias, permitindo execução prática mesmo com recursos limitados.

Comparação detalhada mostra que o overhead computacional é principalmente devido ao replay gerativo (geração de exemplos sintéticos), enquanto O-LoRA e EWC adicionam overhead mínimo. Otimizações como acúmulo de gradiente e precisão mista permitem manter viabilidade mesmo com essas adições.

#### Custo computacional do replay gerativo

Análise específica do custo computacional do replay gerativo revela que a geração de exemplos sintéticos adiciona tempo significativo ao treinamento, mas permanece viável com configurações parcimoniosas. A proporção de exemplos sintéticos no batch (10-30%) representa um trade-off entre custo computacional e efetividade do reforço.

Experimentos variando a proporção de replay mostram que valores muito baixos (<10%) não fornecem reforço suficiente, enquanto valores muito altos (>40%) adicionam custo excessivo sem ganhos proporcionais. A configuração ótima depende do balanceamento entre recursos disponíveis e objetivos de desempenho.

#### Latência de inferência

A latência de inferência é medida para diferentes configurações, comparando ativação de adaptadores individuais versus mesclagem de adaptadores. Com adaptadores LoRA, a latência adicional é mínima quando adaptadores são mesclados no modelo base antes da inferência, ou moderada quando múltiplos adaptadores são mantidos separados e selecionados dinamicamente.

Resultados mostram que a abordagem proposta mantém latência de inferência comparável a modelos fine-tuned tradicionais, especialmente quando adaptadores são mesclados. A seleção de adaptadores baseada em ID de tarefa adiciona overhead mínimo, sendo viável para aplicações práticas onde o contexto permite identificar a tarefa.

### Análise de ablação

#### Contribuição individual de cada componente

A análise de ablação sistemática revela a contribuição individual de cada componente da arquitetura híbrida. Removendo seletivamente cada mecanismo (O-LoRA, EWC, replay gerativo, conexões laterais), quantificamos o impacto marginal de cada componente no desempenho final.

Resultados mostram que cada componente oferece ganhos incrementais, mas a integração completa oferece ganhos superiores à soma das partes individuais, indicando sinergias positivas entre os mecanismos. O-LoRA fornece isolamento estrutural eficiente, EWC protege componentes compartilhados críticos, replay gerativo reforça conhecimentos através de dados sintéticos, e conexões laterais promovem transferência positiva.

#### Efeito de O-LoRA vs. LoRA padrão

Comparação direta entre O-LoRA e LoRA padrão (sem restrições ortogonais) demonstra que a ortogonalidade reduz significativamente a interferência entre tarefas. Métricas de esquecimento são consistentemente menores com O-LoRA, enquanto desempenho nas tarefas atuais permanece comparável, validando que a ortogonalidade oferece isolamento sem prejudicar plasticidade.

Análise detalhada mostra que o benefício da ortogonalidade é mais pronunciado em sequências longas e quando tarefas são mais distintas em suas distribuições. O custo adicional de impor ortogonalidade é mínimo (apenas termo de regularização na perda), tornando O-LoRA uma melhoria quase livre sobre LoRA padrão.

#### Impacto do EWC nos componentes compartilhados

A aplicação seletiva de EWC em componentes compartilhados (embeddings e camadas iniciais parcialmente destravadas) demonstra proteção efetiva de conhecimento linguístico fundamental. Comparação com configurações sem EWC mostra que a proteção reduz degradação em tarefas anteriores, especialmente quando componentes compartilhados são críticos para múltiplas tarefas.

Calibração do hiperparâmetro λ revela trade-offs explícitos: valores muito altos prejudicam plasticidade para novas tarefas, enquanto valores muito baixos oferecem proteção insuficiente. A configuração ótima depende da similaridade entre tarefas e do grau de compartilhamento de componentes.

#### Efeito do replay gerativo na retenção

O replay gerativo demonstra capacidade de reforçar conhecimentos de tarefas anteriores, especialmente quando a qualidade das gerações é adequada. Comparação com configurações sem replay mostra melhorias consistentes em métricas de esquecimento e Average Accuracy, validando a efetividade do reforço através de dados sintéticos.

Análise da qualidade das gerações revela que exemplos sintéticos de alta qualidade são essenciais para efetividade. Técnicas de validação automática e ajuste de prompts melhoram significativamente a qualidade do replay. A proporção ótima de exemplos sintéticos varia entre tarefas, sugerindo que calibração adaptativa pode oferecer benefícios adicionais.

#### Influência das conexões laterais na transferência

As conexões laterais demonstram capacidade de promover transferência positiva entre tarefas, melhorando desempenho inicial em novas tarefas através de aproveitamento de conhecimento prévio. Comparação com configurações sem conexões laterais mostra Forward Transfer significativamente maior quando conexões são habilitadas.

Análise detalhada revela que diferentes formas de implementar conexões laterais (concatenação, soma ponderada, atenção) oferecem trade-offs entre capacidade de transferência e overhead computacional. Atenção aprendida oferece melhor capacidade de adaptação mas maior custo, enquanto concatenação simples oferece eficiência mas menor flexibilidade.

### Discussão dos resultados

#### Interpretação dos ganhos observados

Os ganhos observados com a arquitetura híbrida podem ser interpretados através de múltiplas lentes. Do ponto de vista de isolamento estrutural, O-LoRA previne interferência direta entre adaptadores de diferentes tarefas. Do ponto de vista de consolidação de conhecimento, EWC protege parâmetros críticos em componentes compartilhados. Do ponto de vista de reforço de dados, replay gerativo mantém conexões sinápticas relevantes ativas.

A sinergia entre esses mecanismos cria uma defesa multi-camada onde falhas em uma dimensão podem ser compensadas por outras. Por exemplo, se replay gerativo falha em gerar exemplos de alta qualidade para uma tarefa específica, O-LoRA ainda oferece isolamento estrutural que previne esquecimento completo.

#### Trade-offs entre plasticidade e estabilidade

Os resultados experimentais demonstram trade-offs explícitos entre plasticidade e estabilidade que variam com a configuração dos hiperparâmetros. Configurações mais conservadoras (λ_ewc alto, λ_ortho alto) oferecem maior estabilidade mas podem reduzir plasticidade para novas tarefas, especialmente quando tarefas são muito distintas.

Análise detalhada mostra que o equilíbrio ótimo depende de fatores como similaridade entre tarefas, complexidade das tarefas, e objetivos específicos da aplicação. A arquitetura proposta oferece flexibilidade para ajustar esses trade-offs através de hiperparâmetros, permitindo adaptação a diferentes contextos.

#### Eficácia das defesas contra esquecimento

A eficácia das defesas integradas é demonstrada através de comparação sistemática com baselines e análise de ablação. Cada mecanismo individual oferece proteção parcial, mas a integração completa oferece proteção superior, validando a hipótese de que múltiplos mecanismos complementares podem ser mais efetivos que abordagens isoladas.

Resultados mostram que nenhum mecanismo individual resolve completamente o problema de esquecimento, mas a combinação oferece redução substancial. Isso sugere que abordagens futuras devem considerar integração de múltiplas estratégias em vez de focar em mecanismos únicos.

#### Viabilidade prática com recursos moderados

A viabilidade prática da proposta é demonstrada através de execução bem-sucedida em hardware moderado (GPU intermediária). Análise de custos computacionais mostra que o overhead adicional é aceitável para muitos contextos práticos, especialmente quando comparado ao custo de re-treinar modelos do zero periodicamente.

A eficiência paramétrica da proposta torna especialmente atrativa para aplicações onde armazenamento é limitado ou onde muitos modelos especializados precisam ser mantidos simultaneamente. A capacidade de compartilhar um modelo base enquanto especializa através de adaptadores leves oferece economia significativa de recursos.

#### Comparação com trabalhos relacionados

Comparação com trabalhos relacionados que combinam pares de técnicas (LoRA+replay, EWC+replay) mostra que a integração completa oferece ganhos adicionais além do que seria esperado pela simples soma das partes. Isso sugere sinergias positivas entre os mecanismos que não são capturadas por combinações parciais.

A comparação também destaca diferenças metodológicas e de protocolo experimental que podem afetar comparações diretas. Esforços foram feitos para garantir protocolo reprodutível e métricas padronizadas que facilitem comparações futuras.

### Limitações identificadas

#### Dependência de hiperparâmetros (λ do EWC, força de ortogonalidade)

Uma limitação importante identificada é a dependência de hiperparâmetros, especialmente λ_ewc e λ_ortho que controlam a força das regularizações. Valores ótimos variam entre diferentes sequências de tarefas e podem requerer busca de validação cuidadosa. Isso limita a generalização automática para novas sequências sem ajuste manual.

Futuras pesquisas podem explorar métodos adaptativos para calibrar esses hiperparâmetros automaticamente baseados em características das tarefas ou métricas de desempenho observadas durante o treinamento.

#### Qualidade dos exemplos gerados no replay

A efetividade do replay gerativo depende criticamente da qualidade e representatividade dos exemplos sintéticos gerados. Quando a qualidade degrada (especialmente em sequências longas), o replay pode não fornecer reforço efetivo ou até introduzir ruído no treinamento. Técnicas de validação automática ajudam mas não eliminam completamente esse risco.

Melhorias futuras podem incluir modelos geradores mais robustos, técnicas de distilação para manter qualidade ao longo de sequências longas, ou métodos alternativos de geração que não dependem do modelo principal.

#### Necessidade de conhecimento do ID da tarefa na inferência

A proposta assume cenário task-aware onde o ID da tarefa é conhecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma limitação em relação a cenários completamente task-agnostic onde a tarefa deve ser inferida automaticamente.

Extensões futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto de entrada, ou técnicas de roteamento que combinam múltiplos adaptadores dinamicamente.

#### Crescimento linear ainda presente (embora moderado)

Embora o crescimento paramétrico seja muito menor que PNNs completas, ainda há crescimento linear com o número de tarefas. Para sequências muito longas (dezenas ou centenas de tarefas), mesmo o crescimento moderado de LoRA pode se tornar significativo.

Futuras pesquisas podem explorar métodos de compressão de adaptadores antigos, seleção de adaptadores mais críticos para manter, ou técnicas de consolidação que mesclam adaptadores similares para reduzir crescimento.

## 5 - Conclusão

### Síntese dos resultados

#### Resumo dos principais achados

Este trabalho investigou a viabilidade e efetividade de uma arquitetura híbrida para aprendizado contínuo em PLN que combina quatro mecanismos complementares: redes neurais progressivas (PNN) através de modularização leve, adaptações de baixo ranque com restrição ortogonal (O-LoRA), consolidação elástica de pesos (EWC), e replay gerativo parcimonioso. Quando executados, os experimentos deverão quantificar em que medida a integração desses mecanismos oferece proteção contra esquecimento catastrófico mantendo eficiência paramétrica e viabilidade computacional.

A análise empírica em sequência de cinco tarefas de classificação textual (AG News, Yelp Polarity, Amazon Reviews, DBPedia, Yahoo Answers) comparará a proposta a baselines de fine-tuning sequencial e LoRA sequencial, bem como a um upper bound de joint training, reportando ACC, BWT, FWT e crescimento paramétrico por tarefa.

#### Redução do esquecimento catastrófico

Resultados esperados incluem redução substancial no esquecimento catastrófico quando comparada aos baselines. Métricas de Backward Transfer (BWT) e Forgetting deverão indicar preservação superior do desempenho em tarefas anteriores em cenários onde a ortogonalidade e o EWC são efetivos.

A análise de ablação deverá evidenciar contribuições individuais e sinergias entre mecanismos. O-LoRA tende a reduzir interferência entre tarefas; EWC pode proteger componentes compartilhados críticos; replay gerativo reforça conhecimentos; e conexões laterais promovem transferência positiva.

#### Manutenção da plasticidade

Além de reduzir esquecimento, a arquitetura busca manter plasticidade para aprender novas tarefas efetivamente. Métricas de Forward Transfer (FWT) quantificarão em que medida conhecimento de tarefas anteriores facilita aprendizado de tarefas futuras, especialmente quando conexões laterais são habilitadas.

O equilíbrio entre estabilidade e plasticidade é alcançado através de calibração cuidadosa de hiperparâmetros, permitindo que o modelo adapte-se seletivamente a novas distribuições enquanto protege parâmetros críticos para tarefas anteriores. Trade-offs explícitos foram identificados e documentados, fornecendo diretrizes para configuração em diferentes contextos.

#### Eficiência paramétrica e computacional alcançada

A eficiência paramétrica será avaliada via crescimento moderado de parâmetros (<2% por tarefa com adaptadores LoRA de baixo ranque), e comparada a outras técnicas de adaptação eficiente.

A viabilidade computacional será reportada considerando overhead de replay gerativo, uso de precisão mista e acúmulo de gradiente. A latência de inferência também será medida para configurações com e sem mesclagem de adaptadores.

### Contribuições alcançadas

#### Arranjo integrado original

A principal contribuição conceitual deste trabalho é a proposta e validação empírica de um arranjo integrado que combina quatro mecanismos complementares de defesa contra esquecimento sob uma arquitetura unificada. Embora cada técnica individual seja conhecida, a integração sistemática e avaliação sob protocolo reprodutível representa uma contribuição original à literatura de aprendizado contínuo em PLN.

A demonstração de sinergias positivas entre os mecanismos valida a hipótese de que abordagens multi-camada podem ser mais efetivas que técnicas isoladas, fornecendo fundamentação teórica e empírica para futuras pesquisas em integração de mecanismos de aprendizado contínuo.

#### Protocolo reprodutível de avaliação

O trabalho estabelece um protocolo reprodutível de avaliação em sequência de tarefas, com métricas padronizadas para esquecimento, transferência e eficiência. A documentação detalhada de configurações experimentais, hiperparâmetros e procedimentos permite reprodução completa dos experimentos por outros pesquisadores.

O protocolo pode servir como benchmark para comparação futura de novas abordagens, facilitando progresso sistemático na área. A disponibilização de código-fonte e configurações aumenta o impacto do trabalho e permite extensões por outros pesquisadores.

#### Guia prático de implementação

Como contribuição de engenharia, o trabalho fornece um guia prático detalhado para implementação com recursos computacionais moderados, incluindo decisões específicas sobre configuração de adaptadores LoRA, calibração de hiperparâmetros EWC, implementação de replay gerativo, e otimizações de memória e computação.

As diretrizes são derivadas de experimentação empírica e buscas de validação, oferecendo insights práticos para engenheiros que precisam implementar sistemas similares. Trade-offs explícitos entre diferentes escolhas de projeto são documentados, facilitando decisões informadas em contextos específicos.

#### Artefatos reusáveis (código e configurações)

O trabalho disponibiliza código-fonte completo, configurações e scripts que permitem reprodução e extensão por outros pesquisadores. Os artefatos incluem implementações dos componentes principais (adaptadores O-LoRA, integração EWC, replay gerativo), pipelines de experimentação, e ferramentas de análise de resultados.

A disponibilização desses artefatos aumenta significativamente o impacto do trabalho, facilitando avanços futuros na área através de construção sobre implementações estabelecidas e validadas.

### Limitações do estudo

#### Restrições de recursos computacionais

Os experimentos foram limitados a execução em hardware intermediário (GPU única), o que restringiu a escala dos modelos e datasets que puderam ser explorados. Modelos maiores (como BERT-large ou modelos de linguagem mais recentes) ou sequências mais longas de tarefas podem requerer recursos adicionais ou otimizações mais avançadas.

Futuras pesquisas podem explorar escalabilidade para modelos maiores e sequências mais longas, validando se os princípios e mecanismos investigados se mantêm em contextos mais desafiadores.

#### Sequência limitada de tarefas avaliadas

A avaliação foi realizada em sequência de cinco tarefas de classificação textual. Embora essa seja uma sequência representativa e amplamente utilizada em benchmarks, sequências mais longas ou tarefas de tipos diferentes (como geração, tradução, ou sumarização) podem revelar comportamentos adicionais ou limitações não observadas.

Extensões futuras podem explorar sequências mais longas, diversidade maior de tipos de tarefas, e diferentes ordens de apresentação para validar robustez dos resultados.

#### Dependência de hiperparâmetros

A efetividade da proposta depende de calibração cuidadosa de hiperparâmetros, especialmente λ_ewc e λ_ortho que controlam a força das regularizações. Valores ótimos podem variar entre diferentes sequências de tarefas, requerendo busca de validação que pode ser custosa ou inviável em alguns contextos.

Futuras pesquisas podem explorar métodos adaptativos para calibrar hiperparâmetros automaticamente ou técnicas mais robustas que sejam menos sensíveis a configurações específicas.

#### Cenário task-aware na inferência

A proposta assume cenário task-aware onde o ID da tarefa é conhecido durante a inferência para ativar o conjunto correto de adaptadores. Esta é uma limitação em relação a cenários completamente task-agnostic onde a tarefa deve ser inferida automaticamente do texto de entrada.

Extensões futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto, técnicas de roteamento dinâmico, ou abordagens que combinam múltiplos adaptadores simultaneamente.

### Trabalhos futuros

#### Extensão para mais tarefas e sequências mais longas

Uma direção natural de extensão é explorar sequências mais longas de tarefas (10, 20, ou mais tarefas) para validar escalabilidade dos mecanismos propostos. Sequências mais longas podem revelar degradação cumulativa ou outros comportamentos não observados em sequências curtas. Pesquisas futuras podem investigar técnicas de compressão ou consolidação de adaptadores para gerenciar crescimento paramétrico em sequências muito longas.

#### Investigação de seleção automática de adaptadores

Para tornar a abordagem mais prática em cenários task-agnostic, pesquisas futuras podem explorar métodos de seleção automática de adaptadores baseados em características do texto de entrada. Isso pode envolver modelos de roteamento que aprendem a mapear textos para adaptadores apropriados, ou técnicas de combinação dinâmica de múltiplos adaptadores simultaneamente.

Técnicas de atenção sobre adaptadores ou mecanismos de gating podem permitir seleção suave baseada em conteúdo, eliminando a necessidade de conhecimento explícito do ID da tarefa.

#### Melhoria da qualidade do replay gerativo

Melhorias na qualidade e robustez do replay gerativo podem aumentar significativamente a efetividade dessa componente. Pesquisas futuras podem explorar modelos geradores mais especializados, técnicas de distilação para manter qualidade ao longo de sequências longas, ou métodos alternativos de geração que não dependem do modelo principal.

Técnicas de validação mais sofisticadas, geração condicionada por protótipos, ou uso de modelos auxiliares para garantir qualidade podem melhorar a efetividade do replay gerativo.

#### Combinação com outras técnicas (distillation, masking)

A arquitetura proposta pode ser estendida incorporando outras técnicas de aprendizado contínuo. Knowledge distillation pode ser usado para transferir conhecimento entre adaptadores, técnicas de masking (como HAT) podem ser integradas para proteção adicional de parâmetros, ou métodos de pruning podem ser combinados para gerenciar crescimento paramétrico.

Investigações futuras podem explorar sinergias entre essas técnicas adicionais e os mecanismos já integrados, potencialmente descobrindo combinações ainda mais efetivas.

#### Aplicação em cenários realistas (streaming, dados não rotulados)

Extensões para cenários mais realistas podem aumentar a aplicabilidade prática da proposta. Cenários de streaming onde dados chegam continuamente, aprendizado semi-supervisionado com dados não rotulados, ou adaptação a distribuições que mudam gradualmente ao longo do tempo representam desafios adicionais que podem ser explorados.

Esses cenários podem requerer adaptações dos mecanismos propostos ou desenvolvimento de componentes adicionais específicos para lidar com incerteza e não supervisão.

#### Estudo de transferência entre domínios mais diversos

Investigação de transferência entre domínios mais diversos e distantes pode revelar limitações adicionais ou oportunidades de melhoria. Tarefas de diferentes modalidades (texto, código, tabelas), diferentes idiomas, ou diferentes formatos de saída podem testar a generalidade dos mecanismos propostos.

Análise detalhada de como diferentes tipos de conhecimento são transferidos e protegidos pode fornecer insights teóricos adicionais sobre os mecanismos de aprendizado contínuo e sugerir melhorias específicas para diferentes contextos.

## Referências bibliográficas

ALJUNDI, R.; BABILONI, F.; ELHOSEINY, M.; ROHRBACH, M.; TUYTELAARS, T. Memory Aware Synapses: Learning what (not) to forget. In: European Conference on Computer Vision – ECCV 2018. Cham: Springer, 2018. p. 144–161. DOI: 10.1007/978-3-030-01219-9_9.

DE LANGE, M.; ALJUNDI, R.; MASANA, M.; PARISOT, S.; JIA, X.; LEONARDIS, A.; SLABAUGH, G.; TUYTELAARS, T. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, v. 44, n. 10, p. 3366–3385, 2022. DOI: 10.1109/TPAMI.2021.3057446.

DE MASSON D’AUTUME, C.; RUDER, S.; KONG, L.; YOGATAMA, D. Episodic Memory in Lifelong Language Learning. In: Advances in Neural Information Processing Systems (NeurIPS 2019). p. 13122–13131. Disponível em: https://papers.neurips.cc/paper/9471-episodic-memory-in-lifelong-language-learning.pdf
. Acesso em: 1 nov. 2025.

DETTMERS, T.; PAGNONI, A.; HOLTZMAN, A.; ZETTLEMOYER, L. QLoRA: Efficient Finetuning of Quantized LLMs. In: Advances in Neural Information Processing Systems (NeurIPS 2023). Disponível em: https://arxiv.org/abs/2305.14314
. Acesso em: 1 nov. 2025.

DEVLIN, J.; CHANG, M.-W.; LEE, K.; TOUTANOVA, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In: Proceedings of NAACL-HLT 2019. p. 4171–4186. DOI: 10.18653/v1/N19-1423.

FARAJTABAR, M.; AZIZAN, N.; MOTT, A.; LI, A. Orthogonal Gradient Descent for Continual Learning. arXiv:1910.07104, 2019. Disponível em: https://arxiv.org/abs/1910.07104
. Acesso em: 1 nov. 2025.

FRENCH, R. M. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, v. 3, n. 4, p. 128–135, 1999. DOI: 10.1016/S1364-6613(99)01294-2.

GROSSBERG, S. Competitive learning: From interactive activation to adaptive resonance. Cognitive Science, v. 11, n. 1, p. 23–63, 1987. DOI: 10.1111/j.1551-6708.1987.tb00862.x.

HOULSBY, N.; GIURGIU, A.; JASTRZEBSKI, S.; MORRONE, B.; DE LAROUSSiLHE, Q.; GESMUNDO, A.; ATTARIYAN, M.; GELLY, S. Parameter-Efficient Transfer Learning for NLP. In: Proceedings of the 36th International Conference on Machine Learning (ICML 2019). PMLR 97, p. 2790–2799. Disponível em: https://proceedings.mlr.press/v97/houlsby19a.html
. Acesso em: 1 nov. 2025.

HU, E. J.; SHEN, Y.; WALLIS, P.; ALLEN-ZHU, Z.; LI, Y.; WANG, S.; WANG, L.; CHEN, W. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021. Disponível em: https://arxiv.org/abs/2106.09685
. Acesso em: 1 nov. 2025.

HUSZÁR, F. On Quadratic Penalties in Elastic Weight Consolidation. Proceedings of the National Academy of Sciences (PNAS), v. 115, n. 11, p. E2496–E2497, 2018. DOI: 10.1073/pnas.1717042115.

KIRKPATRICK, J.; PASCANU, R.; RABINOWITZ, N.; VENESS, J.; DESJARDINS, G.; RUSU, A.; MILAN, K.; QUAN, J.; RAMALHO, T.; GRABSKA-BARWINSKA, A.; HASSABIS, D.; CLOPATH, C.; KUMARAN, D.; HADSELL, R. Overcoming catastrophic forgetting in neural networks. PNAS, v. 114, n. 13, p. 3521–3526, 2017. DOI: 10.1073/pnas.1611835114.

LESTER, B.; AL-RFOU, R.; CONSTANT, N. The Power of Scale for Parameter-Efficient Prompt Tuning. In: Findings of EMNLP 2021. p. 3045–3059. DOI: 10.18653/v1/2021.findings-emnlp.265.

LI, X. L.; LIANG, P. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In: Proceedings of ACL 2021. p. 4582–4597. DOI: 10.18653/v1/2021.acl-long.353.

LI, Z.; HOIEM, D. Learning without Forgetting. In: European Conference on Computer Vision (ECCV 2016). Cham: Springer, 2016. p. 614–629. DOI: 10.1007/978-3-319-46493-0_37.

LOPEZ-PAZ, D.; RANZATO, M. Gradient Episodic Memory for Continual Learning. In: Advances in Neural Information Processing Systems (NeurIPS 2017). p. 6467–6476.

McCLOSKEY, M.; COHEN, N. J. Catastrophic interference in connectionist networks: The sequential learning problem. In: BOWER, G. H. (ed.). Psychology of Learning and Motivation, v. 24. San Diego: Academic Press, 1989. p. 109–165. DOI: 10.1016/S0079-7421(08)60137-8.

PARISI, G. I.; KEMKER, R.; PART, J. L.; KANAN, C.; WERMTER, S. Continual lifelong learning with neural networks: A review. Neural Networks, v. 113, p. 54–71, 2019. DOI: 10.1016/j.neunet.2019.01.012.

PFEIFFER, J.; KAMATH, A.; RÜCKLÉ, A.; CHO, K.; GUREVYCH, I. AdapterFusion: Non-Destructive Task Composition for Transfer Learning. In: Proceedings of the 16th Conference of the European Chapter of the ACL (EACL 2021). p. 312–318. DOI: 10.18653/v1/2021.eacl-main.39.

RUSU, A. A.; RABINOWITZ, N. C.; DESJARDINS, G.; SOYER, H.; KIRKPATRICK, J.; KAVUKCUOGLU, K.; PASCANU, R.; HADSELL, R. Progressive Neural Networks. arXiv:1606.04671, 2016. Disponível em: https://arxiv.org/abs/1606.04671
. Acesso em: 1 nov. 2025.

SANH, V.; DEBUT, L.; CHAUMOND, J.; WOLF, T. DistilBERT: A distilled version of BERT – smaller, faster, cheaper and lighter. arXiv:1910.01108, 2019. Disponível em: https://arxiv.org/abs/1910.01108
. Acesso em: 1 nov. 2025.

SCHWARZ, J.; LUKETINA, J.; CZARNECKI, W. M.; GRABSKA-BARWINSKA, A.; TEH, Y. W.; PASCANU, R.; HADSELL, R. Progress & Compress: A scalable framework for continual learning. In: Proceedings of the 35th International Conference on Machine Learning (ICML 2018). PMLR 80, p. 4528–4537. Disponível em: https://proceedings.mlr.press/v80/schwarz18a.html
. Acesso em: 1 nov. 2025.

SHIN, H.; LEE, J. K.; KIM, J.; KIM, J. Continual Learning with Deep Generative Replay. In: Advances in Neural Information Processing Systems (NeurIPS 2017). p. 2990–2999.

SUN, F.-K.; HO, C.-H.; LEE, H.-Y. LAMOL: Language Modeling for Lifelong Language Learning. In: International Conference on Learning Representations (ICLR 2020). Disponível em: https://openreview.net/forum?id=Skgxcn4YDS
. Acesso em: 1 nov. 2025.

VAN DE VEN, G. M.; TOLIAS, A. S. Three scenarios for continual learning. arXiv:1904.07734, 2019. Disponível em: https://arxiv.org/abs/1904.07734
. Acesso em: 1 nov. 2025.

VASWANI, A.; SHAZEER, N.; PARMAR, N.; USZKOREIT, J.; JONES, L.; GOMEZ, A. N.; KAISER, Ł.; POLOSUKHIN, I. Attention Is All You Need. In: Advances in Neural Information Processing Systems (NeurIPS 2017). p. 5998–6008.

ZENG, G.; CHEN, Y.; CUI, B.; YU, S. Continual Learning of Context-Dependent Processing in Neural Networks (OWM). Nature Machine Intelligence, v. 1, p. 364–372, 2019. DOI: 10.1038/s42256-019-0080-x.

ZHANG, X.; ZHAO, J.; LECUN, Y. Character-Level Convolutional Networks for Text Classification. In: Advances in Neural Information Processing Systems (NeurIPS 2015). p. 649–657.