# Capítulo 2 - Referencial Teórico

## Fundamentação teórica sobre aprendizado contínuo em PLN

### Definição e desafios do aprendizado contínuo

O aprendizado contínuo, também referido como continual learning ou lifelong learning, consiste na capacidade de um sistema de inteligência artificial aprender novas tarefas de forma sequencial, acumulando conhecimento ao longo do tempo sem esquecer o que foi aprendido anteriormente. Diferentemente do fine-tuning tradicional, onde um conjunto de dados é fixo e o modelo é ajustado apenas uma vez, no aprendizado contínuo o modelo enfrenta desafios significativos ao adaptar-se a distribuições que evoluem sem reiniciar o treinamento do zero para cada novo contexto.

O principal desafio é o esquecimento catastrófico — a tendência de redes neurais esquecerem abruptamente tarefas antigas ao aprender tarefas novas. Esse fenômeno foi documentado desde os primórdios das redes neurais e permanece um obstáculo central em sistemas adaptativos. Uma arquitetura de aprendizado contínuo deve equilibrar plasticidade (para incorporar novos conhecimentos) e estabilidade (para reter conhecimentos prévios), evitando que gradientes das tarefas recentes sobrescrevam parâmetros importantes das tarefas passadas.

### Dilema estabilidade-plasticidade

O dilema estabilidade-plasticidade, formalizado por Grossberg (1987), é fundamental para entender os desafios do aprendizado contínuo. Plasticidade refere-se à capacidade do sistema de modificar seus parâmetros para incorporar novos conhecimentos, enquanto estabilidade refere-se à capacidade de preservar conhecimentos previamente adquiridos. Em redes neurais tradicionais, essas capacidades estão em conflito: aumentar a plasticidade facilita o aprendizado de novas tarefas mas aumenta a vulnerabilidade ao esquecimento, enquanto aumentar a estabilidade protege contra esquecimento mas pode impedir adaptação efetiva a novas distribuições.

O equilíbrio ótimo depende de múltiplos fatores, incluindo a similaridade entre tarefas, a capacidade do modelo, e o regime de treinamento. Em PLN, onde tarefas podem variar desde classificação de sentimento até tradução ou sumarização, encontrar esse equilíbrio é particularmente desafiador devido à diversidade de objetivos e distribuições.

### Esquecimento catastrófico em redes neurais

O esquecimento catastrófico ocorre quando o treinamento em uma nova tarefa altera parâmetros que eram críticos para tarefas anteriores. Em redes neurais profundas, onde milhões de parâmetros são compartilhados entre diferentes camadas e funções, essa interferência pode ocorrer em múltiplos níveis. O problema é especialmente pronunciado quando tarefas novas e antigas requerem atualizações conflitantes dos mesmos parâmetros.

A natureza do esquecimento catastrófico em Transformers é particularmente complexa devido à arquitetura de atenção e às múltiplas camadas de representação. Parâmetros compartilhados em embeddings, camadas de atenção e redes feed-forward podem ser afetados de forma diferente dependendo de como as tarefas utilizam essas representações. Compreender e mitigar essas interferências requer abordagens que identifiquem e protejam parâmetros críticos ou que isolam atualizações para diferentes tarefas.

### Contexto específico do PLN (diversidade de tarefas, domínios, ambiguidade linguística)

Em PLN, o aprendizado contínuo apresenta desafios adicionais e motivação especial. Enquanto em visão computacional ou robótica as tarefas podem ser bem delimitadas (por exemplo, diferentes conjuntos de classes de imagens), em PLN há grande diversidade de tarefas: classificação de intenção, análise de sentimento, perguntas e respostas, tradução, sumarização, entre outras. Essas tarefas frequentemente envolvem dados textuais de domínios distintos e objetivos variáveis.

A linguagem natural é inerentemente ambígua e dependente do contexto, com vocabulário em constante evolução. Essa natureza dinâmica reforça a motivação para sistemas de PLN que aprendam continuamente: aplicações reais frequentemente precisam incorporar novas gírias, tópicos emergentes, mudanças de estilo ou domínio linguístico sem perder a habilidade em tarefas anteriores. Um assistente virtual pode precisar aprender progressivamente novos tipos de consultas dos usuários ao longo do tempo, sem esquecer como responder às solicitações antigas.

## Redes Neurais Progressivas (PNN)

### Conceito e arquitetura básica

As Redes Neurais Progressivas (Progressive Neural Networks - PNN), proposta por Rusu et al. (2016), constituem uma abordagem baseada em arquitetura para aprendizado contínuo. A ideia central é expandir a capacidade da rede incrementalmente a cada nova tarefa, adicionando um novo conjunto de neurônios (uma nova "coluna" ou módulo de rede) específico para cada tarefa aprendida.

Quando a tarefa T_k é iniciada, cria-se uma nova coluna de parâmetros inicializada (geralmente a partir de uma versão pré-treinada) para aprender T_k. As colunas das tarefas anteriores são congeladas — seus pesos não são alterados durante o treinamento de T_k — e são estabelecidas conexões laterais da saída (ou camadas intermediárias) de cada coluna anterior para a nova coluna. Essas conexões laterais permitem que a nova coluna reutilize e transfira conhecimento das características previamente aprendidas nas tarefas anteriores, promovendo transferência de aprendizado para frente.

### Isolamento de parâmetros por tarefa

A característica fundamental das PNNs é o isolamento completo entre tarefas em termos de parâmetros. Cada coluna atende a uma tarefa específica, garantindo que uma tarefa nova não degrade o desempenho das anteriores através de interferência destrutiva direta. Como os parâmetros das colunas antigas não são modificados, a PNN elimina o esquecimento catastrófico por construção — o aprendizado de T_k não interfere diretamente nos pesos que foram sintonizados para tarefas anteriores.

Esse isolamento é conceitualmente simples e teoricamente garantido, mas vem ao custo de crescimento linear do número de parâmetros em função do número de tarefas. Para cada nova tarefa adiciona-se uma coluna completa de rede, o que pode se tornar impraticável quando as tarefas são numerosas ou quando a arquitetura base é muito grande.

### Conexões laterais e transferência de conhecimento

As conexões laterais são um mecanismo explícito de transferência que permite que a nova coluna consuma representações das colunas anteriores sem atualizá-las. Essas conexões podem ser implementadas de várias formas: concatenando saídas de camadas correspondentes, somando representações ponderadas, ou através de mecanismos de atenção que aprendem a combinar features de diferentes colunas.

O benefício das conexões laterais é duplo: primeiro, permitem que a nova tarefa se beneficie de conhecimento prévio aprendido, potencialmente acelerando o treinamento e melhorando o desempenho inicial. Segundo, fornecem um mecanismo de forward transfer, onde conhecimento de tarefas anteriores facilita o aprendizado de tarefas futuras. Em termos práticos, isso significa que uma nova tarefa pode partir de representações já úteis, em vez de aprender tudo do zero.

### Vantagens e limitações (crescimento paramétrico linear)

As principais vantagens das PNNs são o isolamento completo entre tarefas (garantindo imunidade ao esquecimento), a simplicidade conceitual, e a capacidade de transferência positiva através de conexões laterais. A abordagem reflete, em certo grau, a modularidade do aprendizado biológico, onde novos conhecimentos podem recrutar novas estruturas sem apagar as antigas.

A principal limitação é o crescimento linear pesado de parâmetros: aplicar PNN ingênua a 10 tarefas com um modelo do porte do BERT significaria ter 10 modelos BERT em memória ao final. Além disso, PNNs normalmente assumem que os limites entre tarefas são conhecidos e bem definidos durante o treinamento, e geralmente requerem conhecimento da tarefa na inferência para rotear exemplos à coluna apropriada. Essas limitações motivam alternativas que capturam os benefícios das PNNs (isolamento e transferência) de forma mais parametricamente eficiente.

## Adaptações de Baixo Ranque com Restrições Ortogonais (LoRA e O-LoRA)

### LoRA: adaptação eficiente de parâmetros

O LoRA (Low-Rank Adaptation), proposto por Hu et al. (2021), oferece uma forma de adaptar modelos de linguagem de grande porte de maneira leve e modular. A técnica funciona congelando todos os pesos originais do modelo pré-treinado e injetando pequenos módulos treináveis de baixo ranque em cada camada relevante da arquitetura Transformer.

Para cada matriz de pesos W em camadas selecionadas (por exemplo, nas projeções de atenção ou na rede feed-forward), LoRA introduz duas matrizes menores A e B de dimensões de posto r (tipicamente r bem menor que o tamanho original da camada) de forma que a atualização da camada seja W + ΔW, onde ΔW = AB representa um ajuste de baixo ranque aprendido para a nova tarefa. Em vez de ajustar todos os pesos do modelo, apenas os parâmetros nesses pequenos módulos são aprendidos.

### Decomposição de baixo ranque (matrizes A e B)

A decomposição de baixo ranque explora a hipótese de que as atualizações de pesos necessárias para uma nova tarefa podem ser representadas eficientemente em um subespaço de dimensão muito menor que o espaço original de parâmetros. Se W tem dimensões d×d, LoRA introduz duas matrizes A (d×r) e B (r×d), onde r << d. O produto AB resulta em uma matriz de atualização de posto no máximo r, permitindo representar mudanças complexas com muito menos parâmetros.

A inicialização das matrizes A e B é importante: tipicamente A é inicializada aleatoriamente e B é inicializada com zeros, garantindo que ΔW = 0 inicialmente e o modelo começa com o comportamento do modelo base. Durante o treinamento, apenas A e B são atualizados, mantendo W congelado.

### Eficiência paramétrica e computacional

Uma vantagem crucial do LoRA é a redução dramática de parâmetros ajustáveis. No caso extremo do GPT-3 (175 bilhões de parâmetros), estudos mostraram que é possível ajustar cerca de 18 milhões de parâmetros (via LoRA) para especializar o modelo em uma nova tarefa, mantendo desempenho similar ao fine-tuning completo — isso equivale a apenas ~0,01% dos parâmetros originais. Em modelos menores como BERT ou GPT-2, o overhead típico do LoRA por tarefa costuma ficar na faixa de 0,1% a 3% do total de parâmetros.

Além da eficiência paramétrica, o LoRA mantém eficiência computacional: como o modelo base permanece congelado e compartilhado entre todas tarefas, não há aumento de latência na inferência quando os deltas são mesclados de volta nos pesos originais. Isso torna viável carregar múltiplos adapters LoRA — um por tarefa — em memória sem explodir o uso de recursos.

### O-LoRA: imposição de ortogonalidade entre adaptadores

O LoRA puro não resolve por si só o esquecimento catastrófico quando usado sequencialmente. Se o mesmo conjunto de adaptadores for reutilizado para múltiplas tarefas em sequência, a tarefa nova pode sobrescrever as representações adquiridas pelos adaptadores na tarefa anterior. Para enfrentar essa limitação, o O-LoRA (Orthogonal LoRA), proposto por Wang et al. (2023), impõe restrições ortogonais entre os subespaços de adaptação de cada tarefa.

Ao treinar os adaptadores de uma nova tarefa, adiciona-se um termo de regularização (ou projeta-se explicitamente) para que os novos deltas de baixo ranque fiquem ortogonais aos espaços gerados pelos deltas das tarefas anteriores. Assim, cada tarefa T_i aprende sua atualização ΔW_i = A_i B_i em um subespaço linear distinto, minimizando projeções em direções usadas por ΔW_j de tarefas j < i.

### Isolamento em subespaços distintos

O isolamento em subespaços ortogonais atua como um análogo leve de uma PNN: em vez de dedicar uma coluna inteira de parâmetros para cada tarefa, dedica-se apenas um subespaço de adaptações de baixo ranque, mas assegurando que haja pouca sobreposição entre as direções de atualização de tarefas diferentes. Estudos reportam que métodos como O-LoRA efetivamente reduzem a interferência entre tarefas e, assim, o esquecimento, quando comparados ao uso ingênuo de LoRA sequencial.

O custo adicional do O-LoRA é marginal: os adaptadores ortogonais têm o mesmo número de parâmetros do LoRA convencional (apenas o procedimento de treinamento muda), mantendo a eficiência. No entanto, garantir ortogonalidade perfeita entre subespaços de diversas tarefas pode se tornar difícil conforme o número de tarefas cresce, especialmente se o ranque r for limitado.

### Vantagens e limitações

LoRA e suas variantes ortogonais oferecem um compromisso atraente entre isolamento de tarefas e eficiência. Cada tarefa é especializada por meio de um conjunto pequeno de parâmetros adicionais, resultando em crescimento linear modesto de memória com o número de tarefas (ordens de grandeza menor que adicionar colunas completas como na PNN). Com O-LoRA, obtém-se também isolamento efetivo entre tarefas, aproximando-se do ideal de "uma coluna por tarefa", porém de forma muito mais leve.

As limitações incluem o crescimento linear ainda presente (embora baixo), a necessidade de conhecimento da tarefa na inferência para ativar o conjunto correto de adaptadores, e possíveis dificuldades em manter ortogonalidade perfeita em sequências muito longas. Ainda assim, LoRA e O-LoRA representam avanços importantes para viabilizar aprendizado contínuo em modelos de PLN grandes, fornecendo eficiência paramétrica com isolamento suficiente para mitigar grande parte do esquecimento.

## Elastic Weight Consolidation (EWC)

### Regularização baseada em importância de pesos

Elastic Weight Consolidation (EWC), proposto por Kirkpatrick et al. (2017), é um método de regularização que busca preservar o conhecimento prévio dentro dos mesmos parâmetros ao longo de tarefas sequenciais. A premissa é adicionar um termo de penalização na função de perda durante o treinamento de novas tarefas, desencorajando grandes mudanças nos pesos que foram identificados como importantes para tarefas antigas.

Antes de aprender a tarefa T_k, o algoritmo EWC calcula, para cada peso θ_j do modelo, um valor de importância que quantifica o quanto θ_j contribuiu para o desempenho nas tarefas anteriores. Essa importância é tipicamente estimada através da matriz de informação de Fisher, avaliada nos dados das tarefas passadas. Intuitivamente, se um peso influenciava fortemente as predições corretas nas tarefas antigas, o EWC irá puni-lo caso ele se desvie muito do seu valor original enquanto aprende a nova tarefa.

### Matriz de informação de Fisher

A matriz de informação de Fisher F fornece uma medida da importância de cada parâmetro para o desempenho do modelo. Os elementos diagonais F_j da matriz representam a curvatura da função de perda em relação ao parâmetro θ_j: valores altos indicam que pequenas mudanças em θ_j têm grande impacto no desempenho, enquanto valores baixos indicam que o parâmetro é menos crítico.

A estimativa da matriz de Fisher requer avaliar o gradiente da função de perda em relação aos parâmetros sobre os dados da tarefa anterior. Em implementações práticas, apenas os elementos diagonais são computados (aproximação diagonal), reduzindo significativamente o custo computacional. A matriz de Fisher é estimada após o treinamento em cada tarefa e armazenada para uso nas tarefas subsequentes.

### Termo de penalização na função de perda

Matematicamente, a perda total do modelo ao aprender uma nova tarefa incorpora um termo do tipo:

L(θ) = L_novo(θ) + λ Σ_j (1/2) F_j (θ_j - θ_j*)^2

onde L_novo é a perda normal nos dados da nova tarefa, θ_j* é o valor do peso j após treinamento na tarefa anterior (mantido como referência), F_j é o elemento diagonal da matriz de Fisher, e λ é um hiperparâmetro que controla a força da penalização. Esse termo adicional atua como uma "mola" ancorando cada peso em torno do valor antigo com rigidez proporcional à importância F_j.

### Consolidação de conhecimento crítico

Pesos críticos ficam quase "congelados" (alta penalização se mudarem), enquanto pesos pouco relevantes podem se ajustar livremente à nova tarefa. Dessa forma, o EWC tenta obter um compromisso ótimo entre não esquecer o passado e ainda aprender o novo, encontrando uma região no espaço de parâmetros que minimize a perda da tarefa atual sem sair da região de bom desempenho das tarefas antigas.

A consolidação é particularmente efetiva quando aplicada a componentes compartilhados que mantêm conhecimento linguístico fundamental, como embeddings ou camadas iniciais do modelo. Essas camadas frequentemente codificam conhecimento geral que beneficia múltiplas tarefas, tornando-as candidatas ideais para proteção via EWC.

### Vantagens e limitações (trade-off plasticidade/estabilidade)

O EWC se destaca por sua simplicidade e generalidade: não requer modificar a arquitetura da rede nem adicionar parâmetros extras, apenas a função de custo muda. Não há crescimento de memória conforme novas tarefas são aprendidas (diferente de PNN/LoRA) e não é necessário armazenar dados antigos (diferente de métodos de replay). Em implementações offline, basta armazenar as estimativas de F_j e os valores antigos θ_j após cada tarefa.

Apesar de mitigar o esquecimento, o EWC raramente o elimina por completo. Em cenários de longas sequências de tarefas, as restrições impostas podem se acumular a ponto de prejudicar a plasticidade do modelo para novas tarefas. O efeito depende criticamente do hiperparâmetro λ: se muito alto, o modelo praticamente não aprende a nova tarefa; se muito baixo, o modelo esquece facilmente as antigas. Encontrar um equilíbrio pode exigir validação cuidadosa para cada situação.

## Replay Gerativo

### Conceito de rehearsal e pseudo-rehearsal

O replay gerativo é uma estratégia inspirada no conceito de rehearsal em psicologia, onde memórias antigas são periodicamente revisitadas para consolidação. Em aprendizado de máquina, o replay clássico consiste em reapresentar dados de tarefas antigas durante o treinamento de uma nova tarefa, para que o modelo "não se esqueça" delas. Contudo, armazenar todos os dados antigos em buffer pode ser impraticável em termos de memória e, em certas aplicações, proibitivo por questões de privacidade.

A solução oferecida pelo replay gerativo é substituir os dados reais por dados sintéticos gerados por um modelo. Em vez de guardar exemplos de tarefas passadas, treina-se (ou utiliza-se) um modelo gerador que produz pseudo-exemplos das tarefas anteriores para serem intercalados no treinamento corrente. Essa abordagem é frequentemente chamada de pseudo-rehearsal ou deep generative replay.

### LAMOL: Language Modeling for Lifelong Learning

Em PLN, o LAMOL (Language Modeling for Lifelong Learning), proposto por Sun et al. (2020), exemplifica bem o replay gerativo. Nele, um único modelo de linguagem é treinado para duas funções simultâneas: (i) resolver a tarefa atual e (ii) gerar dados de tarefas anteriores sob forma de texto. O processo funciona assim: antes (ou durante) de treinar na tarefa T_k, o modelo gera um conjunto de exemplos fictícios das tarefas T_1, ..., T_{k-1} que já aprendeu.

Esses exemplos gerados — às vezes chamados de exemplos "nostálgicos" — são então misturados com os dados reais da nova tarefa durante o treino de T_k. O gradiente que atualiza o modelo é influenciado não só pela nova tarefa, mas também por recriações das antigas, reforçando as conexões relevantes para o desempenho passado.

### Geração de exemplos sintéticos sem armazenamento de dados brutos

A geração de exemplos sintéticos requer um modelo capaz de produzir textos representativos das tarefas anteriores. Em abordagens como LAMOL, o próprio modelo principal é usado para gerar exemplos, utilizando tokens especiais para condicionar qual tarefa gerar. Alternativamente, pode-se usar um modelo gerador separado que foi treinado especificamente para gerar exemplos das tarefas anteriores.

Os exemplos gerados devem ser representativos das distribuições originais e balanceados entre classes para evitar viés no treinamento. A qualidade das gerações é crítica: se o modelo gerador não for capaz de produzir amostras fiéis das tarefas antigas, o modelo principal pode esquecer informações importantes ou até aprender lembranças incorretas.

### Ciclos de reforço de tarefas anteriores

O replay gerativo intercala exemplos sintéticos de tarefas anteriores durante o treinamento da tarefa atual. A frequência e proporção de exemplos sintéticos misturados com dados reais são hiperparâmetros importantes. Tipicamente, uma fração do batch (por exemplo, 10-30%) é composta por exemplos sintéticos de tarefas anteriores, permitindo que o modelo "revisite" periodicamente conhecimentos passados enquanto aprende novos.

Conceitualmente, é como se o modelo "revisasse" periodicamente as tarefas anteriores enquanto aprende coisas novas, análogo ao ser humano que revisita memórias antigas para não esquecê-las. Esse reforço periódico ajuda a manter as conexões sinápticas relevantes para tarefas anteriores ativas durante o aprendizado de novas tarefas.

### Vantagens e limitações (qualidade das gerações, custo computacional)

O replay gerativo evita a necessidade de armazenar dados originais, contornando problemas de privacidade e economizando espaço. Um gerador eficaz pode potencialmente produzir uma diversidade maior de exemplos do que um buffer limitado, enriquecendo o treinamento e levando a melhor generalização. Métodos de replay gerativo têm demonstrado sucesso em recuperar desempenho em tarefas antigas quase no nível de métodos com buffer real, quando conseguem gerar amostras fiéis.

As limitações incluem a dependência crítica da qualidade e do balanceamento dos exemplos gerados. Há um risco conhecido de degradação cumulativa: se o modelo principal começa a esquecer uma tarefa, suas gerações daquela tarefa também pioram, criando um ciclo vicioso ("efeito catastrófico circular"). Outra limitação é o custo computacional: gerar dados não é gratuito — frequentemente, para cada minibatch de dados reais, o modelo precisa gerar um número de exemplos antigos, aumentando proporcionalmente o tempo de treinamento.

## Métricas de avaliação em aprendizado contínuo

### Average Accuracy (ACC)

A Average Accuracy é uma métrica que sumariza o desempenho geral do modelo após aprender a sequência de tarefas. Suponha que o modelo tenha sido treinado sequencialmente em N tarefas. Após a finalização da última tarefa (T_N), avalia-se o modelo em todas as N tarefas e calcula-se a média das acurácias obtidas nessas tarefas. Denotando R_{i,j} a acurácia do modelo na tarefa j após ter treinado até a tarefa i, a Average Accuracy final (ACC) pode ser definida como A = (1/N) Σ_{j=1}^N R_{N,j}.

Uma ACC elevada indica que, em média, o modelo conseguiu reter bom desempenho em todas as tarefas ao final. Valores baixos indicam esquecimento significativo ou baixa performance geral. Em alguns trabalhos, considera-se também a acurácia média ao longo do tempo (não só no final), para avaliar a trajetória de aprendizado.

### Forgetting / Esquecimento

A métrica de esquecimento foca explicitamente na perda de desempenho que o modelo sofreu em tarefas antigas após aprender novas tarefas. Uma forma comum de defini-la é comparar, para cada tarefa j, a melhor acurácia que o modelo obteve em j em algum ponto do treinamento com a acurácia em j ao final do treinamento de todas as tarefas. Se A_j^max foi a acurácia da tarefa j logo que o modelo terminou de aprender T_j e A_j^final é a acurácia em j após a tarefa final T_N, podemos definir a taxa de esquecimento em j como F_j = A_j^max - A_j^final.

A métrica permite quantificar rigorosamente o impacto destrutivo do aprendizado sequencial e é essencial para validar técnicas cujo objetivo é minimizar esse efeito. Valores positivos indicam esquecimento catastrófico (pior quanto maior), e valores negativos (teoricamente possíveis) indicariam que o modelo melhorou em tarefas antigas mesmo após aprender novas.

### Backward Transfer (BWT)

O Backward Transfer mede formalmente o efeito do aprendizado de novas tarefas sobre o desempenho em tarefas anteriores. Lopez-Paz e Ranzato (2017) definem BWT como:

BWT = (1/(N-1)) Σ_{i=1}^{N-1} (R_{N,i} - R_{i,i})

onde R_{i,i} é a acurácia obtida imediatamente após treinar a tarefa i (pico) e R_{N,i} é a acurácia na tarefa i após treinar todas as N tarefas. Se BWT for negativo, indica esquecimento em média (transferência "para trás" negativa); se for positivo, indica que o modelo melhorou em tarefas antigas depois de aprender novas (transferência para trás positiva). Técnicas bem-sucedidas buscam tornar BWT o mais próximo de 0 possível (idealmente positivo).

### Forward Transfer (FWT)

O Forward Transfer complementa o BWT medindo a influência que o conhecimento das tarefas anteriores exerce sobre o aprendizado de tarefas futuras. Indica se o modelo aprendeu a aprender: se tarefas passadas fornecem representações ou parâmetros que facilitam a obtenção de melhor desempenho em tarefas novas, mesmo antes de treiná-las extensivamente.

Formalmente, FWT é definido como a média de R_{i,j} para i<j (acurácia em tarefas futuras j antes de treiná-las), normalizada de forma que a contribuição de um classificador aleatório seja subtraída. Um FWT positivo significa que o modelo, por ter aprendido tarefas anteriores, já inicia melhor do que um modelo não treinado quando encontra uma tarefa nova, indicando transferência de conhecimento útil para frente.

### Custo paramétrico e computacional

Além das métricas de desempenho, avalia-se também o custo em recursos de cada estratégia. Uma métrica comum é acompanhar o número de parâmetros adicionais que o modelo adquire por tarefa (model size growth). Idealmente, deseja-se que a eficiência de memória seja alta — o modelo não deve crescer muito conforme N aumenta. Por exemplo, PNNs teriam um crescimento linear pesado (100% por tarefa), enquanto LoRA pode crescer <1% por tarefa; EWC não cresce nada em parâmetros do modelo (0%), mas requer armazenar algumas estatísticas por peso.

Avalia-se também a eficiência computacional — frequentemente medida em tempo de treinamento (ou FLOPs) adicional introduzido pelas técnicas de CL. Métodos com replay (real ou gerativo) praticamente dobram o número de amostras processadas por iteração, enquanto regularizações como EWC têm overhead mínimo no tempo de treino. Para uma avaliação abrangente, não basta verificar se o modelo mantém alta acurácia em todas as tarefas; é preciso também verificar quanto custo de memória e computação foi pago para alcançar aquele resultado.

## Trabalhos correlatos

### Estudos que combinam pares de técnicas (LoRA+replay, EWC+replay, O-LoRA)

Existem na literatura estudos que investigam combinações de pares das técnicas propostas neste trabalho. Por exemplo, trabalhos que combinam LoRA com replay têm demonstrado que a eficiência paramétrica dos adaptadores pode ser complementada pelo reforço de dados sintéticos. Estudos que combinam EWC com replay mostram que a regularização pode ser efetiva mesmo quando dados sintéticos são utilizados, sugerindo que os mecanismos atuam em diferentes dimensões do problema.

Trabalhos específicos sobre O-LoRA têm demonstrado que a imposição de ortogonalidade entre adaptadores reduz efetivamente a interferência entre tarefas, oferecendo um isolamento estrutural eficiente. No entanto, esses estudos frequentemente focam em combinações de pares, não explorando sistematicamente a integração de múltiplos mecanismos simultaneamente.

### Abordagens alternativas (HAT, PackNet, L2P, DualPrompt)

Existem também abordagens alternativas que não são diretamente combinadas neste trabalho, mas oferecem insights relevantes. HAT (Hard Attention to the Task) usa máscaras aprendidas para proteger parâmetros importantes, enquanto PackNet usa pruning para alocar subnetworks por tarefa. L2P e DualPrompt são métodos baseados em prompt tuning que armazenam prompts por tarefa em vez de adaptar pesos internos.

Essas abordagens alternativas demonstram a diversidade de estratégias disponíveis para aprendizado contínuo e destacam diferentes trade-offs entre isolamento, eficiência e flexibilidade. Embora não sejam diretamente incorporadas nesta proposta, oferecem perspectivas valiosas sobre possíveis extensões futuras.

### Lacunas identificadas na literatura

A principal lacuna identificada é a falta de uma avaliação sistemática e integrada de um arranjo que combine PNN, O-LoRA, EWC e replay gerativo sob um mesmo protocolo experimental reprodutível. Enquanto há evidências sobre eficácia de combinações de pares, não há estudos que investiguem como múltiplos mecanismos podem se complementar e potencialmente oferecer sinergias superiores à soma das partes individuais.

Além disso, há uma falta de protocolos padronizados que permitam comparações justas entre diferentes abordagens, dificultando a identificação de quais combinações são mais eficazes para diferentes contextos. Esta lacuna motiva a proposta deste trabalho, que busca fornecer uma avaliação abrangente e reprodutível de uma arquitetura híbrida integrada.