# Persona × Comprimento × Dropout

> **Em uma frase:** damos a um modelo de linguagem uma "ficha de personagem" no começo de uma conversa muito
> longa e medimos se ele **continua lembrando quem é** quando a conversa fica maior do que ele viu no treino.
> Comparamos 6 versões do mesmo modelo, que só diferem na forma como foram treinadas (regime de *dropout*).

---

## 1. A ideia em 30 segundos

Imagine um ator que recebe um papel: *"Você é Tobias, capitão de balsa aposentado, da Noruega, tem um gato
chamado Bolle…"*. Depois ele passa horas numa conversa sobre assuntos aleatórios (receitas, programação,
história). No fim alguém pergunta: **"Qual é o seu nome?"**

- Se a conversa foi curta, ele lembra: *"Tobias."*
- Se a conversa foi longa demais, alguns atores esquecem o papel, outros começam a falar coisas sem sentido.

Aqui o "ator" é um modelo de linguagem, e a "conversa longa" é medida em **tokens** (pedaços de palavra;
1.000 tokens ≈ 750 palavras). Os modelos foram treinados com textos de até **4.096 tokens (4K)**. Testamos
4K, 8K e 16K para ver o que acontece **dentro** (4K) e **além** (8K, 16K) do que eles conhecem.

---

## 2. Glossário rápido

| Termo | Explicação simples |
|---|---|
| **Token** | Pedaço de texto que o modelo lê (uma palavra curta ou parte de uma palavra). |
| **Contexto** | Tudo o que o modelo "vê" de uma vez: instruções + conversa + pergunta. |
| **4K / 8K / 16K** | Tamanho do contexto: 4.096 / 8.192 / 16.384 tokens. |
| **ID / OOD** | *In-distribution* (4K, igual ao treino) / *out-of-distribution* (maior que o treino). |
| **Sliding window (SWA)** | A maioria das camadas do modelo só "enxerga" os últimos 4.096 tokens. Algumas poucas camadas enxergam tudo. Então, acima de 4K, **a ficha da persona só chega ao fim da conversa por essas poucas camadas**. |
| **Dropout** | Técnica de treino que "desliga" partes aleatórias da rede, para ela não depender demais de um caminho só. Na hora de usar o modelo ele fica desligado; só muda **o que o modelo aprendeu**. |
| **Persona** | A "ficha de personagem" colocada no começo do contexto. |
| **Modelo base** | Modelo que só completa texto (não é um chatbot treinado para seguir instruções). |

---

## 3. Os 6 modelos

Todos são o mesmo **SmolLM3-3B**, re-treinado com contexto de 4K. A arquitetura e o tamanho são idênticos; só
muda o regime de treino:

| Nome | O que é | Hipótese do artigo base |
|---|---|---|
| **WD** | Sem dropout (só *weight decay*). É a linha de base. | generaliza mal |
| **PN-D-p30** | Dropout de 30% **antes da normalização** (RMSNorm). | generaliza mal |
| **PL-D-p30** | Dropout de 30% **antes das camadas lineares**. | generaliza bem |
| **VPAD-p10/p20/p30** | Variante de dropout com 10%, 20% e 30%. | generaliza bem |

Artigo base: *Length Generalization Needs Proper Regularization* (Vasylenko et al., 2026). Ele mostra que o
regime de dropout decide se o modelo funciona acima de 4K. **Aqui perguntamos algo novo: a persona também
sobrevive?**

---

## 4. Os dados

### 4.1 As personas (`data/personas.yaml`)
Três personagens inventados, cada um com **7 componentes**:

| Componente | Exemplo (Tobias) |
|---|---|
| biografia | nasceu em 1958 em Tromsø, Noruega; mora com o gato Bolle |
| papel | foi capitão de balsa por 30 anos; hoje é guia no museu do porto |
| personalidade | paciente, calmo, humor seco |
| preferências | adora sopa de peixe, café preto, prefere lugares calmos |
| valores | honestidade e pontualidade |
| memória | amigo de infância Arvid; a nevasca de 1992 |
| estilo | frases curtas, expressões náuticas |

Os outros dois são **Odalys Quintero** (pianista de jazz em New Orleans) e **Kenji Arakawa** (professor de
química em Sapporo). Para cada persona há:
- **6 perguntas factuais** com gabarito ("Qual é o seu nome?" → Tobias);
- **4 perguntas de escolha A/B** ("Prefere um vilarejo calmo ou o centro lotado?");
- **4 pedidos naturais** ("Como é um dia típico seu?");
- **1 frase-afirmação por componente** ("Eu trabalhei como capitão de balsa"), usada pelo juiz NLI.

### 4.2 O "recheio" da conversa (`data/filler.jsonl`)
Para encher o contexto até 4K, 8K ou 16K usamos conversas reais do dataset **UltraChat 200k** (licença MIT):
perguntas e respostas sobre temas variados. Cuidados tomados:
- **É idêntico para todos**: os mesmos turnos, na mesma ordem, para todos os modelos e condições. Só muda o tamanho.
- **É aninhado**: o recheio de 8K é o de 4K mais turnos extras. Assim só a distância cresce.
- **Foi limpo**:
  - removemos turnos em que o Assistant fala de si ("I am…", "my favorite…", "As an AI…");
  - removemos qualquer menção a nomes, cidades, profissões etc. das personas, para não "dar a resposta";
  - não repetimos diálogos.

### 4.3 Como um prompt é montado
```
┌─────────────────────────────────────────────────┐
│ CABEÇALHO   "The Assistant is Tobias Elling..."  │  ← muda conforme a condição
├─────────────────────────────────────────────────┤
│ RECHEIO     User: ... / Assistant: ...           │  ← igual para todos
│             (≈3,5K, 7,6K ou 15,8K tokens)        │
├─────────────────────────────────────────────────┤
│ PERGUNTA    User: What's your name?              │
│             Assistant:                           │  ← o modelo completa daqui
└─────────────────────────────────────────────────┘
```

### 4.4 As condições (o que vai no cabeçalho)
| Condição | Cabeçalho | Para que serve |
|---|---|---|
| **persona** | a ficha completa | é o que queremos medir |
| **neutra** | só "The following is a long conversation between a User and an Assistant." | ponto de comparação "sem persona" |
| **sham** | instruções genéricas do mesmo tamanho, sem identidade | controle: o efeito vem da persona ou só de "ter texto a mais"? |
| **distratora** | a ficha de **outra** persona | controle: o modelo responde pela persona certa? |
| **contraste** | a ficha com **1 traço invertido** (ex.: "prefere cidades lotadas") | testa se a escolha muda junto com a ficha |

---

## 5. A pipeline, passo a passo

| Passo | Arquivo | O que faz, em palavras simples |
|---|---|---|
| 0 | `config.yaml` | Painel de controle: modelos (com versão fixa), tamanhos, nº de repetições, seeds. Nada fica "escondido" no código. |
| 1 | `src/prepare.py` | Baixa o UltraChat, embaralha com seed fixa, filtra e salva o recheio. Também monta cada prompt em tokens. |
| 2 | `src/checks.py` | **Verificações antes de gastar horas rodando** (ver abaixo). |
| 3 | `src/generate.py` | Roda os modelos e salva todas as respostas em `results/`. Se for interrompido, continua de onde parou. |
| 4 | `src/metrics.py` | Calcula as 5 métricas e salva tabelas em `results/metrics/`. |
| 5 | `analysis.ipynb` | Gráficos: cada métrica × tamanho do contexto, 1 linha por modelo. |
| 6 | `RESULTS.md` | Resumo técnico e conciso dos resultados. |

**As verificações (`checks.py`):**
| | Pergunta que responde |
|---|---|
| V1 | Os 6 modelos "leem" o texto do mesmo jeito (mesmo tokenizer)? |
| V2 | A configuração tem mesmo o sliding window de 4.096? |
| V3 | **O código aplica o sliding window de verdade?** Teste da "agulha no palheiro": esconder um número num texto longo e pedir de volta. Esperado: WD falha acima de 4K e PL-D acerta. |
| V4 | Dois backends diferentes dão a mesma resposta? (só no cluster) |
| V5 | Todo prompt cabe no tamanho prometido, já contando a resposta? |
| V6 | O recheio está limpo (sem dicas das personas, sem autorrevelação)? |
| V7 | O recheio de 8K começa exatamente com o de 4K? |
| V8 | As opções " A" e " B" são um único token? (necessário para a métrica 3) |
| V9 | Teste rápido de ponta a ponta antes da coleta completa. |

**Dois truques de justiça na geração:**
- **Respostas "gêmeas":** a resposta com persona e a resposta neutra usam a **mesma semente aleatória**. É como
  jogar o mesmo dado duas vezes: a diferença que sobra vem da persona, não da sorte.
- **Repetições:** cada pergunta é respondida 4 vezes (perfil local) para medir a variação.

---

## 6. As métricas, explicadas de forma fácil

> Todas são **adaptações** de métodos de artigos científicos (citados entre parênteses), não cópias exatas.

### Métrica 1: "As respostas mudaram, no geral?" (MMD² e Energy, adaptação de CoDiTE)
Transformamos cada resposta num ponto num "mapa de significado" (embeddings do modelo E5). Depois comparamos
**a nuvem de pontos com persona** com **a nuvem sem persona**.
- **0** = as nuvens são iguais (a persona não mudou nada).
- **> 0** = as nuvens se afastaram.
- O **p-valor** diz se a diferença pode ser só sorte (p < 0,05 → provavelmente não é sorte).

### Métrica 2: "As respostas mudaram sempre na mesma direção?" (R² pareado, inspirado em PERMANOVA)
Para cada par de respostas gêmeas, medimos a "seta" que vai da resposta neutra até a resposta com persona. Se
todas as setas apontam para o mesmo lado, a persona empurra de forma **consistente**.
- Perto de **0,04** (o valor do acaso) = setas apontam para todo lado.
- Bem acima disso = efeito consistente.
- ⚠️ Não é "porcentagem explicada pela persona".

### Métrica 3: "A persona decide uma escolha?" (perguntas A/B, adaptação de Contrast Sets)
Fazemos uma pergunta de múltipla escolha cuja resposta da persona é conhecida, e olhamos **a probabilidade
que o modelo dá para "A" e para "B"**. Trocamos a ordem das opções para não favorecer uma letra.
- **escolhe a opção da persona**: % de perguntas em que ele prefere a opção certa (50% = cara ou coroa).
- **Δ margem vs. neutra**: quanto a persona aumenta a preferência pela opção certa, comparado a sem persona.
  É a medida mais justa, porque às vezes a opção da persona já é a "mais óbvia".
- **sucesso de inversão**: invertemos um traço na ficha e vemos se a escolha muda junto.

### Métrica 4: "A resposta livre mostra a persona?" (NLI por frase, adaptação de SummaC)
Um segundo modelo, o "juiz" (DeBERTa treinado em NLI), lê cada frase da resposta e diz se ela **confirma**
a afirmação da persona. Exemplo: a frase "Passei 30 anos no comando de balsas" confirma "Eu trabalhei como
capitão de balsa".
- **taxa de apoio** = % de respostas com pelo menos uma frase que confirma.
- ⚠️ "Não mencionou" não é o mesmo que "esqueceu".

### Métrica 5: "O modelo lembra os fatos?" (perguntas diretas, adaptação de QAFactEval)
Um quiz: *"Qual é o seu nome?"*, *"Em que cidade você mora?"*… Comparamos com o gabarito e com variações
aceitas ("Tromsø" ou "Tromso").
- **correct** = acertou; **contradictory** = respondeu outra coisa; **unresolved** = não respondeu ("não sei").
- Sem persona o acerto é 0%, como esperado: não há como adivinhar "Bolle".

### Extra: "A resposta saiu quebrada?" (degeneração)
Contamos respostas **vazias**, **truncadas** (bateram no limite de 256 tokens), **repetitivas** (> 50% de
trechos repetidos) ou **sem texto** (sequências de símbolos sem sentido). Isso é **resultado**, não erro:
um modelo que "quebra" acima de 4K está mostrando que não generaliza.

---

## 7. Resultados (perfil local: 4K, 8K, 16K)

### O teste de sanidade (agulha no palheiro)
| Modelo | 4K | 8K | 16K |
|---|---|---|---|
| WD | 100% | 0% | 0% |
| PL-D-p30 | 100% | 100% | 100% |

O WD acima de 4K responde coisas como `blue-&-&-&-&…`. O código aplica o sliding window corretamente.

### O resultado principal: a persona sobrevive?
**Métrica 5, acerto nas perguntas factuais com persona** (sem persona = 0% sempre):

| Modelo | 4K | 8K | 16K |
|---|---|---|---|
| WD | 56% | **0%** | **0%** |
| PN-D-p30 | 56% | **0%** | **0%** |
| VPAD-p10 | 67% | 39% | 28% |
| VPAD-p20 | 67% | 61% | 44% |
| VPAD-p30 | 72% | 67% | 56% |
| PL-D-p30 | 67% | 78% | 56% |

**Métrica 3, escolhe a opção da persona a 16K** (com persona / sem persona):
WD 25% / 50% · PN-D 25% / 50% · VPAD-p10 50% / 50% · VPAD-p20 83% / 67% · VPAD-p30 75% / 58% · PL-D 83% / 58%.

**Respostas quebradas com persona** (4K / 8K / 16K): WD 0 / **100** / **100%** · PN-D 0 / 8 / 0% · demais 0%.

### O que isso significa
1. **O jeito de treinar decide se a persona sobrevive.** WD e PN-D esquecem tudo acima de 4K. PL-D e VPAD
   continuam lembrando. É o mesmo padrão que o artigo base encontrou para outras tarefas.
2. **No VPAD, mais dropout = mais memória:** p10 < p20 < p30.
3. **Quem falha, falha de jeitos diferentes:**
   - o **WD "quebra"**: só produz símbolos sem sentido;
   - o **PN-D continua falando normalmente, mas perde o fio**. Perguntado "What's your name?", responde
     *"You can find information about your local area…"*.
4. **Mesmo os bons modelos vão esquecendo aos poucos.** O efeito da persona nas escolhas A/B cai de 2 a 6 vezes
   entre 4K e 16K.
5. **Perguntar diretamente funciona; esperar que o modelo "mostre" a persona sozinho, não.** As métricas 3 e 5
   separam bem os modelos. As métricas 1, 2 e 4 ficam perto de zero: mesmo a 4K, o modelo responde de forma
   genérica ("Um dia típico para mim envolve várias tarefas…") e quase nunca menciona a persona espontaneamente.
   Na métrica 1, o controle *sham* dá resultado "significativo" com a mesma frequência que a persona.

Detalhes, intervalos de confiança e controles: `RESULTS.md` e `analysis.ipynb`.

---

## 8. Limitações (o que não dá para concluir)
- **Amostra pequena** (perfil local): 3 personas, 4 repetições, 2 pedidos naturais, até 16K. Os intervalos de
  confiança são largos. O perfil `cluster` (até 64K, 8 repetições) está pronto, mas ainda não foi rodado.
- **As métricas usam regras simples**: aliases por início de palavra, lista de frases evasivas, limiares de
  repetição. Elas podem errar em casos de borda.
- **Lixo conta como "respondeu errado"** na métrica 5. Por isso ela deve ser lida junto com a taxa de respostas
  quebradas.
- **O recheio do UltraChat** é cheio de respostas genéricas de assistente, o que pode "abafar" a persona nas
  respostas livres.
- **Escopo das verificações:** o teste da agulha foi rodado só para WD e PL-D-p30. Só um backend foi usado
  (transformers no Mac).

---

## 9. Como rodar
```bash
# ambiente (Python 3.12)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python src/prepare.py                          # recheio + manifest
.venv/bin/python src/checks.py v1 v2 v5 v6 v7 v8 v3 v9    # verificações
.venv/bin/python src/generate.py                         # coleta (~5 h no Mac; retoma se parar)
.venv/bin/python src/metrics.py                          # métricas → results/metrics/
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
- Perfil: `profile:` no `config.yaml`, ou a variável `PROFILE=cluster` (usa vLLM; servidor em `slurm/vllm_array.sh`).
- Os modelos baixam ~6 GB cada. Sem internet, use `HF_HUB_OFFLINE=1` depois do primeiro download.
- **Reprodutibilidade:** `results/manifest.json` registra:
  - a versão exata (commit) de cada modelo, dataset, E5 e NLI;
  - o hash do recheio;
  - as seeds e as versões das bibliotecas.

## 10. Estrutura
```
config.yaml            painel de controle
data/personas.yaml     as 3 personas, perguntas, gabaritos
data/filler.jsonl      o recheio fixo da conversa
src/prepare.py         recheio + montagem dos prompts
src/checks.py          verificações V1–V9
src/generate.py        roda os modelos (backend hf ou vllm)
src/metrics.py         as 5 métricas + degeneração
slurm/vllm_array.sh    servidores vLLM no cluster (não testado)
results/               respostas, sondas, métricas, manifest, checks
analysis.ipynb         gráficos
RESULTS.md             resumo técnico
```
