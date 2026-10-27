# Persona × Comprimento de Contexto × Dropout

## Resumo
Este repositório avalia se uma **persona** definida no início do contexto continua influenciando um modelo de
linguagem quando o contexto excede o comprimento visto no treino. Seis checkpoints do SmolLM3-3B, idênticos em
arquitetura e diferentes apenas no regime de *dropout* do treino, são testados com contextos de 4K, 8K e 16K
tokens. O treino foi feito a 4K.

**Principal achado:** o regime de dropout determina se a persona sobrevive acima de 4K. WD e PN-D perdem a
persona por completo, enquanto PL-D e VPAD a preservam parcialmente.

## Fundamentação
- **Base:** *Length Generalization Needs Proper Regularization* (Vasylenko et al., 2026). O artigo mostra que
  o regime de dropout decide a generalização de comprimento. Hipótese: WD ≈ PN-D < PL-D ≈ VPAD.
- **Arquitetura:** 27 das 36 camadas usam *sliding window* de 4.096 tokens. As 9 restantes têm atenção global.
  Acima de 4K, a persona só alcança o fim do contexto por meio dessas camadas globais.
- **Inferência:** o dropout é desligado. Os modelos diferem apenas nos pesos aprendidos.

| Modelo | Regime de treino |
|---|---|
| WD | sem dropout (apenas *weight decay*), linha de base |
| PN-D-p30 | dropout de 30% antes do RMSNorm |
| PL-D-p30 | dropout de 30% antes das camadas lineares |
| VPAD-p10 / p20 / p30 | VPAD com 10%, 20% e 30% |

## Dados
- **Personas** (`data/personas.yaml`): três personagens sintéticos (Tobias, Odalys, Kenji), cada um com sete
  componentes (biografia, papel, personalidade, preferências, valores, memória, estilo). Cada persona traz
  6 perguntas factuais com gabarito, 4 perguntas A/B, 4 pedidos abertos e 1 afirmação por componente.
- **Conversa de preenchimento** (`data/filler.jsonl`): turnos do UltraChat 200k (licença MIT) usados para
  completar o contexto até o comprimento desejado. O preenchimento é:
  - idêntico entre modelos e condições;
  - aninhado entre comprimentos (o de 8K começa com o de 4K);
  - filtrado contra autorrevelação do assistente e contra qualquer termo das personas.
- **Estrutura do prompt:** `cabeçalho` → `conversa de preenchimento` → `pergunta avaliada` → `Assistant:`.

| Condição (cabeçalho) | Finalidade |
|---|---|
| persona | ficha completa da persona (condição avaliada) |
| neutra | frase genérica sem identidade (referência) |
| sham | instruções genéricas de mesmo tamanho (controle de comprimento) |
| distratora | ficha de outra persona (controle de especificidade) |
| contraste | ficha com um componente invertido (teste de inversão nas perguntas A/B) |

## Pipeline
| Etapa | Arquivo | Função |
|---|---|---|
| Configuração | `config.yaml` | Modelos com revisão fixada, comprimentos, réplicas, seeds e perfis (`local`, `cluster`). |
| Preparação | `src/prepare.py` | Constrói a conversa de preenchimento e monta os prompts em token ids. |
| Verificação | `src/checks.py` | Testes V1–V9, entre eles o teste da agulha no palheiro, que confirma o *sliding window*. |
| Geração | `src/generate.py` | Executa os modelos. Persona e neutra usam a mesma seed para formar pares comparáveis. A coleta é retomável. |
| Métricas | `src/metrics.py` | Calcula as cinco métricas e a taxa de degeneração em `results/metrics/`. |
| Análise | `analysis.ipynb` | Gráficos de cada métrica por comprimento, um por modelo, com IC de 95%. |

## Métricas
Todas são adaptações dos métodos citados, não reproduções exatas.

| # | Métrica | Pergunta respondida | Interpretação |
|---|---|---|---|
| 1 | MMD² e Energy pareados (CoDiTE) | A persona altera a distribuição das respostas? | 0 indica ausência de efeito. |
| 2 | R² vetorial pareado (PERMANOVA) | A persona desloca as respostas numa direção consistente? | Comparar com o nulo (≈ 0,04). |
| 3 | Escolha A/B por logprob (Contrast Sets) | A persona determina escolhas e se inverte com o contraste? | Δ margem positivo indica efeito. |
| 4 | NLI por frase (SummaC) | A resposta aberta expressa a persona? | % de respostas com *entailment* > 0,5. |
| 5 | Perguntas factuais (QAFactEval) | O modelo recupera os fatos da persona? | % de acertos frente ao gabarito. |
| — | Degeneração | A resposta é vazia, truncada, repetitiva ou sem sentido? | % por modelo e comprimento. |

## Resultados
Perfil `local`: 3 personas, 4 réplicas, comprimentos de 4K, 8K e 16K.

**Verificação do *sliding window*** (agulha no palheiro, % de acerto em 4K / 8K / 16K):
WD 100 / 0 / 0 · PL-D-p30 100 / 100 / 100.

**Recuperação factual com persona** (métrica 5; sem persona, o acerto é 0% em todos os casos):

| Modelo | 4K | 8K | 16K | Degeneração a 16K |
|---|---|---|---|---|
| WD | 56% | 0% | 0% | 100% |
| PN-D-p30 | 56% | 0% | 0% | 0% |
| VPAD-p10 | 67% | 39% | 28% | 0% |
| VPAD-p20 | 67% | 61% | 44% | 0% |
| VPAD-p30 | 72% | 67% | 56% | 0% |
| PL-D-p30 | 67% | 78% | 56% | 0% |

**Escolha da opção da persona a 16K** (métrica 3, com persona / neutra): WD 25 / 50% · PN-D 25 / 50% ·
VPAD-p10 50 / 50% · VPAD-p20 83 / 67% · VPAD-p30 75 / 58% · PL-D 83 / 58%.

**Conclusões**
1. **O regime de dropout determina a retenção da persona fora da distribuição.** WD e PN-D perdem a persona;
   PL-D e VPAD a preservam. A ordem confirma a hipótese do artigo base.
2. **No VPAD, a retenção cresce com a taxa de dropout** (p10 < p20 < p30).
3. **Os modos de falha diferem.** O WD degenera em texto sem sentido. O PN-D permanece fluente, mas perde a
   persona e a pergunta.
4. **A influência decai com o comprimento mesmo nos modelos que generalizam.** O efeito nas escolhas A/B cai
   de 2× a 6× entre 4K e 16K.
5. **Só as sondas diretas (métricas 3 e 5) discriminam os regimes.** As métricas sobre respostas abertas (1, 2 e 4)
   permanecem próximas de zero, porque o modelo base raramente expressa a persona de forma espontânea.

Intervalos de confiança, controles e tabelas completas estão em `RESULTS.md` e `analysis.ipynb`.

## Reprodução
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/prepare.py
.venv/bin/python src/checks.py v1 v2 v5 v6 v7 v8 v3 v9
.venv/bin/python src/generate.py        # PROFILE=cluster para vLLM (slurm/vllm_array.sh)
.venv/bin/python src/metrics.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
O arquivo `results/manifest.json` registra:
- as revisões (commit) dos modelos, do dataset, do E5 e do NLI;
- o hash da conversa de preenchimento;
- as seeds e as versões das bibliotecas.

## Estrutura do repositório
```
config.yaml           configuração central
data/                 personas e conversa de preenchimento
src/                  prepare, checks, generate, metrics
slurm/                servidores vLLM para o cluster
results/              gerações, sondas, métricas, manifest
analysis.ipynb        análise gráfica
RESULTS.md            relatório técnico
```
