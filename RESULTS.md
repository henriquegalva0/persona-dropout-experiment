# Persona × comprimento × dropout: resultados (perfil `local`)

## Pergunta
Uma persona definida no início do contexto continua influenciando o modelo quando o contexto passa do
comprimento de treino (4K)? Isso depende do regime de dropout do continual pre-training?

## Setup
- **Modelos**: 6 checkpoints do mesmo SmolLM3-3B (CPT a 4K; SWA de 4096 nas 27 camadas RoPE; 9 camadas NoPE
  globais). Na inferência todos têm o mesmo grafo e o dropout está desligado: só os pesos diferem.
  WD (sem dropout), PN-D-p30 (antes do RMSNorm), PL-D-p30 (antes do Linear), VPAD-p10/p20/p30.
- **Comprimentos** (orçamento prompt + geração): 4K (ID), 8K e 16K (OOD).
- **Prompt** (completion, sem chat template): `HEADER` → filler fixo → turno avaliado → `Assistant:`.
  O filler (UltraChat 200k, MIT, filtrado contra autorrevelação, identidade de IA e termos das personas) é
  idêntico entre condições e modelos e aninhado entre comprimentos. A persona fica a ~3,5K / 7,6K / 15,8K
  tokens do turno avaliado, ou seja, fora da janela SWA a partir de 8K.
- **Condições**: persona · neutra · sham (instrução genérica de mesmo tamanho) · distratora (outra persona) ·
  contraste (persona com 1 componente invertido, só no A/B).
- **Personas**: 3 sintéticas (Tobias, Odalys, Kenji) × 7 componentes (`data/personas.yaml`).
- **Geração**: 2 pedidos naturais × 4 réplicas (`T=0,7`, 256 tokens; persona e neutra com a mesma seed);
  6 fatos greedy (16 tokens); 4 pares A/B por logprob, nas duas ordens.
- **Backend**: transformers 5.21 em Apple MPS (bf16, sdpa), prefill em chunks com cache por header.
  5.184 registros, 0 falhas. Revisões, SHA-256 do filler e versões em `results/manifest.json`.

## Verificações
V1 (tokenizer idêntico), V2 (`layer_types` e janela 4096), V5–V8 (orçamento, fronteira de turno, filler limpo
e aninhado, `" A"`/`" B"` token único) e V9 (smoke) passaram. **V3 (NIAH, estilo RULER)**: WD 100/0/0% e
PL-D-p30 100/100/100% em 4K/8K/16K, o que confirma que o backend aplica o SWA. Teste direto: com o SWA desligado,
os logits só mudam acima de 4096 tokens. V4 não se aplica (um único backend).

## O que cada métrica mede (adaptações, não implementações originais)
| # | Métrica (inspiração) | Pergunta | Como ler |
|---|---|---|---|
| 1 | MMD² e Energy pareados (CoDiTE) | o header muda a *distribuição* das respostas (E5)? | 0 = sem efeito; p por troca dentro do par |
| 2 | R² vetorial pareado (PERMANOVA) | as respostas mudam numa *direção consistente*? | comparar com o nulo (≈ 0,04); não é % causal |
| 3 | Margem A/B nas duas ordens (Contrast Sets) | a persona *decide* escolhas? inverte com o contraste? | Δ margem vs neutra > 0 = efeito da persona |
| 4 | NLI frase × claim (SummaC) | a resposta livre *expressa* o componente? | apoio = entailment > 0,5 |
| 5 | Gabarito + aliases (QAFactEval) | o modelo *lembra* os fatos quando perguntado? | acerto; neutra = 0% em tudo |

M1 e M2 usam estatísticas U pareadas, que excluem o termo cruzado do próprio par porque ele compartilha a seed.
Os ICs (95%) vêm de bootstrap.

## Resultados (condição persona)

| Modelo | M5 fatos 4K / 8K / 16K | M3 Δ margem vs neutra (nats) 4K / 8K / 16K | Degeneração 4K / 8K / 16K |
|---|---|---|---|
| WD | 56 / **0** / **0** % | 0,52 / 0,00 / −0,00 | 0 / **100** / **100** % |
| PN-D-p30 | 56 / **0** / **0** % | 0,33 / −0,02 / −0,01 | 0 / 8 / 0 % |
| PL-D-p30 | 67 / 78 / 56 % | 0,20 / 0,10 / 0,09 | 0 / 0 / 0 % |
| VPAD-p10 | 67 / 39 / 28 % | 0,41 / 0,07 / −0,02 | 0 / 0 / 0 % |
| VPAD-p20 | 67 / 61 / 44 % | 0,43 / 0,18 / 0,07 | 0 / 0 / 0 % |
| VPAD-p30 | 72 / 67 / 56 % | 0,36 / 0,16 / 0,09 | 0 / 0 / 0 % |

ICs de 95%: M5 a 16K, PL-D-p30 e VPAD-p30 [33, 78]%; M3 a 16K, PL-D-p30 [0,05, 0,14] e VPAD-p30 [0,03, 0,16];
WD e PN-D-p30 ficam com |Δ| ≤ 0,02 em 8K e 16K. As tabelas completas com IC estão no `analysis.ipynb` (seções 1–9).

**A/B, fração que escolhe a opção da persona** (com persona / neutra): a 16K, WD 25/50%, PN-D 25/50%,
PL-D 83/58%, VPAD-p10 50/50%, VPAD-p20 83/67%, VPAD-p30 75/58%. A 4K, todos ficam entre 83 e 100% (neutra 75–83%).
Sucesso de inversão com o contraste: ≤ 42% a 4K e ≤ 25% em OOD em todos os modelos.

**M1/M2**: efeitos pequenos e instáveis. Persona × neutra dá p < 0,05 em 6 de 18 células no M1 e q < 0,05 em
6 de 18 no M2, e o **sham** dá p < 0,05 em 5 de 18 no M1. Não há padrão por modelo ou comprimento.

**M4**: apoio quase nulo em tudo (máx. 8%, VPAD-p30 a 4K). Mesmo a 4K, o modelo base responde aos pedidos
naturais de forma genérica ("A typical day for me involves…").

**Controles**: com a persona distratora, sham ou neutra, o acerto dos fatos da persona-alvo é 0% em todas as
células.

## Conclusões
1. **O regime de dropout decide se a persona sobrevive acima de 4K.** A 16K: WD = PN-D (0% dos fatos, Δ margem
   ≈ 0) < VPAD-p10 (28%) < VPAD-p20 (44%) < PL-D-p30 ≈ VPAD-p30 (56%). A ordem confirma a hipótese do paper
   (WD ≈ PN-D < PL-D ≈ VPAD) e mostra dose-resposta no VPAD (p10 < p20 < p30).
2. **Os modos de falha diferem.** O WD colapsa em texto sem sentido (100% degenerado acima de 4K, 0% no NIAH).
   O PN-D continua fluente mas perde a persona e a pergunta: a "What's your name?" responde
   "You can find information about your local area…".
3. **Mesmo nos modelos que generalizam, a influência decai com o comprimento.** O Δ margem A/B cai de 2× a 6× entre 4K
   e 16K (PL-D-p30: 0,20 → 0,09; VPAD-p30: 0,36 → 0,09; VPAD-p20: 0,43 → 0,07), e o acerto factual cai mais devagar.
4. **As métricas que funcionam neste desenho são as de sonda direta (M3, M5).** As de resposta livre (M1, M2, M4)
   ficam no piso porque o modelo base quase não expressa a persona espontaneamente, nem a 4K.

## Limitações
- Perfil local: só até 16K, 4 réplicas, 2 pedidos naturais e 3 personas. Os ICs são largos (M5: 18 perguntas por
  célula; M3: 12 pares). O perfil `cluster` (até 64K, 8 réplicas, 4 pedidos) ainda não foi rodado.
- As métricas são adaptações com heurísticas próprias: aliases por prefixo de palavra; padrões de evasiva para
  `unresolved`; limiares de degeneração (> 50% de 4-gramas repetidos; < 50% de tokens com cara de palavra).
- A cobertura do M5 é ~100% até para o WD degenerado, porque lixo conta como `contradictory`. Leia junto com a
  degeneração.
- No A/B, a opção da persona costuma ser também a "sensata" (neutra 75% a 4K). Por isso o efeito é medido como
  Δ margem contra a neutra.
- O filler (respostas genéricas de assistente) provavelmente compete com a persona e contribui para o piso de
  M1/M2/M4. M4 depende de claims em 1ª pessoa e do limiar 0,5.
- NIAH (V3) só foi rodado para WD e PL-D-p30. Um único backend (HF/MPS) foi usado, sem comparação com o vLLM (V4).

## Como reproduzir
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/prepare.py                          # filler + manifest
.venv/bin/python src/checks.py v1 v2 v5 v6 v7 v8 v3 v9    # verificações
.venv/bin/python src/generate.py                         # coleta (PROFILE=cluster para o perfil do cluster)
.venv/bin/python src/metrics.py                          # results/metrics/*.csv
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```
A coleta retoma de onde parou, e itens já gravados são pulados. No cluster: `slurm/vllm_array.sh` (não testado).
