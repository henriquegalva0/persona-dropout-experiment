"""As 5 métricas (ADAPTAÇÕES dos artigos citados, não implementações originais) → results/metrics/*.csv

Uso: python src/metrics.py [--tag smoke]

Tudo é estratificado por modelo × comprimento. Comparações: persona, sham e distractor, cada uma contra neutral.
  1. m1_distribution.csv: MMD² e Energy como estatísticas U pareadas (RBF, largura = mediana) (adapt. CoDiTE)
  2. m2_paired_r2.csv:    R² vetorial pareado com troca de sinais + BH (inspirado em PERMANOVA)
  3. m3_contrast.csv:     margem de logprob A/B nas duas ordens, inversão com contraste (adapt. Contrast Sets)
  4. m4_nli.csv:          NLI frase × claim, máx. entailment/contradiction por resposta (adapt. SummaC)
  5. m5_facts.csv:        correct / contradictory / unresolved por gabarito e aliases (adapt. QAFactEval)
  + degeneration.csv:     vazias, truncadas, repetição de n-gramas e saída sem texto
  + summary.csv:          tabela longa com valor e IC de todas as medidas (base do analysis.ipynb)
"""
import argparse
import re

import numpy as np
import pandas as pd

import prepare as P

COMPARISONS = ["persona", "sham", "distractor"]  # cada uma × neutral


# ------------------------------------------------------------------ utilidades
def device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def boot_mean(x, rng, B, ci):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.quantile(bs, [(1 - ci) / 2, (1 + ci) / 2])


def bh(p):
    """Benjamini–Hochberg (q-valores)."""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        ps = p[ok]
        o = np.argsort(ps)
        adj = ps[o] * len(ps) / np.arange(1, len(ps) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        r = np.empty(len(ps))
        r[o] = np.minimum(adj, 1)
        q[ok] = r
    return q


def summary_row(metric, measure, model, length, condition, value, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan):
    return dict(metric=metric, measure=measure, model=model, length=length, condition=condition,
                value=value, lo=lo, hi=hi, p=p, n=n)


def embed(cfg, texts):
    """E5 com prefixo 'passage: ', mean pooling, vetores normalizados."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    e, dev = cfg["embeddings"], device()
    tok = AutoTokenizer.from_pretrained(e["model"], revision=e["revision"])
    model = AutoModel.from_pretrained(e["model"], revision=e["revision"]).to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), e["batch_size"]):
            b = tok(["passage: " + t for t in texts[s:s + e["batch_size"]]], padding=True, truncation=True,
                    max_length=e["max_length"], return_tensors="pt").to(dev)
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ pares (mesma persona, prompt e seed)
def pairs(main, comp):
    """Índices pareados da condição `comp` e da neutra: mesmo modelo, L, persona, prompt e réplica (seed)."""
    key = ["model", "length", "persona_id", "prompt_id", "replica"]
    x = main[main.condition == comp].reset_index().set_index(key)["index"]
    y = main[main.condition == "neutral"].reset_index().set_index(key)["index"]
    both = x.index.intersection(y.index)
    return pd.DataFrame({"ix": x.loc[both].to_numpy(), "iy": y.loc[both].to_numpy()}, index=both).reset_index()


def strata(pr):
    """(model, length, persona_id, grupo): todas as personas juntas ('all') e cada uma separada."""
    for (model, L), g in pr.groupby(["model", "length"]):
        for pid in ["all"] + sorted(g.persona_id.unique()):
            yield model, L, pid, (g if pid == "all" else g[g.persona_id == pid])


# ------------------------------------------------------------------ 1. efeito distribucional
def _mmd_energy(K, D, x, y, ids):
    """Estatísticas U pareadas (Gretton et al. 2012, eq. 3): exclui os termos i = j, inclusive o cruzado
    (x_i, y_i), que compartilha a seed. `ids` = par original (no bootstrap, exclui duplicatas)."""
    off = ids[:, None] != ids[None, :]
    kxy, dxy = K[np.ix_(x, y)], D[np.ix_(x, y)]
    mmd = (K[np.ix_(x, x)] + K[np.ix_(y, y)] - kxy - kxy.T)[off].mean()
    en = (dxy + dxy.T - D[np.ix_(x, x)] - D[np.ix_(y, y)])[off].mean()
    return mmd, en


def m1_cell(Z, ix, iy, rng, mc):
    n = len(ix)
    Zs = Z[np.concatenate([ix, iy])]
    D = np.sqrt(np.maximum(2 - 2 * Zs @ Zs.T, 0))  # vetores normalizados
    sigma = np.median(D[np.triu_indices(2 * n, 1)])
    K = np.exp(-D ** 2 / (2 * sigma ** 2))
    x, y, ids = np.arange(n), np.arange(n, 2 * n), np.arange(n)
    obs = _mmd_energy(K, D, x, y, ids)
    perm = []
    for _ in range(mc["n_permutations"]):  # troca persona ↔ neutra dentro de cada par
        sw = rng.random(n) < 0.5
        perm.append(_mmd_energy(K, D, np.where(sw, y, x), np.where(sw, x, y), ids))
    boot = []
    for _ in range(mc["n_bootstrap"]):  # reamostra pares
        b = rng.integers(0, n, n)
        boot.append(_mmd_energy(K, D, x[b], y[b], b))
    perm, boot, a = np.array(perm), np.array(boot), (1 - mc["ci"]) / 2
    res = {}
    for k, name in enumerate(["mmd2", "energy"]):
        res[name] = obs[k]
        res[f"{name}_lo"], res[f"{name}_hi"] = np.nanquantile(boot[:, k], [a, 1 - a])
        res[f"{name}_p"] = (1 + (perm[:, k] >= obs[k]).sum()) / (1 + len(perm))
    return res


def metric1(main, Z, cfg, rng):
    rows = []
    for comp in COMPARISONS:
        for model, L, pid, s in strata(pairs(main, comp)):
            if len(s) >= 2:
                rows.append({"model": model, "length": L, "comparison": comp, "persona_id": pid, "n_pairs": len(s),
                             **m1_cell(Z, s.ix.to_numpy(), s.iy.to_numpy(), rng, cfg["metrics"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. R² vetorial pareado
def r2(d, ids=None):
    """R² = n·‖média(d)‖² / Σ‖d_i‖² = (Σ_i‖d_i‖² + Σ_{i≠j}⟨d_i,d_j⟩) / (n·Σ‖d_i‖²).
    `ids` (bootstrap): exclui os produtos cruzados entre cópias do mesmo par, que inflariam o R²."""
    G = d @ d.T
    ss = np.trace(G)
    if ss <= 0:
        return np.nan
    same = (ids[:, None] == ids[None, :]) if ids is not None else np.eye(len(d), dtype=bool)
    return (ss + G[~same].sum()) / (len(d) * ss)


def metric2(main, Z, cfg, rng):
    rows, mc = [], cfg["metrics"]
    for comp in COMPARISONS:
        for model, L, pid, s in strata(pairs(main, comp)):
            d = Z[s.ix.to_numpy()] - Z[s.iy.to_numpy()]
            n, obs = len(d), r2(d)
            null = np.array([r2(d * rng.choice([-1, 1], (n, 1))) for _ in range(mc["n_permutations"])])
            boot = np.array([r2(d[b], b) for b in (rng.integers(0, n, n) for _ in range(mc["n_bootstrap"]))])
            lo, hi = np.nanquantile(boot, [(1 - mc["ci"]) / 2, (1 + mc["ci"]) / 2])
            rows.append({"model": model, "length": L, "comparison": comp, "persona_id": pid, "n_pairs": n,
                         "r2": obs, "r2_lo": lo, "r2_hi": hi,
                         "null_mean": null.mean(), "p": (1 + (null >= obs).sum()) / (1 + len(null))})
    df = pd.DataFrame(rows)
    if len(df):  # BH entre células, por comparação e estrato
        df["q"] = df.groupby(["comparison", "persona_id"])["p"].transform(lambda p: bh(p.to_numpy()))
    return df


# ------------------------------------------------------------------ 3. escolha contrastiva
def metric3(ab):
    ab = ab.copy()
    ab["lp_persona_opt"] = np.where(ab.persona_first, ab.lp_A, ab.lp_B)
    ab["lp_contrast_opt"] = np.where(ab.persona_first, ab.lp_B, ab.lp_A)
    ab["margin"] = ab.lp_persona_opt - ab.lp_contrast_opt  # > 0: escolhe a opção da persona
    ab["ab_mass"] = np.exp(ab.lp_A) + np.exp(ab.lp_B)
    idx = ["model", "length", "persona_id", "prompt_id", "component", "condition"]
    w = ab.pivot_table(index=idx, columns="persona_first", values=["margin", "ab_mass"]).reset_index()
    w.columns = idx + ["mass_cf", "mass_pf", "margin_cf", "margin_pf"]
    w["margin"] = (w.margin_pf + w.margin_cf) / 2  # média das duas ordens (remove viés de posição)
    w["both_orders_persona"] = (w.margin_pf > 0) & (w.margin_cf > 0)
    w["ab_mass"] = (w.mass_pf + w.mass_cf) / 2
    return w.drop(columns=["mass_cf", "mass_pf"])


def metric3_summary(items, rng, mc):
    rows = []
    key = ["model", "length", "persona_id", "prompt_id"]
    wide = items.pivot_table(index=key, columns="condition", values="margin")
    for (model, L), g in wide.groupby(level=["model", "length"]):
        for cond in [c for c in ["persona", "neutral", "sham", "distractor", "contrast"] if c in g]:
            v, lo, hi = boot_mean(g[cond] > 0, rng, mc["n_bootstrap"], mc["ci"])
            rows.append(summary_row("m3", "picks_persona_option", model, L, cond, v, lo, hi, n=len(g)))
            if cond != "neutral" and "neutral" in g:
                v, lo, hi = boot_mean(g[cond] - g["neutral"], rng, mc["n_bootstrap"], mc["ci"])
                rows.append(summary_row("m3", "margin_diff_vs_neutral", model, L, cond, v, lo, hi, n=len(g)))
        if {"persona", "contrast"} <= set(g):
            v, lo, hi = boot_mean((g.persona > 0) & (g.contrast < 0), rng, mc["n_bootstrap"], mc["ci"])
            rows.append(summary_row("m3", "flip_success", model, L, "contrast", v, lo, hi, n=len(g)))
    return rows


# ------------------------------------------------------------------ 4. NLI por frase
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 3][:30]


def metric4(main, personas, cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    n, dev = cfg["nli"], device()
    tok = AutoTokenizer.from_pretrained(n["model"], revision=n["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(n["model"], revision=n["revision"]).to(dev).eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    claims = {p["id"]: p["claims"] for p in personas["personas"]}
    pairs, owner = [], []
    for i, r in main.iterrows():
        for s in split_sentences(r.text):
            pairs.append((s, claims[r.persona_id][r.component]))
            owner.append(i)
    probs = []
    with torch.no_grad():
        for s in range(0, len(pairs), n["batch_size"]):
            b = pairs[s:s + n["batch_size"]]
            x = tok([a for a, _ in b], [h for _, h in b], padding=True, truncation=True, return_tensors="pt").to(dev)
            probs.append(torch.softmax(model(**x).logits.float(), -1).cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros((0, 3))
    sent = pd.DataFrame({"i": owner, "entail": probs[:, lab["entailment"]],
                         "contra": probs[:, lab["contradiction"]]})
    agg = sent.groupby("i").agg(max_entail=("entail", "max"), max_contra=("contra", "max"), n_sentences=("i", "size"))
    out = main[["model", "length", "persona_id", "condition", "prompt_id", "component", "replica"]].join(agg)
    out["n_sentences"] = out.n_sentences.fillna(0).astype(int)
    out[["max_entail", "max_contra"]] = out[["max_entail", "max_contra"]].fillna(0.0)  # sem frases: sem apoio
    out["claim"] = [claims[p][c] for p, c in zip(out.persona_id, out.component)]
    return out


def metric4_summary(nli, rng, mc):
    rows = []
    pair = ["persona_id", "prompt_id", "replica"]
    for (model, L), g in nli.groupby(["model", "length"]):
        neu = g[g.condition == "neutral"].set_index(pair)
        for cond in ["persona", "neutral", "sham", "distractor"]:
            c = g[g.condition == cond]
            for measure, x in [("support_rate", c.max_entail > 0.5), ("contradiction_rate", c.max_contra > 0.5),
                               ("mean_max_entail", c.max_entail)]:
                rows.append(summary_row("m4", measure, model, L, cond, *boot_mean(x, rng, mc["n_bootstrap"], mc["ci"]),
                                        n=len(c)))
            if cond != "neutral":
                cc = c.set_index(pair)
                both = cc.index.intersection(neu.index)
                d = cc.loc[both, "max_entail"] - neu.loc[both, "max_entail"]
                rows.append(summary_row("m4", "entail_diff_vs_neutral", model, L, cond,
                                        *boot_mean(d, rng, mc["n_bootstrap"], mc["ci"]), n=len(d)))
    return rows


# ------------------------------------------------------------------ 5. recuperação factual
def classify_fact(text, aliases, unresolved):
    t = text.strip()
    if any(re.search(r"\b" + re.escape(a), t, re.I) for a in aliases):
        return "correct"
    if not t or any(u in t.lower() for u in unresolved):
        return "unresolved"
    return "contradictory"


def metric5(facts, personas, cfg):
    gold = {(p["id"], slot): f for p in personas["personas"] for slot, f in p["facts"].items()}
    out = facts[["model", "length", "persona_id", "condition", "prompt_id", "text", "finish_reason"]].copy()
    out["answer"] = [gold[(p, s)]["answer"] for p, s in zip(out.persona_id, out.prompt_id)]
    out["label"] = [classify_fact(t, gold[(p, s)]["aliases"], cfg["metrics"]["unresolved_patterns"])
                    for t, p, s in zip(out.text, out.persona_id, out.prompt_id)]
    return out


def metric5_summary(f, rng, mc):
    rows = []
    for (model, L), g in f.groupby(["model", "length"]):
        neu = g[g.condition == "neutral"].set_index(["persona_id", "prompt_id"])
        for cond in ["persona", "neutral", "sham", "distractor"]:
            c = g[g.condition == cond]
            answered = c[c.label != "unresolved"]
            for measure, x in [("correct_rate", c.label == "correct"), ("coverage", c.label != "unresolved"),
                               ("accuracy_answered", answered.label == "correct"),
                               ("contradiction_rate", c.label == "contradictory")]:
                rows.append(summary_row("m5", measure, model, L, cond, *boot_mean(x, rng, mc["n_bootstrap"], mc["ci"]),
                                        n=len(x)))
            if cond != "neutral":
                cc = c.set_index(["persona_id", "prompt_id"])
                both = cc.index.intersection(neu.index)
                d = (cc.loc[both].label == "correct").astype(float) - (neu.loc[both].label == "correct").astype(float)
                rows.append(summary_row("m5", "correct_diff_vs_neutral", model, L, cond,
                                        *boot_mean(d, rng, mc["n_bootstrap"], mc["ci"]), n=len(d)))
    return rows


# ------------------------------------------------------------------ degeneração
def degeneration(main, mc):
    def repeat_ratio(t, n=mc["degenerate_ngram"]):
        w = t.split()
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        return 1 - len(set(grams)) / len(grams) if len(grams) >= n else 0.0

    def word_ratio(t):  # fração de "palavras" com cara de texto (heurística extra para saída sem sentido)
        w = t.split()
        return np.mean([bool(re.fullmatch(r"[\"'(*]?[A-Za-z0-9][\w'’\-]*[.,!?;:)\"'”’*%]*", x)) for x in w]) if w else 1.0

    d = main[["model", "length", "persona_id", "condition", "prompt_id", "replica", "finish_reason"]].copy()
    d["empty"] = main.text.str.strip() == ""
    d["truncated"] = main.finish_reason == "length"
    d["repeat_ratio"] = main.text.map(repeat_ratio)
    d["degenerate"] = d.repeat_ratio > mc["degenerate_repeat_ratio"]
    d["non_text"] = main.text.map(word_ratio) < 0.5
    d["any_bad"] = d.empty | d.degenerate | d.non_text
    return d


def degeneration_summary(d, rng, mc):
    rows = []
    for (model, L, cond), g in d.groupby(["model", "length", "condition"]):
        for measure in ["empty", "truncated", "degenerate", "non_text", "any_bad"]:
            rows.append(summary_row("degeneration", measure, model, L, cond,
                                    *boot_mean(g[measure], rng, mc["n_bootstrap"], mc["ci"]), n=len(g)))
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = P.load_config()
    mc, personas = cfg["metrics"], P.load_personas(cfg)
    rng = np.random.default_rng(cfg["seed"])
    out = P.path(cfg, "metrics") / args.tag if args.tag else P.path(cfg, "metrics")
    out.mkdir(parents=True, exist_ok=True)

    gens = pd.read_json(P.out_path(cfg, "generations", args.tag), lines=True)
    probes = pd.read_json(P.out_path(cfg, "probes", args.tag), lines=True)
    errors = pd.concat([gens, probes])
    errors = errors[errors["error"].notna()] if "error" in errors else errors.iloc[:0]
    errors.to_csv(out / "errors.csv", index=False)  # falhas registradas, não descartadas em silêncio
    ok = lambda df: (df[df["error"].isna()] if "error" in df else df).reset_index(drop=True)
    main_, probes = ok(gens), ok(probes)
    main_["text"] = main_.text.fillna("")

    Z = embed(cfg, main_.text.tolist())
    m1 = metric1(main_, Z, cfg, rng)
    m2 = metric2(main_, Z, cfg, rng)
    m3 = metric3(probes[probes.kind == "ab"])
    m4 = metric4(main_, personas, cfg)
    facts = probes[probes.kind == "fact"].copy()
    facts["text"] = facts.text.fillna("")
    m5 = metric5(facts, personas, cfg)
    deg = degeneration(main_, mc)

    rows = []
    for _, r in m1[m1.persona_id == "all"].iterrows():
        for k in ["mmd2", "energy"]:
            rows.append(summary_row("m1", k, r.model, r.length, r.comparison, r[k], r[f"{k}_lo"], r[f"{k}_hi"],
                                    r[f"{k}_p"], r.n_pairs))
    for _, r in m2[m2.persona_id == "all"].iterrows():
        rows.append(summary_row("m2", "r2", r.model, r.length, r.comparison, r.r2, r.r2_lo, r.r2_hi, r.q, r.n_pairs))
        rows.append(summary_row("m2", "r2_null_mean", r.model, r.length, r.comparison, r.null_mean, n=r.n_pairs))
    rows += metric3_summary(m3, rng, mc) + metric4_summary(m4, rng, mc) + metric5_summary(m5, rng, mc)
    rows += degeneration_summary(deg, rng, mc)

    for name, df in [("m1_distribution", m1), ("m2_paired_r2", m2), ("m3_contrast", m3), ("m4_nli", m4),
                     ("m5_facts", m5), ("degeneration", deg), ("summary", pd.DataFrame(rows))]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} linhas")
    print(f"erros registrados: {len(errors)}")


if __name__ == "__main__":
    main()
