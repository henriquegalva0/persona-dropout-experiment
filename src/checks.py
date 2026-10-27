"""Verificações antes da coleta. Uso: python src/checks.py v1 v2 v8 v5 v6 v7 v3 v9 [--models WD,PL-D-p30]

Cada verificação grava {ok, info} em results/checks.json. Sai com código 1 se alguma falhar.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

import prepare as P


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def v1(cfg):
    """Os 6 modelos têm tokenizer idêntico (hash do vocabulário)."""
    info = {}
    for m in cfg["models"]:
        tj = json.loads(Path(hf_hub_download(m["id"], "tokenizer.json", revision=m["revision"])).read_text())
        info[m["label"]] = {"vocab_sha256": sha256(json.dumps(tj["model"]["vocab"], sort_keys=True)),
                            "added_tokens_sha256": sha256(json.dumps(tj["added_tokens"], sort_keys=True)),
                            "merges_sha256": sha256(json.dumps(tj["model"].get("merges"))),
                            "vocab_size": len(tj["model"]["vocab"])}
    ok = len({json.dumps(h, sort_keys=True) for h in info.values()}) == 1
    if ok:
        P.update_manifest(cfg, "tokenizer", next(iter(info.values())))
    return ok, info


def v2(cfg):
    """layer_types: sliding nas camadas RoPE, full nas NoPE; sliding_window=4096."""
    ex, info, ok = cfg["expected_arch"], {}, True
    for m in cfg["models"]:
        c = json.loads(Path(hf_hub_download(m["id"], "config.json", revision=m["revision"])).read_text())
        lt, rope = c["layer_types"], c["no_rope_layers"]  # no_rope_layers[i]=1 → camada usa RoPE
        expected = ["sliding_attention" if r else "full_attention" for r in rope]
        good = (lt == expected and c["sliding_window"] == ex["sliding_window"] and c["use_sliding_window"]
                and c["num_hidden_layers"] == ex["num_hidden_layers"] == len(lt))
        info[m["label"]] = {"ok": good, "sliding_window": c["sliding_window"],
                            "n_sliding": lt.count("sliding_attention"), "n_full": lt.count("full_attention"),
                            "full_layers": [i for i, t in enumerate(lt) if t == "full_attention"],
                            "max_position_embeddings": c["max_position_embeddings"]}
        ok &= good
    return ok, info


def v8(cfg):
    """' A' e ' B' são token único, e prompt + opção tokeniza como prompt_ids + [opção]."""
    tok, personas = P.load_tokenizer(cfg), P.load_personas(cfg)
    opts = {o: P.encode(tok, o) for o in cfg["generation"]["ab_options"]}
    ok = all(len(v) == 1 for v in opts.values())
    for it in (i for i in P.items(cfg, personas) if i["kind"] == "ab"):
        text = P.turn_text(it["user"])
        for o, ids in opts.items():
            ok &= P.encode(tok, text + o) == P.encode(tok, text) + ids
    return ok, {"option_ids": opts}


def _setup(cfg):
    return P.load_tokenizer(cfg), P.load_personas(cfg), P.load_filler(cfg)


def v5(cfg):
    """len(prompt) + max_tokens <= L em todo prompt; filler cortado em fronteira de turno."""
    tok, personas, filler = _setup(cfg)
    gen, ok, info = cfg["generation"], True, {}
    max_new = {"main": gen["main"]["max_tokens"], "fact": gen["facts"]["max_tokens"], "ab": 1}
    for L in cfg["profiles"]["cluster"]["lengths"]:
        hdr, fill, n = P.segments(cfg, tok, personas, L, filler)
        text = "".join(P.exchange_text(e["user"], e["assistant"]) for e in filler[:n])
        boundary = fill == P.encode(tok, text) and tok.decode(fill) == text
        worst = max(len(hdr[it["header_id"]]) + len(fill) + len(P.encode(tok, P.turn_text(it["user"])))
                    + max_new[it["kind"]] for it in P.items({**cfg, "run": cfg["profiles"]["cluster"]}, personas))
        # um prompt completo tokenizado de uma vez == concatenação dos segmentos
        it = P.items(cfg, personas)[0]
        full = P.encode(tok, P.headers(personas)[it["header_id"]] + "\n\n" + text + P.turn_text(it["user"]))
        concat = full == hdr[it["header_id"]] + fill + P.encode(tok, P.turn_text(it["user"]))
        info[L] = {"worst_total": worst, "fits": worst <= L, "turn_boundary": boundary, "concat_ok": concat}
        ok &= worst <= L and boundary and concat
    return ok, info


def v6(cfg):
    """Filler sem fatos/aliases das personas, sem autorrevelação, turnos curtos, sem duplicatas."""
    tok, personas, filler = _setup(cfg)
    disclosure, contaminated = P.disclosure_regex(cfg), P.contamination_regex(personas)
    hits = {"disclosure": [], "contamination": [], "too_long": []}
    for e in filler:
        text = e["user"] + "\n" + e["assistant"]
        if disclosure.search(text):
            hits["disclosure"].append(e["i"])
        if m := contaminated.search(text):
            hits["contamination"].append((e["i"], m.group(0)))
        if max(len(P.encode(tok, e["user"])), len(P.encode(tok, e["assistant"]))) > cfg["filler"]["max_turn_tokens"]:
            hits["too_long"].append(e["i"])
    dups = len(filler) - len({" ".join(e["user"].lower().split()) for e in filler})
    manifest = json.loads(P.path(cfg, "manifest").read_text())
    sha_ok = manifest["filler"]["sha256"] == P.sha256_file(P.path(cfg, "filler"))
    ok = not any(hits.values()) and dups == 0 and sha_ok
    return ok, {**hits, "duplicates": dups, "sha256_matches_manifest": sha_ok}


def v7(cfg):
    """Filler aninhado: o de L é prefixo exato do de 2L."""
    tok, personas, filler = _setup(cfg)
    lengths = cfg["profiles"]["cluster"]["lengths"]
    fills = {L: P.segments(cfg, tok, personas, L, filler)[1] for L in lengths}
    info = {f"{a}⊂{b}": fills[b][: len(fills[a])] == fills[a] and len(fills[a]) < len(fills[b])
            for a, b in zip(lengths, lengths[1:])}
    return all(info.values()), info


NIAH_NOISE = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NIAH_KEYS = ["amber", "falcon", "quartz", "willow", "cobalt", "meadow", "harbor", "ember", "lantern", "glacier"]


def niah_prompt(tok, L, depth, key, value, max_new):
    """NIAH no estilo RULER niah_single_1 (haystack de frases repetidas), cabendo em L - max_new."""
    head = ("Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards.\n")
    needle = f" One of the special magic numbers for {key} is: {value}."
    query = (f"\nWhat is the special magic number for {key} mentioned in the provided text? "
             f"The special magic number for {key} mentioned in the provided text is")
    n = (L - max_new) // len(P.encode(tok, NIAH_NOISE))
    while True:
        k = round(depth * n)
        after = NIAH_NOISE * (n - k) + query
        ids = P.encode(tok, head + NIAH_NOISE * k + needle + after)
        if len(ids) <= L - max_new:
            return ids, len(P.encode(tok, after))  # distância (tokens) da agulha até o fim
        n -= 1


def v3(cfg, models=None):
    """O backend aplica o sliding window: NIAH greedy deve separar WD (falha >4K) de PL-D-p30."""
    import generate as G

    tok, c = P.load_tokenizer(cfg), cfg["checks"]
    out = P.path(cfg, "niah")
    done = {(r["model"], r["length"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())} \
        if out.exists() else set()
    for m in [m for m in cfg["models"] if m["label"] in (models or c["niah_models"])]:
        model = None
        with open(out, "a") as f:
            for L in c["niah_lengths"]:
                for i in range(c["niah_samples"]):
                    if (m["label"], L, i) in done:
                        continue
                    model = model or G.hf_load(cfg, m)
                    rng = random.Random(cfg["seed"] + i)
                    key, value = rng.choice(NIAH_KEYS), str(rng.randint(1_000_000, 9_999_999))
                    depth = i / (c["niah_samples"] - 1)
                    ids, dist = niah_prompt(tok, L, depth, key, value, c["niah_max_tokens"])
                    cache = G.hf_new_cache(model)
                    logits = G.hf_forward(model, ids, cache, cfg["hf"]["prefill_chunk"])
                    text = G.hf_decode(model, tok, cache, logits, [0], 0.0, c["niah_max_tokens"], [])[0]["text"]
                    f.write(json.dumps({"model": m["label"], "length": L, "sample": i, "depth": depth,
                                        "needle_distance": dist, "prompt_tokens": len(ids), "value": value,
                                        "output": text, "correct": value in text}) + "\n")
                    f.flush()
        del model
    import pandas as pd

    df = pd.read_json(out, lines=True)
    acc = df.groupby(["model", "length"])["correct"].mean().unstack().round(2)
    ood = [L for L in acc.columns if L > cfg["train_length"]]
    spread = (acc[ood].max() - acc[ood].min()).max() if len(acc) > 1 and ood else 0.0
    return bool(spread >= 0.5), {"accuracy": acc.to_dict(orient="index"), "max_spread_ood": float(spread)}


def v4(cfg, models=None):
    """vLLM × HF greedy em 5 prompts (só quando os dois backends são usados)."""
    if cfg["run"]["backend"] == "hf":
        return True, {"skipped": "perfil local usa só o backend hf"}
    raise NotImplementedError("V4: rode no cluster com o servidor vLLM e o backend hf no mesmo nó")


def v9(cfg, models=None):
    """Smoke test ponta a ponta: 1 persona × 1 modelo × 4K × 1 réplica → todas as métricas."""
    label, pid = (models or [cfg["models"][0]["label"]])[0], P.load_personas(cfg)["personas"][0]["id"]
    py, src = sys.executable, P.ROOT / "src"
    gen = subprocess.run([py, src / "generate.py", "--tag", "smoke", "--models", label, "--lengths",
                          str(cfg["train_length"]), "--personas", pid, "--replicas", "1"], cwd=P.ROOT)
    met = subprocess.run([py, src / "metrics.py", "--tag", "smoke"], cwd=P.ROOT)
    d = P.path(cfg, "metrics") / "smoke"
    files = {f.name: sum(1 for _ in open(f)) - 1 for f in sorted(d.glob("*.csv"))} if d.exists() else {}
    needed = ["m1_distribution.csv", "m2_paired_r2.csv", "m3_contrast.csv", "m4_nli.csv", "m5_facts.csv",
              "degeneration.csv", "summary.csv"]
    ok = gen.returncode == 0 and met.returncode == 0 and all(files.get(n, 0) > 0 for n in needed)
    return ok, {"model": label, "persona": pid, "csv_rows": files}


CHECKS = {"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6, "v7": v7, "v8": v8, "v9": v9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(CHECKS))
    ap.add_argument("--models", default=None, help="labels para V3/V9")
    args = ap.parse_args()
    cfg = P.load_config()
    names, models = args.names, args.models.split(",") if args.models else None
    p = P.path(cfg, "checks")
    p.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(p.read_text()) if p.exists() else {}
    all_ok = True
    for name in names:
        fn = CHECKS[name]
        ok, info = fn(cfg, models) if name in ("v3", "v4", "v9") else fn(cfg)
        results[name] = {"ok": bool(ok), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "info": info}
        print(f"{name.upper()}: {'PASS' if ok else 'FAIL'}  {CHECKS[name].__doc__.strip()}")
        if not ok:
            print(json.dumps(info, indent=1, ensure_ascii=False, default=str)[:3000])
        all_ok &= bool(ok)
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
