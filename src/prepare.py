"""Prepara os dados: filler fixo (UltraChat) e prompts em token ids.

Uso: python src/prepare.py [--force]    (--force reconstrói data/filler.jsonl)

Prompt = HEADER da condição + FILLER (turnos User/Assistant) + TURNO avaliado, montado em
token ids por segmento. O filler de cada comprimento L é o maior prefixo de turnos que cabe
no orçamento do pior caso (maior header + maior turno + max_tokens), então é idêntico entre
condições/modelos e aninhado entre comprimentos.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import re
import zlib
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN_CONDITIONS = ["persona", "neutral", "sham", "distractor"]
AB_CONDITIONS = MAIN_CONDITIONS + ["contrast"]


# ------------------------------------------------------------------ config / manifest
def load_config(profile=None):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["profile"] = profile or os.environ.get("PROFILE") or cfg["profile"]
    cfg["run"] = cfg["profiles"][cfg["profile"]]
    return cfg


def path(cfg, key):
    return ROOT / cfg["paths"][key]


def out_path(cfg, key, tag=None):
    """Arquivo de resultados; com `tag` (ex.: smoke) vai para results/<tag>/."""
    p = path(cfg, key)
    return p.parent / tag / p.name if tag else p


def load_personas(cfg):
    return yaml.safe_load(path(cfg, "personas").read_text())


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def update_manifest(cfg, key, value):
    p = path(cfg, "manifest")
    p.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(p.read_text()) if p.exists() else {}
    m[key] = value
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def record_environment(cfg):
    libs = ["torch", "transformers", "huggingface_hub", "tokenizers", "numpy", "pandas", "scipy", "httpx"]
    update_manifest(cfg, "environment", {
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": {lib: version(lib) for lib in libs}})
    update_manifest(cfg, "models", [{k: m[k] for k in ("id", "label", "revision")} for m in cfg["models"]])
    update_manifest(cfg, "embeddings", {k: cfg["embeddings"][k] for k in ("model", "revision")})
    update_manifest(cfg, "nli", {k: cfg["nli"][k] for k in ("model", "revision")})
    update_manifest(cfg, "seed", cfg["seed"])
    update_manifest(cfg, "personas_sha256", sha256_file(path(cfg, "personas")))


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    m = cfg["models"][0]  # V1 garante que os 6 modelos têm o mesmo tokenizer
    return AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], clean_up_tokenization_spaces=False)


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


# ------------------------------------------------------------------ filler
def exchange_text(user, assistant):
    return f"User: {user}\nAssistant: {assistant}\n"


def contamination_regex(personas):
    terms = set()
    for p in personas["personas"]:
        terms.update(p["name"].split())
        terms.update(p.get("filter_terms", []))
        for f in p["facts"].values():
            terms.add(f["answer"])
            terms.update(f["aliases"])
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms)) + ")", re.I)


def disclosure_regex(cfg):
    return re.compile("|".join(cfg["filler"]["self_disclosure_regex"]), re.I | re.M)


def build_filler(cfg, personas, tok):
    """Shuffle fixo do UltraChat; de cada diálogo, mantém o prefixo de turnos válidos."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    fc = cfg["filler"]
    local = hf_hub_download(fc["dataset"], fc["file"], repo_type="dataset", revision=fc["revision"])
    rows = pq.read_table(local, columns=["prompt_id", "messages"]).to_pylist()
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)

    disclosure, contaminated = disclosure_regex(cfg), contamination_regex(personas)
    seen, out, total, n_dialogs = set(), [], 0, 0
    for i in order:
        msgs, kept = rows[i]["messages"], 0
        for k in range(0, len(msgs) - 1, 2):
            u, a = msgs[k], msgs[k + 1]
            user, asst = u["content"].strip(), a["content"].strip()
            key = hashlib.sha1(" ".join(user.lower().split()).encode()).hexdigest()
            text = user + "\n" + asst
            if ((u["role"], a["role"]) != ("user", "assistant") or not user or not asst or key in seen
                    or disclosure.search(text) or contaminated.search(text)
                    or max(len(encode(tok, user)), len(encode(tok, asst))) > fc["max_turn_tokens"]):
                break  # o resto do diálogo é descartado (mantém coerência)
            seen.add(key)
            n = len(encode(tok, exchange_text(user, asst)))
            out.append({"i": len(out), "source_id": rows[i]["prompt_id"], "turn": k // 2,
                        "user": user, "assistant": asst, "n_tokens": n})
            total, kept = total + n, kept + 1
        n_dialogs += kept > 0
        if total >= fc["target_tokens"]:
            break

    p = path(cfg, "filler")
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    update_manifest(cfg, "filler", {
        "dataset": fc["dataset"], "revision": fc["revision"], "file": fc["file"], "seed": cfg["seed"],
        "sha256": sha256_file(p), "n_exchanges": len(out), "n_dialogs": n_dialogs, "n_tokens": total})
    return out


def load_filler(cfg):
    return [json.loads(line) for line in path(cfg, "filler").read_text().splitlines()]


# ------------------------------------------------------------------ prompts
def turn_text(user):
    return f"User: {user}\nAssistant:"


def ab_user(cp, persona_first):
    a, b = (cp["persona_option"], cp["contrast_option"])[:: 1 if persona_first else -1]
    return f"{cp['question']}\nA) {a}\nB) {b}\nAnswer with A or B."


def headers(personas):
    """header_id → texto. Inclui um header de contraste por par A/B."""
    def persona_text(p, swap=None):
        comps = {**p["components"], **(swap or {})}
        intro = (f"{personas['frame']} The Assistant is {p['name']} and answers every message "
                 f"in character as {p['first_name']}.")
        return " ".join([intro] + [comps[c] for c in personas["component_order"]])

    out = {"neutral": personas["neutral_header"], "sham": personas["sham_header"]}
    for p in personas["personas"]:
        out[f"persona:{p['id']}"] = persona_text(p)
        for cp in p["contrast_pairs"]:
            out[f"contrast:{p['id']}:{cp['id']}"] = persona_text(p, {cp["component"]: cp["contrast_text"]})
    return out


def header_id(personas, persona_id, condition, pair_id=None):
    ids = [p["id"] for p in personas["personas"]]
    if condition == "persona":
        return f"persona:{persona_id}"
    if condition == "distractor":
        return f"persona:{ids[(ids.index(persona_id) + 1) % len(ids)]}"
    if condition == "contrast":
        return f"contrast:{persona_id}:{pair_id}"
    return condition  # neutral, sham


def items(cfg, personas):
    """Tudo o que é avaliado por comprimento: main (amostrado), fact (greedy) e ab (logprob)."""
    out = []
    nps = personas["natural_prompts"][: cfg["run"]["n_natural_prompts"]]
    for p in personas["personas"]:
        base = {"persona_id": p["id"]}
        for np_ in nps:
            out += [{**base, "kind": "main", "condition": c, "prompt_id": np_["id"],
                     "component": np_["component"], "user": np_["text"]} for c in MAIN_CONDITIONS]
        for slot, q in personas["fact_questions"].items():
            out += [{**base, "kind": "fact", "condition": c, "prompt_id": slot, "user": q}
                    for c in MAIN_CONDITIONS]
        for cp in p["contrast_pairs"]:
            out += [{**base, "kind": "ab", "condition": c, "prompt_id": cp["id"], "component": cp["component"],
                     "persona_first": pf, "user": ab_user(cp, pf)}
                    for c in AB_CONDITIONS for pf in (True, False)]
    for it in out:
        it["header_id"] = header_id(personas, it["persona_id"], it["condition"], it["prompt_id"])
    return out


def all_user_texts(personas):
    """Todos os turnos possíveis (todos os perfis), para o orçamento do pior caso."""
    texts = [np_["text"] for np_ in personas["natural_prompts"]] + list(personas["fact_questions"].values())
    for p in personas["personas"]:
        texts += [ab_user(cp, pf) for cp in p["contrast_pairs"] for pf in (True, False)]
    return texts


def segments(cfg, tok, personas, L, filler):
    """Token ids para o comprimento L: {header_id: ids}, filler_ids, nº de turnos do filler."""
    hdr = {k: encode(tok, v + "\n\n") for k, v in headers(personas).items()}
    max_turn = max(len(encode(tok, turn_text(u))) for u in all_user_texts(personas)) + 1  # +1: opção A/B
    budget = L - max(map(len, hdr.values())) - max_turn - cfg["generation"]["main"]["max_tokens"]
    ids, n = [], 0
    for ex in filler:
        piece = encode(tok, exchange_text(ex["user"], ex["assistant"]))
        if len(ids) + len(piece) > budget:
            break
        ids, n = ids + piece, n + 1
    if n == len(filler):
        raise ValueError(f"filler curto demais para L={L}: aumente filler.target_tokens")
    return hdr, ids, n


def replica_seed(cfg, persona_id, prompt_id, replica):
    """Mesma seed em todas as condições, modelos e comprimentos (pareamento persona × neutra)."""
    return zlib.crc32(f"{cfg['seed']}|{persona_id}|{prompt_id}|{replica}".encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconstrói o filler")
    args = ap.parse_args()
    cfg = load_config()
    personas, tok = load_personas(cfg), load_tokenizer(cfg)
    record_environment(cfg)

    if args.force or not path(cfg, "filler").exists():
        filler = build_filler(cfg, personas, tok)
    else:
        filler = load_filler(cfg)
    print(f"filler: {len(filler)} turnos, {sum(r['n_tokens'] for r in filler)} tokens, "
          f"sha256={sha256_file(path(cfg, 'filler'))[:12]}")

    summary = {}
    for L in cfg["profiles"]["cluster"]["lengths"]:  # o filler é o mesmo para os dois perfis
        hdr, fill, n = segments(cfg, tok, personas, L, filler)
        summary[L] = {"filler_tokens": len(fill), "filler_exchanges": n,
                      "header_tokens": {"min": min(map(len, hdr.values())), "max": max(map(len, hdr.values()))}}
        print(f"L={L:>6}: filler={len(fill):>6} tokens ({n} turnos), header {summary[L]['header_tokens']}")
    update_manifest(cfg, "segments", summary)
    n_items = len(items(cfg, personas))
    print(f"perfil {cfg['profile']}: {n_items} itens por modelo × comprimento")


if __name__ == "__main__":
    main()
