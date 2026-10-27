"""Coleta: gerações principais (amostradas) e sondas (fatos greedy, A/B por logprob).

Uso:
  python src/generate.py                                   # perfil do config.yaml
  python src/generate.py --tag smoke --models WD --lengths 4096 --personas tobias --replicas 1

Os dois backends (hf local, vllm cluster) gravam o mesmo formato em
results[/<tag>]/generations.jsonl (main) e probes.jsonl (fact, ab).
Retoma de onde parou: itens já gravados sem erro são pulados. Falhas são gravadas com `error`.
"""
import argparse
import asyncio
import copy
import gc
import json
import time

import prepare as P

KEY = ("model", "length", "kind", "persona_id", "condition", "prompt_id", "persona_first", "replica")


def key(rec):
    return "|".join(str(rec.get(k)) for k in KEY)


def jobs(cfg, personas, models, lengths, persona_ids, replicas):
    """Um job = um turno avaliado sobre um header, em (modelo, L). Main leva todas as réplicas."""
    out = []
    for m in models:
        for L in lengths:
            for it in P.items(cfg, personas):
                if persona_ids and it["persona_id"] not in persona_ids:
                    continue
                reps = range(replicas) if it["kind"] == "main" else [0]
                seeds = [P.replica_seed(cfg, it["persona_id"], it["prompt_id"], r) if it["kind"] == "main" else None
                         for r in reps]
                out.append({"model": m["label"], "length": L, **it, "replicas": list(zip(reps, seeds))})
    return out


def records(job, results):
    """Expande um job nos registros gravados (um por réplica)."""
    base = {k: v for k, v in job.items() if k != "replicas"}
    return [{**base, "replica": r, "seed": s, **res} for (r, s), res in zip(job["replicas"], results)]


# ------------------------------------------------------------------ backend hf (local)
def hf_load(cfg, m):
    import torch
    from transformers import AutoModelForCausalLM

    h = cfg["hf"]
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], dtype=getattr(torch, h["dtype"]),
                                                 attn_implementation=h["attn_implementation"])
    return model.to(h["device"]).eval()


def hf_new_cache(model):
    from transformers import DynamicCache

    return DynamicCache(config=model.config)  # camadas sliding guardam só a janela


def hf_forward(model, ids, cache, chunk):
    """Alimenta `ids` no cache em chunks; devolve os logits (float32) do último token."""
    import torch

    with torch.no_grad():
        for s in range(0, len(ids), chunk):
            x = torch.tensor([ids[s:s + chunk]], device=model.device)
            logits = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    return logits[0, -1].float()


def hf_decode(model, tok, cache, logits, seeds, temperature, max_tokens, stop):
    """Gera len(seeds) continuações em lote (altera `cache`). Cada réplica tem seu próprio gerador."""
    import torch

    n = len(seeds)
    if n > 1:
        cache.batch_repeat_interleave(n)
    gens = [torch.Generator().manual_seed(s) for s in seeds] if temperature > 0 else None
    logits = logits.unsqueeze(0).expand(n, -1)
    toks, finish = [[] for _ in range(n)], [None] * n
    with torch.no_grad():
        for _ in range(max_tokens):
            if temperature > 0:
                probs = torch.softmax(logits / temperature, -1).cpu()
                nxt = [int(torch.multinomial(probs[r], 1, generator=gens[r])) for r in range(n)]
            else:
                nxt = logits.argmax(-1).tolist()
            for r in range(n):
                if finish[r] is None:
                    toks[r].append(nxt[r])
                    tail = tok.decode(toks[r][-8:])
                    if nxt[r] == tok.eos_token_id or any(s in tail for s in stop):
                        finish[r] = "stop"
            if all(finish):
                break
            x = torch.tensor(nxt, device=model.device).unsqueeze(1)
            logits = model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1].float()
    out = []
    for r in range(n):
        text = tok.decode(toks[r], skip_special_tokens=True)
        for s in stop:
            text = text.split(s)[0]
        out.append({"text": text, "n_gen_tokens": len(toks[r]), "finish_reason": finish[r] or "length"})
    return out


def run_hf(cfg, tok, personas, model_cfg, model_jobs, filler, write):
    import torch

    gen, chunk = cfg["generation"], cfg["hf"]["prefill_chunk"]
    opt_ids = [P.encode(tok, o)[0] for o in gen["ab_options"]]
    model = hf_load(cfg, model_cfg)
    for L in sorted({j["length"] for j in model_jobs}):
        hdr, fill, _ = P.segments(cfg, tok, personas, L, filler)
        for hid in dict.fromkeys(j["header_id"] for j in model_jobs if j["length"] == L):
            group = [j for j in model_jobs if j["length"] == L and j["header_id"] == hid]
            t0 = time.time()
            prefix = hdr[hid] + fill
            cache = hf_new_cache(model)
            hf_forward(model, prefix, cache, chunk)
            t_prefill = time.time() - t0
            for j in group:
                turn = P.encode(tok, P.turn_text(j["user"]))
                n_prompt = len(prefix) + len(turn)
                try:
                    c = copy.deepcopy(cache)
                    logits = hf_forward(model, turn, c, chunk)
                    if j["kind"] == "ab":
                        lp = torch.log_softmax(logits, -1)[opt_ids].tolist()
                        res = [{"lp_A": lp[0], "lp_B": lp[1]}]
                    else:
                        g = gen["main"] if j["kind"] == "main" else gen["facts"]
                        res = hf_decode(model, tok, c, logits, [s or 0 for _, s in j["replicas"]],
                                        g["temperature"], g["max_tokens"], g["stop"])
                    del c
                except Exception as e:  # falha é dado: grava e segue
                    res = [{"error": repr(e)}] * len(j["replicas"])
                for rec in records(j, res):
                    write({**rec, "prompt_tokens": n_prompt, "backend": "hf"})
            print(f"  {model_cfg['label']} L={L} {hid}: prefill {len(prefix)} tok em {t_prefill:.0f}s, "
                  f"{len(group)} jobs em {time.time() - t0:.0f}s", flush=True)
            del cache
            gc.collect()
            torch.mps.empty_cache() if cfg["hf"]["device"] == "mps" else None
    del model
    gc.collect()


# ------------------------------------------------------------------ backend vllm (cluster)
async def run_vllm(cfg, tok, personas, model_cfg, model_jobs, filler, write):
    import httpx

    v, gen = cfg["vllm"], cfg["generation"]
    idx = [m["label"] for m in cfg["models"]].index(model_cfg["label"])
    url, name = f"http://{v['host']}:{v['base_port'] + idx}", model_cfg["id"].split("/")[-1]
    opt_ids = [P.encode(tok, o)[0] for o in gen["ab_options"]]
    segs = {L: P.segments(cfg, tok, personas, L, filler) for L in {j["length"] for j in model_jobs}}

    async with httpx.AsyncClient(timeout=v["timeout"]) as client:
        while True:  # espera o servidor
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(10)
        sem = asyncio.Semaphore(v["concurrency"])

        async def complete(prompt, **kw):
            async with sem:
                r = await client.post(url + "/v1/completions", json={"model": name, "prompt": prompt, **kw})
                r.raise_for_status()
                return r.json()["choices"][0]

        async def one(j):
            hdr, fill, _ = segs[j["length"]]
            prompt = hdr[j["header_id"]] + fill + P.encode(tok, P.turn_text(j["user"]))
            try:
                if j["kind"] == "ab":  # logprob da opção como último token do prompt
                    chs = [await complete(prompt + [o], max_tokens=1, temperature=0, prompt_logprobs=1)
                           for o in opt_ids]
                    lps = [ch["prompt_logprobs"][-1][str(o)]["logprob"] for ch, o in zip(chs, opt_ids)]
                    res = [{"lp_A": lps[0], "lp_B": lps[1]}]
                else:
                    g = gen["main"] if j["kind"] == "main" else gen["facts"]
                    chs = await asyncio.gather(*(complete(prompt, max_tokens=g["max_tokens"], temperature=g["temperature"],
                                                          stop=g["stop"], seed=s) for _, s in j["replicas"]))
                    res = [{"text": ch["text"], "finish_reason": ch["finish_reason"]} for ch in chs]
            except Exception as e:
                res = [{"error": repr(e)}] * len(j["replicas"])
            for rec in records(j, res):
                write({**rec, "prompt_tokens": len(prompt), "backend": "vllm"})

        await asyncio.gather(*(one(j) for j in model_jobs))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="subpasta de results/ (ex.: smoke)")
    ap.add_argument("--models", default=None, help="labels separados por vírgula")
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--personas", default=None)
    ap.add_argument("--replicas", type=int, default=None)
    args = ap.parse_args()

    cfg = P.load_config()
    run, personas, tok, filler = cfg["run"], P.load_personas(cfg), P.load_tokenizer(cfg), P.load_filler(cfg)
    P.record_environment(cfg)
    models = [m for m in cfg["models"] if not args.models or m["label"] in args.models.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else run["lengths"]
    pids = args.personas.split(",") if args.personas else None
    all_jobs = jobs(cfg, personas, models, lengths, pids, args.replicas or run["replicas"])

    paths = {"main": P.out_path(cfg, "generations", args.tag), "probe": P.out_path(cfg, "probes", args.tag)}
    paths["main"].parent.mkdir(parents=True, exist_ok=True)
    done = {key(r) for p in paths.values() if p.exists()
            for r in map(json.loads, p.read_text().splitlines()) if "error" not in r}
    todo = [j for j in all_jobs if any(key(rec) not in done for rec in records(j, [{}] * len(j["replicas"])))]
    print(f"perfil={cfg['profile']} backend={run['backend']}: {len(todo)}/{len(all_jobs)} jobs a fazer", flush=True)

    files = {k: open(p, "a") for k, p in paths.items()}

    def write(rec):
        f = files["main" if rec["kind"] == "main" else "probe"]
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    for m in models:
        mj = [j for j in todo if j["model"] == m["label"]]
        if not mj:
            continue
        t0 = time.time()
        if run["backend"] == "hf":
            run_hf(cfg, tok, personas, m, mj, filler, write)
        else:
            asyncio.run(run_vllm(cfg, tok, personas, m, mj, filler, write))
        print(f"{m['label']}: {len(mj)} jobs em {(time.time() - t0) / 60:.1f} min", flush=True)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
