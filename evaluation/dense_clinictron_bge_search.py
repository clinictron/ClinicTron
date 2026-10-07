#!/usr/bin/env python
"""Exact top-K search of the ClinicTron-BGE paper index for a batch of agent queries.

ClinicTron-BGE is the base model `hanhainebula/reason-embed-qwen3-8b-0928` plus the UNMERGED
fine-tuned LoRA adapter (`adapter_step_200`). The index holds one 4096-dim vector per paper,
written by that same model. Queries are encoded here exactly as the documents were (same
weights, tokenizer, end-of-text handling, last-token pooling, bf16), and before any scan the
script refuses to run unless the adapter on disk is the one named in the index's own shard
metadata, and unless re-encoding a few indexed papers reproduces their stored vectors.

One invocation = load the model, self-check, encode all queries, ONE pass over every shard.

  usage: dense_clinictron_bge_search.py IN.json OUT.json [--topk 25]
  IN.json  = [{"sid": ..., "text": ...}, ...]
  OUT.json = {sid: [{"work_id": ..., "score": ...}, ...]}      cosine, best first
"""
import argparse
import glob
import hashlib
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import env, psql_rows, sql_list

DIM = 4096
MAX_LEN = 8192
# The instruction every biomedical row of the ClinicTron-BGE fine-tune was trained with
# (train_config.json "instruction"), wrapped the way the trainer wrapped it.
INSTRUCTION = "Given a query, retrieve documents that can help answer the query"
QUERY_PREFIX = f"Instruct: {INSTRUCTION}\nQuery: "
SHARD_RE = re.compile(r"([A-Za-z_]*?)(\d+)_of_(\d+)__docids\.json$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- index
def shard_pairs(index_dir):
    """[(docids.json, vectors.raw)] for a COMPLETE index; a missing shard would make its papers
    silently unrankable, so every `<n>_of_<total>` family must be whole."""
    pairs, families = [], {}
    for ids in sorted(glob.glob(os.path.join(index_dir, "*__docids.json"))):
        raw = ids.replace("__docids.json", "__docvecs_fp16.raw")
        if not os.path.exists(raw):
            raise RuntimeError(f"id file without vectors: {ids}")
        pairs.append((ids, raw))
        m = SHARD_RE.search(os.path.basename(ids))
        if m:
            families.setdefault((m.group(1), int(m.group(3))), set()).add(int(m.group(2)))
    if not families:
        raise RuntimeError(f"no '<n>_of_<total>' shards under {index_dir}")
    for (name, total), have in families.items():
        if len(have) != total:
            raise RuntimeError(f"index incomplete: {name} has {len(have)} of {total} shards")
    return pairs


def check_index_matches_model(pairs, adapter_dir, base_dir):
    """Every shard must say it was written by this base revision and this adapter."""
    adapter_sha = sha256_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    base_revision = os.path.basename(os.path.normpath(base_dir))
    unlabelled = 0
    for _ids, raw in pairs:
        with open(raw.replace(".raw", ".meta.json")) as fh:
            meta = json.load(fh)
        if (meta["sv_pool"], meta["append_eos"], meta["normalised"], meta["doc_prefix"]) != \
                ("last", True, True, ""):
            raise RuntimeError(f"{raw}: not the last-token / EOS / normalised surface")
        if "adapter_sha256" not in meta:
            unlabelled += 1
        elif (meta["adapter_sha256"], meta["base_revision"]) != (adapter_sha, base_revision):
            raise RuntimeError(f"{raw} was written by another model: adapter "
                               f"{meta['adapter_sha256'][:12]}, base {meta['base_revision'][:8]}")
    print(f"index matches adapter {adapter_sha[:12]} on base {base_revision[:8]}: "
          f"{len(pairs) - unlabelled} shards confirmed, {unlabelled} carry no model label",
          flush=True)


# ---------------------------------------------------------------- model
def load_model(base_dir, adapter_dir):
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3Model
    model, info = Qwen3Model.from_pretrained(base_dir, dtype=torch.bfloat16,
                                             attn_implementation="sdpa", output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"]:
        raise RuntimeError(f"base model did not load cleanly: {info['missing_keys'][:3]} "
                           f"{info['unexpected_keys'][:3]}")
    model = PeftModel.from_pretrained(model, adapter_dir).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    return model, tok


@torch.no_grad()
def encode(model, tok, texts, batch_size=8):
    """(N, 4096) float32, L2-normalised. Each text gets the tokenizer's end-of-sequence string
    appended, as the documents did, and the last real token's hidden state is the vector."""
    texts = [t + tok.eos_token for t in texts]
    lengths = [len(x) for x in tok(texts, truncation=False, return_token_type_ids=False)["input_ids"]]
    if max(lengths) > MAX_LEN:
        raise RuntimeError(f"a text is {max(lengths)} tokens; over {MAX_LEN} it would be truncated")
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], max_length=MAX_LEN, padding=True, truncation=True,
                  return_token_type_ids=False, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           use_cache=False).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        vec = hidden[torch.arange(len(idx), device="cuda"), last].float()
        out[idx] = F.normalize(vec, dim=-1).cpu().numpy()
    return out


def check_encoder_reproduces_index(model, tok, pairs, n=32):
    """Re-encode n indexed papers from their database text and compare with the stored vectors.
    A wrong adapter, tokenizer setting or pooling fails this; a few papers whose database text
    has changed since the index was written do not (the test is on the median)."""
    ids_path, raw_path = pairs[len(pairs) // 2]
    with open(ids_path) as fh:
        ids = json.load(fh)
    stored = np.fromfile(raw_path, dtype=np.float16).reshape(len(ids), DIM)
    ends = list(range(n)) + list(range(len(ids) - n, len(ids)))  # shards are length-sorted
    rows = psql_rows("SELECT work_id, coalesce(title,''), coalesce(abstract,'') FROM papers_index "
                     f"WHERE work_id IN ({sql_list(ids[i] for i in ends)})")
    text = {w: f"{title} {abstract}".strip() for w, title, abstract in rows}
    keep = sorted((i for i in ends if text.get(ids[i])), key=lambda i: len(text[ids[i]]))[:n]
    fresh = encode(model, tok, [text[ids[i]] for i in keep])
    ref = stored[keep].astype(np.float32)
    cos = (fresh * ref).sum(1) / np.linalg.norm(ref, axis=1)
    print(f"self-check on {len(keep)} indexed papers: cosine median {np.median(cos):.5f}, "
          f"min {cos.min():.5f}", flush=True)
    if len(keep) < n // 2 or np.median(cos) < 0.999:
        raise RuntimeError("the query encoder does not reproduce the index's own vectors")


# ---------------------------------------------------------------- scan
def scan(pairs, queries, topk):
    """One sequential pass -> for each query row, [(work_id, cosine)] best first, no repeats.
    The index stores some papers twice, so 2 x topk candidates are kept and repeats dropped."""
    Q = torch.from_numpy(queries).to("cuda")
    keep = 2 * topk
    best_s = torch.full((len(queries), keep), -2.0, device="cuda")
    best_i = torch.full((len(queries), keep), -1, dtype=torch.int64, device="cuda")
    docids, t0 = [], time.time()
    for si, (ids_path, raw_path) in enumerate(pairs, 1):
        with open(ids_path) as fh:
            ids = json.load(fh)
        V = np.fromfile(raw_path, dtype=np.float16)
        if V.size != len(ids) * DIM:  # misaligned shard: every later row would get the wrong id
            raise RuntimeError(f"{raw_path}: {V.size / DIM} vectors for {len(ids)} ids")
        D = F.normalize(torch.from_numpy(V.reshape(len(ids), DIM)).to("cuda").float(), dim=-1)
        s, i = torch.topk(Q @ D.T, min(keep, len(ids)), dim=1)
        best_s, pos = torch.topk(torch.cat([best_s, s], dim=1), keep, dim=1)
        best_i = torch.gather(torch.cat([best_i, i + len(docids)], dim=1), 1, pos)
        docids.extend(ids)
        if si % 200 == 0 or si == len(pairs):
            print(f"  {si}/{len(pairs)} shards, {len(docids):,} rows, {time.time() - t0:.0f}s",
                  flush=True)
    out = []
    for row_s, row_i in zip(best_s.cpu().tolist(), best_i.cpu().tolist()):
        seen, hits = set(), []
        for score, j in zip(row_s, row_i):
            if j >= 0 and docids[j] not in seen:
                seen.add(docids[j])
                hits.append({"work_id": docids[j], "score": score})
        out.append(hits[:topk])
    return out


def wait_for_free_ram(min_gb):
    """Loading the model and the 60M ids needs ~25 GB of RAM; on a shared box wait for it rather
    than push the machine into its out-of-memory killer."""
    while True:
        with open("/proc/meminfo") as fh:
            free_gb = int(next(l for l in fh if l.startswith("MemAvailable")).split()[1]) / 1e6
        if free_gb >= min_gb:
            return
        print(f"waiting for RAM: {free_gb:.0f} GB available, need {min_gb:.0f}", flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    ap.add_argument("--topk", type=int, default=25)
    a = ap.parse_args()
    with open(a.inp) as fh:
        rows = json.load(fh)
    base_dir, adapter_dir = env("CLINICTRON_BGE_BASE"), env("CLINICTRON_BGE_ADAPTER")

    wait_for_free_ram(float(os.environ.get("MULTI_TURN_MIN_FREE_RAM_GB", 0)))
    pairs = shard_pairs(env("CLINICTRON_BGE_INDEX_DIR"))
    check_index_matches_model(pairs, adapter_dir, base_dir)
    model, tok = load_model(base_dir, adapter_dir)
    check_encoder_reproduces_index(model, tok, pairs)
    queries = encode(model, tok, [QUERY_PREFIX + r["text"] for r in rows])
    del model
    torch.cuda.empty_cache()
    print(f"encoded {len(rows)} queries", flush=True)

    hits = scan(pairs, queries, a.topk)
    with open(a.outp + ".tmp", "w") as fh:
        json.dump({str(r["sid"]): h for r, h in zip(rows, hits)}, fh)
    os.replace(a.outp + ".tmp", a.outp)  # the driver treats an existing OUT.json as a finished scan
    print(f"-> {a.outp}", flush=True)


if __name__ == "__main__":
    main()
