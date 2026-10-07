#!/usr/bin/env python
"""Exact top-K search of a full-corpus paper index for a batch of agent queries, for every encoder
that has the 58.5M-paper OpenAlex corpus embedded besides ClinicTron-BGE. (dense_clinictron_bge_search.py
handles ClinicTron-BGE and lends this script its shard reader, scan loop and Qwen3 encoder.)

  usage: dense_search.py IN.json OUT.json --encoder NAME [--topk 25]
  IN.json  = [{"sid": ..., "text": ...}, ...]
  OUT.json = {sid: [{"work_id": ..., "score": ...}, ...]}      the encoder's own metric, best first

  encoder         index (env.sh)                          query encoder
  reasonembed     AGENT_EVAL_INDEX_REASONEMBED, shards      the base model of ClinicTron-BGE, its default prompt, last token, cosine
  nvembed_v2      AGENT_EVAL_INDEX_NVEMBED, shards          NV-Embed-v2 with its native head (run under AGENT_EVAL_NVEMBED_PYTHON), cosine
  bmretriever_2b  AGENT_EVAL_DOCMAT_DIR/docmat_bmretriever  BMRetriever-2B, instruction + EOS, last token, dot product
  medcpt          AGENT_EVAL_DOCMAT_DIR/docmat_medcpt       MedCPT query tower, [CLS], dot product
  openai_3_small  AGENT_EVAL_DOCMAT_DIR/docmat_openai       text-embedding-3-small through OpenRouter, cosine

Two index layouts: "shards" (one docids.json + fp16 .raw per shard, as the ClinicTron-BGE index) and
"docmat" (one fp16 .npy matrix + one work_ids .npy, with a sidecar JSON naming the metric).

Before scanning, every encoder re-encodes indexed papers from their database text with its DOCUMENT
recipe and must reproduce the stored vectors (cosine median at or above the encoder's floor): a wrong
model, prompt, pooling or text recipe fails here, not silently in the results. One invocation = load
the model, self-check, encode all queries, one pass over the index.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import requests
import torch

import dense_clinictron_bge_search as bge
from common import env, psql_rows, secret, sql_list

HF = os.path.expanduser("~/models/hf/hub")


def snapshot(repo):
    """The local Hugging Face snapshot folder of `repo` (exactly one is expected)."""
    paths = glob.glob(os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if len(paths) != 1:
        raise RuntimeError(f"{repo}: expected one local snapshot under {HF}, found {len(paths)}")
    return paths[0]


# ---------------------------------------------------------------- encoders
# Each has encode_queries(texts) and encode_docs([(title, abstract)]) -> (N, dim) float32.

class ReasonEmbed:
    """ClinicTron-BGE's base model without the adapter, with the model's own default query prompt
    (config_sentence_transformers.json). Documents are title + " " + abstract, as the index was written."""
    PROMPT = "Instruct: Given a query, retrieve documents that can help answer the query.\nQuery: "

    def __init__(self):
        from transformers import AutoTokenizer, Qwen3Model
        base = env("CLINICTRON_BGE_BASE")
        self.model = Qwen3Model.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "right"

    def encode_queries(self, texts):
        return bge.encode(self.model, self.tok, [self.PROMPT + t for t in texts])

    def encode_docs(self, docs):
        return bge.encode(self.model, self.tok, [f"{t} {a}".strip() for t, a in docs])


class NVEmbedV2:
    """NV-Embed-v2 through the class that wrote its index (NVEmbedEncoder, imported from AGENT_EVAL_NVEMBED_CODE_DIR),
    with the same environment flags as that encode. Needs the transformers 4.42 environment
    (AGENT_EVAL_NVEMBED_PYTHON); the driver runs this script with it."""
    PROMPT = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

    def __init__(self):
        os.environ.update(NVEMBED_ATTN="sdpa", NVEMBED_SV_ONLY="1", NVEMBED_NO_TRUNCATION="1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # the module below hides the GPU unless told otherwise
        sys.path.insert(0, env("AGENT_EVAL_NVEMBED_CODE_DIR"))
        from rerank_gold1k import NVEmbedEncoder
        self.enc = NVEmbedEncoder(snapshot("nvidia/NV-Embed-v2"), "cuda", 32768, 32768, self.PROMPT)

    def _encode(self, texts, is_query, batch=16):
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            sv, _, _ = self.enc.encode3([texts[i] for i in idx], is_query=is_query)
            out[idx] = np.concatenate([v.float().cpu().numpy() for v in sv])
        return out

    def encode_queries(self, texts):
        return self._encode(texts, True)

    def encode_docs(self, docs):
        return self._encode([f"{t} {a}".strip() for t, a in docs], False)


class BMRetriever2B:
    """BMRetriever-2B: query = instruction + "\\nQuery: " + text; document = title + " " + abstract; both get the
    end-of-sequence token appended and are pooled at it; 512 tokens. The tokenizer pads on the left by default,
    which would move the pooled position: padding is set to the right. The corpus vectors agree with this
    document recipe at cosine 0.988 (the exact recipe of the corpus encode is not recorded; of nine variants
    tried on 2026-10-02 none did better), hence the 0.98 floor."""
    TASK = "Given a clinical question, retrieve relevant biomedical research passages"

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        path = snapshot("BMRetriever/BMRetriever-2B")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(path, dtype=torch.float32).to("cuda").eval()

    @torch.no_grad()
    def _encode(self, texts, batch=16):
        out = np.zeros((len(texts), 2048), dtype=np.float32)
        for s in range(0, len(texts), batch):
            ids = self.tok(texts[s:s + batch], max_length=511, truncation=True, padding=False)["input_ids"]
            enc = self.tok.pad({"input_ids": [x + [self.tok.eos_token_id] for x in ids]}, padding=True,
                               return_tensors="pt").to("cuda")
            hidden = self.model(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            out[s:s + batch] = hidden[torch.arange(len(ids), device="cuda"), last].float().cpu().numpy()
        return out

    def encode_queries(self, texts):
        return self._encode([f"{self.TASK}\nQuery: {t}" for t in texts])

    def encode_docs(self, docs):
        return self._encode([f"{t} {a}".strip() for t, a in docs])


class MedCPT:
    """MedCPT's two towers: queries through the Query-Encoder, documents through the Article-Encoder as a
    (title, abstract) pair; [CLS] vector, 512 tokens, dot product."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.q = [AutoTokenizer.from_pretrained(snapshot("ncbi/MedCPT-Query-Encoder")),
                  AutoModel.from_pretrained(snapshot("ncbi/MedCPT-Query-Encoder")).to("cuda").eval()]
        self.d = [AutoTokenizer.from_pretrained(snapshot("ncbi/MedCPT-Article-Encoder")),
                  AutoModel.from_pretrained(snapshot("ncbi/MedCPT-Article-Encoder")).to("cuda").eval()]

    @torch.no_grad()
    def _encode(self, tower, texts, pairs=None, batch=64):
        tok, model = tower
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for s in range(0, len(texts), batch):
            args = (texts[s:s + batch],) + ((pairs[s:s + batch],) if pairs else ())
            enc = tok(*args, truncation=True, padding=True, max_length=512, return_tensors="pt").to("cuda")
            out[s:s + batch] = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
        return out

    def encode_queries(self, texts):
        return self._encode(self.q, texts)

    def encode_docs(self, docs):
        return self._encode(self.d, [t for t, _ in docs], [a for _, a in docs])


class OpenAI3Small:
    """text-embedding-3-small through OpenRouter, the service that embedded the corpus. The corpus's exact
    document text is not recorded; title + blank line + abstract reproduces the stored vectors best
    (cosine about 0.99), so the self-check floor is lower for this encoder."""

    def __init__(self):
        self.key = secret("OPENROUTER_API_KEY")

    def _encode(self, texts):
        out = []
        for s in range(0, len(texts), 64):
            r = requests.post("https://openrouter.ai/api/v1/embeddings", timeout=300,
                              json={"model": "openai/text-embedding-3-small", "input": texts[s:s + 64]},
                              headers={"Authorization": f"Bearer {self.key}"}).json()
            if "data" not in r:
                raise RuntimeError(f"embeddings call failed: {str(r)[:300]}")
            out += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_queries(self, texts):
        return self._encode(texts)

    def encode_docs(self, docs):
        return self._encode([f"{t}\n\n{a}".strip() for t, a in docs])


ENCODERS = {  # name: (class, index layout, index location, vector dim, metric, self-check floor on the median cosine)
    "reasonembed": (ReasonEmbed, "shards", "AGENT_EVAL_INDEX_REASONEMBED", 4096, "cos", 0.999),
    "nvembed_v2": (NVEmbedV2, "shards", "AGENT_EVAL_INDEX_NVEMBED", 4096, "cos", 0.999),
    "bmretriever_2b": (BMRetriever2B, "docmat", "bmretriever", 2048, "dot", 0.98),
    "medcpt": (MedCPT, "docmat", "medcpt", 768, "dot", 0.99),
    "openai_3_small": (OpenAI3Small, "docmat", "openai", 1536, "cos", 0.98),
}


# ---------------------------------------------------------------- indexes
def load_docmat(name, dim):
    """(vectors as a read-only fp16 memmap, work ids) of a ctcore docmat, with the sidecar's own metric."""
    d = env("AGENT_EVAL_DOCMAT_DIR")
    V = np.load(os.path.join(d, f"docmat_{name}.npy"), mmap_mode="r")
    ids = np.load(os.path.join(d, f"work_ids_{name}.npy"), allow_pickle=True)
    with open(os.path.join(d, f"docmat_{name}.meta.json")) as fh:
        meta = json.load(fh)
    if V.shape != (len(ids), dim) or meta["dim"] != dim:
        raise RuntimeError(f"docmat {name}: vectors {V.shape}, ids {len(ids)}, dim {dim}")
    print(f"docmat {name}: {len(ids):,} rows, dim {dim}, metric {meta['metric']}, normalised {meta.get('normalized')}",
          flush=True)
    return V, ids, meta


def self_check_shards(enc, pairs, dim, floor, n=32):
    """Re-encode n papers of one shard from their database text and compare with the stored rows."""
    ids_path, raw_path = pairs[len(pairs) // 2]
    with open(ids_path) as fh:
        ids = json.load(fh)
    stored = np.memmap(raw_path, dtype=np.float16, mode="r").reshape(-1, dim)[:len(ids)]
    ends = list(range(n)) + list(range(len(ids) - n, len(ids)))  # shards are length-sorted: short and long papers
    compare(enc, [ids[i] for i in ends], stored[ends], floor, n)


def self_check_docmat(enc, V, ids, floor, n=32):
    rows = np.linspace(0, len(ids) - 1, 2 * n, dtype=np.int64)
    compare(enc, [str(ids[i]) for i in rows], V[rows], floor, n)


def compare(enc, work_ids, stored, floor, n):
    text = {w: (t, a) for w, t, a in psql_rows(
        "SELECT work_id, coalesce(title,''), coalesce(abstract,'') FROM papers_index "
        f"WHERE work_id IN ({sql_list(work_ids)})") if a}
    keep = [i for i, w in enumerate(work_ids) if w in text][:n]
    fresh = enc.encode_docs([text[work_ids[i]] for i in keep])
    ref = np.asarray(stored, dtype=np.float32)[keep]
    cos = (fresh * ref).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(ref, axis=1))
    print(f"self-check on {len(keep)} indexed papers: cosine median {np.median(cos):.5f}, min {cos.min():.5f} "
          f"(floor {floor})", flush=True)
    if len(keep) < n // 2 or np.median(cos) < floor:
        raise RuntimeError("the encoder does not reproduce the index's own vectors")


def scan_docmat(V, ids, queries, topk, as_stored, rows_per_chunk=400_000):
    """One sequential pass over the matrix -> for each query, [(work_id, score)] best first, no repeats.
    `as_stored`: score the rows as they are (dot product, or cosine over a matrix stored unit-length)."""
    Q = torch.from_numpy(queries).to("cuda")
    keep = 2 * topk
    best_s = torch.full((len(queries), keep), -1e30, device="cuda")
    best_i = torch.full((len(queries), keep), -1, dtype=torch.int64, device="cuda")
    t0 = time.time()
    for s in range(0, len(ids), rows_per_chunk):
        D = torch.from_numpy(np.ascontiguousarray(V[s:s + rows_per_chunk])).to("cuda").float()
        if not as_stored:  # cosine over a matrix not stored unit-length
            D = torch.nn.functional.normalize(D, dim=-1)
        sc, i = torch.topk(Q @ D.T, min(keep, len(D)), dim=1)
        best_s, pos = torch.topk(torch.cat([best_s, sc], dim=1), keep, dim=1)
        best_i = torch.gather(torch.cat([best_i, i + s], dim=1), 1, pos)
        if (s // rows_per_chunk) % 25 == 0 or s + rows_per_chunk >= len(ids):
            print(f"  {min(s + rows_per_chunk, len(ids)):,}/{len(ids):,} rows, {time.time() - t0:.0f}s", flush=True)
    out = []
    for row_s, row_i in zip(best_s.cpu().tolist(), best_i.cpu().tolist()):
        seen, hits = set(), []
        for score, j in zip(row_s, row_i):
            if j >= 0 and ids[j] not in seen:
                seen.add(ids[j])
                hits.append({"work_id": str(ids[j]), "score": score})
        out.append(hits[:topk])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--topk", type=int, default=25)
    a = ap.parse_args()
    with open(a.inp) as fh:
        rows = json.load(fh)
    cls, layout, where, dim, metric, floor = ENCODERS[a.encoder]

    bge.wait_for_free_ram(float(os.environ.get("MULTI_TURN_MIN_FREE_RAM_GB", 0)))
    enc = cls()
    if layout == "shards":
        pairs = bge.shard_pairs(env(where))
        self_check_shards(enc, pairs, dim, floor)
    else:
        V, ids, meta = load_docmat(where, dim)
        if meta["metric"] != metric:
            raise RuntimeError(f"docmat metric {meta['metric']} != {metric}")
        self_check_docmat(enc, V, ids, floor)
    queries = enc.encode_queries([r["text"] for r in rows])
    if metric == "cos":
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    del enc
    torch.cuda.empty_cache()
    print(f"{a.encoder}: encoded {len(rows)} queries", flush=True)

    if layout == "shards":
        hits = bge.scan(pairs, queries, a.topk)  # cosine: the scan normalises every stored row
    else:
        hits = scan_docmat(V, ids, queries, a.topk, as_stored=(metric == "dot" or bool(meta.get("normalized"))))
    with open(a.outp + ".tmp", "w") as fh:
        json.dump({str(r["sid"]): h for r, h in zip(rows, hits)}, fh)
    os.replace(a.outp + ".tmp", a.outp)
    print(f"-> {a.outp}", flush=True)


if __name__ == "__main__":
    main()
