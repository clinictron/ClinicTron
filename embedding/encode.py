"""Encode a retrieval corpus and its queries with one embedding model.

    python encode.py --model configs/models/reasonembed.yaml \
                     --benchmark configs/benchmarks/r2med.yaml --task r2med_biology \
                     --out bank/r2med/ReasonEmbed/r2med_biology.npz

Every setting that changes a vector comes from the two config files: the model config fixes
the model's own input format and pooling, the benchmark config fixes lengths, precision and
instructions. The output .npz holds document and query vectors, their ids, and the full
resolved configuration, so any score can be recomputed from the file alone (see score.py).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer, MistralConfig, MistralModel

# --------------------------------------------------------------------------- configuration


def load_config(model_path, bench_path, task):
    m = yaml.safe_load(open(model_path))
    b = yaml.safe_load(open(bench_path))
    if task not in b["tasks"]:
        raise KeyError(f"task {task!r} not in {bench_path}")
    t = b["tasks"][task] or {}
    m = {**m, **b.get("model_overrides", {}).get(m["name"], {})}
    doc_len = m.get("max_length", b["max_length"])
    query_len = doc_len
    cap = m.get("max_length_cap")
    if cap:
        doc_len, query_len = min(doc_len, cap), min(query_len, cap)
    iset = m.get("instruction_set", "default")
    instr = t.get("instructions", {}).get(iset, t.get("instruction", ""))
    doc_instr = t.get("doc_instructions", {}).get(iset, "")
    if instr and m.get("instruction_suffix") and not instr.endswith(m["instruction_suffix"]):
        instr += m["instruction_suffix"]
    return {
        "model": m, "benchmark": b["name"], "task": task,
        "data_dir": b["data"].format(task=t.get("dir", task)),
        "dtype": m.get("dtype") or b["dtype"],
        "doc_max_length": int(doc_len), "query_max_length": int(query_len),
        "instruction": instr if m.get("use_instruction", True) else "",
        "doc_instruction": doc_instr,
        "ignore_identical_ids": bool(t.get("ignore_identical_ids", False)),
    }


# --------------------------------------------------------------------------- data


def read_task(data_dir, title_join):
    """corpus.parquet (_id, title, text) and queries.parquet (_id, text). Duplicate corpus ids
    keep their first occurrence."""
    c = pq.read_table(os.path.join(data_dir, "corpus.parquet")).to_pydict()
    titles = c.get("title") or [""] * len(c["_id"])
    ids, docs, seen = [], [], set()
    for i, ti, tx in zip(c["_id"], titles, c["text"]):
        i = str(i)
        if i in seen:
            continue
        seen.add(i)
        ids.append(i)
        ti, tx = (ti or ""), (tx or "")
        docs.append((ti.strip(), tx.strip()) if title_join == "pair" else f"{ti} {tx}".strip())
    q = pq.read_table(os.path.join(data_dir, "queries.parquet")).to_pydict()
    return ids, docs, [str(x) for x in q["_id"]], [str(x) for x in q["text"]]


# --------------------------------------------------------------------------- models


class BidirectionalMistral(MistralModel):
    """Mistral with full (non-causal) attention, as NV-Embed-v2 uses it: the attention mask
    hides padding keys only, with no causal triangle and no sliding window."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def forward(self, input_ids=None, attention_mask=None, **kw):
        emb = self.embed_tokens(input_ids)
        b, n = input_ids.shape
        keep = attention_mask[:, None, None, :].bool().expand(b, 1, n, n)
        if self.config._attn_implementation == "eager":
            mask = torch.zeros(keep.shape, dtype=emb.dtype, device=emb.device)
            mask = mask.masked_fill(~keep, torch.finfo(emb.dtype).min)
        else:
            mask = keep
        return super().forward(inputs_embeds=emb, attention_mask=mask, use_cache=False, **kw)


class LatentAttentionPool(torch.nn.Module):
    """NV-Embed-v2's pooling head (Lee et al. 2024): every token attends to 512 trained latent
    vectors, passes a feed-forward block, and the masked mean is L2-normalised. Parameter names
    match the released checkpoint's `latent_attention_model.*` tensors."""

    def __init__(self, dim=4096, heads=8, dim_head=4096, latents=512):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.latents = torch.nn.Parameter(torch.zeros(latents, dim))
        attn = torch.nn.Module()
        attn.norm, attn.norm_context = torch.nn.LayerNorm(dim), torch.nn.LayerNorm(dim)
        attn.fn = torch.nn.Module()
        attn.fn.to_q = torch.nn.Linear(dim, inner, bias=False)
        attn.fn.to_kv = torch.nn.Linear(dim, inner * 2, bias=False)
        attn.fn.to_out = torch.nn.Linear(inner, dim, bias=False)
        ff = torch.nn.Module()
        ff.norm = torch.nn.LayerNorm(dim)
        ff.fn = torch.nn.Module()
        ff.fn.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * 8), torch.nn.Identity(),
                                        torch.nn.Linear(dim * 4, dim))
        self.cross_attend_blocks = torch.nn.ModuleList([attn, ff])

    def forward(self, hidden, pool_mask):
        attn, ff = self.cross_attend_blocks
        b, h = hidden.shape[0], self.heads
        x = attn.norm(hidden)
        # the latents are shared by every text, so project them once rather than once per text
        k, v = attn.fn.to_kv(attn.norm_context(self.latents)).chunk(2, dim=-1)
        n = hidden.shape[1]
        q = attn.fn.to_q(x).reshape(b * n, h, -1).transpose(0, 1)            # (h, b*n, d): texts share one matmul
        k, v = (t.reshape(t.shape[0], h, -1).transpose(0, 1) for t in (k, v))  # (h, latents, d)
        w = torch.softmax((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5, dim=-1)
        o = (w.to(v.dtype) @ v).transpose(0, 1).reshape(b, n, -1)
        hidden = attn.fn.to_out(o) + hidden
        y = ff.fn.net[0](ff.norm(hidden))
        y, gates = y.chunk(2, dim=-1)
        hidden = ff.fn.net[2](y * F.gelu(gates)) + hidden
        m = pool_mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / pool_mask.sum(1, keepdim=True).float()


def _nvembed_parts(path, dtype, attn_impl):
    """Load NV-Embed-v2's backbone (and its latent head) straight from the safetensors shards.
    The released remote code does not load under current transformers."""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(path, "config.json")))
    tcfg = MistralConfig(**cfg["text_config"])
    tcfg._attn_implementation = attn_impl
    sd = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(shard))
    backbone = {k[len("embedding_model."):]: v for k, v in sd.items() if k.startswith("embedding_model.")}
    head = {k[len("latent_attention_model."):]: v for k, v in sd.items()
            if k.startswith("latent_attention_model.")}
    with torch.device("meta"):
        model = BidirectionalMistral(tcfg)
    missing, unexpected = model.load_state_dict(backbone, strict=False, assign=True)
    if [k for k in missing if not k.endswith("inv_freq")] or unexpected:
        raise RuntimeError(f"NV-Embed backbone load: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.rotary_emb = type(model.rotary_emb)(config=tcfg)
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta:
        raise RuntimeError(f"NV-Embed backbone left unloaded tensors: {meta[:5]}")
    return model.to(dtype), head


def fetch(spec, root=""):
    """A local directory, a path relative to `root`, or `org/name@revision` on the Hugging Face Hub."""
    if spec is None or os.path.isdir(spec):
        return spec
    if root and os.path.isdir(os.path.join(root, spec)):
        return os.path.join(root, spec)
    from huggingface_hub import snapshot_download
    repo, _, rev = spec.partition("@")
    return snapshot_download(repo, revision=rev or None)


def load_model(m, dtype, device, weights_root=""):
    m = dict(m)
    for k in ("path", "query_path", "doc_path", "tokenizer", "adapter"):
        if m.get(k):
            m[k] = fetch(m[k], weights_root)
    kind = m["kind"]
    attn_impl = m.get("attn_implementation", "sdpa")
    head = None
    if kind == "dual_bert":
        q = AutoModel.from_pretrained(m["query_path"], dtype=dtype).to(device).eval()
        d = AutoModel.from_pretrained(m["doc_path"], dtype=dtype).to(device).eval()
        tq, td = AutoTokenizer.from_pretrained(m["query_path"]), AutoTokenizer.from_pretrained(m["doc_path"])
        if tq.get_vocab() != td.get_vocab():
            raise RuntimeError("query and document towers do not share a vocabulary")
        return {"query": q, "doc": d}, tq, None
    if kind == "nvembed":
        backbone, head_sd = _nvembed_parts(m["path"], dtype, attn_impl)
        if m["pooling"] == "latent":
            head = LatentAttentionPool()
            head.load_state_dict(head_sd, strict=True)
            head = head.to(device=device, dtype=dtype).eval()
    elif kind == "decoder":
        backbone, info = AutoModel.from_pretrained(m["path"], dtype=dtype, attn_implementation=attn_impl,
                                                   output_loading_info=True)
        bad = list(info["missing_keys"]) + [k for k in info["unexpected_keys"] if not k.endswith("lm_head.weight")]
        if bad:
            raise RuntimeError(f"{m['path']}: incomplete load, e.g. {bad[:5]}")
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    if m.get("adapter"):
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, m["adapter"])
    tok = AutoTokenizer.from_pretrained(m.get("tokenizer", m["path"]))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    backbone = backbone.to(device).eval()
    return {"query": backbone, "doc": backbone}, tok, head


# --------------------------------------------------------------------------- encoding


def tokenize(tok, texts, max_len, eos, pair=False):
    if pair:
        return tok([a for a, _ in texts], [b for _, b in texts], max_length=max_len, truncation=True,
                   padding=True, return_tensors="pt")
    if eos == "after_truncation":
        enc = tok(texts, max_length=max_len - 1, truncation=True, return_token_type_ids=False)
        enc["input_ids"] = [x + [tok.eos_token_id] for x in enc["input_ids"]]
        enc["attention_mask"] = [x + [1] for x in enc["attention_mask"]]
        return tok.pad(enc, padding=True, return_tensors="pt")
    if eos == "append":
        texts = [t + tok.eos_token for t in texts]
    return tok(texts, max_length=max_len, truncation=True, padding=True, return_token_type_ids=False,
               return_tensors="pt")


def true_lengths(tok, texts, eos, pair):
    if pair:
        enc = tok([a for a, _ in texts], [b for _, b in texts], truncation=False)
    else:
        enc = tok([t + tok.eos_token if eos in ("append", "after_truncation") else t for t in texts],
                  truncation=False, return_token_type_ids=False)
    return np.array([len(x) for x in enc["input_ids"]])


def pool(hidden, attn_mask, pool_mask, how, head):
    if how == "last":
        last = attn_mask.sum(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    if how == "cls":
        return hidden[:, 0].float()
    if how == "latent":
        return head(hidden, pool_mask).float()
    if how == "mean_tokennorm":  # each token L2-normalised before the mean, as the corpus encodes pooled
        m = pool_mask.unsqueeze(-1).float()
        return (F.normalize(hidden.float(), dim=-1) * m).sum(1) / m.sum(1).clamp(min=1)
    if how == "mean":
        m = pool_mask.unsqueeze(-1).float()
        return (hidden.float() * m).sum(1) / m.sum(1).clamp(min=1)
    raise ValueError(f"unknown pooling {how!r}")


def plan_batches(order, lengths, max_len, batch_size, max_tokens):
    """Longest texts first. A batch closes at `batch_size` texts or, with `max_tokens`, when its padded
    size (count x longest length) would pass the token budget, so short texts share large batches."""
    out, cur, width = [], [], 0
    for i in order:
        n = min(int(lengths[i]), max_len)
        if cur and (len(cur) >= batch_size or (max_tokens and (len(cur) + 1) * width > max_tokens)):
            out.append(cur)
            cur = []
        if not cur:
            width = n
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def encode(texts, side, models, tok, head, m, max_len, prefix_tokens, dtype, device, batch_size, max_tokens=0):
    pair = m.get("title_join") == "pair" and side == "doc"
    eos = m.get("eos", "none")
    lengths = true_lengths(tok, texts, eos, pair)
    order = np.argsort(-lengths, kind="stable")
    vecs = None                     # (n, dim) float32, allocated once the first batch shows the dimension
    fallback = []
    model = models[side]
    use_autocast = device.startswith("cuda") and dtype != torch.float32
    plan = plan_batches(order, lengths, max_len, batch_size, max_tokens)
    ahead = ThreadPoolExecutor(1)   # tokenises the next batches while the GPU runs this one
    todo = [ahead.submit(tokenize, tok, [texts[i] for i in idx], max_len, eos, pair) for idx in plan[:4]]
    for k, idx in enumerate(plan):
        enc = todo[k].result()
        if k % 200 == 0:
            print(f"{side}: batch {k}/{len(plan)}", flush=True)
        if k + 4 < len(plan):
            todo.append(ahead.submit(tokenize, tok, [texts[i] for i in plan[k + 4]], max_len, eos, pair))
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        pm = am.clone()
        if prefix_tokens:
            pm[:, :prefix_tokens] = 0
        kw = {"token_type_ids": enc["token_type_ids"].to(device)} if pair else {}
        with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
            hidden = model(input_ids=ids, attention_mask=am, **kw).last_hidden_state
            v = pool(hidden, am, pm, m["pooling"], head)
        bad = ~torch.isfinite(v).all(1)
        if bad.any() and dtype == torch.float16:
            # fp16 overflows on a handful of texts; re-run only those rows in bfloat16 and record them
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sub = {k: t[bad] for k, t in kw.items()}
                h = model(input_ids=ids[bad], attention_mask=am[bad], **sub).last_hidden_state
                v[bad] = pool(h, am[bad], pm[bad], m["pooling"], head).to(v.dtype)
            fallback.extend(int(idx[j]) for j in bad.nonzero().flatten().tolist())
        if m.get("normalize", True):
            v = F.normalize(v, dim=-1)
        if vecs is None:
            vecs = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        vecs[idx] = v.float().cpu().numpy()
    ahead.shutdown()
    stats = {"n": len(texts), "max_length": max_len, "truncated": int((lengths > max_len).sum()),
             "max_true_length": int(lengths.max()) if len(lengths) else 0,
             "bf16_fallback_rows": sorted(fallback)}
    return vecs, stats


def fmt(template, instruction, text):
    return template.replace("{instruction}", instruction).replace("{text}", text)


def sha256_dir(path, pattern):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(path, pattern))):
        h.update(open(f, "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16, help="most texts per batch")
    ap.add_argument("--max-batch-tokens", type=int, default=0,
                    help="token budget per batch (count x longest length); 0 = fixed --batch-size")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-dir", help="task directory; overrides the benchmark config's data path")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N docs/queries (tests)")
    ap.add_argument("--store-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--weights-root", default=os.environ.get("ENCODE_WEIGHTS_ROOT", ""),
                    help="directory that relative adapter paths in model configs are resolved against")
    a = ap.parse_args()

    cfg = load_config(a.model, a.benchmark, a.task)
    m = cfg["model"]
    dtype = getattr(torch, cfg["dtype"]) if a.device.startswith("cuda") else torch.float32
    # the output records the config's data path; --data-dir only changes where the files are read
    doc_ids, docs, qids, queries = read_task(a.data_dir or cfg["data_dir"], m.get("title_join", "space"))
    if a.limit:
        doc_ids, docs, qids, queries = doc_ids[:a.limit], docs[:a.limit], qids[:a.limit], queries[:a.limit]

    t0 = time.time()
    models, tok, head = load_model(m, dtype, a.device, a.weights_root)
    qtemplate = m.get("query_template", "{text}")
    if not qtemplate.endswith("{text}"):
        raise ValueError("query_template must end with {text}")
    prefix = fmt(qtemplate, cfg["instruction"], "") if cfg["instruction"] else ""
    q_texts = [prefix + q for q in queries]
    prefix_tokens = len(tok.tokenize(prefix)) if (m.get("mask_instruction") and cfg["instruction"]) else 0
    if cfg["doc_instruction"]:
        d_texts = [fmt(qtemplate, cfg["doc_instruction"], "") + d for d in docs]
    elif m.get("title_join") == "pair":
        d_texts = docs
    else:
        d_texts = [fmt(m.get("doc_template", "{text}"), "", d) for d in docs]

    q_vec, q_stats = encode(q_texts, "query", models, tok, head, m, cfg["query_max_length"],
                            prefix_tokens, dtype, a.device, a.batch_size, a.max_batch_tokens)
    d_vec, d_stats = encode(d_texts, "doc", models, tok, head, m, cfg["doc_max_length"],
                            0, dtype, a.device, a.batch_size, a.max_batch_tokens)

    import transformers
    meta = dict(cfg, query_prefix=prefix, instruction_tokens_masked=prefix_tokens,
                queries=q_stats, docs=d_stats, batch_size=a.batch_size, max_batch_tokens=a.max_batch_tokens, device=a.device,
                gpu=torch.cuda.get_device_name() if a.device.startswith("cuda") else platform.processor(),
                torch=torch.__version__, transformers=transformers.__version__,
                adapter_sha256=sha256_dir(m["adapter"], "*.safetensors") if m.get("adapter") else None,
                seconds=round(time.time() - t0, 1), limit=a.limit, store_dtype=a.store_dtype)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, doc_ids=np.array(doc_ids), doc_vectors=d_vec.astype(a.store_dtype),
             query_ids=np.array(qids), query_vectors=q_vec.astype(a.store_dtype),
             config=np.array(json.dumps(meta)))
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps({"out": a.out, "docs": d_stats, "queries": q_stats, "seconds": meta["seconds"]}))


if __name__ == "__main__":
    main()
