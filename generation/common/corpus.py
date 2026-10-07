"""The one place the generation code touches the corpus.

encode_queries  query texts -> vectors, through the release embedding/encode.py
exact_search    vectors -> ranked (doc id, cosine) through the release search/search.py
fetch_metadata  doc ids -> title, abstract, ... from a metadata table you supply

Script locations default to the release layout (<release>/embedding/encode.py,
<release>/search/search.py) and can be moved with GEN_ENCODE_PY / GEN_SEARCH_PY.
The corpus text is not shipped; point GEN_METADATA_TABLE at a parquet or JSONL file
with one row per work (id column `openalex_id` or `id`).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

RELEASE_DIR = Path(__file__).resolve().parents[2]
META_FIELDS = ("title", "abstract", "journal_name", "cited_by_count", "doc_type", "year")


def _load(env: str, default: Path):
    path = os.environ.get(env) or str(default)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found; set {env} to its location")
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ENCODERS = {}   # model config path -> (encode module, config, models, tokenizer, head, dtype, device)


def encode_queries(model_config_path, texts, instruction, max_length=8192, batch_size=16):
    """float32 (n, dim), unit length. `instruction` is used verbatim (no suffix added).
    The model is loaded once per config path and kept for later calls."""
    if model_config_path not in _ENCODERS:
        import torch
        import yaml
        enc = _load("GEN_ENCODE_PY", RELEASE_DIR / "embedding" / "encode.py")
        m = yaml.safe_load(open(model_config_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, m.get("dtype", "bfloat16")) if device == "cuda" else torch.float32
        _ENCODERS[model_config_path] = (enc, m, *enc.load_model(
            m, dtype, device, os.environ.get("ENCODE_WEIGHTS_ROOT", "")), dtype, device)
    enc, m, models, tok, head, dtype, device = _ENCODERS[model_config_path]
    prefix = enc.fmt(m.get("query_template", "{text}"), instruction, "") if instruction else ""
    n_mask = len(tok.tokenize(prefix)) if (m.get("mask_instruction") and instruction) else 0
    vecs, stats = enc.encode([prefix + t for t in texts], "query", models, tok, head, m, max_length,
                             n_mask, dtype, device, batch_size)
    if stats["truncated"]:
        raise SystemExit(f"{stats['truncated']} query texts exceed max_length={max_length}")
    return vecs


def exact_search(index_dir, query_vectors, top_k, device="cpu"):
    """Per query, [(doc id, cosine), ...] best first, over every row of the index."""
    s = _load("GEN_SEARCH_PY", RELEASE_DIR / "search" / "search.py")
    v, i, ids = s.search(str(index_dir), query_vectors, top_k, device)
    return [[(ids[j], float(x)) for x, j in zip(rv, ri) if j >= 0] for rv, ri in zip(v, i)]


def fetch_metadata(ids):
    """{id: {title, abstract, journal_name, cited_by_count, doc_type, year}} for the ids found."""
    path = os.environ.get("GEN_METADATA_TABLE")
    if not path:
        raise SystemExit("no corpus metadata table: set GEN_METADATA_TABLE to a parquet or JSONL file")
    want, out = set(map(str, ids)), {}
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        cols = pq.read_schema(path).names
        key = "openalex_id" if "openalex_id" in cols else "id"
        t = pq.read_table(path, columns=[key] + [f for f in META_FIELDS if f in cols],
                          filters=[(key, "in", list(want))])
        rows = t.to_pylist()
    else:
        rows = (json.loads(l) for l in open(path) if l.strip())
    for r in rows:
        k = str(r.get("openalex_id") or r.get("id"))
        if k in want:
            out[k] = {f: r.get(f) for f in META_FIELDS}
    return out
