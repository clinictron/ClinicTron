"""Corpus search and document metadata for stages 02, 04, 05 and 06, built on common/corpus.py.

The original run used approximate-nearest-neighbour and BM25 search services over the same
corpus. This release substitutes exact search: each dense retriever encodes the query texts
with `corpus.encode_queries` and scans its whole index directory with `corpus.exact_search`.
BM25 goes through `bm25_search`, which you back with your own lexical index.

`CorpusSearch` keeps the surface the stages call (`start`, `stop`, `search`, and `prepare`
for a batch). Each retriever returns its own ranked list to its own `depth`; the lists are
fused by reciprocal-rank fusion (score = sum over retrievers of 1/(rrf_k + rank), rrf_k=60),
ties broken by doc id ascending, and the fused list is capped at `fused_depth`.
`search(query, top_k)` returns the first `top_k` fused documents, each carrying
`retrieval_features.ranks = {retriever: rank}`, so stage 04 can take every retriever's own
best N by its own rank.

`MetaResolver` maps doc id -> metadata: the on-disk doc_meta_cache.jsonl first, then
`corpus.fetch_metadata`, appending every new lookup to the run's cache file.
"""
from __future__ import annotations

import json
from pathlib import Path

EMPTY = {"title": "", "abstract": "", "journal_name": "", "publication_year": None,
         "cited_by_count": None, "doc_type": "article", "is_high_impact": False,
         "is_retracted": False}
RRF_K = 60


def bm25_search(index_dir, texts, top_k):
    """Per query text, [(doc id, score), ...] best first, from YOUR BM25 index over the corpus.

    Backed here by a Pyserini/Lucene index directory; replace the body to use another engine."""
    if not index_dir:
        raise SystemExit("BM25 is required: build a BM25 index over your corpus and set "
                         "corpus_search.retrievers.bm25.index_dir (GEN_BM25_INDEX)")
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(str(index_dir))
    return [[(h.docid, float(h.score)) for h in s.search(t, k=top_k)] for t in texts]


class CorpusSearch:
    """Exact per-retriever search over the corpus, fused per query text."""

    retryable = False          # an exact result does not change when it is recomputed

    def __init__(self, cfg, log=print):
        self.cc = cfg["corpus_search"]
        self.log = log
        self.depth = int(self.cc.get("depth", 300))
        self.fused_depth = int(self.cc.get("fused_depth", 300))
        self.rrf_k = int(self.cc.get("rrf_k", RRF_K))
        self.hits: dict[str, dict[str, list]] = {}      # retriever -> text -> [(id, score)]

    def start(self):
        pass

    def stop(self):
        pass

    def _run(self, name, texts, top_k):
        r = self.cc["retrievers"][name]
        if name == "bm25":
            return bm25_search(r.get("index_dir"), texts, top_k)
        from . import corpus
        # ponytail: encode_queries loads the model per call; 04 batches via prepare(), 02's
        # per-query grounding search reloads it each time. Cache the model if 02 is slow.
        vecs = corpus.encode_queries(r["model_config"], texts, r.get("instruction") or "")
        return corpus.exact_search(r["index_dir"], vecs, top_k, self.cc.get("device", "cpu"))

    def _depth(self, name):
        return int(self.cc["retrievers"][name].get("depth") or self.depth)

    def prepare(self, rows, query_texts):
        """Search every query text of `rows` in one batched pass per retriever."""
        texts = sorted({t for row in rows for t in query_texts(row).values()})
        for name in self.cc["retrievers"]:
            got = self._run(name, texts, self._depth(name)) if texts else []
            self.hits[name] = dict(zip(texts, got))
            self.log(f"SEARCH_PREPARED retriever={name} texts={len(texts)} "
                     f"depth={self._depth(name)}")

    def search(self, query, top_k):
        docs: dict[str, dict] = {}
        for name in self.cc["retrievers"]:
            hits = self.hits.get(name, {}).get(query)
            if hits is None:
                hits = self._run(name, [query], self._depth(name))[0]
            for rank, (wid, _score) in enumerate(hits[:self._depth(name)], 1):
                d = docs.setdefault(str(wid), {"work_id": str(wid),
                                               "retrieval_features": {"ranks": {}}})
                d["retrieval_features"]["ranks"][name] = rank
        fused = sorted(docs.values(), key=lambda d: (
            -sum(1.0 / (self.rrf_k + k) for k in d["retrieval_features"]["ranks"].values()),
            d["work_id"]))[:min(int(top_k), self.fused_depth)]
        if fused:
            from . import corpus
            meta = corpus.fetch_metadata([d["work_id"] for d in fused])
            for d in fused:
                d.update({k: v for k, v in (meta.get(d["work_id"]) or {}).items()
                          if v is not None})
        return fused


def build_search(cfg, log=print) -> CorpusSearch:
    return CorpusSearch(cfg, log)


def load_cache(paths) -> dict[str, dict]:
    """Preload doc_meta_cache.jsonl files (later files win)."""
    mem: dict[str, dict] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                mem[str(rec["work_id"])] = rec
    return mem


class MetaResolver:
    """work_id -> metadata, cache first, `corpus.fetch_metadata` second, appended to a run
    cache. `cfg` and `secrets` are accepted for call-site compatibility and unused."""

    def __init__(self, cfg=None, secrets=None, cache_paths=(), out_path=None, log=print):
        self.mem = load_cache(cache_paths)
        self.out_path = str(out_path) if out_path else None
        self.log = log

    def resolve(self, work_ids, *, cache_only: bool = False) -> dict[str, dict]:
        """Fill and return metadata for `work_ids`. Synthetic ids (`synth:*`) are skipped."""
        wanted = [w for w in dict.fromkeys(map(str, work_ids)) if not w.startswith("synth:")]
        missing = [w for w in wanted if w not in self.mem]
        if missing and not cache_only:
            from . import corpus
            found = corpus.fetch_metadata(missing)
            n_unresolved = 0
            for w in missing:
                got = found.get(w)
                if got is None:
                    n_unresolved += 1
                rec = {"work_id": w, **EMPTY, **{k: v for k, v in (got or {}).items()
                                                 if v is not None}}
                if got and got.get("year") is not None:
                    rec["publication_year"] = got["year"]
                self.mem[w] = rec
                if self.out_path:
                    with open(self.out_path, "a") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_unresolved:
                self.log(f"META_UNRESOLVED n={n_unresolved} of {len(missing)} "
                         f"(rows written with empty metadata, counted here)")
        elif missing:
            self.log(f"META_CACHE_MISS n={len(missing)} (cache_only=True, corpus not consulted)")
        return {w: self.mem.get(w, {"work_id": w, **EMPTY}) for w in wanted}
