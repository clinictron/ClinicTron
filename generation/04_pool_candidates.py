#!/usr/bin/env python3
"""04_pool_candidates.py -- build each question's candidate pool by pool recipe R.

Note: the original run served the three non-base columns from approximate-nearest-neighbour
and BM25 search services over the same corpus. This release substitutes exact search
(common/retrieval.py on common/corpus.py); BM25 needs your own lexical index.

Recipe R: four retrievers by three query texts, per-cell quotas, union, dedupe on work_id,
no cap:

  query text              frozen  bmretriever  openai_vector  bm25   row total
  the written question       50        15            15        15       95
  hypothetical abstract 1    15        10            10        10       45
  hypothetical abstract 2    15        10            10        10       45
  the two abstracts as documents                                         2
  raw total                                                            187
  expected unique                                                     ~145

Mechanics
  * The frozen (base encoder) column costs no search here: all three of a row's frozen
    searches ran in stage 03's full-corpus exact scan, under the ids `<qid>`, `<qid>#s1`,
    `<qid>#s2`.
  * The other three columns come from one batched exact pass per retriever over all query
    texts of the run (`prepare`), then one fused lookup per query text. Each document carries
    its per-retriever rank, and each retriever's own best N is taken by its own rank, not by
    the fused order.
  * `pool.ignored_retrievers` lists source labels that are deliberately never read.

Every document records its full provenance -- every (query text, retriever, rank) that
found it -- so a pool is auditable and readable by a later reranker.

A retriever that returns fewer documents than its quota is never accepted silently: the row
records a `pool_shortfall` field and the shortfall goes to skips.log. Exact search is not
retried, because a rescan returns the same answer.

Per-retriever quota overrides live in `pool.per_retriever_overrides`.

No LLM call. The search backend and the metadata resolver are built at run time, so the
module imports offline.
"""
from __future__ import annotations

import argparse
import atexit
import os
import asyncio
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config                       # noqa: E402
from common.logging import Log, SkipLog, append_jsonl, load_keys  # noqa: E402
from common.retrieval import MetaResolver, build_search      # noqa: E402

META_KEYS = ("title", "abstract", "journal_name", "cited_by_count", "doc_type", "year")

# The three query texts of a row, and the scan ids stage 03 gave them.
QUERY_SLOTS = ("question", "sketch_1", "sketch_2")
SCAN_SUFFIX = {"question": "", "sketch_1": "#s1", "sketch_2": "#s2"}


def scan_id(qid: str, slot: str) -> str:
    """The stage-03 scan id for one of a row's three query texts."""
    return f"{qid}{SCAN_SUFFIX[slot]}"


def query_texts(row: dict) -> dict[str, str]:
    """{slot: text} for the three query texts, in the recipe table's order."""
    out = {"question": row["text"]}
    for k, h in enumerate((row.get("sketches") or [])[:2], 1):
        text = f"{h.get('title', '')} {h.get('abstract', '')}".strip()
        if text:
            out[f"sketch_{k}"] = text
    return out


def per_retriever_lists(fused: list[dict], retrievers, aliases, ignored, top_k,
                        unknown: collections.Counter) -> dict[str, list[dict]]:
    """Split ONE deep fused result into per-source ranked lists, best `top_k` each.

    `top_k` is one number for every retriever, or a {retriever: number} mapping when
    `pool.per_retriever_overrides` gives a retriever its own quota.

    A source label that is neither a configured retriever, nor a configured alias, nor
    deliberately ignored is COUNTED and reported — a silent drop would look exactly like
    a retriever that returned nothing.
    """
    want: dict[str, list] = {r: [] for r in retrievers}
    for d in fused:
        feats = d.get("retrieval_features") or {}
        for src, rank in (feats.get("ranks") or {}).items():
            label = aliases.get(src, src)
            if label in want:
                want[label].append((int(rank), d))
            elif label not in ignored:
                unknown[src] += 1
    out = {}
    for src, pairs in want.items():
        pairs.sort(key=lambda p: (p[0], str(p[1].get("work_id"))))
        k = top_k[src] if isinstance(top_k, dict) else top_k
        out[src] = [d for _, d in pairs[:k]]
    return out


class Pooler:
    """Recipe R for one row: union, dedupe on work_id, no post-union cap, and the two
    hypothetical abstracts added as documents."""

    def __init__(self, cfg, bridge, ranks: dict, log, skiplog):
        self.cfg = cfg
        self.pc = cfg["pool"]
        self.bridge = bridge
        self.ranks = ranks
        self.log = log
        self.skip = skiplog
        self.retrievers = list(self.pc["retrievers"])
        self.aliases = dict(self.pc.get("retriever_aliases") or {})
        self.ignored = set(self.pc.get("ignored_retrievers") or [])
        self.unknown_sources: collections.Counter = collections.Counter()
        self.route_counts: collections.Counter = collections.Counter()
        self.retry_counts: collections.Counter = collections.Counter()
        self.shortfall_counts: collections.Counter = collections.Counter()
        self._no_retry_logged = False

    async def _search(self, query: str, top_k: int) -> list[dict]:
        return await asyncio.to_thread(self.bridge.search, query, top_k)

    def _short_sources(self, docs: list[dict], quotas: dict[str, int]) -> dict[str, int]:
        """{retriever: how many ranked docs it contributed} for every required source
        that did not reach ITS quota. Empty means the response is complete."""
        counts = {r: 0 for r in self.retrievers}
        for d in docs:
            for src in ((d.get("retrieval_features") or {}).get("ranks") or {}):
                label = self.aliases.get(src, src)
                if label in counts:
                    counts[label] += 1
        return {r: n for r, n in counts.items() if n < int(quotas.get(r, 0))}

    async def _search_complete(self, qid: str, slot: str, query: str,
                               quotas: dict[str, int]):
        """One deep fused call that is RETRIED while a required retriever is short.

        Returns (docs, shortfall) where `shortfall` is {} on success. A short response is
        indistinguishable from a healthy one by shape alone, which is exactly why it is
        checked rather than trusted.

        A backend that declares `retryable = False` (the exact backend: one batched pass over
        the corpus already happened) gets NO retries — a rescan returns the same answer, so a
        shortfall there is a real one. The decision is logged the first time it applies rather
        than silently skipping the loop.
        """
        docs: list[dict] = []
        short: dict[str, int] = {}
        retries = int(self.pc.get("shortfall_retries", 3))
        if not getattr(self.bridge, "retryable", True):
            if not self._no_retry_logged:
                self._no_retry_logged = True
                self.log(f"POOL_NO_RETRY backend={type(self.bridge).__name__} declares "
                         f"retryable=False, so shortfall_retries={retries} is not applied: "
                         f"an exact result does not change when it is recomputed. A short "
                         f"retriever is recorded as a real shortfall.")
            retries = 0
        for attempt in range(1 + retries):
            docs = await self._search(query, int(self.pc["fused_depth"]))
            short = self._short_sources(docs, quotas)
            if not short:
                if attempt:
                    self.log(f"POOL_RETRY_OK qid={qid} query={slot} "
                             f"recovered after {attempt} retry(ies)")
                return docs, {}
            if not retries:        # exact backend: nothing to retry, nothing to wait for
                break
            self.retry_counts[slot] += 1
            self.log(f"POOL_SHORT qid={qid} query={slot} attempt={attempt} "
                     f"quotas={quotas} short={short} — retrying")
            await asyncio.sleep(float(self.pc.get("shortfall_backoff_s", 3.0))
                                * (attempt + 1))
        self.skip.note("POOL_SHORTFALL", qid=qid, query=slot, quota=quotas, short=short,
                       reason=("a required retriever was short and this backend is exact "
                               "(no retry could change it)" if not retries else
                               "a required retriever stayed short after "
                               f"{retries} retries"))
        self.shortfall_counts.update(short.keys())
        return docs, short

    def _frozen_quota(self, slot: str) -> int:
        f = self.pc["frozen"]
        return int(f["question_top_k"] if slot == "question" else f["sketch_top_k"])

    def _retriever_quota(self, slot: str, retriever: str | None = None) -> int:
        """The recipe grid's quota for this query text, or a retriever's own override.

        `pool.per_retriever_overrides: {<retriever>: {question_top_k: N, sketch_top_k: M}}`
        lets one column take a different depth from the rest of the grid. Absent overrides, every retriever gets `pool.per_retriever`
        and the grid is exactly the one in the module docstring."""
        key = "question_top_k" if slot == "question" else "sketch_top_k"
        base = int(self.pc["per_retriever"][key])
        if retriever is None:
            return base
        ov = (self.pc.get("per_retriever_overrides") or {}).get(retriever) or {}
        return int(ov.get(key, base))

    def _retriever_quotas(self, slot: str) -> dict[str, int]:
        """{retriever: quota} for one query text."""
        return {r: self._retriever_quota(slot, r) for r in self.retrievers}

    async def pool(self, row: dict) -> dict:
        qid = row["qid"]
        texts = query_texts(row)
        if len(texts) < 3:
            self.skip.note("SKETCHES_MISSING", qid=qid, have=sorted(texts),
                           reason="recipe R needs the question AND two abstracts; run "
                                  "stage 02 --sketches-only before stage 03")

        pool: dict[str, dict] = {}
        prov: dict[str, list] = {}

        def add(wid: str, doc: dict, slot: str, retriever: str, rank: int):
            wid = str(wid)
            if wid not in pool:
                pool[wid] = {"work_id": wid,
                             **{k: doc.get(k) for k in META_KEYS
                                if doc.get(k) is not None}}
                prov[wid] = []
            prov[wid].append({"query": slot, "retriever": retriever, "rank": rank})
            self.route_counts[f"{slot}/{retriever}"] += 1
            for k in META_KEYS:      # keep the richest metadata seen for this document
                if pool[wid].get(k) in (None, "") and doc.get(k) not in (None, ""):
                    pool[wid][k] = doc[k]

        # ── the three deep fused calls, one per query text, each checked and retried ──
        slots = list(texts)
        results = await asyncio.gather(*[
            self._search_complete(qid, s, texts[s], self._retriever_quotas(s))
            for s in slots])
        deep_by_slot = {s: docs for s, (docs, _short) in zip(slots, results)}
        shortfall = {s: short for s, (_docs, short) in zip(slots, results) if short}

        for slot in slots:
            deep = deep_by_slot[slot]
            by_wid = {str(d["work_id"]): d for d in deep}

            # frozen column: read from stage 03's scan, no search of its own
            sid = scan_id(qid, slot)
            fused = (self.ranks.get(sid) or {}).get("fused") or []
            if not fused:
                self.skip.note("NO_FROZEN_RANKS", qid=qid, scan_id=sid,
                               reason="stage 03 has no rankings for this query text")
            for rank, (wid, _score) in enumerate(fused[:self._frozen_quota(slot)], 1):
                wid = str(wid)
                add(wid, by_wid.get(wid, {"work_id": wid}), slot, "frozen", rank)

            # the three search columns, each by its own rank
            lists = per_retriever_lists(deep, self.retrievers, self.aliases, self.ignored,
                                        self._retriever_quotas(slot),
                                        self.unknown_sources)
            for src, docs in lists.items():
                if not docs:
                    self.skip.note("RETRIEVER_EMPTY", qid=qid, query=slot, retriever=src)
                for rank, d in enumerate(docs, 1):
                    add(d["work_id"], d, slot, src, rank)

        docs = list(pool.values())                       # NO post-union cap
        for wid, d in pool.items():
            d["provenance"] = prov[wid]
        if self.pc.get("synthetic_sketches_as_documents", True):
            for k, h in enumerate((row.get("sketches") or [])[:2], 1):
                docs.append({"work_id": f"synth:{qid}:{k}",
                             "title": h.get("title") or "",
                             "abstract": h.get("abstract") or "",
                             "provenance": [{"query": f"sketch_{k}",
                                             "retriever": "synthetic", "rank": k}]})
                self.route_counts[f"sketch_{k}/synthetic"] += 1
        if not docs:
            self.skip.note("EMPTY_POOL", cell=row["cell_id"], qid=qid)
        return {"docs": docs, "texts": texts, "shortfall": shortfall}


async def run(cfg, rows, ranks, out: Path, cache_only: bool):
    log = Log(out / "run.log")
    skip = SkipLog(out / "skips.log")
    pools_path = out / "pools.jsonl"
    meta_path = out / "doc_meta_cache.jsonl"

    bridge = build_search(cfg, log)
    bridge.start()
    try:
        # All of this run's query texts are searched in one pass per retriever before any
        # row is pooled; `search()` is a lookup afterwards.
        if hasattr(bridge, "prepare"):
            bridge.prepare(rows, query_texts)
        pooler = Pooler(cfg, bridge, ranks, log, skip)
        sem = asyncio.Semaphore(int(cfg["pool"].get("concurrency", 16)))

        async def one(row):
            async with sem:
                got = await pooler.pool(row)
            append_jsonl(pools_path, {
                "qid": row["qid"], "cell_id": row["cell_id"],
                "source": row["cell"]["source"], "recipe": "R_locked_2026-09-07",
                "n_docs": len(got["docs"]),
                # the three query texts are recorded ONCE per row; each document's
                # provenance names the slot, not the whole text again
                "query_texts": got["texts"],
                # {} unless a required retriever stayed short after its retries; the row
                # is kept and FLAGGED rather than dropped, so later review can see it.
                "pool_shortfall": got["shortfall"],
                "pool": got["docs"]})
            return got["docs"]

        all_docs = await asyncio.gather(*[one(r) for r in rows])
        raw_total = sum(pooler.route_counts.values())
        if pooler.unknown_sources:
            log(f"RETRIEVER_UNKNOWN labels seen in retrieval_features but neither "
                f"selected nor ignored: {dict(pooler.unknown_sources)} — map them in "
                f"pool.retriever_aliases or list them in pool.ignored_retrievers")
        route_counts = dict(pooler.route_counts)
        unknown = dict(pooler.unknown_sources)
        retries = dict(pooler.retry_counts)
        shortfalls = dict(pooler.shortfall_counts)
    finally:
        bridge.stop()

    # Metadata: the search already gave us most of it; the corpus table fills the rest.
    resolver = MetaResolver(None, None,
                            cache_paths=[meta_path], out_path=str(meta_path), log=log)
    for docs in all_docs:
        for d in docs:
            wid = str(d["work_id"])
            if wid.startswith("synth:") or wid in resolver.mem:
                continue
            if d.get("abstract") and d.get("journal_name") is not None:
                resolver.mem[wid] = {"work_id": wid, **{k: d.get(k) for k in META_KEYS}}
                append_jsonl(meta_path, resolver.mem[wid])
    wanted = {str(d["work_id"]) for docs in all_docs for d in docs}
    # Resume safety: rows pooled by an earlier run that stopped before this point are in
    # pools.jsonl with frozen-only ids that were never resolved, and the grader would score
    # them as empty documents. Resolve every id in the file, not just this run's rows; the
    # cache makes the repeat lookups free.
    n_prev = 0
    if pools_path.exists():
        for line in open(pools_path, encoding="utf-8"):
            if line.strip():
                for d in json.loads(line)["pool"]:
                    if str(d["work_id"]) not in wanted:
                        wanted.add(str(d["work_id"]))
                        n_prev += 1
    if n_prev:
        log(f"META_RESUME resolving {n_prev} ids from rows pooled by earlier runs")
    resolver.resolve(sorted(wanted), cache_only=cache_only)

    uniq = sorted(len({str(d["work_id"]) for d in docs}) for docs in all_docs)
    log(f"POOL done rows={len(rows)} raw={raw_total} unique_total={len(wanted)} "
        f"skips={skip.summary()}")
    return {"recipe": "R_locked_2026-09-07", "rows": len(rows),
            "raw_contributions": raw_total,
            "raw_per_row": round(raw_total / max(len(rows), 1), 1),
            "raw_per_row_expected": 187,
            "unique_per_row_median": uniq[len(uniq) // 2] if uniq else 0,
            "unique_per_row_min": uniq[0] if uniq else 0,
            "unique_per_row_max": uniq[-1] if uniq else 0,
            "unique_per_row_expected": "~145",
            "unique_docs_total": len(wanted),
            "by_query_and_retriever": route_counts,
            "unknown_retriever_labels": unknown,
            "ignored_retrievers": list(cfg["pool"].get("ignored_retrievers") or []),
            "retries_by_query_text": retries,
            "rows_with_shortfall_by_retriever": shortfalls,
            "skips": skip.summary()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True, help="questions.jsonl from stage 02")
    ap.add_argument("--rankings", required=True,
                    help="rankings.jsonl from stage 03 — the frozen column of recipe R")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-only", action="store_true",
                    help="never query the corpus metadata table; use the on-disk "
                         "doc_meta_cache only")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("04_pool")
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    ranks = {}
    for line in open(a.rankings):
        if line.strip():
            o = json.loads(line)
            ranks[str(o.get("id") or o.get("qid"))] = o
    # One pooling client per output directory: two clients on one out-dir would both pool
    # the rows the other has not written yet.
    lock = out / "pool.pid"
    if lock.exists():
        try:
            other = int(lock.read_text().split()[0])
            os.kill(other, 0)
        except (ValueError, ProcessLookupError, PermissionError, IndexError):
            pass
        else:
            raise SystemExit(
                f"FATAL: another pooling client (pid {other}) is already writing {out}. "
                f"Remove {lock} if that process is gone.")
    lock.write_text(f"{os.getpid()}\n")
    atexit.register(lambda: lock.unlink(missing_ok=True))

    done = load_keys(out / "pools.jsonl", "qid")
    rows = [r for r in rows if r["qid"] not in done]
    if a.limit:
        rows = rows[:a.limit]

    summary = asyncio.run(run(cfg, rows, ranks, out, a.cache_only))
    (out / "pool_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
