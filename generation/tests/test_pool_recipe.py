#!/usr/bin/env python3
"""test_pool_recipe.py -- stage 04 builds pool recipe R exactly as run.

Four retrievers by three query texts:

  query text              frozen  bmretriever  openai_vector  bm25   row total
  the written question       50        15            15        15       95
  hypothetical abstract 1    15        10            10        10       45
  hypothetical abstract 2    15        10            10        10       45
  the two abstracts as documents                                         2
  raw total                                                            187

Checked here with a stub search backend, so nothing is searched and nothing is graded:
  * the config carries the recipe grid, and the retired pieces are GONE — no inject
    strings anywhere, no `inject_strings` helper, SPECTER2 and MedCPT not selectors
  * the frozen column reads stage 03's scan under `<qid>`, `<qid>#s1`, `<qid>#s2`, in
    rank order, and costs no search call
  * exactly THREE deep fused calls per row, one per query text, at `pool.fused_depth`
  * each of the three retrievers contributes its own quota BY ITS OWN RANK, per query
    text — 15 from the question, 10 from each abstract
  * SPECTER2 and MedCPT ranks are present in the fused result and are never selected,
    and never counted as unknown labels
  * an unconfigured label IS counted; an alias maps it back
  * a retriever that returned nothing for a query text is logged
  * the two abstracts appear as `synth:<qid>:<k>` documents
  * the raw contributions total exactly 187, the union is deduped, and there is no cap
  * every document carries its full provenance: every (query text, retriever, rank)
  * a row missing its sketches is LOGGED and pools from the question alone

A retriever that comes back short is never accepted silently:
  * a response short of a required retriever's quota is RETRIED, and a response that
    recovers on retry produces a complete pool with no shortfall recorded
  * a response that stays short leaves `pool_shortfall` on the row and a POOL_SHORTFALL
    line in skips.log — the row is flagged, never silently thinner
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parent
sys.path.insert(0, str(GEN))

from common.config import load_config                       # noqa: E402
from common.logging import Log, SkipLog                     # noqa: E402

_sp = importlib.util.spec_from_file_location("pool_candidates",
                                             str(GEN / "04_pool_candidates.py"))
pool_candidates = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(pool_candidates)

SELECTED = ["bmretriever", "openai_vector", "bm25"]
IGNORED = ["specter2", "medcpt"]
ALL_SOURCES = SELECTED + IGNORED
QID = "p7test-failure_groups-stacked_constraints-0000"
FAILURES: list[str] = []
_TMP = tempfile.TemporaryDirectory()



class StubBridge:
    """Offline search stand-in. Records every query it is asked.
    `results` is a callable(query, top_k) -> list[dict], or a fixed list."""

    def __init__(self, results=None, required_sources=()):
        self.queries: list[tuple[str, int]] = []
        self._results = results
        self.required_sources = list(required_sources)

    def start(self):
        return None

    def stop(self):
        return None

    def search(self, query: str, top_k: int, **_kw) -> list[dict]:
        self.queries.append((query, top_k))
        if callable(self._results):
            return self._results(query, top_k)
        return list(self._results or [])

def check(ok: bool, msg: str):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILURES.append(msg)


def deep_result(prefix="W", n=300, extra_source=None, empty=()):
    """A fused list whose per-source ranks deliberately DISAGREE with the fused order, so
    taking each source's own best N is visibly not the same as taking the fused head."""
    out = []
    for i in range(n):
        ranks, sims = {}, {}
        for s_i, src in enumerate(ALL_SOURCES):
            if src in empty:
                continue
            ranks[src] = ((i + s_i * 37) % n) + 1
            sims[src] = 1.0 - ranks[src] / (n + 1)
        if extra_source:
            ranks[extra_source] = i + 1
        out.append({"work_id": f"{prefix}{i:04d}", "title": f"t{i}", "abstract": f"a{i}",
                    "journal_name": "J", "cited_by_count": i, "doc_type": "article",
                    "retrieval_features": {"ranks": ranks, "sims": sims,
                                           "rrf_score": 1.0 / (60 + i + 1)}})
    return out


def make_bridge(deep_by_prefix):
    """One id space per query text, so a document found by two query texts is a real
    overlap rather than an artefact of the stub."""
    calls = []

    def results(query, top_k):
        calls.append((query, top_k))
        if query.startswith("A question"):
            return deep_by_prefix["question"][:top_k]
        if query.startswith("s1"):
            return deep_by_prefix["sketch_1"][:top_k]
        return deep_by_prefix["sketch_2"][:top_k]

    b = StubBridge(results=results)
    b.calls = calls
    return b


def make_row(with_sketches=True):
    cell = {"cell_id": QID.split("-", 1)[1], "run_id": "p7test", "seed": 3,
            "source": "failure_groups", "type_or_group": "stacked_constraints",
            "grounded": False, "masking": "impossible", "masked": False,
            "track": "drugs",
            "entity": {"track": "drugs", "id": "R:1", "display_name": "warfarin",
                       "synonyms": []},
            "length": "medium", "pasted": False, "document_type": None, "style": "clean"}
    sketches = ([{"title": "s1 title", "abstract": "s1 abstract"},
                 {"title": "s2 title", "abstract": "s2 abstract"}]
                if with_sketches else [])
    return {"qid": QID, "cell_id": cell["cell_id"],
            "text": "A question about warfarin.", "sketches": sketches, "cell": cell}


def build(cfg, deep_by_prefix, ranks, row=None):
    log = Log(Path(_TMP.name) / "run.log", echo=False)
    skip = SkipLog(Path(_TMP.name) / "skips.log", echo=False)
    bridge = make_bridge(deep_by_prefix)
    p = pool_candidates.Pooler(cfg, bridge, ranks, log, skip)
    row = row or make_row()
    got = asyncio.run(p.pool(row))
    return p, bridge, row, got["docs"]


def frozen_ranks(cfg):
    """Stage 03 rankings for the three query texts. The question's head OVERLAPS the
    retrievers, as it does in reality."""
    q = [[f"W{i:04d}", 1.0 - i / 100] for i in range(20)] + \
        [[f"F{i:04d}", 0.5 - i / 100] for i in range(60)]
    s1 = [[f"S1{i:04d}", 1.0 - i / 100] for i in range(40)]
    s2 = [[f"S2{i:04d}", 1.0 - i / 100] for i in range(40)]
    return {QID: {"id": QID, "slot": "question", "qid": QID, "fused": q},
            f"{QID}#s1": {"id": f"{QID}#s1", "slot": "sketch_1", "qid": QID, "fused": s1},
            f"{QID}#s2": {"id": f"{QID}#s2", "slot": "sketch_2", "qid": QID, "fused": s2}}


def main() -> int:
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    pc = cfg["pool"]

    # ── the recipe grid, and what was retired ──
    check(pc["frozen"]["question_top_k"] == 50 and pc["frozen"]["sketch_top_k"] == 15,
          "config: frozen column is 50 for the question, 15 per abstract")
    check(pc["per_retriever"]["question_top_k"] == 15
          and pc["per_retriever"]["sketch_top_k"] == 10,
          "config: each search source gives 15 for the question, 10 per abstract")
    check(list(pc["retrievers"]) == SELECTED,
          f"config: the THREE search selectors, in the run's order: {pc['retrievers']}")
    check(sorted(pc["ignored_retrievers"]) == sorted(IGNORED),
          f"config: SPECTER2 and MedCPT are retired as selectors: "
          f"{pc['ignored_retrievers']}")
    for gone in ("inject_top_k", "injected_cap", "hyde_top_k", "frozen_top_k",
                 "per_retriever_top_k"):
        check(gone not in pc, f"config: the retired key `pool.{gone}` is gone")
    check("failure_group_inject" not in cfg,
          "config: `failure_group_inject` is gone — inject strings are retired")
    import common.prompts as P
    check(not hasattr(P, "inject_strings"),
          "common/prompts.py no longer offers inject_strings")
    expected_raw = (pc["frozen"]["question_top_k"]
                    + 3 * pc["per_retriever"]["question_top_k"]
                    + 2 * (pc["frozen"]["sketch_top_k"]
                           + 3 * pc["per_retriever"]["sketch_top_k"])
                    + 2)
    check(expected_raw == 187,
          f"the config's own grid sums to the run's raw total of 187 (got {expected_raw})")

    deep = {"question": deep_result("W"), "sketch_1": deep_result("X"),
            "sketch_2": deep_result("Y")}
    ranks = frozen_ranks(cfg)
    p, bridge, row, docs = build(cfg, deep, ranks)

    # ── the three deep calls ──
    check(len(bridge.calls) == 3 and all(c[1] == pc["fused_depth"] for c in bridge.calls),
          f"exactly THREE deep fused calls, one per query text, at pool.fused_depth: "
          f"{[c[1] for c in bridge.calls]}")
    texts = {c[0] for c in bridge.calls}
    check(texts == {row["text"], "s1 title s1 abstract", "s2 title s2 abstract"},
          "the three query texts are the question and the two hypothetical abstracts")

    def by_route(slot, retriever):
        return [d for d in docs if any(e["query"] == slot and e["retriever"] == retriever
                                       for e in d.get("provenance", []))]

    # ── the frozen column ──
    check(len(by_route("question", "frozen")) == 50,
          f"question x frozen = 50 (got {len(by_route('question', 'frozen'))})")
    for slot in ("sketch_1", "sketch_2"):
        check(len(by_route(slot, "frozen")) == 15,
              f"{slot} x frozen = 15 (got {len(by_route(slot, 'frozen'))})")
    got = sorted(by_route("question", "frozen"),
                 key=lambda d: next(e["rank"] for e in d["provenance"]
                                    if e["query"] == "question"
                                    and e["retriever"] == "frozen"))
    check([d["work_id"] for d in got] == [w for w, _ in ranks[QID]["fused"][:50]],
          "the frozen column takes the scan's head, in rank order")
    check(not any(c[1] < 100 for c in bridge.calls),
          "the frozen column costs no search call of its own")

    # ── the three search columns ──
    for slot, prefix, quota in (("question", "question", 15), ("sketch_1", "sketch_1", 10),
                                ("sketch_2", "sketch_2", 10)):
        for src in SELECTED:
            got = by_route(slot, src)
            check(len(got) == quota,
                  f"{slot} x {src} = {quota} (got {len(got)})")
            want = sorted(deep[prefix],
                          key=lambda d: d["retrieval_features"]["ranks"][src])[:quota]
            check({d["work_id"] for d in got} == {d["work_id"] for d in want},
                  f"{slot} x {src} takes ITS OWN best {quota} by its own rank")
    check({d["work_id"] for d in by_route("question", "bm25")}
          != {d["work_id"] for d in deep["question"][:15]},
          "per-source selection is not the fused head (the stub disagrees on purpose)")

    # ── the retired selectors ──
    for src in IGNORED:
        check(not by_route("question", src) and not by_route("sketch_1", src),
              f"{src} ranks are present in the fused result and never selected")
    check(not p.unknown_sources,
          f"the ignored selectors are NOT reported as unknown labels: "
          f"{dict(p.unknown_sources)}")

    # ── unknown labels and aliases ──
    p2, _, _, _ = build(load_config(str(GEN / "configs/generation.yaml")),
                        {k: deep_result(pfx, extra_source="lexical_v2")
                         for k, pfx in (("question", "W"), ("sketch_1", "X"),
                                        ("sketch_2", "Y"))}, ranks)
    check(p2.unknown_sources.get("lexical_v2"),
          f"an unconfigured retriever label is COUNTED, not dropped silently: "
          f"{dict(p2.unknown_sources)}")
    cfg3 = load_config(str(GEN / "configs/generation.yaml"))
    cfg3["pool"]["retriever_aliases"] = {"lexical_v2": "bm25"}
    p3, _, _, _ = build(cfg3, {k: deep_result(pfx, extra_source="lexical_v2")
                               for k, pfx in (("question", "W"), ("sketch_1", "X"),
                                              ("sketch_2", "Y"))}, ranks)
    check(not p3.unknown_sources,
          "a configured alias maps the label back, leaving none unknown")

    # ── an empty retriever, per query text ──
    p4, _, _, _ = build(load_config(str(GEN / "configs/generation.yaml")),
                        {"question": deep_result("W", empty=("bm25",)),
                         "sketch_1": deep_result("X"), "sketch_2": deep_result("Y")},
                        ranks)
    check(p4.skip.counts.get("RETRIEVER_EMPTY") == 1,
          f"a retriever that returned nothing for ONE query text is LOGGED: "
          f"{dict(p4.skip.counts)}")

    # ── the abstracts as documents, and the union ──
    synth = [d for d in docs if str(d["work_id"]).startswith("synth:")]
    check(len(synth) == 2 and {d["work_id"] for d in synth} ==
          {f"synth:{QID}:1", f"synth:{QID}:2"},
          "the two abstracts appear as synth:<qid>:<k> documents")
    raw = sum(p.route_counts.values())
    check(raw == 187, f"the raw contributions total exactly 187 (got {raw})")
    wids = [d["work_id"] for d in docs]
    check(len(wids) == len(set(wids)), "the union is deduped on work_id")
    check(len(docs) > 140, f"no post-union cap: the pool is {len(docs)} documents")
    multi = [d for d in docs if len(d.get("provenance", [])) > 1]
    check(multi, f"a document found several ways keeps every finding ({len(multi)})")
    check(all(all({"query", "retriever", "rank"} <= set(e) for e in d["provenance"])
              for d in docs),
          "every provenance entry is (query text, retriever, rank)")
    slots_seen = {e["query"] for d in docs for e in d["provenance"]}
    check(slots_seen == {"question", "sketch_1", "sketch_2"},
          f"all three query texts appear in provenance: {sorted(slots_seen)}")

    # ── a row without sketches ──
    p5, b5, _, docs5 = build(load_config(str(GEN / "configs/generation.yaml")), deep,
                             ranks, row=make_row(with_sketches=False))
    check(p5.skip.counts.get("SKETCHES_MISSING") == 1,
          f"a row missing its sketches is LOGGED, not silently under-pooled: "
          f"{dict(p5.skip.counts)}")
    check(len(b5.calls) == 1,
          "such a row makes only the question's call, and nothing is imputed")

    check_shortfall(cfg)

    print(f"\npool size for the worked example: {len(docs)} unique documents "
          f"from {raw} raw contributions")
    print(f"{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def check_shortfall(cfg):
    """Recipe R must notice a retriever that is missing or short, not just malformed."""

    ranks = frozen_ranks(cfg)
    # `answers` is a list of source-sets, consumed one call at a time
    def make(answers, n=300):
        state = {"i": 0}

        def results(query, top_k):
            srcs = answers[min(state["i"], len(answers) - 1)]
            state["i"] += 1
            out = []
            for i in range(n):
                r = {s: ((i + j * 37) % n) + 1 for j, s in enumerate(sorted(srcs))}
                out.append({"work_id": f"W{i:04d}", "title": "t", "abstract": "a",
                            "retrieval_features": {"ranks": r, "sims": {},
                                                   "rrf_score": 0.1}})
            return out[:top_k]
        b = StubBridge(results=results)
        b.calls = []
        return b

    def run(bridge, cfg_):
        log = Log(Path(_TMP.name) / "run.log", echo=False)
        skip = SkipLog(Path(_TMP.name) / "skips.log", echo=False)
        p = pool_candidates.Pooler(cfg_, bridge, ranks, log, skip)
        row = make_row()
        row["sketches"] = []            # one query text keeps the test quick
        got = asyncio.run(p.pool(row))
        return p, got

    cfg_fast = load_config(str(GEN / "configs/generation.yaml"))
    cfg_fast["pool"]["shortfall_backoff_s"] = 0.0

    # 1. a degraded first response, then a healthy one
    degraded = [{"openai_vector", "bm25"}] + [set(SELECTED + IGNORED)] * 5
    p, got = run(make(degraded), cfg_fast)
    check(p.retry_counts.get("question") == 1,
          f"a response missing bmretriever is retried once: {dict(p.retry_counts)}")
    check(not got["shortfall"] and not p.shortfall_counts,
          "a response that recovers on retry leaves no shortfall on the row")
    check(len({d["work_id"] for d in got["docs"]
               if any(e["retriever"] == "bmretriever" for e in d["provenance"])}) == 15,
          "the recovered response fills bmretriever's full quota of 15")

    # 2. a retriever that never comes back
    p2, got2 = run(make([{"openai_vector", "bm25"}]), cfg_fast)
    n_attempts = 1 + int(cfg_fast["pool"]["shortfall_retries"])
    check(p2.retry_counts.get("question") == n_attempts,
          f"the query is tried {n_attempts} times (one call + "
          f"pool.shortfall_retries): {dict(p2.retry_counts)}")
    check(got2["shortfall"].get("question", {}).get("bmretriever") == 0,
          f"the row records WHICH retriever stayed short and by how much: "
          f"{got2['shortfall']}")
    check(p2.skip.counts.get("POOL_SHORTFALL") == 1,
          f"the shortfall reaches skips.log: {dict(p2.skip.counts)}")
    check(got2["docs"], "the row is FLAGGED and kept, not dropped")

    # 3. a retriever that answers but with fewer than the quota
    thin = [{"openai_vector", "bm25", "bmretriever"}]
    p3, got3 = run(make(thin, n=10), cfg_fast)     # 10 docs < the question quota of 15
    check(got3["shortfall"].get("question"),
          f"a retriever that answers with FEWER than the quota is short too: "
          f"{got3['shortfall']}")


def test_pool_recipe():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
