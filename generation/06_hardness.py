
"""06_hardness.py — frozen-model hardness bands and the admission rule on graded pools.

Inputs: the questions, the graded pools (stage 05 scores) and the frozen-model rankings of
each question from stage 03. For each question it records:

  density    the number of the frozen top-n (`hardness.top_n`) graded >= `hardness.answering_threshold`
  band       from density via `hardness.bands`
  admission  for sources in `hardness.admission_for`: keep the row if its pool holds at least
             `min_at_9plus` papers graded 9+ or at least `min_at_8plus` graded 8+ (real papers only;
             synthetic sketch documents are excluded); otherwise it is rejected and logged.
             For sources in `hardness.survival_for`: reject a row whose pool holds fewer than
             `min_answering` papers at the answering threshold.

Any frozen top-n document the grader has not scored is graded now with the stage-05 grader;
`--no-grade` turns that off. `annotate` is a pure function of (rows, scores, rankings).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config, load_secrets
from common.gates import band_for
from common.llm import Budget, build_lanes
from common.logging import Log, SkipLog, append_jsonl
from common.retrieval import MetaResolver


def top10_from_ranks(ranks: dict, qid: str, n: int) -> list[str]:
    """The frozen model's top-n work_ids for one question, from a rankings file
    ({"id": qid, "fused": [[work_id, score], ...]})."""
    row = ranks.get(qid) or {}
    fused = row.get("fused") or row.get("ranking") or []
    return [str(w) for w, _ in fused[:n]]


def annotate(rows, scores: dict, ranks: dict, cfg, log, skip):
    """Attach density, band and survival to every row. Pure function of its inputs —
    this is what the test exercises; the GPU scan only supplies `ranks`."""
    h = cfg["hardness"]
    thr = int(h["answering_threshold"])
    n = int(h["top_n"])
    out = []
    for r in rows:
        qid = r["qid"]


        sc = {w: s for w, s in scores.get(qid, {}).items() if not str(w).startswith("synth:")}
        top = top10_from_ranks(ranks, qid, n) if ranks else \
            sorted(sc, key=lambda w: (-sc[w], w))[:n]
        density = sum(1 for w in top if sc.get(w, 0) >= thr)
        band = band_for(density, h["bands"])
        pool_answering = sum(1 for w, s in sc.items() if s >= thr)


        n9 = sum(1 for v in sc.values() if v >= 9)
        n8 = sum(1 for v in sc.values() if v >= 8)
        rec = {"qid": qid, "cell_id": r["cell_id"], "source": r["cell"]["source"],
               "top_n": top, "density": density, "band": band,
               "pool_answering": pool_answering,
               "pool_max_grade": max(sc.values()) if sc else 0,
               "pool_n_at_9plus": n9, "pool_n_at_8plus": n8,
               "hardness_source": "frozen_top_n" if ranks else "pool_only"}


        adm = h.get("admission") or {}
        need9 = int(adm.get("min_at_9plus", 1))
        need8 = int(adm.get("min_at_8plus", 3))
        if r["cell"]["source"] in h.get("admission_for", []):
            if n9 >= need9 or n8 >= need8:
                rec["survived"] = True
            else:
                rec["survived"] = False
                skip.note("ADMISSION_REJECT", qid=qid, source=r["cell"]["source"],
                          n_at_9plus=n9, n_at_8plus=n8, pool_answering=pool_answering)
        elif r["cell"]["source"] in h.get("survival_for", []):
            if pool_answering < int(h.get("min_answering", 1)):
                rec["survived"] = False
                skip.note("SURVIVAL_REJECT", qid=qid, source=r["cell"]["source"],
                          pool_answering=pool_answering)
            else:
                rec["survived"] = True
        else:
            rec["survived"] = True
        out.append(rec)
    return out


async def grade_missing(cfg, rows, pools, scores, meta, out: Path, log, skip, ranks):
    """Grade any frozen-top-n document the teacher has not already scored."""
    import importlib.util as iu
    sp = iu.spec_from_file_location(
        "grade_pools", str(Path(__file__).resolve().parent / "05_grade_pools.py"))
    g4 = iu.module_from_spec(sp)
    sp.loader.exec_module(g4)

    n = int(cfg["hardness"]["top_n"])
    todo = []
    for r in rows:
        have = scores.get(r["qid"], {})
        missing = [w for w in top10_from_ranks(ranks, r["qid"], n) if w not in have]
        if missing:
            todo.append((r, missing))
    if not todo:
        log("HARDNESS no ungraded top-n documents")
        return scores

    secrets = load_secrets(cfg["paths"]["secrets"])
    budget = Budget(cfg["budget"]["abort_usd"],
                    str(out / cfg["budget"]["cost_state_file"]),
                    warn_usd=cfg["budget"]["warn_usd"],
                    initial_usd=cfg["budget"]["initial_usd"])
    grader = build_lanes(cfg, budget, log, secrets=secrets)["grader"]
    resolver = MetaResolver(None, secrets,
                            cache_paths=[out / "doc_meta_cache.jsonl"],
                            out_path=str(out / "doc_meta_cache.jsonl"), log=log)
    resolver.mem.update(meta)
    for r, missing in todo:
        m = resolver.resolve(missing)
        pool = [{"work_id": w, "title": m[w]["title"], "abstract": m[w]["abstract"]}
                for w in missing]
        try:
            got = await g4.grade_pool(grader, r["text"], pool, m,
                                      seed=r["cell"]["seed"], cfg=cfg, log=log,
                                      qid=r["qid"] + "-hard")
        except Exception as exc:
            skip.note("HARDNESS_GRADE_FAIL", qid=r["qid"], err=str(exc))
            continue
        scores.setdefault(r["qid"], {}).update(got)
        append_jsonl(out / "hardness_scores.jsonl",
                     {"id": r["qid"], "scores": got})
    log(f"HARDNESS graded {len(todo)} rows' missing top-n docs "
        f"spend=${budget.total:.2f}")
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--scores", required=True, help="scores.jsonl from stage 04")
    ap.add_argument("--rankings", required=True,
                    help="rankings.jsonl from stage 03 — the SAME scan pool recipe R used")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-grade", action="store_true",
                    help="do not grade ungraded top-n documents (offline / dry run)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("06_hardness")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    skip = SkipLog(out / "skips.log")

    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(a.pools) if l.strip()}
    scores: dict[str, dict] = {}
    for line in open(a.scores):
        if line.strip():
            o = json.loads(line)
            scores.setdefault(str(o["id"]), {}).update(
                {str(k): int(v) for k, v in o["scores"].items()})
    rows = [r for r in rows if r["qid"] in pools]

    ranks: dict = {}
    n_skipped_slots = 0
    for line in open(a.rankings):
        if not line.strip():
            continue
        o = json.loads(line)
        slot = o.get("slot")
        sid = str(o.get("id") or o.get("qid"))

        if (slot is not None and slot != "question") or (slot is None and "#" in sid):
            n_skipped_slots += 1
            continue
        ranks[str(o.get("qid") or sid)] = o
    log(f"HARDNESS loaded the QUESTION ranking for {len(ranks)} rows from {a.rankings} "
        f"({n_skipped_slots} sketch rankings skipped — the band is the question's)")
    uncovered = [r["qid"] for r in rows if r["qid"] not in ranks]
    if uncovered:
        for qid in uncovered:
            skip.note("NO_FROZEN_RANKS", qid=qid)
        log(f"HARDNESS {len(uncovered)} question(s) have no frozen ranking; their band "
            f"falls back to the graded pool and is tagged hardness_source=pool_only")

    meta = MetaResolver(None, {},
                        cache_paths=[Path(a.pools).parent / "doc_meta_cache.jsonl"]).mem
    if not a.no_grade:
        scores = asyncio.run(grade_missing(cfg, rows, pools, scores, meta, out,
                                           log, skip, ranks))

    recs = annotate(rows, scores, ranks, cfg, log, skip)
    with (out / "hardness.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    import collections
    summary = {
        "rows": len(recs),
        "by_band": dict(collections.Counter(r["band"] for r in recs)),
        "by_source_band": {s: dict(collections.Counter(
            r["band"] for r in recs if r["source"] == s))
            for s in sorted({r["source"] for r in recs})},
        "rejected_by_survival": sum(1 for r in recs if not r["survived"]),
        "sketch_rankings_skipped": n_skipped_slots,
        "survival_for": cfg["hardness"]["survival_for"],
        "skips": skip.summary(),
    }
    (out / "hardness_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
