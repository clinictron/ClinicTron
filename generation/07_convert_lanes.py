"""07_convert_lanes.py — emit the trainer-consumable lanes, one per source and origin.

Layout consumed by build_mixed_batches.py:

  <out>/<lane>/queries.jsonl         {"qid","text","text_sha1","split","arm","provenance"}
  <out>/<lane>/scores.jsonl          {"id","n_docs","n_scored","scores": {work_id: grade}}
  <out>/<lane>/doc_meta_cache.jsonl  {"work_id","title","abstract"}

A lane is named by its source (`ely_types`, `subclasses`, `failure_groups`), with the suffix
`_imported` for rows imported from an earlier generation run.

Rules:
  * Pools with fewer than max(4, k_docs//2) usable documents are excluded here and counted,
    because the batch builder would drop them.
  * No truncation: the longest query is measured against the trainer's max_len and reported.
  * The train/test split is grouped on the seed concept, so one concept cannot appear on
    both sides.
  * Every planning unit is subsampled to its exact target, deterministically by seed, and
    the surplus is logged per unit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config
from common.logging import Log
from common.seeds import seed_int, sha1


def unit_targets_for(cfg):
    """The same per-unit targets stage 01 planned against — read from stage 01, never
    recomputed here, so the two can never drift."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "plan_cells", str(Path(__file__).resolve().parent / "01_plan_cells.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    units = m.source_units(cfg, log=lambda *_: None)
    _ps, targets = m.unit_targets(cfg, units, int(cfg["volume"]["total_rows"]),
                                  log=lambda *_: None)
    return targets


def subsample_to_target(cfg, kept: list[str], rows: dict, log=print):
    """Cut every unit back to its exact target. Returns (kept, dropped, per_unit).

    Generation rounds overshoot on purpose — a round divides what it needs by a measured
    yield — so without this the realised mix would be whatever the yields happened to be
    rather than the mix the plan specifies. The choice is a seeded shuffle on
    sha1(run_id|subsample|source|unit), so it is reproducible and carries no ordering
    bias from the round a row came from.
    """
    targets = unit_targets_for(cfg)
    by_unit = defaultdict(list)
    for qid in kept:
        cell = rows[qid]["cell"]
        by_unit[(cell["source"], cell["type_or_group"])].append(qid)

    selected: list[str] = []
    dropped: list[str] = []
    table: dict = {}
    for (source, unit_id), ids in sorted(by_unit.items()):
        target = (targets.get(source) or {}).get(unit_id)
        ids = sorted(ids)
        if target is None:
            dropped.extend(ids)
            table[f"{source}/{unit_id}"] = {"target": None, "have": len(ids), "kept": 0,
                                            "surplus": len(ids),
                                            "note": "not a unit of this config"}
            log(f"CONVERT unit {source}/{unit_id}: {len(ids)} rows but the config has "
                f"no target for it — all dropped")
            continue
        random.Random(seed_int(cfg["run"]["run_id"], "subsample", source,
                               unit_id)).shuffle(ids)
        take, rest = ids[:target], ids[target:]
        selected.extend(take)
        dropped.extend(rest)
        table[f"{source}/{unit_id}"] = {"target": target, "have": len(ids),
                                        "kept": len(take), "surplus": len(rest),
                                        "short_by": max(0, target - len(take))}
        if rest:
            log(f"CONVERT unit {source}/{unit_id}: {len(ids)} rows -> target {target}, "
                f"{len(rest)} surplus dropped")
    n_short = sum(1 for v in table.values() if v.get("short_by"))
    log(f"CONVERT subsampled to target: kept={len(selected)} "
        f"surplus_dropped={len(dropped)} units_still_short={n_short}")
    return sorted(selected), dropped, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pools", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--hardness", default=None, help="hardness.jsonl from stage 05")
    ap.add_argument("--doc-meta", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("07_lanes")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    cv = cfg["convert"]

    rows = {json.loads(l)["qid"]: json.loads(l) for l in open(a.questions) if l.strip()}
    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(a.pools) if l.strip()}
    scores: dict[str, dict] = defaultdict(dict)
    for line in open(a.scores):
        if line.strip():
            o = json.loads(line)
            for k, v in o["scores"].items():
                scores[str(o["id"])][str(k)] = int(v)
    hard = {}
    if a.hardness:
        hard = {json.loads(l)["qid"]: json.loads(l) for l in open(a.hardness) if l.strip()}
    meta_path = a.doc_meta or str(Path(a.pools).parent / "doc_meta_cache.jsonl")
    meta = {}
    if Path(meta_path).exists():
        for line in open(meta_path):
            if line.strip():
                r = json.loads(line)
                meta[str(r["work_id"])] = {"work_id": str(r["work_id"]),
                                           "title": r.get("title") or "",
                                           "abstract": r.get("abstract") or ""}


    kept, rejected, too_small = [], [], []
    for qid, row in rows.items():
        if qid not in pools or qid not in scores:
            continue
        if hard.get(qid, {}).get("survived") is False:
            rejected.append(qid)
            continue

        usable = sum(1 for d in pools[qid]["pool"]
                     if d["work_id"] in scores[qid] and not str(d["work_id"]).startswith("synth:"))
        (kept if usable >= int(cv["min_pool"]) else too_small).append(qid)
    log(f"CONVERT kept={len(kept)} survival_rejected={len(rejected)} "
        f"pool_below_min={len(too_small)} (min_pool={cv['min_pool']})")

    surplus_dropped: list[str] = []
    unit_table: dict = {}
    if cv.get("subsample_to_target", True):
        kept, surplus_dropped, unit_table = subsample_to_target(cfg, kept, rows, log)


    def lane_of(qid: str) -> str:
        src = rows[qid]["cell"]["source"]
        origin = (rows[qid].get("provenance") or {}).get("origin")
        return f"{src}_imported" if origin else src

    by_source: dict[str, list[str]] = defaultdict(list)
    for qid in kept:
        by_source[lane_of(qid)].append(qid)

    def split_test(ids: list[str], source: str) -> set[str]:
        groups = defaultdict(list)
        for q in ids:
            groups[rows[q]["cell"]["entity"]["display_name"]].append(q)
        keys = sorted(groups)
        random.Random(seed_int(cfg["run"]["run_id"], source, "split")).shuffle(keys)
        n_test = int(len(ids) * float(cv["test_frac"]))
        test, acc = set(), 0
        for g in keys:
            if acc >= n_test:
                break
            test.update(groups[g])
            acc += len(groups[g])
        return test

    stats = {}
    for source, ids in sorted(by_source.items()):
        ids = sorted(ids)
        test_ids = split_test(ids, source)
        d = out / source
        d.mkdir(parents=True, exist_ok=True)
        seen_docs: set[str] = set()
        with (d / "queries.jsonl").open("w") as fq, \
             (d / "scores.jsonl").open("w") as fs, \
             (d / "doc_meta_cache.jsonl").open("w") as fm:
            for qid in ids:
                row, cell = rows[qid], rows[qid]["cell"]
                h = hard.get(qid, {})
                fq.write(json.dumps({
                    "qid": qid, "text": row["text"], "text_sha1": sha1(row["text"]),
                    "split": "test" if qid in test_ids else "train",
                    "arm": "pipeline7",
                    "provenance": {
                        "lane": source, "run_id": cfg["run"]["run_id"],
                        "source": source, "type_or_group": cell["type_or_group"],
                        "subclass": cell.get("subclass", {}).get("subtype_id"),
                        "style": cell["style"], "length": cell["length"],
                        "pasted": cell["pasted"], "document_type": cell["document_type"],
                        "masking": cell["masking"], "masked": cell["masked"],
                        "entity": cell["entity"]["display_name"],
                        "entity_track": cell["track"],
                        "grounded": cell["grounded"],
                        "origin": (row.get("provenance") or {}).get("origin") or "pipeline7",
                        "origin_id": (row.get("provenance") or {}).get("origin_id"),
                        "grounding_ids": row.get("grounding_ids") or [],
                        "hardness_band": h.get("band"),
                        "hardness_density": h.get("density"),
                        "generator_model": row.get("generator_model"),
                        "prompt_sha1": row.get("prompt_sha1"),
                        "teacher_model": cfg["grader"]["model"],
                        "teacher_grader": "pipeline7_locked",
                        "grade_scale": f"{cfg['grader']['score_floor']}-10",
                    }}, ensure_ascii=False) + "\n")


                real_pool = [d_ for d_ in pools[qid]["pool"]
                             if not str(d_["work_id"]).startswith("synth:")]
                sc = {d_["work_id"]: scores[qid][d_["work_id"]]
                      for d_ in real_pool if d_["work_id"] in scores[qid]}
                fs.write(json.dumps({"id": qid, "n_docs": len(real_pool),
                                     "n_scored": len(sc), "scores": sc}) + "\n")
                for d_ in real_pool:
                    w = d_["work_id"]
                    if w in seen_docs:
                        continue
                    seen_docs.add(w)
                    rec = meta.get(w) or {"work_id": w, "title": d_.get("title") or "",
                                          "abstract": d_.get("abstract") or ""}
                    fm.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lens = [len(rows[q]["text"]) // 4 for q in ids]
        stats[source] = {
            "rows": len(ids), "docs": len(seen_docs),
            "train": len(ids) - len(test_ids), "test": len(test_ids),
            "by_band": dict(Counter(hard.get(q, {}).get("band") for q in ids)),
            "by_style": dict(Counter(rows[q]["cell"]["style"] for q in ids)),
            "by_length": dict(Counter(rows[q]["cell"]["length"] for q in ids)),
            "no_truncation_gate": {
                "trainer_max_len": int(cv["max_len"]),
                "approx_max_query_tokens": max(lens) if lens else 0,
                "exceeds_max_len": sum(1 for x in lens if x > int(cv["max_len"])),
                "note": "approximate (chars/4); nothing is ever truncated"},
            "path": str(out / source)}
        log(f"CONVERT lane {source}: {stats[source]['rows']} rows, "
            f"{stats[source]['docs']} docs -> {out / source}")

    summary = {"lanes": stats, "survival_rejected": len(rejected),
               "subsample_to_target": {
                   "enabled": bool(cv.get("subsample_to_target", True)),
                   "surplus_dropped": len(surplus_dropped),
                   "units_short_of_target": sorted(
                       u for u, v in unit_table.items() if v.get("short_by")),
                   "per_unit": unit_table,
                   "note": ("A16 rounds overshoot on purpose; every unit is cut back to "
                            "its exact target here, deterministically by seed")},
               "dropped_pool_below_min": {"n": len(too_small),
                                          "min_pool": int(cv["min_pool"]),
                                          "note": "build_mixed_batches drops these "
                                                  "silently; excluded and counted here"},
               "instruction": cv["instruction"]}
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
