#!/usr/bin/env python3
"""00_import_questions.py — import questions written by earlier generation runs as stage-01 cells.

Two input lanes (`--pipeline5`, `--pipeline2`) are read. Each row keeps its question type or
group, register, entity, entity track, text and up to two sketches (the `synth:*` documents
of its old pool). A row with fewer than two sketches is marked `sketches_missing` for stage
02 to fill.

A row is imported only if the draw test passes:
  1. its type or group is one the current plan can draw (escalation and case_based rows
     from the first lane are excluded);
  2. its track is a current ontology track (`interventions` is renamed to `procedures`;
     `genes` and `labs` are dropped) and is allowed for its type or group;
  3. its entity exists in that track's entity list as the planner loads it.
A masked row whose text leaks its entity is also excluded.

Old pools and old grades are NOT imported. Every exclusion is counted by reason in
`import_summary.json`; nothing is dropped silently.

No LLM, no GPU, no network, no database."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config
from common.gates import leak_hit
from common.logging import Log, SkipLog
from common.seeds import seed_int, sha1


TYPE_RENAMES = {"management_criteria": "management",
                "prognosis": "epidemiology_prognosis",
                "screening": "screening_prevention"}

WRONG_WRITER_TYPES = {"escalation", "case_based"}

TRACK_RENAMES = {"interventions": "procedures"}
DROPPED_TRACKS = {"genes", "labs"}


def build_entity_index(entities_dir: Path, tracks, cfg=None, log=print
                       ) -> dict[str, dict[str, dict]]:
    """{track: {lowercased name or synonym: record}} for the concept-existence test.

    The index is built from the track as the planner sees it: stage 01's entity loader and
    filters, plus the grouper-name screen on diseases, so an imported row cannot sit on a
    concept a fresh cell could never draw."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "p7_plan_cells", str(Path(__file__).resolve().parent / "01_plan_cells.py"))
    pc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(pc)

    filters = (cfg or {}).get("entities_filter")
    loaded, report = pc.load_entities(entities_dir, set(tracks), filters, log)
    grouper = [re.compile(p, re.I)
               for p in ((cfg or {}).get("anchors") or {}).get("grouper_patterns", [])]
    screen_on = bool(((cfg or {}).get("anchors") or {}).get("grouper_screen", True))

    index: dict[str, dict[str, dict]] = {}
    screened_out: dict[str, list[str]] = {}
    for tr, recs in loaded.items():
        by_name: dict[str, dict] = {}
        dropped = []
        for rec in recs:
            name = (rec.get("display_name") or "").strip()
            if not name:
                continue
            if screen_on and tr == "diseases" and any(p.search(name) for p in grouper):
                dropped.append(name)
                continue
            by_name.setdefault(name.lower(), rec)
            for syn in rec.get("synonyms") or []:
                if (syn or "").strip():
                    by_name.setdefault(syn.strip().lower(), rec)
        index[tr] = by_name
        if dropped:
            screened_out[tr] = sorted(dropped)
            log(f"IMPORT_SCREEN {tr}: {len(dropped)} grouper-class names are not drawable "
                f"by this generator, so rows anchored on them are not reused")
    index["_report"] = {"track_filters": report, "grouper_screened": screened_out}
    return index


def subclass_index(cfg) -> dict[str, dict]:
    """{`v4_<type>__<subtype>` and `<type>.<subtype>`: the subclass record}."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "plan_cells", str(Path(__file__).resolve().parent / "01_plan_cells.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    rows = m.load_subclasses(Path(cfg["subclasses"]["taxonomy"]),
                             set(cfg["subclasses"].get("exclude_types") or []),
                             cfg["anchor_to_track"])
    out = {}
    for r in rows:
        out[r["id"]] = r
        out[f"v4_{r['type_id']}__{r['subtype_id']}"] = r
    return out


def read_source_rows(path: Path):
    """Yield (origin_id, question_type, register, entity, track, text, sketches)."""
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)

        sketches = [{"title": d.get("title") or "", "abstract": d.get("abstract") or ""}
                    for d in (r.get("pool") or [])
                    if str(d.get("work_id", "")).startswith("synth:")][:2]
        yield {"origin_id": r["qid"], "unit": r["question_type"],
               "register": r.get("register"), "entity": r["entity"],
               "track": r.get("entity_track"), "text": r["text"],
               "sketches": sketches,
               "working_notes": r.get("working_notes") or {},
               "grounding_ids": r.get("grounding_ids") or [],
               "generator_model": r.get("generator_model")}


def read_source_queries(path: Path):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pv = r.get("provenance") or {}
        yield {"origin_id": r["qid"], "unit": pv.get("group"),
               "register": pv.get("register"), "entity": pv.get("entity"),
               "track": pv.get("entity_track"), "text": r["text"],
               "sketches": [], "working_notes": {}, "grounding_ids": [],
               "generator_model": pv.get("generator_model")}


def draw_test(rec, lane, cfg, subclasses, entity_idx, tally):
    """-> (cell_source, unit_id, track, entity_record, subclass) or (None, reason)."""
    table = cfg["anchors_and_masking"]
    unit = rec["unit"]
    if not unit:
        return None, "no_type_or_group"

    if lane == "pipeline5":
        if unit in WRONG_WRITER_TYPES:
            return None, "wrong_writer_escalation_or_case_based"
        if unit.startswith("v4_"):
            sc = subclasses.get(unit)
            if sc is None:
                return None, "subclass_not_drawable_in_pipeline7"
            source, unit_id, allowed = "subclasses", sc["id"], sc["tracks"]
        else:
            unit_id = TYPE_RENAMES.get(unit, unit)
            if unit_id not in cfg["ely_types"]:
                return None, "type_not_in_the_eleven"
            source, allowed, sc = "ely_types", table[unit_id]["tracks"], None
    else:
        unit_id = unit
        if unit_id not in cfg["failure_group_weights"]:
            return None, "group_not_in_the_eight"
        source, allowed, sc = "failure_groups", table[unit_id]["tracks"], None

    track = TRACK_RENAMES.get(rec["track"], rec["track"])
    if track in DROPPED_TRACKS:
        return None, f"track_dropped:{rec['track']}"
    if track not in entity_idx:
        return None, f"track_unknown:{rec['track']}"
    if track not in allowed:
        return None, f"track_not_allowed_for_{unit_id}"

    ent = entity_idx[track].get((rec["entity"] or "").strip().lower())
    if ent is None:
        return None, "concept_not_in_track_list"
    return (source, unit_id, track, ent, sc), None


def to_cell(rec, source, unit_id, track, ent, sc, cfg, run_id, idx):
    """Emit the cell in exactly the shape stage 01 produces."""
    table = cfg["anchors_and_masking"]
    masking = sc["masking"] if sc else table[unit_id]["masking"]
    if masking == "required":
        masked = True
    elif masking == "impossible":
        masked = False
    else:
        # `optional` was never resolved for an imported row; a fair coin on the row's own
        # seed reproduces stage 01's rule deterministically.
        masked = seed_int(run_id, rec["origin_id"], "mask") % 2 == 0
    cell_id = f"{source}-{unit_id}-imp{idx:05d}"
    style = rec["register"] if rec["register"] in ("clean", "messy") else "clean"
    return {
        "cell_id": cell_id, "run_id": run_id, "seed": seed_int(run_id, cell_id),
        "source": source, "type_or_group": unit_id,
        "grounded": bool(cfg["grounding"][source]),
        "masking": masking, "masked": masked, "track": track,
        "entity": {"track": track, "id": ent.get("id"),
                   "display_name": ent["display_name"],
                   "synonyms": [s for s in (ent.get("synonyms") or []) if s]},

        # The question text already exists, so no length is drawn; the axis is recorded as `imported`.
        "length": "imported", "pasted": rec["register"] == "pasted",
        "document_type": None, "style": style,
        **({"subclass": {k: sc[k] for k in ("type_id", "type_name", "subtype_id",
                                            "subtype_name", "subtype_definition",
                                            "ask")}} if sc else {}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--pipeline5", default=None, help="override the config path")
    ap.add_argument("--pipeline2", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("00_import")
    out.mkdir(parents=True, exist_ok=True)
    log, skip = Log(out / "run.log"), SkipLog(out / "skips.log")
    run_id = cfg["run"]["run_id"]

    imp = cfg["import_prior"]
    lanes = {"pipeline5": (Path(a.pipeline5 or imp["source_rows"]), read_source_rows),
             "pipeline2": (Path(a.pipeline2 or imp["source_queries"]), read_source_queries)}

    entity_idx = build_entity_index(Path(cfg["paths"]["entities_dir"]),
                                    set(cfg["anchor_to_track"].values()), cfg, log)
    idx_report = entity_idx.pop("_report", {})
    subclasses = subclass_index(cfg)

    cells, questions = [], []
    tally = {lane: {"read": 0, "kept": 0,
                    "excluded": collections.Counter(),
                    "by_source": collections.Counter(),
                    "by_track": collections.Counter(),
                    "sketches_missing": 0} for lane in lanes}
    idx = 0
    for lane, (path, reader) in lanes.items():
        if not path.exists():
            raise SystemExit(f"FATAL: {lane} lane not found at {path}")
        log(f"IMPORT reading {lane} from {path}")
        for rec in reader(path):
            tally[lane]["read"] += 1
            got, reason = draw_test(rec, lane, cfg, subclasses, entity_idx, tally)
            if got is None:
                tally[lane]["excluded"][reason] += 1
                skip.note("IMPORT_EXCLUDE", lane=lane, origin=rec["origin_id"],
                          unit=rec["unit"], track=rec["track"], reason=reason)
                continue
            source, unit_id, track, ent, sc = got
            idx += 1
            cell = to_cell(rec, source, unit_id, track, ent, sc, cfg, run_id, idx)

            # A masked row whose text names its entity would fail stage 02's leak check,
            # so it is excluded here.
            if cell.get("masked"):
                e = cell.get("entity") or {}
                banned = list((rec.get("working_notes") or {}).get("banned_terms") or [])
                hit = leak_hit([rec["text"]], e.get("display_name") or "",
                               e.get("synonyms") or [], banned, check_entity=True)
                if hit:
                    idx -= 1
                    tally[lane]["excluded"]["leak"] += 1
                    skip.note("IMPORT_EXCLUDE", lane=lane, origin=rec["origin_id"],
                              unit=rec["unit"], track=rec["track"], reason=f"leak:{hit}")
                    continue
            cells.append(cell)
            sketches = rec["sketches"][:2]
            if len(sketches) < 2:
                tally[lane]["sketches_missing"] += 1
            questions.append({
                "qid": f"{run_id}-{cell['cell_id']}", "cell_id": cell["cell_id"],
                "status": "ok", "text": rec["text"], "text_sha1": sha1(rec["text"]),
                "sketches": sketches,
                "sketches_missing": len(sketches) < 2,
                "working_notes": rec["working_notes"],
                "grounding_ids": rec["grounding_ids"],
                "generator_model": rec["generator_model"],
                "prompt_sha1": None,
                "provenance": {"origin": f"pipeline6_{lane}",
                               "origin_id": rec["origin_id"],
                               "origin_question_type": rec["unit"],
                               "origin_register": rec["register"],
                               "origin_track": rec["track"]},
                "cell": cell})
            tally[lane]["kept"] += 1
            tally[lane]["by_source"][source] += 1
            tally[lane]["by_track"][track] += 1
            if a.limit and len(cells) >= a.limit:
                break

    with (out / "cells.jsonl").open("w") as fh:
        for c in cells:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (out / "questions.jsonl").open("w") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id, "config": cfg.path,
        "lanes": {lane: {"path": str(lanes[lane][0]), "read": t["read"],
                         "kept": t["kept"],
                         "excluded_total": sum(t["excluded"].values()),
                         "excluded_by_reason": dict(t["excluded"]),
                         "by_source": dict(t["by_source"]),
                         "by_track": dict(t["by_track"]),
                         "sketches_missing": t["sketches_missing"]}
                  for lane, t in tally.items()},
        "kept_total": len(cells),
        "by_source": dict(collections.Counter(c["source"] for c in cells)),
        "by_track": dict(collections.Counter(c["track"] for c in cells)),
        "by_style": dict(collections.Counter(c["style"] for c in cells)),
        "masked_applied": sum(1 for c in cells if c["masked"]),
        "sketches_missing_total": sum(1 for q in questions if q["sketches_missing"]),
        "entity_universe": {
            "track_filters": {tr: {k: v for k, v in r.items()
                                   if k in ("before", "kept", "dropped")}
                              for tr, r in (idx_report.get("track_filters") or {}).items()},
            "grouper_screened": {tr: len(v) for tr, v in
                                 (idx_report.get("grouper_screened") or {}).items()},
            "note": ("A concept this generator would never draw "
                     "is not a concept it could have drawn, so the row is not reused"),
        },
        "note": ("old pools and old grades are NOT imported; pools are rebuilt by recipe "
                 "R in stage 04 and every grade is redone by the grader in "
                 "stage 05"),
    }
    (out / "import_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"IMPORT kept {len(cells)} rows -> {out/'questions.jsonl'}")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
