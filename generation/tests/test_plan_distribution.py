#!/usr/bin/env python3
"""test_plan_distribution.py — stage 01 draws what the config says it draws.

Checks, at a scale where sampling noise is small (6,000 cells):
  * source shares are exactly one third each (largest-remainder, not sampling)
  * within ely_types the eleven types are uniform; within failure_groups the eight
    groups follow the measured weights; subclasses draw only from the drawable,
    non-excluded taxonomy subtypes
  * every cell's allowed tracks and masking tag match the config's `anchors_and_masking` table
  * masking is applied per tag: required -> always hidden, impossible -> always named,
    optional -> a fair coin
  * style clean 0.625 / messy 0.375 and length .30/.40/.30 within tolerance
  * pasted is 20% of all cells (two of three long cells) within tolerance
  * document types are drawn only for pasted cells, from the six configured types
  * the plan is DETERMINISTIC: the same (run_id, config) gives byte-identical cells

Top-up mode and the pilot slice:
  * per-unit targets are one third of `volume.total_rows` per source, apportioned over
    that source's units
  * kept + fresh == target for EVERY unit, and the planned total is exactly the configured total
  * a unit with more reusable rows than its target is subsampled, deterministically;
    a unit with fewer keeps all of them and the rest is drawn fresh
  * a unit with no reusable rows is planned entirely fresh
  * the kept/dropped split is a partition of the imported rows, with no row in both
  * the pilot slice reaches every one of the eleven types and eight groups at least
    twice at N=60, and planning it does not change the full plan

Concept filters:
  * the excluded taxonomy types and subtypes are gone from the drawable universe, and
    the per-subclass target is recomputed from what is left
  * the vaccines track keeps only disease-level names, and the kept/dropped lists are
    persisted with the run
  * the grouper screen rejects five MONDO classification nodes and keeps seven real
    diagnoses that must never be screened out

Round-based over-generation and the stage-07 subsample to exact targets are also checked.

Runs with no network, no GPU, no LLM. Executable directly or under pytest.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parent
sys.path.insert(0, str(GEN))

import importlib.util                                       # noqa: E402
from common.config import load_config                       # noqa: E402
from common.seeds import largest_remainder                  # noqa: E402

_sp = importlib.util.spec_from_file_location("plan_cells", str(GEN / "01_plan_cells.py"))
plan_cells = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(plan_cells)

N = 6000
TOL = 0.02          # absolute tolerance on a proportion; 3.2 sigma at n=6000, p=0.3
FAILURES: list[str] = []


def check(ok: bool, msg: str):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILURES.append(msg)


def close(got: float, want: float, tol: float = TOL) -> bool:
    return abs(got - want) <= tol


def build(total=N, run_id="p7test"):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cfg["volume"]["total_rows"] = total
    cfg["run"]["run_id"] = run_id
    cells, summary = plan_cells.plan(cfg, log=lambda *_: None)
    return cfg, cells, summary


def main() -> int:
    cfg, cells, summary = build()
    n = len(cells)
    check(n == N, f"planned exactly {N} cells (got {n})")

    # ── source shares: apportionment, so exact ──
    per_source = Counter(c["source"] for c in cells)
    want = largest_remainder({s: 1.0 for s in plan_cells.SOURCES}, N)
    check(dict(per_source) == want, f"source shares are one third each: {dict(per_source)}")

    # ── within-source weights ──
    ely = Counter(c["type_or_group"] for c in cells if c["source"] == "ely_types")
    check(len(ely) == 11, f"all eleven Ely types drawn (got {len(ely)})")
    check(max(ely.values()) - min(ely.values()) <= 1,
          f"Ely types uniform to within one row: {min(ely.values())}..{max(ely.values())}")

    groups = Counter(c["type_or_group"] for c in cells
                     if c["source"] == "failure_groups")
    gw = cfg["failure_group_weights"]
    want_g = largest_remainder({k: float(v) for k, v in gw.items()},
                               per_source["failure_groups"])
    check(dict(groups) == want_g, f"failure-group weights are the measured shares: {dict(groups)}")
    check(set(gw) == {"stacked_constraints", "case_based", "escalation",
                      "rare_pair_causation", "symptoms_unnamed", "timing_cutoffs",
                      "contradictory_tests", "comparative_superlative"},
          "the EIGHT failure groups are the six NLM-derived plus case_based and escalation")

    drawable = {r["id"] for r in plan_cells.load_subclasses(
        Path(cfg["subclasses"]["taxonomy"]),
        set(cfg["subclasses"]["exclude_types"]), cfg["anchor_to_track"])}
    drawn = {c["type_or_group"] for c in cells if c["source"] == "subclasses"}
    check(drawn <= drawable, f"subclasses drawn only from the {len(drawable)} drawable ones")
    excluded_hit = [c for c in cells if c["source"] == "subclasses"
                    and c["subclass"]["type_id"] in cfg["subclasses"]["exclude_types"]]
    check(not excluded_hit, "no cell drawn from an excluded taxonomy type")

    # ── tracks and masking tags match the config table ──
    table = cfg["anchors_and_masking"]
    bad_track = [c["cell_id"] for c in cells
                 if c["source"] != "subclasses" and c["track"] not in
                 table[c["type_or_group"]]["tracks"]]
    check(not bad_track, f"every track is an allowed track for its type/group "
                         f"({len(bad_track)} violations)")
    bad_tag = [c["cell_id"] for c in cells
               if c["source"] != "subclasses" and c["masking"] !=
               table[c["type_or_group"]]["masking"]]
    check(not bad_tag, f"every masking tag matches the anchor table ({len(bad_tag)} violations)")
    check(all(c["track"] in ("diseases", "drugs", "procedures", "organisms", "toxins",
                             "vaccines") for c in cells),
          "only the six ontology tracks are used (genes and labs dropped)")

    # ── masking applied per tag ──
    req = [c for c in cells if c["masking"] == "required"]
    imp = [c for c in cells if c["masking"] == "impossible"]
    opt = [c for c in cells if c["masking"] == "optional"]
    check(all(c["masked"] for c in req), f"required -> always hidden ({len(req)} cells)")
    check(not any(c["masked"] for c in imp), f"impossible -> always named ({len(imp)} cells)")
    frac = sum(c["masked"] for c in opt) / max(len(opt), 1)
    check(close(frac, 0.5, 0.05), f"optional -> fair coin (masked fraction {frac:.3f} "
                                  f"of {len(opt)})")

    # ── style ──
    style = Counter(c["style"] for c in cells)
    check(close(style["clean"] / n, 0.625), f"clean share {style['clean']/n:.3f} ~ 0.625")
    check(close(style["messy"] / n, 0.375), f"messy share {style['messy']/n:.3f} ~ 0.375")

    # ── length and pasted ──
    length = Counter(c["length"] for c in cells)
    for k, want_p in (("short", 0.30), ("medium", 0.40), ("long", 0.30)):
        check(close(length[k] / n, want_p), f"{k} share {length[k]/n:.3f} ~ {want_p}")
    pasted = sum(1 for c in cells if c["pasted"])
    check(close(pasted / n, 0.20), f"pasted share {pasted/n:.3f} ~ 0.20 (2/3 of long)")
    check(all(c["length"] == "long" for c in cells if c["pasted"]),
          "pasted is only ever a subtype of long")
    check(all(c["document_type"] for c in cells if c["pasted"]),
          "every pasted cell carries a document_type")
    check(not any(c["document_type"] for c in cells if not c["pasted"]),
          "no document_type is drawn for a non-pasted cell")
    doc_types = {c["document_type"] for c in cells if c["pasted"]}
    check(doc_types == set(cfg["lengths"]["document_types"]),
          f"all six document types appear: {sorted(doc_types)}")

    # ── determinism ──
    _, cells2, _ = build()
    check(json.dumps(cells, sort_keys=True) == json.dumps(cells2, sort_keys=True),
          "same (run_id, config) -> byte-identical cells")
    _, cells3, _ = build(run_id="p7test-other")
    check(json.dumps(cells, sort_keys=True) != json.dumps(cells3, sort_keys=True),
          "a different run_id gives a different plan")

    # ── the summary reports what it drew ──
    for key in ("by_source", "by_masking_tag", "by_style", "by_length",
                "pasted_fraction", "draw_order", "seed_recipe"):
        check(key in summary, f"plan_summary reports {key}")

    check_topup(cfg)
    check_pilot(cfg)
    check_round_topup(cfg)
    check_subsample(cfg)
    check_exclusions(cfg)
    check_vaccines_filter(cfg)
    check_grouper_screen(cfg)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def fake_imported(cfg, counts: dict) -> list[dict]:
    """`counts` is {(source, unit): n} -> imported rows in stage 00's questions schema."""
    rows = []
    for (source, unit), n in counts.items():
        for i in range(n):
            cell = {"cell_id": f"{source}-{unit}-imp{i:05d}", "run_id": "p7test",
                    "seed": i, "source": source, "type_or_group": unit,
                    "grounded": bool(cfg["grounding"][source]), "masking": "impossible",
                    "masked": False, "track": "diseases",
                    "entity": {"track": "diseases", "id": "X", "display_name": "x",
                               "synonyms": []},
                    "length": "imported", "pasted": False, "document_type": None,
                    "style": "clean"}
            rows.append({"qid": f"p7test-{cell['cell_id']}", "cell_id": cell["cell_id"],
                         "status": "ok", "text": f"q{i}", "sketches": [],
                         "provenance": {"origin": "pipeline6_pipeline2"}, "cell": cell})
    return rows


def check_topup(cfg_in):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cfg["run"]["run_id"] = "p7test"
    units = plan_cells.source_units(cfg, log=lambda *_: None)
    total = int(cfg["volume"]["total_rows"])
    _per_source, targets = plan_cells.unit_targets(cfg, units, total,
                                                   log=lambda *_: None)
    check(sum(sum(v.values()) for v in targets.values()) == total,
          f"the per-unit targets sum to volume.total_rows ({total})")
    for s in plan_cells.SOURCES:
        check(sum(targets[s].values()) == total // 3,
              f"{s}: targets sum to {total // 3}")
    ely = targets["ely_types"]
    check(max(ely.values()) - min(ely.values()) <= 1,
          f"the eleven Ely targets are uniform ({min(ely.values())}-{max(ely.values())})")

    # one unit over target, one under, one with nothing at all
    over = ("failure_groups", "stacked_constraints")
    under = ("ely_types", "diagnosis")
    empty = ("ely_types", "diagnostic_criteria")
    imported = fake_imported(cfg, {over: targets[over[0]][over[1]] + 200,
                                   under: 10})
    fresh, kept, dropped, table, summary = plan_cells.plan_topup(
        cfg, imported, log=lambda *_: None)

    bad = [(s, u) for s in table for u, v in table[s].items()
           if v["kept"] + v["fresh"] != v["target"]]
    check(not bad, f"kept + fresh == target for EVERY unit ({len(bad)} violations)")
    check(summary["planned_total"] == total,
          f"planned total is exactly {total} (got {summary['planned_total']})")
    check(len(fresh) == summary["fresh_rows"]
          and sum(len(v) for v in kept.values()) == summary["kept_imported_rows"],
          "the summary counts the rows actually returned")

    o = table[over[0]][over[1]]
    check(o["kept"] == o["target"] and o["fresh"] == 0,
          f"a unit with more reusable rows than its target is subsampled to it "
          f"({o['reusable']} reusable -> {o['kept']} kept)")
    u = table[under[0]][under[1]]
    check(u["kept"] == 10 and u["fresh"] == u["target"] - 10,
          f"a unit with fewer keeps all of them and draws the rest ({u})")
    e = table[empty[0]][empty[1]]
    check(e["kept"] == 0 and e["fresh"] == e["target"],
          f"a unit with no reusable rows is planned entirely fresh ({e})")

    kept_ids = {q["cell_id"] for v in kept.values() for q in v}
    drop_ids = {q["cell_id"] for q in dropped}
    check(not (kept_ids & drop_ids), "no imported row is both kept and dropped")
    check(kept_ids | drop_ids == {q["cell_id"] for q in imported},
          "kept + dropped is a partition of the imported rows")
    check(len(dropped) == 200, f"exactly the surplus is dropped (got {len(dropped)})")

    fresh2, kept2, _, _, _ = plan_cells.plan_topup(cfg, imported, log=lambda *_: None)
    check(json.dumps(fresh, sort_keys=True) == json.dumps(fresh2, sort_keys=True)
          and kept_ids == {q["cell_id"] for v in kept2.values() for q in v},
          "top-up is deterministic: same inputs, same kept set and same fresh cells")
    check(all(c["type_or_group"] in targets[c["source"]] for c in fresh),
          "every fresh cell names a unit that has a target")


def check_pilot(cfg_in):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cfg["run"]["run_id"] = "p7test"
    cells, summary = plan_cells.plan_pilot(cfg, 60, log=lambda *_: None)
    check(len(cells) == 180, f"--pilot-per-source 60 plans 180 cells (got {len(cells)})")
    per = Counter(c["source"] for c in cells)
    check(set(per.values()) == {60}, f"60 per source: {dict(per)}")
    for source, n_units in (("ely_types", 11), ("failure_groups", 8)):
        got = Counter(c["type_or_group"] for c in cells if c["source"] == source)
        check(len(got) == n_units,
              f"the pilot reaches all {n_units} {source} units (got {len(got)})")
        check(min(got.values()) >= 2,
              f"every {source} unit gets at least 2 pilot cells "
              f"(min {min(got.values())})")
    check(all(c["cell_id"].startswith("pilot-") for c in cells),
          "pilot cell ids carry the `pilot-` prefix, so their seeds are their own")
    full, _ = plan_cells.plan(cfg, log=lambda *_: None)
    check(not ({c["cell_id"] for c in cells} & {c["cell_id"] for c in full}),
          "no pilot cell id collides with a full-plan cell id")
    full2, _ = plan_cells.plan(cfg, log=lambda *_: None)
    check(json.dumps(full, sort_keys=True) == json.dumps(full2, sort_keys=True),
          "planning a pilot leaves the full plan unchanged")


# ── round-based over-generation ──
def check_round_topup(cfg_in):
    """A round plans target - kept - written per unit, divided by the measured yield."""
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cfg["run"]["run_id"] = "p7round"
    og = cfg["overgeneration"]
    check(og["min_yield"] == 0.35 and og["max_overgen"] == 3.0,
          f"config: yield floor {og['min_yield']}, multiplier cap {og['max_overgen']}")

    units = plan_cells.source_units(cfg, log=lambda *_: None)
    _ps, targets = plan_cells.unit_targets(cfg, units,
                                           int(cfg["volume"]["total_rows"]),
                                           log=lambda *_: None)
    U = ("ely_types", "diagnosis")
    V = ("failure_groups", "stacked_constraints")
    t_u, t_v = targets[U[0]][U[1]], targets[V[0]][V[1]]

    def cells_for(source, unit, n, tag):
        return [{"cell_id": f"{source}-{unit}-{tag}{i:04d}", "source": source,
                 "type_or_group": unit} for i in range(n)]

    def settled(*pairs):
        """{(source, unit): terminal_count} so a round counts as FINISHED: every cell it
        attempted has an outcome, a written row or a terminal skip."""
        return {(s, u): max(0, att - ok) for s, u, att, ok in pairs}

    def written_for(source, unit, n, tag, status="ok"):
        return [{"qid": f"q-{source}-{unit}-{tag}{i}", "status": status,
                 "cell": {"cell_id": f"{source}-{unit}-{tag}{i:04d}", "source": source,
                          "type_or_group": unit}} for i in range(n)]

    # round 1 tried 100 cells for U and got 49 back (the measured Ely yield);
    # V tried 100 and got 97. Nothing imported.
    attempted = cells_for(*U, 100, "r1") + cells_for(*V, 100, "r1")
    written = (written_for(*U, 49, "r1") + written_for(*V, 97, "r1")
               + written_for(*U, 7, "bad", status="discard"))
    cells, kept, dropped, table, summary = plan_cells.plan_round(
        cfg, [], attempted, written, 2, log=lambda *_: None,
        terminal_by_unit=settled((*U, 100, 49), (*V, 100, 97)))

    ru, rv = table[U[0]][U[1]], table[V[0]][V[1]]
    check(ru["written"] == 49,
          f"only status ok counts as written ({ru['written']}, discards ignored)")
    check(abs(ru["yield"] - 0.49) < 1e-9,
          f"the unit yield is measured, not assumed ({ru['yield']})")
    check(ru["remaining"] == t_u - 0 - 49,
          f"remaining = target - kept - written ({ru['remaining']})")
    import math as _m
    check(ru["planned"] == _m.ceil(ru["remaining"] / 0.49),
          f"planned = ceil(remaining / yield) ({ru['planned']} for "
          f"{ru['remaining']} remaining at {ru['yield']})")
    check(rv["planned"] == _m.ceil(rv["remaining"] / rv["yield"])
          and rv["overgen_multiplier"] < 1.1,
          f"a high-yield unit barely over-generates ({rv['overgen_multiplier']}x)")

    # the floor and the cap
    low = plan_cells.plan_round(
        cfg, [], cells_for(*U, 100, "r1"), written_for(*U, 5, "r1"), 2,
        log=lambda *_: None, terminal_by_unit=settled((*U, 100, 5)))[3][U[0]][U[1]]
    check(low["yield"] == 0.05 and low["yield_used"] == og["min_yield"],
          f"a catastrophic yield is floored at min_yield ({low['yield']} -> "
          f"{low['yield_used']})")
    check(low["overgen_multiplier"] <= og["max_overgen"],
          f"the multiplier never exceeds max_overgen ({low['overgen_multiplier']})")
    check(low["planned"] == _m.ceil(low["remaining"] / og["min_yield"]),
          "the floored yield is what the plan divides by")

    # a unit with too little evidence borrows its source's rate
    thin = plan_cells.plan_round(
        cfg, [], cells_for(*U, 100, "r1") + cells_for("ely_types", "dosing", 2, "r1"),
        written_for(*U, 49, "r1") + written_for("ely_types", "dosing", 1, "r1"), 2,
        log=lambda *_: None,
        terminal_by_unit=settled((*U, 100, 49), ("ely_types", "dosing", 2, 1)),
        )[3]["ely_types"]["dosing"]
    check(thin["yield_basis"] == "source",
          f"a unit with < min_attempts borrows its source's yield ({thin['yield_basis']})")
    none_yet = plan_cells.plan_round(
        cfg, [], cells_for(*U, 100, "r1"), written_for(*U, 49, "r1"), 2,
        log=lambda *_: None, terminal_by_unit=settled((*U, 100, 49)))[3]["subclasses"]
    a_sub = next(iter(none_yet.values()))
    check(a_sub["yield_basis"] == "unmeasured" and a_sub["overgen_multiplier"] == 1.0,
          f"an unmeasured source plans no over-generation rather than guessing "
          f"({a_sub['yield_basis']}, {a_sub['overgen_multiplier']}x)")

    # a unit already full plans nothing
    full = plan_cells.plan_round(
        cfg, [], cells_for(*V, 400, "r1"), written_for(*V, t_v, "r1"), 2,
        log=lambda *_: None,
        terminal_by_unit=settled((*V, 400, t_v)))[3][V[0]][V[1]]
    check(full["remaining"] == 0 and full["planned"] == 0,
          f"a unit that reached its target plans nothing ({full})")

    # imported rows still count toward the target
    imported = fake_imported(cfg, {V: 50})
    with_imp = plan_cells.plan_round(
        cfg, imported, cells_for(*V, 100, "r1"), written_for(*V, 97, "r1"), 2,
        log=lambda *_: None,
        terminal_by_unit=settled((*V, 100, 97)))[3][V[0]][V[1]]
    check(with_imp["kept"] == 50
          and with_imp["remaining"] == t_v - 50 - 97,
          f"remaining subtracts BOTH kept imported and written rows ({with_imp})")

    # a round still in flight must not be measured
    part_att = cells_for(*U, 1000, "r1")
    part_written = written_for(*U, 20, "r1")
    try:
        plan_cells.plan_round(cfg, [], part_att, part_written, 2, log=lambda *_: None)
        check(False, "an unfinished round should refuse to be measured")
    except SystemExit as exc:
        check("still in flight" in str(exc),
              "an unfinished round REFUSES to plan and says why")
    forced = plan_cells.plan_round(cfg, [], part_att, part_written, 2,
                                   log=lambda *_: None, allow_unsettled=True)[4]
    check(forced["unsettled_units"],
          "--allow-unsettled plans anyway and NAMES the units it measured mid-flight")
    resolved = plan_cells.plan_round(
        cfg, [], part_att, part_written, 2, log=lambda *_: None,
        terminal_by_unit={U: 980})[3][U[0]][U[1]]
    check(resolved["yield"] == 0.02,
          f"terminal skips settle a round: 20 written + 980 skipped of 1000 "
          f"({resolved['yield']})")

    # ids and seeds are the round's own
    check(all(f"-r2-" in c["cell_id"] for c in cells),
          "every round-2 cell id carries its round")
    r3 = plan_cells.plan_round(cfg, [], attempted, written, 3, log=lambda *_: None,
                               terminal_by_unit=settled((*U, 100, 49),
                                                        (*V, 100, 97)))[0]
    check(not ({c["cell_id"] for c in cells} & {c["cell_id"] for c in r3}),
          "no cell id repeats across rounds")
    seeds2 = {c["cell_id"]: c["seed"] for c in cells}
    check(len(set(seeds2.values())) == len(seeds2),
          "every round-2 cell has its own seed")
    again = plan_cells.plan_round(cfg, [], attempted, written, 2, log=lambda *_: None,
                                  terminal_by_unit=settled((*U, 100, 49),
                                                           (*V, 100, 97)))[0]
    check(json.dumps(cells, sort_keys=True) == json.dumps(again, sort_keys=True),
          "a round is deterministic: same inputs, same cells")
    check(summary["planned_total"] == len(cells)
          and summary["mode"] == "round2" and summary["round"] == 2,
          "the round summary reports what it planned")
    for key in ("target", "kept", "written", "remaining", "yield", "planned"):
        check(key in ru, f"plan_summary per_unit reports {key}")


def check_subsample(cfg_in):
    """Stage 07 cuts every unit back to its exact target, deterministically."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "convert_lanes", str(GEN / "07_convert_lanes.py"))
    cv = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(cv)

    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cfg["run"]["run_id"] = "p7sub"
    check(bool(cfg["convert"].get("subsample_to_target")),
          "config: stage 07 subsamples each unit to its target")
    targets = cv.unit_targets_for(cfg)
    U, V = ("ely_types", "diagnosis"), ("failure_groups", "stacked_constraints")
    t_u, t_v = targets[U[0]][U[1]], targets[V[0]][V[1]]

    rows, kept = {}, []
    def add(source, unit, n, tag):
        for i in range(n):
            qid = f"{source}-{unit}-{tag}{i:04d}"
            rows[qid] = {"cell": {"source": source, "type_or_group": unit}}
            kept.append(qid)
    add(*U, t_u + 40, "over")             # overshot
    add(*V, t_v - 10, "under")            # still short
    add("subclasses", "retired.subtype", 7, "gone")   # no longer a unit

    out, dropped, table = cv.subsample_to_target(cfg, kept, rows, log=lambda *_: None)
    check(len(out) == t_u + (t_v - 10),
          f"the overshooting unit is cut and the short one is left alone ({len(out)})")
    check(table[f"{U[0]}/{U[1]}"]["kept"] == t_u
          and table[f"{U[0]}/{U[1]}"]["surplus"] == 40,
          f"the surplus is exactly the overshoot ({table[f'{U[0]}/{U[1]}']})")
    check(table[f"{V[0]}/{V[1]}"]["short_by"] == 10
          and table[f"{V[0]}/{V[1]}"]["surplus"] == 0,
          f"a unit under target is reported short, not padded "
          f"({table[f'{V[0]}/{V[1]}']})")
    check(table["subclasses/retired.subtype"]["kept"] == 0,
          "rows of a unit the config no longer targets are dropped and named")
    check(len(dropped) == 40 + 7 and not (set(out) & set(dropped)),
          f"kept and dropped partition the input ({len(out)} + {len(dropped)})")
    out2, _, _ = cv.subsample_to_target(cfg, list(reversed(kept)), rows,
                                        log=lambda *_: None)
    check(out == out2,
          "the subsample is deterministic and independent of input order")


# ── the taxonomy exclusions ──
A9_TYPES = ["drug_product", "anatomy_terms", "calculation", "normal_values"]
A9_SUBTYPE = "disease_identity.name_recall_from_description"


def check_exclusions(cfg_in):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    sc = cfg["subclasses"]
    for ty in A9_TYPES:
        check(ty in sc["exclude_types"],
              f"`{ty}` is excluded — its questions are not answered by papers")
    check(A9_SUBTYPE in (sc.get("exclude_subtypes") or []),
          f"`{A9_SUBTYPE}` is excluded")

    tax, a2t = Path(sc["taxonomy"]), cfg["anchor_to_track"]
    all_rows = plan_cells.load_subclasses(tax, set(), a2t)
    kept = plan_cells.load_subclasses(tax, set(sc["exclude_types"]), a2t,
                                      set(sc["exclude_subtypes"]))
    ids = {r["id"] for r in kept}
    check(len(kept) < len(all_rows),
          f"the drawable universe shrank ({len(all_rows)} -> {len(kept)})")
    for ty in A9_TYPES:
        check(not any(r["type_id"] == ty for r in kept),
              f"no `{ty}` subclass survives the exclusion")
    check(A9_SUBTYPE not in ids, f"`{A9_SUBTYPE}` itself is gone")
    check(any(r["type_id"] == "disease_identity" for r in kept),
          "excluding one subtype does not remove its whole type")

    units = plan_cells.source_units(cfg, log=lambda *_: None)
    check(len(units["subclasses"]) == len(kept),
          f"the planner's unit universe is the filtered one ({len(kept)})")
    _ps, targets = plan_cells.unit_targets(cfg, units,
                                           int(cfg["volume"]["total_rows"]),
                                           log=lambda *_: None)
    sub = targets["subclasses"]
    check(sum(sub.values()) == 1920 and max(sub.values()) - min(sub.values()) <= 1,
          f"the per-subclass target is recomputed from the new universe "
          f"({min(sub.values())}-{max(sub.values())} each over {len(sub)})")


# ── the vaccines track filter ──
def check_vaccines_filter(cfg_in):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    rules = (cfg.get("entities_filter") or {}).get("vaccines")
    check(bool(rules) and rules.get("enabled"),
          "the vaccines track carries a mechanical name filter")

    reason = plan_cells.track_filter_reason
    for name in ("Influenza, MDCK, quadrivalent, PF",
                 "COVID-19 PS Non-US Vaccine (Medigen, MVC-COV1901)",
                 "Hep B, adult", "rotavirus, pentavalent"):
        check(reason(name, rules) is not None,
              f"product-level name rejected — {name!r}")
    for name in ("BCG", "Tdap", "HPV9", "measles", "plague", "DTaP-IPV", "IPV",
                 "smallpox", "yellow fever"):
        check(reason(name, rules) is None,
              f"disease-level name kept — {name!r} "
              f"({reason(name, rules)})")

    ents, rep = plan_cells.load_entities(Path(cfg["paths"]["entities_dir"]),
                                         {"vaccines"}, cfg.get("entities_filter"),
                                         log=lambda *_: None)
    r = rep["vaccines"]
    check(r["kept"] + r["dropped"] == r["before"],
          f"kept + dropped == the whole track ({r['kept']} + {r['dropped']} "
          f"= {r['before']})")
    check(len(ents["vaccines"]) == r["kept"],
          "the planner draws from the FILTERED track")
    check(all("," not in e["display_name"] and "(" not in e["display_name"]
              for e in ents["vaccines"]),
          "no product spec survives into the draw")
    check(r["kept_names"] and r["dropped_names"],
          "both lists are recorded, so the filter is auditable, not just counted")
    check(all(len(d) == 2 and d[1] for d in r["dropped_names"]),
          "every dropped name carries the reason it was dropped")

    cfg2 = load_config(str(GEN / "configs/generation.yaml"))
    cfg2["entities_filter"]["vaccines"]["enabled"] = False
    ents2, rep2 = plan_cells.load_entities(Path(cfg["paths"]["entities_dir"]),
                                           {"vaccines"}, cfg2.get("entities_filter"),
                                           log=lambda *_: None)
    check(not rep2 and len(ents2["vaccines"]) > len(ents["vaccines"]),
          "the filter is a config switch, not a hardcoded list")


# ── the grouper screen ──
GROUPER_REJECT = ["eye infectious disorder", "eye accommodation disease",
                  "pregnancy disorder with abortive outcome", "acidosis disorder",
                  "burkholderia infectious disease",
                  "malignant urinary system neoplasm", "unspecified anemia",
                  "disorder of eye", "syndrome"]
GROUPER_KEEP = ["typhoid fever", "Crohn disease", "Ebstein anomaly", "leukopenia",
                "steroid-induced glaucoma", "tumor lysis syndrome",
                "Chiari malformation type I", "cystic fibrosis",
                "Guillain-Barre syndrome", "acute myeloid leukemia"]


def check_grouper_screen(cfg_in):
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    check(bool(cfg["anchors"].get("grouper_screen")),
          "the grouper screen is ON, for the diseases track of every source")
    pats = [re.compile(p, re.I) for p in cfg["anchors"]["grouper_patterns"]]

    def screened(name):
        return any(p.search(name) for p in pats)

    for name in GROUPER_REJECT:
        check(screened(name), f"REJECTED as a classification node — {name!r}")
    for name in GROUPER_KEEP:
        check(not screened(name), f"KEPT as a real diagnosis — {name!r}")

    # the screen has to actually run at draw time, for every source
    units = plan_cells.source_units(cfg, log=lambda *_: None)
    d = plan_cells.Drawer(cfg, units, log=lambda *_: None)
    disease_names = {e["display_name"] for e in d.entities["diseases"]}
    hits = sorted(n for n in disease_names if screened(n))
    check(len(hits) > 100,
          f"the screen matches {len(hits)} names in the real MONDO track")
    for probe in ("eye infectious disorder", "acidosis disorder"):
        if probe in disease_names:
            check(probe in hits, f"the pilot's own offender is caught — {probe!r}")

    cfg2 = load_config(str(GEN / "configs/generation.yaml"))
    cfg2["volume"]["total_rows"] = 3000
    cfg2["run"]["run_id"] = "p7screen"
    cells, summary = plan_cells.plan(cfg2, log=lambda *_: None)
    drawn = [c["entity"]["display_name"] for c in cells if c["track"] == "diseases"]
    leaked = sorted({n for n in drawn if screened(n)})
    check(not leaked,
          f"no screened name reaches a planned cell ({len(leaked)} leaked: "
          f"{leaked[:3]})")
    check(summary["anchor_screen_resamples"] > 0,
          "the resamples are counted in plan_summary, not silent")


def test_plan_distribution():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
