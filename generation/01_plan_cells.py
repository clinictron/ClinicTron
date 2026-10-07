#!/usr/bin/env python3
"""01_plan_cells.py — plan the generation cells: one row per question to be written.

SOURCES. Three sources are planned: Ely types, taxonomy subclasses and failure groups.
Each cell gets an anchor track, a concept from that track, and a masking tag: `required`
(concept always hidden), `impossible` (always named) or `optional` (fair coin at draw).
Ely types and failure groups take their tag and their allowed anchor tracks from the
config's `anchors_and_masking` table; subclasses use the anchors and masking tags the
taxonomy itself carries.

AXES drawn per cell:

  length  short 0.30 / medium 0.40 / long 0.30
  pasted  a subtype of long — two of three long cells open with a {document_type}
          document, so pasted is 20% of all cells
  style   clean 0.625 / messy 0.375


DETERMINISM. Row allocation over sources and over each source's types/groups/subclasses
is largest-remainder apportionment (exact totals, no rounding drift). Each cell then
draws its own fields from `random.Random(sha1(run_id|cell_id))`, so a cell's draw does
not depend on the cells before it and the plan is reproducible from (run_id, config).

DRAW ORDER per cell, fixed and recorded in the manifest:
    track -> concept -> masked_coin_if_optional -> length -> pasted_coin_if_long
          -> document_type_if_pasted -> style

No LLM, no GPU, no network, no database. Pure stdlib plus PyYAML."""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config
from common.logging import Log
from common.seeds import largest_remainder, seed_int, weighted

SOURCES = ("ely_types", "subclasses", "failure_groups")


# ────────────────────────────────────────────────────────────── inputs ──
def track_filter_reason(name: str, rules: dict) -> str | None:
    """Why this display name is unusable as a question concept, or None to keep it.

    Some track entries are product specifications rather than concepts (e.g. "COVID-19 PS
    Non-US Vaccine (Medigen, MVC-COV1901)"), and a question anchored on a product name
    comes out contrived. The rule is mechanical and
    lives in config: a comma or a parenthesis means the name is a product spec rather than
    a disease-level vaccine, and the substring list catches the formulation words that
    survive without either."""
    if rules.get("reject_comma", True) and "," in name:
        return "comma (product spec)"
    if rules.get("reject_parenthesis", True) and ("(" in name or ")" in name):
        return "parenthesis (product spec)"
    low = name.lower()
    for term in rules.get("reject_substrings") or []:
        if term.lower() in low:
            return f"substring:{term}"
    return None


def load_entities(entities_dir: Path, tracks: set[str], filters: dict | None = None,
                  log=print) -> tuple[dict[str, list[dict]], dict]:
    """Load each track's entity list, dropping names `track_filter_reason` rejects.

    Returns (entities by track, report). The report lists every dropped name with its
    reason, not just a count, so the decision is auditable."""
    filters = filters or {}
    out: dict[str, list[dict]] = {}
    report: dict = {}
    for tr in sorted(tracks):
        p = entities_dir / tr / f"{tr}.json"
        if not p.exists():
            raise SystemExit(f"FATAL: entity track file missing: {p}")
        items = [r for r in json.loads(p.read_text())
                 if (r.get("display_name") or "").strip()]
        rules = filters.get(tr)
        if rules and rules.get("enabled", True):
            kept, dropped = [], []
            for r in items:
                why = track_filter_reason(r["display_name"], rules)
                (dropped if why else kept).append(
                    {"display_name": r["display_name"], "reason": why} if why else r)
            report[tr] = {"source": str(p), "before": len(items), "kept": len(kept),
                          "dropped": len(dropped),
                          "kept_names": sorted(r["display_name"] for r in kept),
                          "dropped_names": sorted(
                              (d["display_name"], d["reason"]) for d in dropped)}
            log(f"TRACK_FILTER {tr}: {len(items)} -> {len(kept)} "
                f"({len(dropped)} dropped as product-level names)")
            if not kept:
                raise SystemExit(f"FATAL: the {tr} filter rejected every name in {p}; "
                                 f"loosen entities_filter.{tr} rather than running with "
                                 f"an empty track")
            items = kept
        if not items:
            raise SystemExit(f"FATAL: entity track {tr} is empty: {p}")
        out[tr] = items
    return out, report


def load_subclasses(taxonomy_path: Path, exclude_types: set[str],
                    anchor_to_track: dict[str, str],
                    exclude_subtypes: set[str] = frozenset()) -> list[dict]:
    """Flatten the taxonomy into drawable subclasses.

    `exclude_types` drops whole types whose questions the research literature cannot
    answer (e.g. `drug_product` cells asked about a product's colour and package size),
    and `exclude_subtypes` (dotted `type.subtype` ids) drops individual ones. Each
    subtype's ask is a field of the taxonomy, inline or in subtype_asks.json beside it;
    it is never derived. A subtype whose only anchor is "none" is not drawable."""
    raw = json.loads(taxonomy_path.read_text())
    asks = {(t["id"], s["id"]): s["ask"] for t in raw["types"] for s in t["subtypes"]
            if s.get("ask")}
    side = taxonomy_path.parent / "subtype_asks.json"
    if side.exists():
        for r in json.loads(side.read_text())["asks"]:
            asks[(r["type"], r["subtype"])] = r["ask"]

    rows, missing_ask = [], []
    for t in raw["types"]:
        if t["id"] in exclude_types:
            continue
        for s in t["subtypes"]:
            if f"{t['id']}.{s['id']}" in exclude_subtypes:
                continue
            anchors = list(s.get("anchors") or [])
            unknown = [a for a in anchors if a != "none" and a not in anchor_to_track]
            if unknown:
                raise SystemExit(f"FATAL: subtype {t['id']}.{s['id']} has unknown "
                                 f"anchor(s) {unknown}")
            tracks = [anchor_to_track[a] for a in anchors if a != "none"]
            if not tracks:            # anchors == ["none"] -> not drawable
                continue
            if (t["id"], s["id"]) not in asks:
                missing_ask.append(f"{t['id']}.{s['id']}")
                continue
            rows.append({
                "id": f"{t['id']}.{s['id']}",
                "type_id": t["id"], "type_name": t["name"],
                "subtype_id": s["id"], "subtype_name": s["name"],
                "subtype_definition": s.get("definition", ""),
                "ask": asks[(t["id"], s["id"])],
                "masking": s.get("masking", "optional"), "tracks": tracks})
    if missing_ask:
        raise SystemExit(
            f"FATAL: the taxonomy is missing the ask field on {len(missing_ask)} "
            f"drawable subtypes, e.g. {missing_ask[:3]} — complete the taxonomy, do "
            f"not derive an ask here")
    if not rows:
        raise SystemExit(f"FATAL: no drawable subclasses in {taxonomy_path}")
    return rows


# ─────────────────────────────────────────────────────────────── plan ──
def source_units(cfg, log=print) -> dict[str, dict]:
    """The drawable units of each source, with their weight, tracks and masking tag."""
    table = cfg["anchors_and_masking"]
    units: dict[str, dict] = {}

    ely = {}
    for t in cfg["ely_types"]:
        if t not in table:
            raise SystemExit(f"FATAL: no anchors/masking row for ely type {t!r}")
        ely[t] = {"weight": 1.0, "tracks": table[t]["tracks"],
                  "masking": table[t]["masking"], "meta": {}}
    units["ely_types"] = ely

    groups = {}
    for g, w in cfg["failure_group_weights"].items():
        if g not in table:
            raise SystemExit(f"FATAL: no anchors/masking row for group {g!r}")
        groups[g] = {"weight": float(w), "tracks": table[g]["tracks"],
                     "masking": table[g]["masking"], "meta": {}}
    units["failure_groups"] = groups

    sc_cfg = cfg["subclasses"]
    rows = load_subclasses(Path(sc_cfg["taxonomy"]),
                           set(sc_cfg.get("exclude_types") or []),
                           cfg["anchor_to_track"],
                           set(sc_cfg.get("exclude_subtypes") or []))
    units["subclasses"] = {
        r["id"]: {"weight": 1.0, "tracks": r["tracks"], "masking": r["masking"],
                  "meta": {k: r[k] for k in ("type_id", "type_name", "subtype_id",
                                             "subtype_name", "subtype_definition",
                                             "ask")}}
        for r in rows}
    log(f"PLAN units ely_types={len(ely)} subclasses={len(units['subclasses'])} "
        f"failure_groups={len(groups)}")
    return units


class Drawer:
    """Draws one cell at a time from a source's unit spec.

    Split out of `plan` so the full plan, the top-up plan and the pilot slice all draw
    from ONE implementation. A cell's fields come from `random.Random(sha1(run_id|cell_id))`,
    so a cell's draw never depends on the cells before it and every mode is reproducible.
    """

    def __init__(self, cfg, units, log=print):
        self.cfg = cfg
        self.units = units
        self.run_id = cfg["run"]["run_id"]
        self.entities, self.track_filter = load_entities(
            Path(cfg["paths"]["entities_dir"]),
            {tr for src in units.values() for u in src.values() for tr in u["tracks"]},
            cfg.get("entities_filter"), log)
        self.grouper_screen = bool(cfg["anchors"].get("grouper_screen", True))
        self.grouper_res = [re.compile(p, re.I)
                            for p in cfg["anchors"].get("grouper_patterns", [])]
        self.screened: collections.Counter = collections.Counter()
        self.styles = dict(cfg["styles"])
        self.lengths = dict(cfg["lengths"]["weights"])
        self.pasted_share = float(cfg["lengths"]["long_pasted_share"])
        self.doc_types = list(cfg["lengths"]["document_types"])
        self.track_weights = dict(cfg.get("track_weights") or {})
        self.log = log

    def draw(self, source: str, unit_id: str, cell_id: str) -> dict:
        u = self.units[source][unit_id]
        seed = seed_int(self.run_id, cell_id)
        rng = random.Random(seed)

        # Track choice is uniform over the unit's permitted tracks unless the config sets
        # `track_weights`, which can steer draws toward tracks with more unused concepts.
        # Without the key the draw is uniform.
        _tw = self.track_weights
        if _tw:
            track = rng.choices(u["tracks"], weights=[_tw.get(t, 1.0) for t in u["tracks"]])[0]
        else:
            track = u["tracks"][rng.randrange(len(u["tracks"]))]
        pool = self.entities[track]
        ent = pool[rng.randrange(len(pool))]
        if self.grouper_screen and track == "diseases":
            for _ in range(8):              # 8 straight groupers is ~impossible
                if not any(p.search(ent["display_name"]) for p in self.grouper_res):
                    break
                self.screened[ent["display_name"]] += 1
                ent = pool[rng.randrange(len(pool))]

        if u["masking"] == "required":
            masked = True
        elif u["masking"] == "impossible":
            masked = False
        else:
            masked = rng.random() < 0.5          # coin drawn ONLY for "optional"

        length = weighted(rng, self.lengths)
        pasted = (length == "long") and (rng.random() < self.pasted_share)
        document_type = (self.doc_types[rng.randrange(len(self.doc_types))]
                         if pasted else None)
        style = weighted(rng, self.styles)
        return {
            "cell_id": cell_id, "run_id": self.run_id, "seed": seed,
            "source": source, "type_or_group": unit_id,
            "grounded": bool(self.cfg["grounding"][source]),
            "masking": u["masking"], "masked": masked, "track": track,
            "entity": {"track": track, "id": ent.get("id"),
                       "display_name": ent["display_name"],
                       "synonyms": [s for s in (ent.get("synonyms") or []) if s]},
            "length": length, "pasted": pasted, "document_type": document_type,
            "style": style,
            **({"subclass": u["meta"]} if u["meta"] else {}),
        }

    def redraw_concept(self, cell: dict, attempt: int) -> dict:
        """A fresh concept for an existing cell, same track, deterministic per attempt.

        Used by stage 02's concept literature check, so a cell whose concept has
        essentially no literature is re-drawn rather than written. Only the
        entity changes — style, length, masking and track stay as planned, so the cell
        still fills the slot the planner allocated it."""
        track = cell["track"]
        pool = self.entities[track]
        rng = random.Random(seed_int(self.run_id, cell["cell_id"],
                                     f"concept-redraw-{attempt}"))
        ent = pool[rng.randrange(len(pool))]
        if self.grouper_screen and track == "diseases":
            for _ in range(8):
                if not any(p.search(ent["display_name"]) for p in self.grouper_res):
                    break
                self.screened[ent["display_name"]] += 1
                ent = pool[rng.randrange(len(pool))]
        return {**cell, "entity": {"track": track, "id": ent.get("id"),
                                   "display_name": ent["display_name"],
                                   "synonyms": [s for s in (ent.get("synonyms") or [])
                                                if s]}}

    def draw_many(self, alloc: dict[str, dict[str, int]], id_fmt) -> list[dict]:
        """`alloc` is {source: {unit_id: n}}; `id_fmt(source, unit_id, i)` names the cell."""
        cells = []
        for source in SOURCES:
            for unit_id, n in sorted((alloc.get(source) or {}).items()):
                for i in range(n):
                    cells.append(self.draw(source, unit_id, id_fmt(source, unit_id, i)))
        return cells

    def note_screening(self):
        if self.screened:
            self.log(f"ANCHOR_SCREEN grouper-class names resampled: "
                     f"{sum(self.screened.values())} draws over "
                     f"{len(self.screened)} distinct names")


def unit_targets(cfg, units, total: int, log=print) -> tuple[dict, dict]:
    """(per_source, {source: {unit_id: target}}) by largest-remainder apportionment.

    The total is split over sources by the config's `sources` weights, then each source's
    share over its units by unit weight."""
    per_source = largest_remainder({s: float(cfg["sources"][s]) for s in SOURCES}, total)
    targets = {s: largest_remainder({k: v["weight"] for k, v in units[s].items()},
                                    per_source[s]) for s in SOURCES}
    log(f"PLAN rows total={total} per_source={per_source}")
    return per_source, targets


def select_imported(cfg, targets, imported: list[dict], log=print
                    ) -> tuple[dict[str, list[dict]], list[dict], dict]:
    """Keep min(kept_imported, target) per UNIT; drop the rest. Deterministic.

    Imported rows count toward their own unit's target and fresh cells fill whatever gap is left. A unit with more reusable
    rows than its target is SUBSAMPLED — seeded shuffle on `sha1(run_id|topup|source|unit)`,
    take the first n — so the choice is reproducible and carries no ordering bias from the
    frozen lane files."""
    run_id = cfg["run"]["run_id"]
    by_unit: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    unknown = []
    for q in imported:
        cell = q["cell"] if "cell" in q else q
        key = (cell["source"], cell["type_or_group"])
        if key[1] not in (targets.get(key[0]) or {}):
            unknown.append(cell["cell_id"])
            continue
        by_unit[key].append(q)

    kept: dict[str, list[dict]] = collections.defaultdict(list)
    dropped: list[dict] = []
    table: dict = {}
    for source in SOURCES:
        table[source] = {}
        for unit_id, target in sorted(targets[source].items()):
            rows = sorted(by_unit.get((source, unit_id), []),
                          key=lambda q: (q["cell"] if "cell" in q else q)["cell_id"])
            random.Random(seed_int(run_id, "topup", source, unit_id)).shuffle(rows)
            n_keep = min(len(rows), target)
            kept[source].extend(rows[:n_keep])
            dropped.extend(rows[n_keep:])
            table[source][unit_id] = {"target": target, "reusable": len(rows),
                                      "kept": n_keep, "fresh": target - n_keep}
    if unknown:
        log(f"TOPUP {len(unknown)} imported cell(s) name a unit that is not a target of "
            f"this config and were dropped, e.g. {unknown[:3]}")
    return kept, dropped, table


def summarize(cfg, cells, screened, track_filter=None) -> dict:
    """The per-axis counts every planning mode reports."""
    n = len(cells)
    summary = {
        "run_id": cfg["run"]["run_id"], "config": cfg.path,
        "total_rows": n,
        "by_source": dict(collections.Counter(c["source"] for c in cells)),
        "by_type_or_group": dict(collections.Counter(
            c["type_or_group"] for c in cells if c["source"] != "subclasses")),
        "distinct_subclasses_drawn": len({c["type_or_group"] for c in cells
                                          if c["source"] == "subclasses"}),
        "by_track": dict(collections.Counter(c["track"] for c in cells)),
        "by_masking_tag": dict(collections.Counter(c["masking"] for c in cells)),
        "masked_applied": sum(1 for c in cells if c["masked"]),
        "by_style": dict(collections.Counter(c["style"] for c in cells)),
        "by_length": dict(collections.Counter(c["length"] for c in cells)),
        "pasted": sum(1 for c in cells if c["pasted"]),
        "pasted_fraction": round(sum(1 for c in cells if c["pasted"]) / max(n, 1), 4),
        "by_document_type": dict(collections.Counter(
            c["document_type"] for c in cells if c["document_type"])),
        "distinct_entities": len({(c["track"], c["entity"]["display_name"])
                                  for c in cells}),
        "anchor_screen_resamples": sum(screened.values()),
        "anchor_screen_rejected_names": sorted(screened),

        # lists ride with the run in <track>_filter.json
        "track_filters": {tr: {k: v for k, v in r.items()
                               if k in ("before", "kept", "dropped")}
                          for tr, r in (track_filter or {}).items()},
        # popped by main() and written to <track>_filter.json; never left in the summary
        "_track_filter_full": track_filter or {},
        "draw_weights": {"sources": dict(cfg["sources"]),
                         "styles": dict(cfg["styles"]),
                         "lengths": dict(cfg["lengths"]["weights"]),
                         "long_pasted_share": float(cfg["lengths"]["long_pasted_share"]),
                         "masking_optional_coin": 0.5,
                         "track": "uniform over allowed", "concept": "uniform, no cap",
                         "document_type": "uniform (pasted cells only)"},
        "draw_order": ["track", "concept", "masked_coin_if_optional", "length",
                       "pasted_coin_if_long", "document_type_if_pasted", "style"],
        "seed_recipe": "sha1(run_id|cell_id)[:12] as an int",
    }
    return summary


def plan(cfg, log=print, total: int | None = None) -> tuple[list[dict], dict]:
    """The all-fresh plan: no imported rows, every target filled by a drawn cell."""
    units = source_units(cfg, log)
    total = int(cfg["volume"]["total_rows"] if total is None else total)
    _per_source, targets = unit_targets(cfg, units, total, log)
    d = Drawer(cfg, units, log)
    cells = d.draw_many(targets, lambda s, u, i: f"{s}-{u}-{i:04d}")
    run_id = cfg["run"]["run_id"]
    screened = d.screened
    d.note_screening()
    random.Random(seed_int(run_id, "plan-order")).shuffle(cells)
    if screened:
        log(f"ANCHOR_SCREEN grouper-class names resampled: {sum(screened.values())} "
            f"draws over {len(screened)} distinct names")

    summary = summarize(cfg, cells, screened, getattr(d, "track_filter", None))
    return cells, summary


def measure_yield(attempted: list[dict], written: list[dict]) -> dict:
    """Per-unit and per-source (written ok / cells attempted) from earlier rounds.

    A round is measured, not guessed: `attempted` is the cells earlier rounds planned and
    `written` is the rows those rounds actually produced."""
    att_u: collections.Counter = collections.Counter()
    att_s: collections.Counter = collections.Counter()
    for c in attempted:
        cell = c["cell"] if "cell" in c else c
        att_u[(cell["source"], cell["type_or_group"])] += 1
        att_s[cell["source"]] += 1
    ok_u: collections.Counter = collections.Counter()
    ok_s: collections.Counter = collections.Counter()
    for q in written:
        if q.get("status") not in (None, "ok"):
            continue
        cell = q["cell"] if "cell" in q else q
        ok_u[(cell["source"], cell["type_or_group"])] += 1
        ok_s[cell["source"]] += 1
    return {"attempted_by_unit": att_u, "written_by_unit": ok_u,
            "attempted_by_source": att_s, "written_by_source": ok_s}


def yield_for(meas: dict, source: str, unit_id: str, min_attempts: int
              ) -> tuple[float, str]:
    """(yield, where it came from). A unit with too little evidence borrows its source's
    rate; a source with none plans no over-generation at all, because inventing a
    multiplier would spend money on a guess. A later round measures what this one did."""
    a = meas["attempted_by_unit"].get((source, unit_id), 0)
    if a >= min_attempts:
        return meas["written_by_unit"].get((source, unit_id), 0) / a, "unit"
    a_s = meas["attempted_by_source"].get(source, 0)
    if a_s >= min_attempts:
        return meas["written_by_source"].get(source, 0) / a_s, "source"
    return 1.0, "unmeasured"


def round_settled(meas: dict, source: str, unit_id: str, terminal: int) -> bool:
    """Has this unit's earlier round FINISHED? A yield measured mid-round is meaningless.

    A round is settled for a unit when
    every cell it attempted has an outcome — a written row or a terminal skip."""
    attempted = meas["attempted_by_unit"].get((source, unit_id), 0)
    written = meas["written_by_unit"].get((source, unit_id), 0)
    return attempted == 0 or (written + terminal) >= attempted


def plan_round(cfg, imported: list[dict], attempted: list[dict], written: list[dict],
               round_no: int, log=print, terminal_by_unit=None,
               allow_unsettled: bool = False):
    """Plan round N: fill what earlier rounds did not, over-generating by measured yield.

    Per unit: remaining = target - kept - written; planned = ceil(remaining / yield),
    with the yield floored at `overgeneration.min_yield` and the resulting multiplier
    capped at `overgeneration.max_overgen`. Cell ids carry `-r<N>-`, so a round's seeds
    are its own and a cell can never repeat across rounds.
    """
    og = cfg.get("overgeneration") or {}
    min_yield = float(og.get("min_yield", 0.35))
    max_overgen = float(og.get("max_overgen", 3.0))
    min_attempts = int(og.get("min_attempts", 5))

    units = source_units(cfg, log)
    total = int(cfg["volume"]["total_rows"])
    _per_source, targets = unit_targets(cfg, units, total, log)
    kept, dropped, table = select_imported(cfg, targets, imported, log)
    meas = measure_yield(attempted, written)
    terminal_by_unit = terminal_by_unit or {}


    unsettled = []
    for source in SOURCES:
        for unit_id in table[source]:
            n_att = meas["attempted_by_unit"].get((source, unit_id), 0)
            if n_att and not round_settled(
                    meas, source, unit_id,
                    terminal_by_unit.get((source, unit_id), 0)):
                unsettled.append((source, unit_id, n_att,
                                  meas["written_by_unit"].get((source, unit_id), 0)))
    if unsettled:
        n_a = sum(u[2] for u in unsettled)
        n_w = sum(u[3] for u in unsettled)
        msg = (f"{len(unsettled)} unit(s) have cells still in flight: {n_w} rows written "
               f"of {n_a} attempted, so their measured yield is a floor, not a rate. "
               f"Planning on it over-generates wildly (a half-run Ely round reads 0.03 "
               f"instead of 0.49). Wait for the writers to finish, or pass "
               f"--allow-unsettled to plan on what is there.")
        if not allow_unsettled:
            raise SystemExit("FATAL: " + msg)
        log("ROUND_UNSETTLED " + msg)

    alloc: dict[str, dict[str, int]] = {s: {} for s in SOURCES}
    for source in SOURCES:
        for unit_id, row in table[source].items():
            n_written = meas["written_by_unit"].get((source, unit_id), 0)
            n_att = meas["attempted_by_unit"].get((source, unit_id), 0)
            remaining = max(0, row["target"] - row["kept"] - n_written)
            y, basis = yield_for(meas, source, unit_id, min_attempts)
            y_used = max(y, min_yield)
            mult = min(1.0 / y_used, max_overgen) if y_used > 0 else max_overgen
            planned = math.ceil(remaining * mult) if remaining else 0
            row.update({"written": n_written, "attempted": n_att,
                        "remaining": remaining,
                        "yield": round(y, 4), "yield_basis": basis,
                        "yield_used": round(y_used, 4),
                        "overgen_multiplier": round(mult, 3), "planned": planned})
            if planned:
                alloc[source][unit_id] = planned

    d = Drawer(cfg, units, log)
    cells = d.draw_many(alloc, lambda s, u, i: f"{s}-{u}-r{round_no}-{i:04d}")
    random.Random(seed_int(cfg["run"]["run_id"], f"plan-order-r{round_no}")).shuffle(cells)
    d.note_screening()

    summary = summarize(cfg, cells, d.screened, d.track_filter)
    summary["mode"] = f"round{round_no}"
    summary["round"] = round_no
    summary["target_total"] = total
    summary["kept_imported_rows"] = sum(len(v) for v in kept.values())
    summary["written_earlier_rounds"] = sum(meas["written_by_unit"].values())
    summary["attempted_earlier_rounds"] = sum(meas["attempted_by_unit"].values())
    summary["remaining_total"] = sum(v["remaining"] for s in SOURCES
                                     for v in table[s].values())
    summary["planned_total"] = len(cells)
    summary["overgeneration"] = {"min_yield": min_yield, "max_overgen": max_overgen,
                                 "min_attempts": min_attempts}
    summary["unsettled_units"] = [{"source": s, "unit": u, "attempted": a, "written": w}
                                  for s, u, a, w in unsettled]
    summary["per_unit"] = table
    summary["per_source_totals"] = {
        s: {"target": sum(v["target"] for v in table[s].values()),
            "kept": sum(v["kept"] for v in table[s].values()),
            "written": sum(v["written"] for v in table[s].values()),
            "remaining": sum(v["remaining"] for v in table[s].values()),
            "attempted": sum(v["attempted"] for v in table[s].values()),
            "yield": (round(meas["written_by_source"].get(s, 0)
                            / meas["attempted_by_source"][s], 4)
                      if meas["attempted_by_source"].get(s) else None),
            "planned": sum(v["planned"] for v in table[s].values())}
        for s in SOURCES}
    summary["units_already_full"] = {
        s: sorted(u for u, v in table[s].items() if v["remaining"] == 0)
        for s in SOURCES}
    summary["note"] = (f"round {round_no} plans target - kept - written per unit, divided "
                       f"by the measured yield (A16). Rounds overshoot on purpose; stage "
                       f"07 subsamples each unit back to its exact target.")
    return cells, kept, dropped, table, summary


def plan_topup(cfg, imported: list[dict], log=print):
    """Fill each unit's target with imported rows first, then draw the rest.

    Returns (fresh_cells, kept_by_source, dropped, table, summary).
    """
    units = source_units(cfg, log)
    total = int(cfg["volume"]["total_rows"])
    _per_source, targets = unit_targets(cfg, units, total, log)
    kept, dropped, table = select_imported(cfg, targets, imported, log)

    alloc = {s: {u: v["fresh"] for u, v in table[s].items() if v["fresh"] > 0}
             for s in SOURCES}
    d = Drawer(cfg, units, log)
    fresh = d.draw_many(alloc, lambda s, u, i: f"{s}-{u}-{i:04d}")
    random.Random(seed_int(cfg["run"]["run_id"], "plan-order")).shuffle(fresh)
    d.note_screening()

    summary = summarize(cfg, fresh, d.screened, d.track_filter)
    n_kept = sum(len(v) for v in kept.values())
    summary["mode"] = "topup"
    summary["fresh_rows"] = len(fresh)
    summary["kept_imported_rows"] = n_kept
    summary["dropped_imported_rows"] = len(dropped)
    summary["planned_total"] = len(fresh) + n_kept
    summary["target_total"] = total
    summary["per_unit"] = table
    summary["per_source_totals"] = {
        s: {"target": sum(v["target"] for v in table[s].values()),
            "reusable": sum(v["reusable"] for v in table[s].values()),
            "kept": sum(v["kept"] for v in table[s].values()),
            "fresh": sum(v["fresh"] for v in table[s].values())}
        for s in SOURCES}
    summary["units_short_of_target"] = {
        s: {u: v for u, v in table[s].items() if v["reusable"] < v["target"]
            and v["fresh"] > 0}
        for s in SOURCES}
    summary["note"] = ("The total is 5,760 (1,920 per source); imported rows "
                       "count toward their own unit's target and fresh cells fill the gap. "
                       "Reuse does NOT sit on top.")
    return fresh, kept, dropped, table, summary


def plan_pilot(cfg, per_source: int, log=print) -> tuple[list[dict], dict]:
    """A separate slice of FRESH cells for the pilot gate, spread across each source's units.

    Drawn under its own id prefix (`pilot-...`), so its seeds differ from every cell of
    the full plan and the full plan is unchanged by planning a pilot. The pilot is not a
    subset of the full plan and is not deducted from any target: it is a read-through
    batch, banked or discarded on its own.
    """
    units = source_units(cfg, log)
    alloc = {s: largest_remainder({k: v["weight"] for k, v in units[s].items()},
                                  per_source)
             for s in SOURCES}
    d = Drawer(cfg, units, log)
    cells = d.draw_many(alloc, lambda s, u, i: f"pilot-{s}-{u}-{i:04d}")
    random.Random(seed_int(cfg["run"]["run_id"], "pilot-order")).shuffle(cells)
    d.note_screening()

    summary = summarize(cfg, cells, d.screened, d.track_filter)
    summary["mode"] = "pilot"
    summary["per_source"] = per_source
    summary["units_covered"] = {s: len(alloc[s]) for s in SOURCES}
    summary["units_with_zero"] = {s: sorted(u for u, n in alloc[s].items() if n == 0)
                                  for s in SOURCES}
    summary["min_cells_per_unit"] = {
        s: min(alloc[s].values()) if alloc[s] else 0 for s in SOURCES}
    summary["note"] = ("a separate GATE ONE slice; its cell ids carry a `pilot-` prefix so "
                       "its draws are independent of the full plan's")
    return cells, summary


# ─────────────────────────────────────────────────────────────── main ──
def write_track_filters(out: Path, report: dict, log=print) -> None:
    """Write one <track>_filter.json per track with the entity-filter report."""
    for tr, r in (report or {}).items():
        path = out / f"{tr}_filter.json"
        path.write_text(json.dumps(r, indent=1) + "\n")
        log(f"TRACK_FILTER wrote {path} (kept {r['kept']} of {r['before']})")


def emit_summary(out: Path, summary: dict, log=print) -> dict:
    """Write the per-track filter files, then plan_summary.json without the bulk."""
    write_track_filters(out, summary.pop("_track_filter_full", {}) or {}, log)
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _write(path: Path, rows) -> int:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: <run_root>/<run_id>/01_plan")
    ap.add_argument("--total-rows", type=int, default=None,
                    help="override volume.total_rows")
    ap.add_argument("--topup-from", default=None,
                    help="stage 00's cells.jsonl (or questions.jsonl) — plan only the "
                         "cells the imported rows do not already cover")
    ap.add_argument("--round", type=int, default=None,
                    help="plan round N: fill what earlier rounds left short, "
                         "over-generating by the yield they measured. Writes to "
                         "<out-dir parent>/01_plan_round<N>/")
    ap.add_argument("--written", nargs="*", default=[],
                    help="questions.jsonl of earlier rounds (stage 02 outputs)")
    ap.add_argument("--attempted", nargs="*", default=[],
                    help="cells.jsonl of earlier rounds — what those rounds tried")
    ap.add_argument("--skips", nargs="*", default=[],
                    help="skips.log of earlier rounds; a cell with a terminal skip is "
                         "resolved, which is how a round is known to have finished")
    ap.add_argument("--allow-unsettled", action="store_true",
                    help="plan even though earlier rounds are still writing — their "
                         "yield is then a floor, not a rate, and over-generation will "
                         "overshoot badly")
    ap.add_argument("--pilot-per-source", type=int, default=None,
                    help="also plan a separate GATE ONE slice of N fresh cells per "
                         "source into <out-dir>_pilot/")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.total_rows:
        cfg["volume"]["total_rows"] = a.total_rows
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("01_plan")
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")

    if a.round and not a.topup_from:
        ap.error("--round needs --topup-from: a round fills TARGET - kept - written, "
                 "and the kept set comes from the import")

    if a.topup_from:
        src = Path(a.topup_from)
        imported = [json.loads(l) for l in open(src) if l.strip()]
        # stage 00 writes cells.jsonl and questions.jsonl; accept either, and read the
        # questions file beside a cells file so the kept rows carry their text.
        if imported and "cell" not in imported[0]:
            qpath = src.parent / "questions.jsonl"
            if not qpath.exists():
                raise SystemExit(f"FATAL: {src} holds cells, and {qpath} is not beside "
                                 f"it; --topup-from needs the questions to write "
                                 f"kept_imported.jsonl")
            imported = [json.loads(l) for l in open(qpath) if l.strip()]
        log(f"TOPUP read {len(imported)} imported rows from {src.parent}")

        if a.round:
            out = out.parent / f"01_plan_round{a.round}"
            out.mkdir(parents=True, exist_ok=True)
            log = Log(out / "run.log")
            written = [json.loads(l) for p_ in a.written
                       for l in open(p_) if l.strip()]
            attempted = [json.loads(l) for p_ in a.attempted
                         for l in open(p_) if l.strip()]
            if not attempted:
                raise SystemExit(
                    "FATAL: --round needs --attempted <cells.jsonl ...> — the yield is "
                    "written/attempted, and without what earlier rounds TRIED there is "
                    "no yield to divide by, only a guess.")
            log(f"ROUND{a.round} read {len(written)} written rows from "
                f"{len(a.written)} file(s) and {len(attempted)} attempted cells from "
                f"{len(a.attempted)} file(s)")
            # a terminal skip resolves a cell just as a written row does; without the
            # skips.log an interrupted round would look unfinished forever.
            terminal_by_unit: dict = collections.Counter()
            if a.skips:
                import importlib.util as _iu
                sp2 = _iu.spec_from_file_location(
                    "p7_write", str(Path(__file__).resolve().parent /
                                    "02_write_questions.py"))
                wq = _iu.module_from_spec(sp2)
                sp2.loader.exec_module(wq)
                by_cell = {c["cell_id"]: c for c in attempted}
                for sp_ in a.skips:
                    for cid in wq.terminal_cells(sp_):
                        c = by_cell.get(cid)
                        if c:
                            terminal_by_unit[(c["source"], c["type_or_group"])] += 1
                log(f"ROUND{a.round} read {sum(terminal_by_unit.values())} terminal "
                    f"skips from {len(a.skips)} log(s)")
            fresh, kept, dropped, table, summary = plan_round(
                cfg, imported, attempted, written, a.round, log,
                terminal_by_unit=terminal_by_unit,
                allow_unsettled=a.allow_unsettled)
            _write(out / "cells.jsonl", fresh)
            _write(out / "kept_imported.jsonl",
                   [q for s in SOURCES for q in kept[s]])
            emit_summary(out, summary, log)
            log(f"ROUND{a.round} remaining={summary['remaining_total']} "
                f"planned={summary['planned_total']} -> {out}")
            print(json.dumps({k: summary[k] for k in
                              ("mode", "target_total", "kept_imported_rows",
                               "written_earlier_rounds", "remaining_total",
                               "planned_total", "overgeneration",
                               "per_source_totals")}, indent=1))
            return 0

        fresh, kept, dropped, table, summary = plan_topup(cfg, imported, log)
        kept_rows = [q for s in SOURCES for q in kept[s]]
        _write(out / "cells.jsonl", fresh)
        _write(out / "kept_imported.jsonl", kept_rows)
        _write(out / "kept_imported_cells.jsonl", [q["cell"] for q in kept_rows])
        _write(out / "dropped_imported.jsonl", dropped)
        emit_summary(out, summary, log)
        log(f"TOPUP target={summary['target_total']} kept={summary['kept_imported_rows']} "
            f"fresh={summary['fresh_rows']} dropped={summary['dropped_imported_rows']} "
            f"-> {out}")
        print(json.dumps({k: summary[k] for k in
                          ("mode", "target_total", "planned_total", "kept_imported_rows",
                           "fresh_rows", "dropped_imported_rows", "per_source_totals")},
                         indent=1))
    else:
        cells, summary = plan(cfg, log)
        _write(out / "cells.jsonl", cells)
        emit_summary(out, summary, log)
        log(f"PLAN wrote {len(cells)} cells -> {out/'cells.jsonl'}")
        print(json.dumps(summary, indent=1))

    if a.pilot_per_source:
        pout = out.parent / (out.name + "_pilot")
        pout.mkdir(parents=True, exist_ok=True)
        plog = Log(pout / "run.log")
        cells, psummary = plan_pilot(cfg, int(a.pilot_per_source), plog)
        _write(pout / "cells.jsonl", cells)
        emit_summary(pout, psummary, plog)
        plog(f"PILOT wrote {len(cells)} cells "
             f"({a.pilot_per_source} per source) -> {pout/'cells.jsonl'}")
        print(json.dumps({k: psummary[k] for k in
                          ("mode", "per_source", "total_rows", "units_covered",
                           "min_cells_per_unit", "by_source")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
