#!/usr/bin/env python3
"""02_write_questions.py — write one question per planned cell.

GROUNDED cells (Ely types, subclasses): grounding search -> pick call -> grounded writer
            -> leak check -> Gate A blind verifier (masked rows only), with up to
            `gates.revision_cycles` targeted rewrites as a continued conversation. The
            writer also returns its two ideal-answer sketches inline.

UNGROUNDED cells (failure groups): the ungrounded writer, given the group block, 3-5
            real-NLM style exemplars and the drawn style and length. A second call then
            writes the two ideal-answer sketches.

A masked row whose text leaks its concept, and that the allowed rewrites cannot fix, is
discarded and logged.

A grounded cell has a literature floor built in, through its grounding search and the
pick step: a concept with no literature never reaches the writer. An ungrounded cell has
no such floor. So before an ungrounded cell is written, ONE fused search on
the concept's display name must return at least `concept_check.min_hits` results that
actually name the concept in their title or abstract. If it does not, the concept is
RE-DRAWN from the same track (deterministically, up to `concept_check.max_redraws` times)
and the rejection is logged as CONCEPT_REDRAW. The check result rides with the row.

Every rendered prompt and every raw response is persisted (writer_raw.jsonl), so a
question can be traced to the exact bytes that produced it.

RESUME. A restart never redoes finished work: on start the stage reads the existing
`questions.jsonl` AND `skips.log` in its output directory, and skips every cell that
already has a written row or a TERMINAL outcome (a discard or a skip — a leak the
revisions could not fix, a Gate A failure, a concept with no literature, grounding that
found no qualifying paper). Both files are appended to, never truncated, so a restart
never pays for a cell twice.

SKETCHES ONLY (`--sketches-only`) reads the imported rows from stage 00, some of which carry
`sketches_missing: true`. In this mode the stage skips the writer entirely, makes
ONLY the sketch call for the rows that need it, and passes every other row through
untouched. It never rewrites an imported question — the text is the artifact being reused.
A row that already carries exactly ONE sketch is filled too: recipe R needs two.

A sketch call can return fewer than two sketches. Recipe R (stage 04) gives each sketch its own frozen and per-retriever quotas, so a
row with one sketch loses a third of its pool. Both sketch paths therefore RETRY the call
once when fewer than two come back, and only then mark `sketches_missing`.

NO network is opened at import. `--dry-run` renders and persists prompts without calling
anything, which is what the prompt sign-off gate needs."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import prompts as P
from common.retrieval import build_search
from common.config import load_config, load_secrets
from common.gates import leak_hit
from common.llm import Budget, build_lanes
from common.logging import Log, SkipLog, append_jsonl, load_keys
from common.seeds import sha1

# Skip verbs that END a cell: it will never produce a row, so a resume must not retry it.
# A verb NOT in this set (LLM_FAIL on one attempt, CONCEPT_REDRAW, SKETCHES_SHORT) is a
# step inside a cell that may still succeed, so a resume DOES retry it.
TERMINAL_SKIPS = {"CELL_ERROR", "SKIP", "LEAK_DISCARD", "GATE_A_DISCARD", "CONCEPT_DISCARD",
                  "REVISION_EXHAUSTED"}
_CELL_RE = re.compile(r"\bcell='([^']+)'")


def terminal_cells(skips_path) -> set[str]:
    """Cell ids that already reached a terminal outcome, read back from skips.log.

    The log is the record of what was decided, so a resume reads the same file a human
    would. A malformed or partial line is ignored rather than crashing a restart.
    """
    out: set[str] = set()
    p = Path(skips_path)
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2 or parts[1] not in TERMINAL_SKIPS:
                continue
            m = _CELL_RE.search(line)
            if m:
                out.add(m.group(1))
    return out


def _drawer(cfg, log):
    """Load stage 01's Drawer by file path — a numbered stage cannot be imported by name.
    Used only for the concept re-draw, so the re-draw uses the planner's own logic."""
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        "p7_plan_cells", str(Path(__file__).resolve().parent / "01_plan_cells.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m.Drawer(cfg, m.source_units(cfg, log), log)


def results_block(docs: list[dict]) -> str:
    """Numbered search results for GROUNDING_PICK. NO truncation anywhere."""
    return "\n".join(
        f"[{i}] {d.get('title') or '(no title)'} — {d.get('abstract') or '(no abstract)'}"
        for i, d in enumerate(docs, 1))


class QuestionWriter:
    """One cell -> one written question. Every external system arrives as a client
    object, so a test constructs this with stubs and never touches the network."""

    def __init__(self, cfg, lanes: dict, bridge, log, skiplog, raw_path, drawer=None,
                 grounding_cache=None):
        self.cfg = cfg
        self.lanes = lanes
        self.bridge = bridge
        self.log = log
        self.skip = skiplog
        self.raw_path = raw_path
        self.pool_cfg = cfg["pool"]
        self.gates = cfg["gates"]
        self.cc = cfg.get("concept_check") or {}
        self._drawer = drawer          # built lazily; only the re-draw path needs it


        # Optional grounding cache. Without it, each
        # process runs its own grounding search and its own pick call. Given a path, a
        # cell's papers are read from the file instead, and a fresh pick is appended to
        # it. The pick is a model call, so without this two runs over the same cell get
        # DIFFERENT papers — which would make "every arm saw the same grounding papers"
        # false.
        self.gcache_path = grounding_cache or self.pool_cfg.get("grounding_cache") or None
        self._gcache: dict | None = None

    async def _search(self, query: str, top_k: int) -> list[dict]:
        return await asyncio.to_thread(self.bridge.search, query, top_k)

    def _persist(self, cell, step, prompt, response, extra=None):
        append_jsonl(self.raw_path, {
            "cell_id": cell["cell_id"], "step": step, "prompt": prompt,
            "prompt_sha1": sha1(prompt or ""), "response": response, **(extra or {})})

    # ── concept literature check (ungrounded sources only) ────────────────
    def _names_concept(self, doc: dict, names: list[str]) -> bool:
        hay = f"{doc.get('title') or ''} {doc.get('abstract') or ''}".lower()
        return any(n in hay for n in names)

    async def concept_hits(self, cell: dict) -> int:
        """How many of one fused search's results actually name this cell's concept."""
        ent = cell["entity"]
        names = [n.strip().lower() for n in
                 [ent["display_name"]] + list(ent.get("synonyms") or []) if n.strip()]
        docs = await self._search(ent["display_name"], int(self.cc.get("top_k", 50)))
        return sum(1 for d in docs if self._names_concept(d, names))

    async def check_concept(self, cell: dict) -> tuple[dict | None, dict]:
        """Return (cell_with_a_usable_concept, check_record), or (None, record).

        The cell keeps its planned track, style, length and masking; only the concept
        moves, so a re-drawn cell still fills the slot the planner allocated it.
        """
        rec = {"checked": True, "min_hits": int(self.cc.get("min_hits", 5)),
               "attempts": [], "redraws": 0}
        max_redraws = int(self.cc.get("max_redraws", 5))
        for attempt in range(1 + max_redraws):
            hits = await self.concept_hits(cell)
            rec["attempts"].append({"concept": cell["entity"]["display_name"],
                                    "hits": hits})
            if hits >= rec["min_hits"]:
                rec["hits"] = hits
                rec["concept"] = cell["entity"]["display_name"]
                return cell, rec
            self.skip.note("CONCEPT_REDRAW", cell=cell["cell_id"],
                           rejected=cell["entity"]["display_name"],
                           track=cell["track"], hits=hits,
                           min_hits=rec["min_hits"], attempt=attempt)
            if attempt == max_redraws:
                break
            if self._drawer is None:
                self._drawer = _drawer(self.cfg, self.log)
            cell = self._drawer.redraw_concept(cell, attempt)
            rec["redraws"] += 1
        rec["hits"] = rec["attempts"][-1]["hits"]
        rec["concept"] = None
        return None, rec


    def _cache(self) -> dict:
        """cell_id -> the papers that cell was given (an empty list = none qualified)."""
        if self._gcache is None:
            self._gcache = {}
            p = Path(self.gcache_path)
            if p.exists():
                with open(p) as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line)
                            self._gcache[r["cell_id"]] = r.get("docs") or []
        return self._gcache

    def _cache_put(self, cell, docs: list[dict]) -> None:
        append_jsonl(self.gcache_path, {"cell_id": cell["cell_id"],
                                        "entity": cell["entity"]["display_name"],
                                        "type_or_group": cell.get("type_or_group"),
                                        "docs": docs})
        self._cache()[cell["cell_id"]] = docs

    # ── grounding (grounded sources only) ─────────────────────────────────
    async def grounding(self, cell) -> list[dict] | None:
        """Grounding papers for a cell (cache, else search + pick call), or None if none qualify."""
        if self.gcache_path and cell["cell_id"] in self._cache():
            docs = self._cache()[cell["cell_id"]]
            if docs:
                self.log(f"GROUNDING_CACHE_HIT cell={cell['cell_id']} n={len(docs)}")
                return docs
            self.skip.note("SKIP", cell=cell["cell_id"],
                           entity=cell["entity"]["display_name"],
                           reason="no_qualifying_grounding_cached")
            return None
        entity = cell["entity"]["display_name"]
        synonyms = cell["entity"].get("synonyms") or []
        base = P.grounding_query(self.cfg, cell)
        queries = [base]
        if synonyms:
            k = int(self.pool_cfg.get("grounding_synonyms_top_k", 0) or 0)
            use = synonyms if k <= 0 else synonyms[:k]
            queries.append(base.replace(entity, " OR ".join([entity] + use)))
        for q in queries:
            results = await self._search(q, self.pool_cfg["grounding_top_k"])
            if not results:
                continue
            prompt = P.fill(P.load("grounding_pick.md", self.cfg.prompts_dir),
                            entity_name=entity,
                            synonyms=", ".join(synonyms) or "(none)",
                            n=len(results), results_block=results_block(results))
            try:
                parsed = await self.lanes["aux"].generate_json(prompt)
            except Exception as exc:
                self.skip.note("LLM_FAIL", step="grounding_pick",
                               cell=cell["cell_id"], err=str(exc))
                return None
            self._persist(cell, "grounding_pick", prompt, parsed,
                          {"n_results": len(results)})
            sel = [int(n) for n in (parsed.get("selected") or [])
                   if str(n).lstrip("-").isdigit()]
            docs = [results[n - 1] for n in sel if 1 <= n <= len(results)][:3]
            if docs:
                if self.gcache_path:
                    self._cache_put(cell, docs)
                return docs
        if self.gcache_path:
            self._cache_put(cell, [])
        self.skip.note("SKIP", cell=cell["cell_id"], entity=entity,
                       reason="no_qualifying_grounding")
        return None

    # ── Gate A (grounded sources, masked rows only) ───────────────────────
    async def gate_a(self, cell, query: str) -> tuple[bool, list]:
        """Gate A: a blind verifier reads the question; returns (passed, details)."""
        entity = cell["entity"]["display_name"]
        synonyms = cell["entity"].get("synonyms") or []
        ask_line = P.load("gate_a_ask.md", self.cfg.prompts_dir)
        prompt = P.fill(P.load("gate_a.md", self.cfg.prompts_dir),
                        ask_line=ask_line, query=query)
        parsed = await self.lanes["aux"].generate_json(prompt, temperature=0.2)
        self._persist(cell, "gate_a", prompt, parsed)
        verdict = str(parsed.get("verdict") or "").upper()
        best = str(parsed.get("best_answer") or "")
        rivals = parsed.get("rivals") or []
        if verdict != "UNIQUE":
            return False, rivals
        from common.gates import norm_ws
        bl = norm_ws(best)
        for t in [entity] + list(synonyms):
            tn = norm_ws(t)
            if tn and (tn in bl or bl in tn):
                return True, []
        eq_prompt = P.fill(P.load("equivalence.md", self.cfg.prompts_dir),
                           answer=best, entity=entity)
        eq = await self.lanes["aux"].generate_json(eq_prompt, temperature=0.0)
        self._persist(cell, "equivalence", eq_prompt, eq)
        if eq.get("same") is True:
            return True, []
        return False, [f"best_answer_mismatch:{best}"]

    # ── the sketch call, shared by both paths ─────────────────────────────
    async def call_sketches(self, cell: dict, query: str, step: str,
                            qid: str | None = None) -> list[dict]:
        """Two ideal-answer sketches for `query`, with ONE immediate retry.

        Recipe R scores each sketch as its own query text, so a row that comes back with
        one sketch loses a third of its pool. A short return is retried once before the
        row is marked short — that is cheaper than the pool it would otherwise lose.
        """
        prompt = P.fill(P.load("sketches.md", self.cfg.prompts_dir), query=query)
        sketches: list[dict] = []
        for attempt in range(1 + int(self.cfg.get("sketch_retries", 1))):
            try:
                sk = await self.lanes["aux"].generate_json(prompt)
            except Exception as exc:
                self.skip.note("LLM_FAIL", step=step, qid=qid or cell["cell_id"],
                               attempt=attempt, err=str(exc))
                sk = {}
            self._persist(cell, step, prompt, sk, {"attempt": attempt})
            got = [h for h in (sk.get("hyde") or []) if isinstance(h, dict)][:2]
            if len(got) > len(sketches):
                sketches = got
            if len(sketches) >= 2:
                if attempt:
                    self.log(f"SKETCH_RETRY_OK qid={qid or cell['cell_id']} "
                             f"recovered on attempt {attempt}")
                return sketches
            self.skip.note("SKETCHES_SHORT", qid=qid or cell["cell_id"],
                           got=len(got), attempt=attempt)
        return sketches

    # ── sketches only: fill an imported row's missing sketches ────────────
    async def sketches_for(self, row: dict) -> dict:
        """The sketch call for a row that already has its question text.

        Used by --sketches-only on the imported rows stage 01 kept. The question is never
        rewritten; only `sketches` is filled. A row that arrives with exactly one sketch
        is filled too — recipe R needs two.
        """
        sketches = await self.call_sketches(row["cell"], row["text"], "sketches_only",
                                            qid=row["qid"])
        if not sketches:
            return {**row, "status": "discard", "reason": "sketch_fail"}
        return {**row, "sketches": sketches,
                "sketches_missing": len(sketches) < 2, "status": "ok"}

    # ── one cell ──────────────────────────────────────────────────────────
    async def write(self, cell: dict) -> dict:
        """Returns a row dict, or {'status': 'skip'|'discard', 'reason': ...}."""
        concept_check = None
        if (self.cc.get("enabled")
                and cell["source"] in (self.cc.get("sources") or [])):
            checked, concept_check = await self.check_concept(cell)
            if checked is None:
                self.skip.note("CONCEPT_DISCARD", cell=cell["cell_id"],
                               track=cell["track"],
                               reason=f"no concept on this track passed the literature "
                                      f"check in {concept_check['redraws'] + 1} draws")
                return {"cell_id": cell["cell_id"], "status": "discard",
                        "reason": "concept_no_literature",
                        "concept_check": concept_check}
            cell = checked
        entity = cell["entity"]["display_name"]
        synonyms = cell["entity"].get("synonyms") or []
        grounding = None
        if cell["grounded"]:
            grounding = await self.grounding(cell)
            if grounding is None:
                return {"cell_id": cell["cell_id"], "status": "skip",
                        "reason": "no_qualifying_grounding"}
            gblock = "\n".join(f"[G{i+1}] {d.get('title')} — {d.get('abstract')}"
                               for i, d in enumerate(grounding))
        else:
            gblock = None

        prompt = P.render_writer(self.cfg, cell, grounding_block=gblock)
        messages = [{"role": "user", "content": prompt}]
        lane_id = ("writer_ungrounded" if cell["source"] == "failure_groups"
                   else "writer_a")
        writer = self.lanes[lane_id]

        revisions = (self.gates["revision_cycles"]
                     if cell["source"] in self.gates["revision_sources"] else 0)
        parsed = None
        for cycle in range(1 + revisions):
            try:
                parsed = await writer.generate_json(messages, temperature=0.8)
            except Exception as exc:
                self.skip.note("LLM_FAIL", step="writer", cell=cell["cell_id"],
                               err=str(exc))
                return {"cell_id": cell["cell_id"], "status": "discard",
                        "reason": f"writer_fail:{exc}"}
            self._persist(cell, f"writer:{lane_id}", messages[-1]["content"], parsed,
                          {"cycle": cycle, "model": getattr(writer, "model", lane_id)})
            query = (parsed.get("query") or "").strip()
            if not query:
                return {"cell_id": cell["cell_id"], "status": "discard",
                        "reason": "empty_query"}


            hit = None
            if cell["masked"] and self.gates.get("leak_check", True):
                banned = list((parsed.get("working_notes") or {}).get("banned_terms") or [])
                hit = leak_hit([query], entity, synonyms, banned, check_entity=True)
            if hit:
                if cycle >= revisions:
                    self.skip.note("LEAK_DISCARD", cell=cell["cell_id"],
                                   entity=entity, term=hit)
                    return {"cell_id": cell["cell_id"], "status": "discard",
                            "reason": f"leak:{hit}"}
                messages += [
                    {"role": "assistant", "content": json.dumps(parsed)},
                    {"role": "user",
                     "content": P.fill(P.load("revision_leak.md", self.cfg.prompts_dir),
                                       term=hit, twin_note="")}]
                continue

            # Gate A: grounded sources, masked rows only (a named row has no secret)
            if (cell["source"] in self.gates.get("blind_verifier_sources", [])
                    and self.gates.get("blind_verifier", True) and cell["masked"]):
                try:
                    ok, rivals = await self.gate_a(cell, query)
                except Exception as exc:
                    self.skip.note("LLM_FAIL", step="gate_a", cell=cell["cell_id"],
                                   err=str(exc))
                    ok, rivals = False, [f"llm_fail:{exc}"]
                if not ok:
                    if cycle >= revisions:
                        self.skip.note("GATE_A_DISCARD", cell=cell["cell_id"],
                                       entity=entity, rivals=rivals)
                        return {"cell_id": cell["cell_id"], "status": "discard",
                                "reason": f"gate_a:{rivals}"}
                    messages += [
                        {"role": "assistant", "content": json.dumps(parsed)},
                        {"role": "user",
                         "content": P.fill(
                             P.load("revision_gate_a.md", self.cfg.prompts_dir),
                             rivals=", ".join(map(str, rivals)) or "(none named)",
                             twin_note="")}]
                    continue
            break
        else:
            self.skip.note("REVISION_EXHAUSTED", cell=cell["cell_id"])
            return {"cell_id": cell["cell_id"], "status": "discard",
                    "reason": "revision_exhausted"}

        query = (parsed.get("query") or "").strip()
        sketches = [h for h in (parsed.get("hyde") or []) if isinstance(h, dict)][:2]
        if len(sketches) < 2:

            # A grounded row that returned only one sketch inline lands here too.
            sketches = await self.call_sketches(cell, query, "sketches") or sketches

        return {
            "qid": f"{cell['run_id']}-{cell['cell_id']}", "cell_id": cell["cell_id"],
            "status": "ok", "text": query, "text_sha1": sha1(query),
            "sketches": sketches,
            "sketches_missing": len(sketches) < 2,
            "working_notes": parsed.get("working_notes") or {},
            "grounding_ids": [d["work_id"] for d in (grounding or [])],
            "generator_model": getattr(writer, "model", lane_id),
            "prompt_sha1": sha1(prompt),
            "concept_check": concept_check,
            "cell": cell,
        }


# ─────────────────────────────────────────────────────────────── main ──
async def run(cfg, cells, out: Path, dry_run: bool, sketches_only=None,
              grounding_cache=None, grounding_only=False):
    log = Log(out / "run.log")
    skip = SkipLog(out / "skips.log")
    raw = out / "writer_raw.jsonl"
    qpath = out / "questions.jsonl"

    if grounding_only:
        # Grounding-only mode: run grounding and the pick for every cell,
        # write the cache, make NO writer call. It is how the shared cell set is chosen:
        # the grounding search finds no qualifying paper for roughly half the cells, so
        # the cells that come back with papers are the ones every arm can share. Nothing
        # here runs unless --grounding-only is passed, so the live stage is untouched.
        if not grounding_cache:
            raise ValueError("--grounding-only needs --grounding-cache <path>")
        secrets = load_secrets(cfg["paths"]["secrets"])
        budget = Budget(cfg["budget"]["abort_usd"],
                        str(out / cfg["budget"]["cost_state_file"]),
                        warn_usd=cfg["budget"]["warn_usd"],
                        initial_usd=cfg["budget"]["initial_usd"])
        lanes = build_lanes(cfg, budget, log, secrets=secrets)
        bridge = build_search(cfg, log)
        bridge.start()
        try:
            w = QuestionWriter(cfg, lanes, bridge, log, skip, raw,
                               grounding_cache=grounding_cache)
            sem = asyncio.Semaphore(int(cfg["pool"].get("concurrency", 16)))

            async def one(cell):
                async with sem:
                    return cell["cell_id"], await w.grounding(cell)

            done = await asyncio.gather(*[one(c) for c in cells],
                                        return_exceptions=True)
        finally:
            bridge.stop()
        got, none_, errs = [], [], []
        for r in done:
            if isinstance(r, BaseException):
                errs.append(f"{type(r).__name__}: {r}")
            elif r[1]:
                got.append(r[0])
            else:
                none_.append(r[0])
        for e in errs:
            log(f"GROUNDING_ONLY_ERROR {e}")
        log(f"GROUNDING_ONLY done cells={len(cells)} with_papers={len(got)} "
            f"without={len(none_)} errors={len(errs)} cache={grounding_cache}")
        return {"cells": len(cells), "with_papers": len(got), "without": len(none_),
                "errors": errs, "with_papers_ids": got,
                "grounding_cache": str(grounding_cache),
                "spend_usd": round(budget.total, 4)}

    if sketches_only is not None:
        rows = sketches_only
        # a row with ZERO or exactly ONE sketch needs the call; recipe R needs two
        need = [r for r in rows
                if r.get("sketches_missing") or len(r.get("sketches") or []) < 2]
        log(f"SKETCHES_ONLY rows={len(rows)} needing sketches={len(need)}")
        if dry_run:
            for r in need:
                prompt = P.fill(P.load("sketches.md", cfg.prompts_dir), query=r["text"])
                append_jsonl(raw, {"qid": r["qid"], "step": "sketches_only:dry_run",
                                   "prompt": prompt, "prompt_sha1": sha1(prompt),
                                   "response": None})
            log(f"DRY_RUN rendered {len(need)} sketch prompts -> {raw}")
            return {"rows": len(rows), "needing_sketches": len(need), "dry_run": True}
        secrets = load_secrets(cfg["paths"]["secrets"])
        budget = Budget(cfg["budget"]["abort_usd"],
                        str(out / cfg["budget"]["cost_state_file"]),
                        warn_usd=cfg["budget"]["warn_usd"],
                        initial_usd=cfg["budget"]["initial_usd"])
        lanes = build_lanes(cfg, budget, log, secrets=secrets)
        w = QuestionWriter(cfg, lanes, None, log, skip, raw)
        sem = asyncio.Semaphore(int(cfg["models"]["aux"].get("concurrency", 32)))
        need_ids = {r["qid"] for r in need}

        async def one(row):
            if row["qid"] not in need_ids:
                return row
            async with sem:
                return await w.sketches_for(row)

        done = await asyncio.gather(*[one(r) for r in rows], return_exceptions=True)
        ok = 0
        for r in done:
            if isinstance(r, dict) and r.get("status") == "ok":
                append_jsonl(qpath, r)
                ok += 1
        log(f"SKETCHES_ONLY done ok={ok} of {len(rows)} skips={skip.summary()}")
        return {"rows": len(rows), "needing_sketches": len(need), "ok": ok,
                "skips": skip.summary(), "spend_usd": round(budget.total, 4)}

    if dry_run:
        for cell in cells:
            prompt = P.render_writer(cfg, cell, grounding_block="(dry run: no grounding)"
                                     if cell["grounded"] else None)
            append_jsonl(raw, {"cell_id": cell["cell_id"], "step": "writer:dry_run",
                               "prompt": prompt, "prompt_sha1": sha1(prompt),
                               "response": None})
        log(f"DRY_RUN rendered {len(cells)} writer prompts -> {raw}")
        return {"rendered": len(cells), "dry_run": True}

    secrets = load_secrets(cfg["paths"]["secrets"])
    budget = Budget(cfg["budget"]["abort_usd"],
                    str(out / cfg["budget"]["cost_state_file"]),
                    warn_usd=cfg["budget"]["warn_usd"],
                    initial_usd=cfg["budget"]["initial_usd"])
    lanes = build_lanes(cfg, budget, log, secrets=secrets)
    bridge = build_search(cfg, log)
    bridge.start()
    try:
        w = QuestionWriter(cfg, lanes, bridge, log, skip, raw,
                           grounding_cache=grounding_cache)
        sem = asyncio.Semaphore(int(cfg["models"]["writer_a"].get("concurrency", 24)))

        async def one(cell):
            async with sem:
                row = await w.write(cell)
            if row.get("status") == "ok":
                append_jsonl(qpath, {k: v for k, v in row.items() if k != "cell"}
                             | {"cell": row["cell"]})
            return row

        rows = await asyncio.gather(*[one(c) for c in cells], return_exceptions=True)
    finally:
        bridge.stop()

    # A cell that raised an exception is recorded in skips.log as CELL_ERROR, a terminal
    # outcome with the error, so a rerun moves past it and a human can see it.
    for cell, r in zip(cells, rows):
        if isinstance(r, BaseException):
            skip.note("CELL_ERROR", cell=cell["cell_id"],
                      err=f"{type(r).__name__}: {r}")
            log(f"CELL_ERROR cell={cell['cell_id']} {type(r).__name__}: {r}")
    ok = sum(1 for r in rows if isinstance(r, dict) and r.get("status") == "ok")
    log(f"WRITE done ok={ok} of {len(cells)}  skips={skip.summary()}")
    return {"cells": len(cells), "ok": ok, "skips": skip.summary(),
            "spend_usd": round(budget.total, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--cells", default=None, help="cells.jsonl from stage 01")
    ap.add_argument("--imported", default=None,
                    help="questions.jsonl from stage 00, for --sketches-only")
    ap.add_argument("--sketches-only", action="store_true",
                    help="fill missing sketches on imported rows; write NO questions")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N cells (pilot)")
    ap.add_argument("--source", default=None, help="only cells of this source")
    ap.add_argument("--dry-run", action="store_true",
                    help="render and persist prompts; make NO calls")
    ap.add_argument("--writer-prompt", default=None,
                    help="prompt file name for BOTH writer paths, relative to "
                         "prompts/ (e.g. ablation/arm06_minus_masking.md). Default is "
                         "the config's writer_prompts block, else the live files.")
    ap.add_argument("--grounding-cache", default=None,
                    help="JSONL of cell_id -> grounding papers. Read before any "
                         "search, appended after any pick, so every process over the "
                         "same cell sees the SAME papers. Off unless given.")
    ap.add_argument("--grounding-only", action="store_true",
                    help="run grounding + pick for every cell, fill the cache, make "
                         "NO writer call and write NO questions")
    ap.add_argument("--no-resume", action="store_true",
                    help="retry every cell, including ones already written or already "
                         "terminally skipped (default is to resume)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.writer_prompt:
        cfg["writer_prompts"] = {"grounded": a.writer_prompt,
                                 "ungrounded": a.writer_prompt}
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("02_write")
    out.mkdir(parents=True, exist_ok=True)

    if a.grounding_only:
        if not a.cells:
            ap.error("--grounding-only needs --cells")
        cells = [json.loads(l) for l in open(a.cells) if l.strip()]
        if a.source:
            cells = [c for c in cells if c["source"] == a.source]
        if a.limit:
            cells = cells[:a.limit]
        cache = a.grounding_cache or (cfg["pool"].get("grounding_cache") or None)
        summary = asyncio.run(run(cfg, cells, out, a.dry_run,
                                  grounding_cache=cache, grounding_only=True))
        (out / "grounding_only_summary.json").write_text(
            json.dumps(summary, indent=1) + "\n")
        print(json.dumps({k: v for k, v in summary.items()
                          if k != "with_papers_ids"}, indent=1))
        return 0

    if a.sketches_only:
        if not a.imported:
            ap.error("--sketches-only needs --imported <stage 00 questions.jsonl>")
        rows = [json.loads(l) for l in open(a.imported) if l.strip()]
        if a.source:
            rows = [r for r in rows if r["cell"]["source"] == a.source]
        n_all = len(rows)
        done = set() if a.no_resume else load_keys(out / "questions.jsonl", "qid")
        rows = [r for r in rows if r["qid"] not in done]
        if len(rows) != n_all:
            print(f"RESUME skipped {n_all - len(rows)} done cells of {n_all}; "
                  f"{len(rows)} left", flush=True)
        if a.limit:
            rows = rows[:a.limit]
        summary = asyncio.run(run(cfg, [], out, a.dry_run, sketches_only=rows))
        (out / "write_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        return 0

    if not a.cells:
        ap.error("--cells is required unless --sketches-only is given")
    cells = [json.loads(l) for l in open(a.cells) if l.strip()]
    if a.source:
        cells = [c for c in cells if c["source"] == a.source]

    # RESUME: a cell is done if it produced a row OR reached a terminal outcome.
    n_all = len(cells)
    written = load_keys(out / "questions.jsonl", "cell_id")
    terminal = terminal_cells(out / "skips.log")
    if a.no_resume:
        written, terminal = set(), set()
    done = written | terminal
    cells = [c for c in cells if c["cell_id"] not in done]
    n_done = n_all - len(cells)
    if n_done:
        print(f"RESUME skipped {n_done} done cells "
              f"({len(written)} written, {len(terminal)} terminal) of {n_all}; "
              f"{len(cells)} left", flush=True)
    if a.limit:
        cells = cells[:a.limit]

    summary = asyncio.run(run(cfg, cells, out, a.dry_run,
                              grounding_cache=a.grounding_cache
                              or (cfg["pool"].get("grounding_cache") or None)))
    (out / "write_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
