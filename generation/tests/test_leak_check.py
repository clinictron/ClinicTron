#!/usr/bin/env python3
"""test_leak_check.py — the mechanical leak check and its stage-02 wiring.

The leak check runs on EVERY masked row in all three sources; a leaked row is
discarded and logged. The check is `common.gates.leak_hit`.

Covered here:
  * the concept name, any synonym, and any writer-declared banned term are caught
  * matching is word-boundary and case-insensitive for ordinary terms, so inflections
    and capitalisation do not hide a leak, but a substring inside a longer word does
    not false-positive
  * short ALL-CAPS terms match case-SENSITIVELY, so a trial acronym such as REDUCE does
    not match the ordinary word "reduced"
  * terms shorter than three characters are ignored
  * check_entity=False checks only the banned list, for a named-concept row

Also exercised end to end, all with stub lanes and a stub bridge so no network is touched:
  * stage 02's writer loop discards a leaking masked row and logs it, and passes a clean one
  * the CONCEPT LITERATURE CHECK re-draws an ungrounded concept with no literature,
    counts a hit only when a result actually names the concept, and discards the cell when
    no concept on the track passes
  * the SKETCH COUNT retry fills a row that came back with one sketch
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
from common.gates import leak_hit                           # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pool_recipe import StubBridge                     # noqa: E402
from common.llm import StubLane                             # noqa: E402
from common.logging import Log, SkipLog                     # noqa: E402

_sp = importlib.util.spec_from_file_location("write_questions",
                                             str(GEN / "02_write_questions.py"))
write_questions = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(write_questions)

FAILURES: list[str] = []


def check(ok: bool, msg: str):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILURES.append(msg)


def main() -> int:
    ent = "myasthenia gravis"
    syns = ["MG", "Erb-Goldflam disease"]
    banned = ["Osserman", "edrophonium", "TEN"]

    check(leak_hit(["A 40-year-old with myasthenia gravis."], ent, syns, banned)
          == ent, "the concept name is caught")
    check(leak_hit(["Known Myasthenia Gravis, now worse."], ent, syns, banned)
          == ent, "the concept name is caught case-insensitively")
    check(leak_hit(["Consistent with Erb-Goldflam disease."], ent, syns, banned)
          == "Erb-Goldflam disease", "a synonym is caught")
    check(leak_hit(["The Osserman class was III."], ent, syns, banned)
          == "Osserman", "a writer-declared banned term is caught")
    check(leak_hit(["Ptosis worsened through the day."], ent, syns, banned)
          is None, "a clean vignette passes")

    check(leak_hit(["Symptoms reduced after rest."], ent, [], ["REDUCE"]) is None,
          "short ALL-CAPS terms are case-sensitive: 'REDUCE' does not kill 'reduced'")
    check(leak_hit(["Enrolled in the REDUCE trial."], ent, [], ["REDUCE"]) == "REDUCE",
          "short ALL-CAPS terms still match their own casing")
    check(leak_hit(["Toxic epidermal necrolysis (TEN) was excluded."], ent, [], ["TEN"])
          == "TEN", "an all-caps abbreviation is caught when written as itself")
    check(leak_hit(["Ten days of fever."], ent, [], ["TEN"]) is None,
          "the ordinary word 'Ten' is not a hit for the abbreviation TEN")

    check(leak_hit(["Given SLEEP was ruled out."], ent, ["SLE"], []) is None,
          "word-boundary matching: 'SLE' does not fire inside 'SLEEP'")
    check(leak_hit(["Anti-SLE antibodies present."], ent, ["SLE"], []) == "SLE",
          "a standalone abbreviation is caught")
    check(leak_hit(["Anti-MG antibodies present."], ent, ["MG"], []) is None,
          "a TWO-character synonym is below the three-character floor and is never "
          "checked — carried over from pipeline 1")
    check(leak_hit(["A 40-year-old man."], ent, [], ["at"]) is None,
          "terms shorter than three characters are ignored")
    check(leak_hit(["Started prednisone."], "prednisone", [], ["Osserman"],
                   check_entity=False) is None,
          "check_entity=False checks only the banned list (a named-concept row)")
    check(leak_hit(["Osserman class III."], "prednisone", [], ["Osserman"],
                   check_entity=False) == "Osserman",
          "check_entity=False still catches a banned term")

    # ── end to end through stage 02, with a stub writer ──
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    cell = {
        "cell_id": "ely_types-diagnosis-0000", "run_id": "p7test", "seed": 7,
        "source": "ely_types", "type_or_group": "diagnosis", "grounded": False,
        "masking": "required", "masked": True, "track": "diseases",
        "entity": {"track": "diseases", "id": "MONDO:0", "display_name": ent,
                   "synonyms": syns},
        "length": "short", "pasted": False, "document_type": None, "style": "clean",
    }
    # `grounded: False` keeps the bridge out of this test; the leak gate is source-agnostic.
    # The leaking row is an ely_types cell, so it also burns its two revision cycles first;
    # the clean row is a failure_groups cell, which has neither revisions nor Gate A.
    cell_leak = cell
    cell_ok = dict(cell, cell_id="failure_groups-symptoms_unnamed-0000",
                   source="failure_groups", type_or_group="symptoms_unnamed")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log, skip = Log(td / "run.log", echo=False), SkipLog(td / "skips.log", echo=False)
        leaking = StubLane("writer_a", responses=lambda p: {
            "query": "A 40-year-old with myasthenia gravis and ptosis. What next?"})
        clean = StubLane("writer_ungrounded", responses=lambda p: {
            "query": "Ptosis and fatigable weakness worsening through the day. Cause?"})
        aux = StubLane("aux", responses=lambda p: {
            "hyde": [{"title": "t1", "abstract": "a1"}, {"title": "t2", "abstract": "a2"}]})

        # the concept check searches for the concept; give it a bridge that finds it
        found = StubBridge(results=lambda q, k: [
            {"work_id": f"W{i}", "title": f"A study of {q}", "abstract": "..."}
            for i in range(40)])
        w = write_questions.QuestionWriter(
            cfg, {"writer_a": leaking, "aux": aux}, found, log, skip, td / "raw.jsonl")
        row = asyncio.run(w.write(cell_leak))
        check(row["status"] == "discard" and row["reason"].startswith("leak:"),
              f"stage 02 discards a leaking masked row (got {row})")
        check(skip.summary().get("LEAK_DISCARD") == 1,
              f"the discard is LOGGED, not silent: {skip.summary()}")

        w2 = write_questions.QuestionWriter(
            cfg, {"writer_ungrounded": clean, "aux": aux}, found, log, skip,
            td / "raw.jsonl")
        row2 = asyncio.run(w2.write(cell_ok))
        check(row2["status"] == "ok", f"a clean masked row passes (got {row2['status']})")
        check(len(row2["sketches"]) == 2,
              "the ungrounded path makes the second call and gets two sketches")
        check((td / "raw.jsonl").exists(),
              "every prompt and raw response is persisted")

    check_resume()
    check_concept_check()
    check_sketch_retry()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def check_resume():
    """A restart redoes neither a written cell nor a terminally skipped one."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "questions.jsonl").write_text(
            '{"qid": "q1", "cell_id": "ely_types-diagnosis-0001"}\n')
        (td / "skips.log").write_text(
            "2026-09-07T09:21:04Z  SKIP  cell='ely_types-diagnosis-0002'  "
            "reason='no_qualifying_grounding'\n"
            "2026-09-07T09:21:05Z  GATE_A_DISCARD  cell='ely_types-diagnosis-0003'\n"
            "2026-09-07T09:21:06Z  LEAK_DISCARD  cell='ely_types-diagnosis-0004'\n"
            "2026-09-07T09:21:07Z  CONCEPT_DISCARD  cell='ely_types-diagnosis-0005'\n"
            "2026-09-07T09:21:08Z  CONCEPT_REDRAW  cell='ely_types-diagnosis-0006'\n"
            "2026-09-07T09:21:09Z  LLM_FAIL  cell='ely_types-diagnosis-0007'\n"
            "2026-09-07T09:21:10Z  SKETCHES_SHORT  qid='x'\n")
        term = write_questions.terminal_cells(td / "skips.log")
        for cid, why in (("ely_types-diagnosis-0002", "SKIP"),
                         ("ely_types-diagnosis-0003", "GATE_A_DISCARD"),
                         ("ely_types-diagnosis-0004", "LEAK_DISCARD"),
                         ("ely_types-diagnosis-0005", "CONCEPT_DISCARD")):
            check(cid in term, f"a {why} is terminal and is not retried")
        for cid, why in (("ely_types-diagnosis-0006", "CONCEPT_REDRAW"),
                         ("ely_types-diagnosis-0007", "LLM_FAIL")):
            check(cid not in term,
                  f"a {why} is a step INSIDE a cell, so a resume still retries it")
        check(len(term) == 4, f"nothing else is treated as terminal ({sorted(term)})")
        check(not write_questions.terminal_cells(td / "nope.log"),
              "a missing skips.log resumes cleanly rather than crashing")


def _cell(entity="doxribtimine", synonyms=(), track="drugs"):
    return {"cell_id": "failure_groups-stacked_constraints-0000", "run_id": "p7test",
            "seed": 5, "source": "failure_groups",
            "type_or_group": "stacked_constraints", "grounded": False,
            "masking": "impossible", "masked": False, "track": track,
            "entity": {"track": track, "id": "RxCUI:1372538",
                       "display_name": entity, "synonyms": list(synonyms)},
            "length": "short", "pasted": False, "document_type": None, "style": "clean"}


def check_concept_check():
    """An ungrounded concept with no literature is re-drawn, not written."""
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    check(cfg["concept_check"]["enabled"] and cfg["concept_check"]["sources"]
          == ["failure_groups"],
          "config: the concept check is on, for the ungrounded source only")

    def docs_for(name, n):
        return [{"work_id": f"W{i}", "title": f"A study of {name}",
                 "abstract": "..."} for i in range(n)]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = Log(td / "run.log", echo=False)

        # the first concept returns almost nothing; whatever is drawn next is real
        seen = []

        def results(query, top_k):
            seen.append(query)
            if query == "doxribtimine":
                return docs_for("doxribtimine", 1)
            return docs_for(query, 40)

        skip = SkipLog(td / "skips.log", echo=False)
        w = write_questions.QuestionWriter(
            cfg, {"writer_ungrounded": StubLane(responses=lambda p: {"query": "q?"}),
                  "aux": StubLane(responses=lambda p: {
                      "hyde": [{"title": "t1", "abstract": "a1"},
                               {"title": "t2", "abstract": "a2"}]})},
            StubBridge(results=results), log, skip, td / "raw.jsonl")
        row = asyncio.run(w.write(_cell()))
        check(row["status"] == "ok", f"the cell is written after a re-draw ({row.get('reason')})")
        check(skip.counts.get("CONCEPT_REDRAW") == 1,
              f"the thin concept is logged as CONCEPT_REDRAW: {dict(skip.counts)}")
        check(row["cell"]["entity"]["display_name"] != "doxribtimine",
              f"the concept actually changed (now {row['cell']['entity']['display_name']!r})")
        check(row["cell"]["track"] == "drugs" and row["cell"]["style"] == "clean"
              and row["cell"]["length"] == "short",
              "the re-drawn cell keeps its planned track, style and length")
        check(row["concept_check"]["redraws"] == 1
              and row["concept_check"]["hits"] >= cfg["concept_check"]["min_hits"],
              f"the check result rides with the row: {row['concept_check']}")

        # a search that returns plenty of documents that do NOT name the concept
        skip2 = SkipLog(td / "skips2.log", echo=False)
        w2 = write_questions.QuestionWriter(
            cfg, {"writer_ungrounded": StubLane(responses=lambda p: {"query": "q?"}),
                  "aux": StubLane(responses=lambda p: {})},
            StubBridge(results=lambda q, k: [{"work_id": f"W{i}", "title": "unrelated",
                                              "abstract": "nothing to do with it"}
                                             for i in range(50)]),
            log, skip2, td / "raw.jsonl")
        row2 = asyncio.run(w2.write(_cell()))
        check(row2["status"] == "discard" and row2["reason"] == "concept_no_literature",
              f"a hit counts only when a result NAMES the concept ({row2.get('reason')})")
        check(skip2.counts.get("CONCEPT_REDRAW")
              == cfg["concept_check"]["max_redraws"] + 1
              and skip2.counts.get("CONCEPT_DISCARD") == 1,
              f"every re-draw and the final discard are logged: {dict(skip2.counts)}")

        # a synonym match counts
        skip3 = SkipLog(td / "skips3.log", echo=False)
        w3 = write_questions.QuestionWriter(
            cfg, {"writer_ungrounded": StubLane(responses=lambda p: {"query": "q?"}),
                  "aux": StubLane(responses=lambda p: {
                      "hyde": [{"title": "t", "abstract": "a"}] * 2})},
            StubBridge(results=lambda q, k: docs_for("Coumadin", 20)),
            log, skip3, td / "raw.jsonl")
        row3 = asyncio.run(w3.write(_cell(entity="warfarin", synonyms=["Coumadin"])))
        check(row3["status"] == "ok" and not skip3.counts.get("CONCEPT_REDRAW"),
              "a result that names a SYNONYM counts as a hit")

        # a grounded cell is not checked at all
        grounded = dict(_cell(), source="ely_types", type_or_group="diagnosis",
                        grounded=True)
        b4 = StubBridge(results=lambda q, k: [])
        skip4 = SkipLog(td / "skips4.log", echo=False)
        w4 = write_questions.QuestionWriter(
            cfg, {"writer_a": StubLane(responses=lambda p: {"query": "q?"}),
                  "aux": StubLane(responses=lambda p: {})},
            b4, log, skip4, td / "raw.jsonl")
        row4 = asyncio.run(w4.write(grounded))
        check(row4.get("reason") == "no_qualifying_grounding",
              "a grounded cell keeps its own mechanism and is never concept-checked")


def check_sketch_retry():
    """A call that returns one sketch is retried once before the row is marked short."""
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    check(int(cfg["sketch_retries"]) >= 1, "config: the sketch call is retried")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        log = Log(td / "run.log", echo=False)

        calls = {"n": 0}

        def aux(_prompt):
            calls["n"] += 1
            if calls["n"] == 1:                 # first call: only one sketch back
                return {"hyde": [{"title": "t1", "abstract": "a1"}]}
            return {"hyde": [{"title": "t1", "abstract": "a1"},
                             {"title": "t2", "abstract": "a2"}]}

        skip = SkipLog(td / "skips.log", echo=False)
        w = write_questions.QuestionWriter(
            cfg, {"aux": StubLane(responses=aux)}, StubBridge(), log, skip,
            td / "raw.jsonl")
        row = {"qid": "p7pilot-failure_groups-escalation-0007",
               "cell_id": "failure_groups-escalation-0007", "text": "q?",
               "sketches": [{"title": "only one", "abstract": "a"}],
               "sketches_missing": False, "cell": _cell()}
        got = asyncio.run(w.sketches_for(row))
        check(calls["n"] == 2, f"the short call is retried once (made {calls['n']} calls)")
        check(len(got["sketches"]) == 2 and not got["sketches_missing"],
              f"the retry fills the row to two sketches ({len(got['sketches'])})")
        check(skip.counts.get("SKETCHES_SHORT") == 1,
              f"the short first attempt is still logged: {dict(skip.counts)}")

        # a call that stays short marks the row, and does not lose the one it got
        skip2 = SkipLog(td / "skips2.log", echo=False)
        w2 = write_questions.QuestionWriter(
            cfg, {"aux": StubLane(responses=lambda p: {
                "hyde": [{"title": "t1", "abstract": "a1"}]})},
            StubBridge(), log, skip2, td / "raw.jsonl")
        got2 = asyncio.run(w2.sketches_for(dict(row)))
        check(got2["sketches_missing"] and len(got2["sketches"]) == 1,
              f"a persistently short call marks the row and keeps what it got: "
              f"{len(got2['sketches'])} sketch(es), missing={got2['sketches_missing']}")


def test_leak_check():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
