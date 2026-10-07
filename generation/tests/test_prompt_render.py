#!/usr/bin/env python3
"""test_prompt_render.py — render every prompt the generation stages send, and write them out.

The rendered prompts are written verbatim for review before any LLM call, to
`rendered_examples/`:

  rendered_examples/types/<type>.md            one grounded writer prompt per Ely type
  rendered_examples/subclasses/<id>.md         two taxonomy-subclass writer prompts
  rendered_examples/groups/<group>.md          one ungrounded writer prompt per group
  rendered_examples/style_length/<combo>.md    all eight style x length combinations
  rendered_examples/aux/<name>.md              the grounding pick, Gate A, sketch and
                                               revision prompts, rendered
  rendered_examples/grader.md                  one full grader call, system + user
  rendered_examples/INDEX.md                   what is here and what to check

It also asserts:
  * no rendered prompt carries an unfilled slot
  * every group block appears in its prompt VERBATIM, byte for byte
  * every Ely type's ask templates appear
  * the unified answer-leak sentence is in BOTH writer prompts, and the
    "do not restate the chosen papers' findings" clause is in the grounded one only
  * the old "real physician questions are short" length line is gone from the group prompts
  * masked rows get LATENT ENTITY, named rows get NAMED ENTITY
  * a pasted cell names its document type; a plain long cell does not
  * the grader system prompt is the as-run grader prompt byte for byte, and the
    rendered user message SHOWS journal, year and citations

No network, no GPU, no LLM.
"""
from __future__ import annotations

import re
import sys
import hashlib
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parent
# sha256 of the grader system prompt as run (trailing newlines stripped)
GRADER_SYSTEM_SHA256 = "857370fe07958cb75cae13dd0ccf5c8a94f3b6f271af2fe56318650599aadf6a"
sys.path.insert(0, str(GEN))

import importlib.util                                       # noqa: E402
from common import prompts as P                             # noqa: E402
from common.config import load_config                       # noqa: E402

_sp = importlib.util.spec_from_file_location("grade_pools", str(GEN / "05_grade_pools.py"))
grade_pools = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(grade_pools)

OUT = Path(os.environ.get("RENDER_OUT") or tempfile.mkdtemp()) / "rendered_examples"
LEAK_RULE = "Do NOT include any answer, recommendation, or answer-shaped hint."
GROUNDED_CLAUSE = "restate the chosen papers' findings"
DROPPED_LINE = "real physician questions are short"
FAILURES: list[str] = []


def check(ok: bool, msg: str):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILURES.append(msg)


def write(rel: str, text: str) -> Path:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def cell(source, unit, *, masked, style="clean", length="medium", pasted=False,
         document_type=None, entity="warfarin", track="drugs", synonyms=(),
         subclass=None):
    return {"cell_id": f"{source}-{unit}-0000", "run_id": "p7render", "seed": 1,
            "source": source, "type_or_group": unit,
            "grounded": source != "failure_groups",
            "masking": "required" if masked else "impossible", "masked": masked,
            "track": track,
            "entity": {"track": track, "id": "X:1", "display_name": entity,
                       "synonyms": list(synonyms)},
            "length": length, "pasted": pasted, "document_type": document_type,
            "style": style, **({"subclass": subclass} if subclass else {})}


GROUNDING_BLOCK = (
    "[G1] Warfarin management in atrial fibrillation: a narrative review — Warfarin "
    "remains widely used for stroke prevention in atrial fibrillation. This review "
    "covers initiation, INR targets, bridging, and the interactions that most often "
    "destabilise control.\n"
    "[G2] Bleeding risk with vitamin K antagonists in older adults — A cohort of 4,812 "
    "patients aged 75 and over followed for three years, reporting major bleeding rates "
    "by INR band and by concomitant medication.")


def main() -> int:
    cfg = load_config(str(GEN / "configs/generation.yaml"))
    root = cfg.prompts_dir
    written: list[str] = []

    # ── one grounded writer prompt per Ely type ──
    styles = ["clean", "messy"]
    lengths = [("short", False), ("medium", False), ("long", False), ("long", True)]
    for i, t in enumerate(cfg["ely_types"]):
        spec = P.load_type_spec(t, root)
        masked = spec["masking"] != "impossible"
        style = styles[i % 2]
        length, pasted = lengths[i % 4]
        doc = cfg["lengths"]["document_types"][i % 6] if pasted else None
        c = cell("ely_types", t, masked=masked, style=style, length=length,
                 pasted=pasted, document_type=doc,
                 entity="warfarin" if "drugs" in spec["tracks"] else "sarcoidosis",
                 track=spec["tracks"][0],
                 synonyms=["Coumadin"] if "drugs" in spec["tracks"] else [])
        text = P.render_writer(cfg, c, grounding_block=GROUNDING_BLOCK)
        written.append(str(write(f"types/{t}.md", text).relative_to(OUT.parent)))
        check(not P.slots(text), f"types/{t}: no unfilled slot")
        for ask in spec["asks"]:
            check(ask in text, f"types/{t}: ask template present — {ask[:40]!r}")
        check(LEAK_RULE in text, f"types/{t}: carries the unified answer-leak rule")
        check(GROUNDED_CLAUSE in text, f"types/{t}: carries the grounded-only clause")
        want = "LATENT ENTITY" if masked else "NAMED ENTITY"
        check(want in text, f"types/{t}: masking tag {spec['masking']} -> {want}")

    # ── two taxonomy-subclass prompts, one masked and one named ──
    _sp2 = importlib.util.spec_from_file_location("plan_cells", str(GEN / "01_plan_cells.py"))
    plan_cells = importlib.util.module_from_spec(_sp2)
    _sp2.loader.exec_module(plan_cells)
    subs = plan_cells.load_subclasses(Path(cfg["subclasses"]["taxonomy"]),
                                      set(cfg["subclasses"]["exclude_types"]),
                                      cfg["anchor_to_track"])
    picked = [next(s for s in subs if s["masking"] == "required"),
              next(s for s in subs if s["masking"] == "impossible")]
    for s in picked:
        c = cell("subclasses", s["id"], masked=(s["masking"] == "required"),
                 style="messy", length="medium", track=s["tracks"][0],
                 entity="warfarin" if s["tracks"][0] == "drugs" else "sarcoidosis",
                 subclass={k: s[k] for k in ("type_id", "type_name", "subtype_id",
                                             "subtype_name", "subtype_definition",
                                             "ask")})
        text = P.render_writer(cfg, c, grounding_block=GROUNDING_BLOCK)
        written.append(str(write(f"subclasses/{s['id']}.md", text).relative_to(OUT.parent)))
        check(not P.slots(text), f"subclasses/{s['id']}: no unfilled slot")
        check(s["ask"] in text, f"subclasses/{s['id']}: the taxonomy's own ask is used")

    # ── one ungrounded writer prompt per failure group ──
    for i, g in enumerate(sorted(cfg["failure_group_weights"])):
        style = styles[i % 2]
        length, pasted = lengths[i % 4]
        doc = cfg["lengths"]["document_types"][i % 6] if pasted else None
        tags = cfg["anchors_and_masking"][g]
        c = cell("failure_groups", g, masked=tags["masking"] == "required",
                 style=style, length=length, pasted=pasted, document_type=doc,
                 track=tags["tracks"][0],
                 entity="warfarin" if tags["tracks"][0] == "drugs" else "sarcoidosis")
        text = P.render_writer(cfg, c)
        written.append(str(write(f"groups/{g}.md", text).relative_to(OUT.parent)))
        check(not P.slots(text), f"groups/{g}: no unfilled slot")
        block = P.load_group_block(g, root).replace("{seed_entity}", c["entity"]["display_name"])
        check(block in text, f"groups/{g}: the group block appears VERBATIM")
        check(LEAK_RULE in text, f"groups/{g}: carries the unified answer-leak rule")
        check(GROUNDED_CLAUSE not in text,
              f"groups/{g}: does NOT carry the grounded-only clause")
        check(DROPPED_LINE not in text,
              f"groups/{g}: pipeline 2's 'real physician questions are short' is gone")

    # ── every style x length combination ──
    for style in styles:
        for length, pasted in lengths:
            name = f"{style}_{length}{'_pasted' if pasted else ''}"
            doc = "discharge summary" if pasted else None
            c = cell("failure_groups", "stacked_constraints", masked=False,
                     style=style, length=length, pasted=pasted, document_type=doc)
            text = P.render_writer(cfg, c)
            written.append(str(write(f"style_length/{name}.md", text).relative_to(OUT.parent)))
            check(not P.slots(text), f"style_length/{name}: no unfilled slot")
            check(P.style_block(style, root).split("\n")[0] in text,
                  f"style_length/{name}: the {style} style block is present")
            if pasted:
                check(doc in text, f"style_length/{name}: names its document type")
            elif length == "long":
                check("250 to 450 words." in text and "discharge summary" not in text,
                      "style_length/clean_long: plain long prose, no pasted document")

    # ── the auxiliary prompts, rendered ──
    aux = {
        "grounding_pick.md": dict(entity_name="warfarin", synonyms="Coumadin", n=2,
                                  results_block=GROUNDING_BLOCK),
        "gate_a.md": dict(ask_line=P.load("gate_a_ask.md", root),
                          query="A 78-year-old on long-term anticoagulation presents "
                                "with an INR of 8.4 and no bleeding. What next?"),
        "equivalence.md": dict(answer="vitamin K antagonist over-anticoagulation",
                               entity="warfarin"),
        "sketches.md": dict(query="A 78-year-old with an INR of 8.4 and no bleeding. "
                                  "What next?"),
        "revision_leak.md": dict(term="warfarin", twin_note=""),
        "revision_gate_a.md": dict(rivals="dabigatran toxicity, hepatic failure",
                                   twin_note=""),
    }
    for name, values in aux.items():
        text = P.fill(P.load(name, root), **values)
        written.append(str(write(f"aux/{name}", text).relative_to(OUT.parent)))
        check(not P.slots(text), f"aux/{name}: no unfilled slot")

    # ── one full grader call ──
    papers = [
        grade_pools._Paper({
            "title": "Management of excessive anticoagulation without bleeding",
            "abstract": "A randomised comparison of oral vitamin K against withholding "
                        "warfarin in patients with an INR above 6 and no bleeding.",
            "journal_name": "Annals of Internal Medicine", "publication_year": 2019,
            "cited_by_count": 412, "doc_type": "article"}),
        grade_pools._Paper({
            "title": "Warfarin pharmacogenomics: a narrative overview",
            "abstract": "Reviews CYP2C9 and VKORC1 variants and their effect on dose "
                        "requirement. No guidance on acute over-anticoagulation.",
            "journal_name": "Pharmacogenomics", "publication_year": 2014,
            "cited_by_count": 88, "doc_type": "review"}),
        grade_pools._Paper({
            "title": "Letter: our experience with point-of-care INR meters",
            "abstract": "A single-centre note on device agreement.",
            "journal_name": "Thrombosis Research", "publication_year": 2021,
            "cited_by_count": 3, "doc_type": "letter"}),
    ]
    system, user = grade_pools._render(
        cfg, "A 78-year-old on long-term anticoagulation presents with an INR of 8.4 "
             "and no bleeding. What should be done next?", papers)
    written.append(str(write("grader.md",
                             "# Grader call (system + user), rendered\n\n"
                             "## SYSTEM — prompts/grader.jinja\n\n```\n" + system +
                             "\n```\n\n## USER — prompts/grader_user.jinja\n\n```\n" +
                             user + "\n```\n").relative_to(OUT.parent)))
    check(hashlib.sha256(system.rstrip("\n").encode()).hexdigest() == GRADER_SYSTEM_SHA256,
          "the grader system prompt is the as-run text, byte for byte (sha256)")
    check("Annals of Internal Medicine" in user and "2019" in user
          and "cites:412" in user,
          "the grader user message SHOWS journal, year and citations")
    check("[letter]" in user, "the doc_type tag is shown for every candidate")

    # ── the writer-prompt override keys and comment stripping, and that they are inert ──
    live = sorted(q for q in (GEN / "prompts").rglob("*")
                  if q.is_file() and "ablation" not in q.parts)
    dirty = [q.name for q in live if "<!--" in q.read_text()]
    check(not dirty,
          f"P4: no live prompt file carries an HTML comment, so stripping them changes "
          f"nothing the live pipeline sends (checked {len(live)} files){'' if not dirty else ': ' + str(dirty)}")
    check(P.strip_comments("keep\n<!-- ablation: gone\nstill gone -->\nkeep2\n")
          == "keep\nkeep2\n",
          "P4: an ablation marker is removed, and only the marker")
    check(P.writer_prompt_files({}) == P.WRITER_PROMPT_DEFAULTS,
          "P1: a config that names no writer prompt gets the live files")
    check(P.writer_prompt_files(
              {"writer_prompts": {"grounded": "ablation/arm06_minus_masking.md"}})
          == {"grounded": "ablation/arm06_minus_masking.md",
              "ungrounded": "writer_ungrounded.md"},
          "P1: naming one path leaves the other on its live file")
    _wq = importlib.util.spec_from_file_location(
        "write_questions", str(GEN / "02_write_questions.py"))
    _wm = importlib.util.module_from_spec(_wq)
    _wq.loader.exec_module(_wm)
    _w = _wm.QuestionWriter(cfg, {}, None, lambda *a, **k: None, None, None)
    check(_w.gcache_path is None,
          "P2: the shared grounding cache is OFF for the live config")
    check(_wm.QuestionWriter(cfg, {}, None, lambda *a, **k: None, None, None,
                             grounding_cache="/tmp/gc.jsonl").gcache_path
          == "/tmp/gc.jsonl",
          "P2: a caller can switch the cache on without touching the config")

    # ── index ──
    index = ["# rendered_examples — pipeline 7 prompt sign-off",
             "",
             "Produced by `tests/test_prompt_render.py`. Every file is the EXACT text a "
             "model would receive; nothing here was sent anywhere.",
             "",
             "What to read for:",
             "1. does the task block match the type or group it claims?",
             "2. does the masking rule match the tag (LATENT vs NAMED ENTITY)?",
             "3. does the length and style instruction read like a real instruction?",
             "4. do the two NEW groups (case_based, escalation) and the two NEW types "
             "(diagnostic_criteria, etiology_risk) say what they should?",
             "5. does the grader prompt match the as-run grader prompt?",
             "", "## Files", ""]
    index += [f"- `{p}`" for p in sorted(written)]
    write("INDEX.md", "\n".join(index) + "\n")

    print(f"\nwrote {len(written)} rendered prompts to {OUT}")
    print(f"{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def test_prompt_render():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
