
"""05_grade_pools.py — grade every pooled candidate against its question with the LLM grader.

The grader system message is `prompts/grader.jinja` and the user message is
`prompts/grader_user.jinja` (both named in the config).

Settings, all from the config (`grader:` and `models.grader:`):
  temperature      0.0
  window           100 candidates per call
  shuffle          seeded by the cell's own seed, so candidate order carries no provenance
  metadata         shown: journal, year, citations, doc_type
  truncation       none (abstract_char_cap 0)
  score floor      1; an id the grader omits is re-asked once, then imputed to the floor
                   and logged as TEACHER_OMIT

`--print-prompt` renders one full grader call and exits without calling the model.
"""
from __future__ import annotations

import argparse
import atexit
import os
import time
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import prompts as P
from common.config import load_config, load_secrets
from common.llm import Budget, build_lanes
from common.logging import Log, SkipLog, append_jsonl, load_keys
from common.retrieval import MetaResolver


class _Paper:
    """Attribute bag matching grader_user.jinja's fields. Metadata is shown, so journal,
    year, citations and doc_type carry their real values."""

    def __init__(self, meta: dict):
        self.doc_type = meta.get("doc_type") or "article"
        self.title = meta.get("title") or ""
        self.abstract = meta.get("abstract") or ""
        self.journal_name = meta.get("journal_name") or None
        self.publication_year = meta.get("publication_year") or None
        cited = meta.get("cited_by_count")
        self.cited_by_count = "n/a" if cited is None else cited


def _render(cfg, query: str, papers: list[_Paper]) -> tuple[str, str]:
    """(system, user). Jinja is used only for the user template, which is a jinja file."""
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(cfg.prompts_dir)), autoescape=False,
                      keep_trailing_newline=True)
    system = P.load(cfg["grader"]["system_prompt"], cfg.prompts_dir)
    user = env.get_template(cfg["grader"]["user_prompt"]).render(
        query=query, papers=papers, guidelines=[], fda_labels=[])
    return system, user


async def grade_pool(grader, query: str, pool: list[dict], meta: dict, *, seed: int,
                     cfg, log, qid: str) -> dict[str, int]:
    """Grade every candidate. Windowed calls + one re-ask of omitted ids + floor.

    Calls are windowed at `grader.window`; an omitted id is imputed to the configured
    score floor.
    """
    g = cfg["grader"]
    floor = int(g.get("score_floor", 1))
    window = int(g.get("window", 100))
    cap = int(g.get("abstract_char_cap", 0) or 0)

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    _bf: dict[str, int] = {}

    async def _call(indices: list[int]) -> dict[int, int]:
        papers = []
        for i in indices:
            m = dict(meta.get(pool[i]["work_id"], {}))
            m.setdefault("title", pool[i].get("title") or "")
            m.setdefault("abstract", pool[i].get("abstract") or "")
            if not m.get("title"):
                m["title"] = pool[i].get("title") or ""
            if not m.get("abstract"):
                m["abstract"] = pool[i].get("abstract") or ""


            if not m.get("doc_type") and pool[i].get("doc_type"):
                m["doc_type"] = pool[i]["doc_type"]
                _bf["doc_type"] = _bf.get("doc_type", 0) + 1
            if m.get("publication_year") is None and pool[i].get("year") is not None:
                m["publication_year"] = pool[i]["year"]
                _bf["year"] = _bf.get("year", 0) + 1
            for _k in ("journal_name", "cited_by_count"):
                if m.get(_k) in (None, "") and pool[i].get(_k) not in (None, ""):
                    m[_k] = pool[i][_k]
                    _bf[_k] = _bf.get(_k, 0) + 1
            if cap and len(m["abstract"]) > cap:
                log(f"ABSTRACT_CAP_HIT qid={qid} wid={pool[i]['work_id']} "
                    f"len={len(m['abstract'])} cap={cap}")
                m["abstract"] = m["abstract"][:cap]
            papers.append(_Paper(m))
        system, user = _render(cfg, query, papers)
        parsed = await grader.generate_json(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], temperature=float(g["temperature"]))
        out: dict[int, int] = {}
        for e in parsed.get("scores", []) or []:
            if not isinstance(e, dict) or "id" not in e or "score" not in e:
                continue
            m2 = re.match(r"^[Pp]?(\d+)$", str(e["id"]).strip())
            if not m2:
                continue
            p_idx = int(m2.group(1)) - 1
            if 0 <= p_idx < len(indices):
                try:
                    s = int(round(float(e["score"])))
                except (TypeError, ValueError):
                    continue
                out[indices[p_idx]] = max(floor, min(10, s))
        return out

    scores: dict[int, int] = {}
    for start in range(0, len(order), window):
        scores.update(await _call(order[start:start + window]))
    omitted = [i for i in order if i not in scores]
    if omitted and g.get("reask_omitted", True):
        log(f"TEACHER_REASK qid={qid} n_omitted={len(omitted)}")
        try:
            for start in range(0, len(omitted), window):
                scores.update(await _call(omitted[start:start + window]))
        except Exception as exc:
            log(f"TEACHER_REASK_FAIL qid={qid} err={exc}")

    result: dict[str, int] = {}
    for i in order:
        wid = pool[i]["work_id"]
        if i in scores:
            result[wid] = scores[i]
        else:
            result[wid] = floor
            log(f"TEACHER_OMIT qid={qid} doc={wid} imputed={floor}")


    if _bf:
        log(f"META_BACKFILL qid={qid} n_docs={len(pool)} "
            + " ".join(f"{k}={v}" for k, v in sorted(_bf.items())))
    return result


async def run(cfg, rows, pools, meta, out: Path):
    log = Log(out / "run.log")
    skip = SkipLog(out / "skips.log")
    secrets = load_secrets(cfg["paths"]["secrets"])
    budget = Budget(cfg["budget"]["abort_usd"],
                    str(out / cfg["budget"]["cost_state_file"]),
                    warn_usd=cfg["budget"]["warn_usd"],
                    initial_usd=cfg["budget"]["initial_usd"])
    lanes = build_lanes(cfg, budget, log, secrets=secrets)
    grader = lanes["grader"]
    sem = asyncio.Semaphore(int(cfg["models"]["grader"].get("concurrency", 32)))

    async def one(row):
        pool = pools[row["qid"]]["pool"]
        async with sem:
            try:
                scores = await grade_pool(grader, row["text"], pool, meta,
                                          seed=row["cell"]["seed"], cfg=cfg,
                                          log=log, qid=row["qid"])
            except Exception as exc:
                skip.note("GRADE_FAIL", qid=row["qid"], err=str(exc))
                return None
        append_jsonl(out / "scores.jsonl", {
            "id": row["qid"], "n_docs": len(pool), "n_scored": len(scores),
            "scores": scores})
        return scores

    done = await asyncio.gather(*[one(r) for r in rows])
    ok = sum(1 for d in done if d)
    log(f"GRADE done ok={ok} of {len(rows)} spend=${budget.total:.2f} "
        f"skips={skip.summary()}")
    return {"rows": len(rows), "graded": ok, "spend_usd": round(budget.total, 4),
            "skips": skip.summary()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pools", required=True, help="pools.jsonl from stage 03")
    ap.add_argument("--doc-meta", default=None, help="doc_meta_cache.jsonl from stage 03")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-blank-docs", action="store_true",
                    help="grade even when candidates have no title/abstract ; the scores "
                         "for those documents are meaningless")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--print-prompt", action="store_true",
                    help="render the first grader call and exit; makes NO calls")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out_dir) if a.out_dir else cfg.out_dir("05_grade")
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.questions) if l.strip()]
    pools = {json.loads(l)["qid"]: json.loads(l) for l in open(a.pools) if l.strip()}
    rows = [r for r in rows if r["qid"] in pools]
    meta_path = a.doc_meta or str(Path(a.pools).parent / "doc_meta_cache.jsonl")
    meta = MetaResolver(None, {}, cache_paths=[meta_path]).mem

    if a.print_prompt:
        row = rows[0]
        pool = pools[row["qid"]]["pool"][:int(cfg["grader"]["window"])]
        papers = [_Paper({**meta.get(d["work_id"], {}),
                          "title": d.get("title") or "",
                          "abstract": d.get("abstract") or ""}) for d in pool]
        system, user = _render(cfg, row["text"], papers)
        print("========== SYSTEM ==========\n" + system)
        print("========== USER ==========\n" + user)
        return 0


    lock = out / "grader.pid"
    if lock.exists():
        try:
            other = int(lock.read_text().split()[0])
            os.kill(other, 0)
        except (ValueError, ProcessLookupError, PermissionError, IndexError):
            pass
        else:
            raise SystemExit(
                f"FATAL: another grader (pid {other}) is already writing {out}. Wait for it, "
                f"or point --out-dir somewhere else. Remove {lock} if that process is gone.")
    lock.write_text(f"{os.getpid()} {time.time():.0f}\n")
    atexit.register(lambda: lock.unlink(missing_ok=True))

    done = load_keys(out / "scores.jsonl", "id")
    rows = [r for r in rows if r["qid"] not in done]
    if a.limit:
        rows = rows[:a.limit]


    window = int(cfg["grader"]["window"])
    blank = 0
    for r in rows:
        for d in pools[r["qid"]]["pool"][:window]:
            w = str(d["work_id"])
            if w.startswith("synth:"):
                continue
            m = meta.get(w) or {}
            if not (d.get("title") or m.get("title") or d.get("abstract") or m.get("abstract")):
                blank += 1
    if blank:
        frac = blank / max(sum(len(pools[r["qid"]]["pool"][:window]) for r in rows), 1)
        msg = (f"GRADE_REFUSED {blank} candidate(s) ({frac:.1%} of the grader window) have no "
               f"title and no abstract, in pools.jsonl or in {meta_path}. Grading them scores "
               f"empty documents. Finish the pooling run (it resolves metadata at the end) or "
               f"pass --doc-meta a cache that covers them; --allow-blank-docs overrides.")
        print(msg, flush=True)
        if not a.allow_blank_docs:
            raise SystemExit(f"FATAL: {msg}")
    summary = asyncio.run(run(cfg, rows, pools, meta, out))
    (out / "grade_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
