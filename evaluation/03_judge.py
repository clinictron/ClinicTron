#!/usr/bin/env python
"""Stage 3: an independent model judges, per question, which system returned the better list of sources.

The judge sees the question and two lists, A and B: ours (stage 2, the ranking after a chosen hop) and the
citations of the comparator system's response. It is not told which is which; a seeded coin decides per question
which system is list A. Both lists are written the same way from the same corpus record (type, title,
journal, year, citation count, abstract). A source with no record in our corpus (an FDA label, a guideline,
a paper newer than the corpus, a PubMed hit outside the corpus) is written as title, publisher and year for
both systems. When the run folder holds pubmed_abstracts.json (02b_pubmed_abstracts.py), a source without an
abstract in our corpus is shown with the abstract of its accepted PubMed record, in both lists.
The comparator's answer text is never shown. `citations_per_list` cuts both lists to their first
N sources; 0 shows every source of both.

Three switches in the config give sensitivity checks (all off in the main comparison):
  flip           every question is shown in the opposite order (ours is A where it was B)
  equal_length   both lists are cut to the length of the shorter one
  records_only   only sources with a record in our corpus are kept, in both lists (applied before equal_length)

  usage: 03_judge.py --run-dir RUN/arms/<arm>_<agent> --after-round R [--config judge.yaml]

--after-round R judges the ranking after search round R (hop R); the output folder gets the suffix _hopR.
Without it the ranking after the last round is judged.

Writes RUN/arms/<arm>_<agent>/judge/<name>/ (name from the config, default the model): inputs.jsonl (the exact prompts),
judgments.jsonl, summary.json.
Resumable: a question already in judgments.jsonl is not judged again.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

import tools
from common import HERE, env, secret
from llm import call_openrouter, fill, parse

TRIES = 5  # attempts at one judgment (errors, replies that name no winner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "judge.yaml"))
    ap.add_argument("--after-round", type=int, help="judge the snapshot after this search round, not the final ranking")
    ap.add_argument("--inputs-only", action="store_true", help="write inputs.jsonl and stop: no judge call")
    a = ap.parse_args()
    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(os.path.join(HERE, cfg["prompt"])) as fh:
        prompt = fh.read()
    with open(os.path.join(HERE, cfg["lines"])) as fh:
        line = yaml.safe_load(fh)
    n = cfg["citations_per_list"]
    cut = lambda sources: sources[:n] if n else sources
    out = f"{a.run_dir}/judge/{cfg.get('name') or cfg['model'].replace('/', '_')}" + \
          (f"_hop{a.after_round}" if a.after_round is not None else "")
    os.makedirs(out, exist_ok=True)

    # The two lists per question. A source is {"work_id"} (a corpus record) or {"title", "venue", "year"}.
    def source(c):
        if c["id"].startswith("FDA:"):
            return {"title": fill(line["fda_title"], {"name": c["brand_name"]}), "venue": line["fda_venue"], "year": c["updated"][:4]}
        if c.get("work_id"):
            return {"work_id": c["work_id"], "key": c["id"]}
        return {"title": c["title"], "venue": c["journal"], "year": c["year"], "key": c["id"]}  # a PubMed hit outside the corpus

    with open(f"{a.run_dir}/citations.json") as fh:
        ours = json.load(fh)
    if a.after_round is not None:  # the ranking after that round stands in for the final one (0: the zero_shot arm)
        ours = {q: {**v, "citations": v["snapshots"][str(a.after_round)]} for q, v in ours.items()}
    with open(env("AGENT_EVAL_QUESTIONS")) as fh:
        cmp = {q["question_id"]: cut(q["comparator_citations"]) for q in json.load(fh)["questions"]}
    with open(env("AGENT_EVAL_CITATION_MATCHES")) as fh:
        match = {(int(r["rank"]), int(r["n"])): r["work_id"] for r in csv.DictReader(fh) if r["in_corpus"] == "True"}
    lists = {}
    for qid, q in ours.items():
        lists[qid] = {
            "ours": [source(c) for c in cut(q["citations"])],
            "comparator": [{"key": f"cmp:{q['rank']}:{c['n']}", **({"work_id": match[q["rank"], c["n"]]}
                              if (q["rank"], c["n"]) in match else
                              {"title": c["title"], "venue": c["venue"], "year": c["year"] or ""})} for c in cmp[qid]]}
    for both in lists.values():
        if cfg.get("records_only"):
            both.update({side: [s for s in sources if "work_id" in s] for side, sources in both.items()})
        if cfg.get("equal_length"):
            m = min(map(len, both.values()))
            both.update({side: sources[:m] for side, sources in both.items()})
    meta = tools.papers(s["work_id"] for both in lists.values() for side in both.values() for s in side if "work_id" in s)

    fill_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.run_dir))), "pubmed_abstracts.json")
    pubmed = None
    if os.path.exists(fill_path):
        with open(fill_path) as fh:
            pubmed = json.load(fh)
    filled = set()

    def render(sources):
        rows = []
        for i, s in enumerate(sources, 1):
            rec, kind = ({**meta[s["work_id"]], "n": i}, "paper") if "work_id" in s else ({**s, "n": i}, "no_record")
            if pubmed is not None and len((rec.get("abstract") or "").strip()) < 50 and "key" in s:
                hit = pubmed[s["key"]]["pubmed"]  # a KeyError means the file was built for other lists: rerun 02b
                if hit and hit["accepted"] and len(hit["abstract"]) >= 50:
                    rec["abstract"], kind = hit["abstract"], "paper" if "work_id" in s else "no_record_abstract"
                    filled.add(s["key"])
            rows.append(fill(line[kind], rec))
        return "\n".join(rows)

    inputs = []
    for qid, q in ours.items():
        ours_is = "A" if (random.Random(f"{cfg['seed']}|{qid}").random() < 0.5) != bool(cfg.get("flip")) else "B"
        first, second = ("ours", "comparator") if ours_is == "A" else ("comparator", "ours")
        inputs.append({"qid": qid, "rank": q["rank"], "ours_is": ours_is, "n_ours": len(lists[qid]["ours"]),
                       "n_comparator": len(lists[qid]["comparator"]),
                       "prompt": fill(prompt, {"question": q["question"], "list_a": render(lists[qid][first]),
                                               "list_b": render(lists[qid][second])})})
    with open(f"{out}/inputs.jsonl", "w") as fh:
        for row in inputs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if a.inputs_only:
        print(f"{len(inputs)} inputs, {len(filled)} sources shown with a PubMed abstract -> {out}/inputs.jsonl")
        return

    judged = {}
    if os.path.exists(f"{out}/judgments.jsonl"):
        with open(f"{out}/judgments.jsonl") as fh:
            judged = {j["qid"]: j for j in map(json.loads, fh)}
    key = secret("OPENROUTER_API_KEY")

    def judge(row):
        cost = 0.0
        for attempt in range(TRIES):
            time.sleep(30 * attempt)  # a rate limit or a credit top-up in progress clears in seconds to minutes
            try:
                reply, data, _, c = call_openrouter(cfg["model"], [{"role": "user", "content": row["prompt"]}], key)
            except Exception as e:
                print(f"rank {row['rank']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            if not reply:  # the provider answered with an error body, not a message
                print(f"rank {row['rank']}: empty reply: {json.dumps(data.get('error') or data)[:300]}", flush=True)
                continue
            cost += c
            verdict = parse(reply) or {}
            if verdict.get("better") in ("A", "B"):
                return {**{k: row[k] for k in ("qid", "rank", "ours_is", "n_ours", "n_comparator")},
                        "better": verdict["better"],
                        "winner": "ours" if verdict["better"] == row["ours_is"] else "comparator",
                        "reason": verdict.get("reason"), "model": data.get("model"), "cost_usd": cost, "raw": data}
        print(f"rank {row['rank']}: no verdict after {TRIES} tries", flush=True)

    with ThreadPoolExecutor(cfg["concurrent_calls"]) as ex, open(f"{out}/judgments.jsonl", "a") as fh:
        for j in ex.map(judge, [row for row in inputs if row["qid"] not in judged]):
            if j:
                judged[j["qid"]] = j
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
                fh.flush()

    rows = sorted(judged.values(), key=lambda j: j["rank"])
    wins = sum(j["winner"] == "ours" for j in rows)
    summary = {"model": cfg["model"], "config": cfg,
               "sha256": {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                          for f in (cfg["prompt"], cfg["lines"], "03_judge.py", "llm.py", "tools.py")},
               "pubmed_abstracts": pubmed is not None and {"sources_filled": len(filled),
                                                           "sha256": hashlib.sha256(open(fill_path, "rb").read()).hexdigest()},
               "questions_judged": len(rows), "questions_in_run": len(ours),
               "ours_better": wins, "comparator_better": len(rows) - wins,
               "ours_better_pct": round(100 * wins / len(rows), 1) if rows else None,
               "ours_shown_as_A": sum(j["ours_is"] == "A" for j in rows),
               "list_A_judged_better": sum(j["better"] == "A" for j in rows),
               "cost_usd": round(sum(j["cost_usd"] for j in rows), 4),
               "per_question": [{k: j[k] for k in ("rank", "ours_is", "better", "winner", "n_ours", "n_comparator",
                                                    "reason")} for j in rows]}
    with open(f"{out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(f"{len(rows)}/{len(ours)} judged by {cfg['model']}: ours better {wins}, comparator better "
          f"{len(rows) - wins}; cost ${summary['cost_usd']:.2f} -> {out}/summary.json")


if __name__ == "__main__":
    main()
