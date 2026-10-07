#!/usr/bin/env python
"""Stage 1: write the questions of a run to RUN/sample.jsonl. The config chooses the benchmark.

  pmc_patients   N queries (fixed seed) from the OFFICIAL PMC-Patients ReCDS-PAR test split:
                 {qid, text, gold_pmids}; prints how many gold PMIDs the 58.5M-paper corpus holds
                 (context for the dense arms' ceiling; nothing is filtered).
  physician_questions   the Real-POCQi questions that have a comparator response, in sample order, with the
                 question text exactly as published (the sentence appended for the comparator, "In your
                 answer, please cite at least 15 papers.", removed again): {qid, rank, specialty, text}.

  usage: 01_sample_questions.py --run-dir RUN --config CFG [--n 200] [--seed 42] [--limit N]
"""
import argparse
import collections
import json
import os
import random

import yaml

from common import env, psql_rows, sql_list

OE_SUFFIX = "\n\nIn your answer, please cite at least 15 papers."


def pmc_patients(n, seed):
    raw = env("PMC_PATIENTS_RAW")
    qrels = collections.defaultdict(set)
    with open(f"{raw}/PAR/qrels_test.tsv") as fh:
        next(fh)  # header
        for line in fh:
            qid, pmid, _score = line.rstrip("\n").split("\t")
            qrels[qid].add(pmid)
    queries = {}
    with open(f"{raw}/queries/test_queries.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]
    eligible = sorted(q for q in queries if q in qrels)
    print(f"test queries={len(queries)}  with qrels={len(eligible)}")
    sample = random.Random(seed).sample(eligible, n)
    rows = [{"qid": q, "text": queries[q], "gold_pmids": sorted(qrels[q])} for q in sample]
    gold = sorted({p for r in rows for p in r["gold_pmids"]})
    found = psql_rows(f"SELECT count(*) FROM papers_index WHERE pmid IN ({sql_list(gold)})")
    print(f"sampled={n}  gold pmids total={len(gold)}  found in papers_index: {found[0][0]}/{len(gold)}")
    return rows


def physician_questions():
    with open(env("AGENT_EVAL_QUESTIONS")) as fh:
        sample = json.load(fh)["questions"]
    rows = []
    for q in sorted(sample, key=lambda q: q["rank"]):
        if not q["comparator_answer"]:
            continue
        if not q["prompt"].endswith(OE_SUFFIX):
            raise SystemExit(f"rank {q['rank']}: prompt does not end with the appended sentence")
        rows.append({"qid": q["question_id"], "rank": q["rank"], "specialty": q["specialty"],
                     "text": q["prompt"][:-len(OE_SUFFIX)]})
    print(f"{len(rows)} questions have a comparator response (ranks {[r['rank'] for r in rows]})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=200, help="pmc_patients: sample size")
    ap.add_argument("--seed", type=int, default=42, help="pmc_patients: sampling seed")
    ap.add_argument("--limit", type=int, help="keep only the first N questions")
    a = ap.parse_args()
    with open(a.config) as fh:
        benchmark = yaml.safe_load(fh)["benchmark"]
    out = f"{a.run_dir}/sample.jsonl"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; a run keeps one sample")
    rows = pmc_patients(a.n, a.seed) if benchmark == "pmc_patients" else physician_questions()
    rows = rows[:a.limit] if a.limit else rows
    os.makedirs(a.run_dir, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
