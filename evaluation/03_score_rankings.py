#!/usr/bin/env python
"""Score a PMC-Patients run: nDCG@10 for every arm, agent and hop.

nDCG@10 is pytrec_eval `ndcg_cut_10` (the implementation BEIR and PMC-Patients use), against the
official ReCDS-PAR test qrels of the sampled queries. The list scored at hop k is the ranking the
agent gave after that hop (at most `keep` sources).

A listed source that has no PMID (a corpus row without one) keeps its rank slot as a never-relevant
placeholder. A query the agent has not finished is left out of its cell, and every cell reports its
n; a query with an empty list scores 0.

Paired tests are over the queries both cells finished: mean difference, paired t, and the share of
10,000 bootstrap resamples whose mean difference is above zero.

  usage: 03_score_rankings.py --run-dir RUN [--limit N]
  Scores every arms/{arm}_{agent}/citations.json in the run folder. Writes RUN/scores.json (cells and
  paired tests), RUN/scores_per_query.json and RUN/scores_table.tex (nDCG@10 x 100 per arm, agent and hop).
"""
import argparse
import glob
import json
import math
import os
import random

import pytrec_eval

from common import load_sample

NAMES = {"pubmed": "PubMed Best Match", "bm25": "BM25", "clinictron_bge": "ClinicTron-BGE", "reasonembed": "BGE-Reasoner-Embed",
         "nvembed_v2": "NV-Embed-v2", "bmretriever_2b": "BMRetriever-2B", "medcpt": "MedCPT",
         "openai_3_small": "OpenAI text-embedding-3-small"}


def ranking(sources):
    """A hop's sources, best first -> PMIDs (a source without one keeps its slot as a placeholder)."""
    return list(dict.fromkeys(s["pmid"] or f"UNMAPPED_{s['id']}" for s in sources))


def score_cell(qrels, rankings):
    """rankings: {qid: [pmid best first]} -> (summary, {qid: ndcg@10})."""
    if not rankings:
        return None, {}
    results = {q: {p: float(len(r) - i) for i, p in enumerate(r)} for q, r in rankings.items()}
    scored = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in rankings}, {"ndcg_cut_10"}).evaluate(results)
    ndcg = {q: scored.get(q, {}).get("ndcg_cut_10", 0.0) for q in rankings}
    return {"n": len(rankings), "ndcg@10": round(sum(ndcg.values()) / len(rankings), 4)}, ndcg


def paired(a, b):
    """Per-query scores of two cells -> b minus a over their common queries."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = [b[q] - a[q] for q in common]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    rng = random.Random(0)
    wins = sum(1 for _ in range(10000) if sum(rng.choice(d) for _ in range(n)) > 0)
    return {"n": n, "delta": round(mean, 4), "t": round(mean / (sd / math.sqrt(n)), 2) if sd else None,
            "bootstrap_p_gt_0": wins / 10000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, help="first N sample queries")
    a = ap.parse_args()
    sample = load_sample(a.run_dir, a.limit)
    qids = [r["qid"] for r in sample]
    qrels = {r["qid"]: {p: 1 for p in r["gold_pmids"]} for r in sample}

    cells, per_query = {}, {}
    for path in sorted(glob.glob(f"{a.run_dir}/arms/*/citations.json")):
        arm, agent = os.path.basename(os.path.dirname(path)).rsplit("_", 1)
        with open(path) as fh:
            out = json.load(fh)
        for hop in sorted({h for v in out.values() for h in v["snapshots"]}, key=lambda h: (not h.isdigit(), h.zfill(3))):
            name = f"{agent}/{arm}/hop{hop}"
            summary, ndcg = score_cell(qrels, {q: ranking(out[q]["snapshots"][hop]) for q in qids
                                               if q in out and hop in out[q]["snapshots"]})
            if summary:
                cells[name], per_query[name] = summary, ndcg

    tests = {}
    for name in per_query:  # does each dense arm beat PubMed for the same agent and hop
        agent, arm, hop = name.split("/")
        base = f"{agent}/pubmed/{hop}"
        if arm != "pubmed" and base in per_query:
            result = paired(per_query[base], per_query[name])
            if result:
                tests[f"{agent}/{hop}: {arm} - pubmed"] = result

    for name, c in cells.items():
        if c["n"] != len(qids):
            print(f"WARNING: {name} covers {c['n']} of {len(qids)} queries")
        print(f"{name:36s} n={c['n']:3d}  nDCG@10={c['ndcg@10']:.4f}")
    for label, t in tests.items():
        print(f"{label:44s} n={t['n']:3d}  delta={t['delta']:+.4f}  t={t['t']}  P(>0)={t['bootstrap_p_gt_0']:.4f}")
    with open(f"{a.run_dir}/scores.json", "w") as fh:
        json.dump({"n_queries": len(qids), "cells": cells, "paired": tests}, fh, indent=2)
    with open(f"{a.run_dir}/scores_per_query.json", "w") as fh:
        json.dump(per_query, fh)

    # The paper table: one row per arm, one column per agent and hop, nDCG@10 x 100 to one decimal.
    cols = sorted({tuple(n.split("/")[::2]) for n in cells}, key=lambda c: (c[0], not c[1][3:].isdigit(), c[1][3:].zfill(3)))
    arms = sorted({n.split("/")[1] for n in cells}, key=lambda x: (x != "pubmed", x))
    name = lambda arm: ("Mega Search: " + NAMES.get(arm[5:], arm[5:])) if arm.startswith("mega_") else NAMES.get(arm, arm)
    rows = [[name(arm)] + [f"{100 * cells[f'{ag}/{arm}/{hop}']['ndcg@10']:.1f}" if f"{ag}/{arm}/{hop}" in cells else "--"
                                      for ag, hop in cols] for arm in arms]
    head = ["System / Setting"] + [f"{ag.capitalize()} {hop.replace('hopmega', 'mega').replace('hop', 'r')}" for ag, hop in cols]
    print("\n" + "\n".join("  ".join(f"{v:>10s}" if i else f"{v:30s}" for i, v in enumerate(r)) for r in [head] + rows))
    with open(f"{a.run_dir}/scores_table.tex", "w") as fh:
        fh.write(" & ".join(["System / Setting"] + [f"{ag.capitalize()} " + (f"$r_{hop[3:]}$" if hop[3:].isdigit() else hop[3:]) for ag, hop in cols]) + " \\\\\n")
        fh.write("".join(" & ".join(r) + " \\\\\n" for r in rows))
    print(f"-> {a.run_dir}/scores.json, scores_table.tex")


if __name__ == "__main__":
    main()
