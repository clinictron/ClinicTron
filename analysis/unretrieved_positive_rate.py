#!/usr/bin/env python3
"""Unretrieved-positive rate: the share of relevant articles that the frozen base encoder
does not return in its top 50 for the written question.

  positive    a graded pool article with grade >= --grade (default 7); the invented sketch
              documents (ids starting with "synth:") are never positives
  unretrieved a positive absent from the first --depth (default 50) entries of the base
              encoder's ranking for the written question (rankings.jsonl, slot "question")
  rate        sum(unretrieved) / sum(positives) over all queries (micro, document-weighted)

A query with grades but no question ranking has every positive counted as unretrieved;
the number of such queries is printed so it is never silent.

Per-class rates use the same micro formula inside each class:
  type       question class or difficulty group (field `type_or_group`), split by `source`
  family     parent class of a subclass question (first dotted part of `type_or_group`)

Inputs (formats written by generation stages 03, 05 and 07):
  --scores     scores.jsonl   {"id": qid, "scores": {doc_id: grade}}
  --rankings   rankings.jsonl {"qid", "slot", "fused": [[doc_id, score], ...]}
  --questions  questions.jsonl or a stage-07 queries.jsonl; each row has "qid" and the
               class fields in "cell" (or, for stage-07 rows, "provenance"). Only queries listed here are counted.
"""
from __future__ import annotations

import argparse
import collections
import json


def jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_questions(paths):
    out = {}
    for r in jsonl(paths):
        c = r.get("cell") or r.get("provenance") or {}
        sub = c.get("subclass")
        name = sub.get("type_name") if isinstance(sub, dict) else None
        out[str(r["qid"])] = (str(c.get("source") or ""), str(c.get("type_or_group") or ""), name)
    return out


def load_top(paths, depth):
    top = {}
    for o in jsonl(paths):
        sid = str(o.get("id") or o.get("qid"))
        slot = o.get("slot")
        if (slot is not None and slot != "question") or (slot is None and "#" in sid):
            continue                                   # sketch slots are not the question
        top[str(o.get("qid") or sid)] = {str(d) for d, _ in (o.get("fused") or [])[:depth]}
    return top


def load_grades(paths):
    g = {}
    for o in jsonl(paths):
        g.setdefault(str(o["id"]), {}).update({str(k): int(v) for k, v in o["scores"].items()})
    return g


def rate(pos, miss):
    return round(100 * miss / pos, 1) if pos else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--rankings", nargs="+", required=True)
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--grade", type=int, default=7, help="minimum grade of a positive")
    ap.add_argument("--depth", type=int, default=50, help="rank cutoff for 'retrieved'")
    ap.add_argument("--by", choices=["none", "type", "family"], default="none")
    ap.add_argument("--tsv", action="store_true", help="per-class table as TSV, not JSON")
    a = ap.parse_args(argv)

    qs = load_questions(a.questions)
    top = load_top(a.rankings, a.depth)
    grades = load_grades(a.scores)

    tot = [0, 0, 0]                                    # queries, positives, unretrieved
    groups = collections.defaultdict(lambda: [0, 0, 0])
    names = {}
    no_ranking = 0
    for qid, sc in grades.items():
        if qid not in qs:
            continue
        retrieved = top.get(qid)
        no_ranking += retrieved is None
        retrieved = retrieved or set()
        pos = [w for w, g in sc.items() if g >= a.grade and not w.startswith("synth:")]
        miss = sum(1 for w in pos if w not in retrieved)
        tot[0] += 1
        tot[1] += len(pos)
        tot[2] += miss
        if a.by != "none" and pos:                     # per class: queries with a positive
            acc = groups[key(qs[qid], a.by)]
            acc[0] += 1
            acc[1] += len(pos)
            acc[2] += miss
        if qs[qid][2]:
            names[qs[qid][1].split(".")[0]] = qs[qid][2]

    overall = {"queries": tot[0], "positives": tot[1], "unretrieved": tot[2],
               "rate_pct": rate(tot[1], tot[2]), "grade_threshold": a.grade,
               "depth": a.depth, "queries_without_ranking": no_ranking}
    rows = sorted(({"source": k[0], "class": k[1], "name": names.get(k[1], k[1]),
                    "queries": v[0], "positives": v[1], "unretrieved": v[2],
                    "rate_pct": rate(v[1], v[2])} for k, v in groups.items() if k),
                  key=lambda r: (r["source"], -(r["rate_pct"] or 0)))
    if a.tsv:
        print("source\tclass\tname\tqueries\tpositives\tunretrieved\trate_pct")
        print(f"all\tall\tall\t{tot[0]}\t{tot[1]}\t{tot[2]}\t{overall['rate_pct']}")
        for r in rows:
            print("\t".join(str(r[k]) for k in ("source", "class", "name", "queries",
                                                "positives", "unretrieved", "rate_pct")))
    else:
        print(json.dumps({"overall": overall, "by_class": rows}, indent=1))


def key(q, by):
    source, tog, _ = q
    if by == "type":
        return (source, tog) if source in ("ely_types", "failure_groups") else None
    return (source, tog.split(".")[0]) if source == "subclasses" else None


if __name__ == "__main__":
    main()
