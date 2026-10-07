#!/usr/bin/env python
"""Tests only: stands in for the dense search without a GPU. Same files in and out; the "dense" hits are
the corpus's keyword-search hits.  usage: stub_dense_search.py IN.json OUT.json --topk K"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import psql_rows  # noqa: E402

inp, outp, topk = sys.argv[1], sys.argv[2], int(sys.argv[4])
with open(inp) as fh:
    rows = json.load(fh)
out = {}
for r in rows:
    q = "$kw$" + r["text"].replace("$", " ") + "$kw$"
    hits = psql_rows(f"SELECT work_id FROM papers_index WHERE work_id @@@ paradedb.boolean(should => ARRAY["
                     f"paradedb.match('title', {q}), paradedb.match('abstract', {q})]) "
                     f"ORDER BY paradedb.score(work_id) DESC LIMIT {topk}")
    out[r["sid"]] = [{"work_id": h[0], "score": 1.0 - i / 100} for i, h in enumerate(hits)]
with open(outp, "w") as fh:
    json.dump(out, fh)
print(f"stub scan: {len(rows)} queries")
