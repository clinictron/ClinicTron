#!/usr/bin/env python
"""The acceptance rule of 02b_pubmed_abstracts.py on hand-made records (no network).
Run from multi_turn_evals_v2/ after `. env.example.sh`:  python tests/test_pubmed_abstracts.py"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("step", os.path.join(HERE, "02b_pubmed_abstracts.py"))
step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step)

rec = {"title": "Teriparatide or alendronate in glucocorticoid-induced osteoporosis.", "year": "2007", "first_author": "Saag"}
cited = "Teriparatide or Alendronate in Glucocorticoid-Induced Osteoporosis"
assert step.accepted(cited, 2007, "Saag KG", rec)
assert step.accepted(cited, "2008", "", rec)                      # year within one, no author named
assert not step.accepted(cited, 2010, "Saag KG", rec)             # year
assert not step.accepted(cited, 2007, "Buckley L", rec)           # first author
assert not step.accepted(cited + " in men", 2007, "Saag KG", rec)  # title more than 3 edits away
assert not step.accepted(cited, "", "Saag KG", rec)               # no cited year
assert step.has_abstract("x" * 50) and not step.has_abstract("No abstract") and not step.has_abstract(None)
print("ok")
