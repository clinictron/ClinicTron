#!/usr/bin/env python
"""Offline test of the zero_shot arm (no search tool): a scripted model lists four papers for each of the two
real physician questions and one PMC case: three real titles (with punctuation and spelling differences), one invented title and one repeat. Checks the resolution, the hallucination count, citations.json, that the judge (scripted verdict,
--after-round 0) and the scorer read the arm, and that a rerun is a no-op. Real corpus lookups, no GPU.
Run from multi_turn_evals_v2/ after `. env.example.sh`:  python tests/test_zero_shot.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
os.environ.setdefault("MULTI_TURN_CLAUDE_CWD", "/tmp/multi_turn_evals_claude_cwd")
import llm  # noqa: E402

spec = importlib.util.spec_from_file_location("driver", os.path.join(HERE, "02_run_agent_search.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

LISTED = [
    "Teriparatide or alendronate in glucocorticoid induced osteoporosis",   # punctuation differs from the corpus title
    "How I treat LGL leukaemia",                                            # spelling variant
    "Effects of teriparatide versus alendronate for treating glucocorticoid-induced osteoporosis: thirty-six-month results of a randomized, double-blind, controlled trial",
    "Teriparatide cures glucocorticoid osteoporosis in every patient: a fictional trial",   # hallucinated
    "How I treat LGL leukaemia",                                            # a repeat: dropped before resolution
]
calls = []


def scripted(model, messages, key):
    calls.append(messages)
    if "list_a" in messages[-1]["content"].lower() or "LIST A" in messages[-1]["content"]:
        return json.dumps({"better": "A", "reason": "scripted"}), {"model": model}, None, 0.0
    assert len(messages) == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user"
    assert "You have no search tool." in messages[0]["content"] and "{keep}" not in messages[0]["content"]
    assert messages[1]["content"].startswith(("QUESTION:\n", "PATIENT CASE:\n"))
    return json.dumps({"citations": LISTED}), {"model": model}, None, 0.01


llm.call_openrouter = scripted
tmp = tempfile.mkdtemp(prefix="zero_shot_test_")

for benchmark in ("physician_questions", "pmc_patients"):
    with open(f"{HERE}/configs/{benchmark}.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(keep=3, agents={"scripted": {"backend": "openrouter", "model": "scripted", "concurrent_calls": 2}})
    cfg_path = f"{tmp}/{benchmark}.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    run_dir = f"{tmp}/{benchmark}"
    os.makedirs(run_dir)
    subprocess.run([sys.executable, "01_sample_questions.py", "--config", cfg_path, "--run-dir", run_dir], check=True, capture_output=True) \
        if os.path.exists("01_sample_questions.py") else None
    if not os.path.exists(f"{run_dir}/sample.jsonl"):
        raise SystemExit("01_sample_questions.py did not write sample.jsonl")
    with open(f"{run_dir}/sample.jsonl") as fh:
        rows = [json.loads(l) for l in fh][:2]
    with open(f"{run_dir}/sample.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)

    n0 = len(calls)
    sys.argv = ["02_run_agent_search.py", "--run-dir", run_dir, "--config", cfg_path, "--agent", "scripted", "--arm", "zero_shot"]
    driver.main()
    assert len(calls) - n0 == len(rows), "one model call per question"
    out = json.load(open(f"{run_dir}/arms/zero_shot_scripted/citations.json"))
    assert len(out) == len(rows)
    for q, v in out.items():
        found = [bool(i["work_id"]) for i in v["listed"]]
        assert found == [True, True, True, False], v["listed"]
        assert v["n_listed"] == 4 and v["n_hallucinated"] == 1 and v["rounds"] == 0 and v["problems"] == ["1 listed items were empty, not text, or repeats", "4 papers listed, more than 3"]
        ids = [c["id"] for c in v["citations"]]
        assert ids == [c["id"] for c in v["snapshots"]["0"]] and len(ids) == 3 and ids[0] == "W2042007405", ids
        assert all(c["pmid"] for c in v["citations"]), "resolved records should carry PMIDs"
    # rerun: nothing to do, same output
    driver.main()
    assert len(calls) - n0 == len(rows), "rerun made model calls"
    assert json.load(open(f"{run_dir}/arms/zero_shot_scripted/citations.json")) == out

    if benchmark == "physician_questions":
        r = subprocess.run([sys.executable, "03_judge.py", "--run-dir", f"{run_dir}/arms/zero_shot_scripted", "--after-round", "0",
                            "--config", f"{tmp}/judge.yaml"] if os.path.exists(f"{tmp}/judge.yaml") else ["true"], capture_output=True, text=True)
    else:
        r = subprocess.run([sys.executable, "03_score_rankings.py", "--run-dir", run_dir], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "scripted/zero_shot/hop0" in r.stdout, r.stdout
    print(f"{benchmark}: OK ({len(rows)} questions, 4 titles each after the repeat, 3 in the corpus, 1 hallucinated)")

print(f"ALL GOOD: {tmp}")
