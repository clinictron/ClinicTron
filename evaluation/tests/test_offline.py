#!/usr/bin/env python
"""Offline end-to-end test of version 2: a scripted model, the dense search replaced by a keyword stub (no
GPU), real PubMed, real openFDA, the real question files. Runs both benchmarks:

  physician_questions  2 questions, arms pubmed + clinictron_bge (stub), 2 hops of 5 results, keep 3; then the judge
                with a scripted verdict after hop 1 and hop 2
  pmc_patients  2 cases, arm clinictron_bge (stub); then 03_score_rankings.py

Checks the conversation shape (one query per hop, a correction when the reply is not JSON, FDA labels only on
the physician-questions benchmark, the rank fork never in the conversation, no source shown twice), the identifiers,
citations.json, the judge's inputs (every comparator citation shown, blind) and that a rerun is a no-op.
Run from multi_turn_evals_v2/ after `. env.example.sh`:  python tests/test_offline.py
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
os.environ.setdefault("MULTI_TURN_CLAUDE_CWD", "/tmp/multi_turn_evals_claude_cwd")

import llm  # noqa: E402

RANK_TAIL = '{"citations": ["...", "..."]}'
IDENT = re.compile(r"^(?:FDA LABEL: )?\[([^\]]+)\]", re.M)


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = load("02_run_agent_search.py")
driver.Run.dense_command = lambda self: [sys.executable, os.path.join(HERE, "tests/stub_dense_search.py")]
calls = []


def scripted(model, messages, key):
    """The agent: hop-1 reply first not JSON, then a query plus one FDA label; hop-2 reply a query; a rank
    reply lists the first three identifiers it was shown."""
    last, n = messages[-1]["content"], sum(m["role"] == "assistant" for m in messages)
    if last.rstrip().endswith(RANK_TAIL):
        shown = IDENT.findall("\n".join(m["content"] for m in messages if m["role"] == "user"))
        calls.append(("rank", n))
        return json.dumps({"citations": shown[:3]}), {"model": model}, None, 0.01
    calls.append(("query", n))
    if n == 0:
        return "Sure! My query: teriparatide osteoporosis", {"model": model}, None, 0.01
    if n == 1:
        return json.dumps({"query": "teriparatide glucocorticoid-induced osteoporosis trial",
                           "fda_label": [{"drug": "Forteo", "section": "indications_and_usage"}]}), {"model": model}, None, 0.01
    return json.dumps({"query": "teriparatide versus alendronate bone density",
                       "fda_label": [{"drug": "Forteo", "section": "indications_and_usage"}]}), {"model": model}, None, 0.01


llm.call_openrouter = scripted


def config(benchmark, tmp):
    with open(f"{HERE}/configs/{benchmark}.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg.update(hops=2, per_hop=5, keep=3, agents={"scripted": {"backend": "openrouter", "model": "scripted", "concurrent_calls": 2}})
    path = f"{tmp}/{benchmark}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    return path, cfg


def run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def drive(run_dir, cfg_path, arm):
    sys.argv = ["02_run_agent_search.py", "--run-dir", run_dir, "--config", cfg_path, "--agent", "scripted", "--arm", arm]
    driver.main()
    with open(f"{run_dir}/state/{arm}_scripted.json") as fh:
        return json.load(fh)


def check_conversation(st, cfg, fda):
    for q, s in st.items():
        user = [m["content"] for m in s["messages"] if m["role"] == "user"]
        assert len(s["requests"]) == 2 and s["new_per_hop"] and sorted(s["kept"]) == ["1", "2"], s["kept"]
        assert user[0] == "Round 1 of 2. Write your first search query."
        assert user[1] == "Your reply was not valid JSON. Reply with JSON only.", user[1]
        assert user[2].startswith('SEARCH RESULTS for "teriparatide glucocorticoid-induced osteoporosis trial" (') \
            and user[2].endswith("Round 2 of 2. Write another search query to find relevant sources not yet shown.")
        assert user[3].startswith('SEARCH RESULTS for "teriparatide versus alendronate bone density" (') and "Round 3" not in user[3]
        assert ("FDA LABEL: [FDA:forteo]" in user[2]) == fda and ("FDA" in s["messages"][0]["content"]) == fda
        assert not any(u.rstrip().endswith(RANK_TAIL) for u in user), "the rank fork leaked into the conversation"
        shown = IDENT.findall("\n".join(user))
        papers = [i for i in shown if not i.startswith("FDA:")]  # a label may be requested again (another section)
        assert len(papers) == len(set(papers)), "a paper was shown twice"
        assert set(shown) == set(s["retrieved"]), (set(shown) ^ set(s["retrieved"]))
        for hop, kept in s["kept"].items():
            assert 0 < len(kept) <= cfg["keep"] and set(kept) <= set(s["retrieved"]), kept
        assert set(s["kept"]["1"]) <= set(IDENT.findall(user[2])), "the hop-1 ranking may only use hop-1 results"
        assert len(s["messages"]) == 8, [m["role"] for m in s["messages"]]


tmp = tempfile.mkdtemp(prefix="multi_turn_v2_test_")
print("run folder", tmp)

# ---------------------------------------------------------------- physician-questions benchmark
cmp = f"{tmp}/cmp"
cfg_path, cfg = config("physician_questions", tmp)
print(run(["01_sample_questions.py", "--run-dir", cmp, "--config", cfg_path, "--limit", "2"]).strip().splitlines()[-1])
for arm in ("pubmed", "clinictron_bge"):
    before = len(calls)
    st = drive(cmp, cfg_path, arm)
    check_conversation(st, cfg, fda=True)
    idents = [i for s in st.values() for i in s["retrieved"] if not i.startswith("FDA:")]
    if arm == "pubmed":
        assert all(i.startswith("PMID:") for i in idents) and any(s["retrieved"][i]["work_id"] for s in st.values() for i in idents if i in s["retrieved"])
        assert all(s["retrieved"][i]["journal"] and s["retrieved"][i]["year"] for s in st.values() for i in idents if i in s["retrieved"])
    else:
        assert all(i.startswith("W") for i in idents) and any(s["retrieved"][i]["pmid"] for s in st.values() for i in idents if i in s["retrieved"])
    assert all(s["retrieved"]["FDA:forteo"]["brand_name"].lower() == "forteo" for s in st.values())
    with open(f"{cmp}/arms/{arm}_scripted/citations.json") as fh:
        out = json.load(fh)
    assert len(out) == 2 and all(v["citations"] == v["snapshots"]["2"] and v["snapshots"]["1"] and v["rank"] for v in out.values())
    assert all(c["title"] for v in out.values() for c in v["citations"] if not c["id"].startswith("FDA:"))
    assert os.path.exists(f"{cmp}/arms/{arm}_scripted/manifest.json")
    n_calls = len(calls)
    assert sum(1 for _ in open(f"{cmp}/raw_responses/{arm}_scripted.jsonl")) == n_calls - before, "a model reply was not recorded"
    drive(cmp, cfg_path, arm)  # rerun: nothing to do
    assert len(calls) == n_calls, "a finished arm made model calls on rerun"
    assert json.load(open(f"{cmp}/state/{arm}_scripted.json")) == st
    print(f"  {arm}: {len(calls)} scripted calls so far, {sum(len(s['retrieved']) for s in st.values())} sources retrieved")

# the judge, scripted: always "A"
judge = load("03_judge.py")
judge.call_openrouter = lambda model, messages, key: ('{"better": "A", "reason": "test"}', {"model": model}, None, 0.001)
with open(os.environ["AGENT_EVAL_QUESTIONS"]) as fh:
    cmp_n = {q["question_id"]: len(q["comparator_citations"]) for q in json.load(fh)["questions"]}
for hop in (1, 2):
    sys.argv = ["03_judge.py", "--run-dir", f"{cmp}/arms/pubmed_scripted", "--after-round", str(hop)]
    judge.main()
    out = f"{cmp}/arms/pubmed_scripted/judge/openai_gpt-5.6-sol_hop{hop}"
    inputs = [json.loads(l) for l in open(f"{out}/inputs.jsonl")]
    summary = json.load(open(f"{out}/summary.json"))
    for row in inputs:
        assert row["n_comparator"] == cmp_n[row["qid"]], "a comparator citation was cut"
        assert 0 < row["n_ours"] <= 3
        # no system name and no agent-side identifier (abstracts may mention PubMed as a database)
        assert not re.search(r"ClinicTron|PMID:|\[W\d+\]|FDA:", row["prompt"]), "the prompt names a system"
        assert row["prompt"].count("\nLIST A:\n") == 1 and row["prompt"].count("\nLIST B:\n") == 1
    assert summary["questions_judged"] == 2 and summary["ours_better"] == summary["ours_shown_as_A"]
    assert summary["config"]["citations_per_list"] == 0
    print(f"  judge after hop {hop}: {summary['questions_judged']} judged, lists {[(r['n_ours'], r['n_comparator']) for r in inputs]}")

# ---------------------------------------------------------------- PMC-Patients benchmark
pmc = f"{tmp}/pmc"
cfg_path, cfg = config("pmc_patients", tmp)
print(run(["01_sample_questions.py", "--run-dir", pmc, "--config", cfg_path, "--n", "2"]).strip().splitlines()[-1])
st = drive(pmc, cfg_path, "clinictron_bge")
check_conversation(st, cfg, fda=False)
assert not any(i.startswith("FDA:") for s in st.values() for i in s["retrieved"]), "FDA labels on the PMC-Patients benchmark"
print(run(["03_score_rankings.py", "--run-dir", pmc]).strip().splitlines()[-4:][0])
scores = json.load(open(f"{pmc}/scores.json"))
assert sorted(scores["cells"]) == ["scripted/clinictron_bge/hop1", "scripted/clinictron_bge/hop2"]
assert all(c["n"] == 2 and 0 <= c["ndcg@10"] <= 1 for c in scores["cells"].values())
assert "recall" not in json.dumps(scores)
print(f"ALL GOOD: {len(calls)} scripted model calls; run folder {tmp}")
