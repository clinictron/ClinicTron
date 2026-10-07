#!/usr/bin/env python
"""Multi-hop retrieval benchmark, version 2: one driver for PMC-Patients ReCDS-PAR and for the
comparison against a commercial clinical search product on Real-POCQi physician questions. The config (configs/*.yaml) chooses the benchmark.

The agent reads the question (a patient case report, or a physician's point-of-care question) and
searches in hops. At every hop it writes ONE query for the arm's search tool and receives up to
`per_hop` results it has not seen before. On the physician-questions benchmark it may also request FDA
drug-label sections (openFDA) in any hop. After every hop a throwaway fork of the conversation ranks
the best `keep` sources found so far; that list is the hop's result, scored by 03_score_rankings.py
(PMC-Patients, nDCG@10) or judged by 03_judge.py (physician questions). The conversation never sees the fork.

Arms: pubmed (E-utilities Best Match), bm25 (keyword search over the corpus) and one per dense encoder
with the full corpus embedded (clinictron_bge, reasonembed, nvembed_v2, bmretriever_2b, medcpt,
openai_3_small). The dense arms share one search interface, and one corpus scan per hop serves every question.
The zero_shot arm has no search tool: the model lists the titles of `keep` papers from memory (config
`zero_shot_prompt`); a title that matches a corpus title (tools.resolve) becomes that record, any other is
counted as hallucinated, and the resolved list is the arm's one ranking, snapshot "0" (the protocol of
OpenScholar, Asai et al. 2024, with our corpus as the reference set).
Agents (config `agents`): OpenRouter models, or Claude through the headless CLI (claude -p)
with the system prompt replaced and every tool disabled (llm.py).

  usage: 02_run_agent_search.py --run-dir RUN --config CFG --agent NAME --arm ARM [--limit N]

Resumable: state/{arm}_{agent}.json is rewritten after every step and a rerun continues from it.
Every model reply is appended to raw_responses/{arm}_{agent}.jsonl.
Output: arms/{arm}_{agent}/citations.json (the ranking after every hop, with each source's record).
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

import llm
import tools
from common import HERE, env, load_sample, secret
from llm import fill, parse
from pubmed_search import PubMed, pubmed_tool

DENSE_ARMS = ("clinictron_bge", "reasonembed", "nvembed_v2", "bmretriever_2b", "medcpt", "openai_3_small")
RETRIES = 4       # attempts at one model call (errors, empty replies)
CORRECTIONS = 2   # times the driver may send a reply back because it was not JSON


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Run:
    def __init__(self, run_dir, config_path, agent, arm):
        self.dir, self.name, self.arm = run_dir, agent, arm
        with open(config_path) as fh:
            self.cfg = yaml.safe_load(fh)
        self.agent = self.cfg["agents"][agent]
        self.fda = self.cfg["benchmark"] == "physician_questions"
        with open(os.path.join(HERE, self.cfg["zero_shot_prompt" if arm == "zero_shot" else "prompt"])) as fh:
            self.prompt = fh.read()
        with open(os.path.join(HERE, self.cfg["rank_prompt"])) as fh:
            self.rank = fill(fh.read().strip(), self.cfg)
        with open(os.path.join(HERE, self.cfg["driver_messages"])) as fh:
            self.msg = yaml.safe_load(fh)
        self.key = secret("OPENROUTER_API_KEY") if self.agent["backend"] == "openrouter" else None
        self.cwd = None
        if self.agent["backend"] == "claude_cli":
            self.cwd = env("MULTI_TURN_CLAUDE_CWD")
            llm.check_claude_cwd(self.cwd)
        self.lock = threading.Lock()
        self.over_budget = False
        for sub in ("state", "raw_responses", "logs", f"arms/{arm}_{agent}"):
            os.makedirs(f"{run_dir}/{sub}", exist_ok=True)
        self.claim()
        self.write_manifest(config_path)

    def claim(self):
        """One driver per (arm, agent) and run folder: two at once would interleave writes to one state."""
        self.lockfile = open(f"{self.dir}/state/{self.arm}_{self.name}.lock", "w")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another driver is already running {self.arm}/{self.name} in {self.dir}")

    def write_manifest(self, config_path):
        """What produced this arm's results. An arm is never continued with different settings."""
        files = [config_path, f"{self.dir}/sample.jsonl"] + [os.path.join(HERE, f) for f in (
            self.cfg["zero_shot_prompt" if self.arm == "zero_shot" else "prompt"], self.cfg["rank_prompt"],
            self.cfg["driver_messages"], "02_run_agent_search.py",
            "llm.py", "tools.py", "common.py", "pubmed_search.py")] + self.dense_command()[1:2]
        manifest = {"config": self.cfg, "agent": self.name, "arm": self.arm,
                    "sha256": {os.path.basename(f): sha256(f) for f in files}}
        path = f"{self.dir}/arms/{self.arm}_{self.name}/manifest.json"
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
            if {k: old.get(k) for k in manifest} != manifest:
                raise SystemExit(f"{path} was written with different settings or files; use a new run folder")
            return
        manifest["started"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        if self.agent["backend"] == "claude_cli":
            manifest["claude_cli"] = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=1)

    def dense_command(self):
        """The search program of a dense arm: ClinicTron-BGE has its own script; every other encoder goes
        through dense_search.py, NV-Embed-v2 under the Python environment its index was written with."""
        if self.arm in ("pubmed", "bm25", "zero_shot"):
            return []
        if self.arm == "clinictron_bge":
            return [sys.executable, os.path.join(HERE, "dense_clinictron_bge_search.py")]
        python = env("AGENT_EVAL_NVEMBED_PYTHON") if self.arm == "nvembed_v2" else sys.executable
        return [python, os.path.join(HERE, "dense_search.py"), "--encoder", self.arm]

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            with open(f"{self.dir}/logs/driver_{self.arm}_{self.name}.log", "a") as fh:
                fh.write(line + "\n")

    def load_state(self):
        path = f"{self.dir}/state/{self.arm}_{self.name}.json"
        if not os.path.exists(path):
            return {}
        with open(path) as fh:
            return json.load(fh)

    def save_state(self, st):
        path = f"{self.dir}/state/{self.arm}_{self.name}.json"
        with open(path + ".tmp", "w") as fh:
            json.dump(st, fh, ensure_ascii=False)
        os.replace(path + ".tmp", path)

    def spend(self, usd):
        """OpenRouter spend, cumulative over every arm and agent of the run folder (file-locked); stops the
        run once it passes the budget. The Claude CLI runs on the subscription and reports an estimate only."""
        if self.agent["backend"] != "openrouter" or not usd:
            return
        with open(f"{self.dir}/state/spend.json", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            raw = fh.read()
            total = (json.loads(raw)["usd"] if raw else 0.0) + usd
            fh.seek(0)
            fh.truncate()
            json.dump({"usd": total}, fh)
        if self.cfg["budget_usd"] and total > self.cfg["budget_usd"]:
            self.over_budget = True

    def stop_if_over_budget(self):
        """Called after every state save: the wave that crossed the budget is kept, the next one is not started."""
        if self.over_budget:
            self.log(f"BUDGET STOP: OpenRouter spend passed ${self.cfg['budget_usd']}; state is saved, "
                     f"rerun with a higher budget_usd to continue")
            sys.exit(3)


# ---------------------------------------------------------------- model calls
def ask(run, qid, tag, messages, sid):
    """One exchange: send the last message of `messages`, append the reply -> (reply JSON, session id, cost).
    A reply that is not JSON is sent back up to CORRECTIONS times; the corrections stay in `messages`."""
    cost = 0.0
    for correction in range(CORRECTIONS + 1):
        for attempt in range(RETRIES):
            if run.over_budget:
                raise RuntimeError("over budget")
            try:
                if run.agent["backend"] == "claude_cli":
                    text, data, new_sid, c = llm.call_claude(run.agent["model"], messages, sid, run.cwd)
                else:
                    text, data, new_sid, c = llm.call_openrouter(run.agent["model"], messages, run.key)
            except Exception as e:
                run.log(f"  call error {qid}/{tag} try{attempt}: {type(e).__name__}: {str(e)[:200]}")
                time.sleep(10 * (attempt + 1))
                continue
            if text.strip():
                break
            run.log(f"  empty reply {qid}/{tag} try{attempt}")
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"no reply after {RETRIES} tries: {tag}")
        run.spend(c)
        cost = c if run.agent["backend"] == "claude_cli" else cost + c  # the CLI reports its session's running total
        sid = new_sid or sid
        with run.lock, open(f"{run.dir}/raw_responses/{run.arm}_{run.name}.jsonl", "a") as fh:
            fh.write(json.dumps({"qid": qid, "tag": tag, "t": time.time(), "resp": data}) + "\n")
        messages.append({"role": "assistant", "content": text})
        req = parse(text)
        if req is not None:
            return req, sid, cost
        messages.append({"role": "user", "content": run.msg["not_json"]})
        tag += "_fix"
    raise RuntimeError(f"reply still not JSON after {CORRECTIONS} corrections: {tag}")


def wave(run, tag, jobs):
    """jobs: {qid: (messages, sid)} -> {qid: (reply JSON, sid, cost)}, in parallel. A question whose call
    fails is logged and left out; its messages are restored, and a rerun picks it up."""
    out = {}

    def one(qid):
        messages, sid = jobs[qid]
        n = len(messages)
        try:
            out[qid] = ask(run, qid, tag, messages, sid)
        except RuntimeError as e:
            del messages[n:]
            run.log(f"  SKIP {qid}: {e}")

    with ThreadPoolExecutor(run.agent["concurrent_calls"]) as ex:
        list(ex.map(one, jobs))
    run.log(f"  {tag}: {len(out)}/{len(jobs)} replies")
    return out


# ---------------------------------------------------------------- the search tools
def dense_tool(run, hop, queries):
    """{qid: query} -> {qid: [paper]} from the arm's index: one corpus scan for all queries of the hop."""
    base = f"{run.dir}/state/scan_{run.arm}_{run.name}_h{hop}"
    asked = [{"sid": q, "text": t} for q, t in queries.items()]
    scan = None
    if os.path.exists(f"{base}_out.json"):  # a crash after the scan: reuse it if it answered these queries
        with open(f"{base}_in.json") as fh:
            same = json.load(fh) == asked
        if same:
            run.log(f"  reusing completed scan for hop {hop}")
            with open(f"{base}_out.json") as fh:
                scan = json.load(fh)
    if scan is None:
        with open(f"{base}_in.json", "w") as fh:
            json.dump(asked, fh)
        run.log(f"  dense scan: {len(asked)} queries, one pass (log: logs/scan_{run.arm}_{run.name}_h{hop}.log)")
        with open(f"{run.dir}/logs/scan_{run.arm}_{run.name}_h{hop}.log", "a") as logfh:
            r = subprocess.run(run.dense_command() + [f"{base}_in.json", f"{base}_out.json",
                                                      "--topk", str(run.cfg["per_hop"])],
                               cwd=HERE, stdout=logfh, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not os.path.exists(f"{base}_out.json"):
            raise SystemExit(f"dense scan failed; see logs/scan_{run.arm}_{run.name}_h{hop}.log")
        with open(f"{base}_out.json") as fh:
            scan = json.load(fh)
    meta = tools.papers(h["work_id"] for hits in scan.values() for h in hits)
    absent = sum(1 for hits in scan.values() for h in hits if h["work_id"] not in meta)
    if absent:
        run.log(f"  {absent} retrieved work_ids have no papers_index row")
    return {q: [{"ident": h["work_id"], **meta[h["work_id"]]} for h in scan[q] if h["work_id"] in meta] for q in queries}


def pubmed_arm(run, pm, queries):
    """{qid: query} -> {qid: [paper]} from PubMed Best Match. PubMed gives title, abstract, journal and
    year; document type and citation count come from the corpus record when the paper is in our corpus."""
    hits = pubmed_tool(pm, queries, run.cfg["per_hop"])
    corpus = tools.papers_by_pmid(d["pmid"] for docs in hits.values() for d in docs)
    out = {}
    for q, docs in hits.items():
        out[q] = []
        for d in docs:
            c = corpus.get(d["pmid"], {})
            out[q].append({"ident": "PMID:" + d["pmid"], "work_id": c.get("work_id"), "pmid": d["pmid"],
                           "type": c.get("type", "article"), "cited_by": c.get("cited_by", "n/a"),
                           "title": d["title"], "abstract": d["abstract"] or "No abstract",
                           "journal": d["journal"], "year": d["year"]})
    return out


def bm25_arm(run, queries):
    """{qid: query} -> {qid: [paper]} by BM25 over title and abstract of the corpus."""
    return {q: [{"ident": p["work_id"], **p} for p in tools.keyword_search(t, run.cfg["per_hop"])] for q, t in queries.items()}


def results_block(run, q, query, docs):
    """The results message for one search: the papers not shown to this question before; each is recorded
    under its identifier so the ranking can refer to it."""
    if query is None:
        return run.msg["no_query"], 0
    new = []
    for d in docs:
        if d["ident"] in q["retrieved"]:
            continue
        q["retrieved"][d["ident"]] = {k: d[k] for k in ("work_id", "pmid", "type", "title", "journal", "year", "cited_by")}
        new.append(d)
    lines = [fill(run.msg["results_header"], {"query": query, "n_new": len(new)})]
    lines += [fill(run.msg["paper"], d) for d in new] or [run.msg["no_new_papers"]]
    if len(docs) - len(new):
        lines.append(fill(run.msg["duplicates"], {"n": len(docs) - len(new)}))
    return "\n".join(lines), len(new)


def fda_blocks(run, q, req):
    """One block per FDA label request of a reply (any number; a malformed item is skipped)."""
    blocks = []
    for item in req.get("fda_label") if isinstance(req.get("fda_label"), list) else []:
        drug = item.get("drug") if isinstance(item, dict) else None
        section = item.get("section") if isinstance(item, dict) else None
        if not isinstance(drug, str) or not drug.strip():
            continue
        section = section if isinstance(section, str) else None
        label = tools.fda_label(drug, section)
        if label is None:
            blocks.append(fill(run.msg["fda_not_found"], {"drug": drug}))
            continue
        identifier = "FDA:" + label["brand_name"].lower()  # one identifier per label, whatever name was asked
        values = {**label, "identifier": identifier, "section": section, "sections": ", ".join(label["sections"])}
        if "text" not in label:
            blocks.append(fill(run.msg["fda_missing_section"], {k: values[k] for k in (
                "identifier", "brand_name", "generic_name", "section", "sections")}))
            continue
        q["retrieved"][identifier] = {k: label[k] for k in ("brand_name", "generic_name", "manufacturer",
                                                             "application", "label_id", "updated")}
        blocks.append(fill(run.msg["fda_label"], {k: values[k] for k in (
            "identifier", "brand_name", "generic_name", "manufacturer", "updated", "section", "text")}))
    return blocks


def select(run, q, req, what):
    """A ranking as the agent gave it, cleaned: only identifiers shown to this question, no repeats, at most
    `keep`. Problems are logged on the question."""
    asked = req.get("citations")
    asked = [c.strip("[] ") for c in asked if isinstance(c, str)] if isinstance(asked, list) else []
    asked = [("FDA:" + c[4:].strip().lower()) if c.lower().startswith("fda:") else
             re.sub(r"^(?:PMID)?[:\s]*(\d+)$", r"PMID:\1", c, flags=re.I) for c in asked]  # "PMID 123", "123" -> "PMID:123"
    kept = list(dict.fromkeys(c for c in asked if c in q["retrieved"]))
    if len(kept) != len(asked):
        q["problems"].append(f"{what}: dropped {[c for c in asked if c not in q['retrieved']]} (never shown) or repeats")
    if not kept:
        q["problems"].append(f"{what}: no usable identifier")
    if len(kept) > run.cfg["keep"]:
        q["problems"].append(f"{what}: {len(kept)} sources, cut to {run.cfg['keep']}")
    return kept[:run.cfg["keep"]]


# ---------------------------------------------------------------- the hops
def run_arm(run, sample):
    st = run.load_state()
    for r in sample:
        st.setdefault(r["qid"], {
            "messages": [{"role": "system", "content": fill(run.prompt, {**run.cfg, "text": r["text"], "tool": run.msg["tool"][
                run.arm if run.arm in ("pubmed", "bm25") else "dense"]})},
                         {"role": "user", "content": fill(run.msg["first_query"], run.cfg)}],
            "sid": None, "requests": [], "new_per_hop": [], "retrieved": {}, "kept": {}, "problems": [],
            "cost_usd": 0.0, "fork_cost_usd": 0.0})
    qids = [r["qid"] for r in sample]
    hops = run.cfg["hops"]
    pm = PubMed(run.log) if run.arm == "pubmed" else None

    for hop in range(1, hops + 1):
        # 1. The agent writes this hop's query. A question waits here until its previous hop is ranked, so
        #    the fork below always forks the conversation as it stood at that hop.
        need = [q for q in qids if len(st[q]["requests"]) == hop - 1 and (hop == 1 or str(hop - 1) in st[q]["kept"])]
        if need:
            got = wave(run, f"query_h{hop}", {q: (st[q]["messages"], st[q]["sid"]) for q in need})
            for q, (req, sid, cost) in got.items():
                st[q]["requests"].append(req)
                st[q]["sid"], st[q]["cost_usd"] = sid, cost
            run.save_state(st)
            run.stop_if_over_budget()

        # 2. The tool answers; only sources not shown in earlier hops are added to the conversation.
        pending = {q: st[q]["requests"][hop - 1] for q in qids
                   if len(st[q]["requests"]) == hop and len(st[q]["new_per_hop"]) == hop - 1}
        if pending:
            queries = {q: req["query"].strip() for q, req in pending.items()
                       if isinstance(req.get("query"), str) and req["query"].strip()}
            for q in set(pending) - set(queries):
                st[q]["problems"].append(f"hop {hop}: reply had no search query")
            run.log(f"hop {hop}: {len(queries)} searches" + (f", {len(pending) - len(queries)} replies without a query" if len(pending) > len(queries) else ""))
            results = (pubmed_arm(run, pm, queries) if run.arm == "pubmed" else bm25_arm(run, queries) if run.arm == "bm25"
                       else dense_tool(run, hop, queries))
            for q, req in pending.items():
                block, n_new = results_block(run, st[q], queries.get(q), results.get(q, []))
                blocks = [block] + (fda_blocks(run, st[q], req) if run.fda else [])
                st[q]["messages"].append({"role": "user", "content": "\n\n".join(blocks) + (
                    "\n\n" + fill(run.msg["next_query"], {**run.cfg, "round": hop + 1}) if hop < hops else "")})
                st[q]["new_per_hop"].append(n_new)
            run.save_state(st)
            run.log(f"hop {hop}: {sum(1 for q in pending if not st[q]['new_per_hop'][-1])}/{len(pending)} searches got no new results")

        # 3. A fork ranks the best `keep` of everything shown through this hop. Its messages are the
        #    conversation up to this hop's results, with the rank instruction in place of the next-query request.
        jobs = {}
        for q in qids:
            if len(st[q]["new_per_hop"]) < hop or str(hop) in st[q]["kept"]:
                continue
            if not st[q]["retrieved"]:  # nothing retrieved yet: nothing to rank
                st[q]["kept"][str(hop)] = []
                continue
            msgs = list(st[q]["messages"])
            last = msgs[-1]["content"].removesuffix("\n\n" + fill(run.msg["next_query"], {**run.cfg, "round": hop + 1}))
            jobs[q] = (msgs[:-1] + [{"role": "user", "content": last + "\n\n" + run.rank}], st[q]["sid"])
        if jobs:
            got = wave(run, f"rank_h{hop}", jobs)
            for q, (req, _sid, cost) in got.items():
                st[q]["kept"][str(hop)] = select(run, st[q], req, f"ranking after hop {hop}")
                st[q]["fork_cost_usd"] += max(0.0, cost - st[q]["cost_usd"]) if run.agent["backend"] == "claude_cli" else cost
            run.save_state(st)
            run.stop_if_over_budget()

    done = [q for q in qids if str(hops) in st[q]["kept"]]
    run.log(f"finished: {len(done)}/{len(qids)} questions have all {hops} rankings")
    out = {q: {**{k: r[k] for k in r if k not in ("text", "gold_pmids")}, "question": r["text"], "rounds": hops,
               "citations": [{"id": c, **st[q]["retrieved"][c]} for c in st[q]["kept"][str(hops)]],
               "snapshots": {h: [{"id": c, **st[q]["retrieved"][c]} for c in cs] for h, cs in st[q]["kept"].items()},
               "problems": st[q]["problems"], "cost_usd": round(st[q]["cost_usd"] + st[q]["fork_cost_usd"], 4)}
           for r in sample for q in [r["qid"]] if q in done}
    with open(f"{run.dir}/arms/{run.arm}_{run.name}/citations.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    run.log(f"-> arms/{run.arm}_{run.name}/citations.json; model cost estimate "
            f"${sum(v['cost_usd'] for v in out.values()):.2f}")


# ---------------------------------------------------------------- the arm without a search tool
RECORD = ("work_id", "pmid", "type", "title", "journal", "year", "cited_by")


def run_zero_shot(run, sample):
    """One call per question: the model lists the titles of `keep` papers from memory. Each title is resolved
    to a corpus record (tools.resolve) or counted as hallucinated; the resolved papers, in the order listed,
    are the ranking (snapshot "0")."""
    st = run.load_state()
    head, tail = run.prompt.strip().rsplit("\n\n", 1)  # the last paragraph (QUESTION: / PATIENT CASE:) is the user message
    for r in sample:
        st.setdefault(r["qid"], {
            "messages": [{"role": "system", "content": fill(head, run.cfg)}, {"role": "user", "content": fill(tail, {"text": r["text"]})}],
            "sid": None, "listed": None, "resolved": [], "retrieved": {}, "kept": {}, "problems": [], "cost_usd": 0.0})
    qids = [r["qid"] for r in sample]

    need = [q for q in qids if st[q]["listed"] is None]
    if need:
        got = wave(run, "zero_shot", {q: (st[q]["messages"], st[q]["sid"]) for q in need})
        for q, (req, sid, cost) in got.items():
            items = req.get("citations") if isinstance(req.get("citations"), list) else []
            st[q]["listed"] = list(dict.fromkeys(i.strip() for i in items if isinstance(i, str) and i.strip()))
            if len(st[q]["listed"]) != len(items):
                st[q]["problems"].append(f"{len(items) - len(st[q]['listed'])} listed items were empty, not text, or repeats")
            if len(st[q]["listed"]) > run.cfg["keep"]:
                st[q]["problems"].append(f"{len(st[q]['listed'])} papers listed, more than {run.cfg['keep']}")
            st[q]["sid"], st[q]["cost_usd"] = sid, cost
        run.save_state(st)
        run.stop_if_over_budget()

    todo = [q for q in qids if st[q]["listed"] is not None and "0" not in st[q]["kept"]]
    if todo:
        def one(q):
            kept = []
            for title in st[q]["listed"]:
                rec = tools.resolve(title)
                st[q]["resolved"].append({"title": title, "work_id": rec["work_id"] if rec else None,
                                          "corpus_title": rec["title"] if rec else None})
                if rec and rec["work_id"] not in st[q]["retrieved"]:
                    st[q]["retrieved"][rec["work_id"]] = {k: rec[k] for k in RECORD}
                    kept.append(rec["work_id"])
            st[q]["kept"]["0"] = kept[:run.cfg["keep"]]

        run.log(f"resolving {sum(len(st[q]['listed']) for q in todo)} listed papers of {len(todo)} questions against the corpus")
        with ThreadPoolExecutor(6) as ex:  # a few title searches at a time, never a storm on the database
            list(ex.map(one, todo))
        run.save_state(st)

    done = [q for q in qids if "0" in st[q]["kept"]]
    listed = sum(len(st[q]["listed"]) for q in done)
    missing = sum(1 for q in done for i in st[q]["resolved"] if not i["work_id"])
    run.log(f"finished: {len(done)}/{len(qids)} questions; {listed} titles listed, {listed - missing} in the corpus, "
            f"{missing} hallucinated ({100 * missing / max(listed, 1):.1f}%)")
    out = {q: {**{k: r[k] for k in r if k not in ("text", "gold_pmids")}, "question": r["text"], "rounds": 0,
               "citations": [{"id": c, **st[q]["retrieved"][c]} for c in st[q]["kept"]["0"]],
               "snapshots": {"0": [{"id": c, **st[q]["retrieved"][c]} for c in st[q]["kept"]["0"]]},
               "listed": st[q]["resolved"], "n_listed": len(st[q]["listed"]),
               "n_hallucinated": sum(1 for i in st[q]["resolved"] if not i["work_id"]),
               "problems": st[q]["problems"], "cost_usd": round(st[q]["cost_usd"], 4)}
           for r in sample for q in [r["qid"]] if q in done}
    with open(f"{run.dir}/arms/{run.arm}_{run.name}/citations.json", "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    run.log(f"-> arms/{run.arm}_{run.name}/citations.json; model cost estimate ${sum(v['cost_usd'] for v in out.values()):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--arm", required=True, choices=["pubmed", "bm25", "zero_shot", *DENSE_ARMS])
    ap.add_argument("--limit", type=int, help="first N sample questions")
    a = ap.parse_args()
    run = Run(a.run_dir, a.config, a.agent, a.arm)
    sample = load_sample(a.run_dir, a.limit)
    run.log(f"start: benchmark={run.cfg['benchmark']} arm={a.arm} agent={a.agent} model={run.agent['model']} "
            f"questions={len(sample)} hops={run.cfg['hops']} per_hop={run.cfg['per_hop']} keep={run.cfg['keep']}")
    (run_zero_shot if a.arm == "zero_shot" else run_arm)(run, sample)


if __name__ == "__main__":
    main()
