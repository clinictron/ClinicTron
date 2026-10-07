"""Score a bank file written by encode.py: exact search, then nDCG@10 with pytrec_eval.

    python score.py --bank bank/r2med/ReasonEmbed/r2med_biology.npz

Scores are inner products: cosine for models whose vectors are L2-normalised, dot otherwise.

Queries without test judgments are skipped. A query's `excluded_ids` (BRIGHT) and, with
--ignore-identical-ids (ArguAna, FEVER, FiQA in MTEB), a document with the query's own id are
removed from its ranking before scoring.

With --run, score a search.py run file (TSV: query_id, rank, doc_id, score) against a qrels
file (TSV or parquet: query-id, corpus-id, score) and report MRR, P@10, nDCG@10 and Recall@1000:

    python score.py --run run.tsv --qrels qrels_test.tsv
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytrec_eval
import torch


def rank(bank, excluded, ignore_identical, depth=1000, device="cpu"):
    d = torch.from_numpy(bank["doc_vectors"].astype(np.float32)).to(device)
    q = torch.from_numpy(bank["query_vectors"].astype(np.float32)).to(device)
    doc_ids, qids = [str(x) for x in bank["doc_ids"]], [str(x) for x in bank["query_ids"]]
    pos = {x: i for i, x in enumerate(doc_ids)}
    run = {}
    for s in range(0, len(qids), 256):
        sc = q[s:s + 256] @ d.T
        for j, qid in enumerate(qids[s:s + 256]):
            drop = set(excluded.get(qid, ()))
            if ignore_identical:
                drop.add(qid)
            row = sc[j]
            for x in drop:
                if x in pos:
                    row[pos[x]] = -float("inf")
            top = torch.topk(row, min(depth, len(doc_ids)))
            run[qid] = {doc_ids[i]: float(v) for v, i in zip(top.values.tolist(), top.indices.tolist())
                        if v != -float("inf")}
    return run


def load_qrels(data_dir):
    t = pq.read_table(os.path.join(data_dir, "qrels_test.parquet")).to_pydict()
    qcol = "query-id" if "query-id" in t else "query_id"
    dcol = "corpus-id" if "corpus-id" in t else "corpus_id"
    qrels = {}
    for qid, did, s in zip(t[qcol], t[dcol], t["score"]):
        qrels.setdefault(str(qid), {})[str(did)] = int(s)
    return qrels


def load_excluded(data_dir):
    q = pq.read_table(os.path.join(data_dir, "queries.parquet"))
    if "excluded_ids" not in q.column_names:
        return {}
    return {str(i): [str(x) for x in (xs or []) if x is not None and str(x) != "N/A"]
            for i, xs in zip(q.column("_id").to_pylist(), q.column("excluded_ids").to_pylist())}


def score(bank_path, data_dir, ignore_identical=False, device="cpu"):
    bank = np.load(bank_path)
    qrels = load_qrels(data_dir)
    run = rank(bank, load_excluded(data_dir), ignore_identical, device=device)
    run = {q: r for q, r in run.items() if q in qrels}
    per_q = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"}).evaluate(run)
    return {"ndcg@10": float(np.mean([v["ndcg_cut_10"] for v in per_q.values()])),
            "recall@100": float(np.mean([v["recall_100"] for v in per_q.values()])),
            "n_queries": len(per_q)}


def score_run(run_path, qrels_path):
    if qrels_path.endswith(".parquet"):
        t = pq.read_table(qrels_path).to_pydict()
        rows = zip(*(t[c] for c in ("query-id", "corpus-id", "score")))
    else:
        rows = (line.rstrip("\n").split("\t") for line in list(open(qrels_path))[1:])
    qrels = {}
    for qid, did, s in rows:
        qrels.setdefault(str(qid), {})[str(did)] = int(float(s))
    run = {}
    for line in open(run_path):
        qid, _, did, s = line.rstrip("\n").split("\t")
        run.setdefault(qid, {})[did] = float(s)
    run = {q: r for q, r in run.items() if q in qrels}
    per_q = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", "P.10", "ndcg_cut.10", "recall.1000"}).evaluate(run)
    mean = lambda m: float(np.mean([v[m] for v in per_q.values()]))
    return {"mrr": mean("recip_rank"), "p@10": mean("P_10"), "ndcg@10": mean("ndcg_cut_10"),
            "recall@1000": mean("recall_1000"), "n_queries": len(per_q)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank")
    ap.add_argument("--run", help="search.py TSV; scored against --qrels instead of a bank")
    ap.add_argument("--qrels", help="with --run: qrels TSV (header query-id, corpus-id, score) or parquet")
    ap.add_argument("--data-dir", help="task directory; default: read from the bank's own config")
    ap.add_argument("--ignore-identical-ids", action="store_true",
                    help="default: the task's `ignore_identical_ids` in the benchmark config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.run:
        print(json.dumps(score_run(a.run, a.qrels)))
        return
    cfg = json.loads(str(np.load(a.bank)["config"]))
    data_dir = a.data_dir or cfg["data_dir"]
    ignore = a.ignore_identical_ids or cfg.get("ignore_identical_ids", False)
    print(json.dumps(score(a.bank, data_dir, ignore, a.device)))


if __name__ == "__main__":
    main()
