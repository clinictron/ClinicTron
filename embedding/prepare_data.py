"""Build benchmark task folders from public Hugging Face datasets at pinned revisions.

    python prepare_data.py <benchmark> [--task T] --out data/

Writes data/<benchmark>/<task>/{corpus,queries,qrels_test}.parquet:
    corpus      _id, title, text           (title "" when the source has none)
    queries     _id, text [, excluded_ids] (excluded_ids: BRIGHT only)
    qrels_test  query-id, corpus-id, score
All downloads are anonymous. Row order follows the source files.
"""
import argparse, json, os
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

R2MED = {  # task dir -> (repo, revision); files corpus.jsonl / query.jsonl / qrels.jsonl
    "Biology":          ("R2MED/Biology",          "8b9fec2db9eda4b5742d03732213fbaee8169556"),
    "Bioinformatics":   ("R2MED/Bioinformatics",   "6021fce366892cbfd7837fa85a4128ea93315e18"),
    "Medical-Sciences": ("R2MED/Medical-Sciences", "1b48911514c80bf9182222d99752ad75e23b4b47"),
    "MedXpertQA-Exam":  ("R2MED/MedXpertQA-Exam",  "b457ea43db9ae5db74c3a3e5be0a213d0f85ac3a"),
    "MedQA-Diag":       ("R2MED/MedQA-Diag",       "78b585990279cc01a493f876c1b0cf09557fba57"),
    "PMC-Treatment":    ("R2MED/PMC-Treatment",    "53c489a44a3664ba352c07550b72b4525a5968d5"),
    "PMC-Clinical":     ("R2MED/PMC-Clinical",     "812829522f7eaa407ef82b96717be85788a50f7e"),
    "IIYi-Clinical":    ("R2MED/IIYi-Clinical",    "974abbc9bc281c3169180a6aa5d7586cfd2f5877"),
}
# BEIR biomedical: the hub's parquet conversion branch, pinned by commit; files are already
# in the target schema and are written unchanged (all queries kept; scoring skips unjudged ones).
BEIR_BIO = {  # task dir -> (repo, revision of refs/convert/parquet)
    "NFCorpus":   ("mteb/nfcorpus",   "dcf7a6d3be5c3b19802673fb4a37b2b462015f27"),
    "SciFact":    ("mteb/scifact",    "40ba5b97cb22ea15fa7e5c2e5a0fec5886baba2a"),
    "TREC-COVID": ("mteb/trec-covid", "e4017169ec19a6566ae39bb0b0e46a66f2418ca1"),
}
BEIR_FILES = {"corpus": "corpus/corpus/0000.parquet", "queries": "queries/queries/0000.parquet",
              "qrels_test": "default/test/0000.parquet"}
BRIGHT_REPO, BRIGHT_REV = "xlangai/BRIGHT", "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow",
          "sustainable_living", "leetcode", "pony", "aops", "theoremqa_questions", "theoremqa_theorems"]
MTEB_V2 = {  # task dir -> (repo, revision); configs corpus / queries / default(test qrels)
    "ArguAna":                    ("mteb/arguana",              "c22ab2a51041ffd869aaddef7af8d8215647e41a"),
    "CQADupstackGamingRetrieval": ("mteb/cqadupstack-gaming",   "4885aa143210c98657558c04aaf3dc47cfb54340"),
    "CQADupstackUnixRetrieval":   ("mteb/cqadupstack-unix",     "6c6430d3a6d36f8d2a829195bc5dc94d7e063e53"),
    "ClimateFEVERHardNegatives":  ("mteb/ClimateFEVER_test_top_250_only_w_correct-v2", "3a309e201f3c2c4b13bd4a367a8f37eee2ec1d21"),
    "FEVERHardNegatives":         ("mteb/FEVER_test_top_250_only_w_correct-v2",        "080c9ed6267b65029207906e815d44a9240bafca"),
    "FiQA2018":                   ("mteb/fiqa",                 "27a168819829fe9bcd655c2df245fb19452e8e06"),
    "HotpotQAHardNegatives":      ("mteb/HotpotQA_test_top_250_only_w_correct-v2",     "617612fa63afcb60e3b134bed8b7216a99707c37"),
    "SCIDOCS":                    ("mteb/scidocs",              "f8c2fcf00f625baaa80f62ec5bd9e1fff3b8ae88"),
    "TRECCOVID":                  ("mteb/trec-covid",           "bb9466bac8153a0349341eb1b22e06409e78ef4e"),
    "Touche2020Retrieval.v3":     ("mteb/webis-touche2020-v3",  "431886eaecc48f067a3975b70d0949ea2862463c"),
}
S = pa.string()


def get(repo, rev, path):
    return hf_hub_download(repo, path, repo_type="dataset", revision=rev, token=False)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(out, corpus, queries, qrels):
    os.makedirs(out, exist_ok=True)
    for name, tab in (("corpus", corpus), ("queries", queries), ("qrels_test", qrels)):
        pq.write_table(tab, os.path.join(out, name + ".parquet"))
    print(out, corpus.num_rows, queries.num_rows, qrels.num_rows)


def r2med(task, out):
    repo, rev = R2MED[task]
    c, q, r = (jsonl(get(repo, rev, f)) for f in ("corpus.jsonl", "query.jsonl", "qrels.jsonl"))
    # leading/trailing whitespace is stripped from document and query text
    write(out,
          pa.table({"_id": pa.array([str(x["id"]) for x in c], S), "title": pa.array([""] * len(c), S),
                    "text": pa.array([x["text"].strip() for x in c], S)}),
          pa.table({"_id": pa.array([str(x["id"]) for x in q], S), "text": pa.array([x["text"].strip() for x in q], S)}),
          pa.table({"query-id": pa.array([str(x["q_id"]) for x in r], S),
                    "corpus-id": pa.array([str(x["p_id"]) for x in r], S),
                    "score": pa.array([float(x["score"]) for x in r], pa.float64())}))


def beir_bio(task, out):
    repo, rev = BEIR_BIO[task]
    write(out, *(pq.read_table(get(repo, rev, BEIR_FILES[k])) for k in ("corpus", "queries", "qrels_test")))


def bright(task, out):
    d = pq.read_table(get(BRIGHT_REPO, BRIGHT_REV, f"documents/{task}-00000-of-00001.parquet")).to_pydict()
    e = pq.read_table(get(BRIGHT_REPO, BRIGHT_REV, f"examples/{task}-00000-of-00001.parquet")).to_pydict()
    qids = [str(x) for x in e["id"]]
    # 'N/A' means "nothing excluded"; exclusions are applied as a set, so dedupe and sort.
    excl = [sorted({str(v) for v in (x or []) if str(v) != "N/A"}) for x in e["excluded_ids"]]
    qq, qd = [], []
    for qid, gold in zip(qids, e["gold_ids"]):  # binary relevance, one row per distinct gold id
        for g in sorted({str(x) for x in gold or []}):
            qq.append(qid); qd.append(g)
    write(out,
          pa.table({"_id": pa.array([str(x) for x in d["id"]], S), "title": pa.array([""] * len(d["id"]), S),
                    "text": pa.array([x or "" for x in d["content"]], S)}),
          pa.table({"_id": pa.array(qids, S), "text": pa.array([x or "" for x in e["query"]], S),
                    "excluded_ids": pa.array(excl, pa.list_(S))}),
          pa.table({"query-id": pa.array(qq, S), "corpus-id": pa.array(qd, S),
                    "score": pa.array([1] * len(qq), pa.int64())}))


def mteb_v2(task, out):
    from datasets import load_dataset
    repo, rev = MTEB_V2[task]
    def cfg(name, split):
        dd = load_dataset(repo, name, revision=rev, token=False)
        return dd[split] if split in dd else next(iter(dd.values()))
    c, q, r = cfg("corpus", "corpus"), cfg("queries", "queries"), cfg("default", "test")
    cid = "_id" if "_id" in c.column_names else "id"
    qid = "_id" if "_id" in q.column_names else "id"
    rels = [(str(a), str(b), int(s)) for a, b, s in zip(r["query-id"], r["corpus-id"], r["score"])]
    judged = {a for a, _, _ in rels}
    keep = [(str(i), str(t)) for i, t in zip(q[qid], q["text"]) if str(i) in judged]  # test queries only
    write(out,
          pa.table({"_id": pa.array([str(x) for x in c[cid]], S),
                    "title": pa.array([x or "" for x in c["title"]] if "title" in c.column_names
                                      else [""] * c.num_rows, S),
                    "text": pa.array([x or "" for x in c["text"]], S)}),
          pa.table({"_id": pa.array([a for a, _ in keep], S), "text": pa.array([b for _, b in keep], S)}),
          pa.table({"query-id": pa.array([a for a, _, _ in rels], S), "corpus-id": pa.array([b for _, b, _ in rels], S),
                    "score": pa.array([s for _, _, s in rels], pa.int64())}))


BENCH = {"r2med": (R2MED, r2med), "beir_bio": (BEIR_BIO, beir_bio),
         "bright": (BRIGHT, bright), "mteb_v2": (MTEB_V2, mteb_v2)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=BENCH)
    ap.add_argument("--task", help="one task directory name; default all")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    tasks, fn = BENCH[a.benchmark]
    for t in [a.task] if a.task else list(tasks):
        fn(t, os.path.join(a.out, a.benchmark, t))
