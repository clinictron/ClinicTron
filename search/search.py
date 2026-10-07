"""Exact (brute-force) cosine search over a published embedding index.

    python search.py --index <index_dir> --queries q.npz --top-k 100 --out run.tsv

The index directory holds index_meta.json, vectors/<shard>.fp16 (headerless row-major
float16, `dim` columns) and ids/<shard>.parquet (row_index plus id columns).
Queries: an .npz written by encode.py (query_ids, query_vectors) or a plain .npy matrix.
Scores are fp32 cosine; shards are memory-mapped and read in chunks.
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch


def load_queries(path):
    if path.endswith(".npz"):
        z = np.load(path)
        return [str(x) for x in z["query_ids"]], z["query_vectors"]
    v = np.load(path)
    return [str(i) for i in range(len(v))], v


def search(index, Q, k, device="cpu", chunk=131072, id_col=None):
    meta = json.load(open(os.path.join(index, "index_meta.json")))
    vm = meta["vectors"]
    dim, dt = int(vm["dim"]), np.dtype(vm["dtype"])
    torch.backends.cuda.matmul.allow_tf32 = False                   # exact fp32 products
    Q = torch.nn.functional.normalize(torch.from_numpy(np.asarray(Q, np.float32)).to(device), dim=1)
    nq = Q.shape[0]
    best_v = torch.full((nq, k), -2.0, device=device)
    best_i = torch.full((nq, k), -1, device=device, dtype=torch.int64)
    ids, offset = [], 0
    for shard in sorted(meta["shards"]):
        t = pq.read_table(os.path.join(index, "ids", shard + ".parquet"))
        col = id_col or [c for c in t.column_names if c != "row_index"][0]
        order = np.argsort(t.column("row_index").to_numpy(), kind="stable")
        ids.extend(np.asarray(t.column(col).to_pylist(), dtype=object)[order])
        mm = np.memmap(os.path.join(index, "vectors", shard + ".fp16"), dtype=dt, mode="r").reshape(-1, dim)
        if len(mm) != len(order):
            raise ValueError(f"{shard}: {len(mm)} vectors but {len(order)} ids")
        for s in range(0, len(mm), chunk):
            X = torch.from_numpy(np.ascontiguousarray(mm[s:s + chunk])).to(device).float()
            if not torch.isfinite(X).all():
                raise ValueError(f"{shard}: non-finite vector rows")
            X = X / X.norm(dim=1).clamp_min(1e-12)[:, None]          # cosine whether or not rows are unit
            S = Q @ X.T
            v, j = torch.topk(S, min(k, S.shape[1]), dim=1)
            gid = j + offset + s
            best_v, sel = torch.topk(torch.cat([best_v, v], 1), k, dim=1)
            best_i = torch.gather(torch.cat([best_i, gid], 1), 1, sel)
        offset += len(mm)
        del mm
    return best_v.cpu().numpy(), best_i.cpu().numpy(), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out", required=True, help="TSV: query_id, rank, doc_id, score")
    ap.add_argument("--id-column", help="ids column to report (default: first after row_index)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    qids, Q = load_queries(a.queries)
    v, i, ids = search(a.index, Q, a.top_k, a.device, id_col=a.id_column)
    with open(a.out, "w") as f:
        for q, row_v, row_i in zip(qids, v, i):
            for r, (s, j) in enumerate(zip(row_v, row_i), 1):
                if j >= 0:
                    f.write(f"{q}\t{r}\t{ids[j]}\t{s:.7f}\n")


if __name__ == "__main__":
    main()
