"""Assemble a PMC-Patients index for a model whose PMC vectors are split across its OpenAlex
corpus repo (bridged PMIDs) and a `<model>_gap` folder (PMIDs with no OpenAlex work).

    python assemble_pmc_index.py --selection bridge/selection_reasonembed.parquet \
        --corpus reasonembed-openalex-58m --gap reasonembed_gap --out pmc_reasonembed

The output is a search.py index (11,713,201 rows, id column pmid): one new shard `bridged`
holding the selected corpus rows, plus the gap shards linked in place. Needs ~93 GB of disk.
"""
import argparse, json, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for x in ("--selection", "--corpus", "--gap", "--out"):
    ap.add_argument(x, required=True)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.gap, "index_meta.json")))
vm, gap_shards = meta["vectors"], meta["shards"]
dim, dt, ext = int(vm["dim"]), np.dtype(vm["dtype"]), vm["file_pattern"].rsplit(".", 1)[1]
sel = pd.read_parquet(a.selection).sort_values(["shard", "row_index"])
for d in ("vectors", "ids"):
    os.makedirs(os.path.join(a.out, d), exist_ok=True)
with open(os.path.join(a.out, "vectors", f"bridged.{ext}"), "wb") as f:
    for shard, g in sel.groupby("shard", sort=False):
        X = np.memmap(os.path.join(a.corpus, "vectors", f"{shard}.{ext}"), dtype=dt, mode="r").reshape(-1, dim)
        f.write(np.ascontiguousarray(X[g["row_index"].to_numpy()]).tobytes())
pq.write_table(pa.table({"row_index": np.arange(len(sel), dtype=np.int32), "pmid": sel["pmid"].astype(str)}),
               os.path.join(a.out, "ids", "bridged.parquet"))
for s in gap_shards:
    for d, e in (("vectors", ext), ("ids", "parquet")):
        dst = os.path.join(a.out, d, f"{s}.{e}")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(a.gap, d, f"{s}.{e}")), dst)
json.dump(dict(meta, corpus="bridged corpus rows + gap rows (assemble_pmc_index.py)",
               shards={"bridged": len(sel), **gap_shards}, n_rows=len(sel) + sum(gap_shards.values())),
          open(os.path.join(a.out, "index_meta.json"), "w"), indent=1)
print(f"{a.out}: {len(sel):,} bridged + {sum(gap_shards.values()):,} gap rows")
